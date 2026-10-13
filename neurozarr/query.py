"""Queries over the index: a small expression language, compiled twice.

``datatype == 'ieeg' and sfreq >= 250 and participant.age < 40`` is parsed with
:mod:`ast` (never evaluated as Python) into comparisons joined by ``and`` / ``or`` /
``not``. It then runs in two passes:

1. **prune** -- per subject, against the summary: can this subject hold a match? Uses
   min/max zone maps for numbers, exact value sets or Bloom filters for categories, and
   evaluates ``sub`` / ``participant.*`` exactly. Conservative: it may keep a subject
   with no match, never drop one with a match.
2. **mask** -- per surviving subject, against its index: a numpy boolean mask over only
   the columns the expression names, comparing dictionary codes as integers.

A missing value never matches a comparison (``!=`` included), as in SQL; ``exists(col)``
tests presence.
"""

import ast
import bisect
from dataclasses import dataclass
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import ValidationError
from .index import Summary, Table, _text

if TYPE_CHECKING:
	import pandas as pd

_OPS = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
		ast.In: "in", ast.NotIn: "not in"}
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
_KWARG_OPS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "in": "in"}


@dataclass(frozen=True)
class Cmp:
	column: str
	op: str
	value: Any


@dataclass(frozen=True)
class Exists:
	column: str


@dataclass(frozen=True)
class And:
	parts: tuple[Any, ...]


@dataclass(frozen=True)
class Or:
	parts: tuple[Any, ...]


@dataclass(frozen=True)
class Not:
	part: Any


Expr = Cmp | Exists | And | Or | Not


def _column(node: ast.AST) -> str | None:
	if isinstance(node, ast.Name):
		return node.id
	if isinstance(node, ast.Attribute):
		base = _column(node.value)
		return None if base is None else f"{base}.{node.attr}"
	if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "col" \
			and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
		return node.args[0].value
	return None


def _constant(node: ast.AST) -> Any:
	if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
		return node.value
	if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
			and isinstance(node.operand.value, (int, float)):
		return -node.operand.value
	if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
		return tuple(_constant(e) for e in node.elts)
	raise ValidationError(f"expected a value, got {ast.unparse(node)!r}")


def _convert(node: ast.AST) -> Expr:
	if isinstance(node, ast.BoolOp):
		parts = tuple(_convert(v) for v in node.values)
		return And(parts) if isinstance(node.op, ast.And) else Or(parts)
	if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
		return Not(_convert(node.operand))
	if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "exists" and len(node.args) == 1:
		name = _column(node.args[0]) if not isinstance(node.args[0], ast.Constant) else node.args[0].value
		if not isinstance(name, str):
			raise ValidationError("exists() takes one column name")
		return Exists(name)
	if isinstance(node, ast.Compare):
		terms = [node.left, *node.comparators]
		pairs = []
		for left, op, right in zip(terms, node.ops, terms[1:]):
			if type(op) not in _OPS:
				raise ValidationError(f"unsupported operator {type(op).__name__}")
			symbol = _OPS[type(op)]
			if (name := _column(left)) is not None:
				pairs.append(Cmp(name, symbol, _constant(right)))
			elif (name := _column(right)) is not None and symbol in _FLIP:
				pairs.append(Cmp(name, _FLIP[symbol], _constant(left)))
			else:
				raise ValidationError(f"a comparison needs a column on one side: {ast.unparse(node)!r}")
		return pairs[0] if len(pairs) == 1 else And(tuple(pairs))
	raise ValidationError(f"can't use {ast.unparse(node)!r} in a query")


def parse(text: str | None = None, **kwargs: Any) -> Expr | None:
	"""Parse an expression and/or keyword filters into one expression.

	Keywords compare for equality (``task="rest"``), or use a suffix: ``sfreq__ge=250``
	(``eq ne lt le gt ge in``); ``__`` between words stands for a dot
	(``participant__age__lt=40`` is ``participant.age < 40``).
	"""
	parts: list[Expr] = []
	if text and text.strip():
		try:
			tree = ast.parse(text, mode="eval")
		except SyntaxError as exc:
			raise ValidationError(f"can't parse query {text!r}: {exc.msg}") from exc
		parts.append(_convert(tree.body))
	for key, value in kwargs.items():
		words = key.split("__")
		op = _KWARG_OPS.get(words[-1]) if len(words) > 1 else None
		column = ".".join(words[:-1] if op else words)
		parts.append(Cmp(column, op or "==", tuple(value) if op == "in" else value))
	if not parts:
		return None
	return parts[0] if len(parts) == 1 else And(tuple(parts))


def columns_of(expr: Expr | None) -> set[str]:
	if expr is None:
		return set()
	if isinstance(expr, (Cmp, Exists)):
		return {expr.column}
	if isinstance(expr, Not):
		return columns_of(expr.part)
	return set().union(*(columns_of(p) for p in expr.parts))


# ---- row masks ---------------------------------------------------------------------

def _num_value(value: Any) -> float | None:
	if isinstance(value, bool):
		return None
	if isinstance(value, (int, float)):
		return float(value)
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _cmp_mask(table: Table, column: str, op: str, value: Any) -> np.ndarray:
	values = value if op in ("in", "not in") else (value,)
	if table.kind_of(column) == "num":
		numbers = [_num_value(v) for v in values]
		if all(n is not None for n in numbers):
			col = table.num(column)
			present = ~np.isnan(col)
			if op in ("in", "not in"):
				hit = np.isin(col, np.array(numbers, dtype=float))
				return present & (hit if op == "in" else ~hit)
			n = numbers[0]
			with np.errstate(invalid="ignore"):
				result = {"==": col == n, "!=": col != n, "<": col < n, "<=": col <= n,
						  ">": col > n, ">=": col >= n}[op]
			return present & result
	codes, dictionary = table.codes(column)
	present = codes >= 0
	texts = [_text(v) for v in values]
	if op in ("==", "!=", "in", "not in"):
		lookup = table.lookup(column)
		wanted = [lookup[t] for t in texts if t in lookup]
		if len(wanted) == 1:
			hit = codes == wanted[0]
		else:
			hit = np.isin(codes, np.array(wanted, dtype=np.int32))
		return present & (hit if op in ("==", "in") else ~hit)
	# ordering on text: the dictionary is sorted, so codes order as the values do
	cut = texts[0]
	if op == "<":
		return present & (codes < bisect.bisect_left(dictionary, cut))
	if op == "<=":
		return present & (codes < bisect.bisect_right(dictionary, cut))
	if op == ">":
		return present & (codes >= bisect.bisect_right(dictionary, cut))
	return present & (codes >= bisect.bisect_left(dictionary, cut))


def mask(expr: Expr | None, table: Table, subject: dict[str, Any] | None = None) -> np.ndarray:
	"""Which rows of ``table`` match. ``subject`` holds the subject-level values
	(``participant.*``) that every row of a subject index shares."""
	n = table.n
	if expr is None:
		return np.ones(n, dtype=bool)
	if isinstance(expr, And):
		out = np.ones(n, dtype=bool)
		for part in expr.parts:
			out &= mask(part, table, subject)
		return out
	if isinstance(expr, Or):
		out = np.zeros(n, dtype=bool)
		for part in expr.parts:
			out |= mask(part, table, subject)
		return out
	if isinstance(expr, Not):
		return ~mask(expr.part, table, subject)
	if subject is not None and expr.column.startswith("participant."):
		return np.full(n, _scalar_match(expr, subject.get(expr.column)))
	if isinstance(expr, Exists):
		if table.kind_of(expr.column) == "num":
			return ~np.isnan(table.num(expr.column))
		return table.codes(expr.column)[0] >= 0
	return _cmp_mask(table, expr.column, expr.op, expr.value)


def _scalar_match(expr: "Cmp | Exists", value: Any) -> bool:
	"""One subject-level value against a comparison, with the same rules as the columns:
	missing never matches; numbers compare as numbers when both sides are; else as text."""
	if isinstance(expr, Exists):
		return value is not None
	if value is None:
		return False
	targets = expr.value if expr.op in ("in", "not in") else (expr.value,)
	mine, theirs = _num_value(value), [_num_value(t) for t in targets]
	if mine is not None and all(t is not None for t in theirs):
		left: Any = mine
		rights: list[Any] = theirs
	else:
		left, rights = _text(value), [_text(t) for t in targets]
	if expr.op in ("in", "not in"):
		return (left in rights) == (expr.op == "in")
	right = rights[0]
	return {"==": left == right, "!=": left != right, "<": left < right, "<=": left <= right,
			">": left > right, ">=": left >= right}[expr.op]


def subject_values(summary: Summary, i: int, columns: set[str]) -> dict[str, Any]:
	"""The subject-level (``participant.*``) values of summary row ``i`` that are needed."""
	row = np.array([i])
	return {c: summary.table.values(c, row)[0] for c in columns if c.startswith("participant.")}


# ---- subject pruning -----------------------------------------------------------------

def _exact(expr: Expr) -> bool:
	"""Whether the summary answers this exactly (subject-level columns only)."""
	return all(c == "sub" or c.startswith("participant.") for c in columns_of(expr))


def prune(expr: Expr | None, summary: Summary) -> np.ndarray:
	"""Per subject: may it hold a matching item?"""
	table = summary.table
	n = table.n
	if expr is None:
		return np.ones(n, dtype=bool)
	if _exact(expr):
		return mask(_rename_sub(expr), table)
	if isinstance(expr, And):
		out = np.ones(n, dtype=bool)
		for part in expr.parts:
			out &= prune(part, summary)
		return out
	if isinstance(expr, Or):
		out = np.zeros(n, dtype=bool)
		for part in expr.parts:
			out |= prune(part, summary)
		return out
	if isinstance(expr, Not):
		return np.ones(n, dtype=bool)  # "not" of a maybe is a maybe
	if isinstance(expr, Exists):
		return summary.has_column(expr.column)
	column, op, value = expr.column, expr.op, expr.value
	values = value if op in ("in", "not in") else (value,)
	if op in ("!=", "not in"):
		return summary.has_column(column)
	if column == "path" and op in ("==", "in"):  # a path starts with its subject's id
		return np.isin(np.array(summary.subjects, dtype=object), [str(v).split("/", 1)[0] for v in values])
	lo, hi = table.num(f"zmin.{column}"), table.num(f"zmax.{column}")
	numbers = [_num_value(v) for v in values]
	zone = np.zeros(n, dtype=bool)
	if all(x is not None for x in numbers):
		with np.errstate(invalid="ignore"):
			for x in numbers:
				zone |= {"==": (lo <= x) & (x <= hi), "in": (lo <= x) & (x <= hi), "<": lo < x, "<=": lo <= x,
						 ">": hi > x, ">=": hi >= x}[op]
	if op in ("==", "in"):
		return zone | summary.may_contain(column, [_text(v) for v in values])
	return zone | summary.has_column(column) & np.isnan(lo)  # ordering on text: keep any subject with it


def _rename_sub(expr: Expr) -> Expr:
	"""In the summary, a subject's id is its row's ``path``."""
	if isinstance(expr, Cmp):
		return Cmp("path", expr.op, expr.value) if expr.column == "sub" else expr
	if isinstance(expr, Exists):
		return Exists("path") if expr.column == "sub" else expr
	if isinstance(expr, Not):
		return Not(_rename_sub(expr.part))
	return type(expr)(tuple(_rename_sub(p) for p in expr.parts))


# ---- results -------------------------------------------------------------------------

class QueryResult:
	"""The items a query matched, in path order.

	``len()``, ``paths`` and ``df`` come straight from the index -- nothing is opened;
	iterating (or indexing) yields the usual lazy views, opened one by one.
	"""

	def __init__(self, hits: list[tuple[str, Table, np.ndarray, dict[str, Any]]],
				 view_of: "Callable[[str, str, str], Any]"):
		self._hits = hits
		self._view_of = view_of

	def __len__(self) -> int:
		return sum(len(rows) for _, _, rows, _ in self._hits)

	def __repr__(self) -> str:
		return f"<QueryResult {len(self)} item(s) in {len(self._hits)} subject(s)>"

	def _entries(self) -> "Iterator[tuple[str, Table, int]]":
		for sub, table, rows, _ in self._hits:
			for i in rows:
				yield sub, table, int(i)

	@property
	def paths(self) -> list[str]:
		return [table.header["paths"][i] for _, table, i in self._entries()]

	def __iter__(self) -> "Iterator[Any]":
		for sub, table, rows, _ in self._hits:
			paths = table.header["paths"]
			for i, kind in zip(rows, table.values("kind", rows)):  # kinds decoded once per subject
				yield self._view_of(sub, paths[i], kind)

	def __getitem__(self, position: int) -> Any:
		left = position % len(self) if len(self) else position
		for sub, table, rows, _ in self._hits:
			if left < len(rows):
				i = rows[left]
				return self._view_of(sub, table.header["paths"][i], table.values("kind", np.array([i]))[0])
			left -= len(rows)
		raise IndexError(position)

	@property
	def df(self) -> "pd.DataFrame":
		"""One row per match: path, kind, identifiers, shape, rate, duration, codec,
		``meta.*`` and the subject's ``participant.*`` columns."""
		import pandas as pd
		frames = []
		for sub, table, rows, subject in self._hits:
			data = {name: table.values(name, rows) for name in table.columns}
			for name, value in subject.items():
				if name.startswith("participant."):
					data[name] = [value] * len(rows)
			frames.append(pd.DataFrame(data))
		if not frames:
			return pd.DataFrame(columns=["path", "kind"])
		frame = pd.concat(frames, ignore_index=True)
		first = [c for c in ("path", "kind", "sub", "ses", "datatype") if c in frame.columns]
		return frame[first + [c for c in frame.columns if c not in first]]
