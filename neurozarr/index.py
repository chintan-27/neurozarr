"""The store's index: a manifest at every level, built for pruning and vectorized filters.

Each dataset version carries, in its ``_dataset`` repository under ``_index/``:

- one **subject index** per subject -- a columnar table with a row per item (recording,
  table, array, external file): its path, kind, session, datatype, entities, shape,
  sampling rate, duration, codec, and every scalar of its metadata as ``meta.<key>``;
  plus the subject's sessions and participant/session attributes;
- one **summary** -- a row per subject (the Iceberg "manifest list"): its pin, the
  digest of its index, item counts, min/max of every numeric item column (zone maps),
  and, per categorical column, which values occur (a bit matrix over the column's
  values, or a Bloom filter for high-cardinality columns).

A query reads the summary, skips every subject that cannot match, then reads only the
surviving subjects' indexes and filters them with numpy masks. Each index is one stored
object (one request), and immutable for its digest, so it is cached without invalidation.

The ideas are borrowed, with their measurements, from Apache Iceberg (manifest lists),
DuckDB (zone maps, sorted rows), Arrow/Parquet (dictionary encoding, late
materialization) and Bloom filters -- see ``docs/guide/read.md``.
"""

import bisect
import hashlib
import json
import math
import struct
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, cast

import numpy as np
import zarr

from .read import ArrayView, ExternalFileView, RecordingView, TableView, owns_items, views_at, views_in

INDEX_GROUP = "_index"
SUMMARY = "_summary"  # never a subject id: those start with "sub-"
PACKED = "_packed"
# a query keeping more subjects than this reads them from the packed index (one read)
# instead of one index each; a store with no more subjects than this gets no packed index
PACK_MIN = 16
# the packed index is rewritten at commit once more than max(PACK_MIN, this share of the
# subjects) changed since it was written; until then queries read those few from their own
PACK_STALE = 0.05
_MAGIC = b"NZIX1\n"
# a categorical column whose distinct values across the dataset fit this many gets an
# exact value-membership matrix in the summary; above it, a Bloom filter per subject
_EXACT_VALUES = 4096
_BLOOM_BITS = 2048
_BLOOM_HASHES = 4
_SKIP_META = {"SamplingFrequency", "ch_names", "data_scale", "data_offset", "data_note"}
_CORE = ("path", "kind", "sub", "ses", "datatype", "prefix")


# ---- rows ------------------------------------------------------------------------

def _scalar(value: Any) -> bool:
	return value is None or isinstance(value, (bool, int, float, str))


def _codec(array: zarr.Array) -> str:
	parts = []
	for codec in array.compressors:
		spec: dict[str, Any] = dict(cast(Any, codec).to_dict())
		config: dict[str, Any] = dict(spec.get("configuration") or {})
		name = str(spec.get("name", type(codec).__name__))
		detail = ",".join(f"{k}={v}" for k, v in sorted(config.items()) if k in ("cname", "clevel", "shuffle", "level"))
		parts.append(f"{name}({detail})" if detail else name)
	return "+".join(parts) or "none"


def _place(path: str) -> dict[str, Any]:
	"""sub, prefix, ses and datatype of an item from its path."""
	parts = path.split("/")
	sub, rest = parts[0], parts[1:]
	prefix = None
	if rest[:1] == ["derivatives"] and len(rest) >= 2:
		prefix, rest = f"derivatives/{rest[1]}", rest[2:]
	ses = rest[0] if rest and rest[0].startswith("ses-") else None
	if ses:
		rest = rest[1:]
	return {"sub": sub, "prefix": prefix, "ses": ses, "datatype": rest[0] if len(rest) >= 2 else None}


def make_row(path: str, kind: str, entities: dict[str, Any], meta: dict[str, Any], node: Any = None,
			 columns: list[str] | None = None, uri: str | None = None, media_type: str | None = None) -> dict[str, Any]:
	"""The index row for one item, from what is known about it -- shared by a view
	(re-reading a stored item) and the writer (describing one it just wrote)."""
	row: dict[str, Any] = {"path": path, "kind": kind, **_place(path)}
	row.update({key: str(value) for key, value in entities.items() if key not in row})
	if node is not None:
		row["dtype"] = str(node.dtype)
		row["shape"] = "x".join(map(str, node.shape))
		row["chunks"] = "x".join(map(str, node.chunks))
		row["codec"] = _codec(node)
		row["nbytes"] = int(node.nbytes)
	if kind == "recording" and node is not None:
		row["n_channels"], row["n_samples"] = int(node.shape[0]), int(node.shape[-1])
		sfreq = meta.get("SamplingFrequency")
		if isinstance(sfreq, (int, float)) and not isinstance(sfreq, bool) and sfreq > 0:
			row["sfreq"] = float(sfreq)
			row["duration_s"] = row["n_samples"] / float(sfreq)
	if columns is not None:
		row["n_columns"] = len(columns)
	if uri is not None:
		row["uri"] = uri
	if media_type is not None:
		row["media_type"] = media_type
	for key, value in meta.items():
		if key.startswith("_") or key in _SKIP_META or not _scalar(value) or value is None:
			continue
		row[f"meta.{key}"] = value
	return row


def row_of(view: Any) -> dict[str, Any]:
	"""The index row for one item view."""
	if isinstance(view, RecordingView):
		return make_row(view.path, "recording", view.entities, view.meta, view.array)
	if isinstance(view, TableView):
		return make_row(f"{view.path}/{view.name}", "table", view.entities, view.meta, columns=view.columns)
	if isinstance(view, ArrayView):
		return make_row(f"{view.path}/{view.name}", "array", view.entities, view.meta, view.array)
	return make_row(f"{view.path}/{view.name}", "file", view.entities, view.meta,
					uri=view.uri, media_type=view.media_type)


def scalars(attrs: dict[str, Any], prefix: str) -> dict[str, Any]:
	return {f"{prefix}{k}": v for k, v in attrs.items()
			if not k.startswith("_") and _scalar(v) and v is not None and k not in ("subjects", "codec_plan")}


# ---- the columnar blob -------------------------------------------------------------

def _classify(values: list[Any]) -> str:
	present = [v for v in values if v is not None]
	if present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
		return "num"
	return "cat"


def _text(value: Any) -> str:
	return ("true" if value else "false") if isinstance(value, bool) else str(value)


def encode(rows: list[dict[str, Any]], header: dict[str, Any] | None = None,
		   buffers: dict[str, np.ndarray] | None = None) -> bytes:
	"""Rows -> one blob: a JSON header, then raw column buffers.

	Numeric columns are float64 (NaN = missing); every other column is dictionary-
	encoded: its sorted distinct values in the header, int32 codes (-1 = missing) as a
	buffer -- so equality and ordering compare ints. Rows are sorted by ``path``.
	Nothing is pickled: a blob from anywhere is safe to read.
	"""
	rows = sorted(rows, key=lambda r: r["path"])
	columns: dict[str, Column] = {}
	for name in sorted({key for row in rows for key in row} - {"path"}):
		values = [row.get(name) for row in rows]
		if _classify(values) == "num":
			columns[name] = ("num", np.array([np.nan if v is None else float(v) for v in values]), [])
		else:
			texts = [None if v is None else _text(v) for v in values]
			dictionary = sorted({t for t in texts if t is not None})
			lookup = {t: i for i, t in enumerate(dictionary)}
			columns[name] = ("cat", np.array([-1 if t is None else lookup[t] for t in texts], dtype=np.int32), dictionary)
	return _encode_columns([r["path"] for r in rows], columns, header, buffers)


# ("num", float64 values, []) or ("cat", int32 codes, sorted dictionary)
Column = tuple[str, np.ndarray, list[str]]


def _encode_columns(paths: list[str], columns: dict[str, Column], header: dict[str, Any] | None,
					buffers: dict[str, np.ndarray] | None) -> bytes:
	"""Already-columnar data -> a blob; ``paths`` sorted, every array in that row order."""
	spec: dict[str, Any] = {}
	parts: list[bytes] = []
	offset = 0

	def put(array: np.ndarray) -> dict[str, Any]:
		nonlocal offset
		data = np.ascontiguousarray(array).tobytes()
		entry = {"offset": offset, "dtype": array.dtype.str, "shape": list(array.shape)}
		parts.append(data + b"\0" * (-len(data) % 8))
		offset += len(data) + (-len(data) % 8)
		return entry

	for name, (kind, array, dictionary) in sorted(columns.items()):
		spec[name] = {"type": kind, **({"values": dictionary} if kind == "cat" else {}), **put(array)}
	extra = {name: put(array) for name, array in (buffers or {}).items()}
	head = json.dumps({"n": len(paths), "paths": paths, "columns": spec,
					   "buffers": extra, **(header or {})}, separators=(",", ":")).encode()
	return _MAGIC + struct.pack("<Q", len(head)) + head + b"".join(parts)


@dataclass
class Table:
	"""A decoded blob: columns are read lazily, without copying, from the buffer."""

	header: dict[str, Any]
	_body: memoryview
	_cache: dict[Any, Any] = field(default_factory=dict)

	@classmethod
	def decode(cls, blob: bytes) -> "Table":
		if not blob.startswith(_MAGIC):
			raise ValueError("not a neurozarr index blob")
		(size,) = struct.unpack("<Q", blob[len(_MAGIC):len(_MAGIC) + 8])
		start = len(_MAGIC) + 8
		return cls(json.loads(blob[start:start + size]), memoryview(blob)[start + size:])

	@property
	def n(self) -> int:
		return int(self.header["n"])

	@property
	def paths(self) -> list[str]:
		"""Row paths, sorted. The table's own list, not a copy -- read it, don't change it
		(a copy here made every ``paths[i]`` in a loop quadratic)."""
		return cast("list[str]", self.header["paths"])

	@property
	def columns(self) -> list[str]:
		return ["path", *self.header["columns"]]

	def _array(self, entry: dict[str, Any]) -> np.ndarray:
		key = entry["offset"]  # decoded once per table: blobs are immutable
		cached = self._cache.get(key)
		if cached is not None:
			return cast(np.ndarray, cached)
		dtype = np.dtype(entry["dtype"])
		count = int(np.prod(entry["shape"])) if entry["shape"] else 1
		array = np.frombuffer(self._body, dtype=dtype, count=count, offset=entry["offset"]).reshape(entry["shape"])
		self._cache[key] = array
		return array

	def buffer(self, name: str) -> np.ndarray:
		return self._array(self.header["buffers"][name])

	def kind_of(self, name: str) -> str | None:
		entry = self.header["columns"].get(name)
		return entry["type"] if entry else ("path" if name == "path" else None)

	def num(self, name: str) -> np.ndarray:
		"""A numeric column as float64 (NaN = missing); all-NaN if absent or categorical."""
		entry = self.header["columns"].get(name)
		if entry is None or entry["type"] != "num":
			return np.full(self.n, np.nan)
		return self._array(entry)

	def codes(self, name: str) -> tuple[np.ndarray, list[str]]:
		"""A categorical column: int32 codes (-1 = missing) and its sorted values."""
		if name == "path":
			return np.arange(self.n, dtype=np.int32), self.paths
		entry = self.header["columns"].get(name)
		if entry is None:
			return np.full(self.n, -1, dtype=np.int32), []
		if entry["type"] == "num":  # compared as text: numbers formatted as written
			values = self._array(entry)
			texts = [None if math.isnan(v) else _text(int(v) if float(v).is_integer() else v) for v in values]
			dictionary = sorted({t for t in texts if t is not None})
			lookup = {t: i for i, t in enumerate(dictionary)}
			return np.array([-1 if t is None else lookup[t] for t in texts], dtype=np.int32), dictionary
		return self._array(entry), entry["values"]

	def lookup(self, name: str) -> dict[str, int]:
		"""value -> code of a categorical column, built once per table."""
		key = ("lookup", name)
		cached = self._cache.get(key)
		if cached is None:
			cached = {value: i for i, value in enumerate(self.codes(name)[1])}
			self._cache[key] = cached
		return cast("dict[str, int]", cached)

	def values(self, name: str, rows: np.ndarray | None = None) -> list[Any]:
		"""Decoded values of a column (for the selected rows): floats, strings or None."""
		picked = np.arange(self.n) if rows is None else rows
		if name == "path":
			paths = self.paths
			return [paths[i] for i in picked]
		entry = self.header["columns"].get(name)
		if entry is None:
			return [None] * len(picked)
		if entry["type"] == "num":
			column = self._array(entry)
			return [None if math.isnan(column[i]) else float(column[i]) for i in picked]
		codes, dictionary = self._array(entry), entry["values"]
		return [None if codes[i] < 0 else dictionary[codes[i]] for i in picked]

	def row(self, i: int) -> dict[str, Any]:
		return {name: value for name in self.columns
				if (value := self.values(name, np.array([i]))[0]) is not None}

	def rows(self) -> list[dict[str, Any]]:
		return [self.row(i) for i in range(self.n)]

	def find(self, path: str) -> int | None:
		"""Row of ``path`` by binary search -- rows are sorted by path."""
		paths = self.header["paths"]
		lo, hi = 0, len(paths)
		while lo < hi:
			mid = (lo + hi) // 2
			if paths[mid] < path:
				lo = mid + 1
			else:
				hi = mid
		return lo if lo < len(paths) and paths[lo] == path else None


class _Cache:
	"""Decoded indexes by digest. A digest names immutable content, so nothing here
	ever goes stale -- old entries are only evicted for room."""

	def __init__(self, size: int = 4096):
		self._items: "OrderedDict[str, Table]" = OrderedDict()
		self._size = size

	def get(self, key: str) -> "Table | None":
		table = self._items.get(key)
		if table is not None:
			self._items.move_to_end(key)
		return table

	def put(self, key: str, table: "Table") -> "Table":
		self._items[key] = table
		self._items.move_to_end(key)
		while len(self._items) > self._size:
			self._items.popitem(last=False)
		return table


CACHE = _Cache()


def _disk_dir() -> Path:
	import os
	return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "neurozarr" / "index"


def disk_get(key: str) -> bytes | None:
	"""An index from the local disk cache, if there and intact (checked by digest)."""
	path = _disk_dir() / key
	try:
		blob = path.read_bytes()
	except OSError:
		return None
	return blob if digest(blob) == key else None


def disk_put(key: str, blob: bytes) -> None:
	folder = _disk_dir()
	try:
		folder.mkdir(parents=True, exist_ok=True)
		tmp = folder / f".{key}.tmp"
		tmp.write_bytes(blob)
		tmp.replace(folder / key)
	except OSError:
		pass  # a cache: failing to fill it is never an error


def digest(blob: bytes) -> str:
	return hashlib.blake2b(blob, digest_size=16).hexdigest()


# ---- subject indexes -----------------------------------------------------------------

@dataclass
class Pending:
	"""A subject index computed from a staged tree, waiting for its snapshot id."""

	rows: list[dict[str, Any]]
	header: dict[str, Any]

	def finish(self, snapshot: str) -> bytes:
		return encode(self.rows, {**self.header, "snapshot": snapshot})


def _sessions(root: zarr.Group) -> list[str]:
	return sorted(name for name, node in root.members() if isinstance(node, zarr.Group) and name.startswith("ses-"))


def build_subject(root: zarr.Group, sub_id: str) -> Pending:
	"""A subject's index from scratch: one walk of its tree."""
	sessions = _sessions(root)
	return Pending([row_of(view) for view in views_in(root, sub_id)],
				   {"sub": sub_id, "sessions": sessions, "attrs": scalars(root.attrs.asdict(), ""),
					"session_attrs": {ses: scalars(cast(zarr.Group, root[ses]).attrs.asdict(), "") for ses in sessions}})


_OWN_CHILDREN = ("data", "events", "channels")  # read.py's recording facade children: not items


def update_subject(previous: "Table | None", root: zarr.Group, sub_id: str, changes: dict[str, list[str]],
				   captured: dict[str, dict[str, Any]] | None = None) -> Pending:
	"""A subject's index after a commit, touching only what changed.

	Starts from the published index (none for a new subject); ``changes`` is the
	staged diff summarized to items (see ``repo._summarize``). Removed items drop
	their rows and everything under them. Rows the writer ``captured`` for items it
	wrote are taken as they are -- no read. Only touched items it didn't describe
	(metadata edits, renames) are re-read from ``root``, one group at a time.
	"""
	captured = captured or {}

	def full(rel: str) -> str:
		return f"{sub_id}/{rel}" if rel else sub_id

	rows = {row["path"]: row for row in previous.rows()} if previous is not None else {}
	moved = [m.split(" -> ") for m in changes["moved"]]
	for rel in [*changes["removed"], *(old for old, _ in moved)]:
		path = full(rel)
		for key in [k for k in rows if k == path or k.startswith(path + "/")]:
			del rows[key]
	rows.update(captured)

	def recording(path: str) -> bool:
		return rows.get(path, {}).get("kind") == "recording"

	# a sidecar written before its recording looked like a table item; it belongs to it
	for path in [p for p, r in rows.items() if r["kind"] == "table"]:
		parent, _, name = path.rpartition("/")
		if name in _OWN_CHILDREN and recording(parent):
			del rows[path]

	def described(rel: str) -> bool:
		path = full(rel)
		parent, _, name = path.rpartition("/")
		return path in captured or (name in _OWN_CHILDREN and parent in captured)

	added = {*changes["added"], *(new for _, new in moved)}
	touched = {*added, *changes["changed"], *changes["attrs"]}
	owners: set[str] = set()
	new_trees: set[str] = set()
	for rel in touched:
		if described(rel):
			continue
		try:
			node = root[rel] if rel else root
		except KeyError:
			continue
		owner = rel if owns_items(node) else rel.rpartition("/")[0]
		owners.add(owner)
		if rel in added and owns_items(node):
			new_trees.add(rel)
	for owner in owners:
		owner_path = full(owner)
		for key in [k for k in rows if (k == owner_path or k.rpartition("/")[0] == owner_path) and k not in captured]:
			del rows[key]
		for view in views_at(cast(zarr.Group, root[owner]) if owner else root, owner_path):
			row = row_of(view)
			rows[row["path"]] = row
	for tree in new_trees:
		for view in views_in(cast(zarr.Group, root[tree]), full(tree)):
			row = row_of(view)
			rows.setdefault(row["path"], row)
	sessions = set(previous.header["sessions"]) if previous is not None else set()
	session_attrs = dict(previous.header["session_attrs"]) if previous is not None else {}
	for rel in [*changes["removed"], *(old for old, _ in moved)]:
		if "/" not in rel:
			sessions.discard(rel)
			session_attrs.pop(rel, None)
	for ses in {rel.split("/", 1)[0] for rel in touched} - sessions:
		if ses.startswith("ses-") and ses in root:
			sessions.add(ses)
	for ses in {rel.split("/", 1)[0] for rel in touched} & sessions:
		session_attrs[ses] = scalars(cast(zarr.Group, root[ses]).attrs.asdict(), "")
	return Pending(list(rows.values()), {"sub": sub_id, "sessions": sorted(sessions),
										 "attrs": scalars(root.attrs.asdict(), ""), "session_attrs": session_attrs})


# ---- the summary -------------------------------------------------------------------

def _bloom_positions(value: str) -> np.ndarray:
	raw = hashlib.blake2b(value.encode(), digest_size=4 * _BLOOM_HASHES).digest()
	return np.frombuffer(raw, dtype="<u4") % _BLOOM_BITS


def _bloom(values: Iterable[str]) -> np.ndarray:
	bits = np.zeros(_BLOOM_BITS // 8, dtype=np.uint8)
	positions = [_bloom_positions(value) for value in values]
	if positions:
		at = np.concatenate(positions)
		np.bitwise_or.at(bits, at // 8, (1 << (at % 8)).astype(np.uint8))
	return bits


def bloom_may_contain(bits: np.ndarray, value: str) -> bool:
	positions = _bloom_positions(value)
	return bool(np.all((bits[positions // 8] >> (positions % 8)) & 1))


@dataclass
class SubjectStats:
	"""One subject's summary row, and per categorical column its values."""

	row: dict[str, Any]
	sets: dict[str, set[str]]


def stats_of(table: Table, blob_digest: str) -> SubjectStats:
	"""A subject's summary entry, from its index."""
	row: dict[str, Any] = {"path": table.header["sub"], "pin": table.header["snapshot"], "digest": blob_digest,
						   "n_sessions": len(table.header["sessions"]),
						   **{f"participant.{k}": v for k, v in table.header["attrs"].items()}}
	kinds, names = table.codes("kind")
	for kind in ("recording", "table", "array", "file"):
		row[f"count.{kind}"] = int(np.sum(kinds == names.index(kind))) if kind in names else 0
	# no set for "path": an item's path starts with its subject id, which prune uses directly
	sets: dict[str, set[str]] = {}
	for name in table.columns[1:]:
		if table.kind_of(name) == "num":
			column = table.num(name)
			present = column[~np.isnan(column)]
			if present.size:
				row[f"zmin.{name}"], row[f"zmax.{name}"] = float(present.min()), float(present.max())
		else:
			codes, values = table.codes(name)
			sets[name] = {values[c] for c in np.unique(codes) if c >= 0}
	return SubjectStats(row, sets)


def build_summary(stats: list[SubjectStats], header: dict[str, Any] | None = None) -> bytes:
	"""The dataset's summary from every subject's entry."""
	stats = sorted(stats, key=lambda s: s.row["path"])
	n = len(stats)
	columns = sorted({name for s in stats for name in s.sets})
	exact: dict[str, Any] = {}
	buffers: dict[str, np.ndarray] = {}
	for name in columns:
		entries = [s.sets.get(name, set()) for s in stats]
		universe = sorted(set().union(*entries))
		if len(universe) <= _EXACT_VALUES:
			lookup = {v: j for j, v in enumerate(universe)}
			matrix = np.zeros((n, len(universe)), dtype=bool)
			for i, entry in enumerate(entries):
				matrix[i, [lookup[v] for v in entry]] = True
			exact[name] = {"values": universe}
			buffers[f"set:{name}"] = np.packbits(matrix, axis=1) if universe else np.zeros((n, 0), np.uint8)
		else:
			exact[name] = {"bloom": True}
			buffers[f"bloom:{name}"] = np.stack([_bloom(e) for e in entries]) \
				if n else np.zeros((0, _BLOOM_BITS // 8), np.uint8)
	return encode([s.row for s in stats], {"sets": exact, **(header or {})}, buffers)


def update_summary(previous: "Summary | None", changed: dict[str, "SubjectStats | None"],
				   header: dict[str, Any] | None = None) -> bytes:
	"""The next summary: ``previous`` with the ``changed`` subjects' rows replaced (None =
	removed). Untouched rows are carried as arrays, so a commit costs what it touched
	plus array copies -- not a Python decode and re-encode of every subject."""
	if previous is None:
		return build_summary([s for s in changed.values() if s is not None], header)
	fresh = Table.decode(build_summary([s for s in changed.values() if s is not None]))
	old = previous.table
	keep = np.nonzero(~np.isin(np.array(old.paths, dtype=object), list(changed)))[0] if old.n else np.zeros(0, int)
	parts = [(old, keep), (fresh, np.arange(fresh.n))]
	paths = [old.paths[i] for i in keep] + fresh.paths
	order = np.array(sorted(range(len(paths)), key=paths.__getitem__), dtype=np.intp)
	columns: dict[str, Column] = {}
	for name in sorted(set(old.header["columns"]) | set(fresh.header["columns"])):
		if all(t.kind_of(name) in ("num", None) for t, _ in parts):
			columns[name] = ("num", np.concatenate([t.num(name)[rows] for t, rows in parts])[order], [])
		else:
			codes, dictionary = _merge_codes([(*t.codes(name), rows) for t, rows in parts])
			columns[name] = ("cat", codes[order], dictionary)
	sets: dict[str, Any] = {}
	buffers: dict[str, np.ndarray] = {}
	for name in sorted(set(old.header["sets"]) | set(fresh.header["sets"])):
		specs = [t.header["sets"].get(name, {"values": []}) for t, _ in parts]
		universe = sorted(set().union(*(spec.get("values", []) for spec in specs)))
		matrix = None
		if all("values" in spec for spec in specs):
			lookup = {v: j for j, v in enumerate(universe)}
			matrix = np.zeros((len(paths), len(universe)), dtype=bool)
			at = 0
			for (t, rows), spec in zip(parts, specs):
				if spec["values"]:
					block = np.unpackbits(t.buffer(f"set:{name}"), axis=1, count=len(spec["values"]))[rows].astype(bool)
					matrix[at:at + len(rows), [lookup[v] for v in spec["values"]]] = block
				at += len(rows)
			used = matrix.any(axis=0)  # values only removed subjects had are dropped
			universe, matrix = [v for v, u in zip(universe, used) if u], matrix[:, used]
		if matrix is not None and len(universe) <= _EXACT_VALUES:
			sets[name] = {"values": universe}
			buffers[f"set:{name}"] = np.packbits(matrix[order], axis=1) if universe else np.zeros((len(paths), 0), np.uint8)
		else:
			blooms = [_bloom_rows(t, name, spec, rows) for (t, rows), spec in zip(parts, specs)]
			sets[name] = {"bloom": True}
			buffers[f"bloom:{name}"] = np.concatenate(blooms)[order]
	return _encode_columns([paths[i] for i in order], columns, {"sets": sets, **(header or {})}, buffers)


def _merge_codes(parts: list[tuple[np.ndarray, list[str], np.ndarray]]) -> tuple[np.ndarray, list[str]]:
	"""Concatenate dictionary-encoded columns (codes, dictionary, rows taken) onto one dictionary."""
	dictionary = sorted(set().union(*(values for _, values, _ in parts)))
	lookup = {v: j for j, v in enumerate(dictionary)}
	out = []
	for codes, values, rows in parts:
		remap = np.array([lookup[v] for v in values] + [-1], dtype=np.int32)  # remap[-1] = missing
		out.append(remap[codes[rows]])
	merged = np.concatenate(out) if out else np.zeros(0, np.int32)
	used = np.unique(merged[merged >= 0])  # values only dropped rows had are dropped too
	compact = np.full(len(dictionary) + 1, -1, dtype=np.int32)
	compact[used] = np.arange(len(used), dtype=np.int32)
	return compact[merged], [dictionary[i] for i in used]


def _bloom_rows(table: "Table", name: str, spec: dict[str, Any], rows: np.ndarray) -> np.ndarray:
	"""A set column's rows as Bloom bits, converting exact value sets when the merged
	column outgrows them (only on that transition: a per-row loop)."""
	if "bloom" in spec:
		return cast(np.ndarray, table.buffer(f"bloom:{name}")[rows])
	out = np.zeros((len(rows), _BLOOM_BITS // 8), dtype=np.uint8)
	if spec["values"]:
		matrix = np.unpackbits(table.buffer(f"set:{name}"), axis=1, count=len(spec["values"]))[rows].astype(bool)
		for k, row in enumerate(matrix):
			out[k] = _bloom(v for v, b in zip(spec["values"], row) if b)
	return out


# ---- the packed index ----------------------------------------------------------------

_TEXT = "\x00text"  # the text half of a column that is numeric in some subjects, text in others
_FORMAT_KEYS = ("n", "paths", "columns", "buffers")


def pack(previous: "Table | None", stale: dict[str, "Table | None"]) -> bytes:
	"""Every subject's index rows in one blob: ``previous`` (a packed index) with the
	``stale`` subjects replaced by their current index (None = removed).

	Rows stay sorted by path, so each subject's rows are one contiguous range, recorded
	in the header with the subject's own index header and column types. A column that is
	numeric in some subjects and text in others is stored both ways, so each subject's
	slice reads it back exactly as its own index had it.
	"""
	parts: list[tuple[Table, np.ndarray]] = []
	headers: dict[str, dict[str, Any]] = {}
	if previous is not None:
		kept = {sub: info for sub, info in previous.header["subjects"].items() if sub not in stale}
		ranges = [np.arange(info["start"], info["stop"]) for info in kept.values()]
		parts.append((previous, np.concatenate(ranges) if ranges else np.zeros(0, np.intp)))
		headers.update({sub: {"header": info["header"], "types": info["types"]} for sub, info in kept.items()})
	for sub, table in stale.items():
		if table is not None:
			parts.append((table, np.arange(table.n)))
			headers[sub] = {"header": {k: v for k, v in table.header.items() if k not in _FORMAT_KEYS},
							"types": {name: entry["type"] for name, entry in table.header["columns"].items()}}
	paths = [p for t, rows in parts for p in (t.paths[i] for i in rows)]
	order = np.array(sorted(range(len(paths)), key=paths.__getitem__), dtype=np.intp)
	paths = [paths[i] for i in order]
	columns: dict[str, Column] = {}
	for name in sorted({n.removesuffix(_TEXT) for t, _ in parts for n in t.header["columns"]}):
		kinds = set().union(*({"num", "cat"} if k == "both" else {k} for t, _ in parts if (k := _kind_in(t, name))))
		if "num" in kinds:
			columns[name] = ("num", np.concatenate([_num_in(t, name)[rows] for t, rows in parts])[order], [])
		if "cat" in kinds:
			codes, dictionary = _merge_codes([(*_codes_in(t, name), rows) for t, rows in parts])
			columns[name + _TEXT if "num" in kinds else name] = ("cat", codes[order], dictionary)
	# each subject's rows are contiguous; their order is that of "<sub>/" prefixes
	subjects: dict[str, Any] = {}
	at = 0
	for sub in sorted(headers, key=lambda sub: sub + "/"):
		stop = bisect.bisect_left(paths, sub + "0", lo=at)  # "/" < "0": past every "<sub>/..."
		subjects[sub] = {"start": at, "stop": stop, **headers[sub]}
		at = stop
	return _encode_columns(paths, columns, {"subjects": subjects}, None)


def _kind_in(table: "Table", name: str) -> str | None:
	"""A column's type in an index, where a packed index may hold it both ways."""
	if name + _TEXT in table.header["columns"]:
		return "both"
	return table.kind_of(name)


def _num_in(table: "Table", name: str) -> np.ndarray:
	return table.num(name)  # NaN where the column is text or absent


def _codes_in(table: "Table", name: str) -> tuple[np.ndarray, list[str]]:
	"""Text codes of a column, only where it is text: numeric values belong to the numeric half."""
	if name + _TEXT in table.header["columns"]:
		return table.codes(name + _TEXT)
	if table.kind_of(name) == "num":
		return np.full(table.n, -1, dtype=np.int32), []
	return table.codes(name)


class Slice(Table):
	"""One subject's rows of a packed index, read as that subject's own index: same
	header, same columns with the same types, sharing the packed buffers (no copy)."""

	def __init__(self, packed: Table, sub_id: str):
		info = packed.header["subjects"][sub_id]
		start, stop = info["start"], info["stop"]
		spec = packed.header["columns"]
		columns = {name: spec[name + _TEXT] if kind == "cat" and name + _TEXT in spec else spec[name]
				   for name, kind in info["types"].items()}
		super().__init__({**info["header"], "n": stop - start, "paths": packed.paths[start:stop],
						  "columns": columns, "buffers": {}}, packed._body)
		self._packed, self._range = packed, slice(start, stop)

	def _array(self, entry: dict[str, Any]) -> np.ndarray:
		return self._packed._array(entry)[self._range]


@dataclass
class Summary:
	"""The decoded summary, with the per-subject membership tests the planner needs."""

	table: Table

	@property
	def subjects(self) -> list[str]:
		return self.table.paths

	def digest_of(self, sub_id: str) -> str | None:
		i = self.table.find(sub_id)
		return None if i is None else self.table.values("digest", np.array([i]))[0]

	def may_contain(self, column: str, values: Iterable[str]) -> np.ndarray:
		"""Per subject: could ``column`` hold any of ``values``? Exact for columns with an
		exact value matrix; Bloom filters answer "maybe" for the rest."""
		spec = self.table.header["sets"].get(column)
		n = self.table.n
		if spec is None:
			return np.zeros(n, dtype=bool)  # no subject has this column at all
		wanted = [str(v) for v in values]
		if "values" in spec:
			lookup = {v: j for j, v in enumerate(spec["values"])}
			picked = [lookup[v] for v in wanted if v in lookup]
			if not picked:
				return np.zeros(n, dtype=bool)
			matrix = np.unpackbits(self.table.buffer(f"set:{column}"), axis=1, count=len(spec["values"])).astype(bool)
			return matrix[:, picked].any(axis=1)
		blooms = self.table.buffer(f"bloom:{column}")
		return np.array([any(bloom_may_contain(blooms[i], v) for v in wanted) for i in range(n)], dtype=bool)

	def has_column(self, column: str) -> np.ndarray:
		"""Per subject: does any of its items carry ``column``?"""
		if column == "path":
			return self.table.num("count.recording") + self.table.num("count.table") \
				+ self.table.num("count.array") + self.table.num("count.file") > 0
		spec = self.table.header["sets"].get(column)
		if spec is not None:
			if "values" in spec:
				return np.unpackbits(self.table.buffer(f"set:{column}"), axis=1, count=len(spec["values"])).any(axis=1)
			return self.table.buffer(f"bloom:{column}").any(axis=1)
		return ~np.isnan(self.table.num(f"zmin.{column}"))
