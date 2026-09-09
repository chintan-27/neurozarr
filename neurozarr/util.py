import asyncio
import json
import hashlib
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import zarr

from .errors import ValidationError


def chunk_shape(shape: tuple[int, ...], itemsize: int, target_bytes: int = 8 * 1024 * 1024,
				max_samples: int = 65536) -> tuple[int, ...]:
	"""Pick a chunk shape, dividing along the last (time) axis only.

	Two limits apply, whichever is smaller: a byte target, which keeps chunks
	reasonable for wide many-channel arrays, and a cap on samples per chunk,
	which is what keeps windowed reads cheap.
	"""
	# The sample cap matters because a byte target alone puts an entire
	# multi-hour 2-channel recording in a single chunk, making a 10 s read cost
	# the whole array. Measured on a 2ch/927k-sample recording: at 65536 samples
	# a 10 s window read 6x faster than a single chunk and full reads ~2x faster,
	# for ~14% more storage; dropping to 16k made full reads 3x slower.
	row_bytes = itemsize
	for dim in shape[:-1]:
		row_bytes *= max(1, dim)
	by_size = max(1, target_bytes // row_bytes)
	chunk_len = max(1, min(shape[-1], by_size, max_samples))
	return tuple(max(1, dim) for dim in shape[:-1]) + (chunk_len,)


def set_attrs(node: zarr.Group | zarr.Array, attrs: dict[str, Any]) -> None:
	"""Merge attrs into a zarr node's existing attributes."""
	node.attrs.put({**node.attrs.asdict(), **attrs})


def _column_arrays(df: pd.DataFrame) -> list[dict[str, Any]]:
	"""Convert every column to its physical numpy representation, unchanged from
	before packing existed -- this is purely the per-column dtype/encoding logic."""
	columns: list[dict[str, Any]] = []
	for position in range(len(df.columns)):
		series = df.iloc[:, position]
		logical = str(series.dtype)
		mask = series.isna().to_numpy(dtype=bool)
		categories = None

		if isinstance(series.dtype, pd.CategoricalDtype):
			values = series.cat.codes.to_numpy(dtype=np.int64)
			encoding = "categorical"
			categories = series.cat.categories.tolist()
			try:
				json.dumps(categories, allow_nan=False)
			except (TypeError, ValueError) as exc:
				raise ValidationError(f"table column {series.name!r} has non-JSON categorical labels") from exc
			ordered = bool(series.cat.ordered)
		elif pd.api.types.is_datetime64_any_dtype(series.dtype):
			values = series.to_numpy(dtype="datetime64[ns]").view("int64")
			encoding = "datetime64[ns]"
			ordered = False
		elif pd.api.types.is_timedelta64_dtype(series.dtype):
			values = series.to_numpy(dtype="timedelta64[ns]").view("int64")
			encoding = "timedelta64[ns]"
			ordered = False
		elif pd.api.types.is_integer_dtype(series.dtype):
			integer_dtype: Any = getattr(series.dtype, "numpy_dtype", series.dtype)
			values = series.to_numpy(dtype=integer_dtype, na_value=0)
			encoding = "integer"
			ordered = False
		elif pd.api.types.is_bool_dtype(series.dtype):
			values = series.to_numpy(dtype=bool, na_value=False)
			encoding = "boolean"
			ordered = False
		elif pd.api.types.is_float_dtype(series.dtype):
			float_dtype: Any = getattr(series.dtype, "numpy_dtype", series.dtype)
			values = series.to_numpy(dtype=float_dtype, na_value=np.nan)
			encoding = "float"
			ordered = False
		elif pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
			non_missing = series[~series.isna()]
			if not non_missing.map(lambda value: isinstance(value, str)).all():
				raise ValidationError(
					f"table column {series.name!r} has unsupported object values; use a concrete scalar dtype"
				)
			values = series.fillna("").astype(str).to_numpy(dtype=str)
			encoding = "string"
			ordered = False
		else:
			raise ValidationError(f"table column {series.name!r} has unsupported dtype {series.dtype}")

		columns.append({
			"name": str(df.columns[position]), "values": np.asarray(values), "mask": mask,
			"logical": logical, "encoding": encoding, "categories": categories, "ordered": ordered,
		})
	return columns


def create_table(group: zarr.Group, name: str, df: pd.DataFrame,
				 extra_attrs: dict[str, Any] | None = None) -> None:
	"""Store a DataFrame with a lossless logical dtype schema.

	Columns that share a physical dtype and nullability are packed into one
	shared 2D array instead of each getting its own -- icechunk has a real
	per-array bookkeeping cost that dominates writing many small columns
	(measured ~4x faster on a realistic BIDS sidecar table shape). Decoding a
	column only ever looks at its own schema entry and never depends on what,
	if anything, it was packed alongside, so this changes nothing about what
	comes back on read.
	"""
	table = group.create_group(name, overwrite=True)
	columns = _column_arrays(df)

	groups: dict[tuple[str, bool], list[int]] = {}
	for i, col in enumerate(columns):
		if col["encoding"] == "categorical":
			continue  # each has its own categories/ordered metadata; never packed
		key = ("string", bool(col["mask"].any())) if col["encoding"] == "string" \
			else (str(col["values"].dtype), bool(col["mask"].any()))
		groups.setdefault(key, []).append(i)

	schema: list[dict[str, Any] | None] = [None] * len(columns)
	writes: list[tuple[str, np.ndarray]] = []
	packed: set[int] = set()
	for group_index, (key, indices) in enumerate(g for g in groups.items() if len(g[1]) >= 2):
		has_mask = key[1]
		packed.update(indices)
		array_name = f"g{group_index:06d}"
		cols = [columns[i] for i in indices]
		writes.append((array_name, np.array([c["values"] for c in cols])))
		if has_mask:
			writes.append((f"{array_name}__mask", np.array([c["mask"] for c in cols])))
		for row, i in enumerate(indices):
			c = columns[i]
			schema[i] = {
				"array": array_name, "row": row, "name": c["name"], "dtype": c["logical"],
				"encoding": c["encoding"], "nullable": bool(c["mask"].any()),
			}

	for i, c in enumerate(columns):
		if i in packed:
			continue
		internal = f"c{i:06d}"
		writes.append((internal, c["values"]))
		if c["mask"].any():
			writes.append((f"{internal}__mask", c["mask"]))
		entry: dict[str, Any] = {
			"id": internal, "name": c["name"], "dtype": c["logical"],
			"encoding": c["encoding"], "nullable": bool(c["mask"].any()),
		}
		if c["encoding"] == "categorical":
			entry["categories"] = c["categories"]
			entry["ordered"] = c["ordered"]
		schema[i] = entry

	if writes:
		# ponytail: relies on zarr's private _async_group/_sync (the same thing
		# Group.create_array calls internally) since zarr has no public bulk-create
		# API; switch to one if zarr ever adds it.
		async def _create_all() -> None:
			await asyncio.gather(*(
				table._async_group.create_array(array_name, data=data) for array_name, data in writes
			))
		table._sync(_create_all())

	set_attrs(table, {
		"_neurozarr_item_type": "table",
		"table_schema_version": 3,
		"columns": [str(c) for c in df.columns],
		"schema": schema,
		**(extra_attrs or {}),
	})


def safe_uri(uri: str) -> str:
	"""Remove credentials, query strings, and fragments from a stored URI."""
	parts = urlsplit(uri)
	if not parts.scheme:
		return uri
	try:
		port = f":{parts.port}" if parts.port else ""
	except ValueError as exc:
		raise ValidationError(f"invalid source URI {uri!r}: {exc}") from exc
	host = f"{parts.hostname or ''}{port}"
	return urlunsplit((parts.scheme, host, parts.path, "", ""))


def load_json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text()) if path.exists() else {}


def clean_nan(meta_data: dict[str, Any]) -> dict[str, Any]:
	"""NaN isn't valid JSON, and zarr attrs are JSON -- swap them for None."""
	def clean_value(value: Any) -> Any:
		if isinstance(value, np.generic):
			return clean_value(value.item())
		if isinstance(value, dict):
			return {str(key): clean_value(item) for key, item in value.items()}
		if isinstance(value, (list, tuple)):
			return [clean_value(item) for item in value]
		if isinstance(value, float) and math.isnan(value):
			return None
		return value

	return {str(key): clean_value(value) for key, value in meta_data.items()}


def source_provenance(path: Path, adapter: str, checksum: bool = False) -> dict[str, Any]:
	"""Return inexpensive, JSON-safe source provenance without credentials."""
	result: dict[str, Any] = {"source": safe_uri(str(path)), "adapter": adapter}
	try:
		stat = path.stat()
	except OSError:
		return result
	result.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
	if checksum and path.is_file():
		digest = hashlib.sha256()
		with path.open("rb") as stream:
			for block in iter(lambda: stream.read(1024 * 1024), b""):
				digest.update(block)
		result["sha256"] = digest.hexdigest()
	return result
