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


def create_table(group: zarr.Group, name: str, df: pd.DataFrame,
				 extra_attrs: dict[str, Any] | None = None) -> None:
	"""Store a DataFrame column-wise with a lossless logical dtype schema."""
	table = group.create_group(name, overwrite=True)
	schema: list[dict[str, Any]] = []
	for position in range(len(df.columns)):
		series = df.iloc[:, position]
		internal = f"c{position:06d}"
		logical = str(series.dtype)
		mask = series.isna().to_numpy(dtype=bool)
		encoding = "native"

		if isinstance(series.dtype, pd.CategoricalDtype):
			values = series.cat.codes.to_numpy(dtype=np.int64)
			encoding = "categorical"
			categories = series.cat.categories.tolist()
			try:
				json.dumps(categories, allow_nan=False)
			except (TypeError, ValueError) as exc:
				raise ValidationError(f"table column {series.name!r} has non-JSON categorical labels") from exc
		elif pd.api.types.is_datetime64_any_dtype(series.dtype):
			values = series.to_numpy(dtype="datetime64[ns]").view("int64")
			encoding = "datetime64[ns]"
			categories = None
		elif pd.api.types.is_timedelta64_dtype(series.dtype):
			values = series.to_numpy(dtype="timedelta64[ns]").view("int64")
			encoding = "timedelta64[ns]"
			categories = None
		elif pd.api.types.is_integer_dtype(series.dtype):
			integer_dtype: Any = getattr(series.dtype, "numpy_dtype", series.dtype)
			values = series.to_numpy(dtype=integer_dtype, na_value=0)
			encoding = "integer"
			categories = None
		elif pd.api.types.is_bool_dtype(series.dtype):
			values = series.to_numpy(dtype=bool, na_value=False)
			encoding = "boolean"
			categories = None
		elif pd.api.types.is_float_dtype(series.dtype):
			float_dtype: Any = getattr(series.dtype, "numpy_dtype", series.dtype)
			values = series.to_numpy(dtype=float_dtype, na_value=np.nan)
			encoding = "float"
			categories = None
		elif pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
			non_missing = series[~series.isna()]
			if not non_missing.map(lambda value: isinstance(value, str)).all():
				raise ValidationError(
					f"table column {series.name!r} has unsupported object values; use a concrete scalar dtype"
				)
			values = series.fillna("").astype(str).to_numpy(dtype=str)
			encoding = "string"
			categories = None
		else:
			raise ValidationError(f"table column {series.name!r} has unsupported dtype {series.dtype}")

		table.create_array(internal, data=np.asarray(values))
		if mask.any():
			table.create_array(f"{internal}__mask", data=mask)
		entry: dict[str, Any] = {
			"id": internal,
			"name": str(df.columns[position]),
			"dtype": logical,
			"encoding": encoding,
			"nullable": bool(mask.any()),
		}
		if encoding == "categorical":
			entry["categories"] = categories
			entry["ordered"] = bool(series.cat.ordered)
		schema.append(entry)

	set_attrs(table, {
		"_neurozarr_item_type": "table",
		"table_schema_version": 2,
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
