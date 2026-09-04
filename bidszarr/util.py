import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import zarr


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
		row_bytes *= dim
	by_size = max(1, target_bytes // row_bytes)
	chunk_len = min(shape[-1], by_size, max_samples)
	return shape[:-1] + (chunk_len,)


def set_attrs(node: zarr.Group | zarr.Array, attrs: dict[str, Any]) -> None:
	"""Merge attrs into a zarr node's existing attributes."""
	node.attrs.put({**node.attrs.asdict(), **attrs})


def create_table(group: zarr.Group, name: str, df: pd.DataFrame,
				 extra_attrs: dict[str, Any] | None = None) -> None:
	"""Store a DataFrame as one string array, recording each column's dtype in
	attrs so readers cast the values back faithfully (see ``read._table_df``).
	"""
	# One zarr array per column was measured and rejected: BIDS datasets are
	# mostly many small tables (tens to hundreds of rows), so per-array metadata
	# costs more than typed values save -- it made a real dataset 33% larger
	# (144MB -> 192MB). Readers still understand that layout if it turns up.
	table = group.create_array(name, data=df.astype(str).to_numpy().astype(str), overwrite=True)
	set_attrs(table, {
		"columns": [str(c) for c in df.columns],
		"dtypes": [str(dt) for dt in df.dtypes],
		**(extra_attrs or {}),
	})


def load_json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text()) if path.exists() else {}


def clean_nan(meta_data: dict[str, Any]) -> dict[str, Any]:
	"""NaN isn't valid JSON, and zarr attrs are JSON -- swap them for None."""
	clean: dict[str, Any] = {}
	for key, value in meta_data.items():
		if isinstance(value, dict):
			clean[key] = clean_nan(value)
		elif isinstance(value, float) and math.isnan(value):
			clean[key] = None
		else:
			clean[key] = value
	return clean
