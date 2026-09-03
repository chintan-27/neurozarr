import json
import math
from pathlib import Path

import pandas as pd
import zarr


def chunk_shape(shape: tuple, itemsize: int, target_bytes: int = 30 * 1024 * 1024) -> tuple:
	"""Chunk along the last (time) axis only, targeting ~30 MB per chunk."""
	row_bytes = itemsize
	for dim in shape[:-1]:
		row_bytes *= dim
	chunk_len = min(shape[-1], max(1, target_bytes // row_bytes))
	return shape[:-1] + (chunk_len,)


def set_attrs(node, attrs: dict):
	"""Merge attrs into a zarr node's existing attributes."""
	node.attrs.put({**node.attrs.asdict(), **attrs})


def create_table(group: zarr.Group, name: str, df: pd.DataFrame, extra_attrs: dict = None):
	"""Store a DataFrame as one string array. The per-column dtypes are recorded
	in attrs so readers can cast back (see read.TableView.df) -- the array itself
	stays all-strings, which keeps ragged/mixed BIDS tsv columns storable."""
	table = group.create_array(name, data=df.astype(str).to_numpy().astype(str))
	set_attrs(table, {
		"columns": list(df.columns),
		"dtypes": [str(dt) for dt in df.dtypes],
		**(extra_attrs or {}),
	})


def load_json(path: Path) -> dict:
	return json.loads(path.read_text()) if path.exists() else {}


def clean_nan(meta_data: dict) -> dict:
	"""NaN isn't valid JSON, and zarr attrs are JSON -- swap them for None."""
	clean = {}
	for key, value in meta_data.items():
		if isinstance(value, dict):
			clean[key] = clean_nan(value)
		elif isinstance(value, float) and math.isnan(value):
			clean[key] = None
		else:
			clean[key] = value
	return clean
