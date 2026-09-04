import json
from pathlib import Path

import mne
import pandas as pd

from ..entities import Entities
from ..items import Attrs, Recording, Table
from .bids import MNE_READABLE_EXTS


def _entity_value(value):
	"""A whole number reads back as run-1, not run-1.0 -- pandas widens an int
	column to float as soon as any row in it is blank."""
	if isinstance(value, float) and value.is_integer():
		return int(value)
	return value


class ManifestReader:
	"""Reads arbitrary data described by a manifest table (DataFrame or CSV path):
	one row per file, with columns naming the subject/session/datatype/entities and
	a path to the file. No assumption about folder layout or naming convention --
	the caller supplies the mapping. Column names are configurable via ``*_col``
	kwargs so an existing table doesn't need to be renamed first. Dispatch by file
	extension mirrors BidsReader: .tsv/.csv -> Table, MNE-readable -> Recording,
	anything else -> a non-crashing Attrs fallback. A row_reader callback bypasses
	all of this per-row for fully custom handling."""

	def __init__(self, manifest, *, path_col="path", sub_col="sub", ses_col="ses",
				 datatype_col="datatype", name_col="name", meta_col="meta",
				 entity_cols: list = None, row_reader=None):
		self.df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
		self.path_col, self.sub_col, self.ses_col = path_col, sub_col, ses_col
		self.datatype_col, self.name_col, self.meta_col = datatype_col, name_col, meta_col
		self.entity_cols = entity_cols
		self.row_reader = row_reader

		if row_reader is None:  # a row_reader handles its own columns
			missing = [c for c in (sub_col, path_col) if c not in self.df.columns]
			if missing:
				raise ValueError(
					f"manifest is missing required column(s) {missing} "
					f"(has: {', '.join(map(str, self.df.columns))}). "
					"Pass sub_col=/path_col= if your columns are named differently."
				)

	def read(self):
		for _, row in self.df.iterrows():
			if self.row_reader:
				yield self.row_reader(row)
				continue

			entities = Entities(
				row[self.sub_col],
				row.get(self.ses_col) if pd.notna(row.get(self.ses_col)) else None,
				row.get(self.datatype_col) if pd.notna(row.get(self.datatype_col)) else None,
				self._extra_entities(row),
			)
			meta = self._parse_meta(row)
			path = Path(row[self.path_col])
			name = row[self.name_col] if self.name_col in row and pd.notna(row.get(self.name_col)) else path.stem
			ext = path.suffix

			if ext in (".tsv", ".csv"):
				yield Table(entities, name, pd.read_csv(path, sep="\t" if ext == ".tsv" else ","), meta)
			elif ext in MNE_READABLE_EXTS:
				raw = mne.io.read_raw(path, preload=True, verbose=False)
				yield Recording(entities, raw, meta)
			else:
				yield Attrs((*entities.path(), name), {
					**meta, "unread_file": str(path),
					"note": f"no in-memory reader registered for {ext!r}",
				})

	def _extra_entities(self, row) -> dict:
		reserved = {self.path_col, self.sub_col, self.ses_col, self.datatype_col, self.name_col, self.meta_col}
		cols = self.entity_cols if self.entity_cols is not None else [c for c in row.index if c not in reserved]
		return {c: _entity_value(row[c]) for c in cols if c in row and pd.notna(row[c])}

	def _parse_meta(self, row) -> dict:
		if self.meta_col not in row or pd.isna(row[self.meta_col]):
			return {}
		val = row[self.meta_col]
		return val if isinstance(val, dict) else json.loads(val)
