import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterator

import mne  # type: ignore[import-untyped]  # mne ships no type information
import pandas as pd

from ..entities import Entities
from ..items import Attrs, Recording, Table
from .bids import MNE_READABLE_EXTS


def _entity_value(value: Any) -> Any:
	"""A whole number reads back as run-1, not run-1.0 -- pandas widens an int
	column to float as soon as any row in it is blank."""
	if isinstance(value, float) and value.is_integer():
		return int(value)
	return value


class ManifestReader:
	"""Read data described by a manifest table, one row per file.

	Use this when the data has no standard layout: instead of inferring
	structure from directory names, you describe each file in a table and say
	which columns mean what. Nothing is assumed about folder layout or naming.

	Files are read by extension, as :class:`BidsReader` does: ``.tsv``/``.csv``
	with pandas, and EDF, BDF, GDF, BrainVision, EEGLAB, FIF and CNT through
	mne. A file in any other format is recorded as a reference to its path
	rather than failing the run.

	Parameters
	----------
	manifest : pandas.DataFrame or str or pathlib.Path
		The table itself, or a path to a CSV of it.
	path_col, sub_col : str, default "path", "sub"
		Columns holding the file path and the subject label. Both are required
		unless ``row_reader`` is given.
	ses_col, datatype_col, name_col, meta_col : str
		Columns holding the session label, BIDS datatype, stored name, and
		per-row metadata. Metadata may be a dict or a JSON string. Each is
		optional in the table; only the column *name* is configured here.
	entity_cols : list of str, optional
		Columns to treat as BIDS entities. By default every column that isn't
		one of the six named above becomes an entity, so a ``task`` or ``run``
		column lands where it should.
	row_reader : callable, optional
		Called with each row, returning the item to store. Bypasses all
		column handling above, for sources that need custom logic.

	Raises
	------
	ValueError
		If the subject or path column is missing, unless ``row_reader`` is
		given. The message lists the columns the table does have.

	Examples
	--------
	>>> repo.ingest(ManifestReader("manifest.csv"))
	>>> repo.ingest(ManifestReader(df, sub_col="subject", path_col="filepath"))
	"""

	def __init__(self, manifest: "pd.DataFrame | str | Path", *, path_col: str = "path",
				 sub_col: str = "sub", ses_col: str = "ses", datatype_col: str = "datatype",
				 name_col: str = "name", meta_col: str = "meta",
				 entity_cols: list[str] | None = None,
				 row_reader: Callable[[Any], Recording | Table | Attrs] | None = None):
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

	def read(self) -> Iterator[Recording | Table | Attrs]:
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

	def _extra_entities(self, row: Any) -> dict[str, Any]:
		reserved = {self.path_col, self.sub_col, self.ses_col, self.datatype_col, self.name_col, self.meta_col}
		cols = self.entity_cols if self.entity_cols is not None else [c for c in row.index if c not in reserved]
		return {c: _entity_value(row[c]) for c in cols if c in row and pd.notna(row[c])}

	def _parse_meta(self, row: Any) -> dict[str, Any]:
		if self.meta_col not in row or pd.isna(row[self.meta_col]):
			return {}
		val = row[self.meta_col]
		return val if isinstance(val, dict) else json.loads(val)
