import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterator

import mne  # type: ignore[import-untyped]  # mne ships no type information
import numpy as np
import pandas as pd

from ..entities import Entities
from ..errors import ValidationError
from ..items import Attrs, ExternalFile, Item, Recording, Table
from .bids import MNE_READABLE_EXTS
from ..util import source_provenance


def _entity_value(value: Any) -> Any:
	"""A whole number reads back as run-1, not run-1.0 -- pandas widens an int
	column to float as soon as any row in it is blank."""
	if isinstance(value, np.generic):
		value = value.item()
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
				 checksum: bool = False,
				 row_reader: Callable[[Any], Item] | None = None):
		self.manifest_path = None if isinstance(manifest, pd.DataFrame) else Path(manifest)
		self.base_dir = Path.cwd() if self.manifest_path is None else self.manifest_path.resolve().parent
		self.df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
		self.path_col, self.sub_col, self.ses_col = path_col, sub_col, ses_col
		self.datatype_col, self.name_col, self.meta_col = datatype_col, name_col, meta_col
		self.entity_cols = entity_cols
		self.checksum = checksum
		self.row_reader = row_reader

		if row_reader is None:  # a row_reader handles its own columns
			missing = [c for c in (sub_col, path_col) if c not in self.df.columns]
			if missing:
				raise ValueError(
					f"manifest is missing required column(s) {missing} "
					f"(has: {', '.join(map(str, self.df.columns))}). "
					"Pass sub_col=/path_col= if your columns are named differently."
				)

	def read(self) -> Iterator[Item]:
		for _, row in self.df.iterrows():
			if self.row_reader:
				yield self.row_reader(row)
				continue

			entities = Entities(
				str(row[self.sub_col]),
				str(row.get(self.ses_col)) if pd.notna(row.get(self.ses_col)) else None,
				str(row.get(self.datatype_col)) if pd.notna(row.get(self.datatype_col)) else None,
				self._extra_entities(row),
			)
			meta = self._parse_meta(row)
			path = Path(row[self.path_col])
			if not path.is_absolute():
				path = self.base_dir / path
			name = str(row[self.name_col]) if self.name_col in row and pd.notna(row.get(self.name_col)) else path.stem
			ext = path.suffix.lower()
			meta = {**meta, "_neurozarr_provenance": source_provenance(path, "ManifestReader", self.checksum)}

			if ext in (".tsv", ".csv"):
				yield Table(entities, name, pd.read_csv(path, sep="\t" if ext == ".tsv" else ","), meta)
			elif ext in MNE_READABLE_EXTS:
				raw = mne.io.read_raw(path, preload=False, verbose=False)
				yield Recording(entities, raw, meta)
			else:
				yield ExternalFile(entities, str(name), path,
					reader_hint=f"install or register a reader for {ext!r}", meta=meta)

	def _extra_entities(self, row: Any) -> dict[str, Any]:
		reserved = {self.path_col, self.sub_col, self.ses_col, self.datatype_col, self.name_col, self.meta_col}
		cols = self.entity_cols if self.entity_cols is not None else [c for c in row.index if c not in reserved]
		return {c: _entity_value(row[c]) for c in cols if c in row and pd.notna(row[c])}

	def _parse_meta(self, row: Any) -> dict[str, Any]:
		if self.meta_col not in row:
			return {}
		val = row[self.meta_col]
		if val is None or (not isinstance(val, (dict, list, tuple)) and pd.isna(val)):
			return {}
		if isinstance(val, dict):
			return val
		if not isinstance(val, str):
			raise ValidationError("manifest metadata must be a dictionary or a JSON object string")
		decoded = json.loads(val)
		if not isinstance(decoded, dict):
			raise ValidationError("manifest metadata JSON must decode to an object")
		return decoded
