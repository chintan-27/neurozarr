import copy
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
from ..log import log_progress
from .bids import MNE_READABLE_EXTS
from ..codecs import CodecPlan, check_plan_options, plan_codecs
from .formats import UnclaimedPolicy, decode_embed, decoder_for
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

	A row with no value in ``path_col`` carries no file -- it describes
	dataset/subject/session-level metadata instead, the way ``dataset_description.json``
	or ``participants.tsv`` does for :class:`BidsReader`. Its :class:`~neurozarr.Attrs`
	path is built from whichever of sub/ses/datatype/name are filled in on that row
	(blank columns are skipped; an all-blank row means dataset-wide), and ``meta_col``
	becomes that item's attrs dict directly, not metadata merged onto a file.

	Parameters
	----------
	manifest : pandas.DataFrame or str or pathlib.Path
		The table itself, or a path to it: ``.json`` reads via
		:func:`pandas.read_json` (records-oriented, e.g. ``[{"sub": ..., "path": ...,
		"meta": {...}}, ...]`` -- ``meta`` can be a real nested object, not a JSON
		string), anything else via :func:`pandas.read_csv`.
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
	unclaimed : {"reference", "embed"}, default "reference"
		What to do with a file no built-in or registered format decoder can
		read: ``"reference"`` records its path as an
		:class:`~neurozarr.ExternalFile`, content untouched; ``"embed"`` reads
		its raw bytes into the store instead. See
		:func:`neurozarr.readers.register_format` to add real support for a
		format rather than embedding or referencing it.
	row_reader : callable, optional
		Called with each row, returning the item to store. Bypasses all
		column handling above, for sources that need custom logic.
	optimize : {"balanced", "speed", "size"}, default "balanced"
		How :meth:`codec_plan` trades size against speed: Blosc lz4 (fastest),
		zstd level 5, or zstd level 9 (smallest), each with the shuffle mode that
		probing finds best for the data.
	probe_level : {0, 1, 2, 3, 4}, default 2
		How finely :meth:`codec_plan` groups data, one codec decision per: 0,
		nothing (zarr's default); 1, item type; 2, item type x stored dtype; 3,
		item type x stored dtype x chunk shape; 4, each array, probed as it is
		written. See :mod:`neurozarr.codecs`.
	probe_files : int, default 8
		Arrays probed per group, from files spread across subjects.

	Raises
	------
	ValueError
		If the subject or path column is missing, unless ``row_reader`` is
		given. The message lists the columns the table does have.

	Examples
	--------
	>>> repo.add(ManifestReader("manifest.csv"))
	>>> repo.add(ManifestReader(df, sub_col="subject", path_col="filepath"))
	"""

	def __init__(self, manifest: "pd.DataFrame | str | Path", *, path_col: str = "path",
				 sub_col: str = "sub", ses_col: str = "ses", datatype_col: str = "datatype",
				 name_col: str = "name", meta_col: str = "meta",
				 entity_cols: list[str] | None = None,
				 checksum: bool = False,
				 unclaimed: UnclaimedPolicy | str = UnclaimedPolicy.REFERENCE,
				 row_reader: Callable[[Any], Item] | None = None, optimize: str = "balanced",
				 probe_level: int = 2, probe_files: int = 8):
		self.manifest_path = None if isinstance(manifest, pd.DataFrame) else Path(manifest)
		self.base_dir = Path.cwd() if self.manifest_path is None else self.manifest_path.resolve().parent
		if isinstance(manifest, pd.DataFrame):
			self.df = manifest
		# Entity values are labels, not numbers: left to infer types, pandas turns
		# a column of "01", "02" into 1, 2 and the recording lands under run-1
		# instead of run-01 (and acq_time-style column names get parsed as dates).
		elif Path(manifest).suffix.lower() == ".json":
			self.df = pd.read_json(manifest, dtype=False, convert_dates=False)
		else:
			self.df = pd.read_csv(manifest, dtype=str)
		self.path_col, self.sub_col, self.ses_col = path_col, sub_col, ses_col
		self.datatype_col, self.name_col, self.meta_col = datatype_col, name_col, meta_col
		self.entity_cols = entity_cols
		self.checksum = checksum
		self.unclaimed = UnclaimedPolicy(unclaimed)
		self.row_reader = row_reader
		check_plan_options(optimize, probe_level)
		self.optimize, self.probe_level, self.probe_files = optimize, probe_level, probe_files

		if row_reader is None:  # a row_reader handles its own columns
			missing = [c for c in (sub_col, path_col) if c not in self.df.columns]
			if missing:
				raise ValueError(
					f"manifest is missing required column(s) {missing} "
					f"(has: {', '.join(map(str, self.df.columns))}). "
					"Pass sub_col=/path_col= if your columns are named differently."
				)

	def codec_plan(self, layout: Any = None) -> "CodecPlan":
		"""Probe a few files and decide each kind of data's codec; see :mod:`neurozarr.codecs`."""
		return plan_codecs(self, layout, self.optimize, self.probe_level, self.probe_files)

	def subjects(self) -> list[str] | None:
		"""The subjects named in the table, so :meth:`Repo.add` can split the work.
		``None`` with a ``row_reader``, whose items can't be known without reading."""
		if self.row_reader is not None or self.sub_col not in self.df.columns:
			return None
		return sorted({str(v) for v in self.df[self.sub_col].dropna()})

	def for_subject(self, sub: str | None) -> "ManifestReader":
		"""This reader limited to one subject's rows; ``None`` keeps the rows with no
		subject, which are dataset-level metadata."""
		column = self.df[self.sub_col]
		limited = copy.copy(self)
		limited.df = self.df[column.isna()] if sub is None else self.df[column.astype(str) == sub]
		return limited

	def read(self) -> Iterator[Item]:
		log_progress("Reading manifest", str(self.manifest_path) if self.manifest_path else "(inline DataFrame)",
			f"{len(self.df)} rows")
		for _, row in self.df.iterrows():
			if self.row_reader:
				yield self.row_reader(row)
				continue

			if pd.isna(row.get(self.path_col)):
				yield Attrs(self._attrs_path(row), self._parse_meta(row))
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

			decoder = decoder_for(path)
			if ext in (".tsv", ".csv"):
				yield Table(entities, name, pd.read_csv(path, sep="\t" if ext == ".tsv" else ","), meta)
			elif ext in MNE_READABLE_EXTS:
				raw = mne.io.read_raw(path, preload=False, verbose=False)
				yield Recording(entities, raw, meta)
			elif decoder is not None:
				yield from decoder(path, entities, name, meta)
			elif self.unclaimed is UnclaimedPolicy.EMBED:
				yield from decode_embed(path, entities, name, meta)
			else:
				yield ExternalFile(entities, str(name), path,
					reader_hint=f"install or register a reader for {ext!r}", meta=meta)

	def _attrs_path(self, row: Any) -> tuple[str, ...]:
		"""A pathless row's Attrs path: whichever of sub/ses/datatype/name are
		present, in that order -- an empty tuple means dataset-wide, matching
		BidsReader's dataset_description.json handling."""
		return tuple(
			str(row[col]) for col in (self.sub_col, self.ses_col, self.datatype_col, self.name_col)
			if col in row and pd.notna(row.get(col))
		)

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
