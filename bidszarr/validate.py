from pathlib import Path

import pandas as pd

from .readers.bids import MNE_READABLE_EXTS

TABLE_EXTS = {".tsv", ".csv"}


def validate_manifest(reader) -> list:
	"""Check a ManifestReader's table without reading any data: required columns
	present, files exist, extensions are readable. Returns a list of problem
	strings (empty means fine)."""
	problems = []
	df = reader.df

	for col, label in ((reader.sub_col, "subject"), (reader.path_col, "file path")):
		if col not in df.columns:
			problems.append(f"manifest has no {label} column {col!r} (columns: {', '.join(map(str, df.columns))})")
	if problems:
		return problems  # nothing else is checkable without those

	for i, row in df.iterrows():
		where = f"row {i}"
		if pd.isna(row[reader.sub_col]):
			problems.append(f"{where}: empty subject id")
		path_value = row[reader.path_col]
		if pd.isna(path_value):
			problems.append(f"{where}: empty file path")
			continue
		path = Path(str(path_value))
		if not path.exists():
			problems.append(f"{where}: file not found: {path}")
		elif path.suffix not in TABLE_EXTS | MNE_READABLE_EXTS:
			problems.append(f"{where}: no reader for {path.suffix!r} ({path.name}) -- it will be stored as a reference only")
	return problems


def validate_bids(root_dir) -> list:
	"""Check a BIDS folder is shaped enough to read: it exists, has sub-* dirs,
	and (warning only) has a dataset_description.json."""
	root = Path(root_dir)
	problems = []
	if not root.exists():
		return [f"source not found: {root}"]
	if not root.is_dir():
		return [f"not a directory: {root}"]

	subjects = [d for d in root.glob("sub-*") if d.is_dir()]
	if not subjects:
		problems.append(f"no sub-* directories in {root} -- is this a BIDS dataset?")
	if not (root / "dataset_description.json").exists():
		problems.append(f"missing dataset_description.json in {root} (not fatal, but it isn't valid BIDS without one)")

	for sub in subjects:
		sessions = [d for d in sub.glob("ses-*") if d.is_dir()]
		datatype_dirs = [d for d in (sessions or [sub]) for c in d.iterdir() if c.is_dir()]
		if not datatype_dirs:
			problems.append(f"{sub.name}: no datatype directories (ieeg/, beh/, ...) found")
	return problems


def validate_source(source) -> list:
	"""Check a source before converting it, without writing anything.

	Parameters
	----------
	source : Reader or str or pathlib.Path
		A :class:`BidsReader`, a :class:`ManifestReader`, or a path — treated
		as a manifest if it ends in ``.csv`` or ``.tsv``, and as a BIDS dataset
		otherwise.

	Returns
	-------
	list of str
		One message per problem found: missing files, formats with no reader,
		missing manifest columns, or a source that isn't shaped like BIDS. An
		empty list means the source looks convertible. Messages containing
		"not fatal" are warnings rather than blockers.
	"""
	from .readers import BidsReader, ManifestReader

	if isinstance(source, ManifestReader):
		return validate_manifest(source)
	if isinstance(source, BidsReader):
		return validate_bids(source.root_dir)

	path = Path(str(source))
	if path.suffix in TABLE_EXTS:
		return validate_manifest(ManifestReader(path))
	return validate_bids(path)
