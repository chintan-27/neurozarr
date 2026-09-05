"""Check a converted store against the source it came from, and write a store
back out as a BIDS folder."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .log import logger
from .readers import BidsReader
from .repo import Repo
from .items import Recording, Table

if TYPE_CHECKING:
	import icechunk
	import mne  # type: ignore[import-untyped]  # mne ships no type information
	import zarr


def verify(source: "str | Path", store: "str | Path | icechunk.Storage",
		   tolerance: float = 1e-9, sample_limit: int | None = None) -> list[str]:
	"""Check that a store faithfully matches the source it was converted from.

	Re-reads the source and compares each recording's samples and each table's
	row count against what the store holds.

	Parameters
	----------
	source : str or pathlib.Path
		The BIDS dataset the store was converted from.
	store : str or pathlib.Path or icechunk.Storage
		The store to check.
	tolerance : float, default 1e-9
		Absolute tolerance when comparing sample values.
	sample_limit : int, optional
		Stop after checking this many items, for a quick spot check. Default
		checks everything.

	Returns
	-------
	list of str
		One message per problem found: data missing from the store, a shape
		mismatch, differing values, or a differing row count. An empty list
		means the conversion is faithful.
	"""
	from .read import RecordingView, _table_df

	repo = Repo.open(store)
	problems: list[str] = []
	checked = 0
	roots: dict[str, "zarr.Group | None"] = {}

	for item in BidsReader(source).read():
		if sample_limit is not None and checked >= sample_limit:
			break
		if not isinstance(item, (Recording, Table)):
			continue

		sub_id = item.entities.sub
		if sub_id not in roots:
			try:
				roots[sub_id] = repo.root_of(sub_id)
			except Exception:
				# nothing committed for this subject -- report its items as missing,
				# which is more useful than one opaque "cannot open" line
				roots[sub_id] = None
		root = roots[sub_id]

		# inside a subject's own repo the leading sub-XXX is dropped, as the writer does
		group_path = "/".join((*item.prefix, *item.entities.path()[1:]))
		name = "data" if isinstance(item, Recording) else item.name
		where = f"{sub_id}/{group_path}/{name}"

		if root is None or group_path not in root:
			problems.append(f"missing from store: {where}")
			continue
		group = cast("zarr.Group", root[group_path])
		if name not in group:
			problems.append(f"missing from store: {where}")
			continue
		checked += 1

		if isinstance(item, Recording):
			values, _ = RecordingView(group, item.entities.extra, where).data()
			expected = item.raw.get_data()
			if values.shape != expected.shape:
				problems.append(f"{where}: shape {values.shape} != source {expected.shape}")
			elif not np.allclose(values, expected, atol=tolerance):
				problems.append(f"{where}: values differ (max abs error {np.abs(values - expected).max():.3e})")
		else:
			stored_rows = len(_table_df(group[name]))
			if stored_rows != len(item.df):
				problems.append(f"{where}: {stored_rows} rows != source {len(item.df)}")

	logger.info("verified %d items against %s", checked, source)
	return problems


def export_bids(store: "str | Path | icechunk.Storage", dest: "str | Path",
				subjects: list[str] | None = None) -> Path:
	"""Write a store back out as a BIDS dataset on disk.

	Recordings become signal files, tables become ``.tsv``, and metadata becomes
	JSON sidecars. Use it to hand data to tools that only read BIDS from a
	filesystem.

	Recordings are written as BrainVision if ``pybv`` is installed, EDF if
	``edfio`` is, and otherwise FIF, which mne reads but which is not
	BIDS-conformant for ieeg or eeg data. A warning is logged in that case.

	Parameters
	----------
	store : str or pathlib.Path or icechunk.Storage
		The store to export.
	dest : str or pathlib.Path
		Directory to write the dataset into. Created if it does not exist.
	subjects : list of str, optional
		Export only these subjects. Default exports all of them.

	Returns
	-------
	pathlib.Path
		The dataset directory that was written.
	"""
	import mne

	repo = Repo.open(store)
	dest = Path(dest)
	dest.mkdir(parents=True, exist_ok=True)

	try:
		description = repo.root_of("_dataset").attrs.asdict()
	except Exception:
		description = {}
	description = {key: value for key, value in description.items()
				   if not key.startswith("_neurozarr") and key != "subjects"}
	description.setdefault("Name", dest.name)
	description.setdefault("BIDSVersion", "1.10.0")
	(dest / "dataset_description.json").write_text(json.dumps(description, indent=2, default=str))

	for sub_id in (subjects or repo.subjects()):
		subject = repo.subject(sub_id)
		for recording in subject.recordings():
			out_dir = dest / _bids_dir(recording.path)
			out_dir.mkdir(parents=True, exist_ok=True)
			stem = _bids_stem(recording.path)
			_write_raw(out_dir, stem, recording.raw())
			meta = {k: v for k, v in recording.meta.items()
					if not k.startswith("data_") and not k.startswith("_neurozarr")}
			(out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2, default=str))
		for table in subject.tables():
			out_dir = dest / _bids_dir(table.path)
			out_dir.mkdir(parents=True, exist_ok=True)
			name = f"{_bids_stem(table.path)}_{table.name}" if table.entities else table.name
			table.df().to_csv(out_dir / f"{name}.tsv", sep="\t", index=False)
	return dest


def _write_raw(out_dir: Path, stem: str, raw: "mne.io.BaseRaw") -> Path:
	"""Write a Raw in the most BIDS-appropriate format available.

	BrainVision and EDF are what BIDS wants for ieeg/eeg, but mne needs pybv and
	edfio to write them. If neither is installed, fall back to FIF -- readable by
	mne, though not BIDS-conformant for these datatypes.
	"""
	import mne

	for suffix in (".vhdr", ".edf"):
		try:
			path = out_dir / f"{stem}{suffix}"
			mne.export.export_raw(path, raw, overwrite=True, verbose=False)
			return path
		except (RuntimeError, ValueError) as e:
			logger.debug("cannot export %s: %s", suffix, e)

	logger.warning("writing %s as FIF -- install pybv or edfio for BIDS-conformant output", stem)
	path = out_dir / f"{stem}_raw.fif"
	raw.save(path, overwrite=True, verbose=False)
	return path


def _bids_dir(view_path: str) -> Path:
	"""sub-001/ses-1/ieeg/task-X_run-1 -> sub-001/ses-1/ieeg"""
	return Path("/".join(view_path.split("/")[:-1]))


def _bids_stem(view_path: str) -> str:
	"""sub-001/ses-1/ieeg/task-X_run-1 -> sub-001_ses-1_task-X_run-1"""
	parts = view_path.split("/")
	entities = parts[-1]
	prefix = [p for p in parts[:-1] if p.startswith(("sub-", "ses-"))]
	return "_".join(prefix + ([entities] if entities else []))
