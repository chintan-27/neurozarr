"""Check a converted store against the source it came from, and write a store
back out as a BIDS folder."""

import json
from pathlib import Path

import numpy as np

from .log import logger
from .readers import BidsReader
from .repo import Repo
from .items import Recording, Table


def verify(source, store, tolerance: float = 1e-9, sample_limit: int = None) -> list:
	"""Re-read the source and compare every recording and table against what is in
	the store. Returns a list of problem strings (empty means the conversion is
	faithful). sample_limit stops after that many items, for a quick spot check."""
	from .read import RecordingView, _table_df

	repo = Repo(store)
	problems = []
	checked = 0
	roots = {}

	for item in BidsReader(source).read():
		if sample_limit is not None and checked >= sample_limit:
			break
		if not isinstance(item, (Recording, Table)):
			continue

		sub_id = item.entities.sub
		if sub_id not in roots:
			try:
				roots[sub_id] = repo.root_of(sub_id)
			except Exception as e:
				problems.append(f"{sub_id}: cannot open its repo ({e})")
				roots[sub_id] = None
		root = roots[sub_id]
		if root is None:
			continue

		# inside a subject's own repo the leading sub-XXX is dropped, as the writer does
		group_path = "/".join((*item.prefix, *item.entities.path()[1:]))
		name = "data" if isinstance(item, Recording) else item.name
		where = f"{sub_id}/{group_path}/{name}"

		if group_path not in root or name not in root[group_path]:
			problems.append(f"missing from store: {where}")
			continue
		group = root[group_path]
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


def export_bids(store, dest, subjects: list = None) -> Path:
	"""Write a store back out as a BIDS folder: recordings as BrainVision (a format
	mne always writes without extra dependencies), tables as .tsv, metadata as JSON
	sidecars. Useful for handing data to tools that only speak BIDS on disk."""
	import mne

	repo = Repo(store)
	dest = Path(dest)
	dest.mkdir(parents=True, exist_ok=True)

	try:
		description = repo.root_of("_dataset").attrs.asdict()
	except Exception:
		description = {}
	description.setdefault("Name", dest.name)
	description.setdefault("BIDSVersion", "1.10.0")
	(dest / "dataset_description.json").write_text(json.dumps(description, indent=2, default=str))

	for sub_id in (subjects or repo.subjects()):
		subject = repo.subject(sub_id)
		for view in subject.recordings():
			out_dir = dest / _bids_dir(view.path)
			out_dir.mkdir(parents=True, exist_ok=True)
			stem = _bids_stem(view.path)
			_write_raw(out_dir, stem, view.raw())
			meta = {k: v for k, v in view.meta.items() if not k.startswith("data_")}
			(out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2, default=str))
		for view in subject.tables():
			out_dir = dest / _bids_dir(view.path)
			out_dir.mkdir(parents=True, exist_ok=True)
			name = f"{_bids_stem(view.path)}_{view.name}" if view.entities else view.name
			view.df().to_csv(out_dir / f"{name}.tsv", sep="\t", index=False)
	return dest


def _write_raw(out_dir: Path, stem: str, raw) -> Path:
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
