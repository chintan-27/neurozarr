"""Check a converted store against the source it came from, and write a store
back out as a BIDS folder."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import zarr

from .log import logger
from .readers import reader_for
from .repo import Repo
from .items import Attrs, Recording, Table

if TYPE_CHECKING:
	import icechunk
	import mne  # type: ignore[import-untyped]  # mne ships no type information


def verify(source: "str | Path", store: "str | Path | icechunk.Storage",
		   tolerance: float = 1e-9, sample_limit: int | None = None,
		   reader: str | None = None) -> list[str]:
	"""Check that a store faithfully matches the source it was converted from.

	Re-reads the source and compares each recording sample-for-sample, each
	table cell-for-cell (values, dtypes, column names, and nulls), and each
	item's sidecar metadata against what the store holds.

	Parameters
	----------
	source : str or pathlib.Path
		The dataset the store was converted from -- a BIDS directory or a
		manifest .csv/.tsv, picked the same way ``neurozarr convert`` does.
	store : str or pathlib.Path or icechunk.Storage
		The store to check.
	tolerance : float, default 1e-9
		Absolute tolerance when comparing sample values.
	sample_limit : int, optional
		Stop after checking this many items, for a quick spot check. Default
		checks everything.
	reader : str, optional
		Explicit reader name, for a source ``reader_for`` can't tell apart on
		its own (mirrors ``convert``'s ``--reader``).

	Returns
	-------
	list of str
		One message per problem found: data missing from the store, a shape
		mismatch, a differing value, dtype, column, or a metadata key that
		doesn't match. An empty list means the conversion is faithful.
	"""
	from .read import RecordingView, _table_df

	repo = Repo.open(store)
	problems: list[str] = []
	checked = 0
	roots: dict[str, "zarr.Group | None"] = {}

	def get_root(sub_id: str) -> "zarr.Group | None":
		if sub_id not in roots:
			try:
				roots[sub_id] = repo.root_of(sub_id)
			except Exception:
				# nothing committed for this subject -- report its items as missing,
				# which is more useful than one opaque "cannot open" line
				roots[sub_id] = None
		return roots[sub_id]

	for item in reader_for(source, reader).read():
		if sample_limit is not None and checked >= sample_limit:
			break

		if isinstance(item, Attrs):
			# Same routing rule as Repo._route_attrs: a "sub-XXX" segment anywhere
			# in the path picks that subject's repo and is dropped; no such
			# segment means the shared _dataset repo.
			sub_id, stripped = Repo._route(item.path)
			where = "/".join((sub_id, *stripped)) if stripped else sub_id
			root = get_root(sub_id)
			node = _navigate(root, stripped) if root is not None else None
			if node is None:
				problems.append(f"missing from store: {where}")
				continue
			checked += 1
			problems.extend(_compare_attrs(where, item.attrs, node.attrs.asdict()))
			continue

		if not isinstance(item, (Recording, Table)):
			continue

		sub_id = item.entities.sub
		root = get_root(sub_id)

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
			else:
				max_err = float(np.abs(values - expected).max())
				if not np.allclose(values, expected, atol=tolerance):
					problems.append(f"{where}: values differ (max abs error {max_err:.3e})")
				else:
					logger.debug("%s: verified (max abs error %.3e)", where, max_err)
			problems.extend(_compare_attrs(where, item.meta, group.attrs.asdict()))
		else:
			stored = _table_df(group[name])
			problems.extend(_compare_tables(where, stored, item.df))
			problems.extend(_compare_attrs(where, item.meta, group[name].attrs.asdict()))

	logger.info("verified %d items against %s", checked, source)
	return problems


def _navigate(root: "zarr.Group", path: tuple[str, ...]) -> "zarr.Group | None":
	"""Walk a tuple of group names from root, or None if any segment is missing."""
	node = root
	for part in path:
		if part not in node or not isinstance(node[part], zarr.Group):
			return None
		node = cast("zarr.Group", node[part])
	return node


def _compare_attrs(where: str, expected: dict[str, Any], stored: dict[str, Any]) -> list[str]:
	"""Check that every key in ``expected`` (sidecar metadata a reader supplied)
	survives in ``stored`` (the group's actual attrs) with an equal value.

	A subset check, not full-dict equality: the writer adds its own bookkeeping
	keys (``_neurozarr_item_type``, ``data_scale``, ``table_schema_version``,
	inferred defaults, ...) on top of what a reader passed in, and those are not
	part of what fidelity means here. Values are compared through a JSON
	round-trip so a tuple in ``expected`` (which zarr attrs store as a JSON
	list) doesn't read as a mismatch.
	"""
	canonical = cast("dict[str, Any]", json.loads(json.dumps(expected)))
	missing = sorted(key for key in canonical if key not in stored)
	if missing:
		return [f"{where}: metadata missing from store: {missing}"]
	differing = sorted(key for key in canonical if stored[key] != canonical[key])
	if differing:
		return [f"{where}: metadata differs for key(s): {differing}"]
	return []


def _dtype_compatible(a: Any, b: Any) -> bool:
	"""Whether two column dtypes represent the same fidelity, even if their
	exact repr differs.

	pandas 3's new default text dtype (``"str"``) and the older nullable
	``"string"`` extension dtype are both ``StringDtype``, differing only in
	na_value/storage backend (``nan`` vs ``pd.NA``) -- not in the text they
	hold. Treating that as a mismatch would flag every text column in every
	table as "corrupted" on a newer pandas, which is a false alarm: the
	values compared right after this check are what actually matters.
	"""
	return a == b or (pd.api.types.is_string_dtype(a) and pd.api.types.is_string_dtype(b))


def _compare_tables(where: str, stored: "pd.DataFrame", expected: "pd.DataFrame") -> list[str]:
	"""Compare a decoded stored table against the source DataFrame cell-for-cell.

	Row count alone doesn't catch a dropped column, a dtype that silently
	changed on round-trip, or a value that differs -- this checks all of them.
	"""
	if len(stored) != len(expected):
		return [f"{where}: {len(stored)} rows != source {len(expected)}"]

	problems: list[str] = []
	stored_cols, expected_cols = list(stored.columns), list(expected.columns)
	missing = [c for c in expected_cols if c not in stored_cols]
	extra = [c for c in stored_cols if c not in expected_cols]
	if missing:
		problems.append(f"{where}: columns missing from store: {missing}")
	if extra:
		problems.append(f"{where}: extra columns in store: {extra}")

	for col in expected_cols:
		if col not in stored_cols:
			continue
		s_col, e_col = stored[col], expected[col]
		if not _dtype_compatible(s_col.dtype, e_col.dtype):
			problems.append(f"{where}: column {col!r} dtype {s_col.dtype} != source {e_col.dtype}")
			continue  # comparing values across mismatched dtypes isn't meaningful

		s_null, e_null = s_col.isna(), e_col.isna()
		if not s_null.equals(e_null):
			problems.append(f"{where}: column {col!r} null mask differs in {int((s_null != e_null).sum())} row(s)")
			continue

		try:
			equal = ((s_col == e_col) | s_null).to_numpy(dtype=bool)
		except (TypeError, ValueError):
			equal = np.array([a == b or bool(m) for a, b, m in zip(s_col, e_col, s_null)])
		if not equal.all():
			problems.append(f"{where}: column {col!r} values differ in {int((~equal).sum())} row(s)")

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
