"""Fidelity, checked after every conversion in every experiment: does a format give
back what the source holds? Every recording's samples, every sidecar table and every
metadata file, compared with the source -- numbers, not checksums.

Two verdicts per arm:
- values (pass/fail, every format): each signal within the precision of the type it
  is stored as; each table cell equal once missing values are normalised (HDF5 and NWB
  store missing text as "", missing numbers as NaN); each source metadata key present
  with an equal value (a store may add its own keys).
- types (reported, not required): table columns come back with their source dtypes
  (nullable ints, categoricals, datetimes...). Only formats with a typed table schema
  can; it is a capability column, not a failure.

Keys are normalised the same way for every format: (subject or "_dataset", *path
without the subject segment), so a derivative's ("derivatives", name, "sub-01", ...)
lines up with where neurozarr routes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# relative to each recording's peak: float32 rounds to ~6e-8 of a value; int16 and
# float64 round-trip what mne returns to float64 rounding error
SIGNAL_RTOL = {"int16": 1e-9, "float64": 1e-12, "float32": 1e-6}


def norm(path: tuple[str, ...]) -> str:
	sub = next((part for part in path if part.startswith("sub-")), None)
	return "/".join((sub or "_dataset", *(p for p in path if p != sub)))


def source_content(source: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
	"""(tables, metadata, recording sidecars) the source holds, by normalised key."""
	from neurozarr.items import Attrs, Recording, Table
	from neurozarr.readers import reader_for

	from bench.arms import _rec_id
	tables: dict[str, pd.DataFrame] = {}
	meta: dict[str, dict[str, Any]] = {}
	sidecars: dict[str, dict[str, Any]] = {}
	for item in reader_for(source).read():
		if isinstance(item, Table):
			tables[norm((*item.prefix, *item.entities.path(), item.name))] = item.df
		elif isinstance(item, Attrs):
			meta.setdefault(norm(tuple(item.path)), {}).update(item.attrs)
		elif isinstance(item, Recording):
			sidecars[_rec_id(item.entities)] = item.meta
	return tables, meta, sidecars


# --- what each format gives back: {key: (DataFrame, column names)}, {key: metadata} ---

def _tables_neurozarr(dest: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
	import zarr
	from neurozarr import Repo
	from neurozarr.read import RecordingView, TableView, views_in
	repo = Repo.open(dest)
	roots = {sub: repo.root_of(sub) for sub in repo.subjects()}
	roots["_dataset"] = repo._dataset_root()
	return _walk_zarr(roots)


def _tables_plain_zarr(dest: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
	import zarr
	return _walk_zarr({d.name: zarr.open_group(store=zarr.storage.LocalStore(str(d)), mode="r")
					   for d in sorted(Path(dest).iterdir()) if d.is_dir()})


def _walk_zarr(roots: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
	import zarr
	from neurozarr.read import _table_df
	tables, meta, recs = {}, {}, {}
	for name, root in roots.items():
		stack = [("", root)]
		while stack:
			path, group = stack.pop()
			attrs = group.attrs.asdict()
			key = "/".join(p for p in (name, path) if p)
			kind = attrs.get("_neurozarr_item_type")
			if kind == "table":
				if attrs.get("source") != "mne.Raw.annotations":
					tables[key] = _table_df(group)
				continue
			if kind == "recording":
				recs[key] = attrs
			meta[key] = attrs
			for child, node in group.members():
				if isinstance(node, zarr.Group):
					stack.append((f"{path}/{child}" if path else child, node))
	return tables, meta, recs


def _tables_hdf5(dest: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
	import h5py
	tables, meta, recs = {}, {}, {}
	with h5py.File(Path(dest) / "store.h5", "r") as f:
		def visit(name: str, node: Any) -> None:
			if isinstance(node, h5py.Dataset) and "columns" in node.attrs:
				if node.name.rsplit("/", 1)[-1] == "annotations" and json.loads(node.attrs["meta"]).get("source") == "mne.Raw.annotations":
					return
				columns = json.loads(node.attrs["columns"])
				records = node[()]
				tables[norm(tuple(name.split("/")))] = pd.DataFrame(
					{c: _h5_column(records[f"c{i}"]) for i, c in enumerate(columns)}, columns=columns)
			elif isinstance(node, h5py.Dataset) and name.endswith("/data"):
				recs[name[: -len("/data")]] = json.loads(node.attrs.get("meta", "{}"))
			elif isinstance(node, h5py.Group) and "meta" in node.attrs:
				meta[norm(tuple(name.split("/")))] = json.loads(node.attrs["meta"])
		f.visititems(visit)
		if "meta" in f.attrs:
			meta["_dataset"] = json.loads(f.attrs["meta"])
	return tables, meta, recs


def _h5_column(values: np.ndarray) -> np.ndarray:
	return np.char.decode(values, "utf-8").astype(object) if values.dtype.kind == "S" else values


def _tables_nwb(dest: Path, io_cls: Any, glob: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
	tables, meta, recs = {}, {}, {}
	dataset_file = Path(dest) / "dataset_metadata.json"
	if dataset_file.exists():
		for path, value in json.loads(dataset_file.read_text()).items():
			meta.setdefault(norm(tuple(path.split("__")) if path != "dataset" else ()), {}).update(value)
	for path in sorted(Path(dest).glob(glob)):
		with io_cls(str(path), mode="r") as io:
			nwbfile = io.read()
			sub, ses = path.name.removesuffix(".nwb.zarr").removesuffix(".nwb").split("_", 1)
			module = nwbfile.processing.get("sidecars")
			for table in (module.data_interfaces.values() if module is not None else ()):
				description = json.loads(table.description)
				frame = table.to_dataframe()
				frame.columns = description["columns"]
				tables[norm(tuple(description["key"].split("/")))] = frame.reset_index(drop=True)
			for name, entry in nwbfile.scratch.items():
				value = json.loads(entry.data[()] if hasattr(entry.data, "__getitem__") and not isinstance(entry.data, str) else entry.data)
				if name.endswith("__meta"):
					rec = name[: -len("__meta")]
					recs["/".join((sub, ses, "ieeg", rec)) if ses != "ses-none" else "/".join((sub, "ieeg", rec))] = value
				else:
					meta.setdefault(norm(tuple(name.split("__"))), {}).update(value)
	return tables, meta, recs


def stored_content(arm_name: str, dest: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
	fmt = arm_name.split("_int16")[0].split("_float")[0]
	if fmt == "neurozarr":
		return _tables_neurozarr(dest)
	if fmt == "plain_zarr":
		return _tables_plain_zarr(dest)
	if fmt in ("hdf5", "hdf5_mat73"):
		return _tables_hdf5(dest)
	if fmt == "nwb_hdf5":
		from pynwb import NWBHDF5IO
		return _tables_nwb(dest, NWBHDF5IO, "*.nwb")
	if fmt == "nwb_zarr":
		from hdmf_zarr import NWBZarrIO
		return _tables_nwb(dest, NWBZarrIO, "*.nwb.zarr")
	raise ValueError(f"no fidelity reader for {arm_name!r}")


# --- comparisons ---

def _text(values: Any) -> list[str]:
	return ["" if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA else str(v)
			for v in (values.tolist() if hasattr(values, "tolist") else list(values))]


def tables_equal(stored: pd.DataFrame, expected: pd.DataFrame) -> tuple[bool, bool]:
	"""(values equal, dtypes equal) for one table."""
	if list(map(str, stored.columns)) != list(map(str, expected.columns)) or len(stored) != len(expected):
		return False, False
	types = True
	for i in range(len(expected.columns)):
		s, e = stored.iloc[:, i], expected.iloc[:, i]
		types &= str(s.dtype) == str(e.dtype) or (pd.api.types.is_string_dtype(s.dtype) and pd.api.types.is_string_dtype(e.dtype))
		if pd.api.types.is_numeric_dtype(e.dtype) and not pd.api.types.is_bool_dtype(e.dtype):
			a = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
			b = e.to_numpy(dtype="float64", na_value=np.nan)
			if not np.allclose(a, b, rtol=1e-12, atol=0, equal_nan=True):
				return False, False
		elif pd.api.types.is_bool_dtype(e.dtype):
			if [str(v) for v in _text(e)] != [str(v) for v in _text(s.map(lambda v: bool(v) if v not in ("", None) and not pd.isna(v) else v))]:
				return False, False
		elif _text(s) != _text(e):
			return False, False
	return True, bool(types)


def metadata_equal(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
	canonical = json.loads(json.dumps(expected))
	return all(key in stored and json.loads(json.dumps(stored[key])) == value for key, value in canonical.items())


def stored_dtype(arm_name: str) -> str:
	return next((d for d in SIGNAL_RTOL if f"_{d}_" in f"{arm_name}_"), "int16")


def check(arm_name: str, store: Any, dest: Path, source: Path, content: str,
		  reference: Any, expected: tuple[dict, dict, dict] | None = None,
		  sample: int | None = None, seed: int = 0) -> dict[str, Any]:
	"""Fidelity of one converted arm. ``store`` is the arm's opened ArmStore, ``reference``
	the raw arm's (the source read by mne); ``expected`` is source_content(source), passed
	in when several arms share it. ``sample``: check the samples of only this many
	recordings (the largest plus a seeded random pick) -- for datasets where reading every
	sample of every arm back is a full pass over 100+ GB; tables, metadata and every
	recording's presence and sidecar are still checked in full."""
	tables, meta, sidecars = expected or source_content(source)
	dtype = stored_dtype(arm_name)
	problems: list[str] = []

	# signals: every sample of every recording, or of a seeded sample of them
	worst = 0.0
	recordings = store.recordings()
	checked = recordings
	if sample is not None and len(recordings) > sample:
		import random
		largest = max(recordings, key=lambda r: r.n_samples)
		rest = sorted((r for r in recordings if r.id != largest.id), key=lambda r: r.id)
		checked = [largest] + random.Random(seed).sample(rest, sample - 1)
	for info in checked:
		got, want = np.asarray(store.full_read(info.id), dtype="float64"), np.asarray(reference.full_read(info.id), dtype="float64")
		peak = float(np.max(np.abs(want))) if want.size else 0.0
		err = float(np.max(np.abs(got - want))) if want.size else 0.0
		rel = err / peak if peak else err
		worst = max(worst, rel)
		if got.shape != want.shape or rel > SIGNAL_RTOL[dtype]:
			problems.append(f"signal {info.id}: shape {got.shape} vs {want.shape}, rel error {rel:.2e}")
	missing_recs = sorted(set(r.id for r in reference.recordings()) - set(r.id for r in recordings))
	problems += [f"recording missing: {rid}" for rid in missing_recs]

	stored_tables, stored_meta, stored_recs = stored_content(arm_name, dest)
	# every recording's own sidecar, at both content levels
	for rid, want_meta in sidecars.items():
		got_meta = next((v for k, v in stored_recs.items() if k.endswith(rid.split("/", 1)[-1]) and k.startswith(rid.split("/", 1)[0])), None)
		if got_meta is None or not metadata_equal(got_meta, want_meta):
			problems.append(f"recording metadata: {rid}")

	types_ok = True
	n_tables = n_meta = 0
	if content == "full":
		for key, df in tables.items():
			got = stored_tables.get(key)
			if got is None:
				problems.append(f"table missing: {key}")
				continue
			values_ok, same_types = tables_equal(got, df)
			types_ok &= same_types
			n_tables += 1
			if not values_ok:
				problems.append(f"table values: {key}")
		for key, want_meta in meta.items():
			n_meta += 1
			if not metadata_equal(stored_meta.get(key, {}), want_meta):
				problems.append(f"metadata: {key}")
	return {"fidelity_ok": not problems, "fidelity_problems": problems[:50], "fidelity_n_problems": len(problems),
			"fidelity_recordings": len(recordings), "fidelity_signals_checked": len(checked),
			"fidelity_scope": "all" if len(checked) == len(recordings) else f"sample of {len(checked)} (seed {seed})",
			"fidelity_signal_max_rel_err": worst,
			"fidelity_tables": n_tables, "fidelity_metadata": n_meta,
			"fidelity_table_types_preserved": types_ok if content == "full" else None}


def read_matches(workload: str, got: Any, want: Any, dtype: str) -> tuple[bool, float | None]:
	"""Does one timed read equal the same read of the source? (match, error relative to
	the source read's peak). Searches compare their id lists."""
	if workload == "W4":
		return sorted(got) == sorted(want), None
	got, want = np.asarray(got, dtype="float64"), np.asarray(want, dtype="float64")
	if got.shape != want.shape:
		return False, None
	peak = float(np.max(np.abs(want))) if want.size else 0.0
	err = float(np.max(np.abs(got - want))) if want.size else 0.0
	rel = err / peak if peak else err
	# a scan sums each channel over the recording: float32 rounding accumulates with length
	tol = SIGNAL_RTOL[dtype] * (1e3 if workload == "W5" and dtype == "float32" else 1.0)
	return rel <= tol, rel
