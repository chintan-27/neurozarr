import copy
import datetime
import fnmatch
import hashlib
import json
import re
import tempfile
import warnings
from dataclasses import replace
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from itertools import takewhile
from pathlib import Path
from typing import Any, Iterator

import mne  # type: ignore[import-untyped]  # mne ships no type information
import numpy as np
import pandas as pd

from ..entities import Entities, split_stem
from ..errors import ValidationError
from ..items import Attrs, ExternalFile, Item, Recording, Table
from ..log import log_progress
from ..util import clean_nan, load_json, source_provenance
from .bids import MNE_READABLE_EXTS
from ..codecs import CodecPlan, check_plan_options, plan_codecs
from .formats import UnclaimedPolicy, decode_embed, decoder_for

DATATYPES = ("eeg", "ieeg", "meg", "emg")
_ARRAY_EXTS = {".mat": "mat", ".npy": "npy", ".npz": "npz", ".h5": "hdf5", ".hdf5": "hdf5", ".nwb": "hdf5"}
# never sniffed: text and sidecar files, and the BrainVision/EEGLAB companions mne opens via their header file
_NOT_DATA = {".tsv", ".csv", ".json", ".txt", ".md", ".eeg", ".vmrk", ".fdt"}
_RATE_NAMES = {"sfreq", "fs", "srate", "rate", "samplingrate", "samplingfrequency", "samplerate", "sr", "fsample"}
_CH_NAMES = {"chnames", "channelnames", "channels", "labels", "chanlabels"}
_FINGERPRINT_BYTES = 1 << 20
# ids too generic to tell subjects apart (CHB-MIT's EDF headers all say "Surrogate")
_JUNK_IDS = {"", "x", "surrogate", "anonymous", "unknown", "none", "n/a"}

# a filename prefix with a digit that names a recording, not a subject ("run3_...")
_NOT_IDS = {"run", "rec", "record", "recording", "ses", "session", "trial", "block", "part", "seg",
			"segment", "file", "data", "eeg", "ieeg", "meg", "test", "task", "day", "visit", "block"}
_HEADER_TYPES = {"ieeg": {"seeg", "ecog", "dbs"}, "meg": {"mag", "grad"}, "emg": {"emg"}}

_ROW_KEYS = ("sub", "ses", "datatype", "path", "name", "format", "array", "sfreq", "prefix")


def _label(text: str) -> str:
	return re.sub(r"[^0-9A-Za-z]", "", text)


def _norm(name: str) -> str:
	return re.sub(r"[^a-z]", "", name.rsplit("/", 1)[-1].rsplit("@", 1)[-1].lower())


def _tagged(parts: tuple[str, ...], key: str) -> str | None:
	"""The value of a ``key-value`` folder name (``ses-1`` -> ``1``), if any."""
	return next((p.split("-", 1)[1] for p in parts if p.startswith(key + "-") and len(p) > len(key) + 1), None)


def _sniff(path: Path) -> str | None:
	"""Format of a file from its first bytes, for files whose extension says nothing."""
	try:
		with path.open("rb") as f:
			head = f.read(16)
	except OSError:
		return None
	if head.startswith(b"\x89HDF") or head.startswith(b"MATLAB 7.3"):
		return "hdf5"
	if head.startswith(b"MATLAB 5.0"):
		return "mat"
	if head.startswith(b"\x93NUMPY"):
		return "npy"
	if head[:8] == b"0       ":
		return "edf"
	return None


def _format(path: Path) -> str | None:
	ext = path.suffix.lower()
	if ext in MNE_READABLE_EXTS:
		return "mne"
	if ext == ".mat" and _sniff(path) == "hdf5":
		return "hdf5"  # MATLAB v7.3 is HDF5 underneath
	if ext in _ARRAY_EXTS:
		return _ARRAY_EXTS[ext]
	if ext in _NOT_DATA:
		return None
	return _sniff(path)


def _fingerprint(path: Path) -> tuple[int, str]:
	"""Size plus a hash of the first and last MiB: identical copies of one recording
	collide, distinct recordings with the same length and channels (CHB-MIT's hour-long
	files) do not, without hashing whole multi-GB files."""
	size = path.stat().st_size
	digest = hashlib.blake2b(digest_size=16)
	with path.open("rb") as f:
		digest.update(f.read(_FINGERPRINT_BYTES))
		if size > 2 * _FINGERPRINT_BYTES:
			f.seek(-_FINGERPRINT_BYTES, 2)
			digest.update(f.read())
	return size, digest.hexdigest()


# --- array formats -------------------------------------------------------------------

Entry = tuple[tuple[int, ...], str, Callable[[], Any]]  # shape, numpy dtype kind, loader


def _h5_entries(path: Path) -> dict[str, Entry]:
	try:
		import h5py  # type: ignore[import-untyped]  # h5py ships no type information
	except ImportError as exc:
		raise ValidationError("reading HDF5/NWB/MATLAB v7.3 files needs h5py (pip install h5py)") from exc

	def read(name: str) -> Any:
		with h5py.File(path, "r") as f:
			return f[name][()]

	out: dict[str, Entry] = {}

	def visit(name: str, obj: Any) -> None:
		if isinstance(obj, h5py.Dataset):
			out[name] = (tuple(obj.shape), obj.dtype.kind, partial(read, name))
		for key, value in obj.attrs.items():
			value = np.asarray(value)
			out[f"{name}@{key}"] = (value.shape, value.dtype.kind, partial(np.asarray, value))

	with h5py.File(path, "r") as f:
		f.visititems(visit)
	return out


def _mat_variable(path: Path, name: str) -> Any:
	import scipy.io  # type: ignore[import-untyped]  # a dependency of mne
	return scipy.io.loadmat(path, variable_names=[name], squeeze_me=True)[name]


def _array_entries(fmt: str, path: Path) -> dict[str, Entry]:
	"""Every named array in a file, unloaded where the format allows."""
	if fmt == "npy":
		arr = np.load(path, mmap_mode="r")
		return {"": (arr.shape, arr.dtype.kind, partial(np.load, path))}
	if fmt == "npz":
		with np.load(path) as z:
			arrays = {name: z[name] for name in z.files}
		return {name: (a.shape, a.dtype.kind, partial(np.asarray, a)) for name, a in arrays.items()}
	if fmt == "mat":
		import scipy.io  # a dependency of mne
		kinds = {"double": "f", "single": "f", "char": "U", "cell": "O"}
		return {name: (tuple(shape), kinds.get(cls, "i" if "int" in cls else "V"),
					   partial(_mat_variable, path, name))
				for name, shape, cls in scipy.io.whosmat(path)}
	return _h5_entries(path)


def _closest(names: list[str], to: str) -> str:
	"""The name sharing the most leading path segments with ``to`` -- the rate stored next to the data."""
	def shared(name: str) -> int:
		return len(list(takewhile(lambda pair: pair[0] == pair[1], zip(name.split("/"), to.split("/")))))
	return max(names, key=shared)


def _probe_array(fmt: str, path: Path) -> dict[str, Any]:
	"""Find the signal in an array file: the largest 1-2D numeric array, a sampling rate
	stored beside it, and channel names if there is a list of the right length."""
	entries = _array_entries(fmt, path)
	numeric = {n: e for n, e in entries.items() if e[1] in "fiu" and 1 <= len(e[0]) <= 2 and "@" not in n}
	if not numeric:
		raise ValidationError("no 1-D or 2-D numeric array in the file")
	key = max(numeric, key=lambda n: int(np.prod(numeric[n][0])))
	rates = [n for n, e in entries.items()
			 if _norm(n) in _RATE_NAMES and e[1] in "fiu" and int(np.prod(e[0])) == 1]
	sfreq = float(np.asarray(entries[_closest(rates, key)][2]()).squeeze()) if rates else None
	return {"array": key, "sfreq": sfreq}


def _load_array(fmt: str, path: Path, key: str, sfreq: float, ch_type: str) -> "mne.io.RawArray":
	# ponytail: loads the whole array into memory; stream chunks from h5py/np.memmap if
	# array files larger than RAM show up
	entries = _array_entries(fmt, path)
	data = np.asarray(entries[key][2](), dtype=np.float64)
	data = data[None, :] if data.ndim == 1 else data
	if data.shape[0] > data.shape[1]:
		data = data.T  # channels are the short axis
	n_ch = data.shape[0]
	ch_names = [f"ch{i:03d}" for i in range(n_ch)]
	for name, (shape, kind, load) in entries.items():
		if _norm(name) in _CH_NAMES and kind in "USO" and int(np.prod(shape)) == n_ch:
			labels = [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(load()).ravel()]
			if len(set(labels)) == n_ch:
				ch_names = labels
				break
	info = mne.create_info(ch_names, sfreq, ch_type)
	return mne.io.RawArray(data, info, verbose=False)


# --- inference ---------------------------------------------------------------------------

def _edf_header(path: Path) -> dict[str, Any]:
	"""EDF/BDF identity straight from the fixed 256-byte header. mne's reader also
	loads EDF+ annotations, which scans the whole file -- 26 minutes for Pediatric
	iEEG's 165 GB -- for three fields that sit in the first 256 bytes."""
	with path.open("rb") as f:
		head = f.read(256)
	if len(head) < 256 or head[:8] not in (b"0       ", b"\xffBIOSEMI"):
		raise ValidationError("not an EDF/BDF header")
	header_bytes, n_signals = int(head[184:192]), int(head[252:256])
	if n_signals < 1 or header_bytes != 256 * (n_signals + 1) or path.stat().st_size < header_bytes:
		raise ValidationError("EDF/BDF header is inconsistent or the file is truncated")
	patient = head[8:88].decode("latin-1").split()
	try:
		day, month, year = (int(x) for x in head[168:176].decode().split("."))
		hour, minute, second = (int(x) for x in head[176:184].decode().split("."))
		# EDF's two-digit year: 85-99 is 1985-1999, the rest 2000s (the EDF spec's clipping rule)
		meas_date = datetime.datetime(year + (1900 if year >= 85 else 2000), month, day, hour, minute, second,
									  tzinfo=datetime.timezone.utc)
	except ValueError:
		meas_date = None
	return {"meas_date": meas_date, "his_id": patient[0] if patient and patient[0] != "X" else None, "types": set()}


def _open_header(path: Path, fmt: str) -> dict[str, Any] | None:
	"""Who/when/what a recording's header says, without reading its samples."""
	if fmt == "edf" or path.suffix.lower() in (".edf", ".bdf"):
		return _edf_header(path)
	if fmt != "mne":
		return None
	if path.suffix.lower() == ".set":
		# mne loads a whole EEGLAB .set to open it (HBN: 4 minutes); listing its
		# variables proves it's a readable MATLAB file, and EEGLAB carries no patient
		# id or date worth having
		import scipy.io
		try:
			scipy.io.whosmat(path)
		except (ValueError, NotImplementedError):  # v7.3 .set is HDF5
			if _sniff(path) != "hdf5":
				raise
		return {"meas_date": None, "his_id": None, "types": set()}
	raw = mne.io.read_raw(path, preload=False, verbose=False)
	return {"meas_date": raw.info["meas_date"], "his_id": (raw.info.get("subject_info") or {}).get("his_id"),
			"types": set(raw.get_channel_types())}


def _answers_for(path: Path, answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
	merged: dict[str, Any] = {}
	for pattern, fields in answers.items():
		if fnmatch.fnmatch(str(path), pattern):
			merged.update(fields)
	return merged


def scan_folders(roots: Sequence[str | Path],
				 answers: dict[str, dict[str, Any]] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
	"""Infer what every file in some folders is, and who and what it belongs to.

	See :class:`SmartReader` for the rules. Opens each recording's header (and each
	array file's index) but reads no sample data.

	Parameters
	----------
	roots : list of str or pathlib.Path
		Folders to search, in any layout.
	answers : dict, optional
		Overrides by path glob, as returned in ``report["questions"]`` -- see
		:class:`SmartReader`.

	Returns
	-------
	manifest : pandas.DataFrame
		One row per recording or sidecar: ``sub``, ``ses``, ``datatype``, ``task``,
		``run`` and any other entity found in a filename, then ``path``, ``name``
		(sidecars only), ``format``, and for array files ``array`` and ``sfreq``.
	report : dict
		``recordings``, ``subjects``, ``layouts``, ``sub_rules``, ``datatype_rules``,
		``run_rules``, ``runs_numbered``, ``skipped_extensions``, ``unreadable``,
		``duplicates``, ``dataset_files`` and ``questions``.
	"""
	answers = answers or {}
	recs: list[dict[str, Any]] = []
	skipped: Counter[str] = Counter()
	unreadable: list[dict[str, str]] = []
	duplicates: list[dict[str, str]] = []
	layouts: dict[str, str] = {}
	dataset_files: list[str] = []
	questions: dict[tuple[str, str], dict[str, Any]] = {}

	def ask(field: str, root: Path, guess: Any, why: str, example: Path) -> None:
		q = questions.setdefault((field, str(root)), {
			"field": field, "pattern": f"{root}/*", "guess": guess, "why": why,
			"count": 0, "example": str(example)})
		q["count"] += 1

	for root in (Path(r).resolve() for r in roots):
		if not root.is_dir():
			raise NotADirectoryError(f"{root} is not a folder")
		dataset_files += [str(root / n) for n in ("dataset_description.json", "participants.tsv", "participants.json")
						  if (root / n).exists()]
		dataset_files += [str(p) for p in sorted(root.rglob("*_sessions.tsv"))]
		root_recs: list[dict[str, Any]] = []
		for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")):
			fmt = _format(path)
			if fmt is None:
				if path.suffix.lower() not in _NOT_DATA:
					skipped[path.suffix.lower()] += 1
				continue
			given = _answers_for(path, answers)
			if given.get("skip"):
				continue
			try:
				fp = _fingerprint(path)
				probe: dict[str, Any] = {}
				with warnings.catch_warnings():
					# header probe only: mne's per-file naming/date warnings are noise here
					warnings.simplefilter("ignore")
					header = _open_header(path, fmt)
				if header is None:
					probe = _probe_array(fmt, path)
			except Exception as exc:  # noqa: BLE001 -- any reader failure means "can't use this file"
				unreadable.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"[:300]})
				continue
			parts = path.relative_to(root).parts[:-1]
			stem = path.name[:-len(path.suffix)] if path.suffix else path.name
			ents, suffix = split_stem(stem)
			token = re.split(r"[_\-. ]", stem)[0]
			header_id = header["his_id"] if header else None
			meas_date = header["meas_date"] if header else None
			sidecar = load_json(path.with_suffix(".json")) if path.suffix else {}

			if "sub_from" in given and (found := re.search(given["sub_from"], str(path.relative_to(root)))):
				sub, rule = found.group(1), "answer"
			elif "sub" in given:
				sub, rule = str(given["sub"]).removeprefix("sub-"), "answer"
			elif "sub" in ents:
				sub, rule = ents["sub"], "filename sub-"
			elif (tag := _tagged(parts, "sub")) is not None:
				sub, rule = tag, "folder sub-"
			elif folder := next((p for p in reversed(parts) if p not in DATATYPES and len(p) > 2
								 and token.lower().startswith(p.lower())), None):
				# chb01/chb01_03.edf, and chb17/chb17a_03.edf too: the folder starts the name
				sub, rule = folder, "folder named like the file"
			else:
				sub, rule = "", "unresolved"  # settled below, once the whole folder is seen
			ents.pop("sub", None)

			# answer, then the BIDS filename suffix (sub-01_..._ieeg.edf -- the file's own
			# label beats a folder it was dropped into), then folder, then the
			# header's channel types -- "eeg" there is mne's default for EDF, so only the
			# other types count as evidence
			header_type = next((dt for dt, marks in _HEADER_TYPES.items() if header and header["types"] & marks), None)
			# any part of the suffix: "ieeg_bipolar" is still iEEG
			suffix_type = next((part for part in reversed(suffix.split("_")) if part in DATATYPES), None)
			for dt_rule, value in (("answer", given.get("datatype")),
								   ("filename suffix", suffix_type),
								   ("folder", next((p for p in reversed(parts) if p in DATATYPES), None)),
								   ("channel types", header_type), ("guessed eeg", "eeg")):
				if value:
					datatype = str(value)
					break
			if dt_rule == "guessed eeg":
				ask("datatype", root, "eeg", "nothing in the path or header says what these recordings are", path)

			ses = given.get("ses") or ents.pop("ses", None) or _tagged(parts, "ses")
			ents.pop("ses", None)
			for key in ("task", "run", "acq"):
				if key in given:
					ents[key] = str(given[key])

			sfreq = probe.get("sfreq")
			if fmt not in ("mne", "edf"):
				sfreq = given.get("sfreq") or sfreq or sidecar.get("SamplingFrequency")
				if sfreq is None:
					ask("sfreq", root, None, "array files with no sampling rate inside them or in a JSON sidecar", path)

			# BIDS names the recording "..._ieeg" and its sidecars "..._events"; match on
			# the stem without its suffix. An unstructured name matches whole.
			prefix = stem.rsplit("_", 1)[0] if "task" in ents else stem
			sides = [(f, f.stem[len(prefix) + 1:]) for f in sorted(path.parent.glob(f"{prefix}_*"))
					 if f.suffix.lower() in (".tsv", ".csv", ".json") and f.stem != stem]
			rec = {"sub": sub, "ses": f"ses-{_label(str(ses))}" if ses else None, "datatype": datatype,
				   **{k: _label(v) for k, v in ents.items() if k.isalpha()},
				   "path": path, "sides": sides, "format": fmt, "array": probe.get("array"),
				   "sfreq": sfreq, "_rule": rule, "_header_id": header_id, "_date": meas_date,
				   "prefix": "/".join(parts[parts.index("derivatives"):][:2]) if "derivatives" in parts else None,
				   "_stem": stem, "_token": token, "_top": parts[0] if parts else None, "_fp": fp, "_dt_rule": dt_rule}
			root_recs.append(rec)

		# The subjects no name or folder settled. In order: the header's patient id, when
		# headers actually tell recordings apart (not one anonymized id on every file);
		# then an id-like filename prefix (has a digit, not a word like "run3", shared by
		# several files -- chb01_03.edf); then the top folder; then any prefix. The last
		# two are guesses, so they are asked about.
		open_ = [r for r in root_recs if r["_rule"] == "unresolved"]
		ids = {r["_header_id"] for r in open_}
		ids_useful = len(ids) > 1 and None not in ids and not any(str(i).strip().lower() in _JUNK_IDS for i in ids)
		tokens = Counter(r["_token"] for r in open_)
		for r in open_:
			token = r["_token"]
			if ids_useful:
				r["sub"], r["_rule"] = str(r["_header_id"]), "file header"
			elif re.search(r"\d", token) and tokens[token] > 1 and re.sub(r"\d", "", token).lower() not in _NOT_IDS:
				r["sub"], r["_rule"] = token, "id-like filename prefix"
			elif r["_top"]:
				r["sub"], r["_rule"] = r["_top"], "top folder"
				ask("sub_from", root, "top folder name", "subject taken from the top folder's name", r["path"])
			else:
				r["sub"], r["_rule"] = token, "filename prefix"
				ask("sub_from", root, r"^([A-Za-z0-9]+)", "subject taken from the start of the filename", r["path"])
		for r in root_recs:
			r["sub"] = "sub-" + _label(r["sub"])

		in_subdirs = any(r["path"].parent != root for r in root_recs)
		layouts[str(root)] = ("bids" if (root / "dataset_description.json").exists() or any(
			p.is_dir() and p.name.startswith("sub-") for p in root.iterdir())
			else "folder per subject" if in_subdirs else "flat")
		recs += root_recs

	# Byte-identical copies are ingested once: the copy with the most sidecars beside it
	# (its events/channels/JSON), so dropping the bare copy loses nothing; ties go to
	# the first path.
	copies: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
	for rec in recs:
		copies[rec["_fp"]].append(rec)
	for group in copies.values():
		keep = max(group, key=lambda r: len(r["sides"]))  # max keeps the first on ties
		duplicates += [{"path": str(r["path"]), "same_as": str(keep["path"])} for r in group if r is not keep]
	kept = {id(max(g, key=lambda r: len(r["sides"]))) for g in copies.values()}
	recs = [r for r in recs if id(r) in kept]
	sub_rules = Counter(r["_rule"] for r in recs)
	dtype_rules = Counter(r["_dt_rule"] for r in recs)

	report: dict[str, Any] = {
		"recordings": len(recs), "subjects": len({r["sub"] for r in recs}), "layouts": layouts,
		"sub_rules": dict(sub_rules), "datatype_rules": dict(dtype_rules), "run_rules": {},
		"runs_numbered": 0, "skipped_extensions": dict(skipped), "unreadable": unreadable,
		"duplicates": duplicates, "dataset_files": dataset_files, "questions": list(questions.values())}
	if not recs:
		return pd.DataFrame(), report

	# Recordings must be unique by entities; where several share them (a subject's
	# numbered files with no run in the name), give each a run: the number its filename
	# already ends in, so a run still points at its file; else recording order from the
	# headers; else path order.
	groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
	for rec in recs:
		groups[tuple((k, v) for k, v in rec.items()
					  if k not in ("path", "sides", "format", "array", "sfreq") and not k.startswith("_"))].append(rec)
	run_rules: Counter[str] = Counter()
	for members in groups.values():
		if len(members) < 2:
			continue
		tails = [re.search(r"(\d+)$", m["path"].name[:-len(m["path"].suffix)] if m["path"].suffix else m["path"].name)
				 for m in members]
		own = [t.group(1) for t in tails if t]
		# what the name says beyond the subject: chb17a_03 / chb17b_03 under sub chb17
		rests = [_label(m["_stem"][len(m["sub"]) - 4:]) if m["_stem"].lower().startswith(m["sub"][4:].lower())
				 else "" for m in members]
		dates = [m["_date"] for m in members]
		if len(own) == len(members) and len(set(own)) == len(own):
			runs, rule = own, "filename number"
		elif all(rests) and len(set(rests)) == len(rests):
			runs, rule = rests, "filename after subject"
		elif None not in dates and len(set(dates)) == len(dates):
			ranked = sorted(range(len(members)), key=lambda i: dates[i])
			runs, rule = [str(ranked.index(i) + 1) for i in range(len(members))], "recording date"
		else:
			runs, rule = [str(i) for i in range(1, len(members) + 1)], "path order"
		for member, run in zip(members, runs):
			member["run"] = run
		run_rules[rule] += len(members)
	report["run_rules"] = dict(run_rules)
	report["runs_numbered"] = sum(run_rules.values())

	rows: list[dict[str, Any]] = []
	for rec in recs:
		path, sides = rec.pop("path"), rec.pop("sides")
		for key in [k for k in rec if k.startswith("_")]:
			del rec[key]
		rows.append({**rec, "path": str(path), "name": None})
		rows += [{**rows[-1], "path": str(f), "name": name, "format": "sidecar", "array": None, "sfreq": None}
				 for f, name in sides]
	df = pd.DataFrame(rows)
	order = [c for c in ("sub", "ses", "datatype", "task", "run") if c in df.columns]
	return df[order + [c for c in df.columns if c not in order]], report


_EDF_LINKS: list[tempfile.TemporaryDirectory[str]] = []  # one per process, made on first use


def _as_edf(path: Path) -> Path:
	"""mne opens EDF only by extension; give an extensionless EDF one via a symlink."""
	if not _EDF_LINKS:
		_EDF_LINKS.append(tempfile.TemporaryDirectory(prefix="neurozarr-edf-"))
	link = Path(_EDF_LINKS[0].name) / f"{hashlib.blake2b(str(path).encode(), digest_size=8).hexdigest()}.edf"
	if not link.exists():
		link.symlink_to(path)
	return link


def _with_prefix(item: Item, prefix: tuple[str, ...]) -> Item:
	if not prefix:
		return item
	if isinstance(item, Attrs):  # an Attrs carries its prefix in its path
		return Attrs((*prefix, *item.path), item.attrs)
	return replace(item, prefix=prefix)


class SmartReader:
	"""Read recordings from any folders, working out the layout and formats itself.

	.. note::
		Beta. The rules below are heuristics and may change between releases.
		Check ``reader.df``, ``reader.report`` and ``reader.questions`` before
		ingesting data you care about.

	List the folders the data lives in. BIDS folders, one folder per subject, and
	flat dumps with the subject in the filename all work, alone or mixed.

	**Formats.** Everything mne opens (EDF, BDF, GDF, BrainVision, EEGLAB, FIF, CNT),
	plus array files: MATLAB ``.mat`` (v5 and v7.3), ``.npy``, ``.npz``, HDF5 and NWB.
	In an array file the signal is the largest 1-D or 2-D numeric array (channels are
	the shorter axis); the sampling rate is a scalar named like ``fs``/``srate``/``rate``
	stored nearest to it, or ``SamplingFrequency`` in a same-named JSON sidecar; channel
	names are a list named like ``ch_names``/``labels`` of the right length. Files with
	no extension are recognized by their first bytes. Values are stored as the file
	holds them, in its own units.

	**Who and what.** Subject comes from a ``sub-`` in the filename, then a ``sub-``
	folder, then the top folder under the one you listed, then the patient id in the
	file header (when headers actually tell subjects apart), then the filename prefix.
	Session from ``ses-`` in the name or a folder; datatype from an enclosing
	``eeg``/``ieeg``/``meg``/``emg`` folder, else ``eeg``. Recordings that would share
	every identifier get a run: the number their filename ends in, else their order by
	recording date from the header, else path order.

	**Metadata.** A same-named ``.json`` sidecar is merged into the recording's
	metadata; other ``<recording>_*.tsv``/``.json`` files become tables and attrs on it.
	``dataset_description.json``, ``participants.tsv``/``.json`` and
	``*_sessions.tsv`` are read as dataset, subject and session attrs.

	**Checks.** Files no reader can open are listed in ``report["unreadable"]``
	rather than failing the run, and byte-identical copies (in different folders)
	are ingested once, listed in ``report["duplicates"]``.

	**Asking.** Every guess the reader is unsure of -- a defaulted datatype, a
	subject read off a filename prefix, an array file with no sampling rate -- is
	grouped by folder into ``reader.questions``. Answer with ``answers``, a dict of
	path glob to field overrides, which you can keep in a JSON file and reuse::

		{"/data/siteA/*": {"datatype": "ieeg", "sfreq": 512},
		 "/data/flat/*": {"sub_from": "^p(\\\\d+)_"},
		 "/data/siteA/old/*": {"skip": true}}

	Fields: ``sub``, ``ses``, ``task``, ``run``, ``acq``, ``datatype``, ``sfreq``,
	``skip``, and ``sub_from`` (a regex over the path below the listed folder whose
	first group is the subject). ``neurozarr convert --reader smart`` asks these
	questions interactively and saves the answers. An array file still without a
	sampling rate is kept as a reference to its path, not dropped.

	Parameters
	----------
	folders : str, pathlib.Path or list of them
		Folders to search, recursively.
	answers : dict, optional
		Overrides by path glob, as described above.
	checksum : bool, default False
		Record a SHA-256 of each source file in its provenance.
	unclaimed : {"reference", "embed"}, default "reference"
		What to do with a sidecar no format decoder can read, as in
		:class:`ManifestReader`.
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

	Attributes
	----------
	df : pandas.DataFrame
		What was inferred, one row per recording or sidecar.
	report : dict
		Counts and every rule that decided something; see :func:`scan_folders`.
	questions : list of dict
		The open guesses, each ``{"field", "pattern", "guess", "why", "count", "example"}``;
		``{pattern: {field: value}}`` is the answer.

	Raises
	------
	ValidationError
		If no recording is found in the folders.

	Examples
	--------
	>>> reader = SmartReader(["/data/siteA", "/data/siteB"])
	>>> reader.questions  # check the guesses
	>>> repo.add(SmartReader(["/data/siteA", "/data/siteB"], answers={"/data/siteA/*": {"datatype": "ieeg"}}))
	"""

	def __init__(self, folders: "str | Path | Sequence[str | Path]", *,
				 answers: dict[str, dict[str, Any]] | None = None, checksum: bool = False,
				 unclaimed: UnclaimedPolicy | str = UnclaimedPolicy.REFERENCE, optimize: str = "balanced",
				 probe_level: int = 2, probe_files: int = 8) -> None:
		self.folders = [Path(folders)] if isinstance(folders, (str, Path)) else [Path(f) for f in folders]
		self.df, self.report = scan_folders(self.folders, answers)
		if self.df.empty:
			raise ValidationError(f"no recordings found in {[str(f) for f in self.folders]}")
		self.questions: list[dict[str, Any]] = self.report["questions"]
		self.checksum = checksum
		self.unclaimed = UnclaimedPolicy(unclaimed)
		check_plan_options(optimize, probe_level)
		self.optimize, self.probe_level, self.probe_files = optimize, probe_level, probe_files

	def codec_plan(self, layout: Any = None) -> "CodecPlan":
		"""Probe a few files and decide each kind of data's codec; see :mod:`neurozarr.codecs`."""
		return plan_codecs(self, layout, self.optimize, self.probe_level, self.probe_files)

	def subjects(self) -> list[str]:
		"""The subjects found, so :meth:`Repo.add` can split the work."""
		return sorted(set(self.df["sub"]))

	def for_subject(self, sub: str | None) -> "SmartReader":
		"""This reader limited to one subject; ``None`` reads only the dataset-level
		files (dataset_description.json, participants.json)."""
		limited = copy.copy(self)
		limited.df = self.df[self.df["sub"] == sub] if sub is not None else self.df.iloc[0:0]
		return limited

	def read(self) -> Iterator[Item]:
		log_progress("Reading folders", ", ".join(map(str, self.folders)), f"{len(self.df)} files")
		yield from self._dataset_attrs()
		for row in self.df.to_dict("records"):
			path = Path(row["path"])
			extra = {str(k): v for k, v in row.items() if k not in _ROW_KEYS and pd.notna(v)}
			entities = Entities(row["sub"], row["ses"] if pd.notna(row["ses"]) else None, row["datatype"], extra)
			meta = {"SourceFile": path.name,
					"_neurozarr_provenance": source_provenance(path, "SmartReader", self.checksum)}
			fmt = row["format"]
			# BIDS derivatives (derivatives/<pipeline>/...) are stored apart from the raw data,
			# as BidsReader does
			prefix = tuple(str(row["prefix"]).split("/")) if pd.notna(row.get("prefix")) else ()
			if fmt == "sidecar":
				yield from (_with_prefix(item, prefix) for item in self._sidecar(path, entities, str(row["name"]), meta))
				continue
			meta = {**clean_nan(load_json(path.with_suffix(".json"))), **meta} if path.suffix else meta
			if fmt == "mne":
				yield Recording(entities, mne.io.read_raw(path, preload=False, verbose=False), meta, prefix)
			elif fmt == "edf":
				yield Recording(entities, mne.io.read_raw_edf(_as_edf(path), preload=False, verbose=False), meta, prefix)
			elif pd.notna(row["sfreq"]):
				meta["data_source_array"] = row["array"]
				ch_type = "seeg" if row["datatype"] == "ieeg" else str(row["datatype"])
				yield Recording(entities, _load_array(fmt, path, row["array"], float(row["sfreq"]), ch_type), meta, prefix)
			else:
				yield ExternalFile(entities, path.stem, path,
								   reader_hint="no sampling rate found; answer sfreq for this file", meta=meta, prefix=prefix)

	def _sidecar(self, path: Path, entities: Entities, name: str, meta: dict[str, Any]) -> Iterator[Item]:
		ext = path.suffix.lower()
		if ext == ".json":
			yield Attrs((*entities.path(), name), clean_nan(load_json(path)))
		elif ext in (".tsv", ".csv"):
			yield Table(entities, name, pd.read_csv(path, sep="\t" if ext == ".tsv" else ","), meta)
		elif (decoder := decoder_for(path)) is not None:
			yield from decoder(path, entities, name, meta)
		elif self.unclaimed is UnclaimedPolicy.EMBED:
			yield from decode_embed(path, entities, name, meta)
		else:
			yield ExternalFile(entities, name, path, reader_hint=f"install or register a reader for {ext!r}", meta=meta)

	def _dataset_attrs(self) -> Iterator[Item]:
		"""dataset_description.json, participants.tsv/.json and *_sessions.tsv, wherever a
		listed folder has them -- for subjects that were actually found."""
		subjects = set(self.df["sub"])
		description: dict[str, Any] = {}
		for file in map(Path, self.report["dataset_files"]):
			if file.name == "dataset_description.json":
				description = {**clean_nan(load_json(file)), **description}  # earlier folders win
			elif file.name == "participants.json":
				yield Attrs(("participants_description",), clean_nan(load_json(file)))
			elif file.name == "participants.tsv":
				table = pd.read_csv(file, sep="\t", dtype=str)
				for row in table.to_dict("records"):
					sub = "sub-" + _label(str(row.pop("participant_id", "")).removeprefix("sub-"))
					if sub in subjects:
						yield Attrs((sub,), clean_nan({str(k): v for k, v in row.items()}))
			else:  # sub-XX_sessions.tsv
				sub = "sub-" + _label(file.name.removesuffix("_sessions.tsv").removeprefix("sub-"))
				if sub in subjects:
					for row in pd.read_csv(file, sep="\t", dtype=str).to_dict("records"):
						ses = "ses-" + _label(str(row.pop("session_id", "")).removeprefix("ses-"))
						yield Attrs((sub, ses), clean_nan({str(k): v for k, v in row.items()}))
		if description:
			yield Attrs((), description)
