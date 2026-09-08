from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import mne  # type: ignore[import-untyped]  # mne ships no type information
import pandas as pd

from ..entities import Entities, split_stem
from ..errors import ValidationError
from ..items import Attrs, ExternalFile, Item, Recording, Table
from ..util import clean_nan, load_json, source_provenance
from .formats import UnclaimedPolicy, decode_embed, decoder_for

# Formats mne.io.read_raw() auto-dispatches on, covering eeg/ieeg/meg/nirs data
# generically. Imaging datatypes (anat/func/dwi) are handled by a registered
# format decoder when one claims the extension (NIfTI, automatically, if
# nibabel is installed -- see readers/formats.py); anything else falls back to
# the `unclaimed` policy instead of crashing (see _read_datatype_dir).
MNE_READABLE_EXTS = {".edf", ".bdf", ".gdf", ".vhdr", ".set", ".fif", ".cnt"}

# Path.suffix only ever returns the last dot-segment, so "sub-001_T1w.nii.gz"
# reports ".gz" -- and the BIDS suffix "T1w" would come out as "T1w.nii". Listed
# here rather than derived from whatever happens to be registered in
# readers.formats, so stem-parsing doesn't change depending on whether an
# optional dependency (nibabel) is installed.
_COMPOUND_EXTS = (".nii.gz",)


def _true_ext(path: Path) -> str:
	"""path.suffix, but recognizing a compound extension in _COMPOUND_EXTS."""
	name = path.name.lower()
	for ext in _COMPOUND_EXTS:
		if name.endswith(ext):
			return ext
	return path.suffix.lower()


def _row_dict(row: "pd.Series") -> dict[str, Any]:
	"""A DataFrame row as a plain dict.

	to_dict() rather than dict(row): it converts numpy scalars to Python natives,
	and these end up in zarr attrs, which must be JSON-serializable.
	"""
	return {str(k): v for k, v in row.to_dict().items()}


class BidsReader:
	"""Read a BIDS dataset from disk.

	Works on any valid BIDS dataset: subjects and sessions are discovered by
	directory rather than requiring a ``sessions.tsv``, datatype directories
	(``ieeg``, ``eeg``, ``beh``, ``anat``, …) are walked without a fixed list,
	and JSON sidecars are resolved through the BIDS inheritance principle, where
	the closest matching sidecar wins.

	Data files are read by extension: ``.tsv`` with pandas, and EDF, BDF, GDF,
	BrainVision, EEGLAB, FIF and CNT through mne. A file in any other format is
	recorded as a reference to its path rather than failing the run.

	Parameters
	----------
	root_dir : str or pathlib.Path
		The dataset root — the directory holding ``dataset_description.json``
		and the ``sub-*`` directories.
	subjects : list of str, optional
		Read only these subjects. An empty list reads none, which is how
		dataset-level metadata is ingested on its own. Default reads all.

	Examples
	--------
	>>> repo.ingest(BidsReader("./my_bids_dataset"))
	"""

	def __init__(self, root_dir: str | Path, subjects: list[str] | None = None,
				 checksum: bool = False, unclaimed: UnclaimedPolicy | str = UnclaimedPolicy.REFERENCE):
		# Dataset-level metadata is yielded whatever `subjects` says, so parallel
		# workers each converting one subject still agree on it.
		self.root_dir = Path(root_dir)
		self.subjects = None if subjects is None else set(subjects)
		self.checksum = checksum
		self.unclaimed = UnclaimedPolicy(unclaimed)

	def _wanted(self, sub: str) -> bool:
		return self.subjects is None or sub in self.subjects

	def read(self) -> Iterator[Item]:
		yield Attrs((), clean_nan(load_json(self.root_dir / "dataset_description.json")))

		participantAttrs, participantsFieldInfo = self._load_participants()
		if participantsFieldInfo:
			yield Attrs(("participants_description",), participantsFieldInfo)

		for subDir in sorted(self.root_dir.glob("sub-*")):
			if not subDir.is_dir():
				continue
			sub = subDir.name
			if not self._wanted(sub):
				continue
			if sub in participantAttrs:
				yield Attrs((sub,), participantAttrs[sub])

			sessionAttrs, sessionsFieldInfo = self._load_sessions(sub)
			if sessionsFieldInfo:
				yield Attrs((sub, "sessions_description"), sessionsFieldInfo)

			sesDirs = sorted(subDir.glob("ses-*"))
			levels: list[tuple[str | None, Path]] = [(d.name, d) for d in sesDirs if d.is_dir()] or [(None, subDir)]
			for ses, sesDir in levels:
				if ses and ses in sessionAttrs:
					yield Attrs((sub, ses), sessionAttrs[ses])
				for dtDir in sorted(sesDir.iterdir()):
					if dtDir.is_dir() and dtDir.name != "derivatives":
						yield from self._read_datatype_dir(sub, ses, dtDir.name, dtDir)

		yield from self._read_derivatives()

	def _load_participants(self) -> tuple[dict[str, Any], dict[str, Any]]:
		path = self.root_dir / "participants.tsv"
		if not path.exists():
			return {}, {}
		participants = pd.read_csv(path, sep="\t").set_index("participant_id")
		fieldInfo = clean_nan(load_json(self.root_dir / "participants.json"))
		subjectIds = load_json(self.root_dir / ".bravo_subject_ids.json")  # optional BRAVO extra
		subToHash = {f"sub-{num:03d}": h for h, num in subjectIds.items()}
		attrs = {
			str(sub): clean_nan({**_row_dict(row), **({"bravo_hash": subToHash[str(sub)]} if str(sub) in subToHash else {})})
			for sub, row in participants.iterrows()
		}
		return attrs, fieldInfo

	def _load_sessions(self, sub: str) -> tuple[dict[str, Any], dict[str, Any]]:
		subDir = self.root_dir / sub
		path = subDir / f"{sub}_sessions.tsv"
		if not path.exists():
			return {}, {}
		sessions = pd.read_csv(path, sep="\t").set_index("session_id")
		fieldInfo = clean_nan(load_json(subDir / f"{sub}_sessions.json"))
		devices = load_json(subDir / f"{sub}_devices.json")  # optional BRAVO extra
		manifest = {e["Session"]: e for e in load_json(subDir / f"{sub}_manifest.json").get("Sessions", [])}
		attrs = {
			str(ses): clean_nan({**_row_dict(row), **devices.get(str(ses), {}), "manifest": manifest.get(str(ses), {})})
			for ses, row in sessions.iterrows()
		}
		return attrs, fieldInfo

	def _read_datatype_dir(self, sub: str, ses: str | None, datatype: str, dir_path: Path,
						   prefix: tuple[str, ...] = ()) -> Iterator[Item]:
		groups: dict[tuple[tuple[tuple[str, str], ...], str], dict[str, Path]] = {}
		for f in sorted(dir_path.iterdir()):
			if not f.is_file():
				continue
			ext = _true_ext(f)
			stem = f.name[:-len(ext)] if ext else f.stem
			entities, suffix = split_stem(stem)
			entities.pop("sub", None)
			entities.pop("ses", None)
			key = (tuple(sorted(entities.items())), suffix)
			if ext in groups.setdefault(key, {}):
				raise ValidationError(f"multiple {ext} files represent {stem!r} in {dir_path}")
			groups[key][ext] = f

		for (entityItems, suffix), exts in groups.items():
			item_entities = Entities(sub, ses, datatype, dict(entityItems))
			dataExts = [e for e in exts if e != ".json"]
			if not dataExts:
				# pure sidecar json with no data of its own (e.g. coordsystem.json) --
				# still materialized here, and independently discoverable via
				# _find_sidecar for any sibling group that needs it as inherited meta.
				if ".json" in exts:
					yield Attrs((*prefix, *item_entities.path(), suffix), clean_nan(load_json(exts[".json"])))
				continue

			meta = self._find_sidecar(dir_path, dict(entityItems), suffix)

			# BrainVision is a three-file bundle; only .vhdr is the entry point.
			# Otherwise prefer a supported data file and reject ambiguous bundles.
			preferred = [ext for ext in (".vhdr", ".edf", ".bdf", ".gdf", ".set", ".fif", ".cnt", ".tsv")
						 if ext in dataExts]
			if len(preferred) > 1:
				raise ValidationError(
					f"ambiguous data bundle for {suffix!r} in {dir_path}: {', '.join(preferred)}"
				)
			ext = preferred[0] if preferred else sorted(dataExts)[0]
			path = exts[ext]
			meta = {**meta, "_neurozarr_provenance": source_provenance(path, "BidsReader", self.checksum)}
			decoder = decoder_for(path)
			item_name = suffix or path.stem
			if ext == ".tsv":
				yield Table(item_entities, suffix, pd.read_csv(path, sep="\t"), meta, prefix)
			elif ext in MNE_READABLE_EXTS:
				raw = mne.io.read_raw(path, preload=False, verbose=False)
				yield Recording(item_entities, raw, meta, prefix)
			elif decoder is not None:
				# a format decoder builds items with prefix=() -- it has no notion of
				# derivatives/ routing, which is this reader's concern, not the decoder's.
				yield from (replace(built, prefix=prefix) for built in decoder(path, item_entities, item_name, meta))
			elif self.unclaimed is UnclaimedPolicy.EMBED:
				yield from (replace(built, prefix=prefix) for built in decode_embed(path, item_entities, item_name, meta))
			else:
				yield ExternalFile(item_entities, item_name, path,
					reader_hint=f"install or register a reader for {ext!r}", meta=meta, prefix=prefix)

	def _find_sidecar(self, start_dir: Path, entities: dict[str, str], suffix: str) -> dict[str, Any]:
		"""Merge applicable JSON sidecars from the dataset root to the data file."""
		directories: list[Path] = []
		d = start_dir
		while True:
			directories.append(d)
			if d == self.root_dir:
				break
			if self.root_dir not in d.parents:
				return {}
			d = d.parent

		merged: dict[str, Any] = {}
		for directory in reversed(directories):
			applicable: list[tuple[int, Path]] = []
			for jf in sorted(directory.glob(f"*{suffix}.json")):
				candidate, candidateSuffix = split_stem(jf.stem)
				candidate.pop("sub", None)
				candidate.pop("ses", None)
				if candidateSuffix == suffix and all(entities.get(k) == v for k, v in candidate.items()):
					applicable.append((len(candidate), jf))
			for specificity in sorted({level for level, _ in applicable}):
				layer: dict[str, Any] = {}
				sources: dict[str, Path] = {}
				for _, path in (entry for entry in applicable if entry[0] == specificity):
					for key, value in clean_nan(load_json(path)).items():
						if key in layer and layer[key] != value:
							raise ValidationError(
								f"ambiguous inherited value for {key!r} in {sources[key].name} and {path.name}"
							)
						layer[key] = value
						sources[key] = path
				merged.update(layer)
		return merged

	def _read_derivatives(self) -> Iterator[Item]:
		derivativesDir = self.root_dir / "derivatives"
		if not derivativesDir.exists():
			return
		for derivDir in sorted(derivativesDir.iterdir()):
			if not derivDir.is_dir():
				continue
			prefix = ("derivatives", derivDir.name)
			yield Attrs(prefix, clean_nan(load_json(derivDir / "dataset_description.json")))
			for subDir in sorted(derivDir.glob("sub-*")):
				if not subDir.is_dir():
					continue
				sub = subDir.name
				if not self._wanted(sub):
					continue
				sesDirs = sorted(derivDir.glob(f"{sub}/ses-*"))
				levels: list[tuple[str | None, Path]] = \
					[(d.name, d) for d in sesDirs if d.is_dir()] or [(None, subDir)]
				for ses, sesDir in levels:
					for dtDir in sorted(sesDir.iterdir()):
						if dtDir.is_dir():
							yield from self._read_datatype_dir(sub, ses, dtDir.name, dtDir, prefix=prefix)
