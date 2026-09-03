from pathlib import Path

import mne
import pandas as pd

from ..entities import Entities, split_stem
from ..items import Attrs, Recording, Table
from ..util import clean_nan, load_json

# Formats mne.io.read_raw() auto-dispatches on, covering eeg/ieeg/meg/nirs data
# generically. Imaging datatypes (anat/func/dwi -- NIfTI) need nibabel, which
# isn't a dependency here; files with any other extension fall back to an
# "unread" Attrs placeholder instead of crashing (see _read_datatype_dir).
MNE_READABLE_EXTS = {".edf", ".bdf", ".gdf", ".vhdr", ".set", ".fif", ".cnt"}


class BidsReader:
	"""Reads any valid BIDS dataset generically: subjects/sessions are
	discovered by directory, not by a required sessions.tsv; datatype
	directories (ieeg, eeg, beh, anat, ...) are walked without a fixed list;
	JSON sidecars are resolved via the BIDS inheritance principle (closest
	matching sidecar wins). BRAVO-specific extras (.bravo_subject_ids.json,
	_devices.json, _manifest.json) are used only if present -- never required."""

	def __init__(self, root_dir, subjects: list = None):
		"""subjects, if given, restricts which sub-* directories are read (an empty
		list reads none). Dataset-level metadata is always yielded, so workers
		converting one subject each still agree on it."""
		self.root_dir = Path(root_dir)
		self.subjects = None if subjects is None else set(subjects)

	def _wanted(self, sub: str) -> bool:
		return self.subjects is None or sub in self.subjects

	def read(self):
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
			levels = [(d.name, d) for d in sesDirs if d.is_dir()] or [(None, subDir)]
			for ses, sesDir in levels:
				if ses and ses in sessionAttrs:
					yield Attrs((sub, ses), sessionAttrs[ses])
				for dtDir in sorted(sesDir.iterdir()):
					if dtDir.is_dir() and dtDir.name != "derivatives":
						yield from self._read_datatype_dir(sub, ses, dtDir.name, dtDir)

		yield from self._read_derivatives()

	def _load_participants(self) -> tuple:
		path = self.root_dir / "participants.tsv"
		if not path.exists():
			return {}, {}
		participants = pd.read_csv(path, sep="\t").set_index("participant_id")
		fieldInfo = clean_nan(load_json(self.root_dir / "participants.json"))
		subjectIds = load_json(self.root_dir / ".bravo_subject_ids.json")  # optional BRAVO extra
		subToHash = {f"sub-{num:03d}": h for h, num in subjectIds.items()}
		attrs = {
			sub: clean_nan({**row.to_dict(), **({"bravo_hash": subToHash[sub]} if sub in subToHash else {})})
			for sub, row in participants.iterrows()
		}
		return attrs, fieldInfo

	def _load_sessions(self, sub: str) -> tuple:
		subDir = self.root_dir / sub
		path = subDir / f"{sub}_sessions.tsv"
		if not path.exists():
			return {}, {}
		sessions = pd.read_csv(path, sep="\t").set_index("session_id")
		fieldInfo = clean_nan(load_json(subDir / f"{sub}_sessions.json"))
		devices = load_json(subDir / f"{sub}_devices.json")  # optional BRAVO extra
		manifest = {e["Session"]: e for e in load_json(subDir / f"{sub}_manifest.json").get("Sessions", [])}
		attrs = {
			ses: clean_nan({**row.to_dict(), **devices.get(ses, {}), "manifest": manifest.get(ses, {})})
			for ses, row in sessions.iterrows()
		}
		return attrs, fieldInfo

	def _read_datatype_dir(self, sub, ses, datatype, dir_path, prefix=()):
		groups = {}
		for f in sorted(dir_path.iterdir()):
			if not f.is_file():
				continue
			entities, suffix = split_stem(f.stem)
			entities.pop("sub", None)
			entities.pop("ses", None)
			key = (tuple(sorted(entities.items())), suffix)
			groups.setdefault(key, {})[f.suffix] = f

		for (entityItems, suffix), exts in groups.items():
			entities = Entities(sub, ses, datatype, dict(entityItems))
			dataExts = [e for e in exts if e != ".json"]
			if not dataExts:
				# pure sidecar json with no data of its own (e.g. coordsystem.json) --
				# still materialized here, and independently discoverable via
				# _find_sidecar for any sibling group that needs it as inherited meta.
				if ".json" in exts:
					yield Attrs((*prefix, *entities.path(), suffix), clean_nan(load_json(exts[".json"])))
				continue

			meta = clean_nan(load_json(exts[".json"])) if ".json" in exts \
				else self._find_sidecar(dir_path, dict(entityItems), suffix)

			ext = dataExts[0]
			path = exts[ext]
			if ext == ".tsv":
				yield Table(entities, suffix, pd.read_csv(path, sep="\t"), meta, prefix)
			elif ext in MNE_READABLE_EXTS:
				raw = mne.io.read_raw(path, preload=True, verbose=False)
				yield Recording(entities, raw, meta, prefix)
			else:
				yield Attrs((*prefix, *entities.path(), suffix), {
					**meta, "unread_file": str(path),
					"note": f"no in-memory reader registered for {ext!r}",
				})

	def _find_sidecar(self, start_dir: Path, entities: dict, suffix: str) -> dict:
		"""BIDS inheritance principle: nearest matching JSON sidecar wins,
		walking from the file's own directory up to the dataset root."""
		d = start_dir
		while True:
			for jf in sorted(d.glob(f"*{suffix}.json")):
				candidate, candidateSuffix = split_stem(jf.stem)
				candidate.pop("sub", None)
				candidate.pop("ses", None)
				if candidateSuffix == suffix and all(entities.get(k) == v for k, v in candidate.items()):
					return clean_nan(load_json(jf))
			if d == self.root_dir:
				return {}
			d = d.parent

	def _read_derivatives(self):
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
				levels = [(d.name, d) for d in sesDirs if d.is_dir()] or [(None, subDir)]
				for ses, sesDir in levels:
					for dtDir in sorted(sesDir.iterdir()):
						if dtDir.is_dir():
							yield from self._read_datatype_dir(sub, ses, dtDir.name, dtDir, prefix=prefix)
