import copy
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ..entities import Entities, split_stem
from ..errors import ValidationError
from ..items import Attrs, Item, Table
from ..log import log_progress
from ..util import clean_nan, load_json, source_provenance
from ..codecs import CodecPlan, check_plan_options, plan_codecs
from .filetypes import MNE_EXTS, _text_doc, companions, decodable, decode_file, folder_kind, signal_kind, text_attrs, true_ext
from .formats import UnclaimedPolicy

# Kept for importers: every extension mne opens by name (see readers/filetypes.py).
MNE_READABLE_EXTS = MNE_EXTS
_true_ext = true_ext


def _is_text_doc(path: Path) -> bool:
	return _text_doc(path)


def _is_signal(path: Path) -> bool:
	"""A recording -- the one item per entity set that must not have a twin."""
	kind = folder_kind(path) if path.is_dir() else signal_kind(path)
	return kind is not None and kind != "zarr"


def _size(path: Path) -> int:
	"""Bytes on disk: a file's size, or everything under a folder."""
	return path.stat().st_size if path.is_file() else sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


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

	Every file is decoded the same way every reader decodes it (see
	``neurozarr/readers/filetypes.py``): signals through mne or neo, tables with
	pandas, images and volumes as arrays, media as embedded bytes. A file nothing
	reads is kept as a reference to its path rather than failing the run, and
	files outside the BIDS tree that nothing reads (``stimuli/``, ``code/``...) are
	named in the dataset's ``files_not_stored`` attrs.

	Parameters
	----------
	root_dir : str or pathlib.Path
		The dataset root — the directory holding ``dataset_description.json``
		and the ``sub-*`` directories.
	subjects : list of str, optional
		Read only these subjects. An empty list reads none, which is how
		dataset-level metadata is ingested on its own. Default reads all.
	include_derivatives : bool, default True
		Also convert everything under a ``derivatives/`` folder, if the
		dataset has one — read unconditionally by default, the same as any
		other datatype directory, filtered by ``subjects`` the same way.
		Pass ``False`` to convert only the dataset's raw data.
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

	Examples
	--------
	>>> repo.add(BidsReader("./my_bids_dataset"))
	>>> repo.add(BidsReader("./my_bids_dataset", include_derivatives=False))
	"""

	def __init__(self, root_dir: str | Path, subjects: list[str] | None = None,
				 checksum: bool = False, unclaimed: UnclaimedPolicy | str = UnclaimedPolicy.REFERENCE,
				 include_derivatives: bool = True, optimize: str = "balanced", probe_level: int = 2,
				 probe_files: int = 8):
		# Dataset-level metadata is yielded whatever `subjects` says, so parallel
		# workers each converting one subject still agree on it.
		self.root_dir = Path(root_dir)
		self._only = None if subjects is None else set(subjects)
		# parallel ingest: None reads everything; a dict collects each unit's size instead
		# of reading any unit; a tuple reads only those units (see units())
		self._units: dict[Any, int] | tuple[Any, ...] | None = None
		self.checksum = checksum
		self.unclaimed = UnclaimedPolicy(unclaimed)
		self.include_derivatives = include_derivatives
		check_plan_options(optimize, probe_level)
		self.optimize, self.probe_level, self.probe_files = optimize, probe_level, probe_files

	def codec_plan(self, layout: Any = None) -> "CodecPlan":
		"""Probe a few files and decide each kind of data's codec; see :mod:`neurozarr.codecs`."""
		return plan_codecs(self, layout, self.optimize, self.probe_level, self.probe_files)

	def subjects(self) -> list[str]:
		"""The subjects this reader reads, so :meth:`Repo.add` can split the work."""
		return [d.name for d in sorted(self.root_dir.glob("sub-*")) if d.is_dir() and self._wanted(d.name)]

	def for_subject(self, sub: str | None) -> "BidsReader":
		"""This reader limited to one subject; ``None`` reads only dataset-level metadata."""
		limited = copy.copy(self)
		limited._only = set() if sub is None else {sub}
		return limited

	def units(self) -> list[tuple[Any, int]]:
		"""Each recording with its own sidecars -- one set of entities in one datatype
		folder -- as ``(unit, bytes on disk)``, so :meth:`Repo.add` can spread them over
		processes: no two units write the same item."""
		walker = copy.copy(self)
		found: dict[Any, int] = {}
		walker._units = found
		for _ in walker.read():
			pass
		return list(found.items())

	def for_units(self, units: list[Any]) -> "BidsReader":
		"""This reader limited to some of :meth:`units`, and to nothing outside them."""
		limited = copy.copy(self)
		limited._units = tuple(units)
		return limited

	def without_units(self) -> "BidsReader":
		"""This reader limited to what lies outside every unit: dataset, subject and
		session metadata."""
		limited = copy.copy(self)
		limited._units = {}
		return limited

	def _wanted(self, sub: str) -> bool:
		return self._only is None or sub in self._only

	def read(self) -> Iterator[Item]:
		if isinstance(self._units, tuple):
			folders: dict[tuple[Any, ...], set[Any]] = {}
			for prefix, sub, ses, datatype, folder, entities in self._units:
				folders.setdefault((prefix, sub, ses, datatype, folder), set()).add(entities)
			for (prefix, sub, ses, datatype, folder), wanted in folders.items():
				yield from self._read_datatype_dir(sub, ses, datatype, Path(folder), prefix, wanted)
			return
		log_progress("Reading BIDS dataset", str(self.root_dir))
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
			log_progress("Reading subject", sub)
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
				yield from self._read_scans(sub, ses, sesDir)
				for dtDir in sorted(sesDir.iterdir()):
					if dtDir.is_dir() and dtDir.name != "derivatives":
						yield from self._read_datatype_dir(sub, ses, dtDir.name, dtDir)

		if self.include_derivatives:
			yield from self._read_derivatives()

		for doc in sorted(self.root_dir.iterdir()):
			if _is_text_doc(doc):
				yield text_attrs(doc, ())
		yield from self._read_phenotype()
		yield from self._read_samples()

		# what none of the above reads -- README, stimuli/, phenotype/, sourcedata/... -- is
		# named, so a store never looks complete while silently missing part of its source
		not_stored = self._not_stored()
		if not_stored:
			warnings.warn(f"{len(not_stored)} files in {self.root_dir} are not stored (listed in the "
						  f"dataset's 'files_not_stored' attrs), e.g. {', '.join(list(not_stored)[:3])}", stacklevel=2)
			yield Attrs(("files_not_stored",), not_stored)

	def _read_phenotype(self) -> Iterator[Item]:
		"""``phenotype/<measure>.tsv``: each participant's row on that subject, under
		``phenotype/<measure>``; the measure's JSON description on the dataset."""
		folder = self.root_dir / "phenotype"
		if not folder.is_dir():
			return
		for path in sorted(folder.glob("*.tsv")):
			measure = path.stem
			description = clean_nan(load_json(path.with_suffix(".json")))
			if description:
				yield Attrs(("phenotype", measure), description)
			for row in pd.read_csv(path, sep="\t", dtype=str).to_dict("records"):
				sub = str(row.pop("participant_id", ""))
				if sub.startswith("sub-") and (self.root_dir / sub).is_dir() and self._wanted(sub):
					yield Attrs((sub, "phenotype", measure), clean_nan({str(k): v for k, v in row.items()}))

	def _read_samples(self) -> Iterator[Item]:
		"""Microscopy's ``samples.tsv``: each sample's row on its subject, under ``samples/<id>``."""
		path = self.root_dir / "samples.tsv"
		if not path.exists():
			return
		description = clean_nan(load_json(path.with_suffix(".json")))
		if description:
			yield Attrs(("samples_description",), description)
		for row in pd.read_csv(path, sep="\t", dtype=str).to_dict("records"):
			sub, sample = str(row.get("participant_id", "")), str(row.get("sample_id", ""))
			if sub.startswith("sub-") and sample and (self.root_dir / sub).is_dir() and self._wanted(sub):
				yield Attrs((sub, "samples", sample), clean_nan({str(k): v for k, v in row.items()}))

	def _read_scans(self, sub: str, ses: str | None, level: Path) -> Iterator[Item]:
		"""A session's (or session-less subject's) ``*_scans.tsv``, as a table."""
		for path in sorted(level.glob("*_scans.tsv")):
			meta = clean_nan(load_json(path.with_suffix(".json")))
			yield Table(Entities(sub, ses), "scans", pd.read_csv(path, sep="\t"),
						{**meta, "_neurozarr_provenance": source_provenance(path, "BidsReader", self.checksum)})

	def _not_stored(self) -> dict[str, int]:
		"""Files this reader reads nothing from, as {path under the root: bytes}. Mirrors
		read()'s walk rather than tracking what it consumed, so a parallel ingest (whose
		workers read the data) lists exactly what a serial one does."""
		# ponytail: any .json above a datatype folder counts as read (an inherited sidecar);
		# one no data file inherits is not listed. Exact would need _find_sidecar per file.
		out: dict[str, int] = {}

		def note(path: Path) -> None:
			for f in ([path] if path.is_file() else sorted(path.rglob("*"))):
				if f.is_file() and not any(part.startswith(".") for part in f.relative_to(self.root_dir).parts):
					out[str(f.relative_to(self.root_dir))] = f.stat().st_size

		def level(directory: Path) -> None:  # a session folder, or a session-less subject
			for entry in sorted(directory.iterdir()):
				if entry.name.startswith(".") or entry.is_dir() or entry.suffix == ".json" \
						or entry.name.endswith("_scans.tsv"):
					continue  # datatype folders are read whole; see _read_datatype_dir
				note(entry)

		def subject(sub_dir: Path, known: set[str]) -> None:
			sessions = [d for d in sorted(sub_dir.glob("ses-*")) if d.is_dir()]
			if not sessions:
				level(sub_dir)
				return
			for entry in sorted(sub_dir.iterdir()):
				if entry.name.startswith(".") or entry in sessions or entry.name in known or entry.suffix == ".json":
					continue
				note(entry)
			for ses_dir in sessions:
				level(ses_dir)

		def dataset(root: Path, derivatives: bool) -> None:
			for entry in sorted(root.iterdir()):
				name = entry.name
				if name.startswith(".") or name in ("dataset_description.json", "participants.tsv", "samples.tsv") \
						or entry.suffix == ".json" or _is_text_doc(entry):
					continue
				if entry.is_dir() and name == "phenotype" and root == self.root_dir:
					for f in sorted(entry.iterdir()):
						if f.suffix not in (".tsv", ".json") and not f.name.startswith("."):
							note(f)
					continue
				if entry.is_dir() and name.startswith("sub-"):
					subject(entry, {f"{name}_sessions.tsv"})
				elif entry.is_dir() and name == "derivatives" and derivatives:
					for deriv in sorted(entry.iterdir()):
						if deriv.is_dir():
							dataset(deriv, False)
						elif not deriv.name.startswith("."):
							note(deriv)
				elif not (name == "derivatives" and not self.include_derivatives):
					note(entry)

		dataset(self.root_dir, self.include_derivatives)
		return out

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
						   prefix: tuple[str, ...] = (), wanted: set[Any] | None = None) -> Iterator[Item]:
		groups: dict[tuple[tuple[tuple[str, str], ...], str], dict[str, Path]] = {}
		for f in sorted(dir_path.iterdir()):
			if f.name.startswith("."):
				continue
			ext = _true_ext(f)  # a folder is an entry too: CTF .ds, EGI .mff, MEF3 .mefd
			stem = f.name[:-len(ext)] if ext else f.stem
			entities, suffix = split_stem(stem)
			entities.pop("sub", None)
			entities.pop("ses", None)
			key = (tuple(sorted(entities.items())), suffix)
			if ext in groups.setdefault(key, {}):
				raise ValidationError(f"multiple {ext} files represent {stem!r} in {dir_path}")
			groups[key][ext] = f

		for (entityItems, suffix), exts in groups.items():
			if isinstance(self._units, dict):
				unit = (prefix, sub, ses, datatype, str(dir_path), entityItems)
				self._units[unit] = self._units.get(unit, 0) + sum(_size(f) for f in exts.values())
				continue
			if wanted is not None and entityItems not in wanted:
				continue
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
			# a bundle's companions (BrainVision's .eeg/.vmrk, EEGLAB's .fdt...) are read
			# through its entry point; every other file under these entities is an item
			members = {e: exts[e] for e in dataExts}
			companion_files = set().union(*(companions(f) for f in members.values()))
			candidates = sorted((e for e in members if members[e] not in companion_files),
								key=lambda e: (not _is_signal(members[e]), not decodable(members[e]), e))
			if sum(_is_signal(members[e]) for e in candidates) > 1:
				# two recordings with the same entities would be written to the same group
				raise ValidationError(
					f"ambiguous data bundle for {suffix!r} in {dir_path}: {', '.join(candidates)}")
			for i, ext in enumerate(candidates):
				path = members[ext]
				item_name = (suffix or path.stem) if i == 0 else f"{suffix or path.stem}_{ext.lstrip('.').replace('.', '-')}"
				yield from decode_file(
					path, item_entities, item_name,
					{**meta, "_neurozarr_provenance": source_provenance(path, "BidsReader", self.checksum)},
					self.unclaimed, prefix)

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
