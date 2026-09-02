from pathlib import Path

import mne
import pandas as pd

from ..entities import Entities, parse_entities
from ..items import Attrs, Recording, Table
from ..util import cleanNan, loadJson


class BidsReader:
	"""Reads a BIDS-formatted export folder and yields standardized
	Attrs/Table/Recording items for a Writer to consume."""

	def __init__(self, root_dir):
		self.root_dir = Path(root_dir)

	def read(self):
		yield Attrs((), cleanNan(loadJson(self.root_dir / "dataset_description.json")))
		for sub, attrs in self._loadParticipantAttributes().items():
			for ses, sattrs in self._loadSessionAttributes(sub).items():
				yield Attrs((sub, ses), cleanNan(sattrs))
				sessionDir = self.root_dir / sub / ses
				if (sessionDir / "ieeg").exists():
					yield from self._read_ieeg(sub, ses)
				if (sessionDir / "beh").exists():
					yield from self._read_beh(self.root_dir, sub, ses)
			yield Attrs((sub,), cleanNan(attrs))
		yield from self._read_derivatives()

	def _loadParticipantAttributes(self) -> dict:
		participants = pd.read_csv(self.root_dir / "participants.tsv", sep="\t").set_index("participant_id")
		fieldInfo = loadJson(self.root_dir / "participants.json")
		subjectIds = loadJson(self.root_dir / ".bravo_subject_ids.json")
		subToHash = {f"sub-{num:03d}": h for h, num in subjectIds.items()}
		return {
			sub: {**row.to_dict(), "bravo_hash": subToHash.get(sub), "field_info": fieldInfo}
			for sub, row in participants.iterrows()
		}

	def _loadSessionAttributes(self, sub: str) -> dict:
		subDir = self.root_dir / sub
		sessions = pd.read_csv(subDir / f"{sub}_sessions.tsv", sep="\t").set_index("session_id")
		fieldInfo = loadJson(subDir / f"{sub}_sessions.json")
		devices = loadJson(subDir / f"{sub}_devices.json")
		manifest = {entry["Session"]: entry for entry in loadJson(subDir / f"{sub}_manifest.json").get("Sessions", [])}
		return {
			ses: {**row.to_dict(), **devices.get(ses, {}), "field_info": fieldInfo, "manifest": manifest.get(ses, {})}
			for ses, row in sessions.iterrows()
		}

	def _read_ieeg(self, sub, ses):
		ieegDir = self.root_dir / sub / ses / "ieeg"
		fixedPrefix = f"{sub}_{ses}_space-Other"

		coordsystem = cleanNan(loadJson(ieegDir / f"{fixedPrefix}_coordsystem.json"))
		yield Attrs(("ieeg_descriptions", "coordsystem"), coordsystem)

		electrodes = pd.read_csv(ieegDir / f"{fixedPrefix}_electrodes.tsv", sep="\t")
		yield Table(Entities(sub, ses, "ieeg"), "electrodes", electrodes)
		electrodesDescription = cleanNan(loadJson(ieegDir / f"{fixedPrefix}_electrodes.json"))
		yield Attrs(("ieeg_descriptions", "electrodes"), electrodesDescription)

		scansPath = self.root_dir / sub / ses / f"{sub}_{ses}_scans.tsv"
		scans = pd.read_csv(scansPath, sep="\t") if scansPath.exists() else pd.DataFrame(columns=["filename"])

		for _, row in scans.iterrows():
			fullName = Path(row["filename"]).stem.removesuffix("_ieeg")
			runName = fullName.removeprefix(f"{sub}_{ses}_")
			entities = Entities(sub, ses, "ieeg", parse_entities(runName))

			runAttrs = cleanNan(row.drop("filename").to_dict())
			runAttrs.update(cleanNan(loadJson(ieegDir / f"{fullName}_ieeg.json")))
			edf = mne.io.read_raw_edf(ieegDir / f"{fullName}_ieeg.edf", preload=True, verbose=False)
			yield Recording(entities, edf, runAttrs)

			channels = pd.read_csv(ieegDir / f"{fullName}_channels.tsv", sep="\t")
			yield Table(entities, "channels", channels)
			channelsDescription = cleanNan(loadJson(ieegDir / f"{fullName}_channels.json"))
			yield Attrs(("ieeg_descriptions", "channels"), channelsDescription)

			eventsPath = ieegDir / f"{fullName}_events.tsv"
			if eventsPath.exists():
				events = pd.read_csv(eventsPath, sep="\t")
				eventsDescription = cleanNan(loadJson(self.root_dir / "events.json"))
				yield Table(entities, "events", events, eventsDescription)

	def _read_beh(self, base_dir, sub, ses, prefix=()):
		behDir = base_dir / sub / ses / "beh"
		for tsvPath in behDir.glob("*_beh.tsv"):
			fullName = tsvPath.stem.removesuffix("_beh")
			taskName = fullName.removeprefix(f"{sub}_{ses}_")
			entities = Entities(sub, ses, "beh", parse_entities(taskName))

			beh = pd.read_csv(tsvPath, sep="\t")
			yield Table(entities, "table", beh, prefix=prefix)

			description = cleanNan(loadJson(behDir / f"{fullName}_beh.json"))
			if taskName.startswith("task-PatientSurvey"):
				yield Attrs((*prefix, sub, ses, "beh", taskName), description)
			else:
				yield Attrs((*prefix, "beh_descriptions", taskName), description)

	def _read_derivatives(self):
		derivativesDir = self.root_dir / "derivatives"
		if not derivativesDir.exists():
			return
		for derivDir in derivativesDir.iterdir():
			if not derivDir.is_dir():
				continue
			yield Attrs(("derivatives", derivDir.name), cleanNan(loadJson(derivDir / "dataset_description.json")))
			for sesDir in derivDir.glob("sub-*/ses-*"):
				sub, ses = sesDir.parts[-2], sesDir.parts[-1]
				yield from self._read_beh(derivDir, sub, ses, prefix=("derivatives", derivDir.name))
