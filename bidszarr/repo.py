from pathlib import Path

import icechunk
import mne
import pandas as pd

from .entities import Entities
from .items import Attrs, Recording, Reader, Table
from .writer import CodecConfig, Writer


def _open_writer(storage_path: Path, codec: CodecConfig) -> Writer:
	storage = icechunk.local_filesystem_storage(str(storage_path))
	icechunk_repo = icechunk.Repository.open_or_create(storage)
	session = icechunk_repo.writable_session("main")
	return Writer(session, codec)


class Repo:
	"""Top-level handle over a directory of per-subject Icechunk repositories
	(base_path/sub-XXX/...), one per subject so each can be stored/versioned/shipped
	independently, plus one base_path/_dataset/ repo for attrs that aren't about any
	single subject (dataset_description.json, participants.tsv field descriptions).
	Build it up either by ingesting a Reader (bulk import) or by hand via
	create_subject()/Subject.add_visit()/Visit.add*()."""

	def __init__(self, base_path: str, codec: CodecConfig = None):
		self.base_path = Path(base_path)
		self._codec = codec
		self._writers = {}  # sub_id -> Writer, opened lazily on first use

	def _writer_for(self, sub_id: str) -> Writer:
		if sub_id not in self._writers:
			self._writers[sub_id] = _open_writer(self.base_path / sub_id, self._codec)
		return self._writers[sub_id]

	def create_subject(self, sub_id: str, attrs: dict = None) -> "Subject":
		if attrs:
			self._writer_for(sub_id).add_attrs(Attrs((), attrs))
		return Subject(self, sub_id)

	def ingest(self, reader: Reader):
		for item in reader.read():
			if isinstance(item, Attrs):
				self._route_attrs(item)
			else:
				self._writer_for(item.entities.sub).dispatch(item)

	def _route_attrs(self, item: Attrs):
		"""An Attrs' path may embed a sub-XXX segment anywhere in it (e.g. a
		derivatives path is ("derivatives", name, sub-XXX, ...)). Whichever segment
		looks like a subject id decides which repo it belongs to; that segment is
		dropped since the target repo already is that subject. No sub-XXX segment
		at all means it's dataset-wide, and goes to the _dataset repo unchanged."""
		path = item.path
		for i, part in enumerate(path):
			if part.startswith("sub-"):
				self._writer_for(part).add_attrs(Attrs(path[:i] + path[i + 1:], item.attrs))
				return
		self._writer_for("_dataset").add_attrs(item)

	def save(self, message: str):
		for writer in self._writers.values():
			writer.save(message)


class Subject:
	def __init__(self, repo: Repo, sub_id: str):
		self._repo = repo
		self.sub_id = sub_id

	def add_visit(self, ses_id: str, attrs: dict = None) -> "Visit":
		if attrs:
			self._repo._writer_for(self.sub_id).add_attrs(Attrs((ses_id,), attrs))
		return Visit(self._repo, self.sub_id, ses_id)


class Visit:
	def __init__(self, repo: Repo, sub_id: str, ses_id: str):
		self._repo = repo
		self.sub_id = sub_id
		self.ses_id = ses_id

	def add(self, datatype: str, payload, meta: dict = None, prefix: tuple = (), **entities):
		e = Entities(self.sub_id, self.ses_id, datatype, entities)
		if isinstance(payload, mne.io.BaseRaw):
			item = Recording(e, payload, meta or {}, prefix)
		elif isinstance(payload, pd.DataFrame):
			item = Table(e, "table", payload, meta or {}, prefix)
		else:
			raise TypeError(f"unsupported recording/table payload type {type(payload)!r}")
		self._repo._writer_for(self.sub_id).dispatch(item)

	def add_recording(self, raw: "mne.io.BaseRaw", meta: dict = None, **entities):
		self.add("ieeg", raw, meta, **entities)

	def add_behavioral_table(self, df: pd.DataFrame, meta: dict = None, **entities):
		self.add("beh", df, meta, **entities)
