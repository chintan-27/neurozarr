import icechunk
import mne
import pandas as pd

from .entities import Entities
from .items import Attrs, Recording, Reader, Table
from .writer import CodecConfig, Writer


class Repo:
	"""Top-level handle: an Icechunk repository + writable session + Writer.
	Build it up either by ingesting a Reader (bulk import) or by hand via
	createSubject()/Subject.addVisit()/Visit.add*()."""

	def __init__(self, path: str = None, codec: CodecConfig = None, session: icechunk.Session = None):
		if session is None:
			storage = icechunk.local_filesystem_storage(str(path))
			icechunk_repo = icechunk.Repository.open(storage) if icechunk.Repository.exists(storage) \
				else icechunk.Repository.create(storage)
			session = icechunk_repo.writable_session("main")
		self._writer = Writer(session, codec)

	def createSubject(self, sub_id: str, attrs: dict = None) -> "Subject":
		if attrs:
			self._writer.add_attrs(Attrs((sub_id,), attrs))
		return Subject(self, sub_id)

	def ingest(self, reader: Reader):
		for item in reader.read():
			self._writer.dispatch(item)

	def save(self, message: str):
		return self._writer.save(message)


class Subject:
	def __init__(self, repo: Repo, sub_id: str):
		self._repo = repo
		self.sub_id = sub_id

	def addVisit(self, ses_id: str, attrs: dict = None) -> "Visit":
		if attrs:
			self._repo._writer.add_attrs(Attrs((self.sub_id, ses_id), attrs))
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
		self._repo._writer.dispatch(item)

	def addRecording(self, raw: "mne.io.BaseRaw", meta: dict = None, **entities):
		self.add("ieeg", raw, meta, **entities)

	def addBehavioralTable(self, df: pd.DataFrame, meta: dict = None, **entities):
		self.add("beh", df, meta, **entities)
