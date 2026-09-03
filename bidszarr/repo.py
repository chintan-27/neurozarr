from pathlib import Path

import icechunk
import mne
import pandas as pd
import zarr

from .entities import Entities
from .items import Attrs, Recording, Reader, Table
from .read import RecordingView, TableView, views_in
from .storage import storage_from
from .writer import CodecConfig, Writer


def _open_writer(icechunk_repo: icechunk.Repository, codec: CodecConfig) -> Writer:
	return Writer(icechunk_repo.writable_session("main"), codec)


class Repo:
	"""Top-level handle over a directory of per-subject Icechunk repositories
	(base_path/sub-XXX/...), one per subject so each can be stored/versioned/shipped
	independently, plus one base_path/_dataset/ repo for attrs that aren't about any
	single subject (dataset_description.json, participants.tsv field descriptions).
	Build it up either by ingesting a Reader (bulk import) or by hand via
	create_subject()/Subject.add_visit()/Visit.add*()."""

	def __init__(self, target, codec: CodecConfig = None, **storage_options):
		"""target: a local path, a URI (s3://bucket/study, gs://, az://, memory://),
		or an icechunk.Storage. Extra keyword options pass through to the icechunk
		storage constructor (region=, anonymous=, from_env=, ...)."""
		self.target = target
		self._codec = codec
		self._storage_options = storage_options
		self._repos = {}    # sub_id -> icechunk.Repository, opened lazily
		self._writers = {}  # sub_id -> Writer, opened lazily on first write

	def _icechunk_repo(self, sub_id: str) -> icechunk.Repository:
		if sub_id not in self._repos:
			storage = storage_from(self.target, sub_id, **self._storage_options)
			self._repos[sub_id] = icechunk.Repository.open_or_create(storage)
		return self._repos[sub_id]

	def _writer_for(self, sub_id: str) -> Writer:
		if sub_id not in self._writers:
			self._writers[sub_id] = _open_writer(self._icechunk_repo(sub_id), self._codec)
		return self._writers[sub_id]

	def create_subject(self, sub_id: str, attrs: dict = None) -> "Subject":
		if attrs:
			self._writer_for(sub_id).add_attrs(Attrs((), attrs))
		return Subject(self, sub_id)

	def ingest(self, reader: Reader, progress=None):
		"""Write everything a Reader yields. progress, if given, wraps the item
		stream (e.g. tqdm) -- items are written as they arrive either way."""
		items = reader.read()
		for item in progress(items) if progress else items:
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
		"""Commit every subject repo touched since the last save."""
		for writer in self._writers.values():
			writer.save(message)

	# ---- reading ----------------------------------------------------------

	def root_of(self, sub_id: str, version: str = None) -> zarr.Group:
		"""Read-only zarr root of one subject's repo, optionally at a tag/snapshot."""
		repo = self._icechunk_repo(sub_id)
		session = repo.readonly_session(tag=version) if version and version in repo.list_tags() \
			else repo.readonly_session(snapshot_id=version) if version \
			else repo.readonly_session("main")
		return zarr.open_group(store=session.store, mode="r")

	def subjects(self) -> list:
		"""Subject ids present in the store (every sub-* repo under the target)."""
		if isinstance(self.target, icechunk.Storage):
			return sorted(self._repos)  # opaque storage: only what this handle has opened
		base = Path(str(self.target))
		if not base.exists():
			return sorted(s for s in self._repos if s.startswith("sub-"))
		return sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("sub-"))

	def subject(self, sub_id: str) -> "Subject":
		return Subject(self, sub_id)

	def find(self, datatype: str = None, sub: str = None, **entities):
		"""Every recording across the store matching the given entities, e.g.
		find(task="BrainSenseStream", acq="TD"). Fans out over one repo per
		subject, so it costs a session open per subject."""
		for sub_id in ([sub] if sub else self.subjects()):
			for view in self.subject(sub_id).recordings():
				if datatype and f"/{datatype}/" not in f"/{view.path}/":
					continue
				if all(str(view.entities.get(k)) == str(v) for k, v in entities.items()):
					yield view

	# ---- versioning -------------------------------------------------------

	def history(self, sub_id: str = None) -> list:
		"""Commit history as (snapshot_id, message, written_at), newest first.
		Defaults to the first subject in the store; pass sub_id for a specific one."""
		sub_id = sub_id or (self.subjects() or ["_dataset"])[0]
		return [(s.id, s.message, s.written_at) for s in self._icechunk_repo(sub_id).ancestry(branch="main")]

	def tag(self, name: str):
		"""Tag the current state of every subject repo. A dataset version spans N
		repos (one per subject), so the same tag name is applied to each."""
		for sub_id in self.subjects() + ["_dataset"]:
			repo = self._icechunk_repo(sub_id)
			if name in repo.list_tags():
				continue
			repo.create_tag(name, repo.lookup_branch("main"))

	def tags(self, sub_id: str = None) -> list:
		sub_id = sub_id or (self.subjects() or ["_dataset"])[0]
		return sorted(self._icechunk_repo(sub_id).list_tags())


class Subject:
	def __init__(self, repo: Repo, sub_id: str):
		self._repo = repo
		self.sub_id = sub_id

	def add_visit(self, ses_id: str, attrs: dict = None) -> "Visit":
		if attrs:
			self._repo._writer_for(self.sub_id).add_attrs(Attrs((ses_id,), attrs))
		return Visit(self._repo, self.sub_id, ses_id)

	# ---- reading ----------------------------------------------------------

	def root(self, version: str = None) -> zarr.Group:
		return self._repo.root_of(self.sub_id, version)

	@property
	def attrs(self) -> dict:
		"""This subject's own attributes (its participants.tsv row, typically)."""
		return self.root().attrs.asdict()

	def visits(self, version: str = None) -> list:
		return sorted(name for name, node in self.root(version).members()
					  if isinstance(node, zarr.Group) and name.startswith("ses-"))

	def visit(self, ses_id: str) -> "Visit":
		return Visit(self._repo, self.sub_id, ses_id)

	def recordings(self, version: str = None) -> list:
		return [v for v in views_in(self.root(version), self.sub_id) if isinstance(v, RecordingView)]

	def tables(self, version: str = None) -> list:
		return [v for v in views_in(self.root(version), self.sub_id) if isinstance(v, TableView)]


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

	# ---- reading ----------------------------------------------------------

	def _group(self, version: str = None) -> zarr.Group:
		root = self._repo.root_of(self.sub_id, version)
		if self.ses_id not in root:
			raise KeyError(f"{self.sub_id} has no {self.ses_id} (have: {', '.join(root.keys())})")
		return root[self.ses_id]

	@property
	def attrs(self) -> dict:
		return self._group().attrs.asdict()

	def recordings(self, version: str = None) -> list:
		base = f"{self.sub_id}/{self.ses_id}"
		return [v for v in views_in(self._group(version), base) if isinstance(v, RecordingView)]

	def tables(self, version: str = None) -> list:
		base = f"{self.sub_id}/{self.ses_id}"
		return [v for v in views_in(self._group(version), base) if isinstance(v, TableView)]

	def recording(self, **entities) -> "RecordingView":
		"""The one recording in this visit matching the given entities, e.g.
		recording(task="BrainSenseStream", acq="TD", run=1)."""
		matches = [r for r in self.recordings()
				   if all(str(r.entities.get(k)) == str(v) for k, v in entities.items())]
		if not matches:
			raise KeyError(f"no recording in {self.sub_id}/{self.ses_id} matching {entities}")
		if len(matches) > 1:
			raise KeyError(f"{len(matches)} recordings match {entities}: {[m.path for m in matches]}")
		return matches[0]
