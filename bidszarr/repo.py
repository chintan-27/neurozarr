from pathlib import Path
from typing import Any, Iterator, cast

import icechunk
import mne  # type: ignore[import-untyped]  # mne ships no type information
import pandas as pd
import zarr

from .entities import Entities
from .items import Attrs, Recording, Reader, Table
from .read import RecordingView, TableView, views_in
from .storage import storage_from
from .writer import CodecConfig, Writer


def _open_writer(icechunk_repo: icechunk.Repository, codec: CodecConfig | None) -> Writer:
	return Writer(icechunk_repo.writable_session("main"), codec)


class Repo:
	"""A store of recordings, and the entry point to everything else.

	A store is a directory of Icechunk repositories — one per subject, so any
	single subject can be copied, shared or versioned without moving the whole
	study — plus a ``_dataset`` repository holding metadata that belongs to no
	single subject, such as the dataset description and participant field
	definitions.

	Fill a store either in bulk with :meth:`ingest`, or by hand with
	:meth:`create_subject` and the methods on :class:`Subject` and
	:class:`Visit`. Either way, call :meth:`save` to commit.

	Parameters
	----------
	target : str or pathlib.Path or icechunk.Storage
		Where the store lives: a local path, a URI (``s3://bucket/study``,
		``gs://``, ``az://``, ``r2://``, or ``memory://`` for a throwaway
		in-memory store), or a pre-built :class:`icechunk.Storage`.
	codec : CodecConfig, optional
		How sample data is packed and compressed. Defaults to int16 packing
		with zstd compression.
	**storage_options
		Passed through to the underlying icechunk storage constructor, e.g.
		``region="us-east-1"``, ``anonymous=True``, ``from_env=True``.

	Examples
	--------
	>>> repo = Repo("./study.zarr")
	>>> repo.ingest(BidsReader("./my_bids_dataset"))
	>>> repo.save("initial conversion")
	"""

	def __init__(self, target: "str | Path | icechunk.Storage", codec: CodecConfig | None = None,
				 **storage_options: Any):
		self.target = target
		self._codec = codec
		self._storage_options = storage_options
		self._repos: dict[str, icechunk.Repository] = {}  # opened lazily
		self._writers: dict[str, Writer] = {}  # opened lazily on first write
		self._new_subjects: set[str] = set()  # written into the _dataset index on save()

	def _icechunk_repo(self, sub_id: str) -> icechunk.Repository:
		if sub_id not in self._repos:
			storage = storage_from(self.target, sub_id, **self._storage_options)
			self._repos[sub_id] = icechunk.Repository.open_or_create(storage)
		return self._repos[sub_id]

	def _writer_for(self, sub_id: str) -> Writer:
		if sub_id not in self._writers:
			self._writers[sub_id] = _open_writer(self._icechunk_repo(sub_id), self._codec)
			if sub_id.startswith("sub-"):
				self._new_subjects.add(sub_id)
		return self._writers[sub_id]

	def _subject_index(self) -> list[str]:
		"""Subject ids recorded in the _dataset repo. Object stores can't be listed
		like a directory, so the store keeps its own index of which subjects exist."""
		try:
			return cast("list[str]", self.root_of("_dataset").attrs.asdict().get("subjects", []))
		except Exception:  # no _dataset repo yet, or nothing committed to it
			return []

	def create_subject(self, sub_id: str, attrs: dict[str, Any] | None = None) -> "Subject":
		"""Add a subject to the store.

		Parameters
		----------
		sub_id : str
			BIDS subject label, including the prefix, e.g. ``"sub-001"``.
		attrs : dict, optional
			Metadata about the participant, such as their ``participants.tsv``
			row.

		Returns
		-------
		Subject
			A handle for adding this subject's visits.
		"""
		if attrs:
			self._writer_for(sub_id).add_attrs(Attrs((), attrs))
		return Subject(self, sub_id)

	def ingest(self, reader: Reader, progress: Any = None, skip_existing: bool = False) -> None:
		"""Write everything a reader yields into the store.

		Items are written as they arrive, so memory use stays flat no matter how
		large the source is. Nothing is committed until :meth:`save` is called.

		Parameters
		----------
		reader : Reader
			Any object with a ``read()`` method yielding :class:`Recording`,
			:class:`Table` and :class:`Attrs` items — :class:`BidsReader`,
			:class:`ManifestReader`, or one of your own.
		progress : callable, optional
			Wraps the item stream to report progress, e.g. ``tqdm``.
		skip_existing : bool, default False
			Leave items already present in the store untouched, writing only
			what is new. Use it when re-running after more data arrives.

		See Also
		--------
		save : Commit what was ingested.
		bidszarr.parallel.convert_parallel : Ingest subjects concurrently.
		"""
		existing = self._existing_paths() if skip_existing else None
		items = reader.read()
		for item in progress(items) if progress else items:
			if isinstance(item, Attrs):
				self._route_attrs(item)
			elif existing is None or not self._already_stored(item, existing):
				self._writer_for(item.entities.sub).dispatch(item)

	def _existing_paths(self) -> set[str]:
		"""Every group path already in the store, as "sub-XXX/rest/of/path"."""
		paths: set[str] = set()
		for sub_id in self.subjects():
			try:
				root = self.root_of(sub_id)
			except Exception:  # nothing committed for this subject yet
				continue
			paths.update(f"{sub_id}/{path}" for path, _ in root.members(max_depth=None))
		return paths

	def _already_stored(self, item: Recording | Table, existing: set[str]) -> bool:
		group_path = "/".join((*item.prefix, *item.entities.path()))
		name = "data" if isinstance(item, Recording) else item.name
		return f"{group_path}/{name}" in existing

	def _route_attrs(self, item: Attrs) -> None:
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

	def save(self, message: str) -> None:
		"""Commit every subject repository touched since the last save.

		Each subject is committed independently; subjects with no changes are
		left alone.

		Parameters
		----------
		message : str
			Commit message, shown by :meth:`history`.
		"""
		if self._new_subjects:
			known = set(self._subject_index()) | self._new_subjects
			self._writer_for("_dataset").add_attrs(Attrs((), {"subjects": sorted(known)}))
			self._new_subjects.clear()
		for writer in self._writers.values():
			writer.save(message)

	# ---- reading ----------------------------------------------------------

	def root_of(self, sub_id: str, version: str | None = None) -> zarr.Group:
		"""Open one subject's tree for reading.

		Parameters
		----------
		sub_id : str
			Subject label, e.g. ``"sub-001"``, or ``"_dataset"`` for the
			dataset-level metadata repository.
		version : str, optional
			A tag or snapshot id to read the store as it was at that point.
			Default reads the current state.

		Returns
		-------
		zarr.Group
			Read-only root group for that subject.
		"""
		repo = self._icechunk_repo(sub_id)
		session = repo.readonly_session(tag=version) if version and version in repo.list_tags() \
			else repo.readonly_session(snapshot_id=version) if version \
			else repo.readonly_session("main")
		return zarr.open_group(store=session.store, mode="r")

	def subjects(self) -> list[str]:
		"""List the subjects in the store.

		Returns
		-------
		list of str
			Subject labels, sorted, e.g. ``["sub-001", "sub-002"]``.
		"""
		index = self._subject_index()
		if index:
			return sorted(index)
		if not isinstance(self.target, icechunk.Storage):
			base = Path(str(self.target))
			if base.exists():
				return sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("sub-"))
		return sorted(s for s in self._repos if s.startswith("sub-"))

	def subject(self, sub_id: str) -> "Subject":
		"""Open a subject for reading or for adding visits.

		Parameters
		----------
		sub_id : str
			Subject label, e.g. ``"sub-001"``.

		Returns
		-------
		Subject
		"""
		return Subject(self, sub_id)

	def find(self, datatype: str | None = None, sub: str | None = None,
			 **entities: Any) -> Iterator[RecordingView]:
		"""Search the whole store for recordings matching a set of BIDS entities.

		Because each subject is a separate repository, searching every subject
		opens a session per subject; pass ``sub`` to restrict the search when
		you already know where to look.

		Parameters
		----------
		datatype : str, optional
			Restrict to one datatype, e.g. ``"ieeg"``.
		sub : str, optional
			Restrict to one subject, e.g. ``"sub-001"``.
		**entities
			BIDS entities to match, e.g. ``task="Rest", run=1``. Values are
			compared as strings, so ``run=1`` and ``run="1"`` both match.

		Yields
		------
		RecordingView
			Each matching recording.

		Examples
		--------
		>>> for rec in repo.find(task="BrainSenseStream", acq="TD"):
		...     print(rec.path, rec.shape)
		"""
		for sub_id in ([sub] if sub else self.subjects()):
			for view in self.subject(sub_id).recordings():
				if datatype and f"/{datatype}/" not in f"/{view.path}/":
					continue
				if all(str(view.entities.get(k)) == str(v) for k, v in entities.items()):
					yield view

	# ---- versioning -------------------------------------------------------

	def history(self, sub_id: str | None = None) -> list[tuple[str, str, Any]]:
		"""Read the commit history of one subject's repository.

		Parameters
		----------
		sub_id : str, optional
			Which subject's history to read. Defaults to the first subject in
			the store.

		Returns
		-------
		list of tuple
			``(snapshot_id, message, written_at)`` per commit, newest first.
		"""
		sub_id = sub_id or (self.subjects() or ["_dataset"])[0]
		return [(s.id, s.message, s.written_at) for s in self._icechunk_repo(sub_id).ancestry(branch="main")]

	def tag(self, name: str) -> None:
		"""Name the store's current state so it can be read back later.

		A version of the dataset spans every subject's repository, so the tag is
		applied to each of them. Subjects already carrying the tag are skipped.

		Parameters
		----------
		name : str
			Tag name, e.g. ``"v1"``. Pass it as ``version=`` to
			:meth:`Subject.recordings` or :meth:`root_of` to read that state.
		"""
		for sub_id in self.subjects() + ["_dataset"]:
			repo = self._icechunk_repo(sub_id)
			if name in repo.list_tags():
				continue
			repo.create_tag(name, repo.lookup_branch("main"))

	def tags(self, sub_id: str | None = None) -> list[str]:
		"""List the tags on the store.

		Parameters
		----------
		sub_id : str, optional
			Which subject's repository to read tags from. Defaults to the first
			subject in the store.

		Returns
		-------
		list of str
			Tag names, sorted.
		"""
		sub_id = sub_id or (self.subjects() or ["_dataset"])[0]
		return sorted(self._icechunk_repo(sub_id).list_tags())


class Subject:
	"""One participant, backed by their own repository.

	Obtained from :meth:`Repo.subject` or :meth:`Repo.create_subject`. Add data
	with :meth:`add_visit`; read it back with :meth:`visits`,
	:meth:`recordings` and :meth:`tables`.

	Attributes
	----------
	sub_id : str
		This subject's BIDS label, e.g. ``"sub-001"``.
	"""

	def __init__(self, repo: Repo, sub_id: str):
		self._repo = repo
		self.sub_id = sub_id

	def add_visit(self, ses_id: str, attrs: dict[str, Any] | None = None) -> "Visit":
		"""Add a session to this subject — typically one clinic visit or upload day.

		Parameters
		----------
		ses_id : str
			BIDS session label, including the prefix, e.g. ``"ses-20220908"``.
		attrs : dict, optional
			Metadata about the session, such as the device used.

		Returns
		-------
		Visit
			A handle for adding this session's recordings and tables.
		"""
		if attrs:
			self._repo._writer_for(self.sub_id).add_attrs(Attrs((ses_id,), attrs))
		return Visit(self._repo, self.sub_id, ses_id)

	# ---- reading ----------------------------------------------------------

	def root(self, version: str | None = None) -> zarr.Group:
		"""Open this subject's tree directly, for cases the view API doesn't cover.

		Parameters
		----------
		version : str, optional
			A tag or snapshot id. Default reads the current state.

		Returns
		-------
		zarr.Group
		"""
		return self._repo.root_of(self.sub_id, version)

	@property
	def attrs(self) -> dict[str, Any]:
		"""This subject's metadata, typically their ``participants.tsv`` row."""
		return self.root().attrs.asdict()

	def visits(self, version: str | None = None) -> list[str]:
		"""List this subject's sessions.

		Parameters
		----------
		version : str, optional
			A tag or snapshot id. Default reads the current state.

		Returns
		-------
		list of str
			Session labels, sorted, e.g. ``["ses-20220908"]``.
		"""
		return sorted(name for name, node in self.root(version).members()
					  if isinstance(node, zarr.Group) and name.startswith("ses-"))

	def visit(self, ses_id: str) -> "Visit":
		"""Open one of this subject's sessions.

		Parameters
		----------
		ses_id : str
			Session label, e.g. ``"ses-20220908"``.

		Returns
		-------
		Visit
		"""
		return Visit(self._repo, self.sub_id, ses_id)

	def recordings(self, version: str | None = None) -> list[RecordingView]:
		"""Every recording belonging to this subject, across all their sessions.

		Parameters
		----------
		version : str, optional
			A tag or snapshot id. Default reads the current state.

		Returns
		-------
		list of RecordingView
			Includes derivatives, which carry ``derivatives/<pipeline>/`` in
			their path.
		"""
		return [v for v in views_in(self.root(version), self.sub_id) if isinstance(v, RecordingView)]

	def tables(self, version: str | None = None) -> list[TableView]:
		"""Every table belonging to this subject, across all their sessions.

		Tables stored alongside a recording — its channels and events — belong to
		that recording and are reached through :meth:`RecordingView.channels`
		and :meth:`RecordingView.events` rather than appearing here.

		Parameters
		----------
		version : str, optional
			A tag or snapshot id. Default reads the current state.

		Returns
		-------
		list of TableView
		"""
		return [v for v in views_in(self.root(version), self.sub_id) if isinstance(v, TableView)]


class Visit:
	"""One session of one subject.

	Obtained from :meth:`Subject.add_visit` or :meth:`Subject.visit`. Throughout
	this class, keyword arguments are BIDS entities (``task=``, ``run=``,
	``acq=``, …) and determine where data lands in the tree.

	Attributes
	----------
	sub_id : str
		The subject this session belongs to.
	ses_id : str
		This session's BIDS label.
	"""

	def __init__(self, repo: Repo, sub_id: str, ses_id: str):
		self._repo = repo
		self.sub_id = sub_id
		self.ses_id = ses_id

	def add(self, datatype: str, payload: "mne.io.BaseRaw | pd.DataFrame",
			meta: dict[str, Any] | None = None, prefix: tuple[str, ...] = (),
			**entities: Any) -> None:
		"""Store a recording or a table under any datatype.

		Parameters
		----------
		datatype : str
			BIDS datatype directory, e.g. ``"ieeg"``, ``"eeg"``, ``"beh"``.
		payload : mne.io.BaseRaw or pandas.DataFrame
			The data to store.
		meta : dict, optional
			Sidecar metadata to store alongside it.
		prefix : tuple of str, optional
			Extra path segments placed before the subject, e.g.
			``("derivatives", "my-pipeline")``.
		**entities
			BIDS entities, e.g. ``task="Rest", run=1``.

		Raises
		------
		TypeError
			If ``payload`` is neither an mne ``Raw`` nor a DataFrame.

		See Also
		--------
		add_recording : Shorthand for ``ieeg`` recordings.
		add_behavioral_table : Shorthand for ``beh`` tables.
		add_derivative : Store processed results under ``derivatives/``.
		"""
		e = Entities(self.sub_id, self.ses_id, datatype, entities)
		item: Recording | Table
		if isinstance(payload, mne.io.BaseRaw):
			item = Recording(e, payload, meta or {}, prefix)
		elif isinstance(payload, pd.DataFrame):
			item = Table(e, "table", payload, meta or {}, prefix)
		else:
			raise TypeError(f"unsupported recording/table payload type {type(payload)!r}")
		self._repo._writer_for(self.sub_id).dispatch(item)

	def add_recording(self, raw: "mne.io.BaseRaw", meta: dict[str, Any] | None = None,
					  **entities: Any) -> None:
		"""Store a recording under the ``ieeg`` datatype.

		Parameters
		----------
		raw : mne.io.BaseRaw
			The recording to store.
		meta : dict, optional
			Sidecar metadata. Sampling frequency and channel names are taken
			from ``raw`` itself when not given here.
		**entities
			BIDS entities, e.g. ``task="Rest", run=1``.
		"""
		self.add("ieeg", raw, meta, **entities)

	def add_behavioral_table(self, df: pd.DataFrame, meta: dict[str, Any] | None = None,
							 **entities: Any) -> None:
		"""Store a table under the ``beh`` datatype.

		Parameters
		----------
		df : pandas.DataFrame
			The table to store. Column dtypes are preserved on read-back.
		meta : dict, optional
			Sidecar metadata.
		**entities
			BIDS entities, e.g. ``task="TherapyHistory"``.
		"""
		self.add("beh", df, meta, **entities)

	def add_derivative(self, pipeline: str, payload: "mne.io.BaseRaw | pd.DataFrame",
					   datatype: str = "ieeg", meta: dict[str, Any] | None = None,
					   **entities: Any) -> None:
		"""Store a processed result under ``derivatives/<pipeline>/``.

		This keeps analysis outputs separate from raw data, the way BIDS does.
		Derivatives read back like any other data, through
		:meth:`Subject.recordings`, :meth:`Subject.tables` or :meth:`Repo.find`,
		with ``derivatives/<pipeline>/`` in their path.

		Parameters
		----------
		pipeline : str
			Name of the pipeline that produced this result, e.g.
			``"my-filter"``. Recorded as ``GeneratedBy`` in the metadata.
		payload : mne.io.BaseRaw or pandas.DataFrame
			The processed result.
		datatype : str, default "ieeg"
			BIDS datatype directory to file it under.
		meta : dict, optional
			Additional sidecar metadata.
		**entities
			BIDS entities, e.g. ``task="Rest", run=1``.

		Examples
		--------
		>>> filtered = raw.copy().filter(l_freq=1, h_freq=40)
		>>> visit.add_derivative("my-filter", filtered, task="Rest", run=1)
		"""
		meta = {"GeneratedBy": pipeline, **(meta or {})}
		self.add(datatype, payload, meta, prefix=("derivatives", pipeline), **entities)

	# ---- reading ----------------------------------------------------------

	def _group(self, version: str | None = None) -> zarr.Group:
		root = self._repo.root_of(self.sub_id, version)
		if self.ses_id not in root:
			raise KeyError(f"{self.sub_id} has no {self.ses_id} (have: {', '.join(root.keys())})")
		return cast(zarr.Group, root[self.ses_id])

	@property
	def attrs(self) -> dict[str, Any]:
		"""This session's metadata, such as its ``sessions.tsv`` row."""
		return self._group().attrs.asdict()

	def recordings(self, version: str | None = None) -> list[RecordingView]:
		"""Every recording in this session.

		Parameters
		----------
		version : str, optional
			A tag or snapshot id. Default reads the current state.

		Returns
		-------
		list of RecordingView
		"""
		base = f"{self.sub_id}/{self.ses_id}"
		return [v for v in views_in(self._group(version), base) if isinstance(v, RecordingView)]

	def tables(self, version: str | None = None) -> list[TableView]:
		"""Every table in this session.

		Parameters
		----------
		version : str, optional
			A tag or snapshot id. Default reads the current state.

		Returns
		-------
		list of TableView
		"""
		base = f"{self.sub_id}/{self.ses_id}"
		return [v for v in views_in(self._group(version), base) if isinstance(v, TableView)]

	def recording(self, **entities: Any) -> "RecordingView":
		"""Get the single recording in this session matching the given entities.

		Parameters
		----------
		**entities
			BIDS entities to match, e.g. ``task="Rest", acq="TD", run=1``.
			Values are compared as strings.

		Returns
		-------
		RecordingView

		Raises
		------
		KeyError
			If no recording matches, or if more than one does — in which case
			the message lists the candidates so you can narrow the entities.
		"""
		matches = [r for r in self.recordings()
				   if all(str(r.entities.get(k)) == str(v) for k, v in entities.items())]
		if not matches:
			raise KeyError(f"no recording in {self.sub_id}/{self.ses_id} matching {entities}")
		if len(matches) > 1:
			raise KeyError(f"{len(matches)} recordings match {entities}: {[m.path for m in matches]}")
		return matches[0]
