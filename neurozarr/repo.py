from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Iterator, Literal, cast
import datetime
import logging
import re
import shutil
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor

import icechunk
import mne  # type: ignore[import-untyped]  # mne ships no type information
import pandas as pd
import zarr

from .entities import Entities
from .errors import MergeConflictError, SchemaVersionError, StoreIntegrityError, WriteConflictError
from .items import Array, Attrs, ExternalFile, Recording, Reader, Table
from .log import log_progress, logger, set_verbosity
from .read import ArrayView, ExternalFileView, RecordingView, TableView, WriteBack, views_in
from .schema import MANIFEST_ATTR, SCHEMA_VERSION, StoreManifest, utc_now
from .storage import StorageTarget, storage_from
from . import state, util
from .codecs import CodecPlan
from .writer import CodecConfig, ExistingPolicy, Writer, _clashes, _copy_node, copy_chunks


# "Nothing has been committed here yet" is signalled by four different exception
# types across icechunk and zarr. Every path that probes the _dataset repo must
# accept all of them: two paths disagreeing about this set is what made a freshly
# created store look like a legacy one.
_NO_COMMITTED_ROOT = (
	icechunk.RefNotFoundError,
	icechunk.SnapshotNotFoundError,
	icechunk.RepositoryNotFoundError,
	zarr.errors.GroupNotFoundError,
)


def _reader_for_source(source: Any, name: str | None, options: dict[str, Any]) -> Reader:
	"""The reader for a path source, picked as the ``neurozarr add`` command does."""
	from .readers import SmartReader, reader_for
	if name == "smart" or isinstance(source, (list, tuple)):
		return SmartReader(list(source) if isinstance(source, (list, tuple)) else source, **options)
	return reader_for(source, name, **options)


def _subject_of(item: Any) -> str | None:
	"""The subject an item belongs to, or None for dataset-level metadata."""
	if isinstance(item, Attrs):
		return next((p for p in item.path if p.startswith("sub-")), None)
	return item.entities.sub


def _ingest_subject(job: tuple[Any, ...]) -> tuple[str, Any]:
	"""Runs in a worker process: the ordinary serial ingest of one subject, into a
	fork of that subject's session handed over by the parent."""
	target, storage_options, codec, reader, sub, policy, forked, log_level, plan, branch = job
	if log_level < logging.WARNING:  # spawn starts with no logging setup; mirror the parent's
		set_verbosity(log_level)
	repo = Repo.open(target, mode="a", codec=codec, branch=branch, **storage_options)
	# only this subject's fork: never what the parent staged in the shared index
	repo._writers = {sub: Writer(forked, codec, sub)}  # written to only; the parent stages and commits it
	repo._removed_subjects = set()
	repo._batch = 1  # the parent writes the index, once, after every worker is back
	repo._plan = plan  # the parent's plan: probed once, never re-learned per worker
	items = (item for item in reader.read() if _subject_of(item) is not None)  # dataset-level: the parent's
	repo._ingest_items(items, None, policy)
	if set(repo._writers) != {sub}:
		raise ValueError(f"{type(reader).__name__}.for_subject({sub!r}) yielded items of other subjects: "
						 f"{sorted(set(repo._writers) - {sub})}")
	return sub, forked


def _open_writer(icechunk_repo: icechunk.Repository, codec: CodecConfig | None, label: str = "") -> Writer:
	"""A writer committing straight to ``main`` -- only for creating and migrating a store."""
	return Writer(icechunk_repo.writable_session("main"), codec, label)


class Repo:
	"""A store of recordings, and the entry point to everything else.

	A store is a directory of Icechunk repositories — one per subject, so any
	single subject can be copied, shared or versioned without moving the whole
	study — plus a ``_dataset`` repository holding metadata that belongs to no
	single subject, such as the dataset description and participant field
	definitions.

	It works like git: stage changes -- in bulk from a source with :meth:`add`, or
	item by item with :meth:`create_subject` and the methods on :class:`Subject` and
	:class:`Visit` -- then record them with :meth:`commit`. Staged changes persist
	between processes in a local index shared with the ``neurozarr`` command, on
	the current branch (HEAD), which is also shared.

	Parameters
	----------
	target : str or pathlib.Path or callable
		Where the store lives: a local path, a URI (``s3://bucket/study``,
		``gs://``, ``az://``, ``r2://``, or ``memory://`` for a throwaway
		in-memory store), or a callable returning independent storage for each
		logical repository.
	codec : CodecConfig, optional
		How sample data is packed and compressed. Defaults to int16 packing
		with zstd compression.
	verbose : bool, default False
		Log one line per item as it's written -- what it is, its size, live +
		peak memory actually referenced by real Python objects (not raw
		process footprint), and a running total of bytes written so far --
		plus one line per commit. Equivalent to
		``neurozarr.set_verbosity(logging.INFO)``; the package stays quiet by
		default, as a library should.
	**storage_options
		Passed through to the underlying icechunk storage constructor, e.g.
		``region="us-east-1"``, ``anonymous=True``, ``from_env=True``.

	Examples
	--------
	>>> repo = Repo.init("./study.zarr")
	>>> repo.add("./my_bids_dataset")
	>>> repo.commit("initial conversion")
	"""

	def __init__(self, target: StorageTarget, codec: CodecConfig | None = None,
				 **storage_options: Any):
		warnings.warn(
			"Repo(target) uses legacy open-or-create behavior; prefer Repo.init(target) or Repo.open(target)",
			DeprecationWarning, stacklevel=2,
		)
		self._init(target, codec, "legacy", storage_options)

	def _init(self, target: StorageTarget, codec: CodecConfig | None,
			  mode: Literal["legacy", "r", "a", "x"], storage_options: dict[str, Any]) -> None:
		target, storage_options = state.resolve(target, storage_options)  # a remote name -> its URL
		self.target = target
		self._branch = state.DEFAULT_BRANCH
		self._local: state.LocalState | None = None  # HEAD + index, once the dataset id is known
		self._batch = 0  # >0 while add() runs: stage once at the end, not per item
		self._codec = codec
		self._storage_options = storage_options
		self._mode = mode
		self._repos: dict[str, icechunk.Repository] = {}  # opened lazily
		self._writers: dict[str, Writer] = {}  # opened lazily on first write
		self._base_manifest: StoreManifest | None = None
		# Read-side caches: a readonly_session("main") pins whatever snapshot
		# "main" resolved to at open time, so a cached zarr.Group never goes
		# stale on its own -- only a new commit (from this Repo or another)
		# can outdate it, which is exactly when _invalidate_read_cache() is
		# called. Conflict-detection during commit() must see the true current
		# state, so it bypasses these via _fetch_manifest/_fetch_dataset_root.
		self._manifest_cache: dict[str | None, StoreManifest | None] = {}
		self._root_cache: dict[tuple[str, str | None], zarr.Group] = {}
		self._removed_subjects: set[str] = set()
		self._failed = False
		self._plan: CodecPlan | None = None  # the reader's codec plan, during an ingest only

	@classmethod
	def init(cls, target: StorageTarget, codec: CodecConfig | None = None, verbose: bool = False,
			 **storage_options: Any) -> "Repo":
		"""Create a new store (``git init``), failing if one already exists there.

		``target`` may also be the name of a saved remote (``neurozarr remote add``).
		"""
		if verbose:
			set_verbosity(logging.INFO)
		self = cls.__new__(cls)
		self._init(target, codec, "x", storage_options)
		if self._repo_exists("_dataset"):
			raise FileExistsError(f"a neurozarr store already exists at {self.target}")
		log_progress("Creating store", str(self.target))
		self._initialize_dataset("initialize neurozarr store")
		self._mode = "a"
		self._attach_local_state(None)
		return self

	@classmethod
	def open(cls, target: StorageTarget, mode: Literal["r", "a"] = "r",
			 codec: CodecConfig | None = None, branch: str | None = None, verbose: bool = False,
			 **storage_options: Any) -> "Repo":
		"""Open an existing store without accidentally creating an empty one.

		``target`` may be the name of a saved remote. ``branch`` defaults to HEAD, the
		branch last switched to (``main`` at first). In ``mode="a"``, changes staged
		earlier -- by this API or the ``neurozarr`` command -- are picked up again.
		"""
		if verbose:
			set_verbosity(logging.INFO)
		self = cls.__new__(cls)
		self._init(target, codec, mode, storage_options)
		if not self._repo_exists("_dataset"):
			raise FileNotFoundError(f"no neurozarr store exists at {self.target}")
		manifest = self._manifest()
		if manifest is None and mode == "a":
			raise SchemaVersionError("legacy store is read-only until `neurozarr migrate` is run")
		self._attach_local_state(branch)
		log_progress("Opened store", str(self.target))
		return self

	@property
	def head(self) -> str:
		"""The current branch (git's HEAD): where reads come from and commits go."""
		return self._branch

	def _attach_local_state(self, branch: str | None) -> None:
		"""Pick up HEAD and, when writable, the staged changes from the local index."""
		manifest = self._manifest()
		if manifest is None:  # legacy 0.1 store: no dataset id, no branches
			return
		self._local = state.LocalState(self.target, manifest.dataset_id)
		wanted = branch or self._local.head
		if wanted != self._branch:
			if wanted not in self._icechunk_repo("_dataset").list_branches():
				raise ValueError(f"no such branch: {wanted!r}")
			self._branch = wanted
			self._invalidate_read_cache()
		if self._mode != "a":
			return
		for repo_id, (forked, base) in self._local.staged(self._branch).items():
			self._writers[repo_id] = Writer(forked, self._codec, repo_id, repo=self._icechunk_repo(repo_id),
											branch=self._branch, base=base)
		self._removed_subjects = self._local.removed(self._branch)

	def _repo_exists(self, sub_id: str) -> bool:
		return icechunk.Repository.exists(storage_from(self.target, sub_id, **self._storage_options))

	def _icechunk_repo(self, sub_id: str) -> icechunk.Repository:
		if sub_id not in self._repos:
			storage = storage_from(self.target, sub_id, **self._storage_options)
			if icechunk.Repository.exists(storage):
				log_progress("Opening repository", sub_id)
				self._repos[sub_id] = icechunk.Repository.open(storage)
			elif self._mode == "r":
				raise StoreIntegrityError(f"store manifest references missing repository {sub_id!r}")
			else:
				log_progress("Creating repository", sub_id)
				self._repos[sub_id] = icechunk.Repository.create(storage)
		return self._repos[sub_id]

	def _writer_for(self, sub_id: str) -> Writer:
		if self._mode == "r":
			raise PermissionError("repository was opened read-only")
		if self._failed:
			raise RuntimeError("this transaction failed; call reset() before writing again")
		if self._base_manifest is None:
			self._base_manifest = self._manifest()
			if self._repo_exists("_dataset") and self._base_manifest is None:
				# icechunk.Repository.create() publishes `main` before anything is
				# written to it, so the branch existing does not mean a store does --
				# probe for a committed zarr root instead. A real 0.1 store has one
				# (holding its `subjects` attr); a store we just created has not.
				try:
					self._dataset_root()
				except _NO_COMMITTED_ROOT:
					pass
				else:
					raise SchemaVersionError("legacy store must be migrated before it can be modified")
		if sub_id not in self._writers:
			repo = self._icechunk_repo(sub_id)
			if self._branch not in repo.list_branches():
				# a subject repo gets the branch on its first write there, starting from the
				# snapshot this branch's manifest pins (or empty, for a new subject)
				pinned = (self._base_manifest.subject_snapshots.get(sub_id) if self._base_manifest else None)
				repo.create_branch(self._branch, pinned or repo.lookup_branch(state.DEFAULT_BRANCH))
			self._writers[sub_id] = Writer(repo.writable_session(self._branch).fork(), self._codec, sub_id,
										   repo=repo, branch=self._branch, base=repo.lookup_branch(self._branch))
		return self._writers[sub_id]

	def _stage(self, *repo_ids: str) -> None:
		"""Write staged changes to the local index, so they survive this process and
		``neurozarr status`` / ``commit`` see them. Skipped mid-add (once at its end)."""
		if self._batch or self._local is None:
			return
		for repo_id in repo_ids or tuple(self._writers):
			writer = self._writers.get(repo_id)
			if writer is not None and writer.session.has_uncommitted_changes:
				assert writer.base is not None
				self._local.stage(self._branch, repo_id, writer.session, writer.base)
		self._local.set_removed(self._branch, self._removed_subjects)

	def _dispatch(self, item: Recording | Table | ExternalFile | Array,
				  existing: ExistingPolicy) -> bool:
		"""Dispatch one item and make partial write failures impossible to commit."""
		try:
			writer = self._writer_for(item.entities.sub)
			writer.plan = self._plan  # None outside an ingest: the explicit API decides for itself
			written = writer.dispatch(item, existing)
			self._stage(item.entities.sub)
			return written
		except Exception:
			self._failed = True
			raise

	def _add_attrs(self, item: Attrs, sub_id: str) -> None:
		"""Add metadata while preserving the transaction failure invariant."""
		try:
			self._writer_for(sub_id).add_attrs(item)
			self._stage(sub_id)
		except Exception:
			self._failed = True
			raise

	def _delete(self, path: tuple[str, ...], sub_id: str) -> None:
		try:
			self._writer_for(sub_id).delete(path)
			self._stage(sub_id)
		except Exception:
			self._failed = True
			raise

	def _rename(self, path: tuple[str, ...], new_name: str, sub_id: str) -> None:
		try:
			self._writer_for(sub_id).rename(path, new_name)
			self._stage(sub_id)
		except Exception:
			self._failed = True
			raise

	def _initialize_dataset(self, message: str) -> str:
		manifest = StoreManifest(created_with=self._package_version())
		writer = self._writer_for("_dataset")
		writer.add_attrs(Attrs((), {MANIFEST_ATTR: manifest.as_dict(), "subjects": []}))
		snapshot = writer.save(message)
		assert snapshot is not None
		self._writers.clear()
		self._base_manifest = manifest
		self._invalidate_read_cache()
		return snapshot

	@staticmethod
	def _package_version() -> str:
		try:
			return package_version("neurozarr")
		except PackageNotFoundError:
			return "0.0.0.dev0"

	def _invalidate_read_cache(self) -> None:
		"""Clear memoized manifest/root reads -- call after any commit this
		Repo instance makes, so a read right after commit() doesn't see the
		pre-save state. Reads made *during* a save's own conflict detection
		must never go through the cache at all; those call _fetch_manifest/
		_fetch_dataset_root directly instead of clearing around them."""
		self._manifest_cache.clear()
		self._root_cache.clear()

	def _fetch_dataset_root(self, version: str | None = None) -> zarr.Group:
		repo = self._icechunk_repo("_dataset")
		if version is not None:
			session = repo.readonly_session(tag=version) if version in repo.list_tags() \
				else repo.readonly_session(snapshot_id=version)
		else:
			session = repo.readonly_session(self._branch)
		return zarr.open_group(store=session.store, mode="r")

	def _dataset_root(self, version: str | None = None) -> zarr.Group:
		key = ("_dataset", version)
		if key not in self._root_cache:
			self._root_cache[key] = self._fetch_dataset_root(version)
		return self._root_cache[key]

	def _fetch_manifest(self, version: str | None = None) -> StoreManifest | None:
		if not self._repo_exists("_dataset"):
			return None
		try:
			value = self._fetch_dataset_root(version).attrs.asdict().get(MANIFEST_ATTR)
		except _NO_COMMITTED_ROOT:
			return None
		if value is None:
			return None
		try:
			manifest = StoreManifest.from_dict(cast("dict[str, Any]", value))
		except (TypeError, ValueError) as exc:
			raise StoreIntegrityError(f"invalid dataset manifest: {exc}") from exc
		if manifest.schema_version > SCHEMA_VERSION:
			raise SchemaVersionError(
				f"store schema {manifest.schema_version} is newer than supported schema {SCHEMA_VERSION}"
			)
		if manifest.schema_version < SCHEMA_VERSION:
			raise SchemaVersionError(
				f"store schema {manifest.schema_version} needs migration to schema {SCHEMA_VERSION}"
			)
		return manifest

	def _manifest(self, version: str | None = None) -> StoreManifest | None:
		if version not in self._manifest_cache:
			self._manifest_cache[version] = self._fetch_manifest(version)
		return self._manifest_cache[version]

	def _subject_index(self) -> list[str]:
		"""Subject ids recorded in the _dataset repo. Object stores can't be listed
		like a directory, so the store keeps its own index of which subjects exist."""
		manifest = self._manifest()
		if manifest is not None:
			return sorted(manifest.subject_snapshots)
		try:
			return cast("list[str]", self.root_of("_dataset").attrs.asdict().get("subjects", []))
		except (StoreIntegrityError, icechunk.IcechunkError):  # no committed dataset repo yet
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
		Entities(sub_id)
		self._add_attrs(Attrs((), attrs or {}), sub_id)
		return Subject(self, sub_id)

	def set_attrs(self, attrs: dict[str, Any]) -> None:
		"""Add or update dataset-wide metadata after creation.

		Merges into whatever attrs already exist rather than replacing them.
		Not committed until :meth:`commit`.

		Parameters
		----------
		attrs : dict
			Metadata to merge into the dataset's own attrs.
		"""
		self._add_attrs(Attrs((), attrs), "_dataset")

	def delete_subject(self, sub_id: str) -> None:
		"""Remove a subject from the dataset's published index.

		This does not erase the subject's own repository or its version
		history -- only schema v2's dataset-wide record of which subjects are
		published. A subject id is baked into its repository's storage
		location, so there is no cheap rename for a whole subject; to move one
		under a new id, read its data back out and re-ingest it, then delete
		the old id here.

		Not committed until :meth:`commit`.

		Parameters
		----------
		sub_id : str
			Subject to remove, e.g. ``"sub-001"``.

		Raises
		------
		KeyError
			If no such subject is currently published.
		"""
		if sub_id not in self._subject_index():
			raise KeyError(f"no such subject: {sub_id}")
		self._writers.pop(sub_id, None)
		self._repos.pop(sub_id, None)
		self._removed_subjects.add(sub_id)
		if self._local is not None:
			self._local.unstage(self._branch, sub_id)
		self._stage()

	def add(self, source: "Reader | str | Path | list[str | Path]", *, reader: str | None = None,
			progress: Any = None, existing: ExistingPolicy | str = ExistingPolicy.ERROR,
			workers: int | None = 1, **reader_options: Any) -> None:
		"""Stage everything a source holds (``git add``); :meth:`commit` records it.

		Items are written as they arrive, so memory use stays flat no matter how
		large the source is; the staged changes go to the local index once, at the
		end, where :meth:`status` and the ``neurozarr`` command see them too.

		Codecs come from the reader: if it has ``codec_plan()``, it probes a few of
		its files once and the plan decides every array's codec (recorded in the
		dataset metadata as ``codec_plan``); otherwise zarr's default.
		``CodecConfig(compressors=...)`` on this Repo overrides the plan, with a
		warning.

		Parameters
		----------
		source : str, pathlib.Path, list of them, or Reader
			A folder or manifest table, with its reader picked the way the CLI does
			(a folder -> :class:`BidsReader`, a ``.csv``/``.tsv``/``.json`` ->
			:class:`ManifestReader`; ``reader="smart"`` for :class:`SmartReader`,
			which also takes a list of folders) -- or any reader object, built-in or
			your own: anything with ``read()`` yielding items.
		reader : str, optional
			Name of the reader to use for a path source (``"bids"``, ``"manifest"``,
			``"smart"``, or an installed plugin).
		**reader_options
			Passed to that reader, e.g. ``optimize="size"``, ``probe_level=3``.
		progress : callable, optional
			Wraps the item stream to report progress, e.g. ``tqdm``.
		existing : {"error", "skip", "replace"}, default "error"
			How to handle an item already present at the same entity path.
		workers : int or None, default 1
			Ingest this many subjects at once, each in its own process; ``None``
			means one per CPU. Needs a reader with ``subjects()`` and
			``for_subject()`` (the built-in readers have both) and a store other
			processes can reach (a path or object-store URI, not ``memory://``);
			otherwise it ingests serially. ``progress`` then counts subjects.

		See Also
		--------
		commit : Record what was staged.
		status : What is staged.
		"""
		policy = ExistingPolicy(existing)
		if hasattr(source, "read"):
			if reader or reader_options:
				raise TypeError("reader= and reader options apply to a path source, not a reader object")
			source_reader = cast(Reader, source)
		else:
			source_reader = _reader_for_source(source, reader, reader_options)
		self._batch += 1
		try:
			self._plan = self._codec_plan(source_reader)
			subjects = self._parallel_subjects(source_reader, workers)
			if subjects is None:
				self._ingest_items(source_reader.read(), progress, policy)
				return
			# Dataset-level items once, here; then each subject in a worker process.
			self._ingest_items((item for item in cast(Any, source_reader).for_subject(None).read()
								if _subject_of(item) is None), None, policy)
			self._ingest_parallel(source_reader, subjects, workers, progress, policy)
		finally:
			self._plan = None
			self._batch -= 1
			if not self._failed:
				self._stage()

	def _codec_plan(self, reader: Reader) -> CodecPlan | None:
		"""The reader's codec plan, recorded in the dataset metadata -- or None when
		CodecConfig.compressors pins every codec (or the reader has no plan)."""
		name = type(reader).__name__
		pinned = self._codec.compressors if self._codec is not None else None
		if pinned is not None:
			if hasattr(reader, "codec_plan"):
				warnings.warn(f"CodecConfig.compressors overrides {name}'s codec plan; leave it unset to let "
							  "the reader choose codecs per kind of data", UserWarning, stacklevel=3)
			record: dict[str, Any] = {"pinned_by": "CodecConfig", "compressors": [c.to_dict() for c in pinned]}
			plan = None
		elif hasattr(reader, "codec_plan"):
			plan = cast(Any, reader).codec_plan(self._codec or CodecConfig())
			record = plan.as_dict()
		else:
			return None
		self._route_attrs(Attrs((), {"codec_plan": {"reader": name, "planned_at": utc_now(), **record}}))
		return plan

	def _ingest_items(self, items: Iterator[Any], progress: Any, policy: ExistingPolicy) -> None:
		try:
			for item in progress(items) if progress else items:
				if isinstance(item, Attrs):
					self._route_attrs(item)
				else:
					self._dispatch(item, policy)
		except Exception:
			self._failed = True
			raise

	def _parallel_subjects(self, reader: Reader, workers: int | None) -> list[str] | None:
		"""The subjects to fan out over, or None to ingest serially."""
		if workers is not None and workers <= 1:
			return None
		if not (hasattr(reader, "subjects") and hasattr(reader, "for_subject")):
			logger.warning("%s has no subjects()/for_subject(); ingesting serially", type(reader).__name__)
			return None
		if not isinstance(self.target, (str, Path)) or str(self.target).startswith("memory://"):
			# a worker reopens the store by its address; an in-memory store or a raw
			# icechunk.Storage object is private to this process
			logger.warning("store %r is not reachable from other processes; ingesting serially", self.target)
			return None
		subjects = cast(Any, reader).subjects()
		return list(subjects) if subjects else None

	def _ingest_parallel(self, reader: Reader, subjects: list[str], workers: int | None,
						 progress: Any, policy: ExistingPolicy) -> None:
		"""Each worker writes one subject into that subject's staged fork, handed over and
		handed back; commit() then records it exactly as after a serial add -- one commit
		per subject, with commit()'s message."""
		try:
			jobs = [(self.target, self._storage_options, self._codec, cast(Any, reader).for_subject(sub), sub,
					 policy, self._writer_for(sub).session, logger.getEffectiveLevel(), self._plan, self._branch)
					for sub in subjects]
			# spawn, not fork: forking with icechunk's runtime already open deadlocks the children
			context = multiprocessing.get_context("spawn")
			with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
				results = pool.map(_ingest_subject, jobs)
				for sub, forked in progress(results) if progress else results:
					previous = self._writers[sub]
					self._writers[sub] = Writer(forked, self._codec, sub, repo=self._icechunk_repo(sub),
												branch=self._branch, base=previous.base)
		except Exception:
			self._failed = True
			raise

	@staticmethod
	def _route(path: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
		"""A path (an Attrs' path, or a view's path split on "/") may embed a
		sub-XXX segment anywhere in it (e.g. a derivatives path is
		("derivatives", name, sub-XXX, ...)). Whichever segment looks like a
		subject id decides which repo it belongs to; that segment is dropped
		since the target repo already is that subject. No sub-XXX segment at
		all means it's dataset-wide, routed to the _dataset repo unchanged."""
		for i, part in enumerate(path):
			if part.startswith("sub-"):
				Entities(part)
				return part, path[:i] + path[i + 1:]
		return "_dataset", path

	def _route_attrs(self, item: Attrs) -> None:
		sub_id, stripped = self._route(item.path)
		self._add_attrs(Attrs(stripped, item.attrs), sub_id)

	def _route_rename(self, path: tuple[str, ...], new_name: str) -> None:
		sub_id, stripped = self._route(path)
		self._rename(stripped, new_name, sub_id)

	def _route_delete(self, path: tuple[str, ...]) -> None:
		sub_id, stripped = self._route(path)
		self._delete(stripped, sub_id)

	def _write_back(self) -> WriteBack:
		"""Bound to every view built from this Repo, so a view's set_attrs()/
		rename()/delete() can reach back through it without read.py knowing
		what a Repo is."""
		return WriteBack(self._route_attrs, self._route_rename, self._route_delete)

	def commit(self, message: str) -> str | None:
		"""Record everything staged as one new dataset version (``git commit``).

		Each touched subject is committed, then the dataset version pinning all of
		them is published at once. If the branch moved since something was staged
		and touched the same data, nothing is committed (:class:`WriteConflictError`).

		Parameters
		----------
		message : str
			Commit message, shown by :meth:`log`.

		Returns
		-------
		str or None
			The new dataset version's id, or None if nothing was staged.
		"""
		if self._failed:
			raise RuntimeError("cannot commit a failed transaction; call reset() and retry")
		if not self._writers and not self._removed_subjects:
			return None

		touched: dict[str, str] = {}
		try:
			for sub_id, writer in list(self._writers.items()):
				if sub_id == "_dataset":
					continue
				snapshot = writer.save(message)
				if snapshot is not None:
					touched[sub_id] = snapshot
		except icechunk.ConflictError as exc:
			self._failed = True
			raise WriteConflictError("another writer changed the same subject; abort and retry") from exc
		except Exception:
			self._failed = True
			raise

		try:
			return self._publish_subject_snapshots(touched, message)
		except Exception:
			self._failed = True
			raise

	def _publish_subject_snapshots(self, touched: dict[str, str], message: str) -> str | None:
		"""Publish already-committed subject snapshots as one dataset version."""
		base = self._base_manifest or self._manifest() or StoreManifest(created_with=self._package_version())
		# Conflict detection needs the true current state, not a memoized read
		# from before this save started -- bypasses the read cache on purpose.
		current = self._fetch_manifest()
		if current is not None and current.subject_snapshots != base.subject_snapshots:
			for sub_id, snapshot in touched.items():
				base_snapshot = base.subject_snapshots.get(sub_id)
				current_snapshot = current.subject_snapshots.get(sub_id)
				if current_snapshot not in (None, base_snapshot, snapshot):
					self._failed = True
					raise WriteConflictError(f"another writer published a different snapshot for {sub_id}")
			base = current
		if self._removed_subjects:
			base = replace(
				base,
				subject_snapshots={k: v for k, v in base.subject_snapshots.items()
									if k not in self._removed_subjects},
				catalog={k: v for k, v in base.catalog.items() if k not in self._removed_subjects},
			)
		subject_snapshots = {**base.subject_snapshots, **touched}
		catalog = dict(base.catalog)
		for sub_id, snapshot in touched.items():
			catalog[sub_id] = self._catalog_entry(sub_id, snapshot)
		manifest = replace(
			base,
			updated_at=utc_now(),
			created_with=self._package_version(),
			subject_snapshots=subject_snapshots,
			catalog=catalog,
		)
		dataset_writer = self._writers.get("_dataset") or self._writer_for("_dataset")
		pending_attrs = dataset_writer.root.attrs.asdict()
		try:
			committed_attrs = self._fetch_dataset_root().attrs.asdict()
		except (*_NO_COMMITTED_ROOT, icechunk.IcechunkError, KeyError):
			committed_attrs = {}
		user_changes = {
			key: value for key, value in pending_attrs.items()
			if key not in (MANIFEST_ATTR, "subjects") and committed_attrs.get(key) != value
		}
		dataset_writer.add_attrs(Attrs((), {
			MANIFEST_ATTR: manifest.as_dict(),
			"subjects": sorted(subject_snapshots),
		}))
		try:
			dataset_snapshot = dataset_writer.save(message)
		except icechunk.ConflictError as exc:
			if user_changes:
				self._failed = True
				raise WriteConflictError(
					"another writer changed dataset metadata while this save was publishing; abort and retry"
				) from exc
			# Subject repos are independent. If only the manifest conflicted, merge
			# disjoint subject updates into a fresh dataset session and retry once.
			self._writers.pop("_dataset", None)
			self._repos.pop("_dataset", None)
			current = self._fetch_manifest() or base
			for sub_id, snapshot in touched.items():
				base_snapshot = base.subject_snapshots.get(sub_id)
				current_snapshot = current.subject_snapshots.get(sub_id)
				if current_snapshot not in (None, base_snapshot, snapshot):
					self._failed = True
					raise WriteConflictError(f"another writer published a different snapshot for {sub_id}") from exc
			if self._removed_subjects:
				current = replace(
					current,
					subject_snapshots={k: v for k, v in current.subject_snapshots.items()
										if k not in self._removed_subjects},
					catalog={k: v for k, v in current.catalog.items() if k not in self._removed_subjects},
				)
			merged_snapshots = {**current.subject_snapshots, **touched}
			merged_catalog = dict(current.catalog)
			for sub_id, subject_snapshot in touched.items():
				merged_catalog[sub_id] = self._catalog_entry(sub_id, subject_snapshot)
			manifest = replace(current, updated_at=utc_now(), created_with=self._package_version(),
				subject_snapshots=merged_snapshots, catalog=merged_catalog)
			dataset_writer = self._writer_for("_dataset")
			dataset_writer.add_attrs(Attrs((), {
				MANIFEST_ATTR: manifest.as_dict(), "subjects": sorted(merged_snapshots),
			}))
			try:
				dataset_snapshot = dataset_writer.save(message)
			except icechunk.ConflictError as retry_exc:
				self._failed = True
				raise WriteConflictError("dataset publication conflicted twice; abort and retry") from retry_exc

		self._writers.clear()
		self._removed_subjects.clear()
		self._base_manifest = manifest
		self._invalidate_read_cache()
		if self._local is not None:
			self._local.clear(self._branch)
		return dataset_snapshot

	@staticmethod
	def _catalog_is_current(manifest: StoreManifest, sub_id: str) -> bool:
		"""Whether the catalog can be trusted to describe a subject's published snapshot.

		The catalog is a cache, not a source of truth: it is only usable for the exact
		snapshot it was read from. Both :meth:`find` and ``neurozarr doctor`` ask this
		one question, so a reader and its integrity check cannot disagree.
		"""
		entry = manifest.catalog.get(sub_id)
		if not isinstance(entry, dict):
			return False
		return entry.get("snapshot") == manifest.subject_snapshots.get(sub_id)

	def _catalog_entry(self, sub_id: str, snapshot: str) -> dict[str, Any]:
		repo = self._icechunk_repo(sub_id)
		root = zarr.open_group(store=repo.readonly_session(snapshot_id=snapshot).store, mode="r")
		views = list(views_in(root, sub_id))
		return {
			# The snapshot this summary was read from. A snapshot id names immutable
			# content, so a stamped entry is a permanently true statement about that
			# version -- which is what lets readers tell a usable entry from one
			# describing a version the dataset no longer publishes.
			"snapshot": snapshot,
			"visits": sorted(name for name, node in root.members()
							 if isinstance(node, zarr.Group) and name.startswith("ses-")),
			"recordings": [
				{"path": view.path, "entities": view.entities}
				for view in views if isinstance(view, RecordingView)
			],
			"table_count": sum(isinstance(view, TableView) for view in views),
			"array_count": sum(isinstance(view, ArrayView) for view in views),
			"external_file_count": sum(isinstance(view, ExternalFileView) for view in views),
		}

	def reset(self, *subjects: str) -> None:
		"""Unstage (``git reset``): drop staged changes, for some subjects or all.

		Chunks already written for them stay in storage, unreferenced, until
		garbage-collected -- as git keeps unreachable objects.
		"""
		for repo_id in subjects or tuple(self._writers) + tuple(self._removed_subjects):
			self._writers.pop(repo_id, None)
			self._removed_subjects.discard(repo_id)
			if self._local is not None:
				self._local.unstage(self._branch, repo_id)
		if not subjects:
			if self._local is not None:
				self._local.clear(self._branch)
			self._base_manifest = None
			self._failed = False
		elif self._local is not None:
			self._local.set_removed(self._branch, self._removed_subjects)

	@contextmanager
	def transaction(self, message: str) -> Iterator["Repo"]:
		"""Commit changes on clean exit and discard them when the block fails."""
		try:
			yield self
			self.commit(message)
		except Exception:
			self.reset()
			raise

	def migrate(self, dry_run: bool = False) -> dict[str, Any]:
		"""Upgrade a readable 0.1 store to the current manifest schema in place."""
		if not self._repo_exists("_dataset"):
			raise FileNotFoundError(f"no neurozarr store exists at {self.target}")
		current = self._fetch_manifest()  # "already migrated?" must see true current state
		if current is not None:
			return {"changed": False, "schema_version": current.schema_version,
					"subjects": sorted(current.subject_snapshots)}
		if self._mode == "r" and not dry_run:
			raise PermissionError("repository was opened read-only")
		subjects = self._legacy_subjects()
		snapshots = {sub_id: self._icechunk_repo(sub_id).lookup_branch("main") for sub_id in subjects}
		result = {"changed": bool(not dry_run), "schema_version": SCHEMA_VERSION, "subjects": subjects}
		if dry_run:
			return result
		manifest = StoreManifest(
			created_with=self._package_version(),
			subject_snapshots=snapshots,
			catalog={sub_id: self._catalog_entry(sub_id, snapshot) for sub_id, snapshot in snapshots.items()},
		)
		writer = _open_writer(self._icechunk_repo("_dataset"), self._codec, "_dataset")
		writer.add_attrs(Attrs((), {MANIFEST_ATTR: manifest.as_dict(), "subjects": subjects}))
		writer.save(f"migrate neurozarr schema to {SCHEMA_VERSION}")
		self._writers.clear()
		self._base_manifest = manifest
		self._invalidate_read_cache()
		return result

	def _legacy_subjects(self) -> list[str]:
		try:
			root = self._dataset_root()
			indexed = cast("list[str]", root.attrs.asdict().get("subjects", []))
		except (icechunk.IcechunkError, KeyError):
			indexed = []
		if indexed:
			return sorted(indexed)
		if not isinstance(self.target, icechunk.Storage):
			base = Path(str(self.target))
			if base.exists():
				return sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("sub-"))
		return sorted(s for s in self._repos if s.startswith("sub-"))

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
		if sub_id == "_dataset":
			return self._dataset_root(version)
		key = (sub_id, version)
		if key not in self._root_cache:
			self._root_cache[key] = self._fetch_subject_root(sub_id, version)
		return self._root_cache[key]

	def _fetch_subject_root(self, sub_id: str, version: str | None) -> zarr.Group:
		Entities(sub_id)
		manifest = self._manifest(version)
		repo = self._icechunk_repo(sub_id)
		if manifest is not None:
			if sub_id not in manifest.subject_snapshots:
				raise KeyError(f"dataset version has no subject {sub_id!r}")
			session = repo.readonly_session(snapshot_id=manifest.subject_snapshots[sub_id])
		else:  # legacy 0.1 store: version names belonged to each subject repo
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
		manifest = self._manifest()
		if manifest is not None:
			return sorted(manifest.subject_snapshots)
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

	def describe(self) -> list[dict[str, Any]]:
		"""Summarize every subject in the store, as data rather than printed text.

		The same counts ``neurozarr info`` prints, returned as one dict per
		subject so a Python caller can build a table from them instead of
		parsing CLI output. Opens every subject's repository to count its
		items, so on a store with many subjects this costs one session open
		per subject -- pass ``sub`` to :meth:`find` instead when you already
		know which subject you care about.

		Returns
		-------
		list of dict
			One dict per subject, sorted by subject id, with keys
			``sub_id``, ``visits``, ``recordings``, ``tables``, ``arrays``,
			``external_files``.

		Examples
		--------
		>>> import pandas as pd
		>>> pd.DataFrame(repo.describe())
		"""
		overview = []
		for sub_id in self.subjects():
			subject = self.subject(sub_id)
			overview.append({
				"sub_id": sub_id,
				"visits": len(subject.visits()),
				"recordings": len(subject.recordings()),
				"tables": len(subject.tables()),
				"arrays": len(subject.arrays()),
				"external_files": len(subject.external_files()),
			})
		return overview

	def find(self, datatype: str | None = None, sub: str | None = None,
			 **entities: Any) -> Iterator[RecordingView]:
		"""Search the whole store for recordings matching a set of BIDS entities.

		Answered straight from each subject's catalog entry in the ``_dataset``
		manifest when it's current for that subject (the common case) -- no
		tree walk, one session opened only for a subject that actually has a
		match. A subject whose catalog is missing or stale (older stores,
		mid-migration) falls back to a full walk of just that subject.

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
		candidate_subjects = [sub] if sub else self.subjects()
		manifest = self._manifest()
		for sub_id in candidate_subjects:
			# The catalog is only ever trusted for a subject it can currently vouch
			# for (see _catalog_is_current): missing, unstamped, or stamped with a
			# snapshot this version no longer publishes all mean "unknown", so those
			# subjects take the full-walk fallback rather than risk silently
			# dropping a recording that's really there.
			if manifest is not None and self._catalog_is_current(manifest, sub_id):
				yield from self._find_in_catalog(manifest, sub_id, datatype, entities)
				continue
			for view in self.subject(sub_id).recordings():
				if datatype and f"/{datatype}/" not in f"/{view.path}/":
					continue
				if all(str(view.entities.get(k)) == str(v) for k, v in entities.items()):
					yield view

	def _find_in_catalog(self, manifest: StoreManifest, sub_id: str, datatype: str | None,
						  entities: dict[str, Any]) -> Iterator[RecordingView]:
		"""find()'s fast path: filter the catalog's own path+entities index and
		only then open the subject's root -- and only if something matched.

		No existence check against the tree before indexing into it: a current
		catalog entry (_catalog_is_current already gated this) was built by
		walking this exact snapshot id, and root_of resolves the subject to
		that same snapshot id, so every path it lists is guaranteed present.
		Checking anyway would cost one zarr existence lookup per matching
		recording -- for a query that matches most of a subject, that check
		alone was measured as the dominant cost after the walk itself was
		removed. If the invariant is ever wrong, indexing raises loudly
		instead of silently dropping a recording that is really there.
		"""
		root: zarr.Group | None = None
		write = self._write_back()
		for entry in manifest.catalog[sub_id].get("recordings", []):
			path = cast(str, entry["path"])
			if datatype and f"/{datatype}/" not in f"/{path}/":
				continue
			rec_entities = cast("dict[str, Any]", entry.get("entities", {}))
			if not all(str(rec_entities.get(k)) == str(v) for k, v in entities.items()):
				continue
			if root is None:
				root = self.root_of(sub_id)
			group_path = path.split("/", 1)[1] if "/" in path else ""
			yield RecordingView(cast("zarr.Group", root[group_path]), rec_entities, path, write)

	# ---- versioning -------------------------------------------------------

	def log(self, sub_id: str | None = None) -> list[tuple[str, str, Any]]:
		"""The commit history of the current branch (``git log``), newest first.

		Parameters
		----------
		sub_id : str, optional
			Which subject's internal history to read. By default, return global
			dataset-manifest history.

		Returns
		-------
		list of tuple
			``(snapshot_id, message, written_at)`` per commit, newest first.
		"""
		sub_id = sub_id or "_dataset"
		if sub_id != "_dataset":
			Entities(sub_id)
		repo = self._icechunk_repo(sub_id)
		branch = self._branch if self._branch in repo.list_branches() else state.DEFAULT_BRANCH
		# the last snapshot is icechunk's own empty root, made before any neurozarr
		# commit -- not part of the store's history, as git has nothing before its first commit
		return [(s.id, s.message, s.written_at) for s in repo.ancestry(branch=branch) if s.parent_id is not None]

	def tag(self, name: str, version: str | None = None) -> None:
		"""Name the store's current state so it can be read back later.

		Current-schema stores tag the dataset manifest. That manifest pins the
		exact snapshot of every subject, so one tag names a consistent global
		version without mutating the subject repositories.

		Parameters
		----------
		name : str
			Tag name, e.g. ``"v1"``. Pass it as ``version=`` to
			:meth:`Subject.recordings` or :meth:`root_of` to read that state.
		version : str, optional
			What to tag (see :meth:`resolve`); default the current branch tip.
		"""
		from .constraints import validate_segment

		validate_segment(name, "tag name")
		manifest = self._manifest()
		if manifest is not None:
			repo = self._icechunk_repo("_dataset")
			if name not in repo.list_tags():
				repo.create_tag(name, self.resolve(version))
			return
		for sub_id in self.subjects() + ["_dataset"]:
			repo = self._icechunk_repo(sub_id)
			if name not in repo.list_tags():
				repo.create_tag(name, repo.lookup_branch("main"))

	# ---- git-style inspection and staging -------------------------------

	def resolve(self, version: str | None = None) -> str:
		"""The dataset-version (snapshot) id a name refers to, like ``git rev-parse``.

		Accepts a branch, a tag, a full snapshot id or a unique prefix of one (at
		least 6 characters), each optionally followed by ``~N`` for N versions back.
		``None`` means the current branch.
		"""
		repo = self._icechunk_repo("_dataset")
		name, _, back = (version or self._branch).partition("~")
		steps = int(back or (1 if version and version.endswith("~") else 0))
		if name in repo.list_branches():
			snapshot = repo.lookup_branch(name)
		elif name in repo.list_tags():
			snapshot = repo.lookup_tag(name)
		else:
			if len(name) < 6:
				raise ValueError(f"unknown version {version!r} (a snapshot id prefix needs at least 6 characters)")
			matches = {info.id for branch in repo.list_branches() for info in repo.ancestry(branch=branch)
					   if info.id.startswith(name.upper())}
			if len(matches) != 1:
				raise ValueError(f"unknown version {version!r}" if not matches
								 else f"ambiguous version {version!r}: {', '.join(sorted(matches))}")
			snapshot = matches.pop()
		history = [info.id for info in repo.ancestry(snapshot_id=snapshot) if info.parent_id is not None]
		if steps >= len(history):
			raise ValueError(f"{version!r} goes back further than the history ({len(history) - 1} versions)")
		return history[steps]

	def status(self) -> dict[str, Any]:
		"""What is staged on the current branch (``git status``).

		Returns
		-------
		dict
			``branch``; ``staged``: subject (or ``_dataset``) -> lists of item paths
			``added`` / ``changed`` / ``removed`` / ``attrs`` / ``moved``; and
			``removed_subjects``.
		"""
		staged = {repo_id: _summarize(writer.session.status()) for repo_id, writer in sorted(self._writers.items())
				  if writer.session.has_uncommitted_changes}
		return {"branch": self._branch, "staged": staged, "removed_subjects": sorted(self._removed_subjects)}

	def show(self, version: str | None = None) -> dict[str, Any]:
		"""One dataset version (``git show``): its id, message, time, tags, the subjects
		it changed against its parent, and the codec plan in effect."""
		repo = self._icechunk_repo("_dataset")
		snapshot = self.resolve(version)
		history = list(repo.ancestry(snapshot_id=snapshot))
		info, parent = history[0], (history[1].id if len(history) > 1 else None)
		manifest = self._manifest(snapshot)
		before = self._manifest(parent) if parent else None
		pins = manifest.subject_snapshots if manifest else {}
		old = before.subject_snapshots if before else {}
		return {
			"version": info.id, "message": info.message, "written_at": info.written_at.isoformat(),
			"parent": parent, "tags": sorted(t for t in repo.list_tags() if repo.lookup_tag(t) == info.id),
			"branches": sorted(b for b in repo.list_branches() if repo.lookup_branch(b) == info.id),
			"subjects": len(pins),
			"subjects_added": sorted(pins.keys() - old.keys()),
			"subjects_removed": sorted(old.keys() - pins.keys()),
			"subjects_changed": sorted(k for k in pins.keys() & old.keys() if pins[k] != old[k]),
			"codec_plan": self._dataset_root(snapshot).attrs.asdict().get("codec_plan"),
		}

	def ls(self, path: str = "", version: str | None = None) -> list[dict[str, Any]]:
		"""Browse a version like a directory (``git ls-tree``): subjects at the top,
		then a subject's sessions, then a session's items."""
		snapshot = self.resolve(version) if version else None
		parts = [p for p in path.strip("/").split("/") if p]
		if not parts:
			manifest = self._manifest(snapshot)
			return [{"kind": "subject", "path": sub} for sub in sorted(manifest.subject_snapshots if manifest else [])]
		subject = self.subject(parts[0])
		if len(parts) == 1:
			return [{"kind": "session", "path": f"{parts[0]}/{ses}"} for ses in subject.visits(snapshot)]
		prefix = "/".join(parts)
		return [item for item in subject.visit(parts[1]).describe(snapshot) if item["path"].startswith(prefix)]

	def rm(self, path: str, recursive: bool = False) -> None:
		"""Stage removing a subject, a session or an item (``git rm``).

		Removing a subject or a session needs ``recursive=True``, as removing a
		directory does in git.
		"""
		parts = tuple(p for p in path.strip("/").split("/") if p)
		if not parts:
			raise ValueError("rm needs a path, e.g. sub-03 or sub-01/ses-2")
		if len(parts) <= 2 and not recursive:
			raise ValueError(f"{path!r} holds other items; pass recursive=True (CLI: rm -r) to remove it all")
		if len(parts) == 1:
			self.delete_subject(parts[0])
		else:
			self._route_delete(parts)

	def mv(self, path: str, new_name: str) -> None:
		"""Stage renaming a session or an item in place (``git mv``).

		``new_name`` is the new last path segment, or a full path in the same parent.
		Subjects can't be renamed: a subject's id is its repository's location.
		"""
		parts = tuple(p for p in path.strip("/").split("/") if p)
		target = new_name.strip("/").split("/")
		if len(target) > 1 and tuple(target[:-1]) != parts[:-1]:
			raise ValueError("mv renames in place; the new path must be in the same parent")
		if len(parts) <= 1:
			raise ValueError("a subject can't be renamed (its id is its repository's location); "
							 "add its data under the new id and rm the old one")
		self._route_rename(parts, target[-1])

	# ---- branches, diff, undo ---------------------------------------------

	def branches(self) -> list[str]:
		"""The store's branches (``git branch``), sorted."""
		return sorted(self._icechunk_repo("_dataset").list_branches())

	def branch(self, name: str, start: str | None = None) -> None:
		"""Create a branch at ``start`` (default: the current version) without switching
		to it. Cheap: only ``_dataset`` gets a pointer; each subject repository gets the
		branch on its first write there."""
		from .constraints import validate_segment
		validate_segment(name, "branch name")
		repo = self._icechunk_repo("_dataset")
		if name in repo.list_branches():
			raise ValueError(f"a branch named {name!r} already exists")
		repo.create_branch(name, self.resolve(start))

	def delete_branch(self, name: str, force: bool = False) -> None:
		"""Delete a branch (``git branch -d``); refused if it is the current branch, or
		holds versions no other branch has unless ``force`` (``-D``)."""
		repo = self._icechunk_repo("_dataset")
		if name not in repo.list_branches():
			raise ValueError(f"no such branch: {name!r}")
		if name == self._branch:
			raise ValueError(f"cannot delete the current branch {name!r}; switch to another first")
		tip = repo.lookup_branch(name)
		kept = {info.id for other in repo.list_branches() if other != name for info in repo.ancestry(branch=other)}
		if tip not in kept and not force:
			raise ValueError(f"branch {name!r} has versions no other branch has; delete it with force "
							 "(CLI: branch -D) to drop them")
		manifest = self._manifest(tip)
		for sub_id in (manifest.subject_snapshots if manifest else {}):
			sub_repo = self._icechunk_repo(sub_id)
			if name in sub_repo.list_branches():
				sub_repo.delete_branch(name)
		repo.delete_branch(name)
		if self._local is not None:
			self._local.clear(name)

	def switch(self, name: str, create: bool = False, start: str | None = None) -> None:
		"""Make ``name`` the current branch (``git switch``; ``create`` is ``switch -c``).

		Staged changes stay with the branch they were staged on and come back when you
		switch back to it.
		"""
		if create:
			self.branch(name, start)
		elif name not in self.branches():
			raise ValueError(f"no such branch: {name!r} (create it with switch -c)")
		self._branch = name
		self._writers.clear()
		self._removed_subjects = set()
		self._base_manifest = None
		self._failed = False
		self._invalidate_read_cache()
		if self._local is not None:
			self._local.head = name
			if self._mode == "a":
				self._attach_local_state(name)

	def diff(self, a: str | None = None, b: str | None = None, subject: str | None = None) -> dict[str, Any]:
		"""What changed (``git diff``): staged changes with no versions given; version
		``a`` against the current one; or ``a`` against ``b``.

		Returns
		-------
		dict
			``from``, ``to`` (version ids; ``to`` is ``"staged"`` for staged changes) and
			``subjects``: subject -> ``status`` (added / removed / changed) plus the item
			lists of :meth:`status` and ``chunks``, the number of chunks rewritten.
		"""
		if a is None:
			staged = {sub: {"status": "changed", **summary, "chunks": None}
					  for sub, summary in self.status()["staged"].items()}
			staged.update({sub: {"status": "removed"} for sub in self._removed_subjects})
			result: dict[str, Any] = {"from": self.resolve(), "to": "staged", "subjects": staged}
		else:
			old, new = self.resolve(a), self.resolve(b)
			before, after = self._manifest(old), self._manifest(new)
			old_pins = before.subject_snapshots if before else {}
			new_pins = after.subject_snapshots if after else {}
			subjects: dict[str, Any] = {}
			for sub_id in sorted(old_pins.keys() | new_pins.keys()):
				if old_pins.get(sub_id) == new_pins.get(sub_id):
					continue
				if sub_id not in new_pins:
					subjects[sub_id] = {"status": "removed"}
					continue
				repo = self._icechunk_repo(sub_id)
				start = old_pins.get(sub_id) or next(
					info.id for info in repo.ancestry(snapshot_id=new_pins[sub_id]) if info.parent_id is None)
				changes = repo.diff(from_snapshot_id=start, to_snapshot_id=new_pins[sub_id])
				subjects[sub_id] = {"status": "changed" if sub_id in old_pins else "added", **_summarize(changes),
									"chunks": sum(len(list(c)) for c in changes.updated_chunks.values())}
			result = {"from": old, "to": new, "subjects": subjects}  # same shape as above
		if subject is not None:
			result["subjects"] = {k: v for k, v in result["subjects"].items() if k == subject}
		return result

	def _repin(self, pins: dict[str, str | None], message: str, metadata: dict[str, Any] | None = None) -> str:
		"""Commit a new dataset version on the current branch with these subjects
		re-pinned (None: unpublished), moving each subject's branch to its pin so later
		writes build on it. No sample data is copied."""
		current = self._fetch_manifest()
		assert current is not None
		snapshots = dict(current.subject_snapshots)
		catalog = dict(current.catalog)
		for sub_id, pin in pins.items():
			if pin is None:
				snapshots.pop(sub_id, None)
				catalog.pop(sub_id, None)
				continue
			repo = self._icechunk_repo(sub_id)
			if self._branch in repo.list_branches():
				repo.reset_branch(self._branch, pin)
			else:
				repo.create_branch(self._branch, pin)
			snapshots[sub_id] = pin
			catalog[sub_id] = self._catalog_entry(sub_id, pin)
		manifest = replace(current, updated_at=utc_now(), created_with=self._package_version(),
						   subject_snapshots=snapshots, catalog=catalog)
		writer = self._writer_for("_dataset")
		writer.add_attrs(Attrs((), {MANIFEST_ATTR: manifest.as_dict(), "subjects": sorted(snapshots)}))
		version = writer.save(message, metadata)
		assert version is not None
		self._writers.pop("_dataset", None)
		self._base_manifest = manifest
		self._invalidate_read_cache()
		return version

	def revert(self, version: str, message: str | None = None) -> str:
		"""Undo one version with a new one (``git revert``): every subject it changed is
		put back as it was just before. History is kept and no data is copied.

		Raises
		------
		WriteConflictError
			If a later version changed one of those subjects again -- reverting would
			silently undo that too.
		"""
		if any(w.session.has_uncommitted_changes for w in self._writers.values()) or self._removed_subjects:
			raise ValueError("commit or reset what is staged before reverting")
		target = self.resolve(version)
		shown = self.show(target)
		if shown["parent"] is None:
			raise ValueError("cannot revert the store's first version")
		before, after, now = self._manifest(shown["parent"]), self._manifest(target), self._fetch_manifest()
		assert before is not None and after is not None and now is not None
		pins: dict[str, str | None] = {}
		for sub_id in shown["subjects_added"] + shown["subjects_changed"] + shown["subjects_removed"]:
			if now.subject_snapshots.get(sub_id) != after.subject_snapshots.get(sub_id):
				raise WriteConflictError(f"{sub_id} changed again after {version!r}; revert the later version "
										 "first, or rm/add it yourself")
			pins[sub_id] = before.subject_snapshots.get(sub_id)
		return self._repin(pins, message or f"Revert \"{shown['message']}\"\n\nThis reverts version {target}.")

	def reset_hard(self, version: str) -> None:
		"""Move the current branch to ``version`` and drop what is staged on it
		(``git reset --hard``). Versions after it stay reachable only from other
		branches or tags."""
		target = self.resolve(version)
		manifest = self._manifest(target)
		for sub_id, pin in (manifest.subject_snapshots if manifest else {}).items():
			repo = self._icechunk_repo(sub_id)
			if self._branch in repo.list_branches():
				repo.reset_branch(self._branch, pin)
		self.reset()
		self._icechunk_repo("_dataset").reset_branch(self._branch, target)
		self._invalidate_read_cache()

	# ---- merge, clone, gc -------------------------------------------------

	def _ancestors(self, version: str) -> list[str]:
		"""Every dataset version ``version`` builds on, nearest first: its parents and,
		for a merge, the merged-in branch's history too (kept in commit metadata, since
		an icechunk commit has a single parent)."""
		repo = self._icechunk_repo("_dataset")
		seen: list[str] = []
		queue = [version]
		while queue:
			current = queue.pop(0)
			if current in seen:
				continue
			seen.append(current)
			info = next(iter(repo.ancestry(snapshot_id=current)))
			queue += [p for p in (info.parent_id, *info.metadata.get(_MERGED, [])) if p]
		return seen

	def merge(self, branch: str, message: str | None = None, ours: tuple[str, ...] = (),
			  theirs: tuple[str, ...] = ()) -> str | None:
		"""Bring another branch's changes into the current one (``git merge``).

		Fast-forwards when the current branch has nothing of its own. Otherwise merges
		per subject against the versions' common ancestor: a subject changed on one side
		takes that side, at no copying cost; one changed on both is merged item by item
		when the changes don't overlap (copying the other side's changed chunks); and
		overlapping changes are a conflict, settled by naming the subject in ``ours`` or
		``theirs``. Nothing is written if any conflict remains.

		Returns
		-------
		str or None
			The new version, or None if already up to date.

		Raises
		------
		MergeConflictError
			With ``.conflicts``: subject -> the paths both sides changed.
		"""
		if any(w.session.has_uncommitted_changes for w in self._writers.values()) or self._removed_subjects:
			raise ValueError("commit or reset what is staged before merging")
		mine, other = self.resolve(), self.resolve(branch)
		mine_history = self._ancestors(mine)
		if other in mine_history:
			return None  # already up to date
		other_history = self._ancestors(other)
		if mine in other_history:  # fast-forward: just move the pointers
			manifest = self._manifest(other)
			for sub_id, pin in (manifest.subject_snapshots if manifest else {}).items():
				repo = self._icechunk_repo(sub_id)
				if self._branch in repo.list_branches():
					repo.reset_branch(self._branch, pin)
				else:
					repo.create_branch(self._branch, pin)
			self._icechunk_repo("_dataset").reset_branch(self._branch, other)
			self._invalidate_read_cache()
			return other
		base = next(v for v in other_history if v in set(mine_history))
		pins_base, pins_mine, pins_other = (
			(m.subject_snapshots if m else {}) for m in map(self._manifest, (base, mine, other)))
		pins: dict[str, str | None] = {}
		replays: dict[str, tuple[str, Any]] = {}
		conflicts: dict[str, list[str]] = {}
		for sub_id in sorted(pins_base.keys() | pins_mine.keys() | pins_other.keys()):
			was, mine_pin, other_pin = pins_base.get(sub_id), pins_mine.get(sub_id), pins_other.get(sub_id)
			if mine_pin == other_pin or other_pin == was or sub_id in ours:
				continue
			if mine_pin == was or sub_id in theirs:
				pins[sub_id] = other_pin
				continue
			if was is None or mine_pin is None or other_pin is None:
				conflicts[sub_id] = ["added or removed on one side and changed on the other"]
				continue
			repo = self._icechunk_repo(sub_id)
			their_changes = repo.diff(from_snapshot_id=was, to_snapshot_id=other_pin)
			clashes = _clashes(their_changes, repo.diff(from_snapshot_id=was, to_snapshot_id=mine_pin))
			if clashes:
				conflicts[sub_id] = clashes
			else:
				replays[sub_id] = (other_pin, their_changes)
		if conflicts:
			raise MergeConflictError(
				f"merging {branch!r} conflicts in {', '.join(conflicts)}; nothing was written. Keep one side "
				"per subject with ours=/theirs= (CLI: --ours/--theirs SUBJECT)", conflicts)
		title = message or f"Merge branch '{branch}' into {self._branch}"
		for sub_id, (other_pin, their_changes) in replays.items():
			writer = self._writer_for(sub_id)
			source = zarr.open_group(store=self._icechunk_repo(sub_id).readonly_session(snapshot_id=other_pin).store,
									 mode="r")
			_replay(source, writer.root, their_changes)
			pins[sub_id] = writer.save(title)
			self._writers.pop(sub_id, None)
		return self._repin(pins, title, {_MERGED: [other]})

	@classmethod
	def clone(cls, source: StorageTarget, dest: StorageTarget, version: str | None = None,
			  shallow: bool = False, **storage_options: Any) -> "Repo":
		"""Copy a store (``git clone``): everything, or with ``shallow`` just one version
		(default: the current one) as a fresh store with that single version.

		A full clone copies the store's files as they are, so it needs both ends to be
		local paths; a shallow clone copies the data itself and works between any two
		locations, local or cloud.
		"""
		if not shallow:
			if not (state.persistable(source) and state.persistable(dest)
					and urlparse(str(source)).scheme in ("", "file") and urlparse(str(dest)).scheme in ("", "file")):
				raise ValueError("a full clone needs both stores to be local paths; use shallow=True "
								 "(CLI: clone --depth 1) to copy one version anywhere")
			if Path(str(dest)).exists():
				raise FileExistsError(f"{dest} already exists")
			shutil.copytree(str(source), str(dest))
			return cls.open(dest, mode="a")
		origin = cls.open(source, **storage_options)
		snapshot = origin.resolve(version)
		manifest = origin._manifest(snapshot)
		copy = cls.init(dest, **storage_options)
		for sub_id in sorted(manifest.subject_snapshots if manifest else []):
			src_root = origin.root_of(sub_id, snapshot)
			writer = copy._writer_for(sub_id)
			util.set_attrs(writer.root, src_root.attrs.asdict())
			for name, node in src_root.members():
				_copy_node(node, writer.root, name)
		extra = {k: v for k, v in origin._dataset_root(snapshot).attrs.asdict().items()
				 if k not in (MANIFEST_ATTR, "subjects")}
		if extra:
			copy.set_attrs(extra)
		copy.commit(f"clone of {source} at {snapshot}")
		return copy

	def gc(self, older_than: datetime.timedelta = datetime.timedelta(days=14), dry_run: bool = True) -> dict[str, Any]:
		"""Delete stored data nothing can reach any more (``git gc``): versions dropped by
		``reset_hard`` or a deleted branch, and chunks of changes that were unstaged.

		Nothing reachable is touched: every version on a branch or tag is kept, with all
		subject data it pins. Only objects older than ``older_than`` go, so recent staged
		changes are safe; staged changes older than that are refused, not lost.

		Returns
		-------
		dict
			Per repository: objects and bytes deleted (or that would be, when ``dry_run``).
		"""
		cutoff = datetime.datetime.now(datetime.timezone.utc) - older_than
		if self._local is not None and self._local.oldest_staged() is not None:
			oldest = self._local.oldest_staged()
			if oldest is not None and oldest < cutoff:
				raise ValueError("there are staged changes older than the cutoff; commit or reset them first")
		dataset = self._icechunk_repo("_dataset")
		reachable = {v for ref in (*dataset.list_branches(), *dataset.list_tags())
					 for v in self._ancestors(dataset.lookup_branch(ref) if ref in dataset.list_branches()
											  else dataset.lookup_tag(ref))}
		pinned: dict[str, set[str]] = {}
		for version in reachable:
			manifest = self._manifest(version)
			for sub_id, pin in (manifest.subject_snapshots if manifest else {}).items():
				pinned.setdefault(sub_id, set()).add(pin)
		report: dict[str, Any] = {}
		for repo_id in ["_dataset", *sorted(pinned)]:
			repo = self._icechunk_repo(repo_id)
			if not dry_run and repo_id != "_dataset":
				# a pinned snapshot no subject branch reaches any more (after a revert or
				# reset) is still needed by a dataset version -- give it a ref so the
				# collector keeps it, and drop refs of pins nothing needs now
				wanted = {f"{_PIN_TAG}{pin}" for pin in pinned[repo_id]}
				existing = {t for t in repo.list_tags() if t.startswith(_PIN_TAG)}
				for tag in wanted - existing:
					repo.create_tag(tag, tag[len(_PIN_TAG):])
				for tag in existing - wanted:
					repo.delete_tag(tag)
			summary = repo.garbage_collect(cutoff, dry_run=dry_run)
			report[repo_id] = {"objects": summary.chunks_deleted + summary.manifests_deleted
							   + summary.snapshots_deleted + summary.attributes_deleted
							   + summary.transaction_logs_deleted,
							   "bytes": summary.bytes_deleted}
		return report

	def tags(self, sub_id: str | None = None) -> list[str]:
		"""List the tags on the store.

		Parameters
		----------
		sub_id : str, optional
			Which internal repository to inspect. Defaults to the global dataset
			manifest for current-schema stores.

		Returns
		-------
		list of str
			Tag names, sorted.
		"""
		if sub_id is None:
			sub_id = "_dataset" if self._manifest() is not None \
				else (self.subjects() or ["_dataset"])[0]
		if sub_id != "_dataset":
			Entities(sub_id)
		return sorted(t for t in self._icechunk_repo(sub_id).list_tags() if not t.startswith(_PIN_TAG))


_MERGED = "neurozarr_merged"  # commit metadata: the branch tips a merge brought in
_PIN_TAG = "neurozarr-pin-"  # gc's refs keeping subject snapshots that dataset versions pin


def _replay(source: zarr.Group, dest: zarr.Group, changes: Any) -> None:
	"""Apply one side's changes (an icechunk Diff from the merge base to ``source``'s
	snapshot) onto ``dest``, copying only what changed."""
	def node(root: zarr.Group, path: str) -> Any:
		return root[path.strip("/")] if path.strip("/") else root
	for path in sorted({*changes.deleted_arrays, *changes.deleted_groups}, key=len, reverse=True):
		parent, _, name = path.strip("/").rpartition("/")
		container = node(dest, parent)
		if name in container:
			del container[name]
	for path in sorted({*changes.new_groups, *changes.new_arrays, *changes.updated_arrays}, key=len):
		parent, _, name = path.strip("/").rpartition("/")
		if not name:
			continue
		source_node = node(source, path)
		target_parent = dest.require_group(parent) if parent else dest
		if isinstance(source_node, zarr.Group):
			util.set_attrs(target_parent.require_group(name), source_node.attrs.asdict())
		else:
			_copy_node(source_node, target_parent, name)
	done = {p.strip("/") for p in (*changes.new_arrays, *changes.updated_arrays)}
	for path, coords in changes.updated_chunks.items():
		if path.strip("/") not in done:
			copy_chunks(node(source, path), node(dest, path), [tuple(c) for c in coords])
	for path in changes.updated_groups:
		util.set_attrs(node(dest, path), node(source, path).attrs.asdict())


def _item_of(node_path: str) -> str:
	"""The item a stored node belongs to: a recording's ``data`` array and a table's
	packed column arrays belong to their group; anything else is its own item."""
	parts = node_path.strip("/").split("/")
	if len(parts) > 1 and (parts[-1] == "data" or re.fullmatch(r"[cg]\d{6}(__mask)?", parts[-1])):
		parts = parts[:-1]
	return "/".join(parts)


def _summarize(diff: Any) -> dict[str, list[str]]:
	"""An icechunk Diff (staged or between versions) as items added / changed / removed,
	plus groups whose metadata alone changed."""
	def strip(paths: Any) -> set[str]:
		return {p.strip("/") for p in paths}
	# new items are the groups holding new arrays, plus new groups holding nothing at all;
	# folders on the way down (ses-1, ses-1/eeg) aren't items
	holding = {_item_of(p) for p in diff.new_arrays}
	created = strip(diff.new_groups)
	added = holding | {g for g in created if not any(o.startswith(g + "/") for o in holding | created)}
	# a removed subtree is reported once, at its top
	gone = {_item_of(p) for p in diff.deleted_arrays} | strip(diff.deleted_groups)
	removed = {p for p in gone if not any(p.startswith(o + "/") for o in gone)} - added
	changed = {_item_of(p) for p in (*diff.updated_arrays, *diff.updated_chunks)} - added - removed
	attrs = {p.strip("/") for p in diff.updated_groups} - added - removed - changed
	moved = {f"{a.strip('/')} -> {b.strip('/')}" for a, b in diff.moved_nodes}
	return {key: sorted(p for p in value if p) for key, value in
			(("added", added), ("changed", changed), ("removed", removed), ("attrs", attrs), ("moved", moved))}


class _ItemHost:
	"""Shared add_recording/add_table/add_derivative/add_array
	logic for anything items can be added directly to -- :class:`Subject`
	(session-less) and :class:`Visit` (one session). The two differ only in
	which ``ses_id`` to build :class:`~neurozarr.Entities` with; not a public
	type itself, so a caller never needs to know it exists.

	A subclass must set ``self._repo``/``self.sub_id`` and override
	:meth:`_ses_id` before any of these are called. Not a shared ``ses_id``
	attribute directly: :class:`Visit` needs its own to stay a plain ``str``
	rather than the ``str | None`` a session-less :class:`Subject` needs.
	"""

	_repo: Repo
	sub_id: str

	def _ses_id(self) -> str | None:
		"""Which session items added here belong to. None (the default) for a
		session-less host; Visit overrides this to its own real session id."""
		return None

	def _add_item(self, datatype: str, payload: "mne.io.BaseRaw | pd.DataFrame",
				  meta: dict[str, Any] | None = None, prefix: tuple[str, ...] = (),
				  existing: ExistingPolicy | str = ExistingPolicy.ERROR, **entities: Any) -> None:
		e = Entities(self.sub_id, self._ses_id(), datatype, entities)
		item: Recording | Table
		if isinstance(payload, mne.io.BaseRaw):
			item = Recording(e, payload, meta or {}, prefix)
		elif isinstance(payload, pd.DataFrame):
			item = Table(e, "table", payload, meta or {}, prefix)
		else:
			raise TypeError(f"unsupported recording/table payload type {type(payload)!r}")
		self._repo._dispatch(item, ExistingPolicy(existing))

	def add_recording(self, raw: "mne.io.BaseRaw", meta: dict[str, Any] | None = None,
					  *, datatype: str = "ieeg", existing: ExistingPolicy | str = ExistingPolicy.ERROR,
					  **entities: Any) -> None:
		"""Stage a recording.

		Parameters
		----------
		raw : mne.io.BaseRaw
			The recording to store.
		meta : dict, optional
			Sidecar metadata. Sampling frequency and channel names are taken
			from ``raw`` itself when not given here.
		datatype : str, default "ieeg"
			BIDS datatype directory, e.g. ``"eeg"``, ``"meg"``.
		**entities
			BIDS entities, e.g. ``task="Rest", run=1``.
		"""
		self._add_item(datatype, raw, meta, existing=existing, **entities)

	def add_table(self, df: pd.DataFrame, meta: dict[str, Any] | None = None,
				  *, datatype: str = "beh", existing: ExistingPolicy | str = ExistingPolicy.ERROR,
				  **entities: Any) -> None:
		"""Stage a table.

		Parameters
		----------
		df : pandas.DataFrame
			The table to store. Column dtypes are preserved on read-back.
		meta : dict, optional
			Sidecar metadata.
		datatype : str, default "beh"
			BIDS datatype directory.
		**entities
			BIDS entities, e.g. ``task="TherapyHistory"``.
		"""
		self._add_item(datatype, df, meta, existing=existing, **entities)

	def add_derivative(self, pipeline: str, payload: "mne.io.BaseRaw | pd.DataFrame",
					   datatype: str = "ieeg", meta: dict[str, Any] | None = None,
					   *, pipeline_version: str | None = None,
					   parameters: dict[str, Any] | None = None,
					   inputs: list[str] | None = None,
					   existing: ExistingPolicy | str = ExistingPolicy.ERROR,
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
		manifest = self._repo._manifest()
		provenance = {
			"kind": "derivative",
			"pipeline": pipeline,
			"pipeline_version": pipeline_version,
			"parameters": parameters or {},
			"inputs": inputs or [],
			"source_dataset_id": manifest.dataset_id if manifest else None,
		}
		meta = {"GeneratedBy": pipeline, "_neurozarr_provenance": provenance, **(meta or {})}
		self._add_item(datatype, payload, meta, prefix=("derivatives", pipeline), existing=existing, **entities)

	def add_array(self, name: str, data: Any, dims: tuple[str, ...], *, datatype: str,
				  coords: dict[str, Any] | None = None, meta: dict[str, Any] | None = None,
				  prefix: tuple[str, ...] = (), existing: ExistingPolicy | str = ExistingPolicy.ERROR,
				  **entities: Any) -> None:
		"""Store a named N-dimensional array under a neuroscience datatype."""
		item = Array(Entities(self.sub_id, self._ses_id(), datatype, entities), name, data, dims,
					 coords or {}, meta or {}, prefix)
		self._repo._dispatch(item, ExistingPolicy(existing))


class Subject(_ItemHost):
	"""One participant, backed by their own repository.

	Obtained from :meth:`Repo.subject` or :meth:`Repo.create_subject`.

	For a session-based study, add data with :meth:`add_visit` and the
	methods on the :class:`Visit` it returns. For a session-less one, add
	data directly with :meth:`add_recording`, :meth:`add_table`,
	:meth:`add_derivative` or :meth:`add_array` -- the same methods
	:class:`Visit` has, inherited here, writing with no session segment in
	the path rather than requiring one that doesn't exist for this data.

	Read data back with :meth:`visits`, :meth:`recordings` and :meth:`tables`.

	Attributes
	----------
	sub_id : str
		This subject's BIDS label, e.g. ``"sub-001"``.
	"""

	def __init__(self, repo: Repo, sub_id: str):
		Entities(sub_id)
		self._repo = repo
		self.sub_id = sub_id
		# _ses_id() default (None) is correct here -- items added directly to a
		# Subject carry no session segment; no need to override it.

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
		Entities(self.sub_id, ses_id)
		self._repo._add_attrs(Attrs((ses_id,), attrs or {}), self.sub_id)
		return Visit(self._repo, self.sub_id, ses_id)

	def set_attrs(self, attrs: dict[str, Any]) -> None:
		"""Add or update this subject's metadata after creation.

		Merges into whatever attrs already exist rather than replacing them.
		Not committed until :meth:`Repo.save`.

		Parameters
		----------
		attrs : dict
			Metadata to merge into the subject's own attrs.
		"""
		self._repo._add_attrs(Attrs((), attrs), self.sub_id)

	def rename_visit(self, ses_id: str, new_ses_id: str) -> "Visit":
		"""Rename a session, keeping everything under it.

		A session is just a group inside this subject's own repository, so
		this stays cheap: it copies that one group to the new key and deletes
		the old one, both within the same repository. Not committed until
		:meth:`Repo.save`. This view and any views obtained before the rename
		still show the pre-rename tree until the store is reopened.

		Parameters
		----------
		ses_id : str
			The session's current label.
		new_ses_id : str
			Its new label.

		Returns
		-------
		Visit
			A handle for the session under its new label.
		"""
		Entities(self.sub_id, new_ses_id)
		self._repo._rename((ses_id,), new_ses_id, self.sub_id)
		return Visit(self._repo, self.sub_id, new_ses_id)

	def delete_visit(self, ses_id: str) -> None:
		"""Delete a session and everything under it.

		Not committed until :meth:`Repo.save`.

		Parameters
		----------
		ses_id : str
			The session to delete.
		"""
		self._repo._delete((ses_id,), self.sub_id)

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
		return [v for v in views_in(self.root(version), self.sub_id, self._repo._write_back()) if isinstance(v, RecordingView)]

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
		return [v for v in views_in(self.root(version), self.sub_id, self._repo._write_back()) if isinstance(v, TableView)]

	def arrays(self, version: str | None = None) -> list[ArrayView]:
		"""Every generic N-dimensional array belonging to this subject."""
		return [v for v in views_in(self.root(version), self.sub_id, self._repo._write_back()) if isinstance(v, ArrayView)]

	def external_files(self, version: str | None = None) -> list[ExternalFileView]:
		"""Every unsupported source file preserved by reference for this subject."""
		return [v for v in views_in(self.root(version), self.sub_id, self._repo._write_back()) if isinstance(v, ExternalFileView)]


class Visit(_ItemHost):
	"""One session of one subject.

	Obtained from :meth:`Subject.add_visit` or :meth:`Subject.visit`. Throughout
	this class, keyword arguments are BIDS entities (``task=``, ``run=``,
	``acq=``, …) and determine where data lands in the tree. Add data with
	:meth:`add_recording`, :meth:`add_table`, :meth:`add_derivative`
	or :meth:`add_array`, inherited from the same place :class:`Subject` gets
	its session-less equivalents from.

	Attributes
	----------
	sub_id : str
		The subject this session belongs to.
	ses_id : str
		This session's BIDS label.
	"""

	def __init__(self, repo: Repo, sub_id: str, ses_id: str):
		Entities(sub_id, ses_id)
		self._repo = repo
		self.sub_id = sub_id
		self.ses_id = ses_id

	def _ses_id(self) -> str | None:
		return self.ses_id

	def set_attrs(self, attrs: dict[str, Any]) -> None:
		"""Add or update this session's metadata after creation.

		Merges into whatever attrs already exist rather than replacing them.
		Not committed until :meth:`Repo.save`.

		Parameters
		----------
		attrs : dict
			Metadata to merge into the session's own attrs.
		"""
		self._repo._add_attrs(Attrs((self.ses_id,), attrs), self.sub_id)

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
		return [v for v in views_in(self._group(version), base, self._repo._write_back()) if isinstance(v, RecordingView)]

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
		return [v for v in views_in(self._group(version), base, self._repo._write_back()) if isinstance(v, TableView)]

	def arrays(self, version: str | None = None) -> list[ArrayView]:
		"""Every generic N-dimensional array in this session."""
		base = f"{self.sub_id}/{self.ses_id}"
		return [v for v in views_in(self._group(version), base, self._repo._write_back()) if isinstance(v, ArrayView)]

	def external_files(self, version: str | None = None) -> list[ExternalFileView]:
		"""Every unsupported source file preserved by reference in this session."""
		base = f"{self.sub_id}/{self.ses_id}"
		return [v for v in views_in(self._group(version), base, self._repo._write_back()) if isinstance(v, ExternalFileView)]

	def describe(self, version: str | None = None) -> list[dict[str, Any]]:
		"""Every item in this session, with its kind, path, and BIDS entities.

		Answers "what's actually in here, and what entity is each one" in one
		call, across every item kind, instead of listing each kind separately
		and reading ``.path``/``.entities`` off every result by hand.

		Parameters
		----------
		version : str, optional
			A tag or snapshot id. Default reads the current state.

		Returns
		-------
		list of dict
			One dict per item, with ``kind`` (``"recording"``, ``"table"``,
			``"array"``, or ``"external_file"``), ``path``, and every BIDS
			entity that item carries.

		Examples
		--------
		>>> import pandas as pd
		>>> pd.DataFrame(visit.describe())
		"""
		items: list[dict[str, Any]] = []
		for r in self.recordings(version):
			items.append({"kind": "recording", "path": r.path, **r.entities})
		for t in self.tables(version):
			items.append({"kind": "table", "path": f"{t.path}/{t.name}", **t.entities})
		for a in self.arrays(version):
			items.append({"kind": "array", "path": f"{a.path}/{a.name}", **a.entities})
		for e in self.external_files(version):
			items.append({"kind": "external_file", "path": f"{e.path}/{e.name}", **e.entities})
		# recordings()/tables()/etc. come back in whatever order the underlying
		# tree walk found them, not a useful one -- sort by path so items that
		# sit next to each other in the store also read next to each other here.
		items.sort(key=lambda item: item["path"])
		return items

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
