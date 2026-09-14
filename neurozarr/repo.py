from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterator, Literal, cast
import logging
import warnings

import icechunk
import mne  # type: ignore[import-untyped]  # mne ships no type information
import pandas as pd
import zarr

from .entities import Entities
from .errors import SchemaVersionError, StoreIntegrityError, WriteConflictError
from .items import Array, Attrs, ExternalFile, Recording, Reader, Table
from .log import log_progress, set_verbosity
from .read import ArrayView, ExternalFileView, RecordingView, TableView, WriteBack, views_in
from .schema import MANIFEST_ATTR, SCHEMA_VERSION, StoreManifest, utc_now
from .storage import StorageTarget, storage_from
from .writer import CodecConfig, ExistingPolicy, Writer


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


def _open_writer(icechunk_repo: icechunk.Repository, codec: CodecConfig | None, label: str = "") -> Writer:
	return Writer(icechunk_repo.writable_session("main"), codec, label)


class Repo:
	"""A store of recordings, and the entry point to everything else.

	A store is a directory of Icechunk repositories — one per subject, so any
	single subject can be copied, shared or versioned without moving the whole
	study — plus a ``_dataset`` repository holding metadata that belongs to no
	single subject, such as the dataset description and participant field
	definitions.

	Fill a store either in bulk with :meth:`ingest`, or explicitly with
	:meth:`create_subject` and the methods on :class:`Subject` and
	:class:`Visit`. Either way, call :meth:`save` to commit.

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
		Log one line per item as it's written -- what it is, its size, and
		current + peak process memory right before the write (current needs
		the optional ``psutil``; peak alone otherwise) -- plus one line per
		commit. Equivalent to ``neurozarr.set_verbosity(logging.INFO)``; the
		package stays quiet by default, as a library should.
	**storage_options
		Passed through to the underlying icechunk storage constructor, e.g.
		``region="us-east-1"``, ``anonymous=True``, ``from_env=True``.

	Examples
	--------
	>>> repo = Repo.create("./study.zarr")
	>>> repo.ingest(BidsReader("./my_bids_dataset"))
	>>> repo.save("initial conversion")
	"""

	def __init__(self, target: StorageTarget, codec: CodecConfig | None = None,
				 **storage_options: Any):
		warnings.warn(
			"Repo(target) uses legacy open-or-create behavior; prefer Repo.create(target) or Repo.open(target)",
			DeprecationWarning, stacklevel=2,
		)
		self._init(target, codec, "legacy", storage_options)

	def _init(self, target: StorageTarget, codec: CodecConfig | None,
			  mode: Literal["legacy", "r", "a", "x"], storage_options: dict[str, Any]) -> None:
		self.target = target
		self._codec = codec
		self._storage_options = storage_options
		self._mode = mode
		self._repos: dict[str, icechunk.Repository] = {}  # opened lazily
		self._writers: dict[str, Writer] = {}  # opened lazily on first write
		self._base_manifest: StoreManifest | None = None
		self._removed_subjects: set[str] = set()
		self._failed = False

	@classmethod
	def create(cls, target: StorageTarget, codec: CodecConfig | None = None, verbose: bool = False,
			   **storage_options: Any) -> "Repo":
		"""Create a new store, failing if its dataset repository already exists."""
		if verbose:
			set_verbosity(logging.INFO)
		self = cls.__new__(cls)
		self._init(target, codec, "x", storage_options)
		if self._repo_exists("_dataset"):
			raise FileExistsError(f"a neurozarr store already exists at {target}")
		log_progress("Creating store", str(target))
		self._initialize_dataset("initialize neurozarr store")
		self._mode = "a"
		return self

	@classmethod
	def open(cls, target: StorageTarget, mode: Literal["r", "a"] = "r",
			 codec: CodecConfig | None = None, verbose: bool = False, **storage_options: Any) -> "Repo":
		"""Open an existing store without accidentally creating an empty one."""
		if verbose:
			set_verbosity(logging.INFO)
		self = cls.__new__(cls)
		self._init(target, codec, mode, storage_options)
		if not self._repo_exists("_dataset"):
			raise FileNotFoundError(f"no neurozarr store exists at {target}")
		manifest = self._manifest()
		if manifest is None and mode == "a":
			raise SchemaVersionError("legacy store is read-only until `neurozarr migrate` is run")
		log_progress("Opened store", str(target))
		return self

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
			raise RuntimeError("this transaction failed; call abort() before writing again")
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
			self._writers[sub_id] = _open_writer(self._icechunk_repo(sub_id), self._codec, sub_id)
		return self._writers[sub_id]

	def _dispatch(self, item: Recording | Table | ExternalFile | Array,
				  existing: ExistingPolicy) -> bool:
		"""Dispatch one item and make partial write failures impossible to commit."""
		try:
			return self._writer_for(item.entities.sub).dispatch(item, existing)
		except Exception:
			self._failed = True
			raise

	def _add_attrs(self, item: Attrs, sub_id: str) -> None:
		"""Add metadata while preserving the transaction failure invariant."""
		try:
			self._writer_for(sub_id).add_attrs(item)
		except Exception:
			self._failed = True
			raise

	def _delete(self, path: tuple[str, ...], sub_id: str) -> None:
		try:
			self._writer_for(sub_id).delete(path)
		except Exception:
			self._failed = True
			raise

	def _rename(self, path: tuple[str, ...], new_name: str, sub_id: str) -> None:
		try:
			self._writer_for(sub_id).rename(path, new_name)
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
		return snapshot

	@staticmethod
	def _package_version() -> str:
		try:
			return package_version("neurozarr")
		except PackageNotFoundError:
			return "0.0.0.dev0"

	def _dataset_root(self, version: str | None = None) -> zarr.Group:
		repo = self._icechunk_repo("_dataset")
		if version is not None:
			session = repo.readonly_session(tag=version) if version in repo.list_tags() \
				else repo.readonly_session(snapshot_id=version)
		else:
			session = repo.readonly_session("main")
		return zarr.open_group(store=session.store, mode="r")

	def _manifest(self, version: str | None = None) -> StoreManifest | None:
		if not self._repo_exists("_dataset"):
			return None
		try:
			value = self._dataset_root(version).attrs.asdict().get(MANIFEST_ATTR)
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
		Not committed until :meth:`save`.

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

		Not committed until :meth:`save`.

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

	def ingest(self, reader: Reader, progress: Any = None,
			   existing: ExistingPolicy | str = ExistingPolicy.ERROR,
			   skip_existing: bool | None = None) -> None:
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
		existing : {"error", "skip", "replace"}, default "error"
			How to handle an item already present at the same entity path.
		skip_existing : bool, optional
			Deprecated compatibility alias. ``True`` means ``existing="skip"``
			and explicitly passing ``False`` means ``existing="replace"``.

		See Also
		--------
		save : Commit what was ingested.
		neurozarr.parallel.convert_parallel : Ingest subjects concurrently.
		"""
		policy = ExistingPolicy(existing)
		if skip_existing is not None:
			warnings.warn("skip_existing is deprecated; pass existing='skip' or 'replace'", DeprecationWarning,
						  stacklevel=2)
			policy = ExistingPolicy.SKIP if skip_existing else ExistingPolicy.REPLACE
		items = reader.read()
		try:
			for item in progress(items) if progress else items:
				if isinstance(item, Attrs):
					self._route_attrs(item)
				else:
					self._dispatch(item, policy)
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

	def save(self, message: str) -> str | None:
		"""Commit every subject repository touched since the last save.

		Each subject is committed independently; subjects with no changes are
		left alone.

		Parameters
		----------
		message : str
			Commit message, shown by :meth:`history`.
		"""
		if self._failed:
			raise RuntimeError("cannot save a failed transaction; call abort() and retry")
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
		current = self._manifest()
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
			committed_attrs = self._dataset_root().attrs.asdict()
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
			current = self._manifest() or base
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

	def abort(self) -> None:
		"""Discard all uncommitted sessions held by this Repo instance."""
		self._writers.clear()
		self._base_manifest = None
		self._removed_subjects.clear()
		self._failed = False

	@contextmanager
	def transaction(self, message: str) -> Iterator["Repo"]:
		"""Commit changes on clean exit and discard them when the block fails."""
		try:
			yield self
			self.save(message)
		except Exception:
			self.abort()
			raise

	def migrate(self, dry_run: bool = False) -> dict[str, Any]:
		"""Upgrade a readable 0.1 store to the current manifest schema in place."""
		if not self._repo_exists("_dataset"):
			raise FileNotFoundError(f"no neurozarr store exists at {self.target}")
		current = self._manifest()
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
		candidate_subjects = [sub] if sub else self.subjects()
		manifest = self._manifest()
		if manifest is not None and sub is None and entities:
			# The catalog only ever prunes subjects it can currently vouch for. An entry
			# that is missing, unstamped (written before entries carried a snapshot) or
			# stamped with a snapshot this version no longer publishes means "unknown",
			# never "no match" -- treating it as a match list would silently drop
			# recordings that are really there.
			candidate_subjects = [
				sub_id for sub_id in candidate_subjects
				if not self._catalog_is_current(manifest, sub_id)
				or any(all(str(entry.get("entities", {}).get(k)) == str(v) for k, v in entities.items())
					   for entry in manifest.catalog[sub_id].get("recordings", []))
			]
		for sub_id in candidate_subjects:
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
		return [(s.id, s.message, s.written_at) for s in self._icechunk_repo(sub_id).ancestry(branch="main")]

	def tag(self, name: str) -> None:
		"""Name the store's current state so it can be read back later.

		Current-schema stores tag the dataset manifest. That manifest pins the
		exact snapshot of every subject, so one tag names a consistent global
		version without mutating the subject repositories.

		Parameters
		----------
		name : str
			Tag name, e.g. ``"v1"``. Pass it as ``version=`` to
			:meth:`Subject.recordings` or :meth:`root_of` to read that state.
		"""
		from .constraints import validate_segment

		validate_segment(name, "tag name")
		manifest = self._manifest()
		if manifest is not None:
			repo = self._icechunk_repo("_dataset")
			if name not in repo.list_tags():
				repo.create_tag(name, repo.lookup_branch("main"))
			return
		for sub_id in self.subjects() + ["_dataset"]:
			repo = self._icechunk_repo(sub_id)
			if name not in repo.list_tags():
				repo.create_tag(name, repo.lookup_branch("main"))

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
		return sorted(self._icechunk_repo(sub_id).list_tags())


class _ItemHost:
	"""Shared add/add_recording/add_behavioral_table/add_derivative/add_array
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

	def add(self, datatype: str, payload: "mne.io.BaseRaw | pd.DataFrame",
			meta: dict[str, Any] | None = None, prefix: tuple[str, ...] = (),
			existing: ExistingPolicy | str = ExistingPolicy.ERROR, **entities: Any) -> None:
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
					  *, existing: ExistingPolicy | str = ExistingPolicy.ERROR, **entities: Any) -> None:
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
		self.add("ieeg", raw, meta, existing=existing, **entities)

	def add_behavioral_table(self, df: pd.DataFrame, meta: dict[str, Any] | None = None,
							 *, existing: ExistingPolicy | str = ExistingPolicy.ERROR,
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
		self.add("beh", df, meta, existing=existing, **entities)

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
		self.add(datatype, payload, meta, prefix=("derivatives", pipeline), existing=existing, **entities)

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
	data directly with :meth:`add_recording`, :meth:`add_behavioral_table`,
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
	:meth:`add_recording`, :meth:`add_behavioral_table`, :meth:`add_derivative`
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
