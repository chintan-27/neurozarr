"""Convert subjects concurrently.

Each subject is its own Icechunk repository, so subjects can be written at the
same time without coordinating: one worker process per subject, no shared session,
no write conflicts. Reading EDF and compressing with zstd are both CPU-bound, so
this uses processes rather than threads.
"""

import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

from .items import Attrs, Item, Reader
from .log import logger
from .readers import BidsReader
from .repo import Repo
from .writer import CodecConfig, ExistingPolicy


def _subject_ids(source: str | Path) -> list[str]:
	return sorted(d.name for d in Path(source).glob("sub-*") if d.is_dir())


class _SubjectOnly:
	"""One subject's items, with the dataset-level ones dropped. Those live in the
	shared _dataset repo, and every worker writing it would conflict -- the parent
	writes them once instead."""

	def __init__(self, reader: Reader, sub_id: str):
		self._reader, self._sub_id = reader, sub_id

	def read(self) -> Iterator[Item]:
		for item in self._reader.read():
			if isinstance(item, Attrs) and not any(p.startswith("sub-") for p in item.path):
				continue
			yield item


def _convert_subject(job: tuple) -> tuple[str, int, str]:
	"""Runs in a worker process: convert exactly one subject into its own repo."""
	source, dest, sub_id, codec, message, existing = job
	repo = Repo.open(dest, mode="a", codec=codec)
	reader = _SubjectOnly(BidsReader(source, subjects=[sub_id]), sub_id)
	repo.ingest(reader, existing=existing)
	writer = repo._writers.get(sub_id)
	snapshot = writer.save(message) if writer is not None else None
	if snapshot is None:
		snapshot = repo._icechunk_repo(sub_id).lookup_branch("main")
	count = len(repo._catalog_entry(sub_id, snapshot)["recordings"])
	return sub_id, count, snapshot


def convert_parallel(source: str | Path, dest: str | Path, workers: int | None = None,
					 codec: CodecConfig | None = None, message: str = "convert",
					 existing: ExistingPolicy | str = ExistingPolicy.ERROR,
					 skip_existing: bool | None = None) -> dict[str, int]:
	"""Convert a BIDS dataset using one worker process per subject.

	Since a subject is the unit of work, the wall-clock time is bounded by the
	largest single subject however many workers are available.

	Parameters
	----------
	source : str or pathlib.Path
		The BIDS dataset to convert.
	dest : str or pathlib.Path
		Where to write the store.
	workers : int, optional
		Number of worker processes. Defaults to what
		:class:`~concurrent.futures.ProcessPoolExecutor` chooses.
	codec : CodecConfig, optional
		How sample data is packed and compressed.
	message : str, default "convert"
		Commit message for each subject's commit.
	existing : {"error", "skip", "replace"}, default "error"
		How to handle an item already present at the same path.
	skip_existing : bool, optional
		Deprecated alias for the skip/replace policies.

	Returns
	-------
	dict
		Number of recordings converted, keyed by subject label.

	Raises
	------
	ValueError
		If ``source`` contains no ``sub-*`` directories.

	Examples
	--------
	>>> convert_parallel("./BIDS", "./study.zarr", workers=6)
	{'sub-001': 412, 'sub-002': 173, ...}
	"""
	subjects = _subject_ids(source)
	if not subjects:
		raise ValueError(f"no sub-* directories in {source}")

	# dataset-level metadata (dataset_description, participants) in the parent,
	# so the workers never race to write the same _dataset repo
	try:
		parent = Repo.open(dest, mode="a", codec=codec)
	except FileNotFoundError:
		parent = Repo.create(dest, codec=codec)
	policy = ExistingPolicy(existing)
	if skip_existing is not None:
		warnings.warn("skip_existing is deprecated; pass existing='skip' or 'replace'",
					  DeprecationWarning, stacklevel=2)
		policy = ExistingPolicy.SKIP if skip_existing else ExistingPolicy.REPLACE
	parent.ingest(BidsReader(source, subjects=[]), existing=policy)
	parent.save(message)

	logger.info("converting %d subjects with %s workers", len(subjects), workers or "default")
	jobs = [(str(source), str(dest), sub_id, codec, message, policy) for sub_id in subjects]

	# "spawn", not the Linux default "fork": the parent has already opened icechunk
	# repositories, and forking a process with that runtime's threads live deadlocks
	# the children. A fresh interpreter per worker costs a second and always works.
	context = multiprocessing.get_context("spawn")
	with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
		results = list(pool.map(_convert_subject, jobs))
	parent = Repo.open(dest, mode="a", codec=codec)
	parent._publish_subject_snapshots({sub_id: snapshot for sub_id, _, snapshot in results}, message)
	return {sub_id: count for sub_id, count, _ in results}
