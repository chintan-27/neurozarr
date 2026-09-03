"""Convert subjects concurrently.

Each subject is its own Icechunk repository, so subjects can be written at the
same time without coordinating: one worker process per subject, no shared session,
no write conflicts. Reading EDF and compressing with zstd are both CPU-bound, so
this uses processes rather than threads.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from .items import Attrs
from .log import logger
from .readers import BidsReader
from .repo import Repo


def _subject_ids(source) -> list:
	from pathlib import Path
	return sorted(d.name for d in Path(source).glob("sub-*") if d.is_dir())


class _SubjectOnly:
	"""One subject's items, with the dataset-level ones dropped. Those live in the
	shared _dataset repo, and every worker writing it would conflict -- the parent
	writes them once instead."""

	def __init__(self, reader, sub_id: str):
		self._reader, self._sub_id = reader, sub_id

	def read(self):
		for item in self._reader.read():
			if isinstance(item, Attrs) and not any(p.startswith("sub-") for p in item.path):
				continue
			yield item


def _convert_subject(job) -> tuple:
	"""Runs in a worker process: convert exactly one subject into its own repo."""
	source, dest, sub_id, codec, message, skip_existing = job
	repo = Repo(dest, codec=codec)
	reader = _SubjectOnly(BidsReader(source, subjects=[sub_id]), sub_id)
	repo.ingest(reader, skip_existing=skip_existing)
	repo.save(message)
	return sub_id, len(repo.subject(sub_id).recordings())


def convert_parallel(source, dest, workers: int = None, codec=None,
					 message: str = "convert", skip_existing: bool = False) -> dict:
	"""Convert a BIDS dataset with one worker process per subject.

	Returns {subject_id: recording_count}. Speedup is bounded by the largest
	subject, since a subject is the unit of work.
	"""
	subjects = _subject_ids(source)
	if not subjects:
		raise ValueError(f"no sub-* directories in {source}")

	# dataset-level metadata (dataset_description, participants) in the parent,
	# so the workers never race to write the same _dataset repo
	parent = Repo(dest, codec=codec)
	parent.ingest(BidsReader(source, subjects=[]))
	parent.save(message)

	logger.info("converting %d subjects with %s workers", len(subjects), workers or "default")
	jobs = [(str(source), str(dest), sub_id, codec, message, skip_existing) for sub_id in subjects]

	# "spawn", not the Linux default "fork": the parent has already opened icechunk
	# repositories, and forking a process with that runtime's threads live deadlocks
	# the children. A fresh interpreter per worker costs a second and always works.
	context = multiprocessing.get_context("spawn")
	with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
		return dict(pool.map(_convert_subject, jobs))
