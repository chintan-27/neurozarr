"""Index costs at scale, on the cluster: (1) the summary -- full build and a commit's
splice at 1,000 / 10,000 subjects (synthetic indexes, 100 items each); (2) the packed
index -- a cold query over every subject of a real store with N tiny subjects, read
through per-subject indexes vs the packed one, and what a one-subject commit costs.

	python scripts/bench/index_scale_bench.py --out index_scale.jsonl --scratch DIR
"""

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_index_bench import make  # noqa: E402

from neurozarr import Repo  # noqa: E402
from neurozarr import index as nzindex  # noqa: E402


def timed(fn, reps: int) -> tuple[float, list[float]]:
	times = []
	for _ in range(reps):
		start = time.perf_counter()
		fn()
		times.append(time.perf_counter() - start)
	return statistics.median(times), times


def summary_rows(n_subjects: int, reps: int) -> list[dict]:
	blobs = make(n_subjects, 100)
	tables = {s: nzindex.Table.decode(b) for s, b in blobs.items()}
	stats = [nzindex.stats_of(tables[s], nzindex.digest(b)) for s, b in blobs.items()]
	full, _ = timed(lambda: nzindex.build_summary(stats), reps)
	previous = nzindex.Summary(nzindex.Table.decode(nzindex.build_summary(stats)))
	rows = [{"test": "summary_full_build", "n_subjects": n_subjects, "seconds": full}]
	for k in (1, 10, 100):
		changed = {s: nzindex.stats_of(tables[s], nzindex.digest(blobs[s])) for s in previous.subjects[:k]}
		splice, _ = timed(lambda: nzindex.update_summary(previous, changed), reps)
		rows.append({"test": "summary_commit_splice", "n_subjects": n_subjects, "changed": k, "seconds": splice})
	return rows


def _raw(seed: int) -> mne.io.RawArray:
	data = np.random.default_rng(seed).standard_normal((2, 200)) * 1e-5
	return mne.io.RawArray(data, mne.create_info(2, 100.0, "eeg"), verbose=False)


def packed_rows(n_subjects: int, scratch: Path, reps: int) -> list[dict]:
	store = scratch / f"packed_{n_subjects}"
	shutil.rmtree(store, ignore_errors=True)
	repo = Repo.init(str(store))
	for s in range(n_subjects):
		visit = repo.subject(f"sub-{s:05d}").visit("ses-1")
		for task in ("rest", "task"):
			visit.add_recording(_raw(s), task=task)
	start = time.perf_counter()
	repo.commit("all subjects")
	rows = [{"test": "first_commit", "n_subjects": n_subjects, "seconds": time.perf_counter() - start}]

	def cold_query() -> None:
		nzindex.CACHE = nzindex._Cache()
		assert len(Repo.open(str(store)).query()) == 2 * n_subjects

	for label, pack_min in (("per_subject_indexes", 10**9), ("packed_index", nzindex.PACK_MIN)):
		old, nzindex.PACK_MIN = nzindex.PACK_MIN, pack_min
		median, times = timed(cold_query, reps)
		nzindex.PACK_MIN = old
		rows.append({"test": "cold_query_all", "n_subjects": n_subjects, "via": label, "seconds": median, "all": times})

	def one_subject_commit(i: int = 0) -> None:
		writer = Repo.open(str(store), mode="a")
		writer.subject("sub-00000").visit("ses-1").add_recording(_raw(10_000 + i), task=f"extra{i}")
		writer.commit(f"one more {i}")

	commits = []
	for i in range(reps):
		start = time.perf_counter()
		one_subject_commit(i)
		commits.append(time.perf_counter() - start)
	rows.append({"test": "one_subject_commit", "n_subjects": n_subjects, "seconds": statistics.median(commits), "all": commits})
	shutil.rmtree(store, ignore_errors=True)
	return rows


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--out", required=True)
	parser.add_argument("--scratch", required=True)
	parser.add_argument("--summary-sizes", default="1000,10000")
	parser.add_argument("--packed-sizes", default="150,1000")
	parser.add_argument("--reps", type=int, default=7)
	args = parser.parse_args()
	scratch = Path(args.scratch)
	scratch.mkdir(parents=True, exist_ok=True)
	base = {"node": platform.node(), "loadavg": os.getloadavg()[0]}
	with open(args.out, "a") as out:
		for n in map(int, args.summary_sizes.split(",")):
			for row in summary_rows(n, args.reps):
				out.write(json.dumps({**base, **row}) + "\n")
				print(row, flush=True)
		for n in map(int, args.packed_sizes.split(",")):
			for row in packed_rows(n, scratch, args.reps):
				out.write(json.dumps({**base, **row}) + "\n")
				print({k: v for k, v in row.items() if k != "all"}, flush=True)


if __name__ == "__main__":
	main()
