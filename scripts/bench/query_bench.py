"""Metadata-query latency: neurozarr's index vs the old JSON catalog vs pybids / bids2table.

	python scripts/bench/query_bench.py --impl new  --store STORE --name rns --out q.jsonl
	python scripts/bench/query_bench.py --impl old  --store OLD_STORE --name rns --out q.jsonl   # old code on PYTHONPATH
	python scripts/bench/query_bench.py --impl pybids --source BIDS_DIR --name rns --out q.jsonl
	python scripts/bench/query_bench.py --impl bids2table --source BIDS_DIR --name rns --out q.jsonl

Every row records the implementation, query, cache state, repetition, seconds, the number
of results and a digest of the sorted result paths, so implementations can be checked
against each other, plus the node and load. "cold" means fresh process state each
repetition (a new Repo and empty in-memory caches; the OS page cache is not dropped),
"warm" means one open Repo after two discarded repetitions.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
import warnings
from collections.abc import Callable
from typing import Any

warnings.simplefilter("ignore")


def _digest(paths: list[str]) -> str:
	return hashlib.blake2b("\n".join(sorted(paths)).encode(), digest_size=8).hexdigest()


def _row(args: argparse.Namespace, query: str, cache: str, rep: int, seconds: float, paths: list[str] | int) -> dict[str, Any]:
	n = paths if isinstance(paths, int) else len(paths)
	return {"dataset": args.name, "impl": args.impl, "query": query, "cache": cache, "rep": rep,
			"seconds": seconds, "n": n, "paths_digest": None if isinstance(paths, int) else _digest(paths),
			"node": platform.node(), "loadavg": os.getloadavg()[0], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _measure(args: argparse.Namespace, out: Any, name: str, run: Callable[[], Any], fresh: Callable[[], None] | None) -> None:
	"""cold: fresh() before every repetition; warm: two discarded repetitions first."""
	for cache in ("cold", "warm"):
		if cache == "warm":
			for _ in range(2):
				run()
		for rep in range(args.reps):
			if cache == "cold" and fresh is not None:
				fresh()
			start = time.perf_counter()
			result = run()
			out.write(json.dumps(_row(args, name, cache, rep, time.perf_counter() - start, result)) + "\n")
		out.flush()


def bench_neurozarr(args: argparse.Namespace, out: Any) -> None:
	import neurozarr
	from neurozarr import Repo
	print(f"neurozarr from {neurozarr.__file__}", flush=True)
	state: dict[str, Any] = {}

	def fresh() -> None:
		if args.impl == "new":
			from neurozarr import index
			index.CACHE = index._Cache()
		state["repo"] = Repo.open(args.store)

	fresh()
	subjects = state["repo"].subjects()
	first = subjects[0]
	_measure(args, out, "open", lambda: (fresh(), 0)[1], None)
	_measure(args, out, "find_subject", lambda: [v.path for v in state["repo"].find(sub=first)], fresh)
	_measure(args, out, "find_all", lambda: [v.path for v in state["repo"].find()], fresh)
	if args.impl != "new":
		return
	frame = state["repo"].query(kind="recording").df
	task = frame["task"].mode()[0] if "task" in frame and frame["task"].notna().any() else None
	queries = {"sfreq_ge_250": "sfreq >= 250", "long_multichannel": "duration_s > 60 and n_channels >= 4"}
	if task is not None:
		queries["task_eq"] = f"task == {task!r}"
	for name, text in queries.items():
		_measure(args, out, name, lambda text=text: state["repo"].query(text).paths, fresh)
	_measure(args, out, "count_all", lambda: len(state["repo"].query()), fresh)
	_measure(args, out, "df_all", lambda: len(state["repo"].query().df), fresh)
	one = frame["path"].iloc[len(frame) // 2]
	_measure(args, out, "get_one", lambda: [state["repo"].get(one).path], fresh)


def bench_pybids(args: argparse.Namespace, out: Any) -> None:
	import bids
	print(f"pybids {bids.__version__}", flush=True)
	start = time.perf_counter()
	layout = bids.BIDSLayout(args.source, validate=False)
	out.write(json.dumps(_row(args, "index_build", "cold", 0, time.perf_counter() - start, 0)) + "\n")
	data = [".edf", ".bdf", ".set", ".vhdr", ".fif", ".nwb"]
	first = sorted(layout.get_subjects())[0]
	_measure(args, out, "find_subject", lambda: layout.get(subject=first, extension=data, return_type="filename"), None)
	_measure(args, out, "find_all", lambda: layout.get(extension=data, return_type="filename"), None)
	tasks = layout.get_tasks()
	if tasks:
		counts = {t: len(layout.get(task=t, extension=data, return_type="filename")) for t in tasks}
		task = max(counts, key=lambda t: counts[t])
		_measure(args, out, "task_eq", lambda: layout.get(task=task, extension=data, return_type="filename"), None)


def bench_bids2table(args: argparse.Namespace, out: Any) -> None:
	import bids2table
	version = getattr(bids2table, "__version__", "?")
	print(f"bids2table {version}", flush=True)
	start = time.perf_counter()
	if hasattr(bids2table, "index_dataset"):  # 2.x
		table = bids2table.index_dataset(args.source).to_pandas()
	else:  # 0.x/1.x
		table = bids2table.bids2table(args.source).flat
	out.write(json.dumps(_row(args, "index_build", "cold", 0, time.perf_counter() - start, len(table))) + "\n")
	ext_col = next(c for c in table.columns if c.endswith("ext") or c.endswith("extension"))
	sub_col = next(c for c in table.columns if c.endswith("sub"))
	path_col = next(c for c in table.columns if c.endswith("path") or c.endswith("file_path"))
	data = table[table[ext_col].astype(str).isin([".edf", ".bdf", ".set", ".vhdr", ".fif", ".nwb"])]
	first = sorted(data[sub_col].dropna().unique())[0]
	_measure(args, out, "find_subject", lambda: list(data.loc[data[sub_col] == first, path_col]), None)
	_measure(args, out, "find_all", lambda: list(data[path_col]), None)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--impl", choices=["new", "old", "pybids", "bids2table"], required=True)
	parser.add_argument("--store")
	parser.add_argument("--source")
	parser.add_argument("--name", required=True)
	parser.add_argument("--out", required=True)
	parser.add_argument("--reps", type=int, default=10)
	args = parser.parse_args()
	with open(args.out, "a") as out:
		{"new": bench_neurozarr, "old": bench_neurozarr, "pybids": bench_pybids, "bids2table": bench_bids2table}[args.impl](args, out)
	rows = [json.loads(line) for line in open(args.out) if json.loads(line)["impl"] == args.impl
			and json.loads(line)["dataset"] == args.name]
	for (query, cache) in sorted({(r["query"], r["cache"]) for r in rows}):
		times = [r["seconds"] for r in rows if r["query"] == query and r["cache"] == cache]
		n = [r["n"] for r in rows if r["query"] == query and r["cache"] == cache][0]
		print(f"{args.impl:10s} {query:18s} {cache:4s} median {statistics.median(times) * 1000:9.2f} ms  n={n}")


if __name__ == "__main__":
	main()
