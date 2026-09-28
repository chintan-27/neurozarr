"""The measurement harness: runs every workload against every arm, n times
each, with cache control, and emits one JSON-Lines record per repetition --
raw enough that no claim in a paper depends on trusting a printed summary.

	python -m scripts.bench.runner BIDS/ --arms raw,neurozarr,plain_zarr --out results.jsonl
	python -m scripts.bench.runner results.jsonl --summarize

Workloads (see arms.py's ArmStore for what each calls):
	W1  full recording read (all channels, all samples)
	W2  1.0s window read at 50% offset (all channels)
	W3  single-channel full trace
	W4  cross-subject/task metadata search
	W5  sequential scan in 10s blocks, MB/s

Every workload's result is hashed (after rounding -- see `_checksum`) so a
later analysis can confirm two arms actually returned the same data, not just
that both finished without raising. `t_open` (store construction cost) is
recorded once per (arm, cache-state) rather than per workload, since it's
what W2 and W4 would otherwise silently fold into their own numbers.

Statistical protocol (see the plan): n=10 measured reps per condition after 2
discarded warm-up reps for warm-cache runs; n=5 for --time-convert. Report
median+IQR+min, never mean -- I/O latency is right-skewed with real outliers.
Never pool cold and warm, or local and cluster, in one comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.arms import ARMS, ArmStore, RecInfo  # noqa: E402

WORKLOADS = ("W1", "W2", "W3", "W4", "W5")
WARMUP_REPS = 2
DEFAULT_REPS = 10


# ---------------------------------------------------------------------------
# Reproducibility record
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> str | None:
	try:
		return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
	except (OSError, subprocess.SubprocessError):
		return None


def _git_info() -> dict[str, Any]:
	sha = _run(["git", "rev-parse", "HEAD"])
	dirty = _run(["git", "status", "--porcelain"])
	return {"git_sha": sha, "git_dirty": bool(dirty) if sha else None}


def _pip_freeze() -> list[str]:
	out = _run([sys.executable, "-m", "pip", "freeze"])
	return out.splitlines() if out else []


def _cpu_model() -> str | None:
	try:
		for line in Path("/proc/cpuinfo").read_text().splitlines():
			if line.startswith("model name"):
				return line.split(":", 1)[1].strip()
	except OSError:
		pass
	return platform.processor() or None


def _filesystem_of(path: Path) -> str | None:
	# `df -T` needs no root and works for any mounted path, local disk or
	# cluster-mounted Lustre/GPFS/NFS alike -- the single biggest confound
	# for one-file-per-chunk formats, per the plan.
	out = _run(["df", "-T", str(path)])
	if not out:
		return None
	lines = out.splitlines()
	return lines[1].split()[1] if len(lines) > 1 else None


def environment_record(store_path: Path) -> dict[str, Any]:
	return {
		**_git_info(),
		"pip_freeze": _pip_freeze(),
		"python_version": platform.python_version(),
		"os": platform.platform(),
		"kernel": platform.release(),
		"cpu_model": _cpu_model(),
		"ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else None,
		"filesystem": _filesystem_of(store_path),
		"hostname": socket.gethostname(),
		"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
		"slurm_node": os.environ.get("SLURMD_NODENAME"),
	}


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as f:
		for block in iter(lambda: f.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def dataset_fingerprint(source: Path, max_files: int = 5000) -> dict[str, str]:
	"""SHA-256 per source file -- the download scripts currently resume on
	file-size match only, which isn't adequate provenance for a paper.
	Capped at `max_files` for a 100GB+ dataset: full-content hashing every
	file there costs more than the benchmark itself; a truncated fingerprint
	is flagged, not silently presented as complete."""
	files = sorted(p for p in Path(source).rglob("*") if p.is_file())
	truncated = len(files) > max_files
	return {
		"_truncated": truncated,  # type: ignore[dict-item]
		**{str(p.relative_to(source)): _sha256_file(p) for p in files[:max_files]},
	}


# ---------------------------------------------------------------------------
# Cache control
# ---------------------------------------------------------------------------

def drop_caches_local() -> None:
	"""Real cold cache, local box only. Fails loudly without root rather than
	silently reporting warm numbers as cold -- see the plan's statistical
	protocol."""
	subprocess.run(["sync"], check=True)
	try:
		Path("/proc/sys/vm/drop_caches").write_text("3\n")
	except PermissionError as exc:
		raise PermissionError(
			"cold cache requested but /proc/sys/vm/drop_caches is not writable "
			"(need root). Re-run as root, or pass --cache warm."
		) from exc


def fadvise_dontneed(paths: list[Path]) -> None:
	"""Cold-enough cache on a shared cluster, where drop_caches needs a
	privilege you won't have: evict each file's pages via POSIX_FADV_DONTNEED.
	Validate this against real drop_caches on a local box before trusting a
	cluster cold-cache number -- the plan calls this out explicitly."""
	for path in paths:
		if not path.is_file():
			continue
		fd = os.open(str(path), os.O_RDONLY)
		try:
			os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
		finally:
			os.close(fd)


def _files_under(dest: Path) -> list[Path]:
	return [p for p in Path(dest).rglob("*") if p.is_file()]


# ---------------------------------------------------------------------------
# Checksums and timing
# ---------------------------------------------------------------------------

def _checksum(arr: np.ndarray) -> str:
	"""Rounded before hashing: arms that store float32 (hdf5_mat73, nwb_*)
	vs. neurozarr's lossless int16 calibration differ at ~1e-7 on real data
	(measured), which is an expected representation difference, not a
	fidelity bug -- rounding to 1e-6 absolute lets equivalent arms produce
	the same checksum while a genuinely wrong read still shows a different
	one."""
	rounded = np.round(np.ascontiguousarray(arr, dtype=np.float64), 6)
	return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def _time(fn: Any) -> tuple[Any, float]:
	t0 = time.perf_counter()
	result = fn()
	return result, time.perf_counter() - t0


def _peak_rss_kb() -> int:
	# Cumulative peak since process start, not isolated to one rep -- ru_maxrss
	# is monotonic and stdlib has no cheaper per-call reset; still useful as a
	# ceiling, just not a per-rep delta. Named accordingly in the record.
	return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

def _pick_target(recs: list[RecInfo]) -> RecInfo:
	real = [r for r in recs if r.n_samples > 1 and r.sfreq > 0]
	if not real:
		raise ValueError("no recording with more than one sample and a known sampling rate")
	return max(real, key=lambda r: r.n_samples)


def run_workload(workload: str, store: ArmStore, target: RecInfo) -> tuple[Any, float, str]:
	"""Returns (result, elapsed_seconds, checksum). Every result is forced to
	a concrete array before checksumming, so a lazily-evaluated arm can't
	report a time that doesn't include the actual read."""
	if workload == "W1":
		result, t = _time(lambda: store.full_read(target.id))
		return result, t, _checksum(result)

	if workload == "W2":
		window = max(1, int(round(target.sfreq)))  # 1.0s
		start = target.n_samples // 2
		start = min(start, max(0, target.n_samples - window))
		result, t = _time(lambda: store.window_read(target.id, start, window))
		return result, t, _checksum(result)

	if workload == "W3":
		result, t = _time(lambda: store.channel_read(target.id, 0))
		return result, t, _checksum(result)

	if workload == "W4":
		# A specific, deterministic key -- not "whichever entity comes first":
		# HDF5 attrs don't preserve insertion order the way a dict or a BIDS
		# filename does, so next(iter(...)) picked a different key on
		# different arms for "the same" recording (found by comparing cross-
		# arm checksums on real data, not by inspection).
		key = next((k for k in ("task", "acq", "run") if k in target.entities), None)
		if key is None:
			return [], 0.0, "no-entities"
		value = target.entities[key]
		result, t = _time(lambda: store.search(**{key: value}))
		return result, t, hashlib.sha256(",".join(sorted(result)).encode()).hexdigest()[:16]

	if workload == "W5":
		block = max(1, int(round(target.sfreq)) * 10)  # 10s blocks

		def scan() -> np.ndarray:
			# Every arm's sequential_scan yields channel-major (n_channels,
			# block_len) blocks -- NwbStore transposes to match on the way out.
			total = np.zeros(target.n_channels, dtype=np.float64)
			for block_arr in store.sequential_scan(target.id, block):
				total += np.asarray(block_arr, dtype=np.float64).sum(axis=-1)
			return total

		result, t = _time(scan)
		return result, t, _checksum(result)

	raise ValueError(f"unknown workload {workload!r}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(source: Path, arm_names: list[str], reps: int, warmup: int, cache: str,
		out: Path, dataset_name: str, scratch: Path, checksum_source: bool) -> None:
	env = environment_record(scratch)
	fingerprint = dataset_fingerprint(source) if checksum_source else {"_skipped": True}  # type: ignore[dict-item]

	with out.open("a") as sink:
		for arm_name in arm_names:
			arm = ARMS[arm_name]
			dest = scratch / arm_name
			print(f"[{arm_name}] converting...", file=sys.stderr)
			_, convert_s = _time(lambda: arm.convert(source, dest))

			print(f"[{arm_name}] opening...", file=sys.stderr)
			(store, target), open_s = _time(lambda: _open_and_pick(arm, dest))

			base_row = {
				"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
				"dataset": dataset_name,
				"dataset_fingerprint_truncated": fingerprint.get("_truncated", fingerprint.get("_skipped")),
				"arm": arm_name,
				"cache_state": cache,
				"target_recording": target.id,
				"target_shape": [target.n_channels, target.n_samples],
				"target_chunk_shape": list(target.chunk_shape) if target.chunk_shape else None,
				"convert_s": convert_s,
				"open_s": open_s,
				**env,
			}
			sink.write(json.dumps({**base_row, "workload": "_convert", "rep": 0,
									"elapsed_s": convert_s, "checksum": None}) + "\n")
			sink.write(json.dumps({**base_row, "workload": "_open", "rep": 0,
									"elapsed_s": open_s, "checksum": None}) + "\n")

			for workload in WORKLOADS:
				n_warmup = warmup if cache == "warm" else 0
				for rep in range(-n_warmup, reps):
					if cache == "cold":
						store.close()
						_apply_cold_cache(dest)
						store = arm.open(dest)
					try:
						_, elapsed, checksum = run_workload(workload, store, target)
					except Exception as exc:
						print(f"[{arm_name}] {workload} rep {rep} FAILED: {exc!r}", file=sys.stderr)
						continue
					if rep < 0:
						continue  # discarded warm-up
					row = {**base_row, "workload": workload, "rep": rep, "elapsed_s": elapsed,
						   "checksum": checksum, "peak_rss_kb_cumulative": _peak_rss_kb()}
					sink.write(json.dumps(row) + "\n")
				sink.flush()

			store.close()
			print(f"[{arm_name}] done", file=sys.stderr)


def _open_and_pick(arm: Any, dest: Path) -> tuple[ArmStore, RecInfo]:
	store = arm.open(dest)
	target = _pick_target(store.recordings())
	return store, target


def _apply_cold_cache(dest: Path) -> None:
	try:
		drop_caches_local()
	except PermissionError:
		fadvise_dontneed(_files_under(dest))


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

def _median_iqr_min(values: list[float]) -> dict[str, float]:
	arr = np.asarray(values, dtype=np.float64)
	return {
		"median": float(np.median(arr)),
		"iqr_low": float(np.percentile(arr, 25)),
		"iqr_high": float(np.percentile(arr, 75)),
		"min": float(arr.min()),
		"n": len(values),
	}


def bootstrap_ratio_ci(a: list[float], b: list[float], n_resamples: int = 10_000,
						seed: int = 0) -> tuple[float, float, float]:
	"""Bootstrap CI on the ratio of medians (median(a) / median(b)), not a
	p-value -- see the plan's statistical protocol for why."""
	rng = np.random.default_rng(seed)
	a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
	point = float(np.median(a_arr) / np.median(b_arr))
	ratios = np.empty(n_resamples)
	for i in range(n_resamples):
		ra = rng.choice(a_arr, size=len(a_arr), replace=True)
		rb = rng.choice(b_arr, size=len(b_arr), replace=True)
		ratios[i] = np.median(ra) / np.median(rb)
	lo, hi = np.percentile(ratios, [2.5, 97.5])
	return point, float(lo), float(hi)


def summarize(jsonl_path: Path, compare: tuple[str, str] | None) -> None:
	groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
	with jsonl_path.open() as f:
		for line in f:
			row = json.loads(line)
			if row["workload"] in ("_convert", "_open") or row["rep"] < 0:
				continue
			key = (row["dataset"], row["arm"], row["cache_state"], row["workload"])
			groups[key].append(row["elapsed_s"])

	print(f"{'dataset':12} {'arm':14} {'cache':6} {'workload':4} {'n':>3} {'median_s':>12} {'IQR':>24} {'min_s':>12}")
	for key in sorted(groups):
		dataset, arm, cache_state, workload = key
		stats = _median_iqr_min(groups[key])
		iqr = f"[{stats['iqr_low']:.4g}, {stats['iqr_high']:.4g}]"
		print(f"{dataset:12} {arm:14} {cache_state:6} {workload:4} {stats['n']:>3} "
			  f"{stats['median']:>12.4g} {iqr:>24} {stats['min']:>12.4g}")

	if compare:
		arm_a, arm_b = compare
		print(f"\nratio of medians ({arm_a} / {arm_b}), 95% bootstrap CI:")
		for (dataset, arm, cache_state, workload), values in groups.items():
			if arm != arm_a:
				continue
			other_key = (dataset, arm_b, cache_state, workload)
			if other_key not in groups:
				continue
			point, lo, hi = bootstrap_ratio_ci(values, groups[other_key])
			print(f"  {dataset:12} {cache_state:6} {workload:4}: {point:.3f}  [{lo:.3f}, {hi:.3f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("source", help="a BIDS source directory, or (with --summarize) a results .jsonl file")
	parser.add_argument("--summarize", action="store_true", help="summarize an existing results .jsonl instead of running")
	parser.add_argument("--compare", default=None, help="with --summarize: two arm names, comma-separated, for a bootstrap ratio CI")
	parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated arm names")
	parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
	parser.add_argument("--warmup", type=int, default=WARMUP_REPS)
	parser.add_argument("--cache", choices=["warm", "cold"], default="warm")
	parser.add_argument("--out", default="bench_results.jsonl")
	parser.add_argument("--name", default=None, help="dataset label in the output (default: source's name)")
	parser.add_argument("--scratch", default=None, help="where each arm's converted store is written (default: next to --out)")
	parser.add_argument("--no-checksum-source", action="store_true",
						 help="skip SHA-256 fingerprinting the source (slow on 100GB+ datasets)")
	args = parser.parse_args()

	if args.summarize:
		compare = tuple(args.compare.split(",")) if args.compare else None
		summarize(Path(args.source), compare)  # type: ignore[arg-type]
		return

	source = Path(args.source)
	out = Path(args.out)
	scratch = Path(args.scratch) if args.scratch else out.parent / f"{out.stem}_scratch"
	scratch.mkdir(parents=True, exist_ok=True)
	arm_names = args.arms.split(",")
	unknown = [a for a in arm_names if a not in ARMS]
	if unknown:
		parser.error(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")

	run(source, arm_names, args.reps, args.warmup, args.cache, out,
		args.name or source.name, scratch, not args.no_checksum_source)


if __name__ == "__main__":
	main()
