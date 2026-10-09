"""Phase 6 tables from the pulled results: ingest/size vs Phase 5, read medians vs Phase 4/5,
metadata-query latency across implementations, synthetic scaling.

	python scripts/summarize_phase6.py [.bench_scratch]
"""

import collections
import glob
import json
import re
import statistics
import sys

B = sys.argv[1] if len(sys.argv) > 1 else ".bench_scratch"
P6 = f"{B}/phase6_pulled/phase6"
DATASETS = ["bravo", "rns", "chbmit", "hbn_eeg", "pediatric_ieeg"]
# Phase 5 neurozarr (zstd-19): store bytes and add+commit seconds, from verify_outputs and ingest logs
OLD_BYTES = {"bravo": 139503362, "rns": 249461061, "chbmit": 21115215261, "hbn_eeg": 73838449238, "pediatric_ieeg": 73857887821}
OLD_INGEST_S = {"bravo": 136.8, "rns": 431.8, "chbmit": (17380 + 18805) / 2, "hbn_eeg": 2798.9, "pediatric_ieeg": 4250.0}


def seconds(clock: str) -> float:
	parts = [float(p) for p in clock.split(":")]
	return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def ingest() -> None:
	print("## Ingest and size (Phase 6 codec plans vs Phase 5 zstd-19)")
	print("dataset          add_s r1/r2      commit_s r1/r2   total vs old      bytes vs old          index bytes")
	for ds in DATASETS:
		runs = []
		for rep in (1, 2):
			text = open(f"{P6}/ingest_{ds}_r{rep}.log").read()
			add, commit = [seconds(m) for m in re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)]
			size = int(re.search(r"apparent_bytes (\d+)", text).group(1))
			index = int(re.search(r"index_bytes (\d+)", text).group(1))
			runs.append((add, commit, size, index))
		total = statistics.mean(a + c for a, c, _, _ in runs)
		size = runs[0][2]
		print(f"{ds:15s} {runs[0][0]:6.0f}/{runs[1][0]:<6.0f}   {runs[0][1]:6.1f}/{runs[1][1]:<6.1f}   "
			  f"{total:7.0f}s x{OLD_INGEST_S[ds] / total:5.1f}   {size / 1e9:8.3f} GB {size / OLD_BYTES[ds] - 1:+6.1%}   {runs[0][3]:>9,}")


def reads() -> None:
	print("\n## Reads, neurozarr arm, median ms (old = Phase 5 for small sets, Phase 4 for HBN/Pediatric)")
	old_files = {"bravo": "phase5_pulled/bravo_warm.jsonl", "rns": "phase5_pulled/rns_warm.jsonl",
				 "chbmit_subsample": "phase5_pulled/chbmit_subsample_fixed_warm.jsonl",
				 "hbn_eeg": "phase4/hbn_eeg_warm.jsonl", "pediatric_ieeg": "phase4/pediatric_ieeg_warm.jsonl"}
	for ds, old in old_files.items():
		def load(path: str) -> dict[str, list[float]]:
			out = collections.defaultdict(list)
			for line in open(path):
				r = json.loads(line)
				if r["arm"] == "neurozarr" and not r["workload"].startswith("_"):
					out[r["workload"]].append(r["elapsed_s"] * 1000)
			return out
		o, n = load(f"{B}/{old}"), load(f"{P6}/{ds}_reads.jsonl")
		cells = [f"{w} {statistics.median(o[w]):.1f}->{statistics.median(n[w]):.1f} (min {min(n[w]):.1f})" for w in sorted(n)]
		print(f"{ds:17s} " + " | ".join(cells))


def queries() -> None:
	print("\n## Metadata queries, median ms (cold / warm); latest run per dataset+impl")
	rows = [json.loads(line) for line in open(f"{P6}/query.jsonl")]
	groups = collections.defaultdict(list)
	for r in rows:
		# the old arm's first run (10-08) used the new CLI by mistake; its 10-09 rerun supersedes it
		if not (r["impl"] == "old" and r["utc"] < "2026-10-09T12"):
			groups[(r["dataset"], r["query"], r["impl"], r["cache"])].append(r)
	for ds in sorted({k[0] for k in groups}):
		print(ds)
		for query in sorted({k[1] for k in groups if k[0] == ds}):
			cells = []
			for impl in ("new", "old", "pybids", "bids2table"):
				cold, warm = groups.get((ds, query, impl, "cold")), groups.get((ds, query, impl, "warm"))
				if cold:
					w = f"{statistics.median(r['seconds'] for r in warm) * 1000:.2f}" if warm else "-"
					cells.append(f"{impl} {statistics.median(r['seconds'] for r in cold) * 1000:.2f}/{w} n={cold[0]['n']}")
			print(f"   {query:18s} " + " | ".join(cells))


def synthetic() -> None:
	print("\n## Synthetic index scaling (prune + mask, ms)")
	for line in open(f"{P6}/synthetic.jsonl"):
		r = json.loads(line)
		if r["query"] == "_build":
			print(f"{r['items']:>9,} items: summary build {r['summary_s'] * 1000:.0f} ms, index {r['index_bytes'] / 1e6:.1f} MB, summary {r['summary_bytes'] / 1e3:.0f} kB")
		else:
			print(f"{'':12s}{r['query']:12s} prune {r['prune_ms']:7.2f}  mask {r['mask_ms']:8.2f}  kept {r['subjects_kept']}/{r['n_subjects']}  matches {r['matches']}")


if __name__ == "__main__":
	ingest()
	reads()
	queries()
	synthetic()
	print("\n(paths:", ", ".join(sorted(glob.glob(f"{P6}/*.jsonl"))), ")")
