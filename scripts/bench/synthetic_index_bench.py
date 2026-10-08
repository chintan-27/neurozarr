"""Scaling of the index's query engine, without data files: synthetic subject indexes
of N subjects x M items, shaped like a real store (sessions, datatypes, tasks, runs,
sampling rates, durations, a few sidecar fields), queried at 10^4 .. 10^6 items.

	python scripts/bench/synthetic_index_bench.py --out synthetic.jsonl

Reports per size: summary build time, index bytes, and per query the prune time
(subjects kept), the mask time over the kept subjects' indexes (already decoded, as
when cached) and the matches. Each query is also checked against a brute-force filter.
"""

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

from neurozarr.index import Summary, Table, build_summary, digest, encode, stats_of
from neurozarr.query import columns_of, mask, parse, prune, subject_values

QUERIES = {
	"one_subject": "sub == 'sub-00042'",
	"rare_rate": "sfreq >= 4000",                         # one subject in 500 records at 4 kHz
	"task": "task == 'nback'",
	"combined": "datatype == 'ieeg' and duration_s > 600 and participant.age < 30",
	"meta": "meta.Manufacturer == 'Medtronic'",
	"everything": None,
}


def make(n_subjects: int, items: int, seed: int = 0) -> dict[str, bytes]:
	rng = np.random.default_rng(seed)
	tasks = np.array(["rest", "nback", "stroop", "sleep", "motor", "auditory", "visual", "gonogo"])
	blobs = {}
	for s in range(n_subjects):
		sub = f"sub-{s:05d}"
		rare = s % 500 == 0
		sfreq = 4096.0 if rare else float(rng.choice([250, 256, 500, 512, 1000, 2048]))
		datatype = "ieeg" if s % 3 == 0 else "eeg"
		maker = "Medtronic" if s % 7 == 0 else "Natus"
		rows = []
		for i in range(items):
			n_samples = int(rng.integers(10_000, 2_000_000))
			rows.append({"path": f"{sub}/ses-{i % 4 + 1}/{datatype}/task-{tasks[i % len(tasks)]}_run-{i // len(tasks) + 1}",
						 "kind": "recording", "sub": sub, "ses": f"ses-{i % 4 + 1}", "datatype": datatype,
						 "task": str(tasks[i % len(tasks)]), "run": str(i // len(tasks) + 1), "sfreq": sfreq,
						 "n_channels": float(rng.integers(2, 256)), "n_samples": float(n_samples),
						 "duration_s": n_samples / sfreq, "meta.Manufacturer": maker})
		blobs[sub] = encode(rows, {"sub": sub, "snapshot": "S", "sessions": [f"ses-{k}" for k in range(1, 5)],
								   "attrs": {"age": int(rng.integers(5, 80))}, "session_attrs": {}})
	return blobs


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--out", required=True)
	parser.add_argument("--sizes", default="100x100,1000x100,10000x100")  # subjects x items
	parser.add_argument("--reps", type=int, default=7)
	args = parser.parse_args()
	with open(args.out, "a") as out:
		for size in args.sizes.split(","):
			n_subjects, items = map(int, size.split("x"))
			start = time.perf_counter()
			blobs = make(n_subjects, items)
			made = time.perf_counter() - start
			tables = {sub: Table.decode(blob) for sub, blob in blobs.items()}
			start = time.perf_counter()
			summary_blob = build_summary([stats_of(tables[s], digest(b)) for s, b in blobs.items()])
			summary_s = time.perf_counter() - start
			summary = Summary(Table.decode(summary_blob))
			base = {"n_subjects": n_subjects, "items": n_subjects * items, "node": platform.node(),
					"loadavg": os.getloadavg()[0]}
			out.write(json.dumps({**base, "query": "_build", "make_s": made, "summary_s": summary_s,
								  "index_bytes": sum(map(len, blobs.values())), "summary_bytes": len(summary_blob)}) + "\n")
			for name, text in QUERIES.items():
				expr = parse(text)
				prune_t, mask_t, kept, hits = [], [], 0, 0
				for _ in range(args.reps):
					start = time.perf_counter()
					keep = prune(expr, summary)
					prune_t.append(time.perf_counter() - start)
					start = time.perf_counter()
					kept = int(keep.sum())
					hits = 0
					for i in np.nonzero(keep)[0]:
						sub = summary.subjects[i]
						hits += int(mask(expr, tables[sub], subject_values(summary, int(i), columns_of(expr))).sum())
					mask_t.append(time.perf_counter() - start)
				brute = sum(int(mask(expr, tables[s], summary.table.row(i)).sum()) for i, s in enumerate(summary.subjects))  # all values
				assert hits == brute, (name, hits, brute)
				row = {**base, "query": name, "prune_ms": statistics.median(prune_t) * 1000,
					   "mask_ms": statistics.median(mask_t) * 1000, "subjects_kept": kept, "matches": hits}
				out.write(json.dumps(row) + "\n")
				print(f"{n_subjects * items:>9} items  {name:12s} prune {row['prune_ms']:8.2f} ms  "
					  f"mask {row['mask_ms']:9.2f} ms  kept {kept:6d}/{n_subjects}  matches {hits}", flush=True)
			out.flush()


if __name__ == "__main__":
	main()
