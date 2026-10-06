"""Every Phase 5 number in the paper, computed from the saved result files.

Reads the JSONL/log files pulled from HiPerGator (see RESULTS below) and prints
markdown tables: per-arm medians with the min-max across independent
replicates, between-arm ratios per replicate, apparent store sizes against
apparent source sizes, ingest throughput, and the pre/post-fix find() speedup.
Nothing here depends on a printed summary from a run.

	python scripts/summarize_phase5.py > phase5_summary.md
"""

import glob
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / ".bench_scratch"
ARMS = ["raw", "neurozarr", "plain_zarr", "hdf5_mat73", "nwb_hdf5", "nwb_zarr"]
WORKLOADS = ["W1", "W2", "W3", "W4", "W5"]

# Apparent bytes of each source (du -sb --apparent-size -L), computed once from
# the datasets on disk. chbmit_subsample is the sum of its manifest's files.
SOURCE_BYTES = {
	"bravo": 260_635_851, "rns": 521_942_493, "chbmit_subsample": 6_337_531_648,
	"chbmit": 45_714_707_330, "hbn_eeg": 110_708_824_369, "pediatric_ieeg": 176_779_904_875,
}

# Independent harness runs per dataset: the original phase5 run, then the replicates.
HARNESS = {
	"bravo": ["phase5_pulled/bravo_warm.jsonl", "phase5_pulled2/rep/bravo_r*_harness.jsonl"],
	"rns": ["phase5_pulled/rns_warm.jsonl", "phase5_pulled2/rep/rns_r*_harness.jsonl"],
	"chbmit_subsample": ["phase5_pulled/chbmit_subsample_fixed_warm.jsonl",
						  "phase5_pulled2/rep/chbmit_subsample_r*_harness.jsonl"],
}


def _files(patterns: list[str]) -> list[Path]:
	return [Path(p) for pat in patterns for p in sorted(glob.glob(str(ROOT / pat)))]


def _load(path: Path) -> list[dict]:
	return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run_stats(rows: list[dict]) -> dict[tuple[str, str], float]:
	"""(arm, workload) -> median seconds for one run; ('convert') from the _convert row."""
	out: dict[tuple[str, str], list[float]] = {}
	for r in rows:
		if r["workload"] == "_convert":
			out[(r["arm"], "convert")] = [r["elapsed_s"]]
		elif r["workload"] != "_open" and r["rep"] >= 0:
			out.setdefault((r["arm"], r["workload"]), []).append(r["elapsed_s"])
	return {k: statistics.median(v) for k, v in out.items()}


def _fmt(vals: list[float], scale: float, unit: str) -> str:
	if len(vals) == 1:
		return f"{vals[0] * scale:.1f}{unit}"
	return f"{statistics.median(vals) * scale:.1f}{unit} [{min(vals) * scale:.1f}-{max(vals) * scale:.1f}]"


def harness_tables() -> None:
	for dataset, patterns in HARNESS.items():
		runs = [_run_stats(_load(f)) for f in _files(patterns) if _load(f)]
		if not runs:
			continue
		print(f"\n### {dataset}: median across {len(runs)} independent runs [min-max]\n")
		print("| arm | convert (s) | " + " | ".join(f"{w} (ms)" for w in WORKLOADS) + " |")
		print("|---|---|" + "---|" * len(WORKLOADS))
		for arm in ARMS:
			cells = []
			for key, scale, unit in [("convert", 1, "")] + [(w, 1000, "") for w in WORKLOADS]:
				vals = [run[(arm, key)] for run in runs if (arm, key) in run]
				cells.append(_fmt(vals, scale, unit) if vals else "-")
			print(f"| {arm} | " + " | ".join(cells) + " |")
		print(f"\nRatio to neurozarr, per run, min-max over {len(runs)} runs (<1 means faster than neurozarr):\n")
		print("| arm | " + " | ".join(["convert"] + WORKLOADS) + " |")
		print("|---|" + "---|" * (len(WORKLOADS) + 1))
		for arm in ARMS:
			if arm == "neurozarr":
				continue
			cells = []
			for key in ["convert"] + WORKLOADS:
				rs = [run[(arm, key)] / run[("neurozarr", key)] for run in runs
					  if (arm, key) in run and run.get(("neurozarr", key))]
				cells.append(f"{min(rs):.3g}-{max(rs):.3g}" if rs else "-")
			print(f"| {arm} | " + " | ".join(cells) + " |")


def size_table() -> None:
	tsv = ROOT / "phase5_pulled2/verify_outputs/store_apparent_bytes.tsv"
	if not tsv.exists():
		return
	sizes = {}
	for line in tsv.read_text().splitlines():
		b, path = line.split("\t")
		sizes[path.replace(".bench_scratch/", "").rstrip("/")] = int(b)
	where = {
		"bravo": ("bravo", "phase5/bravo_scratch/{arm}"),
		"rns": ("rns", "phase5/rns_scratch/{arm}"),
		"chbmit_subsample": ("chbmit_subsample", "phase5/chbmit_subsample_fixed_scratch/{arm}"),
		"hbn_eeg": ("hbn_eeg", None), "pediatric_ieeg": ("pediatric_ieeg", None),
	}
	special = {
		"hbn_eeg": {"neurozarr": "phase4/hbn_eeg_scratch/neurozarr", "hdf5_mat73": "phase4/hbn_eeg_scratch/hdf5_mat73",
					"nwb_hdf5": "phase5/hbn_eeg_nwb_hdf5_scratch/nwb_hdf5", "nwb_zarr": "phase5/hbn_eeg_nwb_zarr_scratch/nwb_zarr"},
		"pediatric_ieeg": {"neurozarr": "phase4/pediatric_ieeg_scratch/neurozarr",
							"hdf5_mat73": "phase4/pediatric_ieeg_scratch/hdf5_mat73"},
	}
	extra = {"bravo": {"hdf5_packed_int16": "phase5/bravo_hdf5_fairness_scratch/hdf5_packed_int16"},
			 "rns": {"hdf5_packed_int16": "phase5/rns_hdf5_fairness_scratch/hdf5_packed_int16"},
			 "chbmit_subsample": {"hdf5_packed_int16": "phase5/chbmit_subsample_hdf5_fairness_fixed_scratch/hdf5_packed_int16"}}
	cols = ["neurozarr", "plain_zarr", "hdf5_mat73", "hdf5_packed_int16", "nwb_hdf5", "nwb_zarr"]
	print("\n### Stored size, apparent bytes (MiB), and as % of source\n")
	print("| dataset | source | " + " | ".join(cols) + " |")
	print("|---|---|" + "---|" * len(cols))
	for ds, (_, pat) in where.items():
		cells = []
		for c in cols:
			p = special.get(ds, {}).get(c) or extra.get(ds, {}).get(c) or (pat.format(arm=c) if pat else None)
			b = sizes.get(p) if p else None
			cells.append(f"{b / 2**20:,.0f} ({100 * b / SOURCE_BYTES[ds]:.0f}%)" if b else "-")
		print(f"| {ds} | {SOURCE_BYTES[ds] / 2**20:,.0f} | " + " | ".join(cells) + " |")


def _ingest_seconds(path: Path) -> float | None:
	m = re.search(r"Elapsed \(wall clock\) time.*?: ([\d:.]+)", path.read_text(errors="ignore"))
	if not m:
		return None
	parts = [float(x) for x in m.group(1).split(":")]
	return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def ingest_table() -> None:
	print("\n### Ingest throughput (apparent source MB/s; -j 1 for bravo/chbmit, -j 16 for hbn/pediatric)\n")
	print("| dataset | runs (s) | MB/s (median) |")
	print("|---|---|---|")
	for ds in ["bravo", "chbmit", "hbn_eeg", "pediatric_ieeg"]:
		files = _files([f"phase5_pulled/ingest_{ds}.log", f"phase5_pulled2/rep/ingest_{ds}_r*.log"])
		secs = [s for f in files if (s := _ingest_seconds(f)) is not None]
		if secs:
			rate = statistics.median(SOURCE_BYTES[ds] / s / 1e6 for s in secs)
			print(f"| {ds} | {', '.join(f'{s:.0f}' for s in secs)} | {rate:.1f} |")


def find_table() -> None:
	pre = [r["elapsed_s"] for f in _files(["phase5_pulled2/rns_prefix_find_verified*.jsonl"])
		   for r in _load(f) if r["workload"] == "W4" and r["rep"] >= 0]
	by_run = []
	for f in _files(HARNESS["rns"]):
		w = [r["elapsed_s"] for r in _load(f) if r["arm"] == "neurozarr" and r["workload"] == "W4" and r["rep"] >= 0]
		if w:
			by_run.append(statistics.median(w))
	if pre and by_run:
		print("\n### find() on RNS: pre-fix vs post-fix (same cluster)\n")
		print(f"pre-fix {statistics.median(pre):.1f}s (n={len(pre)} reps over "
			  f"{len(_files(['phase5_pulled2/rns_prefix_find_verified*.jsonl']))} runs); "
			  f"post-fix {statistics.median(by_run):.2f}s (median of {len(by_run)} runs); "
			  f"speedup {statistics.median(pre) / statistics.median(by_run):.1f}x")


if __name__ == "__main__":
	harness_tables()
	size_table()
	ingest_table()
	find_table()
