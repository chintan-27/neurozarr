"""HDF5 fairness controls: a 3-chunk-shape sweep plus an int16-packed arm, so
the cross-format table's HDF5 column isn't judged on one arbitrary default --
and HDF5 gets the same int16+per-channel-calibration domain assumption
neurozarr gets by default, instead of a packed-vs-unpacked comparison being
mistaken for a format difference.

Timing-based -- run on the same hardware as whatever cross-format table row
this is meant to sit next to (matches BRAVO/RNS/CHB-MIT-subsample on
HiPerGator per the paper's "equal hardware for speed claims" rule).

	.venv/bin/python scripts/demo_hdf5_fairness.py datasets/chbmit_subsample \\
		--name CHB-MIT-subsample --out .bench_scratch/paper_logs/hdf5_fairness.jsonl \\
		--scratch .bench_scratch/paper_logs/hdf5_fairness_scratch
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.arms import Hdf5Arm
from bench.runner import run

# Three chunk targets around the default (8MB/65536 samples): half, default,
# double -- enough to show whether HDF5's own number is chunk-shape-sensitive
# without an unbounded sweep.
CHUNK_VARIANTS = {
	"hdf5_chunk_4mb": Hdf5Arm(chunk_target_bytes=4 * 1024 * 1024, max_chunk_samples=32768),
	"hdf5_chunk_8mb_default": Hdf5Arm(chunk_target_bytes=8 * 1024 * 1024, max_chunk_samples=65536),
	"hdf5_chunk_16mb": Hdf5Arm(chunk_target_bytes=16 * 1024 * 1024, max_chunk_samples=131072),
	"hdf5_packed_int16": Hdf5Arm(packed=True),
}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("source")
	parser.add_argument("--name", required=True)
	parser.add_argument("--out", type=Path, required=True)
	parser.add_argument("--scratch", type=Path, required=True)
	parser.add_argument("--reps", type=int, default=10)
	parser.add_argument("--warmup", type=int, default=2)
	args = parser.parse_args()

	args.out.parent.mkdir(parents=True, exist_ok=True)
	run(Path(args.source), list(CHUNK_VARIANTS), args.reps, args.warmup, "warm",
		args.out, args.name, args.scratch, checksum_source=True, arms=CHUNK_VARIANTS)
	print(f"done -- {args.out}")


if __name__ == "__main__":
	main()
