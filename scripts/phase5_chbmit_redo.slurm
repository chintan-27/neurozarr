#!/bin/bash
#SBATCH --job-name=nz-bench-chbmit-redo
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-chbmit-%j.out

# Phase 5h: the CHB-MIT work that phase5_codec_sweep.slurm and
# phase5_ingest_throughput.slurm never actually did. CHB-MIT is a flat EDF dump,
# not BIDS -- both of those pointed a BIDS reader at datasets/chbmit, found zero
# recordings, and "succeeded" (the exit codes they logged were also wrong:
# `$?` was read after a $(date) substitution had already reset it). CHB-MIT
# needs the manifest reader. This script fails loudly instead: a preflight
# refuses to start if the source yields no recordings, and the exit status is
# the real one.
#
#	sbatch phase5_chbmit_redo.slurm ingest    # full 43GB, -j 1, the old "Benchmark 2" row
#	sbatch phase5_chbmit_redo.slurm codec     # 12 lossless configs on the seeded 5GB subsample
#	sbatch phase5_chbmit_redo.slurm harness   # 6-arm W1-W5 + HDF5 fairness on the subsample, fixed reader

set -uo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
cd "$REPO"
mkdir -p .bench_scratch/phase5
PY=.venv/bin/python

FULL_MANIFEST=.bench_scratch/manifest_chbmit.csv
SUB_MANIFEST=.bench_scratch/phase5/manifest_chbmit_subsample.csv

preflight () {
	local n
	n=$($PY - "$1" <<'EOF'
import sys
from neurozarr.items import Recording
from neurozarr.readers import reader_for
print(sum(isinstance(i, Recording) for i in reader_for(sys.argv[1]).read()))
EOF
	)
	echo "preflight: $1 yields $n recordings"
	if [ "${n:-0}" -le 0 ]; then
		echo "preflight FAILED: no recordings -- refusing to run" >&2
		exit 2
	fi
}

MODE="${1:?usage: sbatch phase5_chbmit_redo.slurm <ingest|codec|harness>}"
echo "=== $(date -Is) starting chbmit $MODE ==="
rc=0

case "$MODE" in
ingest)
	preflight "$FULL_MANIFEST"
	rm -rf .bench_scratch/phase5/ingest_chbmit
	/usr/bin/time -v .venv/bin/neurozarr convert "$FULL_MANIFEST" .bench_scratch/phase5/ingest_chbmit -j 1 \
		> .bench_scratch/phase5/ingest_chbmit.log 2>&1
	rc=$?
	;;
codec)
	preflight "$SUB_MANIFEST"
	$PY scripts/benchmark_compressors.py "$SUB_MANIFEST" --reader manifest --name chbmit_subsample \
		> .bench_scratch/phase5/compressors_chbmit_subsample.log 2>&1
	rc=$?
	;;
harness)
	preflight "$SUB_MANIFEST"
	for ARM in raw neurozarr plain_zarr hdf5_mat73 nwb_hdf5 nwb_zarr; do
		echo "=== $(date -Is) chbmit_subsample / $ARM ==="
		$PY scripts/bench/runner.py "$SUB_MANIFEST" \
			--arms "$ARM" --reps 10 --warmup 2 --cache warm \
			--out .bench_scratch/phase5/chbmit_subsample_fixed_warm.jsonl \
			--name chbmit_subsample \
			--scratch .bench_scratch/phase5/chbmit_subsample_fixed_scratch
		s=$?; [ $s -ne 0 ] && rc=$s
	done
	$PY scripts/demo_hdf5_fairness.py "$SUB_MANIFEST" \
		--name chbmit_subsample \
		--out .bench_scratch/phase5/chbmit_subsample_hdf5_fairness_fixed.jsonl \
		--scratch .bench_scratch/phase5/chbmit_subsample_hdf5_fairness_fixed_scratch
	s=$?; [ $s -ne 0 ] && rc=$s
	;;
*)
	echo "unknown mode $MODE" >&2
	exit 64
	;;
esac

echo "=== $(date -Is) finished chbmit $MODE (exit $rc) ==="
exit $rc
