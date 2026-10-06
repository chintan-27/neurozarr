#!/bin/bash
#SBATCH --job-name=nz-bench-hdf5-fairness
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-hdf5fair-%j.out

# Phase 5e: HDF5 fairness controls -- 3 chunk shapes + an int16-packed arm,
# BRAVO/RNS/CHB-MIT-subsample (same hardware as phase5_uniform_small's rows
# for these datasets). Needs scripts/bench/arms.py's Hdf5Arm(chunk_target_bytes=,
# max_chunk_samples=, packed=) and runner.py's run(..., arms=) param synced to
# this checkout first -- both are local-only changes as of this job's writing.

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase5

.venv/bin/python scripts/build_chbmit_subsample.py datasets/chbmit \
	.bench_scratch/phase5/manifest_chbmit_subsample.csv --target-gb 5 --seed 0

declare -A SOURCES=(
	[bravo]="BIDS"
	[rns]="datasets/rns"
	[chbmit_subsample]=".bench_scratch/phase5/manifest_chbmit_subsample.csv"
)

for DATASET in bravo rns chbmit_subsample; do
	echo "=== $(date -Is) starting hdf5 fairness sweep: $DATASET ==="
	.venv/bin/python scripts/demo_hdf5_fairness.py "${SOURCES[$DATASET]}" \
		--name "$DATASET" \
		--out ".bench_scratch/phase5/${DATASET}_hdf5_fairness.jsonl" \
		--scratch ".bench_scratch/phase5/${DATASET}_hdf5_fairness_scratch"
	status=$?
	echo "=== $(date -Is) finished hdf5 fairness sweep: $DATASET (exit $status) ==="
done
