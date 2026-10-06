#!/bin/bash
#SBATCH --job-name=nz-bench-uniform-small
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-uniform-%j.out

# Phase 5a: re-run the cross-format W1-W5 harness for BRAVO/RNS/CHB-MIT-subsample
# on HiPerGator. These three currently only exist as local ("fedora") numbers
# in .bench_scratch/phase4/all_warm.jsonl -- HBN EEG/Pediatric iEEG are already
# cluster-only, so this is what makes every timing claim in the paper share
# hardware, not just the two largest datasets. Separate --out from phase4's so
# the old local numbers aren't silently overwritten; the paper cites phase5.

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase5

# Same seed (0) as the original local subsample -- regenerated here rather
# than copied, since the saved manifest's path column is hardcoded to the
# local machine's absolute paths and won't resolve on this filesystem.
.venv/bin/python scripts/build_chbmit_subsample.py datasets/chbmit \
	.bench_scratch/phase5/manifest_chbmit_subsample.csv --target-gb 5 --seed 0

declare -A SOURCES=(
	[bravo]="BIDS"
	[rns]="datasets/rns"
	[chbmit_subsample]=".bench_scratch/phase5/manifest_chbmit_subsample.csv"
)

for DATASET in bravo rns chbmit_subsample; do
	for ARM in raw neurozarr plain_zarr hdf5_mat73 nwb_hdf5 nwb_zarr; do
		echo "=== $(date -Is) starting $DATASET / $ARM ==="
		.venv/bin/python scripts/bench/runner.py "${SOURCES[$DATASET]}" \
			--arms "$ARM" --reps 10 --warmup 2 --cache warm \
			--out ".bench_scratch/phase5/${DATASET}_warm.jsonl" \
			--name "$DATASET" \
			--scratch ".bench_scratch/phase5/${DATASET}_scratch"
		status=$?
		echo "=== $(date -Is) finished $DATASET / $ARM (exit $status) ==="
	done
done
