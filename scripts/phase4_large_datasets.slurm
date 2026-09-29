#!/bin/bash
#SBATCH --job-name=nz-bench-large
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase4/slurm-%j.out

# Phase 4: raw/neurozarr/hdf5_mat73 on Pediatric iEEG and HBN EEG at full size.
# Single-threaded per the bench harness (no -j parallelism in scripts/bench) --
# CPU count here is headroom, not something the harness itself uses in parallel.
# Runs arm-by-arm so a slow codec's real cost never risks losing progress on
# an already-finished arm (Repo.create()/add_recording(existing="error") both
# refuse to touch a store a previous attempt already wrote into).

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase4

for DATASET in pediatric_ieeg hbn_eeg; do
	for ARM in raw neurozarr hdf5_mat73; do
		echo "=== $(date -Is) starting $DATASET / $ARM ==="
		.venv/bin/python scripts/bench/runner.py "datasets/$DATASET" \
			--arms "$ARM" --reps 10 --warmup 2 --cache warm \
			--out ".bench_scratch/phase4/${DATASET}_warm.jsonl" \
			--name "$DATASET" \
			--scratch ".bench_scratch/phase4/${DATASET}_scratch"
		status=$?
		echo "=== $(date -Is) finished $DATASET / $ARM (exit $status) ==="
	done
done

echo "=== $(date -Is) all datasets/arms done ==="
