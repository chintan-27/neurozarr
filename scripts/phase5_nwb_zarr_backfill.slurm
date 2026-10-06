#!/bin/bash
#SBATCH --job-name=nz-bench-nwbzarr-backfill
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-nwbzarr-backfill-%j.out

# Backfills bravo/nwb_zarr, the one leg of phase5_uniform_small.slurm (job
# 44843891) that failed before arms.py's ZarrDataIO fix was synced -- appends
# to the same bravo_warm.jsonl the rest of that job is writing to (runner.py
# opens --out in append mode), so this doesn't duplicate or conflict with it.
# rns/nwb_zarr and chbmit_subsample/nwb_zarr weren't reached yet when the fix
# landed, so they don't need backfilling -- only bravo does.

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase5

echo "=== $(date -Is) starting bravo / nwb_zarr (backfill) ==="
.venv/bin/python scripts/bench/runner.py BIDS \
	--arms nwb_zarr --reps 10 --warmup 2 --cache warm \
	--out .bench_scratch/phase5/bravo_warm.jsonl \
	--name bravo \
	--scratch .bench_scratch/phase5/bravo_scratch
status=$?
echo "=== $(date -Is) finished bravo / nwb_zarr (backfill, exit $status) ==="
