#!/bin/bash
#SBATCH --job-name=nz-bench-cold-check
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-cold-check-%j.out

# Phase 5k: does the cold-cache mechanism used in phase5d (posix_fadvise
# DONTNEED) actually produce a cold read on this filesystem? Reads the largest
# file of each dataset three ways with page-cache residency alongside; see
# scripts/check_cold_cache_effect.py for how to read the result.

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase5/cold_check
rc=0
for D in hbn_eeg pediatric_ieeg; do
	echo "=== $(date -Is) $D on $(hostname) ==="
	.venv/bin/python scripts/check_cold_cache_effect.py "datasets/$D" --out ".bench_scratch/phase5/cold_check/${D}.json"
	s=$?; [ $s -ne 0 ] && rc=$s
done
which fincore || echo "fincore not available: residency will be null"
exit $rc
