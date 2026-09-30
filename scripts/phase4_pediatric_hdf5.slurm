#!/bin/bash
#SBATCH --job-name=nz-bench-ped-hdf5
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase4/slurm-ped-hdf5-%j.out

# The last missing piece: pediatric_ieeg's hdf5_mat73 arm. It OOM-killed in
# the original job (exit 137, pre-fix) and was never re-run -- the earlier
# recovery job only covered raw+neurozarr. arms.py 0665364 fixes hdf5_mat73's
# own eager-whole-recording-load OOM (confirmed working: hbn_eeg's hdf5_mat73
# succeeded with this same fix in ~1h). Appends to the same results file.

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase4

echo "=== $(date -Is) starting pediatric_ieeg / hdf5_mat73 (recovery 2) ==="
.venv/bin/python scripts/bench/runner.py datasets/pediatric_ieeg \
	--arms hdf5_mat73 --reps 10 --warmup 2 --cache warm \
	--out .bench_scratch/phase4/pediatric_ieeg_warm.jsonl \
	--name pediatric_ieeg \
	--scratch .bench_scratch/phase4/pediatric_ieeg_scratch
status=$?
echo "=== $(date -Is) finished pediatric_ieeg / hdf5_mat73 (recovery 2, exit $status) ==="
