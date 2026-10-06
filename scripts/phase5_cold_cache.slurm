#!/bin/bash
#SBATCH --job-name=nz-bench-cold-cache
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-cold-%j.out

# Phase 5d: cold-cache measurements for the two datasets that only exist on
# HiPerGator, one submission per dataset with a comma-separated arm list:
#
#	sbatch phase5_cold_cache.slurm <hbn_eeg|pediatric_ieeg> <arm[,arm...]>
#
# One job per dataset rather than one per (dataset, arm): cold reads are
# I/O-bound, so several of these hammering the same shared filesystem at once
# would put contention noise into exactly the numbers that are most sensitive
# to it. Arms still run as separate runner.py invocations inside the job, so a
# failure in one doesn't cost the others' already-written rows.
#
# Read-only: raw/neurozarr/hdf5_mat73 stores for both datasets were already
# built by the original Phase 4 campaign (.bench_scratch/phase4/*_scratch) and
# never needed re-converting; --reuse-existing points runner.py at those stores
# directly (the first version of this job burned 6.5+ hours single-threaded
# re-converting hbn_eeg/neurozarr before that was caught). Uses fadvise (not
# drop_caches -- no root on a shared cluster node), per runner.py's existing
# --cache cold support. The fadvise-vs-drop_caches agreement check is a
# separate, one-time local step, not part of this job.

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase5

USAGE="usage: sbatch phase5_cold_cache.slurm <hbn_eeg|pediatric_ieeg> <arm[,arm...]>"
DATASET="${1:?$USAGE}"
ARMS="${2:?$USAGE}"

for ARM in ${ARMS//,/ }; do
	echo "=== $(date -Is) starting cold-cache $DATASET / $ARM ==="
	.venv/bin/python scripts/bench/runner.py "datasets/$DATASET" \
		--arms "$ARM" --reps 10 --warmup 0 --cache cold --reuse-existing \
		--out ".bench_scratch/phase5/${DATASET}_cold.jsonl" \
		--name "$DATASET" \
		--scratch ".bench_scratch/phase4/${DATASET}_scratch"
	status=$?
	echo "=== $(date -Is) finished cold-cache $DATASET / $ARM (exit $status) ==="
done
