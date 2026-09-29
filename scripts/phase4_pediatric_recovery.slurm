#!/bin/bash
#SBATCH --job-name=nz-bench-recovery
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase4/slurm-recovery-%j.out

# Re-run pediatric_ieeg's raw+neurozarr arms, which job 43746966 OOM-killed
# during discovery (32G) even though neurozarr's own conversion had already
# succeeded -- scripts/bench/arms.py 6bf26a2 fixes the harness itself (one
# live handle at a time instead of one per recording). 64G, not 32G, for
# margin over the pre-fix failure point, but the fix removed the per-subject
# scaling that actually caused the OOM, so this isn't trying to brute-force
# past it with memory alone.
# Appends to the same results file the original job is writing to.

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase4

for ARM in raw neurozarr; do
	echo "=== $(date -Is) starting pediatric_ieeg / $ARM (recovery) ==="
	.venv/bin/python scripts/bench/runner.py datasets/pediatric_ieeg \
		--arms "$ARM" --reps 10 --warmup 2 --cache warm \
		--out .bench_scratch/phase4/pediatric_ieeg_warm.jsonl \
		--name pediatric_ieeg \
		--scratch .bench_scratch/phase4/pediatric_ieeg_scratch
	status=$?
	echo "=== $(date -Is) finished pediatric_ieeg / $ARM (recovery, exit $status) ==="
done
