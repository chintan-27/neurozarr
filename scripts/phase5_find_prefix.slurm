#!/bin/bash
#SBATCH --job-name=nz-bench-find-prefix
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-find-prefix-%j.out

# Phase 5g: the pre-catalog-fix find() baseline (RNS, W4), on HiPerGator so it
# shares hardware with the post-fix RNS numbers from phase5_uniform_small.slurm
# -- the earlier local run (64.18s median) predates the "all timing on one
# machine" rule and stays logged only as a supplementary local data point.
#
# Runs the harness against a checkout of the commit just before the catalog
# fix (ce65c29, synced to ../nz-prefix); PYTHONPATH puts that checkout's
# neurozarr package ahead of the cluster repo's editable install, and the
# first thing logged is where neurozarr was actually imported from.

# Optional replicate number (sbatch phase5_find_prefix.slurm 1): writes to its
# own output file so replicates on different nodes stay separate rows.
REP="${1:-}"
SUFFIX="${REP:+_r$REP}"

set -uo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
PREFIX=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/nz-prefix
cd "$REPO"
mkdir -p .bench_scratch/phase5
export PYTHONPATH="$PREFIX"

PY="$REPO/.venv/bin/python"
# Checked from a neutral cwd: from $REPO, '' on sys.path finds the cluster
# repo's neurozarr first and the check would pass or fail regardless of
# whether PYTHONPATH actually beats the editable install (the first version of
# this job logged exactly that misleading path). Refuse to run if the pre-fix
# checkout isn't what gets imported.
WHERE=$(cd /tmp && $PY -c 'import neurozarr; print(neurozarr.__file__)')
echo "neurozarr imported from: $WHERE"
case "$WHERE" in
	"$PREFIX"/*) ;;
	*) echo "WRONG neurozarr (expected under $PREFIX) -- refusing to run" >&2; exit 3 ;;
esac

echo "=== $(date -Is) starting pre-fix find() baseline: rns / neurozarr ==="
$PY "$PREFIX/scripts/bench/runner.py" datasets/rns \
	--arms neurozarr --reps 10 --warmup 2 --cache warm \
	--out .bench_scratch/phase5/rns_prefix_find_verified${SUFFIX}.jsonl \
	--name RNS \
	--scratch .bench_scratch/phase5/rns_prefix_verified${SUFFIX}_scratch
status=$?
echo "=== $(date -Is) finished pre-fix find() baseline (exit $status) ==="
exit $status
