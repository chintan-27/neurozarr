#!/bin/bash
#SBATCH --job-name=nz-p6-reads
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase6/slurm-reads-%j.out

# Phase 6 reads, rerun one job at a time: the first pass ran beside 17 other jobs of ours
# on the same Lustre and nodes (load up to 300; Pediatric W1 8-39 s). --dependency=singleton
# (one nz-p6-reads job at a time) serializes them; the rep-1 stores are reused, not rebuilt.
#
#	sbatch phase6_reads.slurm <bravo|rns|chbmit_subsample|hbn_eeg|pediatric_ieeg>

set -uo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
cd "$REPO"
PY=$REPO/.venv/bin/python
P6=.bench_scratch/phase6
DATASET="${1:?usage: sbatch phase6_reads.slurm <dataset>}"
case "$DATASET" in
	bravo)            SOURCE=BIDS ;;
	rns)              SOURCE=datasets/rns ;;
	chbmit_subsample) SOURCE=.bench_scratch/phase5/manifest_chbmit_subsample.csv ;;
	hbn_eeg)          SOURCE=datasets/hbn_eeg ;;
	pediatric_ieeg)   SOURCE=datasets/pediatric_ieeg ;;
	*) echo "unknown dataset $DATASET" >&2; exit 64 ;;
esac
STORE="$P6/${DATASET}_scratch/neurozarr"
[ -d "$STORE" ] || { echo "no store at $STORE" >&2; exit 2; }
echo "=== $(date -Is) serial reads $DATASET; node=$(hostname) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
$PY scripts/bench/runner.py "$SOURCE" --arms neurozarr --reps 10 --warmup 2 --cache warm \
	--reuse-existing --out "$P6/${DATASET}_reads_serial.jsonl" --name "$DATASET" --scratch "$P6/${DATASET}_scratch"
rc=$?
echo "=== $(date -Is) finished serial reads $DATASET (exit $rc); loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
exit $rc
