#!/bin/bash
#SBATCH --job-name=nz-p8-scan
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase8/slurm-%j.out

# Phase 8: neurozarr reads before vs after the read-path changes (one-pass int16 -> float64,
# background prefetch of the next chunk, no whole-chunk copies for windows straddling two),
# in ONE job on one node: the previous code first (../nz-head, a `git archive` of the
# commit before, imported with python -P so this checkout's code can't shadow it), then
# this checkout. Reuses the Phase 6 neurozarr store.
#
#	sbatch phase8_scan_ab.slurm <bravo|rns|chbmit_subsample|hbn_eeg|pediatric_ieeg>

set -uo pipefail
ROOT=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark
REPO=$ROOT/neurozarr
cd "$REPO"
PY=$REPO/.venv/bin/python
P8=.bench_scratch/phase8
DATASET="${1:?usage: sbatch phase8_scan_ab.slurm <dataset>}"
case "$DATASET" in
	bravo)            SOURCE=BIDS ;;
	rns)              SOURCE=datasets/rns ;;
	chbmit_subsample) SOURCE=.bench_scratch/phase5/manifest_chbmit_subsample.csv ;;
	hbn_eeg|pediatric_ieeg) SOURCE=datasets/$DATASET ;;
	*) echo "unknown dataset $DATASET" >&2; exit 64 ;;
esac
STORE=.bench_scratch/phase6/${DATASET}_scratch/neurozarr
[ -d "$STORE" ] || { echo "MISSING $STORE" >&2; exit 2; }
SCRATCH=$P8/${DATASET}_scratch
mkdir -p "$SCRATCH"
ln -sfn "$REPO/$STORE" "$SCRATCH/neurozarr"
export XDG_STATE_HOME="$REPO/$P8/state_$SLURM_JOB_ID"
rc=0
for code in before after; do
	if [ $code = before ]; then OLDPATH="$ROOT/nz-head"; else OLDPATH=""; fi
	WHERE=$(PYTHONPATH="$OLDPATH" $PY -P -c 'import neurozarr; print(neurozarr.__file__)')
	echo "=== $(date -Is) $DATASET $code: neurozarr from $WHERE; node=$(hostname) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
	case "$code:$WHERE" in before:"$ROOT/nz-head"/*|after:"$REPO"/*) ;; *) echo "WRONG neurozarr for $code" >&2; exit 3 ;; esac
	PYTHONPATH="$OLDPATH" $PY -P scripts/bench/runner.py "$SOURCE" --arms neurozarr --reps 10 --warmup 2 --cache warm \
		--reuse-existing --no-checksum-source --out "$P8/${DATASET}_${code}.jsonl" --name "$DATASET" --scratch "$SCRATCH"
	s=$?; [ $s -ne 0 ] && rc=$s
done
rm -rf "$XDG_STATE_HOME"
echo "=== $(date -Is) finished $DATASET (exit $rc) ==="
exit $rc
