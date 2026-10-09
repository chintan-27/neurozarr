#!/bin/bash
#SBATCH --job-name=nz-p6-ingest
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase6/slurm-ingest-%j.out

# Phase 6: neurozarr with its current code -- codecs planned by the reader (Blosc zstd-5
# + probed shuffle, instead of fixed zstd-19) and the columnar index built at commit.
# Every neurozarr size / ingest / read number in the paper predates both, so this
# re-measures them on the same cluster the earlier numbers came from.
#
#	sbatch phase6_ingest.slurm <bravo|rns|chbmit> <rep>
#	sbatch --cpus-per-task=16 --mem=64G phase6_ingest.slurm <hbn_eeg|pediatric_ieeg> <rep>
#
# init -> add (timed) -> commit (timed separately: that is the index's cost) with the
# git-style CLI. Records stored bytes, the index's bytes and the codec plan. Rep 1
# keeps its store, at the path the harness's --reuse-existing expects, and runs the
# neurozarr read workloads (W1-W5) on it; rep 2's store is deleted after measuring.
# Fails loudly: a preflight refuses an empty source; exit codes are captured directly.

set -uo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
cd "$REPO"
PY=$REPO/.venv/bin/python
NZ=$REPO/.venv/bin/neurozarr
P6=.bench_scratch/phase6
mkdir -p "$P6"
# HEAD and the staging area live per user; keep this job's apart from anyone's shell
export XDG_STATE_HOME="$REPO/$P6/state_$SLURM_JOB_ID" XDG_CONFIG_HOME="$REPO/$P6/config_$SLURM_JOB_ID"

USAGE="usage: sbatch phase6_ingest.slurm <dataset> <rep>"
DATASET="${1:?$USAGE}"; REP="${2:?$USAGE}"
case "$DATASET" in
	bravo)          SOURCE=BIDS ;;
	rns)            SOURCE=datasets/rns ;;
	chbmit)         SOURCE=.bench_scratch/manifest_chbmit.csv ;;
	hbn_eeg)        SOURCE=datasets/hbn_eeg ;;
	pediatric_ieeg) SOURCE=datasets/pediatric_ieeg ;;
	*) echo "unknown dataset $DATASET" >&2; exit 64 ;;
esac
case "$DATASET" in hbn_eeg|pediatric_ieeg) J=16 ;; *) J=1 ;; esac

WHERE=$(cd /tmp && $PY -c 'import neurozarr, neurozarr.index; print(neurozarr.__file__)')
echo "neurozarr imported from: $WHERE (index module present)"
n=$($PY - "$SOURCE" <<'EOF'
import sys
from neurozarr.items import Recording
from neurozarr.readers import reader_for
print(sum(isinstance(i, Recording) for i in reader_for(sys.argv[1]).read()))
EOF
)
echo "preflight: $SOURCE yields $n recordings; node=$(hostname) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
if [ "${n:-0}" -le 0 ]; then echo "preflight FAILED: no recordings" >&2; exit 2; fi

TAG="${DATASET}_r${REP}"
if [ "$REP" = 1 ]; then STORE="$P6/${DATASET}_scratch/neurozarr"; else STORE="$P6/ingest_${TAG}"; fi
LOG="$P6/ingest_${TAG}.log"
rm -rf "$STORE"; mkdir -p "$(dirname "$STORE")"
rc=0

echo "=== $(date -Is) $TAG: init + add (-j $J) ===" | tee "$LOG"
$NZ init "$STORE" >> "$LOG" 2>&1
s=$?; [ $s -ne 0 ] && rc=$s
/usr/bin/time -v $NZ -C "$STORE" add "$SOURCE" -j "$J" -q >> "$LOG" 2>&1
s=$?; [ $s -ne 0 ] && rc=$s
echo "=== $(date -Is) $TAG: commit ===" | tee -a "$LOG"
/usr/bin/time -v $NZ -C "$STORE" commit -m "phase6 $TAG" >> "$LOG" 2>&1
s=$?; [ $s -ne 0 ] && rc=$s

{
	echo "apparent_bytes $(du -sb "$STORE" | cut -f1)"
	echo "index_bytes $(du -sb "$STORE/_dataset" | cut -f1)"
	echo "node=$(hostname) loadavg_at_end=$(cut -d' ' -f1-3 /proc/loadavg)"
	$NZ -C "$STORE" show --json
	$NZ -C "$STORE" fsck
	$NZ -C "$STORE" query --count
} >> "$LOG" 2>&1
s=$?; [ $s -ne 0 ] && rc=$s

if [ "$REP" = 1 ] && [ $rc -eq 0 ]; then
	# reads with the new codec, on the store just built (no second conversion)
	HARNESS_SOURCE="$SOURCE"
	[ "$DATASET" = chbmit ] && HARNESS_SOURCE=.bench_scratch/phase5/manifest_chbmit_subsample.csv
	if [ "$DATASET" = chbmit ]; then
		# the paper's read workloads use the 20-subject subsample; the harness converts it itself
		$PY scripts/bench/runner.py "$HARNESS_SOURCE" --arms neurozarr --reps 10 --warmup 2 --cache warm \
			--out "$P6/chbmit_subsample_reads.jsonl" --name chbmit_subsample --scratch "$P6/chbmit_subsample_scratch"
	else
		$PY scripts/bench/runner.py "$HARNESS_SOURCE" --arms neurozarr --reps 10 --warmup 2 --cache warm \
			--reuse-existing --out "$P6/${DATASET}_reads.jsonl" --name "$DATASET" --scratch "$P6/${DATASET}_scratch"
	fi
	s=$?; [ $s -ne 0 ] && rc=$s
fi
[ "$REP" != 1 ] && rm -rf "$STORE"
rm -rf "$XDG_STATE_HOME" "$XDG_CONFIG_HOME"
echo "=== $(date -Is) finished $TAG (exit $rc) ===" | tee -a "$LOG"
exit $rc
