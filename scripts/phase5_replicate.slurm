#!/bin/bash
#SBATCH --job-name=nz-bench-replicate
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/rep/slurm-rep-%j.out

# Phase 5i: independent replicates of the timing measurements, because one
# run's absolute numbers swing a lot on this shared cluster -- two runs of the
# identical CHB-MIT-subsample cross-format harness (same CPU model, different
# nodes) differed by up to 1.75x on convert and 1.5x on reads. Within-run n=10
# only captures within-run variance; between-run variance needs separate runs.
# The harness now records node load and store apparent-bytes per row, so a
# slow replicate can be attributed to contention instead of guessed at.
#
#	sbatch phase5_replicate.slurm harness <bravo|rns|chbmit_subsample> <rep>
#	sbatch --cpus-per-task=16 --mem=64G phase5_replicate.slurm ingest <hbn_eeg|pediatric_ieeg> <rep>
#	sbatch phase5_replicate.slurm ingest <bravo|chbmit> <rep>
#
# Fails loudly: the preflight refuses to run on a source with no recordings,
# and the exit status is the real one (see phase5_chbmit_redo.slurm for why).

set -uo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
cd "$REPO"
PY=.venv/bin/python
REPDIR=.bench_scratch/phase5/rep
mkdir -p "$REPDIR"

USAGE="usage: sbatch phase5_replicate.slurm <harness|ingest> <dataset> <rep>"
MODE="${1:?$USAGE}"; DATASET="${2:?$USAGE}"; REP="${3:?$USAGE}"

case "$DATASET" in
	bravo)             SOURCE=BIDS ;;
	rns)               SOURCE=datasets/rns ;;
	chbmit_subsample)  SOURCE=.bench_scratch/phase5/manifest_chbmit_subsample.csv ;;
	chbmit)            SOURCE=.bench_scratch/manifest_chbmit.csv ;;
	hbn_eeg)           SOURCE=datasets/hbn_eeg ;;
	pediatric_ieeg)    SOURCE=datasets/pediatric_ieeg ;;
	*) echo "unknown dataset $DATASET" >&2; exit 64 ;;
esac

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
rc=0
echo "=== $(date -Is) starting $MODE $TAG ==="

case "$MODE" in
harness)
	for ARM in raw neurozarr plain_zarr hdf5_mat73 nwb_hdf5 nwb_zarr; do
		echo "=== $(date -Is) $TAG / $ARM ==="
		$PY scripts/bench/runner.py "$SOURCE" \
			--arms "$ARM" --reps 10 --warmup 2 --cache warm \
			--out "$REPDIR/${TAG}_harness.jsonl" \
			--name "$DATASET" \
			--scratch "$REPDIR/${TAG}_scratch"
		s=$?; [ $s -ne 0 ] && rc=$s
	done
	rm -rf "$REPDIR/${TAG}_scratch"
	;;
ingest)
	case "$DATASET" in hbn_eeg|pediatric_ieeg) J=16 ;; *) J=1 ;; esac
	STORE="$REPDIR/ingest_${TAG}"
	rm -rf "$STORE"
	/usr/bin/time -v .venv/bin/neurozarr convert "$SOURCE" "$STORE" -j "$J" > "$REPDIR/ingest_${TAG}.log" 2>&1
	rc=$?
	echo "apparent_bytes $(du -sb "$STORE" | cut -f1)" >> "$REPDIR/ingest_${TAG}.log"
	echo "node=$(hostname) loadavg_at_end=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$REPDIR/ingest_${TAG}.log"
	rm -rf "$STORE"
	;;
*) echo "unknown mode $MODE" >&2; exit 64 ;;
esac

echo "=== $(date -Is) finished $MODE $TAG (exit $rc) ==="
exit $rc
