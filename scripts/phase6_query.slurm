#!/bin/bash
#SBATCH --job-name=nz-p6-query
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase6/slurm-query-%j.out

# Phase 6: metadata-query latency, same node and time for every implementation:
#   new        -- the columnar index (current code), on phase6_ingest.slurm's rep-1 store
#   old        -- the JSON catalog (commit 7320690, synced to ../nz-catalog), on a store
#                 that code built: built here for bravo / rns / chbmit_subsample; for
#                 hbn_eeg / pediatric_ieeg the Phase 4 stores, after the old fsck confirms
#                 their catalog is current (else its find() silently falls back to a walk)
#   pybids     -- BIDSLayout, for BIDS sources
#   bids2table -- if it imports (installed to $P6/pylib), for BIDS sources
#
#	sbatch --dependency=afterok:<ingest rep-1 job> phase6_query.slurm <bravo|rns|chbmit_subsample|hbn_eeg|pediatric_ieeg>
#	sbatch phase6_query.slurm synthetic          # the engine alone at 10^4..10^6 items

set -uo pipefail
ROOT=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark
REPO=$ROOT/neurozarr
OLD=$ROOT/nz-catalog
cd "$REPO"
PY=$REPO/.venv/bin/python
P6=.bench_scratch/phase6
OUT=$P6/query.jsonl
mkdir -p "$P6"
export XDG_STATE_HOME="$REPO/$P6/state_$SLURM_JOB_ID" XDG_CONFIG_HOME="$REPO/$P6/config_$SLURM_JOB_ID"
DATASET="${1:?usage: sbatch phase6_query.slurm <dataset|synthetic>}"
echo "node=$(hostname) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
rc=0

if [ "$DATASET" = synthetic ]; then
	$PY scripts/bench/synthetic_index_bench.py --out "$P6/synthetic.jsonl" --sizes 100x100,1000x100,10000x100
	exit $?
fi

case "$DATASET" in
	bravo)            SOURCE=BIDS;                    NEW=$P6/bravo_scratch/neurozarr ;;
	rns)              SOURCE=datasets/rns;            NEW=$P6/rns_scratch/neurozarr ;;
	chbmit_subsample) SOURCE=.bench_scratch/phase5/manifest_chbmit_subsample.csv; NEW=$P6/chbmit_subsample_scratch/neurozarr ;;
	hbn_eeg)          SOURCE=datasets/hbn_eeg;        NEW=$P6/hbn_eeg_scratch/neurozarr ;;
	pediatric_ieeg)   SOURCE=datasets/pediatric_ieeg; NEW=$P6/pediatric_ieeg_scratch/neurozarr ;;
	*) echo "unknown dataset $DATASET" >&2; exit 64 ;;
esac
[ -d "$NEW" ] || { echo "no phase-6 store at $NEW -- run phase6_ingest.slurm $DATASET 1 first" >&2; exit 2; }

echo "=== $(date -Is) new: index ==="
$PY scripts/bench/query_bench.py --impl new --store "$NEW" --name "$DATASET" --out "$OUT"
s=$?; [ $s -ne 0 ] && rc=$s

echo "=== $(date -Is) old: JSON catalog ==="
WHERE=$(cd /tmp && PYTHONPATH="$OLD" $PY -c 'import neurozarr; print(neurozarr.__file__)')
echo "old neurozarr imported from: $WHERE"
case "$WHERE" in "$OLD"/*) ;; *) echo "WRONG neurozarr for the old arm -- refusing" >&2; exit 3 ;; esac
case "$DATASET" in
	hbn_eeg|pediatric_ieeg)
		OLDSTORE=.bench_scratch/phase4/${DATASET}_scratch/neurozarr
		echo "old fsck of $OLDSTORE (a stale or missing catalog would make old find() walk):"
		PYTHONPATH="$OLD" $PY -m neurozarr.cli -C "$OLDSTORE" fsck
		;;
	*)
		OLDSTORE=$P6/${DATASET}_old_store
		rm -rf "$OLDSTORE"
		PYTHONPATH="$OLD" $PY -m neurozarr.cli init "$OLDSTORE" \
			&& PYTHONPATH="$OLD" $PY -m neurozarr.cli -C "$OLDSTORE" add "$SOURCE" -q \
			&& PYTHONPATH="$OLD" $PY -m neurozarr.cli -C "$OLDSTORE" commit -m "old-catalog store"
		s=$?; [ $s -ne 0 ] && { echo "old store build failed" >&2; rc=$s; }
		;;
esac
PYTHONPATH="$OLD" $PY scripts/bench/query_bench.py --impl old --store "$OLDSTORE" --name "$DATASET" --out "$OUT"
s=$?; [ $s -ne 0 ] && rc=$s

if [ "$DATASET" != chbmit_subsample ]; then  # BIDS sources only
	echo "=== $(date -Is) pybids ==="
	$PY scripts/bench/query_bench.py --impl pybids --source "$SOURCE" --name "$DATASET" --out "$OUT"
	s=$?; [ $s -ne 0 ] && rc=$s
	if [ -d "$P6/pylib" ] && PYTHONPATH="$P6/pylib" $PY -c 'import bids2table' 2>/dev/null; then
		echo "=== $(date -Is) bids2table ==="
		PYTHONPATH="$P6/pylib" $PY scripts/bench/query_bench.py --impl bids2table --source "$SOURCE" --name "$DATASET" --out "$OUT"
		s=$?; [ $s -ne 0 ] && echo "bids2table failed (exit $s); recorded as missing, not fatal"
	else
		echo "bids2table not importable -- skipped"
	fi
fi

case "$DATASET" in hbn_eeg|pediatric_ieeg) ;; *) rm -rf "$P6/${DATASET}_old_store" ;; esac
rm -rf "$XDG_STATE_HOME" "$XDG_CONFIG_HOME"
echo "=== $(date -Is) finished query bench $DATASET (exit $rc) ==="
exit $rc
