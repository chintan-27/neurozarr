#!/bin/bash
#SBATCH --job-name=nz-p7-xformat
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase7/slurm-%j.out

# Phase 7: every format read in ONE job on one node, one dataset at a time (singleton),
# so cross-format ratios never mix nodes or runs. Reuses the stores already built:
# neurozarr from Phase 6 (current codecs), HDF5 / NWB / raw from Phases 4-5; plain Zarr
# is rebuilt here with the current code for the small datasets (it shares neurozarr's
# writer, so the Phase 5 copy has the old codec). Stores converted here (rebuilt because
# an earlier phase's copy was gone) stay in $SCRATCH, so a rerun only reads. Harness
# changes since Phase 6: W1-W3 return float64 in every arm (casts timed), W2's window
# moves every rep.
#
#	sbatch phase7_crossformat.slurm <bravo|rns|chbmit_subsample|hbn_eeg|pediatric_ieeg> [run tag] [warm|fresh]
#
# fresh: before every read each format drops what it holds in memory for the recording
# (HDF5's 8 MiB decoded-chunk cache had served repeated reads of small recordings)

set -uo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
cd "$REPO"
PY=$REPO/.venv/bin/python
B=.bench_scratch
P7=$B/phase7
DATASET="${1:?usage: sbatch phase7_crossformat.slurm <dataset> [run tag]}"
TAG="${2:+_$2}"  # a rerun writes its own results file
CACHE="${3:-warm}"
export XDG_STATE_HOME="$REPO/$P7/state_$SLURM_JOB_ID"
case "$DATASET" in
	bravo)            SOURCE=BIDS;                    OLD=$B/phase5/bravo_scratch ;;
	rns)              SOURCE=datasets/rns;            OLD=$B/phase5/rns_scratch ;;
	chbmit_subsample) SOURCE=$B/phase5/manifest_chbmit_subsample.csv; OLD=$B/phase5/chbmit_subsample_fixed_scratch ;;
	hbn_eeg|pediatric_ieeg) SOURCE=datasets/$DATASET;  OLD=$B/phase4/${DATASET}_scratch ;;
	*) echo "unknown dataset $DATASET" >&2; exit 64 ;;
esac
SCRATCH=$P7/${DATASET}_scratch
mkdir -p "$SCRATCH"
declare -A STORE=([neurozarr]=$B/phase6/${DATASET}_scratch/neurozarr [raw]=$OLD/raw [hdf5_mat73]=$OLD/hdf5_mat73)
case "$DATASET" in
	hbn_eeg|pediatric_ieeg)
		STORE[nwb_hdf5]=$B/phase5/${DATASET}_nwb_hdf5_scratch/nwb_hdf5
		STORE[nwb_zarr]=$B/phase5/${DATASET}_nwb_zarr_scratch/nwb_zarr
		ARMS=raw,neurozarr,hdf5_mat73,nwb_hdf5,nwb_zarr ;;
	*)
		STORE[nwb_hdf5]=$OLD/nwb_hdf5
		STORE[nwb_zarr]=$OLD/nwb_zarr
		ARMS=raw,neurozarr,plain_zarr,hdf5_mat73,nwb_hdf5,nwb_zarr ;;
esac
for arm in "${!STORE[@]}"; do
	if [ -e "${STORE[$arm]}" ]; then
		ln -sfn "$REPO/${STORE[$arm]}" "$SCRATCH/$arm"
	elif [ "$arm" = neurozarr ]; then
		echo "MISSING Phase 6 neurozarr store: ${STORE[$arm]} -- run phase6_ingest.slurm first" >&2; exit 2
	else
		# earlier phases' copy was cleaned up: the harness converts it here (and times it)
		echo "no earlier $arm store at ${STORE[$arm]}; converting into $SCRATCH/$arm"
	fi
done
rm -rf "$SCRATCH/plain_zarr"  # rebuilt below with the current writer (small datasets only)
ls -la "$SCRATCH"
echo "=== $(date -Is) cross-format reads $DATASET [$ARMS]; node=$(hostname) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
$PY scripts/bench/runner.py "$SOURCE" --arms "$ARMS" --reps 10 --warmup 2 --cache "$CACHE" --reuse-existing \
	--no-checksum-source --out "$P7/${DATASET}${TAG}.jsonl" --name "$DATASET" --scratch "$SCRATCH"
rc=$?
rm -rf "$XDG_STATE_HOME" "$SCRATCH/plain_zarr"
echo "=== $(date -Is) finished $DATASET (exit $rc); loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
exit $rc
