#!/bin/bash
#SBATCH --job-name=nz-bench-nwb-spotcheck
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-nwb-%j.out

# Phase 5f: NWB spot-check at scale -- HBN EEG (104GB, the default) or Pediatric
# iEEG (165GB, via the optional second argument). HBN EEG went first: the smaller
# of the two, to keep the first real attempt cheaper to retry if something else
# turned up. An earlier attempt at streaming NWB
# writes hit a broadcast error from passing mismatched chunk/buffer shapes by
# hand to hdmf's DataChunkIterator; arms.py's _build_nwb_files now streams
# through a GenericDataChunkIterator subclass that lets hdmf compute its own
# shapes from maxshape/dtype instead, and that path is round-trip-verified
# locally (demo_nwb_streaming_check.py) before this job -- but this is still
# the first time it's run against real, large, multi-session data, so nwb_hdf5
# and nwb_zarr run as fully separate submissions (not looped together like the
# other phase5 jobs): if nwb_zarr's known many-small-files behavior degrades
# badly at this scale, or the broadcast fix doesn't generalize to some session
# shape this dataset has and raw/neurozarr/hdf5_mat73 don't, that failure
# stays contained to its own job instead of costing the other arm's result too.

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase5

ARM="${1:?usage: sbatch phase5_nwb_spotcheck.slurm <nwb_hdf5|nwb_zarr> [dataset, default hbn_eeg]}"
DATASET="${2:-hbn_eeg}"
P5=.bench_scratch/phase5
rc=0

echo "=== $(date -Is) starting NWB spot-check: $DATASET / $ARM ==="
.venv/bin/python scripts/bench/runner.py "datasets/$DATASET" \
	--arms "$ARM" --reps 10 --warmup 2 --cache warm \
	--out "$P5/${DATASET}_${ARM}_warm.jsonl" \
	--name "$DATASET" \
	--scratch "$P5/${DATASET}_${ARM}_scratch"
s=$?; [ $s -ne 0 ] && rc=$s

# Stored values vs. the raw source (the Phase 4 raw link is a symlink to the
# dataset), and apparent bytes -- run here so the NWB store never has to be
# kept around afterward just to be checked.
mkdir -p "$P5/nwb_checks"
.venv/bin/python scripts/check_arm_equivalence.py --name "${DATASET}_${ARM}" --out "$P5/nwb_checks/equivalence_${DATASET}_${ARM}.json" \
	--store raw=.bench_scratch/phase4/${DATASET}_scratch/raw --store "$ARM=$P5/${DATASET}_${ARM}_scratch/$ARM"
s=$?; [ $s -ne 0 ] && rc=$s
echo "apparent_bytes $(du -sb "$P5/${DATASET}_${ARM}_scratch/$ARM" | cut -f1)" > "$P5/nwb_checks/size_${DATASET}_${ARM}.txt"

echo "=== $(date -Is) finished NWB spot-check: $DATASET / $ARM (exit $rc) ==="
exit $rc
