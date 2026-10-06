#!/bin/bash
#SBATCH --job-name=nz-bench-nwb-spotcheck
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-nwb-%j.out

# Phase 5f: NWB spot-check at scale -- HBN EEG (104GB), the smaller of the two
# large datasets, picked to keep the first real attempt at this cheaper to
# retry if something else turns up. An earlier attempt at streaming NWB
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

ARM="${1:?usage: sbatch phase5_nwb_spotcheck.slurm <nwb_hdf5|nwb_zarr>}"

echo "=== $(date -Is) starting NWB spot-check: hbn_eeg / $ARM ==="
.venv/bin/python scripts/bench/runner.py datasets/hbn_eeg \
	--arms "$ARM" --reps 10 --warmup 2 --cache warm \
	--out ".bench_scratch/phase5/hbn_eeg_${ARM}_warm.jsonl" \
	--name hbn_eeg \
	--scratch ".bench_scratch/phase5/hbn_eeg_${ARM}_scratch"
status=$?
echo "=== $(date -Is) finished NWB spot-check: hbn_eeg / $ARM (exit $status) ==="
