#!/bin/bash
#SBATCH --job-name=nz-bench-verify-outputs
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-verify-outputs-%j.out

# Phase 5j: two checks over the stores the earlier phase5/phase4 runs left on
# disk, so the numbers that were reported don't rest on a hash or on `du`.
#
# 1. check_arm_equivalence.py: every arm's values vs. the raw source, as max
#    absolute error (the harness's rounded checksums mismatch the raw arm on
#    float32-stored data, which a hash can't distinguish from a real
#    difference).
# 2. Apparent bytes per store: `du` without --apparent-size rounds every tiny
#    file up to a block and inflates many-small-files layouts.
#
# Read-only against the existing stores. Memory is 64G because the Pediatric
# iEEG target recording is large (Phase 4 only got it through at 64G).

set -uo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
cd "$REPO"
PY=.venv/bin/python
P4=.bench_scratch/phase4
P5=.bench_scratch/phase5
OUT=$P5/verify_outputs
mkdir -p "$OUT"
rc=0
# Optional single-dataset rerun (equivalence only, skips the size sweep):
#	sbatch phase5_verify_outputs.slurm hbn_eeg
ONLY="${1:-}"
want () { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

equiv () {
	local name=$1; shift
	want "$name" || return 0
	echo "=== $(date -Is) equivalence: $name ==="
	$PY scripts/check_arm_equivalence.py --name "$name" --out "$OUT/equivalence_${name}.json" "$@"
	s=$?; [ $s -ne 0 ] && rc=$s
}

arms5 () { local d=$1; local a; for a in raw neurozarr plain_zarr hdf5_mat73 nwb_hdf5 nwb_zarr; do printf -- "--store %s=%s/%s " "$a" "$d" "$a"; done; }

equiv bravo $(arms5 $P5/bravo_scratch)
equiv rns $(arms5 $P5/rns_scratch)
equiv chbmit_subsample $(arms5 $P5/chbmit_subsample_fixed_scratch)
equiv pediatric_ieeg --store raw=$P4/pediatric_ieeg_scratch/raw --store neurozarr=$P4/pediatric_ieeg_scratch/neurozarr \
	--store hdf5_mat73=$P4/pediatric_ieeg_scratch/hdf5_mat73
equiv hbn_eeg --store raw=$P4/hbn_eeg_scratch/raw --store neurozarr=$P4/hbn_eeg_scratch/neurozarr \
	--store hdf5_mat73=$P4/hbn_eeg_scratch/hdf5_mat73 \
	--store nwb_hdf5=$P5/hbn_eeg_nwb_hdf5_scratch/nwb_hdf5 --store nwb_zarr=$P5/hbn_eeg_nwb_zarr_scratch/nwb_zarr

if [ -n "$ONLY" ]; then echo "=== $(date -Is) finished verify-outputs for $ONLY (exit $rc) ==="; exit $rc; fi
echo "=== $(date -Is) apparent sizes ==="
{
	for d in $P5/ingest_*/ $P5/bravo_scratch/* $P5/rns_scratch/* $P5/chbmit_subsample_fixed_scratch/* \
			$P5/hbn_eeg_nwb_hdf5_scratch/* $P5/hbn_eeg_nwb_zarr_scratch/* \
			$P4/hbn_eeg_scratch/* $P4/pediatric_ieeg_scratch/* \
			$P5/*hdf5_fairness*_scratch/*; do
		[ -e "$d" ] && printf "%s\t%s\n" "$(du -sb --apparent-size "$d" | cut -f1)" "$d"
	done
} > "$OUT/store_apparent_bytes.tsv"
echo "=== $(date -Is) finished verify-outputs (exit $rc) ==="
exit $rc
