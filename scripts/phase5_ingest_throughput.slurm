#!/bin/bash
#SBATCH --job-name=nz-bench-ingest
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-ingest-%j.out

# Phase 5c: re-run ingest throughput for all 4 datasets from the original
# "Benchmark 2" pilot slide, uniformly on HiPerGator (that slide's own
# numbers were split local/cluster). -j N matches convert_parallel's own
# subject-level parallelism; omitted where it never helped on a single
# subject (BRAVO) or wasn't part of the original comparison's point (RNS,
# which Benchmark 2 never measured -- Benchmark 4's ease-of-ingestion table
# covers RNS's wall time separately and isn't re-run here).

# NOTE: the CHB-MIT legs below never ran. CHB-MIT is a flat EDF dump, not BIDS,
# so a BIDS reader pointed at datasets/chbmit finds zero recordings -- the
# "CHB-MIT" step finished in ~2 s and this script's logged "(exit N)" was
# unreliable ($? was read after a $(date) substitution reset it). BRAVO, RNS,
# HBN and Pediatric legs are real. The CHB-MIT work was redone, with a
# zero-recordings preflight and correct exit codes, in phase5_chbmit_redo.slurm.

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase5

run_one () {
	local name=$1 source=$2 jobs=$3
	echo "=== $(date -Is) starting ingest: $name (-j $jobs) ==="
	rm -rf ".bench_scratch/phase5/ingest_${name}"
	/usr/bin/time -v .venv/bin/neurozarr convert "$source" ".bench_scratch/phase5/ingest_${name}" -j "$jobs" \
		> ".bench_scratch/phase5/ingest_${name}.log" 2>&1
	echo "=== $(date -Is) finished ingest: $name (exit $?) ==="
}

run_one bravo BIDS 1
run_one chbmit datasets/chbmit 1
run_one hbn_eeg datasets/hbn_eeg 16
run_one pediatric_ieeg datasets/pediatric_ieeg 16
