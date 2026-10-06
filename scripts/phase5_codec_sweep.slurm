#!/bin/bash
#SBATCH --job-name=nz-bench-codec-sweep
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-codec-%j.out

# Phase 5b: the 12-lossless-config codec sweep, moved to HiPerGator in full
# (BRAVO/RNS too, not just CHB-MIT) so it shares hardware with the rest of
# phase5's timing numbers. CODECS in benchmark_compressors.py already only
# has the 12 lossless configs (bitround was dropped earlier) -- nothing to
# change there, this just runs it uniformly instead of BRAVO/RNS-local +
# CHB-MIT-cluster like the original pilot sweep was.

set -uo pipefail
cd /blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
mkdir -p .bench_scratch/phase5

echo "=== $(date -Is) starting BRAVO codec sweep ==="
.venv/bin/python scripts/benchmark_compressors.py ./BIDS --reader bids \
	> .bench_scratch/phase5/compressors_bravo.log 2>&1
echo "=== $(date -Is) finished BRAVO codec sweep (exit $?) ==="

echo "=== $(date -Is) starting RNS codec sweep ==="
.venv/bin/python scripts/benchmark_compressors.py datasets/rns --reader bids \
	> .bench_scratch/phase5/compressors_rns.log 2>&1
echo "=== $(date -Is) finished RNS codec sweep (exit $?) ==="

echo "=== $(date -Is) starting CHB-MIT (full 43GB) codec sweep -- the long pole ==="
.venv/bin/python scripts/benchmark_compressors.py datasets/chbmit --reader bids \
	> .bench_scratch/phase5/compressors_chbmit.log 2>&1
echo "=== $(date -Is) finished CHB-MIT codec sweep (exit $?) ==="
