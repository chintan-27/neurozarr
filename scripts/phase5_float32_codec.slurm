#!/bin/bash
#SBATCH --job-name=nz-bench-float32-codec
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase5/slurm-float32-codec-%j.out

# Phase 5l: HBN EEG with neurozarr's new float32 default (Blosc/zstd level 5 +
# byte shuffle instead of plain zstd-19). HBN is the only dataset here that
# takes the float32 path (EEGLAB .set has no calibration), so it is the only
# neurozarr result that changes; the EDF datasets use int16 packing, untouched.
# The previous-default numbers (70,418 MiB, 2,799/2,888 s ingest, phase4 warm
# reads) stay on record for the before/after.
#
# Steps: preflight; two timed -j 16 conversions (the second is for the ingest
# replicate and is deleted); warm and cold harness reads against the first
# store (--reuse-existing); array-level equivalence against the raw source;
# apparent bytes. Fails loudly: zero-recordings preflight, real exit codes.

set -uo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
cd "$REPO"
PY=.venv/bin/python
P5=.bench_scratch/phase5
P4=.bench_scratch/phase4
SCR=$P5/hbn_float32_scratch
mkdir -p "$SCR" "$P5/float32"
rc=0

where=$(cd /tmp && PYTHONPATH="$REPO" $PY -c 'import neurozarr; print(neurozarr.__file__)')
echo "neurozarr imported from: $where"
n=$($PY - datasets/hbn_eeg <<'EOF'
import sys
from neurozarr.items import Recording
from neurozarr.readers import reader_for
print(sum(isinstance(i, Recording) for i in reader_for(sys.argv[1]).read()))
EOF
)
echo "preflight: datasets/hbn_eeg yields $n recordings; node=$(hostname) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
if [ "${n:-0}" -le 0 ]; then echo "preflight FAILED" >&2; exit 2; fi
$PY - <<'EOF'
from neurozarr.writer import CodecConfig
print("codec default (None = chosen per recording):", CodecConfig().compressors)
EOF

for R in 1 2; do
	DEST="$SCR/neurozarr"; [ "$R" = 2 ] && DEST="$P5/float32/convert_r2"
	rm -rf "$DEST"
	echo "=== $(date -Is) convert run $R ==="
	{ .venv/bin/neurozarr init "$DEST" && /usr/bin/time -v .venv/bin/neurozarr -C "$DEST" add datasets/hbn_eeg -j 16 -q && .venv/bin/neurozarr -C "$DEST" commit -m convert; } > "$P5/float32/convert_r${R}.log" 2>&1
	s=$?; [ $s -ne 0 ] && rc=$s
	echo "apparent_bytes $(du -sb "$DEST" | cut -f1)" >> "$P5/float32/convert_r${R}.log"
	echo "node=$(hostname) loadavg_at_end=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$P5/float32/convert_r${R}.log"
	[ "$R" = 2 ] && rm -rf "$DEST"
done

for CACHE in warm cold; do
	WU=2; [ "$CACHE" = cold ] && WU=0
	echo "=== $(date -Is) harness $CACHE ==="
	$PY scripts/bench/runner.py datasets/hbn_eeg --arms neurozarr --reps 10 --warmup $WU --cache $CACHE --reuse-existing \
		--out "$P5/hbn_float32_new_${CACHE}.jsonl" --name hbn_eeg --scratch "$SCR"
	s=$?; [ $s -ne 0 ] && rc=$s
done

echo "=== $(date -Is) equivalence ==="
$PY scripts/check_arm_equivalence.py --name hbn_eeg_float32_new --out "$P5/float32/equivalence.json" \
	--store raw=$P4/hbn_eeg_scratch/raw --store neurozarr="$SCR/neurozarr"
s=$?; [ $s -ne 0 ] && rc=$s

echo "=== $(date -Is) finished float32 codec job (exit $rc) ==="
exit $rc
