#!/bin/bash
#SBATCH --job-name=nz-final
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/phase9/slurm-%j.out

# Phase 9: the remaining paper experiments, one job at a time (singleton name nz-final;
# the Phase 7 read jobs are submitted under the same name so nothing overlaps):
#
#	sbatch phase9_final.slurm index                 # summary splice + packed index at scale,
#	                                                  and query latency on every Phase 6 store
#	sbatch phase9_final.slurm readers <rep>         # RNS through BidsReader / ManifestReader /
#	                                                  the explicit API
#	sbatch phase9_final.slurm versioning <bravo|rns|chbmit5>  # neurozarr vs DataLad (git-annex);
#	                                                  chbmit5 = 5 CHB-MIT EDFs as BIDS, for large-file edits

set -uo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
cd "$REPO"
PY=$REPO/.venv/bin/python
NZ=$REPO/.venv/bin/neurozarr
export PATH="$REPO/.venv/bin:$PATH"  # datalad + git-annex installed into the venv
P9=.bench_scratch/phase9
mkdir -p "$P9"
export XDG_STATE_HOME="$REPO/$P9/state_$SLURM_JOB_ID"
WHAT="${1:?usage: sbatch phase9_final.slurm <index|readers REP|versioning DATASET>}"
echo "=== $(date -Is) $* ; node=$(hostname) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
echo "neurozarr from $(cd /tmp && $PY -c 'import neurozarr; print(neurozarr.__file__)')"
rc=0
case "$WHAT" in
	index)
		$PY scripts/bench/index_scale_bench.py --out "$P9/index_scale.jsonl" --scratch "$P9/index_scratch"
		s=$?; [ $s -ne 0 ] && rc=$s
		for d in bravo rns chbmit_subsample hbn_eeg pediatric_ieeg; do
			$PY scripts/bench/query_bench.py --impl new --store ".bench_scratch/phase6/${d}_scratch/neurozarr" \
				--name "$d" --out "$P9/query.jsonl"
			s=$?; [ $s -ne 0 ] && rc=$s
		done
		rm -rf "$P9/index_scratch" ;;
	readers)
		REP="${2:?rep}"
		W="$P9/readers_r$REP"; rm -rf "$W"; mkdir -p "$W"
		$PY scripts/build_rns_manifest.py datasets/rns "$W/manifest_rns.csv" || rc=2
		n=$(grep -c . "$W/manifest_rns.csv"); echo "manifest rows: $n"
		[ "${n:-0}" -gt 1 ] || { echo "manifest is empty" >&2; exit 2; }
		for reader in bids manifest explicit; do
			STORE="$W/$reader"
			echo "=== $(date -Is) reader $reader ==="
			case $reader in
				bids)     /usr/bin/time -v sh -c "$NZ init $STORE && $NZ -C $STORE add datasets/rns -q && $NZ -C $STORE commit -m bids" ;;
				manifest) /usr/bin/time -v sh -c "$NZ init $STORE && $NZ -C $STORE add $W/manifest_rns.csv -q && $NZ -C $STORE commit -m manifest" ;;
				explicit) /usr/bin/time -v $PY scripts/convert_rns_explicit.py datasets/rns "$STORE" ;;
			esac > "$W/$reader.log" 2>&1
			s=$?; [ $s -ne 0 ] && { rc=$s; echo "reader $reader FAILED (exit $s)"; tail -5 "$W/$reader.log"; }
			{ echo "elapsed $(grep 'Elapsed (wall' "$W/$reader.log" | awk '{print $NF}')"
			  echo "apparent_bytes $(du -sb "$STORE" | cut -f1)"
			  echo "recordings $($NZ -C "$STORE" query --kind recording --count)"
			  $NZ -C "$STORE" fsck; } > "$W/$reader.result" 2>&1
			cat "$W/$reader.result"
		done
		rm -rf "$W/bids" "$W/manifest" "$W/explicit" ;;
	versioning)
		D="${2:?dataset}"
		EXTRA=""
		case "$D" in
			bravo) SRC=BIDS ;;
			rns)   SRC=datasets/rns ;;
			chbmit5) SRC=datasets/chbmit; EXTRA="--from-chbmit 5" ;;  # large files: ~40 MB EDFs
			*) echo "unknown $D" >&2; exit 64 ;;
		esac
		command -v datalad >/dev/null && command -v git-annex >/dev/null || { echo "datalad/git-annex missing" >&2; exit 2; }
		$PY scripts/bench/versioning_bench.py "$SRC" $EXTRA --name "$D" --out "$P9/versioning_final.jsonl" --scratch "$P9/versioning_$D"
		rc=$?
		chmod -R u+w "$P9/versioning_$D" 2>/dev/null; rm -rf "$P9/versioning_$D" ;;
	*) echo "unknown $WHAT" >&2; exit 64 ;;
esac
rm -rf "$XDG_STATE_HOME"
echo "=== $(date -Is) finished $* (exit $rc) ==="
exit $rc
