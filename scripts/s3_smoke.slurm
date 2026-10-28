#!/bin/bash
#SBATCH --job-name=nz-s3
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/s3/slurm-%j.out

# Smoke test: a private S3 server on this node's local disk (Versity S3 Gateway, posix
# backend -- MinIO stopped publishing binaries in 2025), then neurozarr converts BRAVO into
# it, verifies it against the source, and times a few reads. Proves the S3 path end to
# end; not a cloud-latency measurement (client and server share the node).
#
#	sbatch scripts/s3_smoke.slurm [dataset dir, default BIDS]

set -euo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
cd "$REPO"
PY=$REPO/.venv/bin/python
GW=$REPO/.bench_scratch/bin/versitygw
SRC="${1:-BIDS}"
OUT=.bench_scratch/s3
mkdir -p "$OUT"
export XDG_STATE_HOME="$REPO/$OUT/state_$SLURM_JOB_ID"
echo "=== $(date -Is) node=$(hostname) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) src=$SRC ==="
"$GW" --version

# throwaway credentials, local to this job; the server lives only as long as the job
export ROOT_ACCESS_KEY=nzbench
export ROOT_SECRET_KEY=$(head -c 24 /dev/urandom | base64 | tr -dc A-Za-z0-9)
DATA=${SLURM_TMPDIR:-/tmp/nz-s3-$SLURM_JOB_ID}/s3
mkdir -p "$DATA"
"$GW" --port 127.0.0.1:7070 posix "$DATA" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
up() { [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:7070/)" != "000" ]; }
for _ in $(seq 60); do up && break; sleep 1; done
up || { echo "S3 gateway did not start" >&2; exit 1; }
echo "S3 gateway up at 127.0.0.1:7070, data on $DATA"

export AWS_ACCESS_KEY_ID=$ROOT_ACCESS_KEY AWS_SECRET_ACCESS_KEY=$ROOT_SECRET_KEY
$PY scripts/s3_smoke.py "$SRC" "$OUT/smoke_$SLURM_JOB_ID.json" --endpoint http://127.0.0.1:7070 --workers 8
echo "=== done $(date -Is) bytes in bucket: $(du -sb "$DATA" | cut -f1) ==="
