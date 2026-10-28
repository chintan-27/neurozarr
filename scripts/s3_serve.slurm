#!/bin/bash
#SBATCH --job-name=nz-s3-serve
#SBATCH --account=neurology-dept
#SBATCH --qos=neurology-dept
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr/.bench_scratch/s3/serve-%j.out

# Serve neurozarr stores over S3 from this node, for reading from outside the cluster
# (a laptop, through an SSH tunnel via the login node). Versity S3 Gateway, posix
# backend; the bucket's files persist on /blue, so each dataset is loaded once and
# later jobs only restart the server over it.
#
#	sbatch scripts/s3_serve.slurm [SOURCE NAME OLD_STORES]...
#	  e.g.  BIDS bravo .bench_scratch/phase5/bravo_scratch
#
# Per dataset: the existing HDF5 / NWB / raw copies from OLD_STORES go under
# bench/formats/NAME/ (s3_formats.py), and neurozarr is loaded as bench/NAME.
#
# The node, port and throwaway credentials for this job go to
# ~/.nz_s3/job_<jobid>.env (mode 600). Stop serving with scancel <jobid>.

set -euo pipefail
REPO=/blue/neurology-dept/chintan.acharya/neurozarr-benchmark/neurozarr
cd "$REPO"
PY=$REPO/.venv/bin/python
GW=$REPO/.bench_scratch/bin/versitygw
DATA=$REPO/.bench_scratch/s3data
OUT=.bench_scratch/s3
mkdir -p "$DATA" "$OUT" "$HOME/.nz_s3"
export XDG_STATE_HOME="$REPO/$OUT/state_$SLURM_JOB_ID"
PORT=$((20000 + SLURM_JOB_ID % 10000))  # per job, so two jobs on one node don't clash
NODE=$(hostname -s)
echo "=== $(date -Is) node=$NODE port=$PORT loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
"$GW" --version

export ROOT_ACCESS_KEY=nz$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')
export ROOT_SECRET_KEY=$(head -c 24 /dev/urandom | base64 | tr -dc A-Za-z0-9)
ENVFILE="$HOME/.nz_s3/job_$SLURM_JOB_ID.env"
( umask 077; printf 'NODE=%s\nPORT=%s\nAWS_ACCESS_KEY_ID=%s\nAWS_SECRET_ACCESS_KEY=%s\n' \
	"$NODE" "$PORT" "$ROOT_ACCESS_KEY" "$ROOT_SECRET_KEY" > "$ENVFILE" )
trap 'rm -f "$ENVFILE"; kill $SERVER 2>/dev/null' EXIT

# listens on every interface: the login node must reach it for the tunnel. Requests
# are signed with the job's credentials; the per-request log goes to its own file.
"$GW" --port ":$PORT" posix "$DATA" > "$OUT/serve-$SLURM_JOB_ID.gateway.log" 2>&1 &
SERVER=$!
up() { [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/")" != "000" ]; }
for _ in $(seq 60); do up && break; sleep 1; done
up || { echo "S3 gateway did not start" >&2; exit 1; }
echo "serving $DATA at $NODE:$PORT; credentials in $ENVFILE"

export AWS_ACCESS_KEY_ID=$ROOT_ACCESS_KEY AWS_SECRET_ACCESS_KEY=$ROOT_SECRET_KEY
# the bucket must exist through S3 before s3_formats.py copies files into its folder
$PY scripts/s3_bucket.py "$DATA/bench" --endpoint "http://127.0.0.1:$PORT"
while [ $# -ge 3 ]; do
	$PY scripts/s3_formats.py "$1" "$2" "$3" "$DATA/bench"
	$PY scripts/s3_load.py "$1" "$2" --endpoint "http://127.0.0.1:$PORT" --workers 8
	shift 3
done
echo "=== $(date -Is) ready: $(du -sb "$DATA" | cut -f1) bytes served ==="
wait $SERVER
