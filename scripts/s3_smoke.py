"""S3 smoke test against a local S3 server (see s3_smoke.slurm): ingest a dataset into a
bucket, verify it against the source through S3, and time a full and a 10 s read.

	python scripts/s3_smoke.py SOURCE OUT.json [--endpoint http://127.0.0.1:7070] [--workers 8]

Credentials come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
"""

import argparse
import functools
import json
import os
import sys
import time

import boto3

from neurozarr import Repo, verify
from neurozarr.storage import storage_from

STORE = "s3://bench/store"


def main() -> int:
	p = argparse.ArgumentParser()
	p.add_argument("source")
	p.add_argument("out")
	p.add_argument("--endpoint", default="http://127.0.0.1:7070")
	p.add_argument("--workers", type=int, default=8)
	args = p.parse_args()

	key, secret = os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"]
	opts = dict(endpoint_url=args.endpoint, allow_http=True, force_path_style=True, region="us-east-1",
				access_key_id=key, secret_access_key=secret)
	boto3.client("s3", endpoint_url=args.endpoint, aws_access_key_id=key, aws_secret_access_key=secret,
				 region_name="us-east-1").create_bucket(Bucket="bench")
	res: dict = {"source": args.source, "node": os.uname().nodename, "workers": args.workers}

	t = time.perf_counter()
	repo = Repo.init(STORE, **opts)
	repo.add(args.source, workers=args.workers)
	repo.commit("smoke: full ingest over S3")
	res["ingest_s"] = time.perf_counter() - t

	# verify() opens the store itself, so hand it a storage factory carrying the same options
	t = time.perf_counter()
	problems = verify(args.source, functools.partial(storage_from, STORE, **opts))
	res["verify_s"], res["verify_problems"], res["n_problems"] = time.perf_counter() - t, problems[:20], len(problems)

	repo = Repo.open(STORE, **opts)
	res["n_items"] = len(repo.query())
	rec = max(repo.find(), key=lambda r: r.shape[0] * r.shape[-1])
	res["recording"], res["shape"] = rec.path, list(rec.shape)
	for name, window in (("full", {}), ("window_10s", {"tmin": 0, "tmax": 10})):
		times = []
		for _ in range(5):
			view = Repo.open(STORE, **opts).get(rec.path)  # a fresh handle each time
			t = time.perf_counter()
			view.data(**window)
			times.append(time.perf_counter() - t)
		res[f"read_{name}_s"] = sorted(times)[len(times) // 2]

	print(json.dumps(res, indent=1))
	with open(args.out, "w") as f:
		json.dump(res, f, indent=1)
	return 1 if problems else 0


if __name__ == "__main__":  # parallel ingest spawns workers that re-import this file
	sys.exit(main())
