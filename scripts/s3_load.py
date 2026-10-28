"""Load a dataset into the S3 server started by s3_serve.slurm, once: skipped if the store
already exists in the bucket (the server's data persists on /blue between jobs).

	python scripts/s3_load.py SOURCE NAME --endpoint http://HOST:PORT [--workers 8]

Credentials come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
"""

import argparse
import os
import sys
import time

import boto3

from neurozarr import Repo


def main() -> int:
	p = argparse.ArgumentParser()
	p.add_argument("source")
	p.add_argument("name")
	p.add_argument("--endpoint", required=True)
	p.add_argument("--workers", type=int, default=8)
	args = p.parse_args()

	key, secret = os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"]
	s3 = boto3.client("s3", endpoint_url=args.endpoint, aws_access_key_id=key, aws_secret_access_key=secret,
					  region_name="us-east-1")
	try:  # the folder may already exist: s3_formats.py writes into it before this runs
		s3.create_bucket(Bucket="bench")
	except (s3.exceptions.BucketAlreadyOwnedByYou, s3.exceptions.BucketAlreadyExists):
		pass
	if s3.list_objects_v2(Bucket="bench", Prefix=f"{args.name}/", MaxKeys=1).get("KeyCount"):
		print(f"s3://bench/{args.name} already loaded; skipping", flush=True)
		return 0

	opts = dict(endpoint_url=args.endpoint, allow_http=True, force_path_style=True, region="us-east-1",
				access_key_id=key, secret_access_key=secret)
	t = time.perf_counter()
	repo = Repo.init(f"s3://bench/{args.name}", **opts)
	repo.add(args.source, workers=args.workers)
	repo.commit(f"load {args.source}")
	print(f"loaded {args.source} -> s3://bench/{args.name} in {time.perf_counter() - t:.1f} s", flush=True)
	return 0


if __name__ == "__main__":  # parallel ingest spawns workers that re-import this file
	sys.exit(main())
