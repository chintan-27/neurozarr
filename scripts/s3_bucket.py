"""Make sure the S3 server (s3_serve.slurm) has bucket "bench", created through S3.

The posix backend treats a directory as a bucket only if it carries the server's
bucket metadata; a directory made by plain file copies (s3_formats.py running first)
is refused ("get bucket acl: no such key"). Such a directory is moved aside, the bucket
created, and its contents moved back -- a rename on one filesystem, nothing re-copied.

	python scripts/s3_bucket.py BUCKET_DIR --endpoint http://HOST:PORT

Credentials come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def main() -> int:
	p = argparse.ArgumentParser()
	p.add_argument("bucket_dir", type=Path)
	p.add_argument("--endpoint", required=True)
	args = p.parse_args()
	s3 = boto3.client("s3", endpoint_url=args.endpoint, aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
					  aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"], region_name="us-east-1",
					  config=boto3.session.Config(retries={"max_attempts": 1}))
	name = args.bucket_dir.name

	def create() -> bool:
		try:
			s3.create_bucket(Bucket=name)
		except (s3.exceptions.BucketAlreadyOwnedByYou, s3.exceptions.BucketAlreadyExists):
			pass
		except ClientError as exc:
			if exc.response["Error"]["Code"] != "InternalError":
				raise
			return False
		return True

	if create():
		print(f"bucket {name}: ready")
		return 0
	aside = args.bucket_dir.with_name(name + ".unregistered")
	args.bucket_dir.rename(aside)
	if not create():
		aside.rename(args.bucket_dir)
		print(f"bucket {name}: could not be created", file=sys.stderr)
		return 1
	for child in aside.iterdir():
		child.rename(args.bucket_dir / child.name)
	aside.rmdir()
	print(f"bucket {name}: registered the existing folder's contents")
	return 0


if __name__ == "__main__":
	sys.exit(main())
