"""Write a store to object storage instead of local disk.

Nothing about the API changes -- only the target you hand Repo().

    python examples/cloud_storage.py
"""

import icechunk

from bidszarr import BidsReader, Repo


def to_s3():
	# credentials come from the environment / instance role by default
	repo = Repo("s3://my-bucket/studies/dbs", region="us-east-1")
	repo.ingest(BidsReader("./BIDS"))
	repo.save("convert to s3")
	return repo


def to_gcs():
	return Repo("gs://my-bucket/studies/dbs")


def to_azure():
	return Repo("az://myaccount/mycontainer/studies/dbs")


def with_explicit_credentials():
	"""When you need full control, build the icechunk Storage yourself and pass it."""
	storage = icechunk.s3_storage(
		bucket="my-bucket",
		prefix="studies/dbs/sub-001",
		region="us-east-1",
		access_key_id="...",
		secret_access_key="...",
	)
	return Repo(storage)


def in_memory():
	"""No storage at all -- useful in tests and scratch work."""
	repo = Repo("memory://scratch")
	print("in-memory store ready:", repo.target)
	return repo


if __name__ == "__main__":
	in_memory()  # the only one that runs without cloud credentials
	print("edit this file and call to_s3() / to_gcs() / to_azure() with your own bucket")
