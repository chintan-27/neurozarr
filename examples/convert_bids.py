"""Convert an existing BIDS dataset into a Zarr/Icechunk store.

    python examples/convert_bids.py ./BIDS ./study.zarr
"""

import logging
import sys

from bidszarr import BidsReader, Repo, set_verbosity


def main(source="./BIDS", dest="./study.zarr"):
	set_verbosity(logging.INFO)

	repo = Repo(dest)
	repo.ingest(BidsReader(source))
	repo.save(f"convert {source}")
	repo.tag("v1")

	print(f"{dest}: {len(repo.subjects())} subjects -> {', '.join(repo.subjects())}")


if __name__ == "__main__":
	main(*sys.argv[1:])
