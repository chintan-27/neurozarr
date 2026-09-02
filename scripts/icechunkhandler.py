from pathlib import Path

from bidszarr import BidsReader, Repo

HOME_DIR = Path("./")

repo = Repo(HOME_DIR / "zarr")
repo.ingest(BidsReader(HOME_DIR / "BIDS"))
repo.save("Convert BIDS dataset to zarr")
