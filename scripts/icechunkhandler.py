from pathlib import Path

from neurozarr import BidsReader, Repo

HOME_DIR = Path("./")

repo = Repo(HOME_DIR / "zarr")
repo.add(BidsReader(HOME_DIR / "BIDS"))
repo.commit("Convert BIDS dataset to zarr")
