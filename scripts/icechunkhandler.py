import os
from pathlib import Path
from tqdm import tqdm
import icechunk as ic
from bidstozarr import getBidsAsZarr

HOME_DIR = Path("./")
ZARR_DIRECTORY = HOME_DIR / "zarr"

storage = ic.local_filesystem_storage(str(ZARR_DIRECTORY))
repo = ic.Repository.create(storage)

session = repo.writable_session("main")
getBidsAsZarr(session)
session.commit("Convert BIDS dataset to zarr")

