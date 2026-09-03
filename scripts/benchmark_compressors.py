import random
import shutil
import tempfile
import time
from pathlib import Path

import icechunk as ic
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle, ZstdCodec
from zarr.codecs.numcodecs import BZ2, LZ4, LZMA, Delta

from bidszarr import BidsReader, CodecConfig, Repo

verbose = False
BIDS_DIR = Path("./BIDS")
REPO_DIR = Path(tempfile.mkdtemp())

ZSTD19 = [ZstdCodec(level=19)]

# (filters, compressors, bitroundK, dtype) — every config tried during compressor exploration
CODECS = {
	"int16+zstd-1": ([], [ZstdCodec(level=1)], 0, "int16"),
	"int16+zstd-9": ([], [ZstdCodec(level=9)], 0, "int16"),
	"int16+zstd-19 (current default)": ([], ZSTD19, 0, "int16"),
	"float16+zstd-19": ([], ZSTD19, 0, "float16"),
	"int16+blosc-zstd-5-shuffle": ([], [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle, typesize=2)], 0, "int16"),
	"int16+blosc-zstd-5-bitshuffle": ([], [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle, typesize=2)], 0, "int16"),
	"int16+blosc-zstd-9-bitshuffle": ([], [BloscCodec(cname="zstd", clevel=9, shuffle=BloscShuffle.bitshuffle, typesize=2)], 0, "int16"),
	"int16+blosc-zstd-5-shuffle-typesize1 (user's config)": ([], [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle, typesize=1)], 0, "int16"),
	"int16+delta+zstd-19": ([Delta(dtype="int16")], ZSTD19, 0, "int16"),
	"int16+lzma": ([], [LZMA()], 0, "int16"),
	"int16+bz2": ([], [BZ2()], 0, "int16"),
	"int16+lz4": ([], [LZ4()], 0, "int16"),
	"int16+zstd-19+bitround-k3 (lossy)": ([], ZSTD19, 3, "int16"),
	"int16+zstd-19+bitround-k5 (lossy)": ([], ZSTD19, 5, "int16"),
	"int16+zstd-19+bitround-k7 (lossy)": ([], ZSTD19, 7, "int16"),
}

def _dir_size(path: Path) -> int:
	return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def runQueries(sub: zarr.Group) -> dict:
	"""sub is sub-001's own repo root (one repo per subject -- no shared root to index into)."""
	random.seed(0)
	runGroups = [(p, n) for p, n in sub.members(max_depth=None) if isinstance(n, zarr.Group) and "data" in n]
	biggest = max(runGroups, key=lambda pn: pn[1]["data"].shape[-1])

	t0 = time.perf_counter()
	biggest[1]["data"][:]
	fullRunS = time.perf_counter() - t0

	nSamp = biggest[1]["data"].shape[-1]
	start = random.randint(0, max(0, nSamp - 1000))
	t0 = time.perf_counter()
	biggest[1]["data"][:, start:start + 1000]
	sliceMs = (time.perf_counter() - t0) * 1000

	# scan every channels table for sub-001, filter rows whose first column contains "TD"
	t0 = time.perf_counter()
	for path, node in sub.members(max_depth=None):
		if isinstance(node, zarr.Array) and path.endswith("channels"):
			arr = node[:]
			_ = [row for row in arr if "TD" in row[0]]
	tableScanS = time.perf_counter() - t0

	# cross-run aggregate: mean abs amplitude over every acq-TD run for sub-001
	t0 = time.perf_counter()
	total, count = 0.0, 0
	for p, n in runGroups:
		if "TD" in p:
			vals = n["data"][:].astype(np.float64)
			total += np.abs(vals).sum()
			count += vals.size
	aggregateS = time.perf_counter() - t0

	# task scan: sum every BrainSenseSurvey run's data across sub-001's sessions
	t0 = time.perf_counter()
	for path, node in sub.members(max_depth=None):
		if isinstance(node, zarr.Array) and path.endswith("/data") and "BrainSenseSurvey" in path:
			_ = node[:].astype(np.float64).sum()
	taskScanS = time.perf_counter() - t0

	# random scattered access: small slice from 20 random data arrays within sub-001
	allDataArrays = [n for p, n in sub.members(max_depth=None) if isinstance(n, zarr.Array) and p.endswith("/data")]
	random.shuffle(allDataArrays)
	t0 = time.perf_counter()
	for arr in allDataArrays[:20]:
		n = arr.shape[-1]
		s = random.randint(0, max(0, n - 100))
		_ = arr[:, s:s + 100]
	randomAccessMs = (time.perf_counter() - t0) * 1000

	t0 = time.perf_counter()
	for _, node in sub.members(max_depth=None):
		_ = node.attrs.asdict()
	attrsS = time.perf_counter() - t0

	return dict(fullRunS=fullRunS, sliceMs=sliceMs, tableScanS=tableScanS,
		aggregateS=aggregateS, taskScanS=taskScanS, randomAccessMs=randomAccessMs, attrsS=attrsS)

def main():
	cols = ["write_s", "commit_s", "full_run_s", "slice_ms", "table_scan_s", "aggregate_s", "task_scan_s", "rand_ms", "attrs_s", "stored_MB"]
	print(f"{'codec':45} " + " ".join(f"{c:>12}" for c in cols))
	for name, (filters, compressors, bitroundK, dtype) in CODECS.items():
		# one repo per subject means no shared store to branch across configs --
		# each codec gets its own fresh base_path instead.
		codecDir = REPO_DIR / name
		repo = Repo(codecDir, codec=CodecConfig(filters, compressors, bitroundK, dtype))

		t0 = time.perf_counter()
		repo.ingest(BidsReader(BIDS_DIR))
		writeTime = time.perf_counter() - t0

		t0 = time.perf_counter()
		repo.save(f"convert with {name}")
		commitTime = time.perf_counter() - t0

		storedBytes = _dir_size(codecDir)

		subStorage = ic.local_filesystem_storage(str(codecDir / "sub-001"))
		subRepo = ic.Repository.open(subStorage)
		readSession = subRepo.readonly_session("main")
		subRoot = zarr.open_group(store=readSession.store, mode="r")
		q = runQueries(subRoot)

		if verbose:
			print(subRoot.tree())
		vals = [writeTime, commitTime, q["fullRunS"], q["sliceMs"], q["tableScanS"],
			q["aggregateS"], q["taskScanS"], q["randomAccessMs"], q["attrsS"], storedBytes / 1e6]
		print(f"{name:45} " + " ".join(f"{v:12.2f}" for v in vals))

	shutil.rmtree(REPO_DIR)

if __name__ == "__main__":
	main()
