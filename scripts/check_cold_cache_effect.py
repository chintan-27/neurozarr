"""Does posix_fadvise(DONTNEED) actually make the next read cold on this
filesystem? The harness's cold-cache mode drops the cache that way (no root on a
shared cluster node for drop_caches), and the first cold-cache results came out
barely slower than warm (cold/warm 0.7-1.2x) -- either the cache wasn't really
dropped, or something outside this node's page cache (Lustre client/server
caches) dominates. This reads one large file three ways and reports page-cache
residency alongside each timing:

	read 1 (as found) -> read 2 (immediately again, warm) -> fadvise DONTNEED -> read 3

Residency is measured with `fincore` (util-linux) when present. Interpretation:
  - residency stays high after fadvise  -> fadvise did not evict; cold numbers are warm.
  - residency drops to ~0 but read 3 ~ read 2 -> evicted locally, but a cache
    elsewhere (Lustre) still serves it -- "cold" here means cold only on the node.
  - residency drops AND read 3 >> read 2     -> fadvise gives a genuinely cold read.

	python scripts/check_cold_cache_effect.py datasets/hbn_eeg [--out result.json]
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def largest_file(root: Path) -> Path:
	best, best_size = None, -1
	for dirpath, _, files in os.walk(root):
		for name in files:
			p = Path(dirpath) / name
			try:
				size = p.stat().st_size
			except OSError:
				continue
			if size > best_size:
				best, best_size = p, size
	if best is None:
		raise SystemExit(f"no files under {root}")
	return best


def resident_bytes(path: Path) -> int | None:
	if not shutil.which("fincore"):
		return None
	out = subprocess.run(["fincore", "--bytes", "--noheadings", "--output", "RES", str(path)],
						 capture_output=True, text=True).stdout.split()
	return int(out[0]) if out else None


def read_all(path: Path, block: int = 8 * 1024 * 1024) -> float:
	start = time.perf_counter()
	with path.open("rb", buffering=0) as f:
		while f.read(block):
			pass
	return time.perf_counter() - start


def dontneed(path: Path) -> None:
	fd = os.open(path, os.O_RDONLY)
	try:
		os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
	finally:
		os.close(fd)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("root", type=Path)
	parser.add_argument("--out", type=Path, default=None)
	args = parser.parse_args()

	path = largest_file(args.root)
	size = path.stat().st_size
	steps: list[dict] = []

	def record(label: str, seconds: float | None = None) -> None:
		steps.append({"step": label, "seconds": seconds, "resident_bytes": resident_bytes(path),
					  "mb_per_s": (size / seconds / 1e6) if seconds else None})

	record("start")
	record("read 1 (as found)", read_all(path))
	record("read 2 (warm, immediately after)", read_all(path))
	dontneed(path)
	record("after fadvise DONTNEED")
	record("read 3 (after fadvise)", read_all(path))
	dontneed(path)
	record("after second fadvise")
	record("read 4 (after second fadvise)", read_all(path))

	result = {"file": str(path), "size_bytes": size, "host": os.uname().nodename, "steps": steps}
	print(json.dumps(result, indent=2))
	if args.out:
		args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
