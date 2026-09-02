import random
import time
from pathlib import Path
import icechunk as ic
import zarr

REPO_DIR = Path("zarr")

def timeIt(fn):
	t0 = time.perf_counter()
	result = fn()
	return result, time.perf_counter() - t0

def main():
	storage = ic.local_filesystem_storage(str(REPO_DIR))
	repo = ic.Repository.open(storage)
	session = repo.readonly_session(branch="main")
	root = zarr.open_group(store=session.store, mode="r")

	sub = root["sub-001"]
	runGroups = [(p, n) for p, n in sub.members(max_depth=None) if isinstance(n, zarr.Group) and "data" in n]

	biggest = max(runGroups, key=lambda pn: pn[1]["data"].shape[-1])
	_, t = timeIt(lambda: biggest[1]["data"][:])
	print(f"single full-run read ({biggest[0]}, shape={biggest[1]['data'].shape}): {t*1000:.1f} ms")

	n = biggest[1]["data"].shape[-1]
	start = random.randint(0, max(0, n - 1000))
	_, t = timeIt(lambda: biggest[1]["data"][:, start:start + 1000])
	print(f"1000-sample slice read from same run: {t*1000:.1f} ms")

	chArr = runGroups[0][1]["channels"]
	_, t = timeIt(lambda: chArr[:])
	print(f"channels table decode ({chArr.shape}): {t*1000:.1f} ms")

	ses = sub["ses-20221007"]
	def readSession():
		for path, node in ses.members(max_depth=None):
			if isinstance(node, zarr.Array) and path.endswith("data"):
				_ = node[:]
	_, t = timeIt(readSession)
	print(f"full session read (ses-20221007, all runs): {t:.2f} s")

	def readSubject():
		for path, node in sub.members(max_depth=None):
			if isinstance(node, zarr.Array) and path.endswith("data"):
				_ = node[:]
	_, t = timeIt(readSubject)
	print(f"full subject read (sub-001, all sessions): {t:.2f} s")

	def scanAttrs():
		count = 0
		for _, node in root.members(max_depth=None):
			_ = node.attrs.asdict()
			count += 1
		return count
	count, t = timeIt(scanAttrs)
	print(f"attrs-only scan, no chunk data ({count} nodes, full dataset): {t:.2f} s")

if __name__ == "__main__":
	main()
