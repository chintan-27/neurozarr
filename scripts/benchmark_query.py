"""Query benchmarks against a converted store.

	python scripts/benchmark_query.py ./study.zarr

Ported to the neurozarr read API: each subject is its own Icechunk repository now,
so there is no single root to index into with root["sub-001"].
"""

import random
import sys
import time

import zarr

from neurozarr import Repo

STORE = "zarr"
SUBJECT = "sub-001"


def timeIt(fn):
	t0 = time.perf_counter()
	result = fn()
	return result, time.perf_counter() - t0


def main(store=STORE, sub_id=SUBJECT):
	repo = Repo(store)
	subject = repo.subject(sub_id)
	recordings = subject.recordings()
	if not recordings:
		print(f"{store}: no recordings for {sub_id}")
		return

	biggest = max(recordings, key=lambda r: r.shape[-1])
	_, t = timeIt(lambda: biggest.data())
	print(f"single full-run read ({biggest.path}, shape={biggest.shape}): {t*1000:.1f} ms")

	n = biggest.shape[-1]
	start = random.randint(0, max(0, n - 1000))
	_, t = timeIt(lambda: biggest.data(start=start, stop=start + 1000))
	print(f"1000-sample slice read from same run: {t*1000:.1f} ms")

	_, t = timeIt(lambda: biggest.channels())
	print(f"channels table decode: {t*1000:.1f} ms")

	# one session, every run in it
	ses_id = biggest.path.split("/")[1]
	visit = subject.visit(ses_id)
	_, t = timeIt(lambda: [r.data() for r in visit.recordings()])
	print(f"full session read ({ses_id}, {len(visit.recordings())} runs): {t:.2f} s")

	_, t = timeIt(lambda: [r.data() for r in recordings])
	print(f"full subject read ({sub_id}, {len(recordings)} runs): {t:.2f} s")

	def scanAttrs():
		count = 0
		for sub in repo.subjects():
			for _, node in repo.root_of(sub).members(max_depth=None):
				_ = node.attrs.asdict()
				count += 1
		return count

	count, t = timeIt(scanAttrs)
	print(f"attrs-only scan, no chunk data ({count} nodes, full dataset): {t:.2f} s")

	# cross-subject entity search, which the old single-repo layout had no API for
	found, t = timeIt(lambda: list(repo.find(task="BrainSenseStream", acq="TD")))
	print(f"cross-subject find(task=BrainSenseStream, acq=TD): {len(found)} hits in {t:.2f} s")


if __name__ == "__main__":
	main(*sys.argv[1:])
