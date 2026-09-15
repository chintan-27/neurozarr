"""Query benchmarks against a converted store -- any dataset, not just BRAVO's.

	python scripts/benchmark_query.py ./study.zarr [sub_id] [--sample N]

Each subject is its own Icechunk repository, so there is no single root to index
into with root["sub-001"]; goes through Repo/Subject/Visit instead. `--sample` caps
how many recordings a "read everything" query actually reads, so this stays a quick
probe on a 100GB dataset instead of reading the whole thing.
"""

import argparse
import random
import time

from neurozarr import Repo


def timeIt(fn):
	t0 = time.perf_counter()
	result = fn()
	return result, time.perf_counter() - t0


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("store")
	parser.add_argument("sub_id", nargs="?", default=None, help="default: the store's first subject")
	parser.add_argument("--sample", type=int, default=50, help="cap on recordings touched per query")
	args = parser.parse_args()

	repo = Repo.open(args.store)
	sub_id = args.sub_id or next(iter(repo.subjects()), None)
	if sub_id is None:
		print(f"{args.store}: no subjects")
		return
	subject = repo.subject(sub_id)
	recordings = subject.recordings()
	if not recordings:
		print(f"{args.store}: no recordings for {sub_id}")
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

	# one session, up to `sample` runs in it -- session-less subjects (no ses- segment
	# in a recording's path at all, see Subject's own add_recording) have none
	segments = biggest.path.split("/")
	ses_id = segments[1] if len(segments) > 1 and segments[1].startswith("ses-") else None
	if ses_id:
		visit = subject.visit(ses_id)
		visit_sample = visit.recordings()[:args.sample]
		_, t = timeIt(lambda: [r.data() for r in visit_sample])
		print(f"session read ({ses_id}, {len(visit_sample)} of {len(visit.recordings())} runs): {t:.2f} s")

	subject_sample = recordings[:args.sample]
	_, t = timeIt(lambda: [r.data() for r in subject_sample])
	print(f"subject read ({sub_id}, {len(subject_sample)} of {len(recordings)} runs): {t:.2f} s")

	def scanAttrs():
		count = 0
		for sub in repo.subjects():
			for _, node in repo.root_of(sub).members(max_depth=None):
				_ = node.attrs.asdict()
				count += 1
		return count

	count, t = timeIt(scanAttrs)
	print(f"attrs-only scan, no chunk data ({count} nodes, full dataset): {t:.2f} s")

	# cross-subject entity search -- reuses whatever entities the biggest run
	# actually has (e.g. task/acq for BRAVO, task alone for a dataset with no
	# acquisition split) rather than assuming any dataset-specific label
	search_entities = {k: v for k, v in biggest.entities.items() if k in ("task", "acq")}
	if search_entities:
		found, t = timeIt(lambda: list(repo.find(**search_entities)))
		query = ", ".join(f"{k}={v}" for k, v in search_entities.items())
		print(f"cross-subject find({query}): {len(found)} hits in {t:.2f} s")


if __name__ == "__main__":
	main()
