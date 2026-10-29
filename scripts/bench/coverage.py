"""Coverage pass over the development sampler: what neurozarr does with each item.

	python scripts/bench/coverage.py [--sampler .bench_scratch/sampler]

Each manifest row is read the way a user would: a BIDS folder with BidsReader, any
other folder with SmartReader (the "point it at my files" path). The items are
then written to an in-memory store, and every recording is read back and compared
with what mne reads from the source.

Verdict per row, written to coverage.tsv next to manifest.tsv:
	native      every data file decoded (recording / array / table), every recording
	            read back equal to the source
	partial     some decoded, but some files only referenced, unreadable or not stored, or a
	            read-back mismatch
	reference   nothing decoded; files kept only as path references
	unsupported nothing usable: unreadable, not stored, or reading/writing raised
"""

import argparse
import csv
import multiprocessing
import itertools
import logging
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RTOL = 1e-6  # float32 storage; int16-packed EDF-family data is exact


def check(path: Path, bids: bool, store: str) -> dict:
	from neurozarr import Repo
	from neurozarr.items import Array, Attrs, ExternalFile, Recording, Table
	from neurozarr.readers import BidsReader, SmartReader

	out: dict = {"decoded": "", "referenced": 0, "unreadable": 0, "not_stored": 0, "recordings_checked": 0,
				 "max_rel_err": "", "error": ""}
	try:
		reader = BidsReader(path) if bids else SmartReader([path])
		items = list(reader.read())
	except Exception as e:
		out["error"] = f"read: {type(e).__name__}: {e}"[:300]
		return out
	kinds = Counter(type(i).__name__ for i in items if isinstance(i, (Recording, Array, Table)))
	out["decoded"] = " ".join(f"{k}={n}" for k, n in sorted(kinds.items()))
	out["referenced"] = sum(isinstance(i, ExternalFile) for i in items)
	unreadable = getattr(reader, "report", {}).get("unreadable", []) if not bids else []
	out["unreadable"] = len(unreadable)
	out["unreadable_files"] = " | ".join(Path(u["path"]).name for u in unreadable)[:300]
	# BidsReader names what it reads nothing from (README, stimuli/, ...) instead of dropping it
	not_stored = next((i.attrs for i in items if isinstance(i, Attrs) and tuple(i.path) == ("files_not_stored",)), {})
	failed = next((i.attrs for i in items if isinstance(i, Attrs) and tuple(i.path) == ("files_unreadable",)), {})
	out["unreadable"] = max(out["unreadable"], len(failed))  # the attrs also list the scan's
	out["unreadable_files"] = (out["unreadable_files"] + " | " + " | ".join(
		f"{Path(k).name}: {v}" for k, v in failed.items()))[:300] if failed else out["unreadable_files"]
	out["not_stored"] = len(not_stored)
	out["not_stored_files"] = " | ".join(list(not_stored)[:10])[:300]

	try:
		repo = Repo.init(store)
		repo.add(reader)
		repo.commit("coverage")
	except Exception as e:
		out["error"] = f"write: {type(e).__name__}: {e}"[:300]
		return out

	source = {"/".join((*i.prefix, *i.entities.path())): i.raw for i in items if isinstance(i, Recording)}
	worst = 0.0
	try:
		for sub in repo.subjects():
			for view in repo.subject(sub).recordings():
				raw = source.get(view.path)
				if raw is None:
					continue
				got = np.asarray(view.data()[0], dtype="float64")
				want = raw.get_data()
				peak = float(np.abs(want).max()) if want.size else 0.0
				err = float(np.abs(got - want).max()) if got.shape == want.shape and want.size else np.inf
				worst = max(worst, err / peak if peak else err)
				out["recordings_checked"] += 1
	except Exception as e:
		out["error"] = f"read back: {type(e).__name__}: {e}"[:300]
	if out["recordings_checked"]:
		out["max_rel_err"] = f"{worst:.1e}"
	out["_missing_readback"] = len(source) - out["recordings_checked"]
	return out


def verdict(r: dict) -> str:
	decoded = bool(r["decoded"])
	clean = not (r["referenced"] or r["unreadable"] or r.get("not_stored") or r["error"] or r["_missing_readback"]
				 or (r["max_rel_err"] and float(r["max_rel_err"]) > RTOL))
	if decoded and clean:
		return "native"
	if decoded:
		return "partial"
	if r["referenced"] and not r["error"]:
		return "reference"
	return "unsupported"


def _quiet() -> None:
	warnings.filterwarnings("ignore")
	logging.disable(logging.WARNING)
	import mne
	mne.set_log_level("ERROR")


def main(sampler: Path) -> None:
	_quiet()
	with open(sampler / "manifest.tsv") as f:
		rows = list(csv.DictReader(f, delimiter="\t"))
	counter = itertools.count()
	results = []
	for row in rows:
		# one process per item: a reader that segfaults (native code) fails its row, not the run
		with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"), initializer=_quiet) as pool:
			try:
				r = pool.submit(check, sampler / row["path"], row["group"] == "BIDS",
								f"memory://coverage{next(counter)}").result(timeout=600)
			except Exception as exc:
				detail = "process crashed" if isinstance(exc, BrokenProcessPool) else \
					traceback.format_exception_only(exc)[-1].strip()[:300]
				r = {"decoded": "", "referenced": 0, "unreadable": 0, "recordings_checked": 0, "max_rel_err": "",
					 "error": "crash: " + detail, "_missing_readback": 0}
		r["verdict"] = verdict(r)
		results.append({"group": row["group"], "format": row["format"], "path": row["path"], **r})
		print(f"{r['verdict']:<12} {row['group']:<12} {row['format']:<32} {r['decoded'] or '-'}"
			  f"{'  ERR ' + r['error'][:90] if r['error'] else ''}", flush=True)

	cols = ["group", "format", "path", "verdict", "decoded", "referenced", "unreadable", "not_stored",
			"recordings_checked", "max_rel_err", "unreadable_files", "not_stored_files", "error"]
	with open(sampler / "coverage.tsv", "w", newline="") as f:
		w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore")
		w.writeheader()
		w.writerows(results)
	print(Counter(r["verdict"] for r in results))


if __name__ == "__main__":
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--sampler", type=Path, default=ROOT / ".bench_scratch/sampler")
	main(ap.parse_args().sampler)
