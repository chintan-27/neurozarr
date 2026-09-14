"""Step-by-step demo of neurozarr, run top to bottom in a terminal.

	python examples/demo_stepbystep.py

Uses the real BRAVO BIDS export at ./BIDS (gitignored, present on this
machine only). Every write goes to ./zarr/demo_*.zarr (also gitignored,
scratch output).

1. Build one subject explicitly, one file at a time -- no reader.
2. Convert the same subject automatically with BidsReader.
3. Convert a couple of files through ManifestReader instead.
4. Convert several subjects in bulk, one process each, and report the numbers.

Every step runs with verbose=True, so neurozarr's own progress output -- what
it's writing, how big, and how much memory it's holding right before the
write -- does most of the talking.
"""

import functools
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import mne  # type: ignore[import-untyped]
import pandas as pd

from neurozarr import Repo
from neurozarr.log import logger as neurozarr_logger, set_verbosity
from neurozarr.parallel import convert_parallel
from neurozarr.readers import BidsReader, ManifestReader

print = functools.partial(print, flush=True)  # correct interleaving with neurozarr's own log lines below

# neurozarr's verbose logging defaults to stderr, same as any well-behaved
# library; retargeted to stdout here purely so this demo's own narration and
# the package's progress output interleave correctly in one terminal or log file.
set_verbosity(logging.INFO)
for _handler in neurozarr_logger.handlers:
	if isinstance(_handler, logging.StreamHandler):
		_handler.stream = sys.stdout

SOURCE = Path("BIDS")
SUBJECT = "sub-004"


def banner(title: str) -> None:
	print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def dir_size_mb(path: Path) -> float:
	total = 0
	for root, _dirs, files in os.walk(path, followlinks=True):
		for name in files:
			total += os.path.getsize(os.path.join(root, name))
	return total / 1e6


def step1_explicit() -> None:
	banner("STEP 1 -- explicit construction, one file at a time (no reader)")
	dest = Path("zarr/demo_explicit.zarr")
	shutil.rmtree(dest, ignore_errors=True)

	ses_dir = sorted((SOURCE / SUBJECT).glob("ses-*"))[0]
	edf = sorted((ses_dir / "ieeg").glob("*task-ChronicLFP_run-2_ieeg.edf"))[0]
	therapy_tsv = sorted((ses_dir / "beh").glob("*task-TherapyHistory_beh.tsv"))[0]

	repo = Repo.create(str(dest), verbose=True)
	subject = repo.create_subject(SUBJECT, attrs={"note": "added via explicit construction for the demo"})
	visit = subject.add_visit(ses_dir.name)

	print(f"\n-- reading {edf.name} with mne ourselves, then handing it to add_recording --")
	raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
	visit.add_recording(raw, task="ChronicLFP", run=2)

	print(f"\n-- reading {therapy_tsv.name} with pandas ourselves, then handing it to add_behavioral_table --")
	therapy_df = pd.read_csv(therapy_tsv, sep="\t")
	visit.add_behavioral_table(therapy_df, task="TherapyHistory")

	snapshot = repo.save("built sub-004 explicitly")
	print(f"\ndone -- {dest} now holds 1 recording + 1 table, snapshot {snapshot}")

	readback = Repo.open(str(dest)).subject(SUBJECT).visit(ses_dir.name)
	print(f"read back: {len(readback.recordings())} recording(s), {len(readback.tables())} table(s)")


def step2_bids_reader() -> None:
	banner("STEP 2 -- the same subject, automatically, with BidsReader")
	dest = Path("zarr/demo_bids.zarr")
	shutil.rmtree(dest, ignore_errors=True)

	t0 = time.perf_counter()
	repo = Repo.create(str(dest), verbose=True)
	repo.ingest(BidsReader(str(SOURCE), subjects=[SUBJECT]))
	snapshot = repo.save("converted sub-004 with BidsReader")
	elapsed = time.perf_counter() - t0

	readback = Repo.open(str(dest)).subject(SUBJECT)
	visit = readback.visit(readback.visits()[0])
	print(
		f"\ndone -- {len(visit.recordings())} recordings + {len(visit.tables())} tables "
		f"(BIDS discovers every session, sidecar and datatype on its own) in {elapsed:.1f}s, "
		f"snapshot {snapshot}"
	)


def step3_manifest_reader() -> None:
	banner("STEP 3 -- a couple of the same files, through ManifestReader instead")
	dest = Path("zarr/demo_manifest.zarr")
	shutil.rmtree(dest, ignore_errors=True)

	ses_dir = sorted((SOURCE / SUBJECT).glob("ses-*"))[0]
	manifest = pd.DataFrame([
		{
			"path": str(sorted((ses_dir / "ieeg").glob("*task-ChronicLFP_run-4_ieeg.edf"))[0]),
			"sub": SUBJECT, "ses": ses_dir.name, "datatype": "ieeg", "task": "ChronicLFP", "run": 4,
		},
		{
			"path": str(sorted((ses_dir / "beh").glob("*task-Impedance_beh.tsv"))[0]),
			"sub": SUBJECT, "ses": ses_dir.name, "datatype": "beh", "task": "Impedance",
			"name": "impedance",
		},
	])
	print("manifest:\n", manifest.to_string(index=False))

	repo = Repo.create(str(dest), verbose=True)
	repo.ingest(ManifestReader(manifest))
	snapshot = repo.save("converted 2 files with ManifestReader")

	readback = Repo.open(str(dest)).subject(SUBJECT)
	visit = readback.visit(readback.visits()[0])
	print(f"\ndone -- {len(visit.recordings())} recording(s), {len(visit.tables())} table(s), snapshot {snapshot}")


def step4_bulk_and_benchmark() -> None:
	banner("STEP 4 -- bulk conversion, one process per subject, then the numbers")
	dest = Path("zarr/demo_bulk.zarr")
	staging = Path("zarr/_demo_bulk_src")
	shutil.rmtree(dest, ignore_errors=True)

	subjects = sorted(p.name for p in SOURCE.glob("sub-*") if p.is_dir() and p.name != "sub-001")
	print(f"converting {len(subjects)} subjects in parallel (skipping sub-001, the 222 MB one, for demo speed): "
		f"{', '.join(subjects)}")

	shutil.rmtree(staging, ignore_errors=True)
	staging.mkdir(parents=True)
	for name in ("dataset_description.json", "participants.json", "participants.tsv", "events.json"):
		src = SOURCE / name
		if src.exists():
			shutil.copy(src, staging / name)
	for sub_id in subjects:
		os.symlink((SOURCE / sub_id).resolve(), staging / sub_id)

	source_mb = sum(dir_size_mb(SOURCE / s) for s in subjects)
	t0 = time.perf_counter()
	counts = convert_parallel(str(staging), str(dest), workers=len(subjects), verbose=True)
	elapsed = time.perf_counter() - t0
	store_mb = dir_size_mb(dest)
	total_recordings = sum(counts.values())

	print(f"\n{'-' * 72}\nBENCHMARK\n{'-' * 72}")
	for sub_id, n in sorted(counts.items()):
		print(f"  {sub_id}: {n} recordings")
	print(f"  total: {total_recordings} recordings, {len(subjects)} worker processes")
	print(f"  wall time: {elapsed:.1f}s ({total_recordings / elapsed:.1f} recordings/s)")
	print(f"  source on disk: {source_mb:.1f} MB  ->  store on disk: {store_mb:.1f} MB "
		f"({source_mb / store_mb:.2f}x)")


if __name__ == "__main__":
	step1_explicit()
	step2_bids_reader()
	step3_manifest_reader()
	step4_bulk_and_benchmark()
