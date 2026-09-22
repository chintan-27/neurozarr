"""Convert RNS_Epilepsy-iBIDS by hand -- Repo/Subject/Visit calls you write
yourself, walking the real directory tree, instead of `repo.ingest(BidsReader(...))`.

	python scripts/convert_rns_explicit.py datasets/rns .bench_scratch/rns_explicit --subjects sub-01

This is what "explicit" means in the docs/README sense (create_subject ->
add_visit -> add_recording), applied to a real dataset instead of a couple
of hand-built recordings. One thing it can't reach through the public API:
a *named* sidecar table ("channels", "events") -- Visit.add()/add_recording()
have no such parameter, only BidsReader's own reader-side Table(entities,
suffix, ...) construction does. So this uses that same low-level path
(repo._dispatch) for channels.tsv/events.tsv specifically, with a comment
at that call explaining why -- everything else goes through the real
public API.
"""

import argparse
import time
from pathlib import Path

import mne
import pandas as pd

from neurozarr import Repo
from neurozarr.entities import Entities, parse_entities
from neurozarr.items import Table
from neurozarr.writer import ExistingPolicy


def convert_visit(repo: Repo, visit, ieeg_dir: Path, existing: ExistingPolicy) -> int:
	count = 0
	for edf_path in sorted(ieeg_dir.glob("*_ieeg.edf")):
		stem = edf_path.stem.removesuffix("_ieeg")
		# parse_entities parses the whole stem, sub-/ses- included -- passing those
		# through unfiltered leaks them into the recording's path as extra entities
		# (e.g. "task-seizure_run-01_ses-01_sub-01" instead of "task-seizure_run-01"),
		# since visit.add_recording already places the recording under this sub/ses
		entities = {k: v for k, v in parse_entities(stem).items() if k not in ("sub", "ses")}

		raw = mne.io.read_raw(edf_path, preload=False, verbose=False)
		visit.add_recording(raw, existing=existing, **entities)

		# The real per-run sidecars. add_recording's own _add_annotations only
		# auto-derives "events" from raw.annotations as a fallback -- these are
		# the dataset's *actual* channels/events tables, and there's no public
		# add_channels()/add_events() naming them explicitly, so this reaches
		# one level down to the same Table+dispatch a Reader itself calls.
		full_entities = Entities(visit.sub_id, visit.ses_id, "ieeg", entities)
		for suffix in ("channels", "events"):
			tsv_path = edf_path.with_name(f"{stem}_{suffix}.tsv")
			if tsv_path.exists():
				table = Table(full_entities, suffix, pd.read_csv(tsv_path, sep="\t"), {})
				repo._dispatch(table, existing)

		count += 1

	# One electrodes.tsv per session, sibling to the runs but describing no
	# particular one -- same low-level path as channels/events, entities scoped
	# to just (sub, ses, "ieeg") since it carries no task/run of its own.
	electrodes_path = next(ieeg_dir.glob("*_electrodes.tsv"), None)
	if electrodes_path is not None:
		session_entities = Entities(visit.sub_id, visit.ses_id, "ieeg", {})
		table = Table(session_entities, "electrodes", pd.read_csv(electrodes_path, sep="\t"), {})
		repo._dispatch(table, existing)

	return count


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("source", help="RNS dataset root, e.g. datasets/rns")
	parser.add_argument("dest", help="output store")
	parser.add_argument("--subjects", default=None, help="comma-separated subject ids, e.g. sub-01,sub-02")
	parser.add_argument("--existing", default="replace", choices=[p.value for p in ExistingPolicy])
	args = parser.parse_args()

	source = Path(args.source)
	existing = ExistingPolicy(args.existing)
	wanted = set(args.subjects.split(",")) if args.subjects else None

	repo = Repo.create(args.dest)
	t0 = time.perf_counter()
	total_recordings = 0

	for sub_dir in sorted(source.glob("sub-*")):
		sub_id = sub_dir.name
		if wanted is not None and sub_id not in wanted:
			continue
		subject = repo.create_subject(sub_id)
		print(f"{sub_id}:")

		for ses_dir in sorted(sub_dir.glob("ses-*")):
			ses_id = ses_dir.name
			ieeg_dir = ses_dir / "ieeg"
			if not ieeg_dir.is_dir():
				continue
			visit = subject.add_visit(ses_id)
			n = convert_visit(repo, visit, ieeg_dir, existing)
			total_recordings += n
			print(f"  {ses_id}: {n} recording(s)")

	repo.save("explicit RNS conversion")
	print(f"\n{total_recordings} recordings, {len(repo.subjects())} subject(s), "
		  f"{time.perf_counter() - t0:.1f}s -> {args.dest}")


if __name__ == "__main__":
	main()
