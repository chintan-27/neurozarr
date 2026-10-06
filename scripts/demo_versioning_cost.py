"""Two correctness claims from one real correction, logged instead of printed
and discarded: how many bytes a one-sample correction actually adds (vs. a
full raw-BIDS re-copy), and whether a pinned snapshot stays stable while
`main` reflects the fix.

	.venv/bin/python scripts/demo_versioning_cost.py BIDS/sub-001 .bench_scratch/paper_demos/versioning_cost

Hardware doesn't matter for either claim -- both are about what Icechunk's
copy-on-write model does on disk and what a pinned read returns, not about
speed -- so this is meant to run locally, not on HiPerGator.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neurozarr import Repo
from neurozarr.writer import ExistingPolicy


def _dir_size(path: Path) -> int:
	return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("source", type=Path, help="one BIDS subject directory")
	parser.add_argument("dest", type=Path, help="scratch store path, wiped first")
	parser.add_argument("--out", type=Path, default=Path(".bench_scratch/paper_logs/versioning_cost.json"))
	args = parser.parse_args()

	if args.dest.exists():
		shutil.rmtree(args.dest)
	args.dest.parent.mkdir(parents=True, exist_ok=True)
	args.out.parent.mkdir(parents=True, exist_ok=True)

	sub_id = args.source.name
	from neurozarr.readers import BidsReader
	repo = Repo.create(args.dest)
	repo.ingest(BidsReader(args.source.parent, subjects=[sub_id]))
	repo.save("initial conversion")

	subject = repo.subject(sub_id)
	recordings = subject.recordings()
	if not recordings:
		raise RuntimeError(f"no recordings found for {sub_id} -- can't run the demo")
	target = recordings[0]
	entities = dict(target.entities)  # already just the extra BIDS entities (task/run/...)
	ses_id = next((seg for seg in target.path.split("/") if seg.startswith("ses-")), None)

	before_bytes = _dir_size(args.dest)
	baseline_data, _ = target.data()
	baseline_mean = baseline_data.mean(axis=-1)

	before_history = repo.history(sub_id)
	pre_correction_snapshot = before_history[0][0]

	# Re-read the same source file, preload it, nudge one real sample, and
	# write it back under the same entities -- ExistingPolicy.REPLACE is what
	# makes this a correction rather than a duplicate.
	source_ieeg_dir = args.source / ses_id / "ieeg" if ses_id else args.source / "ieeg"
	edf_candidates = sorted(source_ieeg_dir.glob("*_ieeg.edf"))
	if not edf_candidates:
		raise RuntimeError(f"no source EDF found under {source_ieeg_dir}")
	raw = mne.io.read_raw(edf_candidates[0], preload=True, verbose=False)
	# One LSB step in physical units -- any larger offset risks pushing an
	# already-near-ceiling sample outside the channel's int16 calibration
	# range (confirmed by hitting exactly that ValidationError on a first try).
	extras = raw._raw_extras[0]
	delta = float(extras["cal"][0] * extras["units"][0])
	raw._data[0, 0] += delta

	visit = subject.visit(ses_id) if ses_id else subject
	visit.add_recording(raw, existing=ExistingPolicy.REPLACE, **entities)
	repo.save("correct one sample")

	after_bytes = _dir_size(args.dest)
	after_history = repo.history(sub_id)
	post_correction_snapshot = after_history[0][0]

	# Pinned read: the OLD snapshot id must still return the OLD value.
	pinned_recordings = subject.recordings(version=pre_correction_snapshot)
	pinned_target = next(r for r in pinned_recordings if r.path == target.path)
	pinned_data, _ = pinned_target.data()
	pinned_mean = pinned_data.mean(axis=-1)

	# main: the current read must reflect the correction.
	current_recordings = subject.recordings()
	current_target = next(r for r in current_recordings if r.path == target.path)
	current_data, _ = current_target.data()
	current_mean = current_data.mean(axis=-1)

	source_bytes = sum(f.stat().st_size for f in args.source.rglob("*") if f.is_file())

	result = {
		"subject": sub_id,
		"recording_path": target.path,
		"store_bytes_before": before_bytes,
		"store_bytes_after": after_bytes,
		"store_bytes_added": after_bytes - before_bytes,
		"source_bytes_full_recopy_equivalent": source_bytes,
		"ratio_fewer_bytes": round(source_bytes / max(1, after_bytes - before_bytes), 1),
		"pre_correction_snapshot": pre_correction_snapshot,
		"post_correction_snapshot": post_correction_snapshot,
		"pinned_snapshot_mean_matches_baseline": bool(np.allclose(pinned_mean, baseline_mean, atol=1e-12)),
		"main_mean_reflects_correction": bool(not np.allclose(current_mean, baseline_mean, atol=1e-12)),
		"baseline_mean_ch0": float(baseline_mean[0]),
		"pinned_mean_ch0": float(pinned_mean[0]),
		"current_mean_ch0": float(current_mean[0]),
		"delta_applied_volts": delta,
	}
	args.out.write_text(json.dumps(result, indent=2))
	print(json.dumps(result, indent=2))
	print(f"\nwritten to {args.out}")


if __name__ == "__main__":
	main()
