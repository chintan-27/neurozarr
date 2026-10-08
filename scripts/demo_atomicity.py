"""Demonstrate, don't time: what a crashed writer leaves behind, plain Zarr
vs. Icechunk. Synthetic data, small and fast on purpose -- the claim is about
a commit boundary existing or not, which has nothing to do with dataset size
or hardware speed, so this runs locally.

Rather than race a real SIGKILL against an in-flight write (flaky, and the
outcome it would show is identical to this), the "crash" is simulated
deterministically: write some items, then stop without finalizing -- for
plain Zarr that means never writing a completion marker (there isn't one to
write), for Icechunk it means never calling repo.commit(). That's the same
state a real crash at that exact point would leave; this way it's
reproducible instead of timing-dependent.

	.venv/bin/python scripts/demo_atomicity.py
"""

import json
import shutil
import sys
from pathlib import Path

import mne
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neurozarr import Repo
from neurozarr.writer import ExistingPolicy


def _synthetic_raw(seed: int) -> "mne.io.RawArray":
	info = mne.create_info(["LFP_L", "LFP_R"], sfreq=250.0, ch_types="eeg")
	rng = np.random.default_rng(seed)
	return mne.io.RawArray(rng.normal(scale=1e-5, size=(2, 500)), info, verbose=False)


def plain_zarr_crash(dest: Path) -> dict:
	"""Write 4 recordings directly to a plain zarr store (no Icechunk), crash
	after 2 complete + 1 mid-write (some chunks of its array written, the
	group's own metadata already visible)."""
	if dest.exists():
		shutil.rmtree(dest)
	root = zarr.open_group(store=zarr.storage.LocalStore(str(dest)), mode="a")

	for i in range(2):
		g = root.create_group(f"rec-{i}")
		raw = _synthetic_raw(i)
		arr = g.create_array("data", shape=raw.get_data().shape, dtype="float32", chunks=(2, 100))
		arr[:, :] = raw.get_data().astype("float32")

	# rec-2: group + array metadata created (so it's visible to a reader as
	# "present"), but only the first chunk of samples is ever written --
	# simulates the crash landing mid-array-write, which is the dangerous
	# case: a reader sees a complete-looking item with silently wrong data.
	g = root.create_group("rec-2")
	raw = _synthetic_raw(2)
	full = raw.get_data().astype("float32")
	arr = g.create_array("data", shape=full.shape, dtype="float32", chunks=(2, 100))
	arr[:, :100] = full[:, :100]
	# "crash" here -- samples 100:500 never written, rec-3 never started

	# What a reader sees right now, with no idea a crash happened:
	present = sorted(n for n in root.keys())
	rec2 = root["rec-2"]["data"][:]
	written_region_nonzero = bool(np.any(rec2[:, :100]))
	unwritten_region = rec2[:, 100:]
	# zarr fills an unwritten region with the array's fill value (0 by default)
	# -- it reads back cleanly, no error, just silently wrong.
	looks_fine_but_is_wrong = bool(np.all(unwritten_region == 0) and not np.allclose(unwritten_region, full[:, 100:]))

	return {
		"items_visible_to_a_reader": present,
		"rec2_written_region_has_real_data": written_region_nonzero,
		"rec2_unwritten_region_silently_reads_as_zero_not_an_error": looks_fine_but_is_wrong,
		"a_reader_has_no_signal_that_rec2_or_rec3_are_incomplete": True,
	}


def icechunk_crash(dest: Path) -> dict:
	"""Same interrupted pattern, through neurozarr's own Writer, but inside an
	Icechunk session that never calls save()."""
	if dest.exists():
		shutil.rmtree(dest)
	repo = Repo.init(dest)
	repo.create_subject("sub-01")
	visit = repo.subject("sub-01").add_visit("ses-1")
	for i in range(2):
		visit.add_recording(_synthetic_raw(i), existing=ExistingPolicy.REPLACE, task=f"rec{i}")
	repo.commit("first two recordings, committed")

	# Now start a second session: write rec-2 (mid-write equivalent -- the
	# writer always writes the whole array per add_recording(), so "mid-write"
	# here means "written in this session" rather than a partial array; the
	# point under test is the save() boundary, not array-level chunking) and
	# deliberately never call save() -- this IS the crash.
	visit.add_recording(_synthetic_raw(2), existing=ExistingPolicy.REPLACE, task="rec2")
	# no repo.commit() -- simulated crash, this session's changes are abandoned

	# A FRESH, independent Repo instance -- a reader with no knowledge of the
	# crashed process, exactly like a reader coming along after a real crash.
	fresh = Repo.open(dest)
	visible = [r.path for r in fresh.subject("sub-01").visit("ses-1").recordings()]

	return {
		"items_visible_to_a_fresh_reader": visible,
		"rec2_visible_at_all": any("rec2" in p for p in visible),
		"store_matches_last_commit_exactly": len(visible) == 2,
	}


def main() -> None:
	out_path = Path(".bench_scratch/paper_logs/atomicity_demo.json")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	scratch = Path(".bench_scratch/paper_demos")

	result = {
		"plain_zarr": plain_zarr_crash(scratch / "atomicity_plain_zarr"),
		"icechunk": icechunk_crash(scratch / "atomicity_icechunk"),
	}
	out_path.write_text(json.dumps(result, indent=2))
	print(json.dumps(result, indent=2))
	print(f"\nwritten to {out_path}")


if __name__ == "__main__":
	main()
