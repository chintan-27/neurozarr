"""Incremental conversion, derivatives, verify and export."""

import json

import mne
import numpy as np
import pandas as pd
import pytest

from neurozarr import Repo
from neurozarr.readers import BidsReader
from neurozarr.verify import export_bids, verify


@pytest.fixture
def bids_source(tmp_path):
	root = tmp_path / "bids"
	ieeg = root / "sub-001" / "ses-1" / "ieeg"
	ieeg.mkdir(parents=True)
	(root / "dataset_description.json").write_text(json.dumps({"Name": "Tiny", "BIDSVersion": "1.10.0"}))
	pd.DataFrame({"onset": [0.5], "duration": [1.0]}).to_csv(
		ieeg / "sub-001_ses-1_task-rest_run-1_events.tsv", sep="\t", index=False)
	return root


def test_skip_existing_leaves_stored_data_alone(store, raw):
	repo = Repo(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="X")
	repo.save("first")
	before, _ = Repo(store).subject("sub-001").visit("ses-1").recording(task="X").data()

	# a second pass whose "source" has different values for the same entities
	other = mne.io.RawArray(np.ones((2, 500)) * 1e-6,
							mne.create_info(["LFP_L", "LFP_R"], 250.0, "eeg"), verbose=False)

	class OneRecording:
		def read(self):
			from neurozarr.entities import Entities
			from neurozarr.items import Recording
			yield Recording(Entities("sub-001", "ses-1", "ieeg", {"task": "X"}), other, {})

	repo = Repo(store)
	repo.ingest(OneRecording(), skip_existing=True)
	after, _ = Repo(store).subject("sub-001").visit("ses-1").recording(task="X").data()
	assert np.array_equal(before, after), "skip_existing overwrote existing data"


def test_without_skip_existing_the_data_is_replaced(store, raw):
	repo = Repo(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="X")
	repo.save("first")

	louder = mne.io.RawArray(raw.get_data() * 2, raw.info.copy(), verbose=False)
	repo = Repo(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(
		louder, task="X", existing="replace")
	repo.save("second")

	after, _ = Repo(store).subject("sub-001").visit("ses-1").recording(task="X").data()
	assert np.abs(after).max() > np.abs(raw.get_data()).max()


def test_derivatives_are_namespaced(store, raw):
	repo = Repo(store)
	visit = repo.create_subject("sub-001").add_visit("ses-1")
	visit.add_recording(raw, task="Stream", run=1)
	visit.add_derivative("my-filter", raw, task="Stream", run=1)
	repo.save("with a derivative")

	paths = sorted(r.path for r in Repo(store).subject("sub-001").recordings())
	assert paths == [
		"sub-001/derivatives/my-filter/ses-1/ieeg/task-Stream_run-1",
		"sub-001/ses-1/ieeg/task-Stream_run-1",
	]
	derivative = [r for r in Repo(store).subject("sub-001").recordings() if "derivatives" in r.path][0]
	assert derivative.meta["GeneratedBy"] == "my-filter"


def test_verify_passes_for_a_faithful_conversion(bids_source, store):
	repo = Repo(store)
	repo.ingest(BidsReader(bids_source))
	repo.save("converted")
	assert verify(bids_source, store) == []


def test_verify_reports_missing_data(bids_source, store):
	repo = Repo.create(store)
	repo.create_subject("sub-001").add_visit("ses-1")  # nothing actually ingested
	repo.save("empty")
	problems = verify(bids_source, store)
	assert any("missing from store" in p for p in problems)


def test_export_writes_a_bids_folder(store, raw, tmp_path):
	repo = Repo(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="Stream", run=1)
	repo.save("for export")

	dest = export_bids(store, tmp_path / "exported")
	assert (dest / "dataset_description.json").exists()

	out_dir = dest / "sub-001/ses-1/ieeg"
	written = sorted(p.name for p in out_dir.iterdir())
	assert "sub-001_ses-1_task-Stream_run-1.json" in written

	# whichever signal format was available (BrainVision/EDF/FIF), mne must read it back
	signal = [p for p in out_dir.iterdir() if p.suffix in (".vhdr", ".edf", ".fif")]
	assert signal, f"no signal file exported, only {written}"
	back = mne.io.read_raw(signal[0], verbose=False)
	assert back.ch_names == raw.ch_names
	assert back.info["sfreq"] == raw.info["sfreq"]
