import numpy as np
import pytest

from bidszarr import Repo


def test_recording_data_roundtrips(written, raw):
	rec = Repo(written).subject("sub-001").visit("ses-1").recording(task="Stream", run=1)
	values, _ = rec.data()
	assert values.shape == raw.get_data().shape
	# int16 packing is lossy at the LSB, but only there
	assert np.abs(values - raw.get_data()).max() < 1e-9


def test_recording_rebuilds_mne_raw(written, raw):
	rec = Repo(written).subject("sub-001").visit("ses-1").recording(task="Stream", run=1)
	back = rec.raw()
	assert back.info["sfreq"] == raw.info["sfreq"]
	assert back.ch_names == raw.ch_names


def test_raw_without_sfreq_raises_clearly(store):
	"""Data stored without any sampling rate (e.g. by an older version) should say
	so plainly rather than guessing, and still be readable with an explicit sfreq."""
	from bidszarr.read import RecordingView

	repo = Repo(store)
	group = repo._writer_for("sub-001").root.require_group("ses-1/ieeg/task-X")
	group.create_array("data", data=np.zeros((2, 10)))
	repo.save("no sfreq")

	rec = RecordingView(Repo(store).subject("sub-001").root()["ses-1/ieeg/task-X"], {}, "task-X")
	with pytest.raises(ValueError, match="sampling frequency"):
		rec.raw()
	assert rec.raw(sfreq=250.0).info["sfreq"] == 250.0  # explicit override still works


def test_windowed_reads_match_the_full_read(written, raw):
	rec = Repo(written).subject("sub-001").visit("ses-1").recording(task="Stream", run=1)
	full, _ = rec.data()

	by_sample, _ = rec.data(start=100, stop=200)
	assert np.array_equal(by_sample, full[:, 100:200])

	by_time, _ = rec.data(tmin=0.4, tmax=0.8)      # 250 Hz -> samples 100:200
	assert np.array_equal(by_time, full[:, 100:200])

	one_channel, _ = rec.data(picks=["LFP_R"])
	assert np.array_equal(one_channel[0], full[1])


def test_chunking_allows_partial_reads(store, raw):
	"""A recording must not land in a single chunk, or a windowed read still
	fetches the whole array."""
	import mne

	long_raw = mne.io.RawArray(
		np.random.default_rng(1).normal(scale=1e-5, size=(2, 200_000)),
		mne.create_info(["a", "b"], sfreq=250.0, ch_types="eeg"), verbose=False)
	repo = Repo(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(long_raw, task="Long")
	repo.save("long")

	rec = Repo(store).subject("sub-001").visit("ses-1").recording(task="Long")
	assert rec.array.chunks[-1] < rec.shape[-1], "whole recording is one chunk"
	assert rec.duration == 800.0
	assert rec.sfreq == 250.0


def test_raw_accepts_a_window(written):
	rec = Repo(written).subject("sub-001").visit("ses-1").recording(task="Stream", run=1)
	windowed = rec.raw(tmin=0.0, tmax=1.0)
	assert windowed.get_data().shape == (2, 250)
	assert rec.raw(picks=["LFP_L"]).ch_names == ["LFP_L"]


def test_table_dtypes_are_restored(written, table):
	tables = Repo(written).subject("sub-001").visit("ses-1").tables()
	got = tables[0].df()
	assert list(got.columns) == list(table.columns)
	assert got["amp"].dtype == table["amp"].dtype
	assert got["onset"].tolist() == table["onset"].tolist()


def test_navigation(written):
	repo = Repo(written)
	assert repo.subjects() == ["sub-001"]
	subject = repo.subject("sub-001")
	assert subject.attrs["age"] == 63
	assert subject.visits() == ["ses-1"]
	assert subject.visit("ses-1").attrs["device"] == "Percept"
	assert len(subject.recordings()) == 1


def test_find_across_store(written):
	found = list(Repo(written).find(task="Stream"))
	assert len(found) == 1
	assert found[0].entities["run"] == "1"


def test_missing_recording_raises(written):
	visit = Repo(written).subject("sub-001").visit("ses-1")
	with pytest.raises(KeyError, match="no recording"):
		visit.recording(task="Nope")
