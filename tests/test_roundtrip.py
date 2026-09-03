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
