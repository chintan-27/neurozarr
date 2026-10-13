import numpy as np
import pytest
from zarr.codecs import GzipCodec, ZstdCodec

from neurozarr import CodecConfig, Repo


def test_recording_data_roundtrips(written, raw):
	rec = Repo(written).subject("sub-001").visit("ses-1").recording(task="Stream", run=1)
	values, _ = rec.data()
	assert values.shape == raw.get_data().shape
	# `raw` has no _raw_extras[0]["cal"], so this is the float32 fallback path,
	# not int16 packing -- see test_int16_calibration_path_roundtrips for that.
	assert np.abs(values - raw.get_data()).max() < 1e-9


def test_int16_calibration_path_roundtrips(store, raw_calibrated):
	"""The default codec path (int16 packing keyed off _raw_extras[0]["cal"])
	is what real EDF-family reads use, but no fixture exercised it until now --
	`raw` above has no such calibration and only ever hits the float32
	fallback. `raw_calibrated`'s values are exact multiples of its LSB, so a
	correct implementation round-trips them exactly, not just within a
	tolerance."""
	repo = Repo(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw_calibrated, task="Stream")
	repo.commit("calibrated")

	rec = Repo(store).subject("sub-001").visit("ses-1").recording(task="Stream")
	values, _ = rec.data()
	assert rec.array.dtype == np.dtype("int16")
	assert np.array_equal(values, raw_calibrated.get_data())


def _compressors(store):
	rec = Repo(store).subject("sub-001").visit("ses-1").recording(task="Stream")
	return rec, list(rec.array.compressors)


def test_explicit_api_uses_zarrs_default_codec_unless_told(store, raw, raw_calibrated):
	"""The explicit API never probes: CodecConfig.compressors if set, else zarr's default."""
	repo = Repo(store)
	visit = repo.create_subject("sub-001").add_visit("ses-1")
	visit.add_recording(raw, task="Stream")
	visit.add_recording(raw_calibrated, task="Calibrated")
	repo.commit("explicit")
	for task, source in (("Stream", raw), ("Calibrated", raw_calibrated)):
		rec = Repo(store).subject("sub-001").visit("ses-1").recording(task=task)
		assert list(rec.array.compressors) == [ZstdCodec(level=0)]  # zarr's "auto"
		assert np.abs(rec.data()[0] - source.get_data()).max() < 1e-9


def test_explicit_compressors_are_used_as_given(tmp_path, raw):
	repo = Repo.init(tmp_path / "s", codec=CodecConfig(compressors=[GzipCodec(level=4)]))
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="Stream")
	repo.commit("custom")
	_, compressors = _compressors(tmp_path / "s")
	assert compressors == [GzipCodec(level=4)]


def test_recording_rebuilds_mne_raw(written, raw):
	rec = Repo(written).subject("sub-001").visit("ses-1").recording(task="Stream", run=1)
	back = rec.raw()
	assert back.info["sfreq"] == raw.info["sfreq"]
	assert back.ch_names == raw.ch_names


def test_raw_without_sfreq_raises_clearly(store):
	"""Data stored without any sampling rate (e.g. by an older version) should say
	so plainly rather than guessing, and still be readable with an explicit sfreq."""
	from neurozarr.read import RecordingView

	repo = Repo(store)
	group = repo._writer_for("sub-001").root.require_group("ses-1/ieeg/task-X")
	group.create_array("data", data=np.zeros((2, 10)))
	repo.commit("no sfreq")

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
	repo.commit("long")

	rec = Repo(store).subject("sub-001").visit("ses-1").recording(task="Long")
	assert rec.array.chunks[-1] < rec.shape[-1], "whole recording is one chunk"
	assert rec.duration == 800.0
	assert rec.sfreq == 250.0


def test_raw_accepts_a_window(written):
	rec = Repo(written).subject("sub-001").visit("ses-1").recording(task="Stream", run=1)
	windowed = rec.raw(tmin=0.0, tmax=1.0)
	assert windowed.get_data().shape == (2, 250)
	assert rec.raw(picks=["LFP_L"]).ch_names == ["LFP_L"]


def test_annotations_survive_the_round_trip(store):
	"""mne annotations (events, stim markers) used to be dropped on write."""
	import mne

	raw = mne.io.RawArray(np.zeros((2, 1000)),
						  mne.create_info(["a", "b"], sfreq=100.0, ch_types="eeg"), verbose=False)
	raw.set_annotations(mne.Annotations(onset=[1.0, 3.5], duration=[0.5, 0.25],
										description=["stim_on", "artifact"]), verbose=False)
	repo = Repo(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="X")
	repo.commit("annotated")

	rec = Repo(store).subject("sub-001").visit("ses-1").recording(task="X")
	events = rec.events()
	assert events["trial_type"].tolist() == ["stim_on", "artifact"]
	assert events["onset"].tolist() == [1.0, 3.5]

	restored = rec.raw().annotations
	assert list(restored.description) == ["stim_on", "artifact"]
	assert np.allclose(restored.onset, [1.0, 3.5])
	assert np.allclose(restored.duration, [0.5, 0.25])


def test_recording_without_annotations_has_no_events_table(written):
	rec = Repo(written).subject("sub-001").visit("ses-1").recording(task="Stream", run=1)
	assert rec.events().empty
	assert rec.raw().annotations is not None  # just empty, not broken


def test_bids_events_sidecar_wins_over_annotations_written_first(store):
	"""A source format that carries its own annotations (EEGLAB's boundary markers,
	for instance) auto-derives an "events" table when a recording is added -- but a
	reader streams that recording and its real BIDS events.tsv sidecar as two
	independent items with no ordering guarantee between them (see BidsReader's
	per-directory grouping). If the recording is processed first, its annotations
	placeholder must not block the sidecar from landing right after -- the sidecar
	is always the real data and should win regardless of which came first."""
	import mne
	import pandas as pd

	from neurozarr.items import Entities, Table
	from neurozarr.writer import ExistingPolicy

	raw = mne.io.RawArray(np.zeros((2, 1000)),
						  mne.create_info(["a", "b"], sfreq=100.0, ch_types="eeg"), verbose=False)
	raw.set_annotations(mne.Annotations(onset=[0.5], duration=[0.0], description=["boundary"]), verbose=False)

	repo = Repo(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="X")

	sidecar = pd.DataFrame({"onset": [1.0, 2.0], "duration": [0.0, 0.0],
							 "trial_type": ["stim_on", "stim_off"]})
	entities = Entities("sub-001", "ses-1", "ieeg", {"task": "X"})
	repo._writer_for("sub-001").add_table(Table(entities, "events", sidecar, {}), ExistingPolicy.ERROR)
	repo.commit("recording then sidecar")

	events = Repo(store).subject("sub-001").visit("ses-1").recording(task="X").events()
	assert events["trial_type"].tolist() == ["stim_on", "stim_off"]


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


def test_narrow_reads_from_cached_chunks_match_direct_reads():
	import zarr
	from neurozarr.read import RecordingView
	group = zarr.open_group(zarr.storage.MemoryStore(), mode="a")
	stored = np.arange(3 * 1000, dtype=np.int16).reshape(3, 1000)
	group.create_array("data", shape=stored.shape, chunks=(3, 100), dtype=stored.dtype)[:] = stored
	group.attrs.update({"data_scale": [0.5, 2.0, 1.0], "data_offset": [0.0, 1.0, -3.0]})
	scale, offset = np.array([[0.5], [2.0], [1.0]]), np.array([[0.0], [1.0], [-3.0]])
	rec = RecordingView(group, {}, "sub-x/rec")
	windows = [(0, 10), (95, 105), (99, 101), (250, 450), (990, 1000), (0, 1000), (500, 500), (400, 410), (400, 410)]
	for (start, stop), picks in [(w, p) for w in windows for p in (None, [2, 0])]:  # 400-410 twice: from the cache
		values, _ = rec.data(start=start, stop=stop, picks=picks)
		rows = [0, 1, 2] if picks is None else picks
		assert np.array_equal(values, stored[rows, start:stop] * scale[rows] + offset[rows]), (start, stop, picks)
	from neurozarr.read import _PREFETCH
	assert len(rec._chunks) <= 3 + _PREFETCH

	unscaled = zarr.open_group(zarr.storage.MemoryStore(), mode="a")
	unscaled.create_array("data", shape=stored.shape, chunks=(3, 100), dtype=stored.dtype)[:] = stored
	rec = RecordingView(unscaled, {}, "sub-x/rec")
	first, _ = rec.data(start=0, stop=10)
	first[:] = -1  # writing to a result must not reach the cache
	assert np.array_equal(rec.data(start=0, stop=10)[0], stored[:, :10])
	scan = np.concatenate([rec.data(start=s, stop=min(1000, s + 30))[0] for s in range(0, 1000, 30)], axis=-1)
	assert np.array_equal(scan, stored)  # a forward scan, with the next chunk prefetched


def test_data_can_return_float32(written):
	import zarr
	from neurozarr.read import RecordingView
	rec = Repo(written).subject("sub-001").visit("ses-1").recording(task="Stream", run=1)
	stored, _ = rec.data()
	assert stored.dtype == rec.array.dtype  # unpacked recordings come back as stored by default
	assert rec.data(dtype=np.float64)[0].dtype == np.float64

	group = zarr.open_group(zarr.storage.MemoryStore(), mode="a")  # an int16-packed recording
	packed = np.arange(2 * 500, dtype=np.int16).reshape(2, 500)
	group.create_array("data", shape=packed.shape, chunks=(2, 100), dtype=packed.dtype)[:] = packed
	group.attrs.update({"data_scale": [1e-6, 2e-6], "data_offset": [0.0, 1e-3]})
	rec = RecordingView(group, {}, "sub-x/rec")
	full, half = rec.data()[0], rec.data(dtype=np.float32)[0]
	assert full.dtype == np.float64 and half.dtype == np.float32
	assert np.allclose(half, full, rtol=1e-6)
	window = rec.data(start=150, stop=250, dtype=np.float32)[0]
	assert window.dtype == np.float32 and np.array_equal(window, half[:, 150:250])
