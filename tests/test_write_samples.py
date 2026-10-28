import numpy as np
import pytest

from neurozarr import Repo


def _store(store, raw):
	repo = Repo.init(store)
	repo.create_subject("sub-1").add_visit("ses-1").add_recording(raw, task="A")
	repo.commit("first")
	return Repo.open(store, mode="a")


def _rec(repo):
	return repo.subject("sub-1").visit("ses-1").recording(task="A")


def test_a_one_sample_correction_reads_back_and_keeps_the_old_version(store, raw_calibrated):
	repo = _store(store, raw_calibrated)
	first = repo.resolve()
	rec = _rec(repo)
	before, _ = rec.data()
	fixed = before[:, 10:11].copy()
	fixed[0, 0] += rec.meta["data_scale"][0]  # one int16 step
	rec.write(fixed, start=10)
	repo.commit("fix sample 10")

	after, _ = _rec(Repo.open(store)).data()
	expected = before.copy()
	expected[:, 10:11] = fixed
	assert np.array_equal(after, expected)
	old, _ = Repo.open(store).get(rec.path, version=first).data()
	assert np.array_equal(old, before)


def test_only_the_touched_chunk_is_rewritten(store, raw_calibrated):
	from neurozarr import CodecConfig
	repo = Repo.init(store, codec=CodecConfig(max_chunk_samples=100))  # 500 samples: 5 chunks
	repo.create_subject("sub-1").add_visit("ses-1").add_recording(raw_calibrated, task="A")
	repo.commit("first")
	repo = Repo.open(store, mode="a")
	rec = _rec(repo)
	values, _ = rec.data(start=250, stop=251)
	rec.write(values + rec.meta["data_scale"][0], start=250)
	changed = repo._writers["sub-1"].session.status().updated_chunks
	assert {tuple(c) for chunks in changed.values() for c in chunks} == {(0, 2)}


def test_picks_write_only_those_channels(store, raw):
	repo = _store(store, raw)  # float32: no int16 grid
	rec = _rec(repo)
	before, _ = rec.data()
	rec.write(np.full((1, 3), 7.0), start=0, picks="LFP_R")
	repo.commit("fix LFP_R")
	after, _ = _rec(Repo.open(store)).data()
	assert np.array_equal(after[0], before[0]) and np.array_equal(after[1, :3], [7.0, 7.0, 7.0])
	assert np.array_equal(after[1, 3:], before[1, 3:])


@pytest.mark.parametrize("values, start, match", [
	(np.zeros((1, 1)), 0, "shape"),  # two channels stored
	(np.zeros((2, 2)), 499, "outside"),  # 500 samples
	(np.full((2, 1), 0.5e-9), 0, "whole steps"),  # half an LSB
	(np.full((2, 1), 1.0), 0, "range"),  # 1e9 steps
])
def test_bad_writes_are_refused_and_change_nothing(store, raw_calibrated, values, start, match):
	repo = _store(store, raw_calibrated)
	with pytest.raises(ValueError, match=match):
		_rec(repo).write(values, start=start)
	assert not repo.status()["staged"]  # nothing staged, and the Repo can still commit
	assert not repo._failed


def test_a_read_only_view_cannot_write(store, raw):
	_store(store, raw)
	with pytest.raises(PermissionError):  # as set_attrs() there
		_rec(Repo.open(store)).write(np.zeros((2, 1)))
