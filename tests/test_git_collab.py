import datetime

import mne
import numpy as np
import pytest

from neurozarr import MergeConflictError, Repo, cli


@pytest.fixture
def imported(tmp_path, tiny_bids):
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	repo.add(tiny_bids)
	repo.commit("import")
	return store


def _raw(seed):
	data = np.random.default_rng(seed).standard_normal((2, 300)) * 1e-5
	return mne.io.RawArray(data, mne.create_info(2, 100.0, "eeg"), verbose=False)


def _rec(store, sub, task, version=None):
	for rec in Repo.open(store).subject(sub).recordings(version):
		if rec.entities.get("task") == task:
			return rec.data()[0]
	return None


def _add(repo, sub, task, seed, existing="error"):
	repo.subject(sub).visit("ses-1").add_recording(_raw(seed), task=task, existing=existing)


def test_fast_forward(imported):
	repo = Repo.open(imported, mode="a")
	repo.switch("feature", create=True)
	_add(repo, "sub-01", "new", 1)
	tip = repo.commit("on feature")
	repo.switch("main")
	assert repo.merge("feature") == tip  # main just moves
	assert repo.resolve() == tip and _rec(imported, "sub-01", "new") is not None
	assert repo.merge("feature") is None  # already up to date


def test_merge_combines_both_sides(imported):
	repo = Repo.open(imported, mode="a")
	repo.switch("feature", create=True)
	_add(repo, "sub-02", "theirs", 2)      # another subject
	_add(repo, "sub-01", "theirs", 3)      # same subject, different item
	repo.commit("feature work")
	repo.switch("main")
	_add(repo, "sub-03", "ours", 4)
	_add(repo, "sub-01", "ours", 5)
	repo.commit("main work")

	merged = repo.merge("feature")
	assert merged is not None and repo.show()["message"] == "Merge branch 'feature' into main"
	for sub, task, seed in [("sub-02", "theirs", 2), ("sub-01", "theirs", 3), ("sub-03", "ours", 4), ("sub-01", "ours", 5)]:
		assert np.abs(_rec(imported, sub, task) - _raw(seed).get_data()).max() < 1e-9, (sub, task)

	# more work on feature merges again: the earlier merge is part of the history now
	repo.switch("feature")
	_add(repo, "sub-02", "later", 6)
	repo.commit("later feature work")
	repo.switch("main")
	assert repo.merge("feature") is not None
	assert _rec(imported, "sub-02", "later") is not None
	assert repo.merge("feature") is None


def test_merge_conflict_and_resolution(imported, capsys):
	repo = Repo.open(imported, mode="a")
	repo.switch("feature", create=True)
	_add(repo, "sub-01", "dup", 7)  # both add the same new item, with different data
	repo.commit("their version")
	repo.switch("main")
	_add(repo, "sub-01", "dup", 8)
	before = repo.commit("our version")

	with pytest.raises(MergeConflictError) as raised:
		repo.merge("feature")
	assert list(raised.value.conflicts) == ["sub-01"] and repo.resolve() == before  # nothing written
	assert cli.main(["-C", imported, "merge", "feature"]) == 1 and "CONFLICT" in capsys.readouterr().err

	repo.merge("feature", theirs=("sub-01",))
	assert np.abs(_rec(imported, "sub-01", "dup") - _raw(7).get_data()).max() < 1e-9


def test_clone(imported, tmp_path):
	full = Repo.clone(imported, str(tmp_path / "copy"))
	assert full.log() == Repo.open(imported).log()
	shallow = Repo.clone(imported, str(tmp_path / "shallow"), version="main", shallow=True)
	assert len(shallow.log()) == 2  # init + the cloned version
	assert np.array_equal(_rec(str(tmp_path / "shallow"), "sub-02", "rest"), _rec(imported, "sub-02", "rest"))
	with pytest.raises(ValueError, match="local paths"):
		Repo.clone(imported, "s3://bucket/x")


def test_gc_keeps_everything_reachable(imported):
	repo = Repo.open(imported, mode="a")
	_add(repo, "sub-01", "tagged", 9)
	repo.commit("tagged work")
	repo.tag("keep")
	_add(repo, "sub-01", "dropped", 10)
	repo.commit("work to drop")
	repo.reset_hard("main~2")  # sub-01's branch now points before both

	now = datetime.timedelta(0)
	dry = repo.gc(now)
	assert sum(r["bytes"] for r in dry.values()) > 0  # the dropped version's data
	repo.gc(now, dry_run=False)
	assert np.abs(_rec(imported, "sub-01", "tagged", version="keep") - _raw(9).get_data()).max() < 1e-9
	assert _rec(imported, "sub-01", "rest") is not None  # current data intact
	assert sum(r["bytes"] for r in repo.gc(now).values()) == 0  # nothing left to collect
	assert "neurozarr-pin" not in " ".join(repo.tags("sub-01"))
