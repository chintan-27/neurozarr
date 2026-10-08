import threading
import time

import mne
import numpy as np
import pytest

from neurozarr import Repo
from neurozarr import index as nzindex


def _raw(seed):
	data = np.random.default_rng(seed).standard_normal((2, 300)) * 1e-5
	return mne.io.RawArray(data, mne.create_info(2, 100.0, "eeg"), verbose=False)


@pytest.fixture
def imported(tmp_path, tiny_bids):
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	repo.add(tiny_bids)
	repo.commit("import")
	return store


def test_refresh_fetches_only_what_changed(imported, monkeypatch):
	reader = Repo.open(imported)
	assert len(reader.query()) == 3
	writer = Repo.open(imported, mode="a")
	writer.subject("sub-02").visit("ses-1").add_recording(_raw(1), task="new")
	writer.commit("one more")

	assert len(reader.query()) == 3  # a Repo reads one version until it refreshes
	fetched = []
	real = Repo._blob
	monkeypatch.setattr(Repo, "_blob", lambda self, name, *a, **kw: (fetched.append(name), real(self, name, *a, **kw))[1])
	assert reader.refresh() is True
	assert len(reader.query()) == 4
	assert sorted(fetched) == ["_summary", "sub-02"], "unchanged subjects come from the cache"
	assert reader.refresh() is False


def test_watch_yields_new_versions(imported):
	reader = Repo.open(imported)

	def commit_later():
		time.sleep(0.3)
		writer = Repo.open(imported, mode="a")
		writer.subject("sub-01").visit("ses-1").add_recording(_raw(2), task="live")
		writer.commit("live")

	thread = threading.Thread(target=commit_later)
	thread.start()
	versions = []
	for version in reader.watch(interval=0.05, timeout=10):
		versions.append(version)
		break
	thread.join()
	assert versions and versions[0] == Repo.open(imported).resolve()
	assert "sub-01/ses-1/ieeg/task-live" in reader.query(task="live").paths


def test_query_can_include_staged_changes(imported):
	repo = Repo.open(imported, mode="a")
	repo.subject("sub-01").visit("ses-1").add_recording(_raw(3), task="staged")
	repo.rm("sub-03", recursive=True)
	assert repo.query(task="staged").paths == []
	staged = repo.query(include_staged=True)
	assert "sub-01/ses-1/ieeg/task-staged" in staged.paths
	assert not any(p.startswith("sub-03/") for p in staged.paths)
	view = repo.query("task == 'staged'", include_staged=True)[0]
	assert np.abs(view.data()[0] - _raw(3).get_data()).max() < 1e-9  # read from the staged tree
	repo.commit("now published")
	assert repo.query(task="staged").paths == ["sub-01/ses-1/ieeg/task-staged"]


def test_disk_cache_serves_indexes(imported, monkeypatch, tmp_path):
	monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
	repo = Repo.open(imported)
	repo.index_cache = True  # on by default only for remote stores
	monkeypatch.setattr(nzindex, "CACHE", nzindex._Cache())
	assert len(repo.query()) == 3
	assert len(list((tmp_path / "cache" / "neurozarr" / "index").iterdir())) == 3

	again = Repo.open(imported)
	again.index_cache = True
	monkeypatch.setattr(nzindex, "CACHE", nzindex._Cache())
	real = Repo._blob

	def no_subject_reads(self, name, *a, **kw):
		assert name == "_summary", f"{name} should have come from the disk cache"
		return real(self, name, *a, **kw)

	monkeypatch.setattr(Repo, "_blob", no_subject_reads)
	assert len(again.query()) == 3
