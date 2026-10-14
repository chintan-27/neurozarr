import json

import numpy as np
import pytest

from neurozarr import Repo, cli
from neurozarr.errors import WriteConflictError
from neurozarr.readers import BidsReader


def run(*argv: str) -> int:
	return cli.main(list(argv))


def test_cli_add_status_commit_in_a_store_directory(tmp_path, tiny_bids, monkeypatch, capsys):
	store = tmp_path / "study"
	store.mkdir()
	monkeypatch.chdir(store)  # found like git finds .git
	assert run("init") == 0
	assert run("status") == 0 and "nothing to commit" in capsys.readouterr().out
	assert run("commit", "-m", "empty") == 1  # nothing staged

	assert run("add", str(tiny_bids), "-q") == 0
	assert run("status") == 0
	out = capsys.readouterr().out
	assert "On branch main" in out and "sub-02: 2 added" in out and "added:   ses-1/eeg/task-rest\n" in out

	assert run("commit", "-m", "first import") == 0
	assert "[main " in capsys.readouterr().out
	assert run("status") == 0 and "nothing to commit" in capsys.readouterr().out

	assert run("log", "--oneline") == 0
	log = capsys.readouterr().out.splitlines()
	assert log[0].endswith("first import") and "(main)" in log[0] and len(log) == 2  # + the init version
	assert run("show", "--json") == 0
	shown = json.loads(capsys.readouterr().out)
	assert shown["subjects_added"] == ["sub-01", "sub-02", "sub-03"] and shown["codec_plan"]["reader"] == "BidsReader"
	assert run("ls") == 0 and capsys.readouterr().out.split() == ["subject", "sub-01", "subject", "sub-02", "subject", "sub-03"]
	assert run("find", "task=rest", "sub=sub-02") == 0 and capsys.readouterr().out.strip() == "sub-02/ses-1/eeg/task-rest"


def test_cli_and_python_share_the_index(tmp_path, tiny_bids):
	store = str(tmp_path / "store")
	Repo.init(store)
	repo = Repo.open(store, mode="a")
	repo.add(BidsReader(tiny_bids).for_subject("sub-01"))  # staged in Python ...
	assert run("-C", store, "add", str(tiny_bids), "--existing", "skip", "-q") == 0  # ... more from the CLI
	staged = Repo.open(store, mode="a").status()["staged"]
	assert {"sub-01", "sub-02", "sub-03"} <= set(staged)
	assert run("-C", store, "commit", "-m", "both") == 0  # committed from the CLI
	assert Repo.open(store).subjects() == ["sub-01", "sub-02", "sub-03"]


def test_reset_unstages(tmp_path, tiny_bids):
	store = str(tmp_path / "store")
	Repo.init(store).add(tiny_bids)
	repo = Repo.open(store, mode="a")
	repo.reset("sub-02")
	assert set(repo.status()["staged"]) >= {"sub-01", "sub-03"} and "sub-02" not in repo.status()["staged"]
	repo = Repo.open(store, mode="a")  # unstaging persisted
	assert "sub-02" not in repo.status()["staged"]
	repo.reset()
	assert Repo.open(store, mode="a").status()["staged"] == {}
	assert Repo.open(store, mode="a").commit("nothing") is None


def test_rm_and_mv(tmp_path, tiny_bids):
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	repo.add(tiny_bids)
	repo.commit("import")
	with pytest.raises(ValueError, match="recursive"):
		repo.rm("sub-03")
	repo.rm("sub-03", recursive=True)
	repo.mv("sub-01/ses-1/eeg/task-rest", "task-baseline")
	status = Repo.open(store, mode="a").status()  # staged across processes
	assert status["removed_subjects"] == ["sub-03"]
	repo.commit("tidy")
	repo = Repo.open(store)
	assert repo.subjects() == ["sub-01", "sub-02"]
	assert [e["path"] for e in repo.ls("sub-01/ses-1")] == ["sub-01/ses-1/eeg/task-baseline"]
	with pytest.raises(ValueError, match="can't be renamed"):
		Repo.open(store, mode="a").mv("sub-01", "sub-09")


def test_versions_resolve_like_git(tmp_path, tiny_bids):
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	init_version = repo.resolve()
	repo.add(tiny_bids)
	first = repo.commit("import")
	repo.tag("v1")
	repo.rm("sub-03", recursive=True)
	second = repo.commit("drop sub-03")
	assert repo.resolve() == second and repo.resolve("main") == second
	assert repo.resolve("v1") == first and repo.resolve(first[:8]) == first
	assert repo.resolve("main~1") == first and repo.resolve("main~2") == init_version
	assert Repo.open(store).ls(version="v1")[-1]["path"] == "sub-03"  # read the past
	with pytest.raises(ValueError):
		repo.resolve("main~9")


def test_remotes(tmp_path, tiny_bids, capsys):
	store = str(tmp_path / "lab-store")
	assert run("remote", "add", "lab", store) == 0
	assert run("remote") == 0 and capsys.readouterr().out.strip() == f"lab\t{store}"
	assert run("init", "lab") == 0
	assert run("-C", "lab", "add", str(tiny_bids), "-q") == 0
	assert run("-C", "lab", "commit", "-m", "via remote") == 0
	assert Repo.open("lab").subjects() == ["sub-01", "sub-02", "sub-03"]


def test_outside_a_store_says_so(tmp_path, monkeypatch, capsys):
	monkeypatch.chdir(tmp_path)
	assert run("status") == 1
	assert "not a neurozarr store" in capsys.readouterr().err


def _as_user(monkeypatch, tmp_path, name):
	"""Each user has their own index and HEAD, as each git clone has its own."""
	monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / name))


def test_concurrent_commits_conflict_only_when_they_touch_the_same_data(tmp_path, tiny_bids, raw, monkeypatch):
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	repo.add(tiny_bids)
	repo.commit("import")

	_as_user(monkeypatch, tmp_path, "alice")
	alice = Repo.open(store, mode="a")
	alice.subject("sub-01").visit("ses-1").add_recording(raw, task="alice")  # staged, not committed

	_as_user(monkeypatch, tmp_path, "bob")
	bob = Repo.open(store, mode="a")
	bob.subject("sub-02").visit("ses-1").add_recording(raw, task="bob")
	bob.commit("bob's change")  # a different subject

	_as_user(monkeypatch, tmp_path, "alice")
	Repo.open(store, mode="a").commit("alice's change")  # no overlap: both land
	found = {r.path for r in Repo.open(store).find()}
	assert {"sub-01/ses-1/ieeg/task-alice", "sub-02/ses-1/ieeg/task-bob"} <= found

	# now both change the same recording, differently (an identical rewrite changes
	# nothing, so it is no conflict -- as in git)
	alice = Repo.open(store, mode="a")
	alice.subject("sub-01").visit("ses-1").add_recording(_scaled(raw, 2.0), task="alice", existing="replace")
	_as_user(monkeypatch, tmp_path, "bob")
	bob = Repo.open(store, mode="a")
	bob.subject("sub-01").visit("ses-1").add_recording(_scaled(raw, 3.0), task="alice", existing="replace")
	bob.commit("bob rewrites it")
	published = Repo.open(store).resolve()
	_as_user(monkeypatch, tmp_path, "alice")
	with pytest.raises(WriteConflictError, match="since this was staged"):
		Repo.open(store, mode="a").commit("alice rewrites it too")
	assert Repo.open(store).resolve() == published  # nothing half-published
	values = Repo.open(store).subject("sub-01").visit("ses-1").recording(task="alice").data()[0]
	assert np.abs(values - 3.0 * raw.get_data()).max() < 1e-9  # bob's, the one published


def test_commit_leaves_no_fork_snapshot_behind(tmp_path, tiny_bids):
	import icechunk
	store = tmp_path / "s"
	repo = Repo.init(str(store))
	repo.add(tiny_bids)
	first = repo.commit("import")
	sub = repo.subjects()[0]
	path = repo.query(kind="recording", sub=sub).paths[0]
	before = repo.get(path).data()[0]
	sub_repo, ses = repo.subject(sub), path.split("/")[1]
	sub_repo.visit(ses).add_recording(_scaled(repo.get(path).raw(), 2.0), existing="replace", **repo.get(path).entities)
	repo.commit("edit")
	for repo_id in [*repo.subjects(), "_dataset"]:
		ice = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store / repo_id)))
		reachable = {s.id for b in ice.list_branches() for s in ice.ancestry(branch=b)}
		on_disk = {p.name for p in (store / repo_id / "snapshots").iterdir()}
		assert on_disk <= reachable, f"{repo_id}: unreferenced snapshots {on_disk - reachable}"
	assert np.allclose(Repo.open(str(store)).get(path, version=first).data()[0], before)  # history intact
	from neurozarr.validate import inspect_store
	assert not inspect_store(str(store)).errors


def _scaled(raw, factor):
	copy = raw.copy().load_data()
	copy._data *= factor
	return copy


def test_a_correction_writes_only_the_chunks_it_changes(tmp_path):
	import mne
	store = tmp_path / "s"
	repo = Repo.init(str(store))
	data = np.random.default_rng(0).standard_normal((2, 5 * 65536)) * 1e-5  # 5 chunks along time
	raw = mne.io.RawArray(data, mne.create_info(2, 1000.0, "eeg"), verbose=False)
	repo.subject("sub-01").visit("ses-1").add_recording(raw, task="long")
	first = repo.commit("import")
	chunks = lambda: set((store / "sub-01" / "chunks").iterdir())
	before = chunks()
	fixed = raw.copy().load_data()
	fixed._data[0, 3 * 65536 + 7] += 1e-3  # one sample, in the fourth chunk
	repo.subject("sub-01").visit("ses-1").add_recording(fixed, task="long", existing="replace")
	repo.commit("correct one sample")
	assert len(chunks() - before) == 1
	rec = Repo.open(str(store)).subject("sub-01").visit("ses-1").recording(task="long")
	assert np.allclose(rec.data()[0], fixed.get_data(), atol=1e-12)
	assert np.allclose(Repo.open(str(store)).get(rec.path, version=first).data()[0], raw.get_data(), atol=1e-12)

	shorter = mne.io.RawArray(data[:, :1000], mne.create_info(2, 1000.0, "eeg"), verbose=False)
	repo.subject("sub-01").visit("ses-1").add_recording(shorter, task="long", existing="replace")
	repo.commit("a different layout is rewritten whole")
	assert Repo.open(str(store)).subject("sub-01").visit("ses-1").recording(task="long").shape == (2, 1000)
