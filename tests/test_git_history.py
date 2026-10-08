import numpy as np
import pytest

from neurozarr import Repo, cli
from neurozarr.errors import WriteConflictError


@pytest.fixture
def imported(tmp_path, tiny_bids):
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	repo.add(tiny_bids)
	repo.commit("import")
	return store


def _data(store, sub, task="rest", version=None, branch=None):
	repo = Repo.open(store, branch=branch)
	for rec in repo.subject(sub).recordings(version):
		if rec.entities.get("task") == task:
			return rec.data()[0]
	return None


def test_branches_isolate_work(imported, raw, capsys):
	repo = Repo.open(imported, mode="a")
	repo.switch("cleanup", create=True)
	assert repo.head == "cleanup" and repo.branches() == ["cleanup", "main"]
	repo.subject("sub-01").visit("ses-1").add_recording(raw, task="extra")
	repo.commit("extra recording on cleanup")

	assert _data(imported, "sub-01", "extra", branch="cleanup") is not None
	assert _data(imported, "sub-01", "extra", branch="main") is None  # main untouched
	assert Repo.open(imported).head == "cleanup"  # HEAD persisted, shared with the CLI
	assert cli.main(["-C", imported, "branch"]) == 0
	assert capsys.readouterr().out.split() == ["*", "cleanup", "main"]

	# staged changes stay with the branch they were staged on
	repo.subject("sub-02").visit("ses-1").add_recording(raw, task="pending")
	repo.switch("main")
	assert repo.status()["staged"] == {}
	repo.switch("cleanup")
	assert "sub-02" in repo.status()["staged"]


def test_diff(imported, raw, capsys):
	repo = Repo.open(imported, mode="a")
	before = repo.resolve()
	repo.subject("sub-01").visit("ses-1").add_recording(raw, task="extra")
	assert repo.diff()["subjects"]["sub-01"]["added"] == ["ses-1/ieeg/task-extra"]  # staged
	repo.rm("sub-03", recursive=True)
	repo.commit("edit")

	changes = repo.diff(before)["subjects"]
	assert changes["sub-01"]["status"] == "changed" and changes["sub-01"]["added"] == ["ses-1/ieeg/task-extra"]
	assert changes["sub-03"] == {"status": "removed"} and "sub-02" not in changes
	assert repo.diff(before, before)["subjects"] == {}
	first = repo.diff("main~2", "main~1")["subjects"]  # init -> import: every subject added
	assert {s["status"] for s in first.values()} == {"added"} and len(first) == 3
	assert cli.main(["-C", imported, "diff", "main~1", "--exit-code", "--stat"]) == 1
	assert "changed  sub-01: 1 added" in capsys.readouterr().out


def test_revert_restores_exactly(imported):
	repo = Repo.open(imported, mode="a")
	original = _data(imported, "sub-03")
	repo.rm("sub-03", recursive=True)
	removal = repo.commit("drop sub-03")
	assert "sub-03" not in Repo.open(imported).subjects()

	repo = Repo.open(imported, mode="a")
	repo.revert(removal)
	assert np.array_equal(_data(imported, "sub-03"), original)
	assert [m for _, m, _ in repo.log()][0].startswith('Revert "drop sub-03"')
	assert len(repo.log()) == 4  # init, import, drop, revert -- history kept


def test_revert_refuses_to_undo_later_changes(imported, raw):
	repo = Repo.open(imported, mode="a")
	repo.subject("sub-01").visit("ses-1").add_recording(raw, task="a")
	first = repo.commit("a")
	repo.subject("sub-01").visit("ses-1").add_recording(raw, task="b")
	repo.commit("b")
	with pytest.raises(WriteConflictError, match="changed again"):
		repo.revert(first)


def test_reset_hard(imported, raw):
	repo = Repo.open(imported, mode="a")
	imported_version = repo.resolve()
	repo.subject("sub-01").visit("ses-1").add_recording(raw, task="mistake")
	repo.commit("mistake")
	repo.tag("kept")
	repo.reset_hard("main~1")
	assert repo.resolve() == imported_version
	assert _data(imported, "sub-01", "mistake") is None
	assert _data(imported, "sub-01", "mistake", version="kept") is not None  # still reachable by tag

	# later writes build on the reset state, not on the dropped version
	repo = Repo.open(imported, mode="a")
	repo.subject("sub-01").visit("ses-1").add_recording(raw, task="fresh")
	repo.commit("fresh")
	tasks = {r.entities["task"] for r in Repo.open(imported).subject("sub-01").recordings()}
	assert tasks == {"rest", "fresh"}
	assert cli.main(["-C", imported, "reset", "--hard", "main~1"]) == 1  # needs --yes


def test_branch_deletion_is_safe(imported, raw):
	repo = Repo.open(imported, mode="a")
	repo.branch("merged")
	repo.delete_branch("merged")  # its versions are on main too
	repo.switch("work", create=True)
	repo.subject("sub-01").visit("ses-1").add_recording(raw, task="w")
	repo.commit("only on work")
	with pytest.raises(ValueError, match="current branch"):
		repo.delete_branch("work")
	repo.switch("main")
	with pytest.raises(ValueError, match="force"):
		repo.delete_branch("work")
	repo.delete_branch("work", force=True)
	assert repo.branches() == ["main"]
