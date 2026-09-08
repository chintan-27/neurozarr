"""rename()/delete() on Subject, Visit, and every item view.

Neither zarr nor icechunk has a move/rename primitive, so a rename is a
generic recursive copy-then-delete (Writer._copy_node); these tests exercise
that path for a Group (session, table, external file) and an Array item."""

import numpy as np
import pandas as pd
import pytest

from neurozarr import Entities, ExternalFile, Repo


def _populated_repo(store, raw):
	repo = Repo(store)
	visit = repo.create_subject("sub-001", attrs={"age": 40}).add_visit("ses-1", attrs={"device": "A"})
	visit.add_recording(raw, task="Rest")
	visit.add_behavioral_table(pd.DataFrame({"x": [1, 2]}), task="Impedance")
	visit.add_array("arr", np.arange(4.0).reshape(2, 2), ("x", "y"), datatype="ieeg", task="Rest2")
	repo._writer_for("sub-001").add_external_file(
		ExternalFile(Entities("sub-001", "ses-1", "ieeg", {"task": "Rest3"}), "sidecar", "somewhere.mystery"))
	repo.save("initial")
	return repo


def test_rename_visit_keeps_everything_under_it(store, raw):
	_populated_repo(store, raw)

	repo = Repo(store)
	subject = repo.subject("sub-001")
	subject.rename_visit("ses-1", "ses-2")
	repo.save("renamed session")

	subject = Repo(store).subject("sub-001")
	assert subject.visits() == ["ses-2"]
	visit = subject.visit("ses-2")
	assert visit.attrs == {"device": "A"}
	assert visit.recordings()[0].entities["task"] == "Rest"
	assert visit.tables()[0].df()["x"].tolist() == [1, 2]
	assert np.array_equal(visit.arrays()[0].data(), np.arange(4.0).reshape(2, 2))
	assert visit.external_files()[0].uri == "somewhere.mystery"


def test_rename_visit_rejects_an_existing_target(store, raw):
	repo = _populated_repo(store, raw)
	repo.subject("sub-001").add_visit("ses-2")
	repo.save("second visit")

	repo = Repo(store)
	with pytest.raises(FileExistsError):
		repo.subject("sub-001").rename_visit("ses-1", "ses-2")


def test_delete_visit_removes_it(store, raw):
	_populated_repo(store, raw)

	repo = Repo(store)
	repo.subject("sub-001").delete_visit("ses-1")
	repo.save("deleted session")

	assert Repo(store).subject("sub-001").visits() == []


def test_rename_and_delete_on_every_item_view(store, raw):
	_populated_repo(store, raw)

	repo = Repo(store)
	visit = repo.subject("sub-001").visit("ses-1")
	visit.recordings()[0].rename("renamed-recording")
	visit.tables()[0].rename("renamed-table")
	visit.arrays()[0].rename("renamed-array")
	visit.external_files()[0].rename("renamed-file")
	repo.save("renamed items")

	visit = Repo(store).subject("sub-001").visit("ses-1")
	assert visit.recordings()[0].path.endswith("renamed-recording")
	assert visit.tables()[0].name == "renamed-table"
	assert visit.arrays()[0].name == "renamed-array"
	assert visit.external_files()[0].name == "renamed-file"
	assert np.array_equal(visit.arrays()[0].data(), np.arange(4.0).reshape(2, 2))
	assert visit.tables()[0].df()["x"].tolist() == [1, 2]

	repo = Repo(store)
	visit = repo.subject("sub-001").visit("ses-1")
	visit.recordings()[0].delete()
	visit.tables()[0].delete()
	visit.arrays()[0].delete()
	visit.external_files()[0].delete()
	repo.save("deleted items")

	visit = Repo(store).subject("sub-001").visit("ses-1")
	assert visit.recordings() == []
	assert visit.tables() == []
	assert visit.arrays() == []
	assert visit.external_files() == []


def test_view_rename_and_delete_need_a_writable_repo(store, raw):
	"""Regression: a view built without a write= (the default, and what any
	direct views_in() caller outside repo.py still gets) must fail clearly."""
	repo = Repo(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="Rest")
	repo.save("initial")

	from neurozarr.read import RecordingView
	bare = RecordingView(repo.root_of("sub-001")["ses-1"]["ieeg"]["task-Rest"], {}, "sub-001/ses-1/ieeg/task-Rest")
	with pytest.raises(TypeError, match="writable"):
		bare.rename("x")
	with pytest.raises(TypeError, match="writable"):
		bare.delete()


def test_delete_subject_removes_it_from_the_dataset_index(store, raw):
	"""delete_subject is a soft removal from schema v2's published index, not a
	physical erase -- the subject's own repository and history are untouched."""
	repo = _populated_repo(store, raw)
	repo.delete_subject("sub-001")
	repo.save("removed subject")

	reopened = Repo(store)
	assert reopened.subjects() == []
	assert list(reopened.find(task="Rest")) == []
	with pytest.raises(KeyError, match="sub-001"):
		reopened.root_of("sub-001")  # no longer part of this dataset version -- by design

	# but the subject's own repository is untouched, not physically erased
	assert reopened._repo_exists("sub-001")
	import zarr
	root = zarr.open_group(
		store=reopened._icechunk_repo("sub-001").readonly_session("main").store, mode="r")
	assert root.attrs.asdict()["age"] == 40


def test_delete_subject_rejects_an_unknown_subject(store):
	repo = Repo(store)
	repo.create_subject("sub-001")
	repo.save("initial")

	with pytest.raises(KeyError):
		Repo(store).delete_subject("sub-002")


def test_save_publishes_a_subject_deletion_with_no_other_writes(store, raw):
	"""Regression: save() used to bail out early whenever self._writers was
	empty, which is exactly the state right after delete_subject (it discards
	any writer for the removed subject rather than creating one) -- so a
	delete-only save silently published nothing."""
	repo = _populated_repo(store, raw)

	repo = Repo(store)
	repo.delete_subject("sub-001")
	snapshot = repo.save("delete only")

	assert snapshot is not None
	assert Repo(store).subjects() == []
