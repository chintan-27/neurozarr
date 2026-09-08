"""Layout and durability: one repo per subject, and reopening never destroys data."""

from neurozarr import Repo


def test_reopening_does_not_wipe_existing_data(store, raw):
	"""Regression: Writer used to open zarr with mode="w" ("overwrite if exists"),
	so reopening a store to add data silently erased everything already in it."""
	first = Repo(store)
	first.create_subject("sub-001", attrs={"age": 63}).add_visit("ses-1").add_recording(raw, task="First")
	first.save("first")

	second = Repo(store)  # reopen the SAME subject and add another visit
	second.create_subject("sub-001").add_visit("ses-2").add_recording(raw, task="Second")
	second.save("second")

	subject = Repo(store).subject("sub-001")
	assert subject.visits() == ["ses-1", "ses-2"], "reopening wiped an existing session"
	assert subject.attrs["age"] == 63, "reopening wiped existing attrs"
	assert {r.entities["task"] for r in subject.recordings()} == {"First", "Second"}


def test_reconverting_the_same_source_is_idempotent(store, raw):
	"""Re-running a conversion into an existing store must replace, not crash --
	it used to fail with a bare "An array exists in store"."""
	for message in ("first pass", "second pass"):
		repo = Repo(store)
		repo.create_subject("sub-001").add_visit("ses-1").add_recording(
			raw, task="X", run=1, existing="replace")
		repo.save(message)

	subject = Repo(store).subject("sub-001")
	assert len(subject.recordings()) == 1, "re-ingest duplicated the recording"
	assert [m for _, m, _ in Repo(store).history("sub-001")][:2] == ["second pass", "first pass"]


def test_each_subject_is_an_independent_repo(store, raw):
	repo = Repo(store)
	for sub_id in ("sub-001", "sub-002"):
		repo.create_subject(sub_id).add_visit("ses-1").add_recording(raw, task="X")
	repo.save("two subjects")

	# each subject's tree starts at its own sessions -- no redundant sub-XXX level
	root = Repo(store).subject("sub-002").root()
	assert list(root.keys()) == ["ses-1"]


def test_dataset_level_attrs_go_to_their_own_repo(store):
	from neurozarr import Attrs

	repo = Repo(store)
	repo._route_attrs(Attrs((), {"Name": "My Study"}))
	repo._route_attrs(Attrs(("sub-001",), {"age": 40}))
	repo.save("attrs")

	assert repo._icechunk_repo("_dataset") is not None
	assert Repo(store).subject("sub-001").attrs == {"age": 40}


def test_set_attrs_merges_at_every_level_after_creation(store):
	repo = Repo(store)
	subject = repo.create_subject("sub-001", attrs={"age": 40})
	subject.add_visit("ses-1", attrs={"device": "A"})
	repo.save("first")

	repo = Repo(store)
	repo.set_attrs({"Name": "My Study"})
	repo.subject("sub-001").set_attrs({"handedness": "right"})
	repo.subject("sub-001").visit("ses-1").set_attrs({"technician": "B"})
	repo.save("attrs added later")

	reopened = Repo(store)
	assert reopened.subject("sub-001").attrs == {"age": 40, "handedness": "right"}
	assert reopened.subject("sub-001").visit("ses-1").attrs == {"device": "A", "technician": "B"}
	assert reopened._dataset_root().attrs.asdict()["Name"] == "My Study"


def test_history_and_tags(written):
	repo = Repo(written)
	history = repo.history("sub-001")
	assert any(message == "test data" for _, message, _ in history)

	repo.tag("v1")
	assert "v1" in repo.tags()
	assert "v1" not in repo.tags("sub-001")
