"""Dataset manifests, global versions, and typed generalized items."""

import icechunk
import numpy as np
import pandas as pd

from neurozarr import Attrs, Entities, Repo, inspect_store
from neurozarr.items import Recording
from neurozarr.schema import MANIFEST_ATTR, SCHEMA_VERSION
from neurozarr.storage import storage_from
from neurozarr.writer import Writer


def test_save_publishes_global_manifest_and_returns_version(store, raw):
	repo = Repo.init(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="X")
	version = repo.commit("first")
	assert version

	opened = Repo.open(store)
	manifest = opened.root_of("_dataset").attrs.asdict()[MANIFEST_ATTR]
	assert manifest["schema_version"] == SCHEMA_VERSION
	assert set(manifest["subject_snapshots"]) == {"sub-001"}
	assert opened.subjects() == ["sub-001"]


def test_global_tag_pins_each_subject_snapshot(store, raw):
	repo = Repo.init(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="First")
	repo.commit("first")
	repo.tag("v1")

	repo = Repo.open(store, mode="a")
	repo.subject("sub-001").add_visit("ses-2").add_recording(raw, task="Second")
	repo.commit("second")
	assert Repo.open(store).subject("sub-001").visits(version="v1") == ["ses-1"]


def test_transaction_discards_changes_on_error(store, raw):
	repo = Repo.init(store)
	try:
		with repo.transaction("must not commit"):
			repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="X")
			raise RuntimeError("stop")
	except RuntimeError:
		pass
	assert Repo.open(store).subjects() == []


def test_typed_nullable_table_and_nd_array_round_trip(store):
	repo = Repo.init(store)
	visit = repo.create_subject("sub-001").add_visit("ses-1")
	table = pd.DataFrame({
		"count": pd.Series([1, None], dtype="Int64"),
		"ok": pd.Series([True, None], dtype="boolean"),
		"kind": pd.Series(["a", None], dtype="string"),
		"score": pd.Series([1.5, None], dtype="Float32"),
		"category": pd.Series(pd.Categorical(["a", None], categories=["a", "b"], ordered=True)),
		"when": pd.Series(pd.to_datetime(["2024-01-01T12:00:00Z", None]).tz_convert("America/New_York")),
		"elapsed": pd.Series(pd.to_timedelta([1, None], unit="s")),
	})
	visit.add_table(table, task="Log")
	visit.add_array("power", np.arange(6).reshape(2, 3), ("channel", "frequency"),
		datatype="eeg", coords={"frequency": [1, 2, 3]}, task="Spectrum")
	repo.commit("generalized")

	visit = Repo.open(store).subject("sub-001").visit("ses-1")
	result = visit.tables()[0].df()
	assert [str(dtype) for dtype in result.dtypes] == [
		"Int64", "boolean", "string", "Float32", "category",
		"datetime64[ns, America/New_York]", "timedelta64[ns]",
	]
	assert result["category"].cat.ordered
	assert np.array_equal(visit.arrays()[0].data(), np.arange(6).reshape(2, 3))


def test_array_named_data_is_not_misclassified_as_a_recording(store):
	repo = Repo.init(store)
	visit = repo.create_subject("sub-001").add_visit("ses-1")
	visit.add_array("data", np.arange(3), ("sample",), datatype="anat", task="Vector")
	repo.commit("array")
	visit = Repo.open(store).subject("sub-001").visit("ses-1")
	assert len(visit.arrays()) == 1
	assert visit.recordings() == []


def test_explicit_creation_materializes_empty_subject_and_visit(store):
	repo = Repo.init(store)
	repo.create_subject("sub-001").add_visit("ses-1")
	repo.commit("empty subject")
	assert Repo.open(store).subject("sub-001").visits() == ["ses-1"]


def test_doctor_accepts_a_current_store(store):
	Repo.init(store)
	assert inspect_store(store).ok


def test_disjoint_subject_writers_merge_at_publication(store, raw):
	Repo.init(store)
	left = Repo.open(store, mode="a")
	right = Repo.open(store, mode="a")
	left.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="L")
	right.create_subject("sub-002").add_visit("ses-1").add_recording(raw, task="R")
	left.commit("left")
	right.commit("right")
	assert Repo.open(store).subjects() == ["sub-001", "sub-002"]


def test_legacy_migration_preserves_subject_snapshot(store, raw):
	subject_repo = icechunk.Repository.create(storage_from(store, "sub-001"))
	writer = Writer(subject_repo.writable_session("main"))
	writer.add_recording(Recording(Entities("sub-001", "ses-1", "ieeg", {"task": "X"}), raw))
	subject_snapshot = writer.save("legacy subject")

	dataset_repo = icechunk.Repository.create(storage_from(store, "_dataset"))
	dataset_writer = Writer(dataset_repo.writable_session("main"))
	dataset_writer.add_attrs(Attrs((), {"subjects": ["sub-001"]}))
	dataset_writer.save("legacy dataset")

	repo = Repo(store)
	assert repo.migrate(dry_run=True)["subjects"] == ["sub-001"]
	repo.migrate()
	manifest = Repo.open(store).root_of("_dataset").attrs.asdict()[MANIFEST_ATTR]
	assert manifest["subject_snapshots"]["sub-001"] == subject_snapshot


def test_migrating_a_missing_store_does_not_create_one(tmp_path):
	missing = tmp_path / "missing"
	with np.testing.assert_raises(FileNotFoundError):
		Repo(missing).migrate()
	assert not missing.exists()


def _with_manifest(repo, mangle):
	"""Repo whose manifest is read back mangled, simulating catalog drift."""
	real = repo._manifest

	def patched(version=None):
		manifest = real(version)
		if manifest is not None:
			mangle(manifest)
		return manifest

	repo._manifest = patched
	return repo


def _two_subjects(store, raw):
	repo = Repo.init(store)
	for sub_id in ("sub-001", "sub-002"):
		repo.create_subject(sub_id).add_visit("ses-1").add_recording(raw, task="Rest", run=1)
	repo.commit("two subjects")
	return store


def test_index_records_the_snapshot_it_describes(store, raw):
	from neurozarr.index import Table
	_two_subjects(store, raw)
	repo = Repo.open(store)
	manifest = repo._manifest()
	for sub_id, snapshot in manifest.subject_snapshots.items():
		assert Table.decode(repo._blob(sub_id)).header["snapshot"] == snapshot, "index is not keyed by its snapshot"
	assert repo._summary().subjects == sorted(manifest.subject_snapshots)


def test_find_opens_nothing_until_data_is_read(store, raw):
	repo = Repo.init(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="Rest")
	repo.create_subject("sub-002").add_visit("ses-1").add_recording(raw, task="Other")
	repo.commit("two tasks")

	opened, scanned = Repo.open(store), []
	real_root_of = opened.root_of
	opened.root_of = lambda sub_id, version=None: (scanned.append(sub_id), real_root_of(sub_id, version))[1]
	found = list(opened.find(task="Rest"))
	assert [view.entities["task"] for view in found] == ["Rest"]
	assert scanned == [], "find() answers from the index; it opens no subject"
	found[0].data()
	assert scanned == ["sub-001"], "reading the match opens only its own subject"


def test_find_does_not_walk_when_the_catalog_is_current(store, raw, monkeypatch):
	"""Regression: find() used to answer from the catalog only to decide which
	subjects to re-walk -- it still fully re-walked every surviving one via
	Subject.recordings() (a members(max_depth=None) scan of the whole tree).
	Assert on whether that walk ran at all, not on wall-clock: a timing
	assertion here would be flaky by construction."""
	from neurozarr import Subject

	repo = Repo.init(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="Rest")
	repo.commit("one subject")

	walked: list[str] = []
	real_recordings = Subject.recordings
	monkeypatch.setattr(Subject, "recordings",
		lambda self, *a, **kw: (walked.append(self.sub_id), real_recordings(self, *a, **kw))[1])

	found = list(Repo.open(store).find(task="Rest"))
	assert [v.entities["task"] for v in found] == ["Rest"]
	assert walked == [], "find() walked a subject its current catalog could already answer"


def test_find_agrees_with_a_full_walk(store, raw):
	"""find() answers from the index; it must never hide a recording that is really there."""
	from neurozarr.read import RecordingView, views_in
	_two_subjects(store, raw)
	repo = Repo.open(store)
	walked = {v.path for sub in repo.subjects() for v in views_in(repo.root_of(sub), sub) if isinstance(v, RecordingView)}
	assert {v.path for v in repo.find(task="Rest")} == walked


def test_fsck_reports_an_index_that_does_not_match(store, raw, monkeypatch):
	from neurozarr.index import Table, encode
	_two_subjects(store, raw)
	assert inspect_store(store).ok

	real = Repo._blob

	def patched(self, name, version=None, fresh=False):
		blob = real(self, name, version, fresh)
		if name == "sub-002" and blob is not None:
			table = Table.decode(blob)
			return encode(table.rows(), {**table.header, "snapshot": "OUTDATEDSNAPSHOT0000"})
		return blob

	monkeypatch.setattr(Repo, "_blob", patched)
	report = inspect_store(store)
	assert [issue.location for issue in report if issue.code == "store.corrupt_index"] == ["sub-002"]
	assert not report.ok, "an index that disagrees with the data makes queries wrong: an error"

