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
	repo = Repo.create(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="X")
	version = repo.save("first")
	assert version

	opened = Repo.open(store)
	manifest = opened.root_of("_dataset").attrs.asdict()[MANIFEST_ATTR]
	assert manifest["schema_version"] == SCHEMA_VERSION
	assert set(manifest["subject_snapshots"]) == {"sub-001"}
	assert opened.subjects() == ["sub-001"]


def test_global_tag_pins_each_subject_snapshot(store, raw):
	repo = Repo.create(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="First")
	repo.save("first")
	repo.tag("v1")

	repo = Repo.open(store, mode="a")
	repo.subject("sub-001").add_visit("ses-2").add_recording(raw, task="Second")
	repo.save("second")
	assert Repo.open(store).subject("sub-001").visits(version="v1") == ["ses-1"]


def test_transaction_discards_changes_on_error(store, raw):
	repo = Repo.create(store)
	try:
		with repo.transaction("must not commit"):
			repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="X")
			raise RuntimeError("stop")
	except RuntimeError:
		pass
	assert Repo.open(store).subjects() == []


def test_typed_nullable_table_and_nd_array_round_trip(store):
	repo = Repo.create(store)
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
	visit.add_behavioral_table(table, task="Log")
	visit.add_array("power", np.arange(6).reshape(2, 3), ("channel", "frequency"),
		datatype="eeg", coords={"frequency": [1, 2, 3]}, task="Spectrum")
	repo.save("generalized")

	visit = Repo.open(store).subject("sub-001").visit("ses-1")
	result = visit.tables()[0].df()
	assert [str(dtype) for dtype in result.dtypes] == [
		"Int64", "boolean", "string", "Float32", "category",
		"datetime64[ns, America/New_York]", "timedelta64[ns]",
	]
	assert result["category"].cat.ordered
	assert np.array_equal(visit.arrays()[0].data(), np.arange(6).reshape(2, 3))


def test_array_named_data_is_not_misclassified_as_a_recording(store):
	repo = Repo.create(store)
	visit = repo.create_subject("sub-001").add_visit("ses-1")
	visit.add_array("data", np.arange(3), ("sample",), datatype="anat", task="Vector")
	repo.save("array")
	visit = Repo.open(store).subject("sub-001").visit("ses-1")
	assert len(visit.arrays()) == 1
	assert visit.recordings() == []


def test_explicit_creation_materializes_empty_subject_and_visit(store):
	repo = Repo.create(store)
	repo.create_subject("sub-001").add_visit("ses-1")
	repo.save("empty subject")
	assert Repo.open(store).subject("sub-001").visits() == ["ses-1"]


def test_doctor_accepts_a_current_store(store):
	Repo.create(store)
	assert inspect_store(store).ok


def test_disjoint_subject_writers_merge_at_publication(store, raw):
	Repo.create(store)
	left = Repo.open(store, mode="a")
	right = Repo.open(store, mode="a")
	left.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="L")
	right.create_subject("sub-002").add_visit("ses-1").add_recording(raw, task="R")
	left.save("left")
	right.save("right")
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
	repo = Repo.create(store)
	for sub_id in ("sub-001", "sub-002"):
		repo.create_subject(sub_id).add_visit("ses-1").add_recording(raw, task="Rest", run=1)
	repo.save("two subjects")
	return store


def test_catalog_entries_record_the_snapshot_they_describe(store, raw):
	_two_subjects(store, raw)
	manifest = Repo.open(store)._manifest()
	for sub_id, snapshot in manifest.subject_snapshots.items():
		assert manifest.catalog[sub_id]["snapshot"] == snapshot, "catalog entry is not keyed by its snapshot"


def test_a_valid_catalog_still_prunes_subjects_that_cannot_match(store, raw):
	repo = Repo.create(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="Rest")
	repo.create_subject("sub-002").add_visit("ses-1").add_recording(raw, task="Other")
	repo.save("two tasks")

	opened, scanned = Repo.open(store), []
	real_subject = opened.subject
	opened.subject = lambda sub_id: (scanned.append(sub_id), real_subject(sub_id))[1]
	assert [view.entities["task"] for view in opened.find(task="Rest")] == ["Rest"]
	assert scanned == ["sub-001"], "the catalog should have spared sub-002 from being opened"


def test_find_does_not_hide_subjects_whose_catalog_entry_is_unusable(store, raw):
	"""Regression: find() pruned on the catalog without checking it was current, so a
	missing, stale or unstamped entry silently dropped recordings that were really there."""
	_two_subjects(store, raw)
	both = {"sub-001", "sub-002"}

	drifts = {
		"missing": lambda m: m.catalog.pop("sub-002", None),
		"stale": lambda m: m.catalog["sub-002"].__setitem__("snapshot", "OUTDATEDSNAPSHOT0000"),
		"unstamped": lambda m: m.catalog["sub-002"].pop("snapshot", None),
	}
	for label, mangle in drifts.items():
		repo = _with_manifest(Repo.open(store), mangle)
		found = {view.path.split("/")[0] for view in repo.find(task="Rest")}
		assert found == both, f"{label} catalog entry hid a subject from find()"


def test_doctor_reports_a_stale_catalog_entry(store, raw, monkeypatch):
	_two_subjects(store, raw)
	assert inspect_store(store).ok

	real = Repo._manifest

	def patched(self, version=None):
		manifest = real(self, version)
		if manifest is not None and "sub-002" in manifest.catalog:
			manifest.catalog["sub-002"]["snapshot"] = "OUTDATEDSNAPSHOT0000"
		return manifest

	monkeypatch.setattr(Repo, "_manifest", patched)

	report = inspect_store(store)
	stale = [issue for issue in report if issue.code == "store.stale_catalog_entry"]
	assert [issue.location for issue in stale] == ["sub-002"]
	assert report.ok, "a stale catalog entry costs a scan; it is not a corrupt store"
