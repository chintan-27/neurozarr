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
