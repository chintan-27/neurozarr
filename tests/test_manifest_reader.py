import json

import pandas as pd
import pytest

from neurozarr import Repo, inspect_source
from neurozarr.items import Attrs, ExternalFile, Table
from neurozarr.readers import ManifestReader, reader_for


@pytest.fixture
def tsv_file(tmp_path):
	path = tmp_path / "log.tsv"
	pd.DataFrame({"onset": [0, 1], "label": ["a", "b"]}).to_csv(path, sep="\t", index=False)
	return path


def test_reads_a_tsv_row_with_default_columns(tsv_file):
	df = pd.DataFrame([{"sub": "sub-001", "ses": "ses-1", "datatype": "beh",
						"task": "Log", "path": str(tsv_file)}])
	items = list(ManifestReader(df).read())
	assert len(items) == 1 and isinstance(items[0], Table)
	assert items[0].entities.sub == "sub-001"
	assert items[0].entities.extra == {"task": "Log"}   # extra columns become entities
	assert list(items[0].df.columns) == ["onset", "label"]


def test_column_names_are_configurable(tsv_file):
	df = pd.DataFrame([{"subject": "sub-002", "filepath": str(tsv_file)}])
	items = list(ManifestReader(df, sub_col="subject", path_col="filepath").read())
	assert items[0].entities.sub == "sub-002"


def test_unsupported_extension_becomes_an_explicit_reference(tmp_path):
	odd = tmp_path / "scan.xyz"
	odd.write_text("a format nothing reads")
	items = list(ManifestReader(pd.DataFrame([{"sub": "sub-003", "path": str(odd)}])).read())
	assert isinstance(items[0], ExternalFile)
	assert items[0].uri == odd


def test_meta_column_accepts_json_string(tsv_file):
	df = pd.DataFrame([{"sub": "sub-004", "path": str(tsv_file), "meta": '{"device": "X"}'}])
	assert list(ManifestReader(df).read())[0].meta["device"] == "X"


def test_meta_column_requires_a_json_object(tsv_file):
	reader = ManifestReader(pd.DataFrame([
		{"sub": "sub-004", "path": str(tsv_file), "meta": '["not", "an", "object"]'},
	]))
	report = inspect_source(reader)
	assert report.errors[0].code == "manifest.invalid_metadata"
	with pytest.raises(ValueError, match="JSON must decode to an object"):
		list(reader.read())


def test_a_pathless_row_yields_dataset_level_attrs():
	df = pd.DataFrame([{"sub": None, "path": None, "meta": '{"Name": "chb-mit"}'}])
	items = list(ManifestReader(df).read())
	assert items == [Attrs((), {"Name": "chb-mit"})]


def test_a_pathless_row_yields_subject_level_attrs():
	df = pd.DataFrame([{"sub": "sub-001", "path": None, "meta": {"age": 42}}])
	items = list(ManifestReader(df).read())
	assert items == [Attrs(("sub-001",), {"age": 42})]


def test_a_pathless_row_with_datatype_and_name_builds_a_longer_path(tsv_file):
	df = pd.DataFrame([
		{"sub": "sub-001", "ses": "ses-1", "datatype": "ieeg", "name": "coordsystem",
		 "path": None, "meta": {"space": "Other"}},
		{"sub": "sub-001", "ses": "ses-1", "datatype": "ieeg", "path": str(tsv_file)},
	])
	items = list(ManifestReader(df).read())
	assert items[0] == Attrs(("sub-001", "ses-1", "ieeg", "coordsystem"), {"space": "Other"})
	assert isinstance(items[1], Table)  # the file row beside it still reads normally


def test_pathless_row_passes_preflight_validation():
	df = pd.DataFrame([{"sub": None, "path": None, "meta": '{"Name": "x"}'}])
	report = inspect_source(ManifestReader(df))
	assert not report.errors


def test_reads_a_json_manifest_with_nested_meta(tmp_path, tsv_file):
	manifest = tmp_path / "manifest.json"
	manifest.write_text(json.dumps([
		{"sub": "sub-001", "path": str(tsv_file), "meta": {"device": "X"}},
	]))
	items = list(ManifestReader(manifest).read())
	assert items[0].meta["device"] == "X"  # a real nested object, no JSON-string escaping needed


def test_reader_for_dispatches_json_manifests_to_manifest_reader(tmp_path, tsv_file):
	manifest = tmp_path / "manifest.json"
	manifest.write_text(json.dumps([{"sub": "sub-001", "path": str(tsv_file)}]))
	assert isinstance(reader_for(manifest), ManifestReader)


def test_zero_padded_entities_survive_a_csv_manifest(tmp_path, tsv_file):
	"""A column of "01", "02" must not be read back as the integers 1, 2 --
	that filed run-01 under run-1, which no longer matched its BIDS source."""
	manifest = tmp_path / "manifest.csv"
	pd.DataFrame([
		{"sub": "sub-001", "run": "01", "path": str(tsv_file)},
		{"sub": "sub-001", "run": "02", "path": str(tsv_file)},
		{"sub": "sub-001", "run": "11", "path": str(tsv_file)},
	]).to_csv(manifest, index=False)
	items = list(ManifestReader(manifest).read())
	assert [i.entities.extra["run"] for i in items] == ["01", "02", "11"]
	assert [i.entities.group_name() for i in items] == ["run-01", "run-02", "run-11"]


def test_reads_a_tsv_manifest(tmp_path, tsv_file):
	"""The CLI advertises .tsv manifests; read as comma-separated they came back as one column."""
	manifest = tmp_path / "manifest.tsv"
	pd.DataFrame([{"sub": "sub-001", "run": "01", "path": str(tsv_file)}]).to_csv(manifest, index=False, sep="\t")
	assert list(ManifestReader(manifest).read())[0].entities.extra == {"run": "01"}


def test_blank_cells_in_a_csv_manifest_are_still_missing(tmp_path, tsv_file):
	manifest = tmp_path / "manifest.csv"
	pd.DataFrame([
		{"sub": "sub-001", "run": "01", "path": str(tsv_file)},
		{"sub": "sub-001", "run": None, "path": str(tsv_file)},
	]).to_csv(manifest, index=False)
	items = list(ManifestReader(manifest).read())
	assert items[0].entities.extra == {"run": "01"}
	assert items[1].entities.extra == {}


def test_zero_padded_entities_survive_a_json_manifest(tmp_path, tsv_file):
	manifest = tmp_path / "manifest.json"
	manifest.write_text(json.dumps([{"sub": "sub-001", "run": "01", "path": str(tsv_file)}]))
	assert list(ManifestReader(manifest).read())[0].entities.extra == {"run": "01"}


def test_row_reader_bypasses_all_interpretation():
	sentinel = Attrs(("sub-005",), {"custom": True})
	reader = ManifestReader(pd.DataFrame([{"anything": "goes"}]), row_reader=lambda row: sentinel)
	assert list(reader.read()) == [sentinel]


def test_whole_number_entities_do_not_become_floats(tsv_file):
	"""A blank in a numeric column makes pandas widen it to float -- run must
	still land as run-1, not run-1.0."""
	df = pd.DataFrame([
		{"sub": "sub-001", "run": 1, "path": str(tsv_file)},
		{"sub": "sub-001", "run": None, "path": str(tsv_file)},
	])
	items = list(ManifestReader(df).read())
	assert items[0].entities.extra == {"run": 1}
	assert items[0].entities.group_name() == "run-1"


def test_missing_required_column_fails_upfront():
	with pytest.raises(ValueError, match="missing required column"):
		ManifestReader(pd.DataFrame([{"subject": "sub-001", "filepath": "x.edf"}]))


def test_manifest_ingests_into_a_store(store, tsv_file):
	df = pd.DataFrame([{"sub": "sub-001", "ses": "ses-1", "datatype": "beh",
						"task": "Log", "path": str(tsv_file)}])
	repo = Repo(store)
	repo.add(ManifestReader(df))
	repo.commit("from manifest")

	tables = Repo(store).subject("sub-001").visit("ses-1").tables()
	assert len(tables) == 1
	assert tables[0].df()["label"].tolist() == ["a", "b"]
