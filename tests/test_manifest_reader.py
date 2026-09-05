import pandas as pd
import pytest

from neurozarr import Repo, inspect_source
from neurozarr.items import Attrs, ExternalFile, Table
from neurozarr.readers import ManifestReader


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
	odd = tmp_path / "scan.dcm"
	odd.write_text("not a real dicom")
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
	repo.ingest(ManifestReader(df))
	repo.save("from manifest")

	tables = Repo(store).subject("sub-001").visit("ses-1").tables()
	assert len(tables) == 1
	assert tables[0].df()["label"].tolist() == ["a", "b"]
