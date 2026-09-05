"""Safe defaults and structured extension contracts."""

import numpy as np
import pandas as pd
import pytest

from neurozarr import (
	Array, CodecConfig, Entities, ExistingPolicy, ExternalFile, Repo,
	ValidationError, WriteConflictError, inspect_source,
)
from neurozarr.items import Table
from neurozarr.readers import ManifestReader, open_reader, register_reader
from neurozarr.util import chunk_shape, safe_uri


@pytest.mark.parametrize("sub", ["001", "sub-../x", "sub-a/b", "sub-"])
def test_unsafe_subject_ids_are_rejected(sub):
	with pytest.raises(ValidationError):
		Entities(sub)


@pytest.mark.parametrize("kwargs", [
	{"dtype": "float32"},
	{"bitround_k": 16},
	{"bitround_k": 1.5},
	{"dtype": "float16", "bitround_k": 1},
	{"chunk_target_bytes": 0},
	{"chunk_target_bytes": True},
	{"max_chunk_samples": -1},
])
def test_invalid_codec_config_is_rejected(kwargs):
	with pytest.raises(ValidationError):
		CodecConfig(**kwargs)


def test_manifest_report_has_structured_warning_for_unknown_format(tmp_path):
	path = tmp_path / "scan.dcm"
	path.write_bytes(b"DICOM")
	reader = ManifestReader(pd.DataFrame({"sub": ["sub-001"], "path": [str(path)]}))
	report = inspect_source(reader)
	assert report.ok
	assert report.warnings[0].code == "source.unsupported_format"
	assert isinstance(list(reader.read())[0], ExternalFile)


def test_manifest_relative_paths_resolve_from_manifest_directory(tmp_path):
	data = tmp_path / "events.tsv"
	pd.DataFrame({"onset": [1]}).to_csv(data, sep="\t", index=False)
	manifest = tmp_path / "manifest.csv"
	pd.DataFrame({"sub": ["sub-001"], "path": ["events.tsv"]}).to_csv(manifest, index=False)
	item = list(ManifestReader(manifest).read())[0]
	assert isinstance(item, Table)
	assert item.meta["_neurozarr_provenance"]["source"] == str(data)


def test_safe_uri_strips_credentials_and_tokens():
	assert safe_uri("s3://user:secret@bucket/path?token=x#frag") == "s3://bucket/path"


def test_metadata_rejects_nested_non_string_keys():
	with pytest.raises(ValidationError, match="non-string key"):
		ExternalFile(Entities("sub-001"), "source", "scan.bin", meta={"nested": {1: "bad"}})


def test_chunk_shape_never_emits_zero_sized_chunks():
	assert chunk_shape((0, 3), 8) == (1, 3)


def test_array_validates_dimensions():
	with pytest.raises(ValueError, match="axes"):
		Array(Entities("sub-001", "ses-1", "anat"), "image", np.zeros((2, 3)), ("x",))


def test_in_process_reader_registry(tmp_path):
	class EmptyReader:
		def __init__(self, source):
			self.source = source

		def read(self):
			return iter(())

	register_reader("empty-test", EmptyReader, extensions=(".empty",))
	assert isinstance(open_reader("empty-test", tmp_path / "x.empty"), EmptyReader)
	assert inspect_source(open_reader("empty-test", tmp_path / "x.empty")).ok


def test_reader_registry_requires_normalized_extensions():
	with pytest.raises(ValueError, match="leading dot"):
		register_reader("bad-extension-test", lambda source: source, extensions=("nwb",))


def test_duplicate_write_errors_unless_policy_is_explicit(store, raw):
	repo = Repo.create(store)
	visit = repo.create_subject("sub-001").add_visit("ses-1")
	visit.add_recording(raw, task="X")
	repo.save("first")

	repo = Repo.open(store, mode="a")
	with pytest.raises(WriteConflictError):
		repo.subject("sub-001").visit("ses-1").add_recording(raw, task="X")
	with pytest.raises(RuntimeError, match="failed transaction"):
		repo.save("must not commit partial work")
	repo.abort()

	repo = Repo.open(store, mode="a")
	repo.subject("sub-001").visit("ses-1").add_recording(
		raw, task="X", existing=ExistingPolicy.REPLACE)
	assert repo.save("replace") is not None
