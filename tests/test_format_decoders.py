"""Format decoder registry: per-file decoding shared by ManifestReader and
BidsReader, distinct from the source-reader registry in test_robustness.py."""

import json

import numpy as np
import pandas as pd
import pytest

from neurozarr import Entities, ExternalFile, Repo
from neurozarr.readers import BidsReader, ManifestReader, register_format, registered_formats
from neurozarr.readers.formats import _DECODERS, decoder_for


@pytest.fixture(autouse=True)
def _restore_registered_formats():
	"""register_format is process-global; a test that registers a fake decoder
	must not leak it into every test that runs after it."""
	saved = dict(_DECODERS)
	yield
	_DECODERS.clear()
	_DECODERS.update(saved)


def test_register_format_requires_leading_dot():
	with pytest.raises(ValueError, match="leading dot"):
		register_format("nwb", lambda path, entities, name, meta: iter(()))


def test_decoder_for_matches_a_compound_extension(tmp_path):
	"""Regression: path.suffix only sees the last dot-segment, so a decoder
	registered for '.nii.gz' never matched a *.nii.gz file -- the same class of
	bug fixed in the reader registry (test_reader_for_matches_a_compound_extension),
	here in the per-file decoder registry instead."""
	calls = []

	def fake_decoder(path, entities, name, meta):
		calls.append(path)
		return iter(())

	register_format(".fake.gz", fake_decoder)
	target = tmp_path / "sub-001_thing.fake.gz"
	target.write_bytes(b"")
	assert decoder_for(target) is fake_decoder

	plain = tmp_path / "plain.gz"
	plain.write_bytes(b"")
	assert decoder_for(plain) is None


def test_manifest_reader_embeds_binary_when_asked(tmp_path):
	target = tmp_path / "data.unknownformat"
	target.write_bytes(b"raw bytes nobody has a decoder for")
	manifest = pd.DataFrame([{"sub": "sub-001", "ses": "ses-1", "datatype": "beh",
							  "task": "Blob", "path": str(target)}])
	manifest_path = tmp_path / "manifest.csv"
	manifest.to_csv(manifest_path, index=False)

	repo = Repo.create("memory://embed-test")
	repo.ingest(ManifestReader(manifest_path, unclaimed="embed"))
	repo.save("embedded")

	array = Repo.open("memory://embed-test").subject("sub-001").visit("ses-1").arrays()[0]
	assert bytes(array.data()) == b"raw bytes nobody has a decoder for"
	assert array.meta["_neurozarr_decoder"] == "binary"
	assert Repo.open("memory://embed-test").subject("sub-001").visit("ses-1").external_files() == []


def test_manifest_reader_default_unclaimed_is_still_a_reference(tmp_path):
	"""Regression: unclaimed defaults to reference, unchanged from before this
	feature existed -- embedding is opt-in, never silent."""
	target = tmp_path / "data.unknownformat"
	target.write_bytes(b"x")
	manifest = pd.DataFrame([{"sub": "sub-001", "ses": "ses-1", "datatype": "beh",
							  "task": "Blob", "path": str(target)}])
	manifest_path = tmp_path / "manifest.csv"
	manifest.to_csv(manifest_path, index=False)

	repo = Repo.create("memory://reference-test")
	repo.ingest(ManifestReader(manifest_path))
	repo.save("referenced")

	visit = Repo.open("memory://reference-test").subject("sub-001").visit("ses-1")
	assert visit.arrays() == []
	assert len(visit.external_files()) == 1


def test_views_in_surfaces_a_sibling_item_next_to_a_recording(store, raw):
	"""Regression: a group that is a recording only ever yielded the
	RecordingView, so any sibling table/array/external-file sharing the same
	entities as the recording (e.g. an unsupported-format sidecar file next to
	a BIDS recording under the same task) was silently invisible outside
	RecordingView's own channels()/events() facade."""
	repo = Repo.create(store)
	visit = repo.create_subject("sub-001").add_visit("ses-1")
	visit.add_recording(raw, task="Rest")
	sidecar = ExternalFile(Entities("sub-001", "ses-1", "ieeg", {"task": "Rest"}), "sidecar", "somewhere.mystery")
	repo._writer_for("sub-001").add_external_file(sidecar)
	repo.save("recording plus a sibling reference")

	visit = Repo.open(store).subject("sub-001").visit("ses-1")
	assert len(visit.recordings()) == 1, "the recording itself must still be found"
	assert [f.name for f in visit.external_files()] == ["sidecar"], \
		"a sibling item under the same entities as a recording must still be found"


def test_views_in_does_not_double_list_events_as_a_top_level_table(store):
	"""events/channels stay facade-only (rec.events()), not independently
	listed -- the fix for the bug above must not also break this."""
	import mne
	info = mne.create_info(["a"], sfreq=100.0, ch_types="eeg")
	raw = mne.io.RawArray(np.zeros((1, 100)), info, verbose=False)
	raw.set_annotations(mne.Annotations(onset=[0.1], duration=[0.0], description=["stim"]))

	repo = Repo.create(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="X")
	repo.save("with events")

	visit = Repo.open(store).subject("sub-001").visit("ses-1")
	assert visit.recordings()[0].events()["trial_type"].tolist() == ["stim"]
	assert visit.tables() == []


def test_bids_reader_compound_extension_suffix_is_not_mangled(tmp_path, store):
	"""Regression: Path.stem only strips '.gz' from 'sub-001_T1w.nii.gz', so the
	BIDS suffix was recovered as 'T1w.nii' instead of 'T1w'. Tested against the
	embed fallback rather than the real NIfTI decoder, so it holds regardless of
	which decoder ends up handling the file."""
	_DECODERS.pop(".nii.gz", None)  # isolate the naming fix from decoder priority
	anat = tmp_path / "bids" / "sub-001" / "ses-1" / "anat"
	anat.mkdir(parents=True)
	(tmp_path / "bids" / "dataset_description.json").write_text(json.dumps({"Name": "x"}))
	(anat / "sub-001_ses-1_T1w.nii.gz").write_bytes(b"not a real nifti, just checking the filename parsing")

	repo = Repo.create(store)
	repo.ingest(BidsReader(tmp_path / "bids", unclaimed="embed"))
	repo.save("imaging")

	array = Repo.open(store).subject("sub-001").visit("ses-1").arrays()[0]
	assert array.name == "T1w"


def test_nifti_decoder_reads_real_image_data(tmp_path, store):
	nib = pytest.importorskip("nibabel")

	image = nib.Nifti1Image(np.arange(24, dtype=np.float32).reshape(2, 3, 4), affine=np.eye(4))
	anat = tmp_path / "bids" / "sub-001" / "ses-1" / "anat"
	anat.mkdir(parents=True)
	(tmp_path / "bids" / "dataset_description.json").write_text(json.dumps({"Name": "x"}))
	nib.save(image, str(anat / "sub-001_ses-1_T1w.nii.gz"))

	assert ".nii.gz" in registered_formats(), "NIfTI decoder should auto-register when nibabel is installed"

	repo = Repo.create(store)
	repo.ingest(BidsReader(tmp_path / "bids"))
	repo.save("imaging")

	array = Repo.open(store).subject("sub-001").visit("ses-1").arrays()[0]
	assert array.name == "T1w"
	assert array.dims == ("x", "y", "z")
	assert np.array_equal(array.data(), np.arange(24, dtype=np.float32).reshape(2, 3, 4))
	assert array.meta["_neurozarr_decoder"] == "nibabel:nifti"
	assert array.meta["affine"] == np.eye(4).tolist()


def test_inspect_source_does_not_warn_on_a_file_a_decoder_claims(tmp_path):
	"""Regression: inspect_source() checked only the built-in tabular/mne
	extension sets, with no knowledge of the format decoder registry, so it
	warned 'no reader for .gz' on a NIfTI file even when nibabel was installed
	and the real conversion would decode it successfully -- a false warning
	from the exact check meant to be trusted before converting."""
	pytest.importorskip("nibabel")
	from neurozarr import inspect_source

	anat = tmp_path / "bids" / "sub-001" / "ses-1" / "anat"
	anat.mkdir(parents=True)
	(tmp_path / "bids" / "dataset_description.json").write_text(json.dumps({"Name": "x"}))
	(anat / "sub-001_ses-1_T1w.nii.gz").write_bytes(b"")  # decoder claims by extension, content unread here

	report = inspect_source(tmp_path / "bids")
	assert "source.unsupported_format" not in {issue.code for issue in report}
