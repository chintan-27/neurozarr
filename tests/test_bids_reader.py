"""BidsReader against a synthetic BIDS tree -- no dependency on the real BIDS/ folder."""

import json

import pandas as pd
import pytest

from neurozarr import Repo
from neurozarr.readers import BidsReader
from neurozarr.validate import validate_bids


@pytest.fixture
def tiny_bids(tmp_path):
	root = tmp_path / "bids"
	ieeg = root / "sub-001" / "ses-1" / "ieeg"
	ieeg.mkdir(parents=True)
	(root / "dataset_description.json").write_text(json.dumps({"Name": "Tiny", "BIDSVersion": "1.10.0"}))
	pd.DataFrame({"participant_id": ["sub-001"], "age": [63]}).to_csv(root / "participants.tsv", sep="\t", index=False)

	pd.DataFrame({"onset": [0.5], "duration": [1.0]}).to_csv(
		ieeg / "sub-001_ses-1_task-rest_run-1_events.tsv", sep="\t", index=False)
	# a sidecar one level up: the inheritance principle should still find it
	(root / "sub-001" / "ses-1" / "task-rest_events.json").write_text(json.dumps({"onset": {"Units": "s"}}))
	return root


def test_reads_dataset_and_subject_levels(tiny_bids, store):
	repo = Repo(store)
	repo.add(BidsReader(tiny_bids))
	repo.commit("tiny")

	subject = Repo(store).subject("sub-001")
	assert subject.attrs["age"] == 63          # participants.tsv row landed on the subject
	assert subject.visits() == ["ses-1"]


def test_sidecar_is_inherited_from_a_parent_directory(tiny_bids, store):
	(tiny_bids / "events.json").write_text(json.dumps({"duration": {"Units": "s"}}))
	repo = Repo(store)
	repo.add(BidsReader(tiny_bids))
	repo.commit("tiny")

	tables = Repo(store).subject("sub-001").visit("ses-1").tables()
	events = [t for t in tables if t.name == "events"]
	assert events, "events table was not stored"
	assert events[0].meta["onset"] == {"Units": "s"}, "parent-directory sidecar was not inherited"
	assert events[0].meta["duration"] == {"Units": "s"}, "root and nearer sidecars were not merged"


def test_session_less_dataset_is_handled(tmp_path, store):
	root = tmp_path / "bids"
	(root / "sub-001" / "beh").mkdir(parents=True)
	(root / "dataset_description.json").write_text(json.dumps({"Name": "NoSessions"}))
	pd.DataFrame({"x": [1]}).to_csv(root / "sub-001" / "beh" / "sub-001_task-t_beh.tsv", sep="\t", index=False)

	repo = Repo(store)
	repo.add(BidsReader(root))
	repo.commit("no sessions")

	root_group = Repo(store).subject("sub-001").root()
	assert "beh" in root_group          # datatype sits directly under the subject
	assert Repo(store).subject("sub-001").visits() == []


def test_validate_flags_a_non_bids_directory(tmp_path):
	(tmp_path / "empty").mkdir()
	problems = validate_bids(tmp_path / "empty")
	assert any("no sub-*" in p for p in problems)


def test_validate_accepts_the_tiny_dataset(tiny_bids):
	assert validate_bids(tiny_bids) == []


def test_include_derivatives_flag(tiny_bids, store):
	"""derivatives/ is read unconditionally by default; include_derivatives=False
	must skip it entirely, converting only the dataset's raw data."""
	deriv_beh = tiny_bids / "derivatives" / "my-pipeline" / "sub-001" / "ses-1" / "beh"
	deriv_beh.mkdir(parents=True)
	pd.DataFrame({"score": [1.0]}).to_csv(
		deriv_beh / "sub-001_ses-1_task-summary_beh.tsv", sep="\t", index=False)

	with_derivatives = Repo(store)
	with_derivatives.add(BidsReader(tiny_bids))
	with_derivatives.commit("with derivatives")
	assert any("derivatives" in t.path for t in Repo(store).subject("sub-001").tables())

	without_derivatives = Repo(f"{store}-no-derivatives")
	without_derivatives.add(BidsReader(tiny_bids, include_derivatives=False))
	without_derivatives.commit("without derivatives")
	assert not any("derivatives" in t.path for t in without_derivatives.subject("sub-001").tables())


def test_nothing_in_the_tree_is_dropped_silently(tiny_bids, store):
	"""A folder no reader knows becomes a reference, scans.tsv a table, README text
	metadata, and files nothing reads (stimuli/) are named in 'files_not_stored'."""
	ses = tiny_bids / "sub-001" / "ses-1"
	unknown = ses / "ieeg" / "sub-001_ses-1_task-rest_run-1_ieeg.xyzd"
	(unknown / "inner").mkdir(parents=True)
	(unknown / "inner" / "part.bin").write_bytes(b"\0" * 10)
	pd.DataFrame({"filename": ["ieeg/x.xyzd"], "acq_time": ["2020-01-01T00:00:00"]}).to_csv(
		ses / "sub-001_ses-1_scans.tsv", sep="\t", index=False)
	(tiny_bids / "README").write_text("about")
	(tiny_bids / "stimuli").mkdir()
	(tiny_bids / "stimuli" / "face.png").write_bytes(b"png")

	repo = Repo.init(store)
	with pytest.warns(UserWarning, match="1 files .* are not stored"):
		repo.add(BidsReader(tiny_bids))
	repo.commit("tiny")

	reopened = Repo.open(store)
	visit = reopened.subject("sub-001").visit("ses-1")
	assert [f.name for f in visit.external_files()] == ["ieeg"]
	assert "scans" in [t.name for t in visit.tables()]
	dataset = reopened.root_of("_dataset")
	assert dataset["README"].attrs["text"] == "about"
	assert dataset["files_not_stored"].attrs.asdict() == {"stimuli/face.png": 3}
