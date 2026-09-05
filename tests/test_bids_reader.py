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
	repo.ingest(BidsReader(tiny_bids))
	repo.save("tiny")

	subject = Repo(store).subject("sub-001")
	assert subject.attrs["age"] == 63          # participants.tsv row landed on the subject
	assert subject.visits() == ["ses-1"]


def test_sidecar_is_inherited_from_a_parent_directory(tiny_bids, store):
	(tiny_bids / "events.json").write_text(json.dumps({"duration": {"Units": "s"}}))
	repo = Repo(store)
	repo.ingest(BidsReader(tiny_bids))
	repo.save("tiny")

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
	repo.ingest(BidsReader(root))
	repo.save("no sessions")

	root_group = Repo(store).subject("sub-001").root()
	assert "beh" in root_group          # datatype sits directly under the subject
	assert Repo(store).subject("sub-001").visits() == []


def test_validate_flags_a_non_bids_directory(tmp_path):
	(tmp_path / "empty").mkdir()
	problems = validate_bids(tmp_path / "empty")
	assert any("no sub-*" in p for p in problems)


def test_validate_accepts_the_tiny_dataset(tiny_bids):
	assert validate_bids(tiny_bids) == []
