import json
import logging

import numpy as np
import pytest

from neurozarr import Repo
from neurozarr.readers import BidsReader, ManifestReader, SmartReader

# spawned workers import mne and icechunk from scratch, a few seconds each
pytestmark = pytest.mark.timeout(180)

SUBJECTS = ("sub-01", "sub-02", "sub-03")  # conftest.tiny_bids


def _contents(target):
	repo = Repo.open(target)
	return {rec.path: rec.data()[0] for rec in repo.find()}, repo


@pytest.mark.parametrize("make_reader", [
	lambda root: BidsReader(root),
	lambda root: SmartReader(root),
], ids=["bids", "smart"])
def test_parallel_ingest_matches_serial(tmp_path, tiny_bids, make_reader):
	root = tiny_bids
	stores = {}
	for workers in (1, 2):
		target = tmp_path / f"store{workers}"
		repo = Repo.init(target)
		repo.add(make_reader(root), workers=workers)
		repo.commit("convert")
		stores[workers] = _contents(target)

	(serial, _), (parallel, repo) = stores[1], stores[2]
	assert sorted(parallel) == sorted(serial) and len(parallel) == 3
	for path in serial:
		assert np.array_equal(parallel[path], serial[path])
	# workers apply the parent's codec plan, not one of their own
	codecs = [{r.path: r.array.compressors for r in Repo.open(tmp_path / f"store{w}").find()} for w in (1, 2)]
	assert codecs[0] == codecs[1]
	assert repo.subjects() == list(SUBJECTS)
	# one commit per subject, with save()'s message -- workers commit nothing themselves
	assert [message for _, message, _ in repo.log("sub-02")][:1] == ["convert"]


def test_manifest_reader_in_parallel(tmp_path, tiny_bids):
	root = tiny_bids
	rows = [{"sub": sub, "ses": "ses-1", "datatype": "eeg", "task": "rest",
			 "path": str(root / sub / "ses-1" / "eeg" / f"{sub}_ses-1_task-rest_eeg.fif")} for sub in SUBJECTS]
	rows.append({"sub": None, "path": None, "meta": json.dumps({"Name": "from manifest"})})  # dataset-level row
	(tmp_path / "m.json").write_text(json.dumps(rows))
	reader = ManifestReader(tmp_path / "m.json")
	assert reader.subjects() == list(SUBJECTS)

	repo = Repo.init(tmp_path / "store")
	repo.add(reader, workers=3)
	repo.commit("convert")
	found, repo = _contents(tmp_path / "store")
	assert len(found) == 3 and repo.subjects() == list(SUBJECTS)


def test_in_memory_store_falls_back_to_serial(tiny_bids, store, caplog):
	root = tiny_bids
	repo = Repo.init(store)
	with caplog.at_level(logging.WARNING, logger="neurozarr"):
		repo.add(BidsReader(root), workers=4)
	repo.commit("convert")
	assert "ingesting serially" in caplog.text
	assert len(list(Repo.open(store).find())) == 3


def test_a_failing_subject_publishes_nothing(tmp_path, tiny_bids):
	root = tiny_bids
	(root / "sub-02/ses-1/eeg/sub-02_ses-1_task-rest_eeg.fif").write_bytes(b"not a fif file")
	repo = Repo.init(tmp_path / "store")
	with pytest.raises(Exception):
		# probe_files=1: planning samples sub-01 only, so the broken file fails in its worker
		repo.add(BidsReader(root, probe_files=1), workers=2)
	with pytest.raises(RuntimeError, match="failed transaction"):
		repo.commit("convert")
	assert Repo.open(tmp_path / "store").subjects() == []  # all or nothing


def test_cli_parallel_smart_reader(tmp_path, tiny_bids):
	from neurozarr import cli
	root = tiny_bids
	store = str(tmp_path / "store")
	assert cli.main(["init", store]) == 0
	assert cli.main(["-C", store, "add", str(root), "--reader", "smart", "-j", "2",
					 "--answers", str(tmp_path / "answers.json"), "-q"]) == 0
	assert cli.main(["-C", store, "commit", "-m", "parallel"]) == 0
	assert Repo.open(tmp_path / "store").subjects() == list(SUBJECTS)
