import json
import logging

import numpy as np
import pytest

from neurozarr import Repo
from neurozarr.items import Attrs
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


def test_a_workers_last_small_table_is_not_lost(tmp_path, tiny_bids):
	# a subject's stream ends with a derivatives table small enough to be staged, not
	# written; a worker must flush it before handing its fork back
	beh = tiny_bids / "derivatives" / "pipe" / "sub-02" / "ses-1" / "beh"
	beh.mkdir(parents=True)
	(tiny_bids / "derivatives" / "pipe" / "dataset_description.json").write_text(json.dumps({"Name": "pipe"}))
	(beh / "sub-02_ses-1_task-x_beh.tsv").write_text("a\tb\n1\t2\n")
	trees = []
	for workers in (1, 2):
		repo = Repo.init(tmp_path / f"store{workers}")
		repo.add(BidsReader(tiny_bids), workers=workers)
		repo.commit("convert")
		root = Repo.open(tmp_path / f"store{workers}").subject("sub-02").root()
		trees.append(sorted(path for path, _ in root.members(max_depth=None)))
	assert trees[0] == trees[1] and any(p.startswith("derivatives/pipe") for p in trees[1])


@pytest.mark.parametrize("existing", ["skip", "replace"])
def test_parallel_re_add_follows_the_existing_policy(tmp_path, tiny_bids, existing):
	# workers see the committed store through their overlay, so a re-add is decided
	# exactly as a serial one -- and a replace's old chunks are deleted, not left behind
	target = tmp_path / "store"
	repo = Repo.init(target)
	repo.add(BidsReader(tiny_bids), workers=2)
	repo.commit("first")
	before, _ = _contents(target)
	repo = Repo.open(target, mode="a")
	repo.add(BidsReader(tiny_bids), workers=2, existing=existing)
	repo.commit("again") if existing == "replace" else None
	after, repo = _contents(target)
	assert sorted(after) == sorted(before) and all(np.array_equal(after[p], before[p]) for p in before)
	with pytest.raises(Exception, match="already exists"):
		Repo.open(target, mode="a").add(BidsReader(tiny_bids), workers=2)


def test_replay_keeps_every_workers_attrs_on_a_shared_group(store):
	from neurozarr.util import OverlayStore
	from neurozarr.writer import Writer
	import icechunk
	repo = icechunk.Repository.create(icechunk.in_memory_storage())
	parent = Writer(repo.writable_session("main"))
	parent._flush()
	results = []
	for name, attrs in (("a", {"x": 1, "both": "a"}), ("b", {"y": 2, "both": "b"})):
		overlay = OverlayStore(parent.session.store)
		worker = Writer(type("S", (), {"store": overlay})())
		worker._require(("ses-1", "eeg", name))
		worker._flush()
		worker.add_attrs(Attrs(("ses-1",), attrs))
		results.append((overlay.written(), overlay.deleted))
	for written, deleted in results:
		parent.replay(written, deleted)
	assert dict(parent.root["ses-1"].attrs) == {"x": 1, "y": 2, "both": "b"}
	assert sorted(parent.root["ses-1/eeg"].group_keys()) == ["a", "b"]
