import datetime
import json

import h5py
import mne
import numpy as np
import pytest
import scipy.io

from neurozarr import Repo
from neurozarr.errors import ValidationError
from neurozarr.readers import SmartReader
from neurozarr.readers.smart import scan_folders

UTC = datetime.timezone.utc


def _raw(seed=0, his_id=None, day=1):
	data = np.random.default_rng(seed).standard_normal((3, 500)) * 1e-5
	raw = mne.io.RawArray(data, mne.create_info(["a", "b", "c"], 250.0, "eeg"), verbose=False)
	raw.set_meas_date(datetime.datetime(2020, 1, day, tzinfo=UTC))
	if his_id:
		raw.info["subject_info"] = {"his_id": his_id}
	return raw


def _save(raw, path):
	path.parent.mkdir(parents=True, exist_ok=True)
	raw.save(path, overwrite=True, verbose=False)


def _write(path, text=""):
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text)


def _recs(df):
	return df[df["format"] != "sidecar"]


def test_layouts_metadata_duplicates_and_unreadable(tmp_path):
	bids, flat = tmp_path / "bids", tmp_path / "flat"
	stem = "sub-01_ses-a_task-rest_run-1_ieeg"
	_save(_raw(0), bids / "sub-01/ses-a/ieeg" / f"{stem}.fif")
	_write(bids / "sub-01/ses-a/ieeg" / f"{stem}.json", json.dumps({"PowerLineFrequency": 60}))
	_write(bids / "sub-01/ses-a/ieeg/sub-01_ses-a_task-rest_run-1_events.tsv", "onset\tduration\n1\t0\n")
	_write(bids / "dataset_description.json", json.dumps({"Name": "demo", "BIDSVersion": "1.9"}))
	_write(bids / "participants.tsv", "participant_id\tage\nsub-01\t31\nsub-99\t40\n")
	_write(bids / "sub-01/notes.pdf")
	_save(_raw(1), flat / "chb01/chb01_03.fif")
	_save(_raw(2), flat / "chb01/chb01_04.fif")
	_save(_raw(3), flat / "chb02/chb02_01.fif")
	(flat / "chb02/copy").mkdir()
	(flat / "chb02/copy/chb02_01.fif").write_bytes((flat / "chb02/chb02_01.fif").read_bytes())
	_write(flat / "chb02/broken.edf", "not an edf")

	reader = SmartReader([bids, flat])
	report, recs = reader.report, _recs(reader.df)

	assert report["recordings"] == 4 and report["subjects"] == 3
	assert set(report["layouts"].values()) == {"bids", "folder per subject"}
	assert report["skipped_extensions"] == {".pdf": 1}
	assert [d["path"].endswith("copy/chb02_01.fif") for d in report["duplicates"]] == [True]
	assert [u["path"].endswith("broken.edf") for u in report["unreadable"]] == [True]
	b = recs[recs["sub"] == "sub-01"].iloc[0]
	assert (b["ses"], b["datatype"], b["task"], b["run"]) == ("ses-a", "ieeg", "rest", "1")
	assert sorted(recs[recs["sub"] == "sub-chb01"]["run"]) == ["03", "04"]  # from chb01_03 / chb01_04
	assert report["run_rules"] == {"filename number": 2}
	assert [q["field"] for q in reader.questions] == ["datatype"]  # flat files have no datatype folder

	store = tmp_path / "store"
	repo = Repo.init(store)
	repo.add(reader)
	repo.commit("smart")
	repo = Repo.open(store)
	assert len(list(repo.find())) == 4
	rec = repo.subject("sub-01").visit("ses-a").recording(task="rest", run=1)
	assert rec.meta["PowerLineFrequency"] == 60  # same-stem JSON sidecar merged
	assert abs(rec.data()[0] - _raw(0).get_data()).max() < 1e-9
	assert "sub-99" not in repo.subjects()  # participants.tsv rows only for subjects found


def test_header_ids_dates_and_answers(tmp_path):
	_save(_raw(0, his_id="P7", day=5), tmp_path / "flat/rec_b.fif")
	_save(_raw(1, his_id="P7", day=2), tmp_path / "flat/rec_a.fif")
	_save(_raw(2, his_id="Q2"), tmp_path / "flat/other.fif")
	df, report = scan_folders([tmp_path / "flat"])
	recs = _recs(df)
	assert report["sub_rules"] == {"file header": 3}
	assert set(recs["sub"]) == {"sub-P7", "sub-Q2"}
	# rec_a was recorded first though it sorts second by path
	p7 = recs[recs["sub"] == "sub-P7"].set_index("path")["run"]
	assert p7[str(tmp_path / "flat/rec_a.fif")] == "1" and report["run_rules"] == {"recording date": 2}

	# one anonymized id on every file says nothing: fall back to the filename and ask
	_save(_raw(0, his_id="Surrogate"), tmp_path / "anon/p1_x.fif")
	_save(_raw(1, his_id="Surrogate"), tmp_path / "anon/p2_x.fif")
	df, report = scan_folders([tmp_path / "anon"])
	assert report["sub_rules"] == {"filename prefix": 2}
	q = next(q for q in report["questions"] if q["field"] == "sub_from")
	df, report = scan_folders([tmp_path / "anon"], {q["pattern"]: {"sub_from": r"^p(\d)_", "datatype": "ieeg"}})
	assert set(_recs(df)["sub"]) == {"sub-1", "sub-2"} and set(df["datatype"]) == {"ieeg"}
	assert report["questions"] == [] and report["sub_rules"] == {"answer": 2}


def test_array_formats(tmp_path):
	data = np.random.default_rng(0).standard_normal((4, 1000))
	(tmp_path / "s1").mkdir()
	np.save(tmp_path / "s1/a.npy", data.T)  # samples x channels: transposed back
	_write(tmp_path / "s1/a.json", json.dumps({"SamplingFrequency": 100}))
	scipy.io.savemat(tmp_path / "s1/b.mat", {"eeg": data, "fs": 200.0, "labels": np.array(["Fz", "Cz", "Pz", "Oz"])})
	with h5py.File(tmp_path / "s1/c.nwb", "w") as f:  # NWB's shape: rate beside the data
		f["acquisition/es/data"] = data.T
		f["acquisition/es/starting_time"] = 0.0
		f["acquisition/es/starting_time"].attrs["rate"] = 300.0
		f["processing/other/starting_time"] = 0.0
		f["processing/other/starting_time"].attrs["rate"] = 1.0
	np.save(tmp_path / "s1/d.npy", data)
	(tmp_path / "s1/d.npy").rename(tmp_path / "s1/nodot")  # no extension: sniffed
	np.save(tmp_path / "s1/e.npy", data[:2])  # no rate anywhere

	reader = SmartReader(tmp_path)
	recs = _recs(reader.df).set_index(_recs(reader.df)["path"].str.rsplit("/", n=1).str[-1])
	assert recs.loc["a.npy", "sfreq"] == 100 and recs.loc["b.mat", "sfreq"] == 200
	assert recs.loc["c.nwb", "sfreq"] == 300 and recs.loc["nodot", "format"] == "npy"
	assert sorted((q["field"], q["count"]) for q in reader.questions) == [("datatype", 5), ("sfreq", 2), ("sub_from", 5)]

	store = tmp_path / "store"
	repo = Repo.init(store)
	repo.add(SmartReader(tmp_path, answers={str(tmp_path / "s1/nodot"): {"sfreq": 50}}))
	repo.commit("arrays")
	repo = Repo.open(store)
	found = {r.meta["SourceFile"]: r for r in repo.find()}
	assert set(found) == {"a.npy", "b.mat", "c.nwb", "nodot"}  # e.npy kept as a reference
	assert found["b.mat"].raw().ch_names == ["Fz", "Cz", "Pz", "Oz"]
	assert found["nodot"].raw().info["sfreq"] == 50
	for rec in found.values():
		assert np.abs(rec.data()[0] - data).max() < 1e-6


def test_nothing_found_raises(tmp_path):
	_write(tmp_path / "x/readme.txt", "hi")
	with pytest.raises(ValidationError):
		SmartReader(tmp_path)


def test_smart_reader_cli(tmp_path, capsys):
	from neurozarr import cli
	_save(_raw(0), tmp_path / "a/s1/x.fif")
	_save(_raw(1), tmp_path / "b/s2/x.fif")
	dest = tmp_path / "store"
	answers = tmp_path / "answers.json"
	assert cli.main(["init", str(dest)]) == 0
	assert cli.main(["-C", str(dest), "add", str(tmp_path / "a"), str(tmp_path / "b"),
					 "--answers", str(answers), "-q"]) == 0
	assert cli.main(["-C", str(dest), "commit", "-m", "two sites"]) == 0
	assert len(Repo.open(dest).subjects()) == 2
	assert "guessing datatype='eeg'" in capsys.readouterr().err and not answers.exists()


def _edf_bytes(patient=b"P042 M 01-JAN-1980 X", start=b"14.07.2415.40.51", ns=2):
	head = b"0       " + patient.ljust(80) + b"Startdate X".ljust(80) + start
	head += str(256 * (ns + 1)).encode().ljust(8) + b" " * 44 + b"1".ljust(8) + b"1".ljust(8) + str(ns).encode().ljust(4)
	assert len(head) == 256
	return head + b" " * 256 * ns


def test_edf_header_is_read_without_mne(tmp_path):
	from neurozarr.readers.smart import _edf_header
	good = tmp_path / "a.edf"
	good.write_bytes(_edf_bytes())
	h = _edf_header(good)
	assert h["his_id"] == "P042"
	assert h["meas_date"] == datetime.datetime(2024, 7, 14, 15, 40, 51, tzinfo=UTC)
	(tmp_path / "unknown.edf").write_bytes(_edf_bytes(patient=b"X X X X"))
	assert _edf_header(tmp_path / "unknown.edf")["his_id"] is None  # EDF+ "X" means unknown
	(tmp_path / "cut.edf").write_bytes(_edf_bytes()[:400])
	with pytest.raises(ValidationError):
		_edf_header(tmp_path / "cut.edf")


def test_derivatives_and_lettered_runs(tmp_path, store):
	_save(_raw(0), tmp_path / "d/sub-01/ses-1/ieeg/sub-01_ses-1_task-sleep_ieeg.fif")
	_save(_raw(1), tmp_path / "d/derivatives/bipolar/sub-01_ses-1_task-sleep_ieeg_bipolar.fif")
	_save(_raw(2), tmp_path / "d/chb17/chb17a_03.fif")
	_save(_raw(3), tmp_path / "d/chb17/chb17b_03.fif")
	reader = SmartReader(tmp_path / "d")
	recs = _recs(reader.df)
	assert set(recs["sub"]) == {"sub-01", "sub-chb17"}
	assert sorted(recs[recs["sub"] == "sub-chb17"]["run"]) == ["a03", "b03"]  # "03" alone would collide
	assert sorted(recs["prefix"].dropna()) == ["derivatives/bipolar"]

	repo = Repo.init(store)
	repo.add(reader)
	repo.commit("deriv")
	paths = sorted(r.path for r in Repo.open(store).find())
	assert len(paths) == 4 and sum("derivatives/bipolar" in p for p in paths) == 1  # stored apart from the raw one
