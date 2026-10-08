import json
import random

import mne
import numpy as np
import pytest

from neurozarr import Repo
from neurozarr.index import Summary, Table, build_subject, build_summary, encode, stats_of, digest
from neurozarr.query import mask, parse, prune
from neurozarr.readers import BidsReader


def _raw(seed, n_ch=2, sfreq=100.0):
	data = np.random.default_rng(seed).standard_normal((n_ch, 300)) * 1e-5
	return mne.io.RawArray(data, mne.create_info(n_ch, sfreq, "eeg"), verbose=False)


# ---- the blob ------------------------------------------------------------------------

def test_blob_round_trip():
	rows = [{"path": "b", "kind": "recording", "sfreq": 250.0, "run": "01", "meta.flag": True},
			{"path": "a", "kind": "table", "n_columns": 3},
			{"path": "c", "kind": "recording", "sfreq": 1000, "meta.flag": False, "meta.note": "x"}]
	table = Table.decode(encode(rows, {"extra": 1}))
	assert table.paths == ["a", "b", "c"] and table.header["extra"] == 1
	assert table.values("sfreq") == [None, 250.0, 1000.0]
	assert table.values("run") == [None, "01", None]  # text stays text: run-01 stays "01"
	assert table.values("meta.flag") == [None, "true", "false"]
	assert table.find("b") == 1 and table.find("zz") is None
	assert table.row(0) == {"path": "a", "kind": "table", "n_columns": 3.0}
	assert Table.decode(encode([])).n == 0
	with pytest.raises(ValueError):
		Table.decode(b"not a blob")


# ---- pruning never drops a match ---------------------------------------------------------

def test_prune_and_mask_agree_with_brute_force():
	rng = random.Random(0)
	tasks, datatypes = ["rest", "go", "nogo", "sleep"], ["eeg", "ieeg", "beh"]
	subjects = {}
	for s in range(12):
		rows = []
		for i in range(rng.randint(0, 15)):
			row = {"path": f"sub-{s:02d}/ses-1/x/i{i}", "kind": rng.choice(["recording", "table"]),
				   "sub": f"sub-{s:02d}", "task": rng.choice(tasks), "datatype": rng.choice(datatypes)}
			if rng.random() < 0.8:
				row["sfreq"] = float(rng.choice([128, 250, 256, 500, 1000]))
			rows.append(row)
		blob = encode(rows, {"sub": f"sub-{s:02d}", "snapshot": "S", "sessions": ["ses-1"],
							 "attrs": {"age": rng.randint(5, 70)}, "session_attrs": {}})
		subjects[f"sub-{s:02d}"] = blob
	summary = Summary(Table.decode(build_summary([stats_of(Table.decode(b), digest(b)) for b in subjects.values()])))
	queries = ["task == 'rest'", "sfreq >= 500", "sfreq < 200 and task in ('go', 'nogo')", "not task == 'rest'",
			   "datatype == 'ieeg' or sfreq == 256", "exists(sfreq)", "participant.age < 30 and kind == 'recording'",
			   "task != 'sleep'", "sfreq > 10000", "task == 'missing'", "sub == 'sub-03'", "task > 'n'"]
	for text in queries:
		expr = parse(text)
		keep = prune(expr, summary)
		for i, sub in enumerate(summary.subjects):
			table = Table.decode(subjects[sub])
			subject = summary.table.row(i)
			hits = mask(expr, table, subject)
			brute = [_matches(text, {**row, "participant.age": table.header["attrs"]["age"]}) for row in table.rows()]
			assert list(hits) == brute, (text, sub)
			if any(brute):
				assert keep[i], f"{text!r} pruned {sub}, which has a match"


def _matches(text, row):
	"""The same semantics in plain Python: a missing value never matches a comparison."""
	class Row(dict):
		def __missing__(self, key):
			return _MISSING
	names = Row({k.replace(".", "__"): v for k, v in row.items()})
	expr = text.replace("participant.age", "participant__age").replace("exists(sfreq)", "(sfreq is not _MISSING)")
	return bool(eval(expr, {"_MISSING": _MISSING}, names))


class _Missing:
	def _no(self, *_):
		return False
	__eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _no
	__hash__ = object.__hash__


_MISSING = _Missing()


# ---- the index follows every kind of change ---------------------------------------------

def _rows_now(repo):
	"""What a full rebuild of every subject would produce, versus what the store holds."""
	repo = Repo.open(repo.target)
	for sub in repo.subjects():
		stored = {r["path"]: r for r in repo._subject_table(sub).rows()}
		fresh = {r["path"]: r for r in Table.decode(build_subject(repo.root_of(sub), sub).finish("S")).rows()}
		assert stored == fresh, sub
	return repo


def test_incremental_index_equals_a_full_rebuild(tmp_path, tiny_bids):
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	repo.add(tiny_bids)
	repo.commit("import")
	_rows_now(repo)

	v = repo.subject("sub-01").visit("ses-1")
	v.add_recording(_raw(1), task="extra")                                 # add
	v.add_recording(_raw(2, n_ch=3), task="extra", existing="replace")     # replace
	repo.commit("edit")
	_rows_now(repo)
	repo.get("sub-01/ses-1/ieeg/task-extra").set_attrs({"Manufacturer": "Acme"})  # metadata only
	repo.commit("describe")
	assert _rows_now(repo)._subject_table("sub-01").values("meta.Manufacturer").count("Acme") == 1
	repo.mv("sub-01/ses-1/ieeg/task-extra", "task-renamed")                # mv
	repo.subject("sub-02").add_visit("ses-2").add_recording(_raw(3), task="new")  # a new session
	repo.commit("move + new session")
	_rows_now(repo)
	repo.rm("sub-03", recursive=True)                                      # rm subject
	repo.rm("sub-02/ses-2", recursive=True)                                # rm session
	dropped = repo.commit("removals")
	assert _rows_now(repo).subjects() == ["sub-01", "sub-02"]
	assert Repo.open(store)._subject_table("sub-02").header["sessions"] == ["ses-1"]

	repo.revert(dropped)                                                   # revert
	assert _rows_now(repo).subjects() == ["sub-01", "sub-02", "sub-03"]
	repo.switch("side", create=True)
	repo.subject("sub-02").visit("ses-1").add_recording(_raw(4), task="side")
	repo.commit("side work")
	repo.switch("main")
	repo.subject("sub-02").visit("ses-1").add_recording(_raw(5), task="main")
	repo.commit("main work")
	repo.merge("side")                                                     # merge (item-level)
	assert {"task-side", "task-main"} <= {p.rsplit("/", 1)[-1] for p in _rows_now(repo).query(sub="sub-02").paths}
	repo.reset_hard("main~2")                                              # reset --hard
	_rows_now(repo)


def test_parallel_add_is_indexed(tmp_path, tiny_bids):
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	repo.add(BidsReader(tiny_bids), workers=2)
	repo.commit("parallel")
	assert len(_rows_now(repo).query(kind="recording")) == 3


# ---- querying -------------------------------------------------------------------------

def test_query(tmp_path, tiny_bids):
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	repo.add(tiny_bids)
	repo.subject("sub-02").visit("ses-1").add_recording(_raw(9, n_ch=5, sfreq=1000.0), task="fast")
	first = repo.commit("import")
	repo = Repo.open(store)

	assert len(repo.query()) == 4
	fast = repo.query("sfreq >= 500")
	assert fast.paths == ["sub-02/ses-1/ieeg/task-fast"]
	assert repo.query(sfreq__ge=500, kind="recording").paths == fast.paths
	assert repo.query("participant.age < 31").paths == ["sub-01/ses-1/eeg/task-rest"]   # participants.tsv: 30, 31, 32
	assert repo.query(participant__age__ge=31, task="rest").paths == ["sub-02/ses-1/eeg/task-rest", "sub-03/ses-1/eeg/task-rest"]
	assert len(repo.query("exists(meta.PowerLineFrequency)")) == 0
	frame = repo.query(datatype="eeg").df
	assert list(frame["path"]) == [f"sub-0{i}/ses-1/eeg/task-rest" for i in (1, 2, 3)]
	assert set(frame["sfreq"]) == {200.0} and set(frame["n_channels"]) == {3.0} and "participant.age" in frame
	view = repo.get("sub-02/ses-1/ieeg/task-fast")
	assert view.data()[0].shape == (5, 300)
	with pytest.raises(KeyError):
		repo.get("sub-02/ses-1/ieeg/nope")
	assert [v.path for v in repo.query(task="fast")] == ["sub-02/ses-1/ieeg/task-fast"]

	writable = Repo.open(store, mode="a")
	writable.rm("sub-02", recursive=True)
	writable.commit("drop sub-02")
	now = Repo.open(store)
	assert len(now.query()) == 2 and len(now.query(version=first)) == 4      # any version
	assert len(now.query(version="main~1")) == 4 and now.subjects(version=first[:8]) == ["sub-01", "sub-02", "sub-03"]
	assert [d["recordings"] for d in now.describe()] == [1, 1]


def test_cli_query(tmp_path, tiny_bids, capsys):
	from neurozarr import cli
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	repo.add(tiny_bids)
	repo.commit("import")
	assert cli.main(["-C", store, "query", "participant.age >= 31", "--count"]) == 0
	assert capsys.readouterr().out.strip() == "2"
	assert cli.main(["-C", store, "query", "--where", "sub=sub-01", "--json", "--columns", "sfreq"]) == 0
	assert json.loads(capsys.readouterr().out) == [{"path": "sub-01/ses-1/eeg/task-rest", "sfreq": 200.0}]


def test_migrate_from_schema_2(tmp_path, tiny_bids):
	"""A schema-2 store (JSON catalog, no index) is upgraded by one walk per subject."""
	from neurozarr.writer import Writer
	store = str(tmp_path / "store")
	repo = Repo.init(store)
	repo.add(tiny_bids)
	repo.commit("import")
	expected = sorted(Repo.open(store).query().paths)

	# turn it back into schema 2: catalog attr, no index
	dataset = repo._icechunk_repo("_dataset")
	session = dataset.writable_session("main")
	writer = Writer(session, None, "_dataset")
	manifest = dict(writer.root.attrs.asdict()["_neurozarr"], schema_version=2, catalog={})
	writer.root.attrs.put({**writer.root.attrs.asdict(), "_neurozarr": manifest})
	writer.drop("_index")
	session.commit("as schema 2")
	with pytest.raises(Exception, match="needs migration"):
		Repo.open(store)

	result = Repo._for_migration(store).migrate()
	assert result["changed"] and result["schema_version"] == 3
	assert sorted(Repo.open(store).query().paths) == expected
	assert Repo._for_migration(store).migrate()["changed"] is False
