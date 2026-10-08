import mne
import numpy as np
import pytest
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

from neurozarr import CodecConfig, Repo
from neurozarr.codecs import CodecPlan, group_keys, plan_from_items, probe
from neurozarr.errors import ValidationError
from neurozarr.entities import Entities
from neurozarr.items import Recording
from neurozarr.readers import BidsReader


def _shuffle(chain):
	return str(getattr(chain[0].shuffle, "value", chain[0].shuffle))


def test_probe_follows_the_data_and_optimize():
	rng = np.random.default_rng(0)
	noise = rng.standard_normal((4, 65536)).astype("float32")
	# mostly-zero int16 with rare spikes: byte/bit shuffle lose to no shuffle
	sparse = ((rng.random((4, 131072)) < 0.02) * rng.integers(-500, 500, (4, 131072))).astype("int16")
	assert _shuffle(probe(noise)[0]) != "noshuffle"
	chain, sizes = probe(sparse)
	assert _shuffle(chain) == "noshuffle" and chain[0].typesize == 2 and set(sizes) == {"noshuffle", "shuffle", "bitshuffle"}
	speed = probe(sparse, "speed")[0][0]
	assert (str(getattr(speed.cname, "value", speed.cname)), speed.clevel) == ("lz4", 1)
	assert (probe(sparse, "balanced")[0][0].clevel, probe(sparse, "size")[0][0].clevel) == (5, 9)
	assert probe(np.array(["a", "bb"]))[0] == [ZstdCodec(level=5)]  # strings: no shuffle to probe
	with pytest.raises(ValidationError):
		probe(noise, "smallest")


def test_reader_plan_decides_every_codec_and_is_recorded(tmp_path, tiny_bids):
	reader = BidsReader(tiny_bids)
	plan = reader.codec_plan()
	assert {"Recording", "Recording/float32", "Table"} <= set(plan.groups)
	assert plan.evidence["Recording/float32"]["files"] == 3  # one per subject, spread

	repo = Repo.init(tmp_path / "store")
	repo.add(reader)
	repo.commit("convert")
	repo = Repo.open(tmp_path / "store")
	rec = repo.subject("sub-02").visit("ses-1").recording(task="rest")
	(codec,) = rec.array.compressors
	assert isinstance(codec, BloscCodec) and codec.clevel == 5
	assert str(getattr(codec.shuffle, "value", codec.shuffle)) == plan.groups["Recording/float32"]
	events = repo.subject("sub-02").root()["ses-1/eeg/task-rest/events"]  # a sidecar lives with its recording
	codecs = {str(a.dtype): list(a.compressors) for _, a in events.arrays()}
	assert codecs
	assert all(isinstance(c[0], (BloscCodec, ZstdCodec)) for c in codecs.values())  # planned, not zarr's default
	recorded = repo._dataset_root().attrs["codec_plan"]
	assert recorded["reader"] == "BidsReader" and (recorded["optimize"], recorded["probe_level"]) == ("balanced", 2)
	assert recorded["shuffle"]["Recording/float32"] == plan.groups["Recording/float32"]


def test_codecconfig_pin_overrides_the_plan_with_a_warning(tmp_path, tiny_bids):
	repo = Repo.init(tmp_path / "store", codec=CodecConfig(compressors=[GzipCodec(level=4)]))
	with pytest.warns(UserWarning, match="overrides BidsReader's codec plan"):
		repo.add(BidsReader(tiny_bids))
	repo.commit("convert")
	repo = Repo.open(tmp_path / "store")
	assert {tuple(r.array.compressors) for r in repo.find()} == {(GzipCodec(level=4),)}
	assert repo._dataset_root().attrs["codec_plan"]["pinned_by"] == "CodecConfig"


def _rec(sub, n_ch, seed, sparse=False):
	rng = np.random.default_rng(seed)
	data = rng.standard_normal((n_ch, 4000)) * 1e-5
	if sparse:
		data *= rng.random((n_ch, 4000)) < 0.02
	raw = mne.io.RawArray(data, mne.create_info(n_ch, 250.0, "eeg"), verbose=False)
	return Recording(Entities(sub, "ses-1", "eeg", {"task": "rest"}), raw, {})


def test_probe_levels():
	items = [_rec("sub-1", 2, 0), _rec("sub-2", 8, 1)]
	float32 = np.dtype("float32")
	assert plan_from_items(items, probe_level=1).groups.keys() == {"Recording"}
	assert plan_from_items(items, probe_level=2).groups.keys() == {"Recording", "Recording/float32"}
	level3 = plan_from_items(items, probe_level=3)
	assert {"Recording/float32/2x4000", "Recording/float32/8x4000"} <= set(level3.groups)
	assert group_keys("Recording", float32, (2, 4000))[2] == "Recording/float32/2x4000"
	# a chunk shape the sample never saw falls back to the dtype's choice
	unseen = level3.compressors("Recording", float32, (99, 7))
	assert str(getattr(unseen[0].shuffle, "value", unseen[0].shuffle)) == level3.groups["Recording/float32"]

	assert CodecPlan(probe_level=0).compressors("Recording", float32, (2, 4000)) == "auto"
	sparse = _rec("sub-3", 4, 2, sparse=True).raw.get_data().astype("float32")
	assert CodecPlan(probe_level=4).compressors("Recording", float32, sparse.shape, sample=lambda: sparse) == probe(sparse)[0]
	# never probed: strings and embedded files
	assert CodecPlan().compressors("File", np.dtype("uint8"), (100,)) == [ZstdCodec(level=5)]
	assert CodecPlan().compressors("Table", np.dtypes.StringDType(), (10,)) == [ZstdCodec(level=5)]


def test_options_are_checked():
	with pytest.raises(ValidationError):
		BidsReader(".", optimize="fastest")
	with pytest.raises(ValidationError):
		BidsReader(".", probe_level=5)
	with pytest.raises(ValidationError):
		BidsReader(".", probe_level=True)


def test_rename_keeps_the_codec(store, raw):
	repo = Repo.init(store, codec=CodecConfig(compressors=[BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")]))
	repo.create_subject("sub-1").add_visit("ses-1").add_recording(raw, task="A")
	repo.commit("a")
	repo = Repo.open(store, mode="a")
	before = list(repo.subject("sub-1").visit("ses-1").recording(task="A").array.compressors)
	repo.subject("sub-1").visit("ses-1").recording(task="A").rename("task-B")
	repo.commit("b")
	after = Repo.open(store).subject("sub-1").visit("ses-1").recordings()[0]
	assert after.path.endswith("task-B") and list(after.array.compressors) == before


def test_explicit_add_array_uses_codecconfig(store):
	repo = Repo.init(store, codec=CodecConfig(compressors=[GzipCodec(level=2)]))
	repo.create_subject("sub-1").add_visit("ses-1").add_array("spec", np.zeros((3, 10)), dims=("ch", "f"), datatype="eeg")
	repo.commit("a")
	group = Repo.open(store).subject("sub-1").root()
	assert list(group["ses-1/eeg/spec"].compressors) == [GzipCodec(level=2)]


def test_cli_optimize_flag(tmp_path, tiny_bids):
	from neurozarr import cli
	store = str(tmp_path / "store")
	assert cli.main(["init", store]) == 0
	assert cli.main(["-C", store, "add", str(tiny_bids), "--optimize", "speed", "-q"]) == 0
	assert cli.main(["-C", store, "commit", "-m", "fast"]) == 0
	codecs = {str(getattr(c.cname, "value", c.cname)) for r in Repo.open(tmp_path / "store").find()
			  for c in r.array.compressors}
	assert codecs == {"lz4"}
