"""The shared file decoder (readers/filetypes.py): one small file per format family,
made here, so no test depends on downloaded data."""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from neurozarr import Repo
from neurozarr.entities import Entities
from neurozarr.errors import UnsupportedFormatError
from neurozarr.items import Array, Recording, Table
from neurozarr.readers import BidsReader, ManifestReader, SmartReader
from neurozarr.readers import filetypes
from neurozarr.readers.filetypes import decode_file

ENT = Entities("sub-01", "ses-1", "beh")


def _one(path, name="data", meta=None):
	items = list(decode_file(path, ENT, name, meta or {}))
	assert len(items) == 1, items
	return items[0]


def test_bids_physio_tsv_gz_is_a_recording(tmp_path):
	path = tmp_path / "sub-01_task-rest_physio.tsv.gz"
	path.write_bytes(gzip.compress(b"0.1\t5\n0.2\t6\n0.3\t7\n"))
	rec = _one(path, "physio", {"SamplingFrequency": 100.0, "Columns": ["cardiac", "trigger"]})
	assert isinstance(rec, Recording) and rec.raw.ch_names == ["cardiac", "trigger"]
	assert rec.raw.info["sfreq"] == 100.0
	assert np.array_equal(rec.raw.get_data()[1], [5, 6, 7])


def test_motion_tsv_with_a_header_row_is_tolerated(tmp_path):
	path = tmp_path / "sub-01_tracksys-x_motion.tsv"
	path.write_text("HeelX\tHeelY\n1\t2\n3\t4\n")
	rec = _one(path, "motion", {"SamplingFrequency": 50})
	assert rec.raw.ch_names == ["HeelX", "HeelY"] and rec.raw.n_times == 2


def test_wav_is_a_recording(tmp_path):
	from scipy.io import wavfile
	samples = np.array([[0, 1], [-2, 3], [4, -5]], dtype=np.int16)
	wavfile.write(tmp_path / "beep.wav", 8000, samples)
	rec = _one(tmp_path / "beep.wav")
	assert rec.raw.info["sfreq"] == 8000 and np.array_equal(rec.raw.get_data(), samples.T)


def test_media_is_embedded_byte_for_byte(tmp_path, store):
	png = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
	(tmp_path / "face.png").write_bytes(png)
	item = _one(tmp_path / "face.png", "face")
	assert isinstance(item, Array) and item.meta["media_type"] == "image/png"
	assert bytes(item.data) == png


def test_tiff_is_an_array_with_its_axes(tmp_path):
	tifffile = pytest.importorskip("tifffile")
	data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
	tifffile.imwrite(tmp_path / "stack.tif", data)
	item = _one(tmp_path / "stack.tif")
	assert isinstance(item, Array) and np.array_equal(item.data, data) and len(item.dims) == 3


def test_ome_zarr_levels_keep_their_axes(tmp_path):
	import zarr
	group = zarr.open_group(str(tmp_path / "img.ome.zarr"), mode="w", zarr_format=2)
	group.create_array("0", shape=(2, 4, 4), dtype="u1")[:] = 7
	group.attrs["multiscales"] = [{"version": "0.4", "axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}],
								   "datasets": [{"path": "0"}]}]
	[item] = decode_file(tmp_path / "img.ome.zarr", ENT, "img", {})
	assert item.name == "img_level-0" and item.dims == ("c", "y", "x") and int(item.data.sum()) == 7 * 32


def test_dicom_keeps_pixels_and_no_patient_identifiers():
	pydicom = pytest.importorskip("pydicom")
	from pathlib import Path
	path = Path(pydicom.data.get_testdata_file("CT_small.dcm"))
	item = _one(path)
	assert item.data.shape == (128, 128) and item.meta["dicom"]["Modality"] == "CT"
	assert not any("Patient" in key and key != "ImagePositionPatient" and key != "ImageOrientationPatient"
				   for key in item.meta["dicom"])


def test_nwb_series_and_tables(tmp_path):
	pynwb = pytest.importorskip("pynwb")
	from datetime import datetime, timezone
	nwb = pynwb.NWBFile(session_description="t", identifier="x", session_start_time=datetime.now(timezone.utc))
	nwb.add_acquisition(pynwb.TimeSeries(name="lick", data=np.arange(10.0), unit="V", rate=10.0, conversion=2.0))
	nwb.add_trial_column("cue", "cue name")
	nwb.add_trial(start_time=0.0, stop_time=1.0, cue="a")
	with pynwb.NWBHDF5IO(str(tmp_path / "s.nwb"), "w") as io:
		io.write(nwb)
	items = list(decode_file(tmp_path / "s.nwb", ENT, "s", {}))
	rec = next(i for i in items if isinstance(i, Recording))
	assert rec.entities.extra["recording"] == "lick" and np.array_equal(rec.raw.get_data()[0], np.arange(10.0) * 2)
	trials = next(i for i in items if isinstance(i, Table))
	assert list(trials.df["cue"]) == ["a"]


def test_neuroscan_cnt_never_reaches_the_ant_reader(tmp_path, monkeypatch):
	"""antio segfaults on a non-RIFF file; an unreadable NeuroScan CNT must fail cleanly."""
	path = tmp_path / "x.cnt"
	path.write_bytes(b"Version 3.0\0" + b"\0" * 1000)
	monkeypatch.setattr(filetypes.mne.io, "read_raw_ant", lambda *a, **k: pytest.fail("ANT reader called"))
	with pytest.raises(Exception) as caught:
		filetypes.open_raw(path)
	assert not isinstance(caught.value, pytest.fail.Exception)


def test_every_reader_decodes_a_file_the_same_way(tmp_path, store):
	from scipy.io import wavfile
	wav = tmp_path / "bids" / "sub-01" / "beh" / "sub-01_task-x_beh.wav"
	wav.parent.mkdir(parents=True)
	(tmp_path / "bids" / "dataset_description.json").write_text(json.dumps({"Name": "t"}))
	wavfile.write(wav, 1000, np.arange(20, dtype=np.int16))
	readers = [BidsReader(tmp_path / "bids"), SmartReader([tmp_path / "bids"]),
			   ManifestReader(pd.DataFrame([{"sub": "sub-01", "path": str(wav)}]))]
	for reader in readers:
		recs = [i for i in reader.read() if isinstance(i, Recording)]
		assert len(recs) == 1 and np.array_equal(recs[0].raw.get_data()[0], np.arange(20)), type(reader).__name__


def test_a_folder_nothing_reads_is_a_reference(tmp_path):
	folder = tmp_path / "rec.xyzd"
	folder.mkdir()
	(folder / "part").write_bytes(b"1")
	item = _one(folder)
	assert type(item).__name__ == "ExternalFile"


def test_missing_optional_library_names_the_extra(tmp_path, monkeypatch):
	import builtins
	real_import = builtins.__import__

	def no_tifffile(name, *args, **kwargs):
		if name == "tifffile":
			raise ImportError(name)
		return real_import(name, *args, **kwargs)

	(tmp_path / "a.tif").write_bytes(b"II*\0")
	monkeypatch.setattr(builtins, "__import__", no_tifffile)
	with pytest.raises(UnsupportedFormatError, match=r"neurozarr\[formats\]"):
		_one(tmp_path / "a.tif")
