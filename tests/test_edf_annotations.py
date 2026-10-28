import numpy as np
from mne.io.edf import edf

import neurozarr.readers  # noqa: F401  # installs the joined annotation reader


def _tal(text: str, records: int, per_record: int) -> np.ndarray:
	"""An EDF+ annotation channel as mne hands it over: int16 samples, one row per record."""
	raw = text.encode().ljust(records * per_record * 2, b"\x00")
	return np.frombuffer(raw, dtype="<i2").reshape(records, per_record).astype(np.float64)


def test_joined_records_give_the_same_annotations_as_mne():
	assert getattr(edf._read_annotations_edf, "_neurozarr_joined", False)
	text = "+0\x14\x14\x00" + "+1.5\x150.25\x14stim A\x14\x00" + "+3\x14seizure\x14note\x14\x00" + "+7\x14end\x14\x00"
	tal = _tal(text, records=9, per_record=6)  # annotations cross record boundaries
	got = edf._read_annotations_edf(tal)
	# mne's own function, row by row, as it was before neurozarr wrapped it
	from inspect import getclosurevars
	original = getclosurevars(edf._read_annotations_edf).nonlocals["original"]
	want = original(tal)
	assert list(got.onset) == list(want.onset) and list(got.duration) == list(want.duration)
	assert list(got.description) == list(want.description) == ["stim A", "seizure", "note", "end"]
