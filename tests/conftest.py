import itertools

import mne
import numpy as np
import pandas as pd
import pytest

from neurozarr import Repo

_counter = itertools.count()


@pytest.fixture
def store():
	"""A fresh in-memory store per test -- no files touched, no cleanup needed."""
	return f"memory://test{next(_counter)}"


@pytest.fixture
def raw():
	info = mne.create_info(["LFP_L", "LFP_R"], sfreq=250.0, ch_types="eeg")
	rng = np.random.default_rng(0)
	return mne.io.RawArray(rng.normal(scale=1e-5, size=(2, 500)), info, verbose=False)


@pytest.fixture
def raw_calibrated():
	"""Same shape and rate as `raw`, but carrying the per-channel EDF-style
	calibration (_raw_extras[0]["cal"/"units"/"offsets"]) that only EDF-family
	mne readers populate -- a plain RawArray's _raw_extras is just
	{"orig_nchan": N}, so `raw` above only ever exercises writer.py's float32
	fallback, never the int16 packing path that is the actual default codec.
	Values are exact multiples of the chosen LSB, so the int16 round trip is
	lossless by construction rather than merely within some tolerance."""
	lsb = 1e-9
	rng = np.random.default_rng(0)
	counts = rng.integers(-1000, 1000, size=(2, 500))
	info = mne.create_info(["LFP_L", "LFP_R"], sfreq=250.0, ch_types="eeg")
	raw = mne.io.RawArray(counts.astype(np.float64) * lsb, info, verbose=False)
	raw._raw_extras = [{
		"cal": np.array([1.0, 1.0]),
		"units": np.array([lsb, lsb]),
		"offsets": np.array([0.0, 0.0]),
	}]
	return raw


@pytest.fixture
def table():
	return pd.DataFrame({"onset": [0.5, 1.5], "amp": [1.25, 2.5], "label": ["a", "b"]})


@pytest.fixture
def written(store, raw, table):
	"""One subject, one visit, one recording and one table, already committed."""
	repo = Repo(store)
	visit = repo.create_subject("sub-001", attrs={"age": 63}).add_visit("ses-1", attrs={"device": "Percept"})
	visit.add_recording(raw, task="Stream", run=1)
	visit.add_behavioral_table(table, task="Log")
	repo.save("test data")
	return store
