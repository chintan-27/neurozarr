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
