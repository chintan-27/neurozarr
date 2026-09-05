"""Convert a pile of files that follow no particular layout, by describing them
in a manifest table. This is the "universal" path: nothing about your folder
names or file naming has to match anything.

    python examples/convert_manifest.py
"""

import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neurozarr import ManifestReader, Repo

scratch = Path(tempfile.mkdtemp())


def make_example_files() -> pd.DataFrame:
	"""Stand-ins for your own files, in deliberately unstructured locations."""
	# any format mne can read works here (EDF, BrainVision, FIF, ...) -- FIF just
	# happens to need no extra export dependency
	recording_path = scratch / "patient1_visit1_raw.fif"
	info = mne.create_info(["LFP_L", "LFP_R"], sfreq=250.0, ch_types="eeg")
	raw = mne.io.RawArray(np.random.default_rng(0).normal(scale=1e-5, size=(2, 2500)), info, verbose=False)
	raw.save(recording_path, overwrite=True, verbose=False)

	log_path = scratch / "patient1_therapy_log.csv"
	pd.DataFrame({"onset": [0.0, 12.5], "amplitude": [1.5, 2.0]}).to_csv(log_path, index=False)

	# one row per file: who it belongs to, what it is, and where it lives
	return pd.DataFrame([
		{"sub": "sub-001", "ses": "ses-1", "datatype": "ieeg", "task": "Stream", "run": 1,
		 "path": str(recording_path), "meta": '{"Manufacturer": "Medtronic"}'},
		{"sub": "sub-001", "ses": "ses-1", "datatype": "beh", "task": "TherapyLog",
		 "path": str(log_path)},
	])


def main():
	manifest = make_example_files()
	print(manifest.to_string(index=False), "\n")

	repo = Repo.create("./manifest_study.zarr")
	repo.ingest(ManifestReader(manifest))
	repo.save("convert from manifest")

	visit = repo.subject("sub-001").visit("ses-1")
	print("recordings:", [r.path for r in visit.recordings()])
	print("tables    :", [t.path for t in visit.tables()])


if __name__ == "__main__":
	main()
