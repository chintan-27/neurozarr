"""Build a ManifestReader table for a downloaded chunk of CHB-MIT (see
fetch_chbmit.py). CHB-MIT has no entities of its own -- just chbXX/chbXX_YY.edf
files -- so this is the "you describe it yourself" step ManifestReader exists
for: no sub-/ses- folders, no BIDS sidecars, nothing to infer.

Each patient's own chbXX-summary.txt is real per-file metadata (recording
start/end time and, critically, the real seizure onset/offset times -- the
whole reason this dataset exists) -- parsed here and attached to each
recording's `meta` column, not dropped on the floor.

	python scripts/build_chbmit_manifest.py ./datasets/chbmit manifest.csv
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

_FILE_BLOCK = re.compile(
	r"File Name:\s*(\S+)\s*"
	r"File Start Time:\s*([\d:]+)\s*"
	r"File End Time:\s*([\d:]+)\s*"
	r"Number of Seizures in File:\s*(\d+)"
)
_SEIZURE = re.compile(r"Seizure(?:\s*\d*)\s*Start Time:\s*(\d+)\s*seconds.*?"
					   r"Seizure(?:\s*\d*)\s*End Time:\s*(\d+)\s*seconds", re.DOTALL)


def _parse_summary(path: Path) -> dict[str, dict]:
	"""One entry per File Name block: start/end time-of-day, and the real
	seizure onset/offset times (seconds into that file) if any."""
	text = path.read_text(errors="replace")
	per_file = {}
	# split on each "File Name:" block so seizure lines are scoped to their own file
	blocks = re.split(r"(?=File Name:)", text)[1:]
	for block in blocks:
		match = _FILE_BLOCK.match(block)
		if not match:
			continue
		name, start, end, n_seizures = match.groups()
		seizures = []
		if int(n_seizures) > 0:
			# seizure lines are individually numbered ("Seizure 1 Start Time", "Seizure 2 Start Time", ...)
			starts = re.findall(r"Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds", block)
			ends = re.findall(r"Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds", block)
			seizures = [{"onset_s": int(s), "offset_s": int(e)} for s, e in zip(starts, ends)]
		per_file[name] = {"file_start_time": start, "file_end_time": end, "seizures": seizures}
	return per_file


def main(root: str, out_csv: str) -> None:
	root_dir = Path(root)
	rows = []
	for patient_dir in sorted(root_dir.iterdir()):
		if not patient_dir.is_dir():
			continue
		summary_path = patient_dir / f"{patient_dir.name}-summary.txt"
		per_file = _parse_summary(summary_path) if summary_path.exists() else {}
		for edf in sorted(patient_dir.glob("*.edf")):
			match = re.match(r"chb(\d+)_(\d+)", edf.stem)
			if not match:
				continue
			patient, run = match.groups()
			rows.append({
				"sub": f"sub-chb{patient}",
				"path": str(edf.resolve()),
				"datatype": "eeg",
				# CHB-MIT is session-less -- these are just sequential recording
				# segments, no ses- column at all
				"task": "scalp",
				"run": run,
				"meta": json.dumps(per_file.get(edf.name, {})),
			})
	df = pd.DataFrame(rows)
	df.to_csv(out_csv, index=False)
	with_seizures = sum(1 for r in rows if json.loads(r["meta"]).get("seizures"))
	print(f"{len(df)} rows across {df['sub'].nunique()} patient(s), "
		  f"{with_seizures} file(s) with at least one real seizure annotation -> {out_csv}")


if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])
