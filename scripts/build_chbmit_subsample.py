"""Build a fixed, seeded byte-budget subsample of CHB-MIT's manifest, for
benchmarks where running the full 43GB dataset through every comparator arm
isn't a good use of time (see the paper plan's Phase 4).

	python scripts/build_chbmit_subsample.py ./datasets/chbmit manifest_subsample.csv --target-gb 5 --seed 0

Patients are shuffled with the given seed and added greedily until the byte
budget is reached -- not a fixed patient count. CHB-MIT's per-patient size is
highly skewed (measured: 20 of 24 patients, picked uniformly at random, came
to 37GB of the 43GB total, not the ~5GB a "20 of 24" fraction suggests), so a
fixed patient count doesn't actually control wall-clock or disk cost the way
a byte budget does. The resulting patient count varies run to run only with
the seed, not with which patients happen to be small.

Reuses build_chbmit_manifest.py's own summary-parsing for the real seizure
metadata -- this only adds the patient selection and a sidecar recording
which patients were picked and why, so the choice is reproducible and citable
rather than "whatever the shell glob happened to list first".
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

import build_chbmit_manifest


def _patient_bytes(patient_dir: Path) -> int:
	return sum(f.stat().st_size for f in patient_dir.glob("*.edf"))


def main(root: str, out_csv: str, target_bytes: int, seed: int) -> None:
	root_dir = Path(root)
	patients = sorted(p.name for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("chb"))

	rng = random.Random(seed)
	order = patients[:]
	rng.shuffle(order)

	chosen: list[str] = []
	running_bytes = 0
	for patient in order:
		if running_bytes >= target_bytes:
			break
		chosen.append(patient)
		running_bytes += _patient_bytes(root_dir / patient)
	chosen.sort()

	rows = []
	for patient in chosen:
		patient_dir = root_dir / patient
		summary_path = patient_dir / f"{patient}-summary.txt"
		per_file = build_chbmit_manifest._parse_summary(summary_path) if summary_path.exists() else {}
		for edf in sorted(patient_dir.glob("*.edf")):
			match = re.match(r"chb(\d+)_(\d+)", edf.stem)
			if not match:
				continue
			pid, run = match.groups()
			rows.append({
				"sub": f"sub-chb{pid}", "path": str(edf.resolve()), "datatype": "eeg",
				"task": "scalp", "run": run, "meta": json.dumps(per_file.get(edf.name, {})),
			})

	df = pd.DataFrame(rows)
	df.to_csv(out_csv, index=False)

	sidecar = Path(out_csv).with_suffix(".subsample.json")
	sidecar.write_text(json.dumps({
		"seed": seed, "target_bytes": target_bytes, "actual_bytes": running_bytes,
		"patients_available": patients, "patients_chosen": chosen, "rows": len(df),
	}, indent=2))
	print(f"{len(df)} rows across {len(chosen)} of {len(patients)} patients, "
		  f"{running_bytes / 1e9:.2f} GB (target {target_bytes / 1e9:.2f} GB, seed={seed}) -> {out_csv}")
	print(f"selection recorded -> {sidecar}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("root")
	parser.add_argument("out_csv")
	parser.add_argument("--target-gb", type=float, default=5.0)
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()
	main(args.root, args.out_csv, int(args.target_gb * 1e9), args.seed)
