"""Do every arm's stored values match the raw source, in numbers rather than
rounded hashes? The harness's checksums round before hashing; for data stored
as float32 (non-EDF sources such as EEGLAB .set) the raw arm reads float64, and
rounding to a fixed number of decimals lands differently on the two even though
the values agree to ~1e-12 -- so the raw arm's checksum differs from every
stored arm's on those datasets, and a hash can't distinguish that from a real
data difference. This compares the arrays themselves: a mid-recording 1 s
window plus the first and last channel's full trace, for each arm's own target
recording (the largest one, same pick as the harness), against the raw arm.

	python scripts/check_arm_equivalence.py --name hbn_eeg \\
		--store raw=.../raw --store neurozarr=.../neurozarr ... --out equiv.json

An arm passes if its max absolute error is within --atol, or its error is
within --rtol of the signal's peak: float32 storage rounds each value to ~6e-8
of itself, so a window holding one large-amplitude sample (EEG DC offsets reach
0.1 V) legitimately exceeds any fixed absolute tolerance while being a
faithful copy -- an absolute-only check flagged HBN EEG's four float32-stored
arms identically at 7e-9 for exactly that reason. A real data difference would
be orders of magnitude above 1e-6 of peak, not a rounding residue.

Exits non-zero if any arm fails both tests against raw.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench.arms import ARMS  # noqa: E402
from bench.runner import _pick_target  # noqa: E402


def _samples(store, target) -> dict[str, np.ndarray]:
	n = max(1, int(round(target.sfreq)))
	start = min(target.n_samples // 2, max(0, target.n_samples - n))
	return {
		"window": np.asarray(store.window_read(target.id, start, n), dtype=np.float64),
		"first_channel": np.asarray(store.channel_read(target.id, 0), dtype=np.float64),
		"last_channel": np.asarray(store.channel_read(target.id, target.n_channels - 1), dtype=np.float64),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--store", action="append", required=True, metavar="ARM=PATH",
						help="repeat per arm; one of them must be raw=")
	parser.add_argument("--name", default="")
	parser.add_argument("--atol", type=float, default=1e-9)
	parser.add_argument("--rtol", type=float, default=1e-6, help="max error as a fraction of the signal's peak")
	parser.add_argument("--out", type=Path, default=None)
	args = parser.parse_args()

	stores = {}
	for spec in args.store:
		arm, path = spec.split("=", 1)
		stores[arm] = ARMS[arm].open(Path(path))
	if "raw" not in stores:
		parser.error("need raw=PATH as the reference")

	ref_target = _pick_target(stores["raw"].recordings())
	reference = _samples(stores["raw"], ref_target)
	report: dict[str, dict] = {}
	failed = False
	for arm, store in stores.items():
		target = _pick_target(store.recordings())
		got = _samples(store, target)
		entry: dict = {"target": target.id, "shape": [target.n_channels, target.n_samples]}
		ok = (target.n_channels, target.n_samples) == (ref_target.n_channels, ref_target.n_samples)
		for key, ref in reference.items():
			if got[key].shape != ref.shape:
				entry[key] = f"SHAPE {got[key].shape} != {ref.shape}"
				ok = False
				continue
			diff = float(np.abs(got[key] - ref).max())
			peak = float(np.abs(ref).max())
			rel = diff / peak if peak > 0 else (0.0 if diff == 0 else float("inf"))
			entry[key] = diff
			entry[key + "_rel_peak"] = rel
			ok = ok and (diff <= args.atol or rel <= args.rtol)
		entry["status"] = "PASS" if ok else "FAIL"
		failed = failed or not ok
		report[arm] = entry
		print(f"{args.name:18s} {arm:12s} {entry['status']}  "
			  + "  ".join((f"{k}={entry[k]:.3e} (rel {entry[k + '_rel_peak']:.1e})") if isinstance(entry[k], float)
						  else f"{k}={entry[k]}" for k in ("window", "first_channel", "last_channel")))
		store.close()

	if args.out:
		args.out.write_text(json.dumps({"dataset": args.name, "atol": args.atol, "rtol": args.rtol, "arms": report}, indent=2))
	return 1 if failed else 0


if __name__ == "__main__":
	sys.exit(main())
