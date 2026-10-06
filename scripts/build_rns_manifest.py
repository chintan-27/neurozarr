"""Build a ManifestReader table for RNS_Epilepsy-iBIDS -- RNS is real BIDS, so
this manifest is redundant with BidsReader, but it's what makes the three-way
reader comparison possible (BidsReader vs. ManifestReader vs.
convert_rns_explicit.py, same source data, three different ingest paths).

One row per file: the recording itself, plus its channels.tsv/events.tsv
sidecars each as their own row with name= set explicitly, since ManifestReader
maps one row to one item.

	python scripts/build_rns_manifest.py datasets/rns manifest_rns.csv
"""

import sys
from pathlib import Path

import pandas as pd

from neurozarr.entities import parse_entities


def main(root: str, out_csv: str) -> None:
	root_dir = Path(root)
	rows = []
	for edf_path in sorted(root_dir.glob("sub-*/ses-*/ieeg/*_ieeg.edf")):
		sub_id, ses_id = edf_path.parts[-4], edf_path.parts[-3]
		stem = edf_path.stem.removesuffix("_ieeg")
		# parse_entities parses the whole stem, sub-/ses- included -- those two
		# must lose to the folder names (unprefixed "01" vs "sub-01"), so spread
		# entities first and let the explicit columns win
		entities = {k: v for k, v in parse_entities(stem).items() if k not in ("sub", "ses")}
		rows.append({**entities, "sub": sub_id, "ses": ses_id, "path": str(edf_path.resolve()),
					 "datatype": "ieeg"})
		for suffix in ("channels", "events"):
			tsv_path = edf_path.with_name(f"{stem}_{suffix}.tsv")
			if tsv_path.exists():
				rows.append({**entities, "sub": sub_id, "ses": ses_id, "path": str(tsv_path.resolve()),
							 "datatype": "ieeg", "name": suffix})
	df = pd.DataFrame(rows)
	df.to_csv(out_csv, index=False)
	print(f"{len(df)} rows across {df['sub'].nunique()} subject(s) -> {out_csv}")


if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])
