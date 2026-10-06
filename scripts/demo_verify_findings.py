"""Re-run verify() against the three real RNS stores and log what it finds,
instead of transcribing a remembered number into a slide. Hardware doesn't
matter here either -- these are correctness counts, not timing -- so this
runs locally against the stores already on disk.

	.venv/bin/python scripts/demo_verify_findings.py
"""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neurozarr.verify import verify

SOURCE_BIDS = Path("datasets/rns")
# bravo_bids_vs_source and rns_bids_vs_source already confirmed 0 problems on
# the first run -- only the two informative legs are worth the re-run cost.
STORES = {
	# rns_manifest_vs_source / rns_explicit_vs_source already logged from the
	# first pass (the old manifest store predates the zero-padding fix, so its
	# 3,167 "missing" recordings were run-01 filed as run-1, not data loss).
	"rns_manifest_fixed_vs_source": (SOURCE_BIDS, Path(".bench_scratch/stores/rns_manifest_fixed"), None),
}


def _categorize(problems: list[str]) -> dict[str, Any]:
	"""A real value/shape/dtype mismatch always says so explicitly ('values
	differ', 'shape ... != source', 'dtype ... != source', 'null mask
	differs'); 'metadata missing/differs' is a key-level sidecar gap; a bare
	'missing from store: <path>' is a whole item never captured -- genuinely
	missing recording DATA (path ends in '/data') is the one case that would
	actually be a fidelity bug, not a sidecar-completeness gap, so it's
	called out on its own rather than folded into either bucket."""
	real_mismatch_markers = ("values differ", "!= source", "null mask differs",
							  "columns missing from store", "extra columns in store")
	metadata_only = 0
	missing_recording_data = []
	missing_sidecar_item = 0
	real_mismatches = 0
	for p in problems:
		if p.startswith("missing from store: "):
			where = p[len("missing from store: "):]
			if where.endswith("/data"):
				missing_recording_data.append(where)
			else:
				missing_sidecar_item += 1
		elif ": metadata missing from store:" in p or ": metadata differs for key(s):" in p:
			metadata_only += 1
		elif any(marker in p for marker in real_mismatch_markers):
			real_mismatches += 1
		else:
			real_mismatches += 1  # unrecognized shape -- err toward flagging it, not hiding it
	return {
		"total": len(problems),
		"metadata_key_completeness": metadata_only,
		"missing_sidecar_item_whole": missing_sidecar_item,
		"missing_recording_data_REAL_BUG_IF_NONZERO": len(missing_recording_data),
		"missing_recording_data_examples": missing_recording_data[:10],
		"real_value_shape_dtype_mismatches": real_mismatches,
	}


def main() -> None:
	out_path = Path(".bench_scratch/paper_logs/verify_findings.json")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	# Preserve bravo/rns_bids's already-confirmed 0-problem results from the
	# first pass instead of re-running them -- only STORES' two legs rerun here.
	results: dict[str, object] = json.loads(out_path.read_text()) if out_path.exists() else {}

	for name, (source, store, reader) in STORES.items():
		if not store.exists():
			results[name] = {"skipped": f"store not found at {store}"}
			print(f"{name}: SKIPPED (store not found at {store})")
			continue
		print(f"running verify({source}, {store}) ...")
		problems = verify(source, store, sample_limit=None, reader=reader)
		counts = _categorize(problems)
		raw_path = out_path.parent / f"{name}_raw_problems.txt"
		raw_path.write_text("\n".join(problems))
		results[name] = {"source": str(source), "store": str(store), **counts,
						  "sample_problems": problems[:10], "raw_problems_file": str(raw_path)}
		print(f"{name}: {counts}")

	results["_note"] = (
		"The historical '18,412 missing items' figure (convert_rns_explicit.py's "
		"parse_entities bug) describes a now-fixed bug's first detection and is not "
		"expected to reproduce -- it's a bug-history fact, not a steady-state number."
	)
	out_path.write_text(json.dumps(results, indent=2))
	print(f"\nwritten to {out_path}")


if __name__ == "__main__":
	main()
