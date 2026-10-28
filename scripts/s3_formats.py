"""Put the comparison formats into the S3 server's bucket folder (run before the server
starts; the posix backend serves plain files as objects, so a filesystem copy is an
upload). For one dataset, copies the existing HDF5 / NWB-HDF5 / NWB-Zarr stores built by
the read benchmarks, and each recording's raw source files with an index of which files
belong to which recording -- under <bucket dir>/formats/<name>/. Skips what is already
there. Also writes a 64 MiB probe object for measuring the network path itself.

	python scripts/s3_formats.py SOURCE NAME OLD_STORES_DIR BUCKET_DIR

OLD_STORES_DIR holds hdf5_mat73/, nwb_hdf5/, nwb_zarr/ (as phase7_crossformat.slurm uses).
"""

import json
import os
import shutil
import sys
from pathlib import Path

from neurozarr.items import Recording
from neurozarr.readers import reader_for


def rec_id(entities) -> str:
	return "/".join(entities.path())  # the id every benchmark arm uses


def main() -> int:
	source, name, old, bucket = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
	dest = bucket / "formats" / name
	dest.mkdir(parents=True, exist_ok=True)

	probe = bucket / "formats" / "_probe" / "64mib.bin"
	if not probe.exists():
		probe.parent.mkdir(parents=True, exist_ok=True)
		probe.write_bytes(os.urandom(64 << 20))

	for arm in ("hdf5_mat73", "nwb_hdf5", "nwb_zarr"):
		src, out = (old / arm).resolve(), dest / arm
		if out.exists():
			print(f"{name}/{arm}: already there")
		elif src.exists():
			shutil.copytree(src, out, symlinks=False)
			print(f"{name}/{arm}: copied from {src}")
		else:
			print(f"{name}/{arm}: MISSING at {src}, not served", file=sys.stderr)

	index_file = dest / "raw_index.json"
	if index_file.exists():
		print(f"{name}/raw: already there")
		return 0
	index: dict[str, list[str]] = {}
	for item in reader_for(source).read():
		if not isinstance(item, Recording):
			continue
		files = [Path(f) for f in item.raw.filenames if f]
		main_file = files[0]
		# companions a reader opens through the main file (EEGLAB .fdt, BrainVision .eeg/.vmrk)
		files += [p for p in main_file.parent.glob(main_file.stem + ".*") if p != main_file and p.suffix in (".fdt", ".eeg", ".vmrk")]
		keys = []
		for f in files:
			key = f"raw/{len(index):05d}/{f.name}"  # flat and unique: a recording's files sit together
			(dest / key).parent.mkdir(parents=True, exist_ok=True)
			shutil.copyfile(f, dest / key)
			keys.append(key)
		index[rec_id(item.entities)] = keys
	index_file.write_text(json.dumps(index, indent=1))
	print(f"{name}/raw: {len(index)} recordings copied")
	return 0


if __name__ == "__main__":
	sys.exit(main())
