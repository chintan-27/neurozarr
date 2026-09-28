"""Download patients from PhysioNet's CHB-MIT Scalp EEG Database -- a real,
genuinely non-BIDS dataset (flat chbXX/ folders, sequentially-numbered EDF
files, no entities, no sidecars) for testing ManifestReader against real data.

	python scripts/fetch_chbmit.py chb01 chb02 ./datasets/chbmit

PhysioNet's own web server throttles anonymous downloads hard (a single file
measured under 1 MB/s, vs. 11+ MB/s from the mirror below) and physionet.org's
own `wget -r` has no parallelism on top of that. This pulls from the same
public, unsigned S3 mirror PhysioNet publishes under the AWS Open Data
program instead -- same approach as fetch_openneuro.py.

Skips a file whose expected size already matches what's on disk, so a killed
run resumes by just re-running the same command. That resume check is
size-match only -- not provenance a paper can cite -- so every freshly
downloaded file's SHA-256 is recorded in `<dest>/_checksums.sha256`
(`sha256sum -c`-compatible), appended immediately so a killed run doesn't
lose already-computed hashes. A file this run skipped keeps whatever
checksum a prior run recorded; it isn't re-hashed, or "resume" would cost as
much as downloading again.
"""

import hashlib
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BUCKET = "https://s3.amazonaws.com/physionet-open"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
DB_PREFIX = "chbmit/1.0.0/"


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as fid:
		for block in iter(lambda: fid.read(1 << 20), b""):
			digest.update(block)
	return digest.hexdigest()


def _load_checksums(path: Path) -> dict[str, str]:
	if not path.exists():
		return {}
	checksums = {}
	for line in path.read_text().splitlines():
		digest, _, rel = line.partition("  ")
		if digest and rel:
			checksums[rel] = digest
	return checksums


def _list_objects(prefix: str):
	token = None
	while True:
		params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
		if token:
			params["continuation-token"] = token
		resp = requests.get(BUCKET, params=params, timeout=30)
		resp.raise_for_status()
		root = ET.fromstring(resp.text)
		for contents in root.findall("s3:Contents", NS):
			key = contents.find("s3:Key", NS).text
			size = int(contents.find("s3:Size", NS).text)
			yield key, size
		truncated = root.find("s3:IsTruncated", NS).text == "true"
		if not truncated:
			return
		token = root.find("s3:NextContinuationToken", NS).text


def _download_one(key: str, size: int, out_path: Path) -> bool:
	"""Returns whether this file was freshly downloaded (and so needs
	hashing) as opposed to already present at the right size."""
	if out_path.exists() and out_path.stat().st_size == size:
		return False
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with requests.get(f"{BUCKET}/{key}", stream=True, timeout=60) as resp:
		resp.raise_for_status()
		with open(out_path, "wb") as fid:
			for chunk in resp.iter_content(chunk_size=1 << 20):
				fid.write(chunk)
	return True


def main(patients: list[str], dest: str) -> None:
	dest_dir = Path(dest)
	wanted = {f"{DB_PREFIX}{p}/" for p in patients}
	objects = [(key, size) for key, size in _list_objects(DB_PREFIX)
			   if any(key.startswith(w) for w in wanted)]
	total_bytes = sum(size for _, size in objects)
	print(f"{len(objects)} files across {len(patients)} patient(s), {total_bytes / 1e9:.2f} GB")

	checksums_path = dest_dir / "_checksums.sha256"
	checksums = _load_checksums(checksums_path)

	done_bytes = 0
	with ThreadPoolExecutor(max_workers=8) as pool:
		futures = {
			pool.submit(_download_one, key, size, dest_dir / key[len(DB_PREFIX):]): (key, size)
			for key, size in objects
		}
		with checksums_path.open("a") as checksums_file:
			for future in as_completed(futures):
				key, size = futures[future]
				rel = key[len(DB_PREFIX):]
				fresh = future.result()
				if fresh or rel not in checksums:
					digest = _sha256_file(dest_dir / rel)
					checksums[rel] = digest
					checksums_file.write(f"{digest}  {rel}\n")
					checksums_file.flush()
				done_bytes += size
				print(f"  {done_bytes / 1e9:6.2f} / {total_bytes / 1e9:.2f} GB  {key}")


if __name__ == "__main__":
	main(sys.argv[1:-1], sys.argv[-1])
