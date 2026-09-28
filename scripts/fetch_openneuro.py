"""Download an OpenNeuro dataset from its public, unsigned S3 bucket -- no aws-cli,
datalad, or openneuro-py needed, just `requests` (already a transitive dep via mne).

	python scripts/fetch_openneuro.py ds007095 ./datasets/rns [--workers 8]

Skips a file whose expected size already matches what's on disk, so a killed run
resumes by just re-running the same command. Downloads `--workers` files at once:
a single S3 connection tops out well below this network's real capacity, so one
file at a time leaves most of the available bandwidth unused on a multi-GB dataset.

A resume check is size-match only (cheap, and good enough to resume a killed
run), but that alone isn't provenance a paper can cite -- every freshly
downloaded file's SHA-256 is recorded in `<dest>/_checksums.sha256`
(`sha256sum -c`-compatible), appended immediately so a killed run doesn't lose
already-computed hashes. A file this run *skipped* (already on disk, right
size) keeps whatever checksum a prior run recorded for it; it isn't
re-hashed, since that would make "resume" as slow as downloading again.
"""

import argparse
import hashlib
import sys
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BUCKET = "https://s3.amazonaws.com/openneuro.org"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


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


def main(dataset_id: str, dest: str, workers: int) -> None:
	dest_dir = Path(dest)
	prefix = f"{dataset_id}/"
	objects = list(_list_objects(prefix))
	total_bytes = sum(size for _, size in objects)
	print(f"{dataset_id}: {len(objects)} files, {total_bytes / 1e9:.2f} GB, {workers} workers")

	checksums_path = dest_dir / "_checksums.sha256"
	checksums = _load_checksums(checksums_path)

	done_bytes = 0
	lock = threading.Lock()
	with ThreadPoolExecutor(max_workers=workers) as pool:
		futures = {
			pool.submit(_download_one, key, size, dest_dir / key[len(prefix):]): (key, size)
			for key, size in objects
		}
		with checksums_path.open("a") as checksums_file:
			for future in as_completed(futures):
				key, size = futures[future]
				rel = key[len(prefix):]
				fresh = future.result()
				if fresh or rel not in checksums:
					digest = _sha256_file(dest_dir / rel)
					checksums[rel] = digest
					checksums_file.write(f"{digest}  {rel}\n")
					checksums_file.flush()
				with lock:
					done_bytes += size
					print(f"  {done_bytes / 1e9:6.2f} / {total_bytes / 1e9:.2f} GB  {key}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("dataset_id")
	parser.add_argument("dest")
	parser.add_argument("--workers", type=int, default=8)
	args = parser.parse_args()
	main(args.dataset_id, args.dest, args.workers)
