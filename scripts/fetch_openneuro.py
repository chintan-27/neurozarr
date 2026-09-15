"""Download an OpenNeuro dataset from its public, unsigned S3 bucket -- no aws-cli,
datalad, or openneuro-py needed, just `requests` (already a transitive dep via mne).

	python scripts/fetch_openneuro.py ds007095 ./datasets/rns

Skips a file whose expected size already matches what's on disk, so a killed run
resumes by just re-running the same command.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BUCKET = "https://s3.amazonaws.com/openneuro.org"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


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


def main(dataset_id: str, dest: str) -> None:
	dest_dir = Path(dest)
	prefix = f"{dataset_id}/"
	objects = list(_list_objects(prefix))
	total_bytes = sum(size for _, size in objects)
	print(f"{dataset_id}: {len(objects)} files, {total_bytes / 1e9:.2f} GB")

	done_bytes = 0
	for key, size in objects:
		out_path = dest_dir / key[len(prefix):]
		if out_path.exists() and out_path.stat().st_size == size:
			done_bytes += size
			continue
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with requests.get(f"{BUCKET}/{key}", stream=True, timeout=60) as resp:
			resp.raise_for_status()
			with open(out_path, "wb") as fid:
				for chunk in resp.iter_content(chunk_size=1 << 20):
					fid.write(chunk)
		done_bytes += size
		print(f"  {done_bytes / 1e9:6.2f} / {total_bytes / 1e9:.2f} GB  {key}")


if __name__ == "__main__":
	main(*sys.argv[1:3])
