"""Compression + read-speed + accessibility benchmark, generalized to run against
any BIDS dataset (not just BRAVO's) -- or, with --reader manifest, any source
ManifestReader can describe, BIDS or not (e.g. PhysioNet's CHB-MIT, see
build_chbmit_manifest.py) -- so the same numbers are comparable across them.

	python scripts/benchmark_compressors.py [source] [--name LABEL] [--sample N]
	python scripts/benchmark_compressors.py manifest.csv --reader manifest

`source` defaults to ./BIDS (BRAVO). `--sample` caps how many run groups/tables the
per-codec query benchmarks touch, so a 100GB dataset finishes in the same ballpark of
time as a 500MB one -- these are meant to compare codecs/datasets against each other,
not to be an exhaustive read of everything ever written.
"""

import argparse
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from zarr.codecs import BloscCodec, BloscShuffle, ZstdCodec
from zarr.codecs.numcodecs import BZ2, LZ4, LZMA, Delta

from neurozarr import BidsReader, CodecConfig, ManifestReader, Repo
from neurozarr.repo import Subject

verbose = False

ZSTD19 = [ZstdCodec(level=19)]

# (filters, compressors, bitroundK, dtype) — every config tried during compressor exploration
CODECS = {
	"int16+zstd-1": ([], [ZstdCodec(level=1)], 0, "int16"),
	"int16+zstd-9": ([], [ZstdCodec(level=9)], 0, "int16"),
	"int16+zstd-19 (current default)": ([], ZSTD19, 0, "int16"),
	"float16+zstd-19": ([], ZSTD19, 0, "float16"),
	"int16+blosc-zstd-5-shuffle": ([], [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle, typesize=2)], 0, "int16"),
	"int16+blosc-zstd-5-bitshuffle": ([], [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle, typesize=2)], 0, "int16"),
	"int16+blosc-zstd-9-bitshuffle": ([], [BloscCodec(cname="zstd", clevel=9, shuffle=BloscShuffle.bitshuffle, typesize=2)], 0, "int16"),
	"int16+blosc-zstd-5-shuffle-typesize1 (user's config)": ([], [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle, typesize=1)], 0, "int16"),
	"int16+delta+zstd-19": ([Delta(dtype="int16")], ZSTD19, 0, "int16"),
	"int16+lzma": ([], [LZMA()], 0, "int16"),
	"int16+bz2": ([], [BZ2()], 0, "int16"),
	"int16+lz4": ([], [LZ4()], 0, "int16"),
}


def _dir_size(path: Path) -> dict[str, int]:
	"""Split stored bytes into actual chunk data vs. Icechunk's own bookkeeping
	(manifests/snapshots/transactions/overwritten) -- summing rglob("*") alone
	conflates the two, so a codec comparison was really comparing
	chunk-bytes-plus-a-roughly-fixed-overhead rather than chunk bytes alone."""
	chunk_bytes = overhead_bytes = 0
	for f in path.rglob("*"):
		if not f.is_file():
			continue
		size = f.stat().st_size
		if "chunks" in f.relative_to(path).parts:
			chunk_bytes += size
		else:
			overhead_bytes += size
	return {"chunk_bytes": chunk_bytes, "overhead_bytes": overhead_bytes, "total_bytes": chunk_bytes + overhead_bytes}


def run_queries(subject: Subject, sample: int) -> dict:
	"""Goes through the public read API (RecordingView/TableView), not raw zarr
	internals, so this works unchanged across table_schema_version bumps and any
	BIDS dataset's own task/entity naming. Every query is capped at `sample` items
	so this stays a comparable, bounded-time probe regardless of whether the
	subject holds megabytes or hundreds of gigabytes."""
	random.seed(0)
	recordings = subject.recordings()
	if not recordings:
		return {}
	biggest = max(recordings, key=lambda r: r.shape[-1])

	t0 = time.perf_counter()
	biggest.data()
	full_run_s = time.perf_counter() - t0

	n_samp = biggest.shape[-1]
	start = random.randint(0, max(0, n_samp - 1000))
	t0 = time.perf_counter()
	biggest.data(start=start, stop=start + 1000)
	slice_ms = (time.perf_counter() - t0) * 1000

	# decode up to `sample` recordings' channels tables (whatever the dataset calls them)
	sampled = recordings.copy()
	random.shuffle(sampled)
	t0 = time.perf_counter()
	for rec in sampled[:sample]:
		rec.channels()
	table_scan_s = time.perf_counter() - t0

	# cross-run aggregate: mean abs amplitude over up to `sample` recordings sharing
	# the most common task entity, so this exercises "one task across many runs"
	# without assuming any particular task name
	tasks = [r.entities.get("task") for r in recordings if r.entities.get("task")]
	target_task = max(set(tasks), key=tasks.count) if tasks else None
	matching = [r for r in recordings if r.entities.get("task") == target_task] if target_task else recordings
	matching = matching.copy()
	random.shuffle(matching)
	t0 = time.perf_counter()
	total, count = 0.0, 0
	for rec in matching[:sample]:
		values, _ = rec.data()
		total += np.abs(values).sum()
		count += values.size
	aggregate_s = time.perf_counter() - t0

	# random scattered access: small slice from `sample` random recordings
	random.shuffle(sampled)
	t0 = time.perf_counter()
	for rec in sampled[:sample]:
		n = rec.shape[-1]
		s = random.randint(0, max(0, n - 100))
		rec.data(start=s, stop=s + 100)
	random_access_ms = (time.perf_counter() - t0) * 1000

	# listing cost: what it takes to discover this subject's recordings/tables at all
	t0 = time.perf_counter()
	subject.recordings(), subject.tables(), subject.arrays(), subject.external_files()
	listing_s = time.perf_counter() - t0

	return dict(fullRunS=full_run_s, sliceMs=slice_ms, tableScanS=table_scan_s,
		aggregateS=aggregate_s, randomAccessMs=random_access_ms, listingS=listing_s)


def _build_reader(args: argparse.Namespace) -> "BidsReader | ManifestReader":
	subjects = args.subjects.split(",") if args.subjects else None
	if args.reader == "manifest":
		# ManifestReader has no subjects= filter of its own -- filter the table
		# ourselves before constructing it, or a --subjects filter silently
		# does nothing and every codec ingests the whole source.
		df = pd.read_csv(args.source)
		if subjects is not None:
			df = df[df["sub"].isin(subjects)]
		return ManifestReader(df)
	return BidsReader(args.source, subjects=subjects)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("source", nargs="?", default="./BIDS", help="a BIDS directory, or a manifest .csv with --reader manifest")
	parser.add_argument("--reader", choices=["bids", "manifest"], default="bids")
	parser.add_argument("--name", default=None, help="label for the printed table (default: source's name)")
	parser.add_argument("--sample", type=int, default=20, help="cap on run groups/tables touched per query")
	parser.add_argument("--subjects", default=None,
						 help="comma-separated subject ids to ingest, BidsReader only (default: the whole "
							  "dataset -- scope this on anything with more than a handful of subjects)")
	args = parser.parse_args()

	source = Path(args.source)
	name = args.name or source.name
	# Not tempfile.mkdtemp()'s default location: /tmp is commonly a small,
	# RAM-backed tmpfs, and a full 12-codec sweep's converted output for even a
	# modest subject can run into multiple GB -- scratch next to the repo, on
	# real disk, instead.
	scratch_root = Path(__file__).resolve().parent.parent / ".bench_scratch"
	scratch_root.mkdir(exist_ok=True)
	repo_dir = Path(tempfile.mkdtemp(dir=scratch_root))

	cols = ["write_s", "commit_s", "full_run_s", "slice_ms", "table_scan_s", "aggregate_s", "rand_ms", "listing_s",
			"chunk_MB", "overhead_MB"]
	print(f"dataset: {name} ({source}, reader={args.reader})")
	print(f"{'codec':55} " + " ".join(f"{c:>12}" for c in cols))
	for codec_name, (filters, compressors, bitround_k, dtype) in CODECS.items():
		# one repo per subject means no shared store to branch across configs --
		# each codec gets its own fresh base_path instead.
		codec_dir = repo_dir / codec_name
		try:
			repo = Repo.create(codec_dir, codec=CodecConfig(filters, compressors, bitround_k, dtype))

			t0 = time.perf_counter()
			repo.ingest(_build_reader(args))
			write_time = time.perf_counter() - t0

			t0 = time.perf_counter()
			repo.save(f"convert with {codec_name}")
			commit_time = time.perf_counter() - t0

			stored_bytes = _dir_size(codec_dir)

			# benchmark whichever subject actually has recordings
			q: dict = {}
			for sub_id in repo.subjects():
				subject = repo.subject(sub_id)
				q = run_queries(subject, args.sample)
				if q:
					if verbose:
						print(repo.root_of(sub_id).tree())
					break

			vals = [write_time, commit_time, q.get("fullRunS", 0.0), q.get("sliceMs", 0.0), q.get("tableScanS", 0.0),
				q.get("aggregateS", 0.0), q.get("randomAccessMs", 0.0), q.get("listingS", 0.0),
				stored_bytes["chunk_bytes"] / 1e6, stored_bytes["overhead_bytes"] / 1e6]
			print(f"{codec_name:55} " + " ".join(f"{v:12.2f}" for v in vals))
		except Exception as exc:
			# One misbehaving codec (e.g. a numcodecs filter that mishandles a
			# ragged final chunk on some particular data shape) shouldn't cost the
			# whole comparison -- report it and move on to the rest.
			print(f"{codec_name:55} FAILED: {exc!r}")

	shutil.rmtree(repo_dir)


if __name__ == "__main__":
	main()
