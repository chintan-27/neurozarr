"""Read a dataset served over S3 (s3_serve.slurm, through your SSH tunnel; or a local
server) -- neurozarr and the comparison formats, measured the same way.

	python scripts/s3_remote_read.py --env job_<id>.env --store bravo --out remote.jsonl
		[--scenario network,cold,session] [--formats ...] [--recordings 10] [--reps 3] [--port 7070]

Scenarios, identical for every format:

- network: a 1-byte GET (round trip); a 64 MiB object as one stream and as 16 ranges at once.
- cold: from nothing to the data of one recording -- open the dataset, open the
  recording, read it.
- session: open the dataset once, then open and read --recordings recordings in turn,
  as an analysis loop does.

Every cold and session measurement runs in a fresh child process (imports done before
the clock starts), so no format reuses another measurement's caches. Each format is
opened the way its users open it:

- neurozarr: Repo.open, then get + data;
- nwb_zarr: pynwb's NWBZarrIO on the session's file (kept open within a session);
  nwb_zarr_direct: the recording's data array opened by its path -- NWB-Zarr's best case;
- hdf5_mat73: h5py on the dataset's one file (an s3fs file object), then the recording;
- nwb_hdf5: pynwb's NWBHDF5IO on the session's file through s3fs (kept open within a session);
- raw: the recording's files downloaded, then read with mne (no partial reads).

Reads: full recording, a 10 s window from the middle, channel 0. One JSON row per
measurement: dataset-open time, and per recording its open and read times.
"""

import argparse
import json
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

BUCKET = "bench"
FORMATS = ("neurozarr", "nwb_zarr", "nwb_zarr_direct", "hdf5_mat73", "nwb_hdf5", "raw")
TESTS = ("full", "window_10s", "one_channel")


def now() -> str:
	return datetime.now(timezone.utc).isoformat()


def settings(env_file: str, port: int) -> dict[str, Any]:
	env = dict(line.strip().split("=", 1) for line in open(env_file) if "=" in line)
	endpoint = f"http://127.0.0.1:{port}"
	key, secret = env["AWS_ACCESS_KEY_ID"], env["AWS_SECRET_ACCESS_KEY"]
	return {
		"env": env, "endpoint": endpoint, "key": key, "secret": secret,
		"nz": dict(endpoint_url=endpoint, allow_http=True, force_path_style=True, region="us-east-1",
				   access_key_id=key, secret_access_key=secret),
		"fs": dict(key=key, secret=secret, client_kwargs={"endpoint_url": endpoint, "region_name": "us-east-1"}),
	}


def boto(s: dict[str, Any]) -> Any:
	import boto3
	return boto3.client("s3", endpoint_url=s["endpoint"], aws_access_key_id=s["key"],
						aws_secret_access_key=s["secret"], region_name="us-east-1")


def nwb_file(rec_id: str) -> tuple[str, str]:
	"""(file stem, series name): one NWB file per subject+session, as the benchmark arms wrote them."""
	parts = rec_id.split("/")
	ses = parts[1] if len(parts) == 4 else "ses-none"
	return f"{parts[0]}_{ses}", parts[-1]


# ---- formats: dataset() opens the dataset, recording(rid) a recording, which then
# answers full() / window(a, n) / channel(i) ----

class Neurozarr:
	def __init__(self, s: dict[str, Any], store: str):
		self.s, self.target = s, f"s3://{BUCKET}/{store}"

	def dataset(self) -> None:
		from neurozarr import Repo
		self.repo = Repo.open(self.target, **self.s["nz"])

	def recording(self, rid: str) -> Any:
		view = self.repo.get(rid)

		class R:
			def full(self) -> Any: return view.data()[0]
			def window(self, a: int, n: int) -> Any: return view.data(start=a, stop=a + n)[0]
			def channel(self, i: int) -> Any: return view.data(picks=i)[0]
		return R()


class TimeMajor:
	"""NWB stores (samples, channels); return channel-major like every other format."""
	def __init__(self, data: Any):
		self.data = data
	def full(self) -> Any: return self.data[:, :].T
	def window(self, a: int, n: int) -> Any: return self.data[a:a + n, :].T
	def channel(self, i: int) -> Any: return self.data[:, i]


class NwbZarr:
	def __init__(self, s: dict[str, Any], store: str):
		self.s, self.prefix, self.files = s, f"formats/{store}/nwb_zarr", {}

	def dataset(self) -> None:
		pass

	def recording(self, rid: str) -> Any:
		from hdmf_zarr import NWBZarrIO
		stem, name = nwb_file(rid)
		if stem not in self.files:
			io = NWBZarrIO(f"s3://{BUCKET}/{self.prefix}/{stem}.nwb.zarr", mode="r", storage_options=self.s["fs"])
			self.files[stem] = (io, io.read())
		return TimeMajor(self.files[stem][1].acquisition[name].data)


class NwbZarrDirect:
	def __init__(self, s: dict[str, Any], store: str):
		self.s, self.prefix = s, f"formats/{store}/nwb_zarr"

	def dataset(self) -> None:
		pass

	def recording(self, rid: str) -> Any:
		import zarr
		from zarr.storage import FsspecStore
		stem, name = nwb_file(rid)
		url = f"s3://{BUCKET}/{self.prefix}/{stem}.nwb.zarr/acquisition/{name}/data"
		return TimeMajor(zarr.open_array(store=FsspecStore.from_url(url, storage_options=self.s["fs"], read_only=True), mode="r"))


class Hdf5:
	"""The read benchmarks' HDF5 layout: one store.h5 for the dataset, a group per recording, channel-major."""
	def __init__(self, s: dict[str, Any], store: str):
		self.s, self.key = s, f"{BUCKET}/formats/{store}/hdf5_mat73/store.h5"

	def dataset(self) -> None:
		import h5py
		import s3fs  # type: ignore[import-untyped]
		self.file = h5py.File(s3fs.S3FileSystem(**self.s["fs"]).open(self.key, "rb"), "r")

	def recording(self, rid: str) -> Any:
		import numpy as np
		ds = self.file[rid + "/data"]
		scale = np.asarray(ds.attrs["data_scale"]) if "data_scale" in ds.attrs else None
		offset = np.asarray(ds.attrs["data_offset"]) if scale is not None else None

		def decode(raw: Any, ch: Any = slice(None)) -> Any:
			if scale is None:
				return raw
			return raw * scale[ch] + offset[ch] if raw.ndim == 1 else raw * scale[:, None] + offset[:, None]

		class R:
			def full(self) -> Any: return decode(ds[:, :])
			def window(self, a: int, n: int) -> Any: return decode(ds[:, a:a + n])
			def channel(self, i: int) -> Any: return decode(ds[i, :], i)
		return R()


class NwbHdf5:
	def __init__(self, s: dict[str, Any], store: str):
		self.s, self.prefix, self.files = s, f"{BUCKET}/formats/{store}/nwb_hdf5", {}

	def dataset(self) -> None:
		import s3fs  # type: ignore[import-untyped]
		self.fs = s3fs.S3FileSystem(**self.s["fs"])

	def recording(self, rid: str) -> Any:
		import h5py
		from pynwb import NWBHDF5IO
		stem, name = nwb_file(rid)
		if stem not in self.files:
			io = NWBHDF5IO(file=h5py.File(self.fs.open(f"{self.prefix}/{stem}.nwb", "rb"), "r"), mode="r")
			self.files[stem] = (io, io.read())
		return TimeMajor(self.files[stem][1].acquisition[name].data)


class Raw:
	"""No partial reads: every read downloads the recording's files, then mne reads them."""
	def __init__(self, s: dict[str, Any], store: str):
		self.s, self.prefix = s, f"formats/{store}"

	def dataset(self) -> None:
		self.s3 = boto(self.s)
		self.index = json.loads(self.s3.get_object(Bucket=BUCKET, Key=f"{self.prefix}/raw_index.json")["Body"].read())

	def recording(self, rid: str) -> Any:
		import mne
		keys, s3, prefix = self.index[rid], self.s3, self.prefix

		def read(**kw: Any) -> Any:
			with tempfile.TemporaryDirectory() as tmp:
				for key in keys:
					s3.download_file(BUCKET, f"{prefix}/{key}", str(Path(tmp) / Path(key).name))
				return mne.io.read_raw(Path(tmp) / Path(keys[0]).name, preload=False, verbose=False).get_data(**kw)

		class R:
			def full(self) -> Any: return read()
			def window(self, a: int, n: int) -> Any: return read(start=a, stop=a + n)
			def channel(self, i: int) -> Any: return read(picks=[i])[0]
		return R()


OPENERS: dict[str, Callable[..., Any]] = {"neurozarr": Neurozarr, "nwb_zarr": NwbZarr, "nwb_zarr_direct": NwbZarrDirect,
										  "hdf5_mat73": Hdf5, "nwb_hdf5": NwbHdf5, "raw": Raw}


def child(args: argparse.Namespace) -> int:
	"""One measurement in this fresh process: dataset open, then each recording opened and read."""
	s = settings(args.env, args.port)
	# the same rule for every format: imports -- including readers a library loads lazily
	# on first use -- happen before the clock; everything from creating the S3 client on
	# (botocore/aiobotocore/Icechunk client set-up included) is timed
	import boto3, h5py, mne, s3fs, zarr  # noqa: F401,E401
	import hdmf_zarr, pynwb  # noqa: F401,E401
	import neurozarr  # noqa: F401
	from zarr.storage import FsspecStore  # noqa: F401
	from hdmf_zarr import NWBZarrIO  # noqa: F401
	from pynwb import NWBHDF5IO  # noqa: F401
	for reader in ("read_raw", "read_raw_edf", "read_raw_bdf", "read_raw_eeglab", "read_raw_brainvision"):
		getattr(mne.io, reader)  # mne imports each reader module on first access
	fmt = OPENERS[args.format](s, args.store)
	recs = json.loads(args.recs)  # [[rid, window_start, window_len], ...]
	reads = {"full": lambda r, a, n: r.full(), "window_10s": lambda r, a, n: r.window(a, n),
			 "one_channel": lambda r, a, n: r.channel(0)}[args.test]
	t = time.perf_counter()
	fmt.dataset()
	out: dict[str, Any] = {"dataset_open_s": time.perf_counter() - t, "recordings": []}
	for rid, a, n in recs:
		t0 = time.perf_counter()
		handle = fmt.recording(rid)
		t1 = time.perf_counter()
		data = reads(handle, a, n)
		out["recordings"].append({"recording": rid, "open_s": t1 - t0, "read_s": time.perf_counter() - t1,
								  "bytes": int(data.nbytes)})
	print("RESULT " + json.dumps(out), flush=True)
	return 0


def network(s: dict[str, Any], row: Callable[..., None]) -> None:
	s3 = boto(s)
	s3.head_bucket(Bucket=BUCKET)  # sets up the connection; not timed

	def timed(fn: Callable[[], Any], reps: int) -> list[float]:
		times = []
		for _ in range(reps):
			t = time.perf_counter()
			fn()
			times.append(time.perf_counter() - t)
		return times

	probe = "formats/_probe/64mib.bin"
	row("network", "round_trip_get1b", seconds=timed(
		lambda: s3.get_object(Bucket=BUCKET, Key=probe, Range="bytes=0-0")["Body"].read(), 30))
	row("network", "get_64mib_1x", seconds=timed(lambda: s3.get_object(Bucket=BUCKET, Key=probe)["Body"].read(), 3),
		bytes=64 << 20)

	def parallel() -> None:
		def part(i: int) -> bytes:
			return s3.get_object(Bucket=BUCKET, Key=probe, Range=f"bytes={i << 22}-{((i + 1) << 22) - 1}")["Body"].read()
		with ThreadPoolExecutor(16) as pool:
			assert sum(map(len, pool.map(part, range(16)))) == 64 << 20
	row("network", "get_64mib_16x4mib", seconds=timed(parallel, 3), bytes=64 << 20)


def main() -> int:
	p = argparse.ArgumentParser()
	p.add_argument("--env", required=True)
	p.add_argument("--store", required=True, help="dataset name loaded by s3_serve.slurm, e.g. bravo")
	p.add_argument("--out")
	p.add_argument("--port", type=int, default=7070)
	p.add_argument("--scenario", default="network,cold,session")
	p.add_argument("--formats", default=",".join(FORMATS))
	p.add_argument("--tests", default=",".join(TESTS))
	p.add_argument("--recordings", type=int, default=10)
	p.add_argument("--reps", type=int, default=3)
	p.add_argument("--seed", type=int, default=0)
	# internal: one measurement in a child process
	p.add_argument("--child", action="store_true")
	p.add_argument("--format")
	p.add_argument("--test")
	p.add_argument("--recs")
	args = p.parse_args()
	if args.child:
		return child(args)

	s = settings(args.env, args.port)
	base = {"client": socket.gethostname(), "server": f"{s['env']['NODE']}:{s['env']['PORT']}", "store": args.store}
	out = open(args.out, "a")

	def row(fmt: str, scenario: str, **fields: Any) -> None:
		r = {**base, "utc": now(), "format": fmt, "scenario": scenario, **fields}
		out.write(json.dumps(r) + "\n")
		out.flush()
		secs = fields.get("seconds") or fields.get("total_s")
		label = fields.get("test", "")
		if isinstance(secs, list):
			print(f"{fmt:<16}{scenario:<20}{label:<12} {statistics.median(secs) * 1000:10.1f} ms", flush=True)
		elif secs is not None:
			print(f"{fmt:<16}{scenario:<20}{label:<12} {secs * 1000:10.1f} ms  {fields.get('recording', fields.get('note', ''))}",
				  flush=True)
		else:
			print(f"{fmt:<16}{scenario:<20}{label:<12} ERROR {fields.get('error', '')[:120]}", flush=True)

	scenarios = args.scenario.split(",")
	if "network" in scenarios:
		network(s, row)

	# the recordings: a seeded sample from neurozarr's index, the same ids for every format
	from neurozarr import Repo
	repo = Repo.open(f"s3://{BUCKET}/{args.store}", **s["nz"])
	views = {v.path: v for v in repo.find()}
	sample = random.Random(args.seed).sample(sorted(views), min(args.recordings, len(views)))
	recs = []
	for rid in sample:
		n_s = views[rid].shape[-1]
		win = min(n_s, int(10 * (views[rid].sfreq or 1.0)))
		recs.append([rid, (n_s - win) // 2, win])

	def run(fmt: str, test: str, chosen: list[list[Any]]) -> dict[str, Any]:
		cmd = [sys.executable, __file__, "--child", "--env", args.env, "--port", str(args.port), "--store", args.store,
			   "--format", fmt, "--test", test, "--recs", json.dumps(chosen)]
		proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
		result = next((line[7:] for line in proc.stdout.splitlines() if line.startswith("RESULT ")), None)
		if result is None:
			tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
			raise RuntimeError(tail)
		return json.loads(result)

	for fmt in args.formats.split(","):
		for test in args.tests.split(","):
			for rep in range(args.reps):
				if "cold" in scenarios:
					for rid, a, n in recs:
						try:
							r = run(fmt, test, [[rid, a, n]])
							one = r["recordings"][0]
							row(fmt, "cold", test=test, rep=rep, recording=rid, dataset_open_s=r["dataset_open_s"],
								open_s=one["open_s"], read_s=one["read_s"], bytes=one["bytes"],
								total_s=r["dataset_open_s"] + one["open_s"] + one["read_s"])
						except Exception as exc:  # noqa: BLE001 -- keep the other formats' rows
							row(fmt, "cold", test=test, rep=rep, recording=rid, error=str(exc)[:300])
				if "session" in scenarios:
					try:
						r = run(fmt, test, recs)
						total = r["dataset_open_s"] + sum(x["open_s"] + x["read_s"] for x in r["recordings"])
						row(fmt, "session", test=test, rep=rep, n_recordings=len(recs), dataset_open_s=r["dataset_open_s"],
							recordings=r["recordings"], total_s=total, note=f"{len(recs)} recordings")
					except Exception as exc:  # noqa: BLE001
						row(fmt, "session", test=test, rep=rep, error=str(exc)[:300])
	out.close()
	return 0


if __name__ == "__main__":
	sys.exit(main())
