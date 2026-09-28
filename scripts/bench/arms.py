"""One adapter per comparator arm, behind a single interface the runner drives
identically regardless of which storage format is underneath.

Every arm turns the same BIDS source into its own format (`convert`) and opens
it for reading (`open`), producing an `ArmStore` that answers the five
workloads (W1-W5, see runner.py) plus a `recordings()` listing used to pick
targets and report each arm's own chunk shape. Recording ids are the BIDS
entity path joined with "/" (e.g. "sub-001/ses-1/ieeg/task-Rest_run-1"),
identical across every arm, so the runner can point two arms at "the same
recording" and compare their answers.

Arms are written idiomatically for their own format -- an author-written weak
baseline is a standard reviewer objection, so NWB uses pynwb's own electrode
table, HDF5 uses plain h5py, and neither is bent to imitate neurozarr's
layout.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class RecInfo:
	"""What the runner needs to know about one recording, independent of arm."""
	id: str
	entities: dict[str, str]
	n_channels: int
	n_samples: int
	sfreq: float
	chunk_shape: tuple[int, ...] | None  # None if the format has no chunking concept


class ArmStore(Protocol):
	def recordings(self) -> list[RecInfo]: ...
	def full_read(self, rec_id: str) -> np.ndarray: ...
	def window_read(self, rec_id: str, start: int, n: int) -> np.ndarray: ...
	def channel_read(self, rec_id: str, channel: int) -> np.ndarray: ...
	def search(self, **entities: str) -> list[str]: ...
	def sequential_scan(self, rec_id: str, block_samples: int) -> Iterator[np.ndarray]: ...
	def close(self) -> None: ...


class Arm(Protocol):
	name: str
	def convert(self, source: Path, dest: Path) -> None: ...
	def open(self, dest: Path) -> ArmStore: ...


def _rec_id(entities: Any) -> str:
	return "/".join(entities.path())


# ---------------------------------------------------------------------------
# raw: the "do nothing" control -- no conversion, mne reads the source EDFs
# directly with preload=False, exactly as BidsReader does for discovery.
# ---------------------------------------------------------------------------

class RawArm:
	name = "raw"

	def convert(self, source: Path, dest: Path) -> None:
		# Writes nothing -- a symlink costs nothing and keeps open(dest)
		# uniform across arms (it always reads from `dest`, never `source`).
		if dest.is_symlink() or dest.exists():
			dest.unlink() if dest.is_symlink() else shutil.rmtree(dest)
		dest.parent.mkdir(parents=True, exist_ok=True)
		dest.symlink_to(Path(source).resolve())

	def open(self, dest: Path) -> "RawStore":
		return RawStore(dest)


class RawStore:
	def __init__(self, source: Path):
		from neurozarr.readers import BidsReader
		from neurozarr.items import Recording

		self._raws: dict[str, Any] = {}
		self._info: dict[str, RecInfo] = {}
		for item in BidsReader(source).read():
			if not isinstance(item, Recording):
				continue
			rec_id = _rec_id(item.entities)
			self._raws[rec_id] = item.raw
			self._info[rec_id] = RecInfo(
				id=rec_id, entities=dict(item.entities.extra),
				n_channels=len(item.raw.ch_names), n_samples=int(item.raw.n_times),
				sfreq=float(item.raw.info["sfreq"]), chunk_shape=None,
			)

	def recordings(self) -> list[RecInfo]:
		return list(self._info.values())

	def full_read(self, rec_id: str) -> np.ndarray:
		return self._raws[rec_id].get_data()

	def window_read(self, rec_id: str, start: int, n: int) -> np.ndarray:
		return self._raws[rec_id].get_data(start=start, stop=start + n)

	def channel_read(self, rec_id: str, channel: int) -> np.ndarray:
		return self._raws[rec_id].get_data(picks=[channel])[0]

	def search(self, **entities: str) -> list[str]:
		return [rid for rid, info in self._info.items()
				if all(str(info.entities.get(k)) == str(v) for k, v in entities.items())]

	def sequential_scan(self, rec_id: str, block_samples: int) -> Iterator[np.ndarray]:
		raw = self._raws[rec_id]
		n = int(raw.n_times)
		for start in range(0, n, block_samples):
			yield raw.get_data(start=start, stop=min(n, start + block_samples))

	def close(self) -> None:
		pass


# ---------------------------------------------------------------------------
# neurozarr: the system under test.
# ---------------------------------------------------------------------------

class NeurozarrArm:
	name = "neurozarr"

	def convert(self, source: Path, dest: Path) -> None:
		from neurozarr import BidsReader, Repo

		repo = Repo.create(dest)
		repo.ingest(BidsReader(source))
		repo.save("bench")

	def open(self, dest: Path) -> "NeurozarrStore":
		return NeurozarrStore(dest)


class NeurozarrStore:
	def __init__(self, dest: Path):
		from neurozarr import Repo

		self._repo = Repo.open(dest)
		self._recs: dict[str, Any] = {}
		for sub_id in self._repo.subjects():
			for rec in self._repo.subject(sub_id).recordings():
				self._recs[rec.path] = rec

	def recordings(self) -> list[RecInfo]:
		return [
			RecInfo(id=path, entities=dict(rec.entities), n_channels=rec.shape[0],
					n_samples=rec.shape[-1], sfreq=float(rec.sfreq or 0.0), chunk_shape=rec.array.chunks)
			for path, rec in self._recs.items()
		]

	def full_read(self, rec_id: str) -> np.ndarray:
		values, _ = self._recs[rec_id].data()
		return values

	def window_read(self, rec_id: str, start: int, n: int) -> np.ndarray:
		values, _ = self._recs[rec_id].data(start=start, stop=start + n)
		return values

	def channel_read(self, rec_id: str, channel: int) -> np.ndarray:
		values, _ = self._recs[rec_id].data(picks=[channel])
		return values[0]

	def search(self, **entities: str) -> list[str]:
		return [rec.path for rec in self._repo.find(**entities)]

	def sequential_scan(self, rec_id: str, block_samples: int) -> Iterator[np.ndarray]:
		rec = self._recs[rec_id]
		n = rec.shape[-1]
		for start in range(0, n, block_samples):
			values, _ = rec.data(start=start, stop=min(n, start + block_samples))
			yield values

	def close(self) -> None:
		pass


# ---------------------------------------------------------------------------
# plain_zarr: same Writer, same chunk shape, same default codec as neurozarr
# -- just no Icechunk session wrapping it. Isolates what versioning costs.
# ---------------------------------------------------------------------------

class _FakeSession:
	"""Writer only ever touches `session.store` -- this is the entire adapter
	needed to point it at a plain zarr store instead of an icechunk one."""
	def __init__(self, store: Any):
		self.store = store


class PlainZarrArm:
	name = "plain_zarr"

	def convert(self, source: Path, dest: Path) -> None:
		import zarr
		from neurozarr import BidsReader, CodecConfig, Writer
		from neurozarr.items import Recording, Table

		dest = Path(dest)
		dest.mkdir(parents=True, exist_ok=True)
		codec = CodecConfig()
		writers: dict[str, Writer] = {}
		for item in BidsReader(source).read():
			if not isinstance(item, (Recording, Table)):
				continue
			sub_id = item.entities.sub
			if sub_id not in writers:
				store = zarr.storage.LocalStore(str(dest / sub_id))
				writers[sub_id] = Writer(_FakeSession(store), codec=codec)  # type: ignore[arg-type]
			writer = writers[sub_id]
			if isinstance(item, Recording):
				writer.add_recording(item)
			else:
				writer.add_table(item)

	def open(self, dest: Path) -> "PlainZarrStore":
		return PlainZarrStore(dest)


class PlainZarrStore:
	def __init__(self, dest: Path):
		import zarr
		from neurozarr.read import RecordingView, views_in

		self._recs: dict[str, Any] = {}
		dest = Path(dest)
		for sub_dir in sorted(p for p in dest.iterdir() if p.is_dir()):
			root = zarr.open_group(store=zarr.storage.LocalStore(str(sub_dir)), mode="r")
			for view in views_in(root, base_path=sub_dir.name):
				if isinstance(view, RecordingView):
					self._recs[view.path] = view

	def recordings(self) -> list[RecInfo]:
		return [
			RecInfo(id=path, entities=dict(rec.entities), n_channels=rec.shape[0],
					n_samples=rec.shape[-1], sfreq=float(rec.sfreq or 0.0), chunk_shape=rec.array.chunks)
			for path, rec in self._recs.items()
		]

	def full_read(self, rec_id: str) -> np.ndarray:
		values, _ = self._recs[rec_id].data()
		return values

	def window_read(self, rec_id: str, start: int, n: int) -> np.ndarray:
		values, _ = self._recs[rec_id].data(start=start, stop=start + n)
		return values

	def channel_read(self, rec_id: str, channel: int) -> np.ndarray:
		values, _ = self._recs[rec_id].data(picks=[channel])
		return values[0]

	def search(self, **entities: str) -> list[str]:
		# No catalog here (that's Icechunk's _dataset manifest) -- a plain,
		# unaided walk-and-filter, same as neurozarr's own find() does today
		# once the catalog is exhausted (see Phase 3 of the plan).
		return [path for path, rec in self._recs.items()
				if all(str(rec.entities.get(k)) == str(v) for k, v in entities.items())]

	def sequential_scan(self, rec_id: str, block_samples: int) -> Iterator[np.ndarray]:
		rec = self._recs[rec_id]
		n = rec.shape[-1]
		for start in range(0, n, block_samples):
			values, _ = rec.data(start=start, stop=min(n, start + block_samples))
			yield values

	def close(self) -> None:
		pass


# ---------------------------------------------------------------------------
# hdf5_mat73: plain h5py. A MATLAB v7.3 .mat file IS HDF5 with MATLAB's own
# attribute conventions on top -- for storage/timing characteristics (what
# this harness measures) those conventions are irrelevant, so this arm
# doesn't add them or claim MATLAB can load its output.
# ---------------------------------------------------------------------------

class Hdf5Arm:
	name = "hdf5_mat73"

	def convert(self, source: Path, dest: Path) -> None:
		import h5py
		from neurozarr import BidsReader, util
		from neurozarr.items import Recording

		dest = Path(dest)
		dest.mkdir(parents=True, exist_ok=True)
		with h5py.File(dest / "store.h5", "w") as f:
			for item in BidsReader(source).read():
				if not isinstance(item, Recording):
					continue
				rec_id = _rec_id(item.entities)
				data = item.raw.get_data().astype("float32")
				chunks = util.chunk_shape(data.shape, data.itemsize, 8 * 1024 * 1024, 65536)
				grp = f.require_group(rec_id)
				ds = grp.create_dataset("data", data=data, chunks=chunks,
										 compression="gzip", compression_opts=4, shuffle=True)
				ds.attrs["sfreq"] = float(item.raw.info["sfreq"])
				ds.attrs["ch_names"] = list(item.raw.ch_names)
				for key, value in item.entities.extra.items():
					grp.attrs[key] = str(value)

	def open(self, dest: Path) -> "Hdf5Store":
		return Hdf5Store(dest)


class Hdf5Store:
	def __init__(self, dest: Path):
		import h5py

		self._file = h5py.File(Path(dest) / "store.h5", "r")
		self._recs: dict[str, Any] = {}

		def visit(name: str, obj: Any) -> None:
			if isinstance(obj, h5py.Dataset) and name.endswith("/data"):
				self._recs[name[: -len("/data")]] = obj

		self._file.visititems(visit)

	def recordings(self) -> list[RecInfo]:
		out = []
		for rec_id, ds in self._recs.items():
			grp = ds.parent
			entities = {k: v for k, v in grp.attrs.items()}
			out.append(RecInfo(id=rec_id, entities=entities, n_channels=ds.shape[0],
								n_samples=ds.shape[-1], sfreq=float(ds.attrs.get("sfreq", 0.0)),
								chunk_shape=ds.chunks))
		return out

	def full_read(self, rec_id: str) -> np.ndarray:
		return self._recs[rec_id][:, :]

	def window_read(self, rec_id: str, start: int, n: int) -> np.ndarray:
		return self._recs[rec_id][:, start:start + n]

	def channel_read(self, rec_id: str, channel: int) -> np.ndarray:
		return self._recs[rec_id][channel, :]

	def search(self, **entities: str) -> list[str]:
		return [rid for rid, ds in self._recs.items()
				if all(str(ds.parent.attrs.get(k)) == str(v) for k, v in entities.items())]

	def sequential_scan(self, rec_id: str, block_samples: int) -> Iterator[np.ndarray]:
		ds = self._recs[rec_id]
		n = ds.shape[-1]
		for start in range(0, n, block_samples):
			yield ds[:, start:min(n, start + block_samples)]

	def close(self) -> None:
		self._file.close()


# ---------------------------------------------------------------------------
# NWB (HDF5- and Zarr-backed): one NWBFile per (subject, session), matching
# NWB's own convention -- an NWBFile represents one session, not a dataset.
# Data is stored time-major (n_samples, n_channels), NWB's own convention,
# not bent to match neurozarr's channel-major layout: W3 (single-channel
# trace) is therefore a strided read here, not a contiguous one, which is a
# real, reportable structural difference, not a bug in this arm.
# ---------------------------------------------------------------------------

def _nwb_group_key(entities: Any) -> tuple[str, str]:
	return entities.sub, entities.ses or "ses-none"


def _build_nwb_files(source: Path) -> dict[tuple[str, str], Any]:
	"""One in-memory NWBFile per (subject, session), with every recording in
	that session added as its own ElectricalSeries. Electrodes are deduped by
	channel name within a file -- a session's different tasks/runs often
	share the same physical channels, and NWB's electrode table is one table
	per file, not per series."""
	from datetime import datetime, timezone

	from pynwb import NWBFile
	from pynwb.ecephys import ElectricalSeries
	from hdmf.backends.hdf5.h5_utils import H5DataIO

	from neurozarr import BidsReader, util
	from neurozarr.items import Recording

	files: dict[tuple[str, str], Any] = {}
	electrode_rows: dict[tuple[str, str], dict[str, int]] = {}

	for item in BidsReader(source).read():
		if not isinstance(item, Recording):
			continue
		key = _nwb_group_key(item.entities)
		if key not in files:
			nwbfile = NWBFile(session_description=f"{key[0]} {key[1]}", identifier="/".join(key),
							   session_start_time=datetime.now(timezone.utc))
			device = nwbfile.create_device(name="device")
			nwbfile.create_electrode_group(name="group", description="all channels", location="unknown", device=device)
			files[key] = nwbfile
			electrode_rows[key] = {}

		nwbfile = files[key]
		rows = electrode_rows[key]
		region_indices = []
		for ch_name in item.raw.ch_names:
			if ch_name not in rows:
				rows[ch_name] = len(rows)
				nwbfile.add_electrode(id=rows[ch_name], location="unknown", group_name="group",
									   group=nwbfile.electrode_groups["group"])
			region_indices.append(rows[ch_name])
		region = nwbfile.create_electrode_table_region(region=region_indices, description=item.entities.group_name() or "data")

		data = item.raw.get_data().astype("float32").T  # NWB wants (n_samples, n_channels)
		chunks = util.chunk_shape(data.shape[::-1], data.itemsize, 8 * 1024 * 1024, 65536)[::-1]
		wrapped = H5DataIO(data=data, chunks=chunks, compression="gzip", compression_opts=4, shuffle=True)
		name = item.entities.group_name() or "data"
		es = ElectricalSeries(name=name, data=wrapped, electrodes=region,
							   rate=float(item.raw.info["sfreq"]), starting_time=0.0)
		nwbfile.add_acquisition(es)

	return files


class NwbHdf5Arm:
	name = "nwb_hdf5"

	def convert(self, source: Path, dest: Path) -> None:
		from pynwb import NWBHDF5IO

		dest = Path(dest)
		dest.mkdir(parents=True, exist_ok=True)
		for (sub_id, ses_id), nwbfile in _build_nwb_files(source).items():
			path = dest / f"{sub_id}_{ses_id}.nwb"
			with NWBHDF5IO(str(path), mode="w") as io:
				io.write(nwbfile)

	def open(self, dest: Path) -> "NwbStore":
		from pynwb import NWBHDF5IO
		return NwbStore(dest, NWBHDF5IO, "*.nwb")


class NwbZarrArm:
	name = "nwb_zarr"

	def convert(self, source: Path, dest: Path) -> None:
		from hdmf_zarr import NWBZarrIO

		dest = Path(dest)
		dest.mkdir(parents=True, exist_ok=True)
		for (sub_id, ses_id), nwbfile in _build_nwb_files(source).items():
			path = dest / f"{sub_id}_{ses_id}.nwb.zarr"
			with NWBZarrIO(str(path), mode="w") as io:
				io.write(nwbfile)

	def open(self, dest: Path) -> "NwbStore":
		from hdmf_zarr import NWBZarrIO
		return NwbStore(dest, NWBZarrIO, "*.nwb.zarr")


class NwbStore:
	"""Shared reader for both NWB IO backends -- the file layout and lookup
	logic don't depend on whether pynwb or hdmf-zarr opens the file."""

	def __init__(self, dest: Path, io_cls: Any, glob: str):
		from neurozarr.entities import parse_entities

		self._ios: list[Any] = []
		self._recs: dict[str, Any] = {}
		self._info: dict[str, RecInfo] = {}
		for path in sorted(Path(dest).glob(glob)):
			io = io_cls(str(path), mode="r")
			nwbfile = io.read()
			self._ios.append(io)
			parts = path.name.removesuffix(".nwb.zarr").removesuffix(".nwb").split("_", 1)
			sub_id, ses_id = parts[0], parts[1] if len(parts) > 1 else "ses-none"
			for name, series in nwbfile.acquisition.items():
				entities = parse_entities(name)
				rec_id = "/".join((sub_id, ses_id, "ieeg", name)) if ses_id != "ses-none" \
					else "/".join((sub_id, "ieeg", name))
				self._recs[rec_id] = series
				self._info[rec_id] = RecInfo(
					id=rec_id, entities=entities, n_channels=series.data.shape[1],
					n_samples=series.data.shape[0], sfreq=float(series.rate or 0.0),
					chunk_shape=getattr(series.data, "chunks", None),
				)

	def recordings(self) -> list[RecInfo]:
		return list(self._info.values())

	def full_read(self, rec_id: str) -> np.ndarray:
		return self._recs[rec_id].data[:, :].T  # back to channel-major for the runner's comparisons

	def window_read(self, rec_id: str, start: int, n: int) -> np.ndarray:
		return self._recs[rec_id].data[start:start + n, :].T

	def channel_read(self, rec_id: str, channel: int) -> np.ndarray:
		return self._recs[rec_id].data[:, channel]

	def search(self, **entities: str) -> list[str]:
		return [rid for rid, info in self._info.items()
				if all(str(info.entities.get(k)) == str(v) for k, v in entities.items())]

	def sequential_scan(self, rec_id: str, block_samples: int) -> Iterator[np.ndarray]:
		data = self._recs[rec_id].data
		n = data.shape[0]
		for start in range(0, n, block_samples):
			yield data[start:min(n, start + block_samples), :].T

	def close(self) -> None:
		for io in self._ios:
			io.close()


ARMS: dict[str, Any] = {
	"raw": RawArm(),
	"neurozarr": NeurozarrArm(),
	"plain_zarr": PlainZarrArm(),
	"hdf5_mat73": Hdf5Arm(),
	"nwb_hdf5": NwbHdf5Arm(),
	"nwb_zarr": NwbZarrArm(),
}
