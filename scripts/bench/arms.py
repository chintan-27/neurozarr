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
# directly with preload=False, exactly as the auto-detected reader
# (BidsReader for a directory, ManifestReader for a .csv) does for discovery.
# ---------------------------------------------------------------------------

class RawArm:
	name = "raw"

	def convert(self, source: Path, dest: Path) -> None:
		# Writes nothing -- a symlink costs nothing and keeps open(dest)
		# uniform across arms (it always reads from `dest`, never `source`).
		# dest holds a same-named symlink rather than being the symlink
		# itself: reader_for() picks BidsReader vs ManifestReader by the
		# path's own suffix (a directory, or a bare extensionless `dest`,
		# both read as "no suffix" -> BidsReader), so a manifest .csv source
		# needs its real filename to survive the symlink for auto-detection
		# to still work.
		if dest.is_symlink() or dest.exists():
			dest.unlink() if dest.is_symlink() else shutil.rmtree(dest)
		dest.mkdir(parents=True, exist_ok=True)
		(dest / Path(source).name).symlink_to(Path(source).resolve())

	def open(self, dest: Path) -> "RawStore":
		return RawStore(dest)


class RawStore:
	"""Holds only a file path + cheap metadata per recording, not a live mne
	Raw handle for every one -- a dataset with hundreds of subjects (measured:
	185, real OOM at 32GB) means hundreds of simultaneously-open Raw objects
	if discovery holds onto what BidsReader hands it. mne.io.read_raw with
	preload=False doesn't load samples, but each Raw's own info/annotation
	structures still add up across that many live objects. Only the one
	recording actually being read is ever open at a time."""

	def __init__(self, dest: Path):
		from neurozarr.readers import reader_for
		from neurozarr.items import Recording

		source = next(Path(dest).iterdir())
		self._paths: dict[str, Path] = {}
		self._info: dict[str, RecInfo] = {}
		self._cache: tuple[str, Any] | None = None
		for item in reader_for(source).read():
			if not isinstance(item, Recording):
				continue
			rec_id = _rec_id(item.entities)
			self._paths[rec_id] = Path(item.raw.filenames[0])
			self._info[rec_id] = RecInfo(
				id=rec_id, entities=dict(item.entities.extra),
				n_channels=len(item.raw.ch_names), n_samples=int(item.raw.n_times),
				sfreq=float(item.raw.info["sfreq"]), chunk_shape=None,
			)
			# item.raw is now unreferenced and can be garbage-collected before
			# the next iteration opens the next file.

	def _raw(self, rec_id: str) -> Any:
		import mne
		if self._cache is not None and self._cache[0] == rec_id:
			return self._cache[1]
		raw = mne.io.read_raw(self._paths[rec_id], preload=False, verbose=False)
		self._cache = (rec_id, raw)
		return raw

	def recordings(self) -> list[RecInfo]:
		return list(self._info.values())

	def full_read(self, rec_id: str) -> np.ndarray:
		return self._raw(rec_id).get_data()

	def window_read(self, rec_id: str, start: int, n: int) -> np.ndarray:
		return self._raw(rec_id).get_data(start=start, stop=start + n)

	def channel_read(self, rec_id: str, channel: int) -> np.ndarray:
		return self._raw(rec_id).get_data(picks=[channel])[0]

	def search(self, **entities: str) -> list[str]:
		return [rid for rid, info in self._info.items()
				if all(str(info.entities.get(k)) == str(v) for k, v in entities.items())]

	def sequential_scan(self, rec_id: str, block_samples: int) -> Iterator[np.ndarray]:
		raw = self._raw(rec_id)
		n = int(raw.n_times)
		for start in range(0, n, block_samples):
			yield raw.get_data(start=start, stop=min(n, start + block_samples))

	def close(self) -> None:
		self._cache = None


# ---------------------------------------------------------------------------
# neurozarr: the system under test.
# ---------------------------------------------------------------------------

class NeurozarrArm:
	name = "neurozarr"

	def convert(self, source: Path, dest: Path) -> None:
		from neurozarr import Repo
		from neurozarr.readers import reader_for

		# A benchmark run always wants a fresh store, never a resume/append --
		# Repo.init() refuses an existing store, which a prior run killed
		# mid-conversion (e.g. by a timeout) can leave behind.
		if dest.exists():
			shutil.rmtree(dest)
		repo = Repo.init(dest)
		repo.add(reader_for(source))
		repo.commit("bench")

	def open(self, dest: Path) -> "NeurozarrStore":
		return NeurozarrStore(dest)


class NeurozarrStore:
	"""Holds only lightweight RecInfo per recording, not a live RecordingView
	for every one, and explicitly drops each subject's opened icechunk
	session/root before moving to the next -- Repo caches every root it opens
	for its own lifetime (deliberate, see repo.py's _invalidate_read_cache:
	it's what makes repeated find() calls fast), which is fine for a handful
	of subjects but pins real memory across hundreds (measured: OOM at 32GB
	on a 185-subject store, discovery alone, after conversion had already
	succeeded). Reaches into Repo's cache dicts directly rather than adding a
	bound-the-cache-size knob to the library for a harness-only need."""

	def __init__(self, dest: Path):
		from neurozarr import Repo

		self._repo = Repo.open(dest)
		self._sub_of: dict[str, str] = {}
		self._info: dict[str, RecInfo] = {}
		self._cache: tuple[str, Any] | None = None
		for sub_id in self._repo.subjects():
			for rec in self._repo.subject(sub_id).recordings():
				self._sub_of[rec.path] = sub_id
				self._info[rec.path] = RecInfo(
					id=rec.path, entities=dict(rec.entities), n_channels=rec.shape[0],
					n_samples=rec.shape[-1], sfreq=float(rec.sfreq or 0.0), chunk_shape=rec.array.chunks,
				)
			self._drop_cached_subjects(keep=None)

	def _drop_cached_subjects(self, keep: str | None) -> None:
		self._repo._repos = {k: v for k, v in self._repo._repos.items() if k == keep}
		self._repo._root_cache = {k: v for k, v in self._repo._root_cache.items() if k[0] == keep}

	def _view(self, rec_id: str) -> Any:
		if self._cache is not None and self._cache[0] == rec_id:
			return self._cache[1]
		sub_id = self._sub_of[rec_id]
		view = next(v for v in self._repo.subject(sub_id).recordings() if v.path == rec_id)
		self._drop_cached_subjects(keep=sub_id)
		self._cache = (rec_id, view)
		return view

	def recordings(self) -> list[RecInfo]:
		return list(self._info.values())

	def full_read(self, rec_id: str) -> np.ndarray:
		values, _ = self._view(rec_id).data()
		return values

	def window_read(self, rec_id: str, start: int, n: int) -> np.ndarray:
		values, _ = self._view(rec_id).data(start=start, stop=start + n)
		return values

	def channel_read(self, rec_id: str, channel: int) -> np.ndarray:
		values, _ = self._view(rec_id).data(picks=[channel])
		return values[0]

	def search(self, **entities: str) -> list[str]:
		return [rec.path for rec in self._repo.find(**entities)]

	def sequential_scan(self, rec_id: str, block_samples: int) -> Iterator[np.ndarray]:
		rec = self._view(rec_id)
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
		from neurozarr import CodecConfig, Writer
		from neurozarr.items import Recording, Table
		from neurozarr.readers import reader_for

		dest = Path(dest)
		# Same reasoning as NeurozarrArm.convert(): a killed prior run can
		# leave some subjects converted and others not, and add_recording's
		# default existing="error" would then raise on the ones that exist.
		if dest.exists():
			shutil.rmtree(dest)
		dest.mkdir(parents=True, exist_ok=True)
		codec = CodecConfig()
		writers: dict[str, Writer] = {}
		for item in reader_for(source).read():
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
	"""chunk_target_bytes/max_chunk_samples let a fairness sweep try HDF5 at
	several chunk shapes instead of judging it on one arbitrary default --
	same knobs neurozarr's own util.chunk_shape() takes. packed=True gives
	HDF5 the same int16+per-channel-calibration domain assumption neurozarr
	gets by default, instead of comparing packed-vs-unpacked as if that were
	a format difference."""
	name = "hdf5_mat73"

	def __init__(self, chunk_target_bytes: int = 8 * 1024 * 1024, max_chunk_samples: int = 65536,
				 packed: bool = False):
		self.chunk_target_bytes = chunk_target_bytes
		self.max_chunk_samples = max_chunk_samples
		self.packed = packed

	def convert(self, source: Path, dest: Path) -> None:
		import h5py
		from neurozarr import util
		from neurozarr.items import Recording
		from neurozarr.readers import reader_for

		dest = Path(dest)
		dest.mkdir(parents=True, exist_ok=True)
		with h5py.File(dest / "store.h5", "w") as f:
			for item in reader_for(source).read():
				if not isinstance(item, Recording):
					continue
				rec_id = _rec_id(item.entities)
				shape = (len(item.raw.ch_names), int(item.raw.n_times))

				scale = offset = None
				dtype = np.dtype("float32")
				if self.packed:
					extras = getattr(item.raw, "_raw_extras", None)
					if extras and "cal" in extras[0]:
						cal = np.array(extras[0]["cal"])
						units = np.array(extras[0]["units"])
						offsets = np.array(extras[0]["offsets"])
						scale, offset = cal * units, offsets * units
						if scale.shape == (shape[0],) and offset.shape == scale.shape and np.all(scale != 0):
							dtype = np.dtype("int16")
						else:
							scale = offset = None  # fall back to float32, same as neurozarr's own writer would

				chunks = util.chunk_shape(shape, dtype.itemsize, self.chunk_target_bytes, self.max_chunk_samples)
				grp = f.require_group(rec_id)
				ds = grp.create_dataset("data", shape=shape, dtype=dtype, chunks=chunks,
										 compression="gzip", compression_opts=4, shuffle=True)
				# Stream chunk-by-chunk like neurozarr's own writer does --
				# item.raw.get_data() with no start/stop loads the whole
				# recording at once, which is what OOM-killed this arm on a
				# real 165GB dataset (a single large recording, not the
				# "many subjects" shape the other OOM fix addressed).
				for start in range(0, shape[-1], chunks[-1]):
					stop = min(shape[-1], start + chunks[-1])
					physical = item.raw.get_data(start=start, stop=stop)
					if scale is not None:
						encoded = np.round((physical - offset[:, None]) / scale[:, None])
						ds[:, start:stop] = np.clip(encoded, -32768, 32767).astype("int16")
					else:
						ds[:, start:stop] = physical.astype("float32")
				ds.attrs["sfreq"] = float(item.raw.info["sfreq"])
				ds.attrs["ch_names"] = list(item.raw.ch_names)
				if scale is not None:
					ds.attrs["data_scale"] = scale.tolist()
					ds.attrs["data_offset"] = offset.tolist()
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

	def _decode(self, ds: Any, raw: np.ndarray, channels: "slice | int | None" = None) -> np.ndarray:
		"""Packed (int16) datasets store codes, not physical values -- decode
		the same way neurozarr's own reader does, so a packed-vs-float32
		comparison is still comparing the same physical quantity."""
		if "data_scale" not in ds.attrs:
			return raw
		scale = np.asarray(ds.attrs["data_scale"])[channels] if channels is not None else np.asarray(ds.attrs["data_scale"])
		offset = np.asarray(ds.attrs["data_offset"])[channels] if channels is not None else np.asarray(ds.attrs["data_offset"])
		if raw.ndim == 1:  # single-channel read
			return raw.astype("float64") * scale + offset
		return raw.astype("float64") * scale[:, None] + offset[:, None]

	def full_read(self, rec_id: str) -> np.ndarray:
		ds = self._recs[rec_id]
		return self._decode(ds, ds[:, :])

	def window_read(self, rec_id: str, start: int, n: int) -> np.ndarray:
		ds = self._recs[rec_id]
		return self._decode(ds, ds[:, start:start + n])

	def channel_read(self, rec_id: str, channel: int) -> np.ndarray:
		ds = self._recs[rec_id]
		return self._decode(ds, ds[channel, :], channels=channel)

	def search(self, **entities: str) -> list[str]:
		return [rid for rid, ds in self._recs.items()
				if all(str(ds.parent.attrs.get(k)) == str(v) for k, v in entities.items())]

	def sequential_scan(self, rec_id: str, block_samples: int) -> Iterator[np.ndarray]:
		ds = self._recs[rec_id]
		n = ds.shape[-1]
		for start in range(0, n, block_samples):
			yield self._decode(ds, ds[:, start:min(n, start + block_samples)])

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


class _RawChunkIterator:
	"""Streams an mne Raw's samples to NWB chunk-by-chunk instead of loading
	the whole recording at once -- the same eager-get_data() OOM pattern
	already fixed for hdf5_mat73 (arms.py's own Hdf5Arm), just hit here via
	NWB's own chunked-write primitive. A first attempt at this passed
	mismatched chunk_shape/buffer_shape dimensionality and hit hdmf's own
	broadcast error; this one lets GenericDataChunkIterator compute its own
	defaults from maxshape/dtype instead of guessing shapes by hand, and is
	round-trip-verified (see demo_nwb_streaming_check.py) before being
	trusted against a real multi-hour conversion."""

	def __new__(cls, raw: Any, **kwargs: Any) -> Any:
		from hdmf.data_utils import GenericDataChunkIterator

		class _Impl(GenericDataChunkIterator):
			def __init__(self, raw: Any, **kw: Any) -> None:
				self._raw = raw
				super().__init__(**kw)

			def _get_data(self, selection: tuple) -> np.ndarray:
				time_sel, chan_sel = selection
				start = time_sel.start or 0
				stop = time_sel.stop if time_sel.stop is not None else int(self._raw.n_times)
				data = self._raw.get_data(start=start, stop=stop).T.astype("float32")
				return data[:, chan_sel]

			def _get_dtype(self) -> np.dtype:
				return np.dtype("float32")

			def _get_maxshape(self) -> tuple[int, int]:
				return (int(self._raw.n_times), len(self._raw.ch_names))

		return _Impl(raw, **kwargs)


def _build_nwb_files(source: Path, backend: str) -> Iterator[tuple[tuple[str, str], Any]]:
	"""Yields one (key, NWBFile) per (subject, session) as soon as that
	session's items are exhausted, instead of returning a dict of every
	session in the dataset at once -- holding all of them (each session's own
	real file handles, electrode tables, and chunk-iterator closures) until
	the very end OOM-killed a real 104GB/138-subject run even though each
	individual series streams its sample data lazily; the fix is the same
	"process and release one at a time" shape already used for RawStore/
	NeurozarrStore elsewhere in this harness. Relies on the source reader
	grouping a session's items together in emission order -- true for
	BidsReader's own sub/ses/datatype walk, which is the only reader this
	harness points NWB arms at.

	Electrodes are deduped by channel name within a file -- a session's
	different tasks/runs often share the same physical channels, and NWB's
	electrode table is one table per file, not per series.

	backend picks the DataIO wrapper: H5DataIO and hdmf_zarr's ZarrDataIO
	look like drop-in alternatives (same constructor shape) but aren't --
	H5DataIO assumes its wrapped data supports direct [item] indexing once
	hdmf-zarr's writer gets hold of it, which a chunked iterator doesn't
	("'_Impl' object is not subscriptable", hit by actually trying this
	against real data rather than assuming the HDF5 path's success implied
	the Zarr path would behave the same way)."""
	from datetime import datetime, timezone

	from pynwb import NWBFile
	from pynwb.ecephys import ElectricalSeries
	if backend == "hdf5":
		from hdmf.backends.hdf5.h5_utils import H5DataIO as DataIOCls
		dataio_kwargs: dict[str, Any] = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
	elif backend == "zarr":
		from hdmf_zarr.utils import ZarrDataIO as DataIOCls  # type: ignore[no-redef]
		dataio_kwargs = {"compressors": True}  # zarr's own default codec, not forced to imitate HDF5's gzip
	else:
		raise ValueError(f"unknown NWB backend {backend!r}")

	from neurozarr import util
	from neurozarr.items import Recording
	from neurozarr.readers import reader_for

	open_key: tuple[str, str] | None = None
	nwbfile: Any = None
	rows: dict[str, int] = {}

	for item in reader_for(source).read():
		if not isinstance(item, Recording):
			continue
		key = _nwb_group_key(item.entities)
		if key != open_key:
			if open_key is not None:
				yield open_key, nwbfile
			nwbfile = NWBFile(session_description=f"{key[0]} {key[1]}", identifier="/".join(key),
							   session_start_time=datetime.now(timezone.utc))
			device = nwbfile.create_device(name="device")
			nwbfile.create_electrode_group(name="group", description="all channels", location="unknown", device=device)
			rows = {}
			open_key = key

		region_indices = []
		for ch_name in item.raw.ch_names:
			if ch_name not in rows:
				rows[ch_name] = len(rows)
				nwbfile.add_electrode(id=rows[ch_name], location="unknown", group_name="group",
									   group=nwbfile.electrode_groups["group"])
			region_indices.append(rows[ch_name])
		region = nwbfile.create_electrode_table_region(region=region_indices, description=item.entities.group_name() or "data")

		# NWB wants (n_samples, n_channels); chunk shape computed the same way
		# as every other arm's, then reversed to match that axis order.
		shape = (len(item.raw.ch_names), int(item.raw.n_times))
		chunks = util.chunk_shape(shape, np.dtype("float32").itemsize, 8 * 1024 * 1024, 65536)[::-1]
		iterator = _RawChunkIterator(item.raw, chunk_shape=chunks)
		wrapped = DataIOCls(data=iterator, **dataio_kwargs)
		name = item.entities.group_name() or "data"
		es = ElectricalSeries(name=name, data=wrapped, electrodes=region,
							   rate=float(item.raw.info["sfreq"]), starting_time=0.0)
		nwbfile.add_acquisition(es)

	if open_key is not None:
		yield open_key, nwbfile


class NwbHdf5Arm:
	name = "nwb_hdf5"

	def convert(self, source: Path, dest: Path) -> None:
		from pynwb import NWBHDF5IO

		dest = Path(dest)
		dest.mkdir(parents=True, exist_ok=True)
		for (sub_id, ses_id), nwbfile in _build_nwb_files(source, backend="hdf5"):
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
		for (sub_id, ses_id), nwbfile in _build_nwb_files(source, backend="zarr"):
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
