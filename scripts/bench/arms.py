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

import json
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
	# Drop whatever this store holds in memory for one recording (open handles, decoded
	# chunks), so the next read opens it as a first read would; files stay in the OS cache.
	def forget(self, rec_id: str) -> None: ...


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

	def forget(self, rec_id: str) -> None:
		import mne
		# a new handle, opened here (outside the timed read) like every other format's
		self._cache = (rec_id, mne.io.read_raw(self._paths[rec_id], preload=False, verbose=False))


# ---------------------------------------------------------------------------
# neurozarr: the system under test.
# ---------------------------------------------------------------------------

class _SignalsOnly:
	"""A reader passing through only its recordings (each with its own sidecar
	metadata; the writer adds its annotations) -- the "signals" content level."""

	def __init__(self, reader: Any):
		self._reader = reader

	def read(self) -> Iterator[Any]:
		from neurozarr.items import Recording
		return (item for item in self._reader.read() if isinstance(item, Recording))

	def codec_plan(self, layout: Any = None) -> Any:
		return self._reader.codec_plan(layout)

	def subjects(self) -> list[str]:
		return self._reader.subjects()

	def for_subject(self, sub: str | None) -> "_SignalsOnly":
		return _SignalsOnly(self._reader.for_subject(sub))


class NeurozarrArm:
	name = "neurozarr"

	def __init__(self, dtype: str = "int16", content: str = "full", workers: int = 1):
		self.dtype, self.content, self.workers = dtype, _check_content(content), workers

	def convert(self, source: Path, dest: Path) -> None:
		from neurozarr import CodecConfig, Repo
		from neurozarr.readers import reader_for

		# A benchmark run always wants a fresh store, never a resume/append --
		# Repo.init() refuses an existing store, which a prior run killed
		# mid-conversion (e.g. by a timeout) can leave behind.
		if dest.exists():
			shutil.rmtree(dest)
		repo = Repo.init(dest, codec=CodecConfig(dtype=self.dtype))
		reader = reader_for(source)
		repo.add(reader if self.content == "full" else _SignalsOnly(reader), workers=self.workers)
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

	def forget(self, rec_id: str) -> None:
		from neurozarr.read import RecordingView
		# a new view over the same group, its array opened here (outside the timed read):
		# no decoded chunks carried over. Not looked up again -- that walks the subject.
		old = self._view(rec_id)
		fresh = RecordingView(old._group, old.entities, old.path)
		fresh.array
		self._cache = (rec_id, fresh)


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

	def __init__(self, dtype: str = "int16", content: str = "full"):
		self.dtype, self.content = dtype, _check_content(content)

	def convert(self, source: Path, dest: Path) -> None:
		import zarr
		from neurozarr import CodecConfig, Writer
		from neurozarr.items import Attrs, Recording, Table
		from neurozarr.readers import reader_for

		dest = Path(dest)
		# Same reasoning as NeurozarrArm.convert(): a killed prior run can
		# leave some subjects converted and others not, and add_recording's
		# default existing="error" would then raise on the ones that exist.
		if dest.exists():
			shutil.rmtree(dest)
		dest.mkdir(parents=True, exist_ok=True)
		codec = CodecConfig(dtype=self.dtype)
		reader = reader_for(source)
		plan = reader.codec_plan(codec)  # the same codecs neurozarr picks: this arm isolates icechunk
		writers: dict[str, Writer] = {}

		def writer_for(name: str) -> Writer:
			if name not in writers:
				store = zarr.storage.LocalStore(str(dest / name))
				writers[name] = Writer(_FakeSession(store), codec=codec)  # type: ignore[arg-type]
				writers[name].plan = plan
			return writers[name]

		for item in reader.read():
			if isinstance(item, Attrs):
				if self.content == "full":  # routed as Repo does: to its subject, else dataset-wide
					sub = next((part for part in item.path if part.startswith("sub-")), None)
					writer_for(sub or "_dataset").add_attrs(Attrs(tuple(p for p in item.path if p != sub), item.attrs))
				continue
			if isinstance(item, Table) and self.content != "full":
				continue
			if not isinstance(item, (Recording, Table)):
				continue
			writer = writer_for(item.entities.sub)
			if isinstance(item, Recording):
				writer.add_recording(item)
			else:
				writer.add_table(item)
		for writer in writers.values():
			writer._flush()  # no commit here to do it: write the nodes still batched

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

	def forget(self, rec_id: str) -> None:
		from neurozarr.read import RecordingView
		old = self._recs[rec_id]  # a new view over the same group: no decoded chunks
		self._recs[rec_id] = RecordingView(old._group, old.entities, old.path)
		self._recs[rec_id].array  # opened here, outside the timed read

	def close(self) -> None:
		pass


# ---------------------------------------------------------------------------
# hdf5_mat73: plain h5py. A MATLAB v7.3 .mat file IS HDF5 with MATLAB's own
# attribute conventions on top -- for storage/timing characteristics (what
# this harness measures) those conventions are irrelevant, so this arm
# doesn't add them or claim MATLAB can load its output.
# ---------------------------------------------------------------------------

def _int16_calibration(raw: Any) -> "tuple[np.ndarray, np.ndarray] | None":
	"""Per-channel (scale, offset) such that physical = code * scale + offset, from the
	EDF-family reader's own calibration -- the same source neurozarr's writer packs
	int16 from. None when the reader doesn't provide it (then every arm stays float32)."""
	extras = getattr(raw, "_raw_extras", None)
	if not extras or "cal" not in extras[0]:
		return None
	units = np.array(extras[0]["units"])
	scale, offset = np.array(extras[0]["cal"]) * units, np.array(extras[0]["offsets"]) * units
	if scale.shape != (len(raw.ch_names),) or offset.shape != scale.shape or not np.all(scale != 0):
		return None
	return scale, offset


def _encode_int16(physical: np.ndarray, scale: np.ndarray, offset: np.ndarray) -> np.ndarray:
	"""Channel-major physical values -> int16 codes."""
	return np.clip(np.round((physical - offset[:, None]) / scale[:, None]), -32768, 32767).astype("int16")


def _check_content(content: str) -> str:
	"""signals: recordings with their own sidecar metadata and annotations; full: also
	every sidecar table and metadata file -- what neurozarr stores by default."""
	if content not in ("signals", "full"):
		raise ValueError(f"content must be 'signals' or 'full', got {content!r}")
	return content


def _annotations(raw: Any) -> "Any | None":
	import pandas as pd
	annotations = getattr(raw, "annotations", None)
	if not annotations or len(annotations) == 0:
		return None
	return pd.DataFrame({"onset": annotations.onset, "duration": annotations.duration,
						 "trial_type": [str(d) for d in annotations.description]})


def _h5_table(parent: Any, name: str, df: Any, meta: dict[str, Any]) -> None:
	"""A table the way an HDF5 user stores one fastest: one compound dataset (a record
	array, as MATLAB tables and pandas' fixed format do) -- numbers as they are, missing
	as NaN; text as UTF-8, missing as "". Column names and sidecar metadata as attributes.
	Measured 2.4 ms per BRAVO table, against 4.1 ms for one dataset per column."""
	import pandas as pd
	missing = df.isna().to_numpy().reshape(len(df), len(df.columns))
	fields, columns = [], []
	for i, (_, series) in enumerate(df.items()):
		if pd.api.types.is_numeric_dtype(series.dtype):
			values = series.to_numpy(dtype="float64" if missing[:, i].any() else None, na_value=np.nan)
		else:
			values = np.char.encode(series.to_numpy(dtype=object, na_value="").astype(str), "utf-8")
			if values.dtype.itemsize == 0:
				values = values.astype("S1")
		fields.append((f"c{i}", values.dtype))
		columns.append(values)
	records = np.empty(len(df), dtype=fields)
	for (field, _), values in zip(fields, columns):
		records[field] = values
	dataset = parent.create_dataset(name, data=records)
	dataset.attrs["columns"] = json.dumps([str(c) for c in df.columns])
	dataset.attrs["meta"] = json.dumps(meta)


class Hdf5Arm:
	"""chunk_target_bytes/max_chunk_samples let a fairness sweep try HDF5 at
	several chunk shapes instead of judging it on one arbitrary default --
	same knobs neurozarr's own util.chunk_shape() takes. packed=True gives
	HDF5 the same int16+per-channel-calibration domain assumption neurozarr
	gets by default, instead of comparing packed-vs-unpacked as if that were
	a format difference."""
	name = "hdf5_mat73"

	def __init__(self, chunk_target_bytes: int = 8 * 1024 * 1024, max_chunk_samples: int = 65536,
				 packed: bool = False, content: str = "signals", dtype: str | None = None):
		self.chunk_target_bytes = chunk_target_bytes
		self.max_chunk_samples = max_chunk_samples
		self.dtype = dtype or ("int16" if packed else "float32")
		self.packed = self.dtype == "int16"
		self.content = _check_content(content)

	def convert(self, source: Path, dest: Path) -> None:
		import h5py
		from neurozarr import util
		from neurozarr.items import Attrs, Recording, Table
		from neurozarr.readers import reader_for

		dest = Path(dest)
		dest.mkdir(parents=True, exist_ok=True)
		with h5py.File(dest / "store.h5", "w") as f:
			for item in reader_for(source).read():
				if self.content == "full" and isinstance(item, Table):
					_h5_table(f.require_group("/".join((*item.prefix, *item.entities.path()))), item.name, item.df, item.meta)
					continue
				if self.content == "full" and isinstance(item, Attrs):
					f.require_group("/".join(item.path) or "/").attrs["meta"] = json.dumps(item.attrs)
					continue
				if not isinstance(item, Recording):
					continue
				rec_id = _rec_id(item.entities)
				shape = (len(item.raw.ch_names), int(item.raw.n_times))

				cal = _int16_calibration(item.raw) if self.packed else None
				scale, offset = cal if cal is not None else (None, None)
				# int16 where the source is calibrated; else lossless float32, as neurozarr does
				dtype = np.dtype("int16" if cal is not None else ("float32" if self.packed else self.dtype))

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
						ds[:, start:stop] = _encode_int16(physical, scale, offset)
					else:
						ds[:, start:stop] = physical.astype(dtype)
				ds.attrs["sfreq"] = float(item.raw.info["sfreq"])
				ds.attrs["ch_names"] = list(item.raw.ch_names)
				# what neurozarr keeps with every recording too: its sidecar metadata and the
				# file's own annotations (as an events table), so "signals" is equal content
				ds.attrs["meta"] = json.dumps(item.meta)
				events = _annotations(item.raw)
				if events is not None:
					_h5_table(grp, "annotations", events, {"source": "mne.Raw.annotations"})
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

		self._path = Path(dest) / "store.h5"
		self._file = h5py.File(self._path, "r")
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

	def forget(self, rec_id: str) -> None:
		import h5py
		# re-open the file: HDF5's decoded-chunk cache (8 MiB per dataset in HDF5 2.0)
		# lives on the open dataset, and otherwise served repeated reads from memory
		names = list(self._recs)
		self._file.close()
		self._file = h5py.File(self._path, "r")
		self._recs = {name: self._file[name + "/data"] for name in names}

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

	def __new__(cls, raw: Any, cal: "tuple[np.ndarray, np.ndarray] | None" = None, float_dtype: str = "float32",
				**kwargs: Any) -> Any:
		from hdmf.data_utils import GenericDataChunkIterator

		class _Impl(GenericDataChunkIterator):
			def __init__(self, raw: Any, **kw: Any) -> None:
				self._raw = raw
				super().__init__(**kw)

			def _get_data(self, selection: tuple) -> np.ndarray:
				time_sel, chan_sel = selection
				start = time_sel.start or 0
				stop = time_sel.stop if time_sel.stop is not None else int(self._raw.n_times)
				physical = self._raw.get_data(start=start, stop=stop)
				data = _encode_int16(physical, *cal) if cal is not None else physical.astype(float_dtype)
				return data.T[:, chan_sel]

			def _get_dtype(self) -> np.dtype:
				return np.dtype("int16" if cal is not None else float_dtype)

			def _get_maxshape(self) -> tuple[int, int]:
				return (int(self._raw.n_times), len(self._raw.ch_names))

		return _Impl(raw, **kwargs)


def _build_nwb_files(source: Path, backend: str, dtype: str = "float32", content: str = "signals",
					 dataset_meta: dict[str, Any] | None = None) -> Iterator[tuple[tuple[str, str], Any]]:
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
	the Zarr path would behave the same way).

	dtype="int16" stores int16 codes the way NeuroConv does: ElectricalSeries
	`channel_conversion` carries the per-channel gain and `offset` the shift. NWB has
	one offset per series, so a series whose channels' offsets differ stays float32.

	Every series keeps its sidecar JSON (in scratch) and its file's annotations (a
	TimeIntervals table), as neurozarr does. content="full" adds every sidecar table
	(DynamicTable, "sidecars" module) and metadata file: session/subject JSON in the
	scratch of that session's file(s); dataset-wide JSON in ``dataset_meta``, which the
	arm writes beside the files (where DANDI keeps dataset metadata); anything arriving
	after its file is done goes there too, keyed by path, so nothing is dropped."""
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

	from neurozarr.items import Attrs, Table

	open_key: tuple[str, str] | None = None
	nwbfile: Any = None
	rows: dict[str, int] = {}
	scratch: dict[str, dict[str, Any]] = {}  # this file's metadata, by path, merged
	subject_meta: dict[str, dict[str, dict[str, Any]]] = {}
	pending: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
	done: set[tuple[str, str]] = set()
	dataset_meta = dataset_meta if dataset_meta is not None else {}

	def finish() -> Any:
		for path, meta in scratch.items():
			nwbfile.add_scratch(json.dumps(meta), name=path, description="BIDS metadata")
		return nwbfile

	for item in reader_for(source).read():
		if isinstance(item, Attrs):
			if content != "full":
				continue
			path = "__".join(item.path) or "dataset"
			sub = next((part for part in item.path if part.startswith("sub-")), None)
			ses = next((part for part in item.path if part.startswith("ses-")), None)
			if sub is None:
				dataset_meta.setdefault(path, {}).update(item.attrs)
			elif ses is None:  # subject-level: into each of its session files
				subject_meta.setdefault(sub, {}).setdefault(path, {}).update(item.attrs)
				if open_key is not None and open_key[0] == sub:
					scratch.setdefault(path, {}).update(item.attrs)
			else:
				key = (sub, ses or "ses-none")
				if key == open_key:
					scratch.setdefault(path, {}).update(item.attrs)
				elif key in done:
					dataset_meta.setdefault(path, {}).update(item.attrs)
				else:
					pending.setdefault(key, {}).setdefault(path, {}).update(item.attrs)
			continue
		if isinstance(item, Table) and content != "full":
			continue
		if not isinstance(item, (Recording, Table)):
			continue
		key = _nwb_group_key(item.entities)
		if key != open_key:
			if open_key is not None:
				yield open_key, finish()
				done.add(open_key)
			nwbfile = NWBFile(session_description=f"{key[0]} {key[1]}", identifier="/".join(key),
							   session_start_time=datetime.now(timezone.utc))
			device = nwbfile.create_device(name="device")
			nwbfile.create_electrode_group(name="group", description="all channels", location="unknown", device=device)
			rows = {}
			scratch = {path: dict(meta) for path, meta in subject_meta.get(key[0], {}).items()}
			for path, meta in pending.pop(key, {}).items():
				scratch.setdefault(path, {}).update(meta)
			open_key = key

		within = "__".join(item.entities.path()[2:])  # inside this session's file
		if isinstance(item, Table):
			module = nwbfile.processing.get("sidecars") or nwbfile.create_processing_module("sidecars", "BIDS sidecar tables")
			module.add(_nwb_table(f"{within}__{item.name}" if within else item.name, item.df, item.meta,
								  "/".join((*item.prefix, *item.entities.path(), item.name))))
			continue

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
		cal = _int16_calibration(item.raw) if dtype == "int16" else None
		if cal is not None and not np.all(cal[1] == cal[1][0]):
			cal = None
		float_dtype = "float32" if dtype == "int16" else dtype
		shape = (len(item.raw.ch_names), int(item.raw.n_times))
		itemsize = 2 if cal is not None else np.dtype(float_dtype).itemsize
		chunks = util.chunk_shape(shape, itemsize, 8 * 1024 * 1024, 65536)[::-1]
		iterator = _RawChunkIterator(item.raw, cal=cal, float_dtype=float_dtype, chunk_shape=chunks)
		wrapped = DataIOCls(data=iterator, **dataio_kwargs)
		name = item.entities.group_name() or "data"
		scaling: dict[str, Any] = {} if cal is None else {"channel_conversion": cal[0].tolist(), "offset": float(cal[1][0])}
		es = ElectricalSeries(name=name, data=wrapped, electrodes=region,
							   rate=float(item.raw.info["sfreq"]), starting_time=0.0, **scaling)
		nwbfile.add_acquisition(es)
		scratch.setdefault(f"{name}__meta", {}).update(item.meta)
		events = _annotations(item.raw)
		if events is not None:
			nwbfile.add_time_intervals(_nwb_intervals(f"{name}__annotations", events))

	if open_key is not None:
		yield open_key, finish()
	for meta in pending.values():  # for a session no file was made for: kept, dataset-wide
		for path, value in meta.items():
			dataset_meta.setdefault(path, {}).update(value)


def _nwb_frame(df: Any) -> Any:
	"""A DataFrame hdmf can write: text (any string or categorical dtype) as plain str
	with missing as "", numbers as numpy with missing as NaN."""
	import pandas as pd
	out = {}
	for i, (_, series) in enumerate(df.items()):
		if pd.api.types.is_numeric_dtype(series.dtype):
			out[f"c{i}"] = series.to_numpy(dtype="float64" if series.isna().any() else None, na_value=np.nan)
		else:
			out[f"c{i}"] = series.to_numpy(dtype=object, na_value="").astype(str)
	return pd.DataFrame(out, index=pd.RangeIndex(len(df)))


def _nwb_table(name: str, df: Any, meta: dict[str, Any], key: str) -> Any:
	from hdmf.common import DynamicTable
	description = {"key": key, "columns": [str(c) for c in df.columns], "meta": meta}
	return DynamicTable.from_dataframe(_nwb_frame(df), name=name, table_description=json.dumps(description))


def _nwb_intervals(name: str, events: Any) -> Any:
	import pandas as pd
	from pynwb.epoch import TimeIntervals
	frame = pd.DataFrame({"start_time": events["onset"].to_numpy(float),
						  "stop_time": (events["onset"] + events["duration"]).to_numpy(float),
						  "trial_type": events["trial_type"].astype(str).to_numpy()})
	return TimeIntervals.from_dataframe(frame, name=name, table_description="mne.Raw.annotations")


class NwbHdf5Arm:
	name = "nwb_hdf5"

	def __init__(self, packed: bool = False, content: str = "signals", dtype: str | None = None):
		self.dtype = dtype or ("int16" if packed else "float32")
		self.packed = self.dtype == "int16"
		self.content = _check_content(content)

	def convert(self, source: Path, dest: Path) -> None:
		from pynwb import NWBHDF5IO

		dest = Path(dest)
		if dest.exists():
			shutil.rmtree(dest)
		dest.mkdir(parents=True)
		dataset_meta: dict[str, Any] = {}
		for (sub_id, ses_id), nwbfile in _build_nwb_files(source, backend="hdf5", dtype=self.dtype, content=self.content,
															dataset_meta=dataset_meta):
			path = dest / f"{sub_id}_{ses_id}.nwb"
			with NWBHDF5IO(str(path), mode="w") as io:
				io.write(nwbfile)
		if dataset_meta:
			(dest / "dataset_metadata.json").write_text(json.dumps(dataset_meta))

	def open(self, dest: Path) -> "NwbStore":
		from pynwb import NWBHDF5IO
		return NwbStore(dest, NWBHDF5IO, "*.nwb")


class NwbZarrArm:
	name = "nwb_zarr"

	def __init__(self, packed: bool = False, content: str = "signals", dtype: str | None = None):
		self.dtype = dtype or ("int16" if packed else "float32")
		self.packed = self.dtype == "int16"
		self.content = _check_content(content)

	def convert(self, source: Path, dest: Path) -> None:
		from hdmf_zarr import NWBZarrIO

		dest = Path(dest)
		if dest.exists():
			shutil.rmtree(dest)
		dest.mkdir(parents=True)
		dataset_meta: dict[str, Any] = {}
		for (sub_id, ses_id), nwbfile in _build_nwb_files(source, backend="zarr", dtype=self.dtype, content=self.content,
															dataset_meta=dataset_meta):
			path = dest / f"{sub_id}_{ses_id}.nwb.zarr"
			with NWBZarrIO(str(path), mode="w") as io:
				io.write(nwbfile)
		if dataset_meta:
			(dest / "dataset_metadata.json").write_text(json.dumps(dataset_meta))

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
		self._io_cls, self._file_of = io_cls, {}  # rec_id -> (its file, its io)
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
				self._file_of[rec_id] = (path, io)
				self._info[rec_id] = RecInfo(
					id=rec_id, entities=entities, n_channels=series.data.shape[1],
					n_samples=series.data.shape[0], sfreq=float(series.rate or 0.0),
					chunk_shape=getattr(series.data, "chunks", None),
				)

	def recordings(self) -> list[RecInfo]:
		return list(self._info.values())

	def _decode(self, rec_id: str, data: np.ndarray, channel: "int | None" = None) -> np.ndarray:
		"""Time-major codes -> channel-major physical values. Same arithmetic as pynwb's
		get_data_in_units (data * conversion * channel_conversion + offset), but on the
		slice already read, since that method always reads the whole series."""
		series = self._recs[rec_id]
		if series.channel_conversion is None:
			return data.T if channel is None else data
		gain = np.asarray(series.channel_conversion) * series.conversion
		if channel is not None:
			return data.astype("float64") * gain[channel] + series.offset
		return data.T.astype("float64") * gain[:, None] + series.offset

	def full_read(self, rec_id: str) -> np.ndarray:
		return self._decode(rec_id, self._recs[rec_id].data[:, :])  # back to channel-major for the runner's comparisons

	def window_read(self, rec_id: str, start: int, n: int) -> np.ndarray:
		return self._decode(rec_id, self._recs[rec_id].data[start:start + n, :])

	def channel_read(self, rec_id: str, channel: int) -> np.ndarray:
		return self._decode(rec_id, self._recs[rec_id].data[:, channel], channel)

	def search(self, **entities: str) -> list[str]:
		return [rid for rid, info in self._info.items()
				if all(str(info.entities.get(k)) == str(v) for k, v in entities.items())]

	def sequential_scan(self, rec_id: str, block_samples: int) -> Iterator[np.ndarray]:
		data = self._recs[rec_id].data
		n = data.shape[0]
		for start in range(0, n, block_samples):
			yield self._decode(rec_id, data[start:min(n, start + block_samples), :])

	def forget(self, rec_id: str) -> None:
		path, io = self._file_of[rec_id]  # re-open just this recording's file
		io.close()
		fresh = self._io_cls(str(path), mode="r")
		nwbfile = fresh.read()
		self._ios = [fresh if x is io else x for x in self._ios]
		for other, (p, o) in list(self._file_of.items()):
			if o is io:
				self._recs[other] = nwbfile.acquisition[self._recs[other].name]
				self._file_of[other] = (p, fresh)

	def close(self) -> None:
		for io in self._ios:
			io.close()


FORMATS: dict[str, Any] = {
	"neurozarr": NeurozarrArm,
	"plain_zarr": PlainZarrArm,
	"hdf5": lambda dtype, content: Hdf5Arm(dtype=dtype, content=content),
	"nwb_hdf5": lambda dtype, content: NwbHdf5Arm(dtype=dtype, content=content),
	"nwb_zarr": lambda dtype, content: NwbZarrArm(dtype=dtype, content=content),
}
DTYPES = ("int16", "float32", "float64")
CONTENTS = ("signals", "full")

# every format x stored dtype x content level, as "<format>_<dtype>_<content>" -- the
# experiment-1 matrix. int16 needs the source's calibration (EDF family); without it
# each format stores float32, and that cell is reported as int16 unavailable.
ARMS: dict[str, Any] = {"raw": RawArm()}
for _format, _make in FORMATS.items():
	for _dtype in DTYPES:
		for _content in CONTENTS:
			ARMS[f"{_format}_{_dtype}_{_content}"] = _make(dtype=_dtype, content=_content)
# names earlier results and scripts use
ARMS.update({
	"neurozarr": ARMS["neurozarr_int16_full"],
	"plain_zarr": ARMS["plain_zarr_int16_full"],
	"hdf5_mat73": ARMS["hdf5_float32_signals"],
	"nwb_hdf5": ARMS["nwb_hdf5_float32_signals"],
	"nwb_zarr": ARMS["nwb_zarr_float32_signals"],
	"hdf5_int16": ARMS["hdf5_int16_signals"],
	"nwb_hdf5_int16": ARMS["nwb_hdf5_int16_signals"],
	"nwb_zarr_int16": ARMS["nwb_zarr_int16_signals"],
})
