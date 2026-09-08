from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import icechunk
import mne  # type: ignore[import-untyped]  # mne ships no type information
import numpy as np
import pandas as pd
import zarr
from zarr.codecs import ZstdCodec

from . import util
from .entities import Entities
from .errors import ValidationError, WriteConflictError
from .items import Array, Attrs, ExternalFile, Item, Recording, Table
from .log import log_mem


class ExistingPolicy(StrEnum):
	"""How a writer handles an item already present at the same path."""

	ERROR = "error"
	SKIP = "skip"
	REPLACE = "replace"


@dataclass
class CodecConfig:
	"""How sample data is packed and compressed. Pass one to :class:`Repo`.

	Parameters
	----------
	filters : list, optional
		Zarr filter codecs applied before compression. Empty by default.
	compressors : list, optional
		Zarr compression codecs. Defaults to zstd at level 19.
	bitround_k : int, default 0
		Drop this many low bits from each sample before compressing. Lossy, but
		compresses better. 0 disables it.
	dtype : {"int16", "float16"}, default "int16"
		How samples are packed. ``"int16"`` stores integers with a per-channel
		scale and offset, which is lossless for data that arrived as integers,
		as EDF does, and compresses best. ``"float16"`` stores physical values
		rounded to half precision. Recordings whose source carries no
		calibration are stored as float32 regardless.
	chunk_target_bytes : int, default 8 MiB
		Target size of one chunk, which bounds chunk size for arrays with many
		channels.
	max_chunk_samples : int, default 65536
		Cap on samples per chunk. Smaller chunks make reading short windows
		cheaper and full reads slightly more expensive.

	Examples
	--------
	>>> repo = Repo.create("./study.zarr", codec=CodecConfig(dtype="float16"))
	"""

	filters: list[Any] = field(default_factory=list)
	compressors: list[Any] = field(default_factory=lambda: [ZstdCodec(level=19)])
	bitround_k: int = 0
	dtype: str = "int16"
	chunk_target_bytes: int = 8 * 1024 * 1024
	max_chunk_samples: int = 65536

	def __post_init__(self) -> None:
		if self.dtype not in ("int16", "float16"):
			raise ValidationError("codec dtype must be 'int16' or 'float16'")
		if isinstance(self.bitround_k, bool) or not isinstance(self.bitround_k, int) \
				or not 0 <= self.bitround_k <= 15:
			raise ValidationError("bitround_k must be an integer from 0 through 15")
		if self.dtype != "int16" and self.bitround_k:
			raise ValidationError("bitround_k is only valid with dtype='int16'")
		if isinstance(self.chunk_target_bytes, bool) or not isinstance(self.chunk_target_bytes, int) \
				or self.chunk_target_bytes <= 0:
			raise ValidationError("chunk_target_bytes must be positive")
		if isinstance(self.max_chunk_samples, bool) or not isinstance(self.max_chunk_samples, int) \
				or self.max_chunk_samples <= 0:
			raise ValidationError("max_chunk_samples must be positive")


class Writer:
	"""Internal engine that places standardized items in the store's tree, which
	is laid out by subject/session/datatype/entities following BIDS naming.
	Not user-facing directly -- use Repo/Subject/Visit, which own a Writer."""

	def __init__(self, session: icechunk.Session, codec: CodecConfig | None = None):
		self._session = session
		self._codec = codec or CodecConfig()
		# mode="a" (create if missing), never "w" -- "w" means "overwrite if exists",
		# which silently wipes a store you reopened to add more data to.
		self.root = zarr.open_group(store=session.store, mode="a")

	def _group_for(self, prefix: tuple[str, ...], entities: Entities) -> zarr.Group:
		# a Writer always scopes to one subject's own repo, so the leading
		# sub-XXX from entities.path() is redundant -- the repo already is that subject.
		path = (*prefix, *entities.path()[1:])
		group = self.root
		for part in path:
			group = group.require_group(part)
		return group

	@staticmethod
	def _should_write(container: zarr.Group, name: str, policy: ExistingPolicy) -> bool:
		if name not in container:
			return True
		if policy is ExistingPolicy.SKIP:
			return False
		if policy is ExistingPolicy.ERROR:
			raise WriteConflictError(f"an item already exists at {container.path}/{name}; use existing='skip' or 'replace'")
		return True

	def add_recording(self, item: Recording, existing: ExistingPolicy = ExistingPolicy.ERROR) -> bool:
		group = self._group_for(item.prefix, item.entities)
		if not self._should_write(group, "data", existing):
			return False
		attrs = dict(item.meta)
		attrs["_neurozarr_item_type"] = "recording"

		# Keep what the Raw already knows, so a recording ingested without BIDS
		# sidecars (manual API, ManifestReader) still reads back as a real Raw.
		attrs.setdefault("SamplingFrequency", float(item.raw.info["sfreq"]))
		attrs.setdefault("ch_names", list(item.raw.ch_names))
		self._add_annotations(group, item.raw)

		storage_dtype: Any
		scale: np.ndarray | None = None
		offset: np.ndarray | None = None
		if self._codec.dtype == "float16":
			storage_dtype = np.dtype("float16")
			attrs["data_note"] = "data is physical value in the channel's native unit, rounded to float16"
		else:
			extras = getattr(item.raw, "_raw_extras", None)
			if extras and "cal" in extras[0]:
				cal = np.array(extras[0]["cal"])
				units = np.array(extras[0]["units"])
				offsets = np.array(extras[0]["offsets"])
				scale = cal * units
				offset = offsets * units
				if scale.shape != (len(item.raw.ch_names),) or offset.shape != scale.shape or np.any(scale == 0):
					raise ValidationError("recording calibration must contain one non-zero scale and offset per channel")
				storage_dtype = np.dtype("int16")
				attrs["data_scale"] = scale.tolist()
				attrs["data_offset"] = offset.tolist()
				attrs["data_note"] = "physical value = data * data_scale + data_offset (per channel)"
			else:
				# ponytail: no EDF-style per-channel calibration available (non-EDF reader) -- store
				# losslessly as float32 rather than guessing a scale/offset.
				storage_dtype = np.dtype("float32")
				attrs["data_note"] = "data is physical value in the channel's native unit (float32, no calibration metadata available)"

		shape = (len(item.raw.ch_names), int(item.raw.n_times))
		chunks = util.chunk_shape(shape, storage_dtype.itemsize,
								  self._codec.chunk_target_bytes, self._codec.max_chunk_samples)
		util.set_attrs(group, attrs)
		array = group.create_array(
			"data",
			shape=shape,
			dtype=storage_dtype,
			chunks=chunks,
			filters=self._codec.filters,
			compressors=self._codec.compressors,
			overwrite=True,  # re-converting a source into an existing store replaces it
		)
		for start in range(0, shape[-1], chunks[-1]):
			stop = min(shape[-1], start + chunks[-1])
			physical = item.raw.get_data(start=start, stop=stop)
			if storage_dtype == np.dtype("int16"):
				assert scale is not None and offset is not None
				encoded = np.round((physical - offset[:, None]) / scale[:, None])
				if not np.isfinite(encoded).all() or encoded.min() < -32768 or encoded.max() > 32767:
					raise ValidationError(
					"recording samples cannot be represented by their int16 calibration; use dtype='float16'"
				)
				data = encoded.astype(np.int16)
				if self._codec.bitround_k:
					step = 1 << self._codec.bitround_k
					data = np.clip(np.round(data.astype(np.int32) / step) * step, -32768, 32767).astype(np.int16)
			else:
				data = physical.astype(storage_dtype)
			array[:, start:stop] = data
		log_mem()
		return True

	def _add_annotations(self, group: zarr.Group, raw: "mne.io.BaseRaw") -> None:
		"""An mne.Raw's annotations are events -- store them the way BIDS does, as an
		events table, so they survive a round trip. A recording that already has an
		events table from its BIDS sidecar keeps that one."""
		annotations = getattr(raw, "annotations", None)
		if not annotations or len(annotations) == 0 or "events" in group:
			return
		util.create_table(group, "events", pd.DataFrame({
			"onset": annotations.onset,
			"duration": annotations.duration,
			"trial_type": [str(d) for d in annotations.description],
		}), {"source": "mne.Raw.annotations"})

	def add_table(self, item: Table, existing: ExistingPolicy = ExistingPolicy.ERROR) -> bool:
		group = self._group_for(item.prefix, item.entities)
		if not self._should_write(group, item.name, existing):
			return False
		util.create_table(group, item.name, item.df, item.meta)
		log_mem()
		return True

	def add_attrs(self, item: Attrs) -> None:
		group = self.root
		for part in item.path[:-1]:
			group = group.require_group(part)
		if not item.path:
			target: zarr.Group | zarr.Array = group
		elif item.path[-1] in group:
			target = group[item.path[-1]]  # an existing item -- a plain Array included, not just a Group
		else:
			target = group.require_group(item.path[-1])
		util.set_attrs(target, item.attrs)
		log_mem()

	def add_external_file(self, item: ExternalFile, existing: ExistingPolicy = ExistingPolicy.ERROR) -> bool:
		group = self._group_for(item.prefix, item.entities)
		if not self._should_write(group, item.name, existing):
			return False
		reference = group.create_group(item.name, overwrite=True)
		util.set_attrs(reference, {
			"_neurozarr_item_type": "external_file",
			"uri": util.safe_uri(str(item.uri)),
			"media_type": item.media_type,
			"reader_hint": item.reader_hint,
			**item.meta,
		})
		log_mem()
		return True

	def add_array(self, item: Array, existing: ExistingPolicy = ExistingPolicy.ERROR) -> bool:
		group = self._group_for(item.prefix, item.entities)
		if not self._should_write(group, item.name, existing):
			return False
		shape = tuple(int(size) for size in item.data.shape)
		dtype = np.dtype(item.data.dtype)
		chunks = util.chunk_shape(shape, dtype.itemsize,
								  self._codec.chunk_target_bytes, self._codec.max_chunk_samples)
		array = group.create_array(item.name, shape=shape, dtype=dtype, chunks=chunks, overwrite=True)
		for start in range(0, shape[-1], chunks[-1]):
			stop = min(shape[-1], start + chunks[-1])
			selection = (slice(None),) * (len(shape) - 1) + (slice(start, stop),)
			array[selection] = np.asarray(item.data[selection])
		util.set_attrs(array, {
			"_neurozarr_item_type": "array",
			"dims": list(item.dims),
			"coords": item.coords,
			**item.meta,
		})
		log_mem()
		return True

	def dispatch(self, item: Item, existing: ExistingPolicy = ExistingPolicy.ERROR) -> bool:
		if isinstance(item, Recording):
			return self.add_recording(item, existing)
		elif isinstance(item, Table):
			return self.add_table(item, existing)
		elif isinstance(item, Attrs):
			self.add_attrs(item)
			return True
		elif isinstance(item, ExternalFile):
			return self.add_external_file(item, existing)
		elif isinstance(item, Array):
			return self.add_array(item, existing)
		else:
			raise TypeError(f"unknown item type {type(item)!r}")

	def save(self, message: str) -> str | None:
		"""Commit this subject's session. A session with nothing new in it is left
		alone -- icechunk refuses empty commits, and reopening a store to touch one
		subject shouldn't fail because the others had no changes."""
		if not self._session.has_uncommitted_changes:
			return None
		snapshot = self._session.commit(message)
		log_mem()
		return snapshot
