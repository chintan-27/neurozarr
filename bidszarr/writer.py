from dataclasses import dataclass, field

import icechunk
import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from . import util
from .entities import Entities
from .items import Attrs, Recording, Table
from .log import log_mem


@dataclass
class CodecConfig:
	"""How sample data is packed and compressed.

	dtype:       "int16" packs with per-channel scale/offset (lossless for EDF-sourced
	             data, best compression); "float16" stores rounded physical values.
	bitround_k:  drop k low bits before compressing -- lossy, smaller. 0 disables.
	filters/compressors: zarr codecs, defaulting to zstd level 19.
	chunk_target_bytes / max_chunk_samples: chunking along time. Bigger chunks
	             compress slightly better; smaller ones make windowed reads cheaper.
	"""

	filters: list = field(default_factory=list)
	compressors: list = field(default_factory=lambda: [ZstdCodec(level=19)])
	bitround_k: int = 0
	dtype: str = "int16"
	chunk_target_bytes: int = 8 * 1024 * 1024
	max_chunk_samples: int = 65536


class Writer:
	"""Internal engine that writes standardized items into a BIDS-shaped Zarr tree.
	Not user-facing directly -- use Repo/Subject/Visit, which own a Writer."""

	def __init__(self, session: icechunk.Session, codec: CodecConfig = None):
		self._session = session
		self._codec = codec or CodecConfig()
		# mode="a" (create if missing), never "w" -- "w" means "overwrite if exists",
		# which silently wipes a store you reopened to add more data to.
		self.root = zarr.open_group(store=session.store, mode="a")

	def _group_for(self, prefix: tuple, entities: Entities) -> zarr.Group:
		# a Writer always scopes to one subject's own repo, so the leading
		# sub-XXX from entities.path() is redundant -- the repo already is that subject.
		path = (*prefix, *entities.path()[1:])
		group = self.root
		for part in path:
			group = group.require_group(part)
		return group

	def add_recording(self, item: Recording):
		group = self._group_for(item.prefix, item.entities)
		physical = item.raw.get_data()
		attrs = dict(item.meta)

		# Keep what the Raw already knows, so a recording ingested without BIDS
		# sidecars (manual API, ManifestReader) still reads back as a real Raw.
		attrs.setdefault("SamplingFrequency", float(item.raw.info["sfreq"]))
		attrs.setdefault("ch_names", list(item.raw.ch_names))

		if self._codec.dtype == "float16":
			data = physical.astype(np.float16)
			attrs["data_note"] = "data is physical value in the channel's native unit, rounded to float16"
		else:
			extras = getattr(item.raw, "_raw_extras", None)
			if extras and "cal" in extras[0]:
				cal = np.array(extras[0]["cal"])
				units = np.array(extras[0]["units"])
				offsets = np.array(extras[0]["offsets"])
				scale = cal * units
				offset = offsets * units
				data = np.round((physical - offset[:, None]) / scale[:, None]).astype(np.int16)
				if self._codec.bitround_k:
					step = 1 << self._codec.bitround_k
					data = np.clip(np.round(data.astype(np.int32) / step) * step, -32768, 32767).astype(np.int16)
				attrs["data_scale"] = scale.tolist()
				attrs["data_offset"] = offset.tolist()
				attrs["data_note"] = "physical value = data * data_scale + data_offset (per channel)"
			else:
				# ponytail: no EDF-style per-channel calibration available (non-EDF reader) -- store
				# losslessly as float32 rather than guessing a scale/offset.
				data = physical.astype(np.float32)
				attrs["data_note"] = "data is physical value in the channel's native unit (float32, no calibration metadata available)"

		util.set_attrs(group, attrs)
		group.create_array(
			"data",
			data=data,
			chunks=util.chunk_shape(data.shape, data.itemsize,
									self._codec.chunk_target_bytes, self._codec.max_chunk_samples),
			filters=self._codec.filters,
			compressors=self._codec.compressors,
			overwrite=True,  # re-converting a source into an existing store replaces it
		)
		log_mem()

	def add_table(self, item: Table):
		group = self._group_for(item.prefix, item.entities)
		util.create_table(group, item.name, item.df, item.meta)
		log_mem()

	def add_attrs(self, item: Attrs):
		group = self.root
		for part in item.path:
			group = group.require_group(part)
		util.set_attrs(group, item.attrs)
		log_mem()

	def dispatch(self, item):
		if isinstance(item, Recording):
			self.add_recording(item)
		elif isinstance(item, Table):
			self.add_table(item)
		elif isinstance(item, Attrs):
			self.add_attrs(item)
		else:
			raise TypeError(f"unknown item type {type(item)!r}")

	def save(self, message: str):
		"""Commit this subject's session. A session with nothing new in it is left
		alone -- icechunk refuses empty commits, and reopening a store to touch one
		subject shouldn't fail because the others had no changes."""
		if not self._session.has_uncommitted_changes:
			return None
		snapshot = self._session.commit(message)
		log_mem()
		return snapshot
