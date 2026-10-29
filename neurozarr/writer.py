import itertools
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import icechunk
import mne  # type: ignore[import-untyped]  # mne ships no type information
import numpy as np
import pandas as pd
import zarr
from zarr.core.buffer import default_buffer_prototype

from . import util
from .codecs import CodecPlan
from .constraints import validate_segment
from .entities import Entities
from .errors import ValidationError, WriteConflictError
from .items import Array, Attrs, ExternalFile, Item, Recording, Table
from .log import log_mem, log_progress, record_write


def copy_chunks(source: zarr.Array, dest: zarr.Array,
				chunks: "Iterable[tuple[int, ...]] | None" = None) -> None:
	"""Copy an array's values chunk by chunk -- all chunks, or the given chunk
	coordinates -- so a large array is never held in memory whole."""
	if source.ndim == 0:
		dest[...] = source[...]
		return
	grid = [range(-(-dim // size)) for dim, size in zip(source.shape, source.chunks)]
	for coords in chunks if chunks is not None else itertools.product(*grid):
		region = tuple(slice(c * size, min((c + 1) * size, dim))
					   for c, size, dim in zip(coords, source.chunks, source.shape))
		dest[region] = source[region]


def _copy_node(source: zarr.Group | zarr.Array, dest_parent: zarr.Group, dest_name: str) -> None:
	"""Recursively copy a zarr node under a new key. zarr has no move/rename
	primitive (zarr.Group.move raises NotImplementedError). icechunk does
	(Repository.rearrange_session().move()), but that session type commits
	independently of a writable_session and has its own lifecycle -- adopting
	it means rename()/delete() would need their own commit path, breaking the
	"not committed until Repo.commit()" contract every other write follows.
	ponytail: copy-then-delete is O(size) instead of O(1); move to
	rearrange_session if a rename-heavy workload ever makes that cost matter."""
	if isinstance(source, zarr.Array):
		# carry the codec over: left out, zarr would re-encode with its default
		dest_array = dest_parent.create_array(dest_name, shape=source.shape, dtype=source.dtype,
											   chunks=source.chunks, filters=source.filters,
											   compressors=source.compressors, serializer=source.serializer or "auto",
											   overwrite=True)
		copy_chunks(source, dest_array)
		util.set_attrs(dest_array, source.attrs.asdict())
		return
	dest_group = dest_parent.create_group(dest_name, overwrite=True)
	util.set_attrs(dest_group, source.attrs.asdict())
	for name, child in source.members():
		_copy_node(child, dest_group, name)


class ExistingPolicy(StrEnum):
	"""How a writer handles an item already present at the same path."""

	ERROR = "error"
	SKIP = "skip"
	REPLACE = "replace"


@dataclass
class RecordingLayout:
	"""How a recording's samples are stored: dtype, and the per-channel int16
	calibration when there is one."""
	dtype: np.dtype
	scale: np.ndarray | None
	offset: np.ndarray | None
	note: str


def recording_layout(raw: "mne.io.BaseRaw", codec: "CodecConfig") -> RecordingLayout:
	"""The storage layout the writer uses for ``raw`` -- shared with codec probing,
	so a probe compresses exactly the bytes that will be written."""
	if codec.dtype in ("float16", "float32", "float64"):
		return RecordingLayout(np.dtype(codec.dtype), None, None,
							   f"data is physical value in the channel's native unit, as {codec.dtype}")
	extras = getattr(raw, "_raw_extras", None)
	# the calibration only makes int16 exact for 2-byte samples: BDF stores 3 bytes, GDF
	# up to 8, and their values overflow int16 (every write of a BDF used to fail here)
	# ponytail: wider sources go to float32 (~6e-8 relative); int32 packing would keep them exact
	if extras and "cal" in extras[0] and np.all(np.asarray(extras[0].get("dtype_byte", 0)) == 2):
		scale = np.array(extras[0]["cal"]) * np.array(extras[0]["units"])
		offset = np.array(extras[0]["offsets"]) * np.array(extras[0]["units"])
		if scale.shape != (len(raw.ch_names),) or offset.shape != scale.shape or np.any(scale == 0):
			raise ValidationError("recording calibration must contain one non-zero scale and offset per channel")
		return RecordingLayout(np.dtype("int16"), scale, offset,
							   "physical value = data * data_scale + data_offset (per channel)")
	# ponytail: no EDF-style per-channel calibration available (non-EDF reader) -- store
	# losslessly as float32 rather than guessing a scale/offset.
	return RecordingLayout(np.dtype("float32"), None, None,
						   "data is physical value in the channel's native unit (float32, no calibration metadata available)")


def encode_recording(raw: "mne.io.BaseRaw", start: int, stop: int, layout: RecordingLayout,
					 codec: "CodecConfig") -> np.ndarray:
	"""Samples ``start:stop`` of ``raw`` as stored."""
	physical = raw.get_data(start=start, stop=stop)
	if layout.dtype != np.dtype("int16"):
		return physical.astype(layout.dtype)
	assert layout.scale is not None and layout.offset is not None
	encoded = np.round((physical - layout.offset[:, None]) / layout.scale[:, None])
	if not np.isfinite(encoded).all() or encoded.min() < -32768 or encoded.max() > 32767:
		raise ValidationError("recording samples cannot be represented by their int16 calibration; use dtype='float16'")
	data = encoded.astype(np.int16)
	if codec.bitround_k:
		step = 1 << codec.bitround_k
		data = np.clip(np.round(data.astype(np.int32) / step) * step, -32768, 32767).astype(np.int16)
	return data


def recording_chunks(raw: "mne.io.BaseRaw", layout: RecordingLayout, codec: "CodecConfig") -> tuple[int, ...]:
	shape = (len(raw.ch_names), int(raw.n_times))
	return util.chunk_shape(shape, layout.dtype.itemsize, codec.chunk_target_bytes, codec.max_chunk_samples)


def middle_chunk(raw: "mne.io.BaseRaw", codec: "CodecConfig") -> np.ndarray:
	"""One real chunk from the middle of the recording, as stored -- the probe input.
	The middle, because the start is often a settling transient."""
	layout = recording_layout(raw, codec)
	width = recording_chunks(raw, layout, codec)[-1]
	start = max(0, (int(raw.n_times) - width) // 2)
	return encode_recording(raw, start, min(int(raw.n_times), start + width), layout, codec)


@dataclass
class CodecConfig:
	"""How sample data is packed and compressed. Pass one to :class:`Repo`.

	Parameters
	----------
	filters : list, optional
		Zarr filter codecs applied before compression. Empty by default.
	compressors : list, optional
		Zarr compression codecs for everything added through the explicit API
		(``visit.add_recording`` and friends). Left unset, those use zarr's default.
		Data ingested from a reader uses the reader's codec plan instead (see
		:meth:`Repo.add`) -- unless this is set, in which case it overrides the
		plan, with a warning.
	bitround_k : int, default 0
		Drop this many low bits from each sample before compressing. Lossy, but
		compresses better. 0 disables it.
	dtype : {"int16", "float16", "float32", "float64"}, default "int16"
		How samples are packed. ``"int16"`` stores integers with a per-channel
		scale and offset, which is lossless for data that arrived as integers,
		as EDF does, and compresses best. ``"float16"``, ``"float32"`` and
		``"float64"`` store physical values at that precision (``"float64"`` is
		exactly what mne returns). Recordings whose source carries no
		calibration, or stores samples wider than 16 bits (BDF, most GDF),
		are stored as float32 under ``"int16"``.
	chunk_target_bytes : int, default 8 MiB
		Target size of one chunk, which bounds chunk size for arrays with many
		channels.
	max_chunk_samples : int, default 65536
		Cap on samples per chunk. Smaller chunks make reading short windows
		cheaper and full reads slightly more expensive.

	Examples
	--------
	>>> repo = Repo.init("./study.zarr", codec=CodecConfig(dtype="float16"))
	"""

	filters: list[Any] = field(default_factory=list)
	compressors: list[Any] | None = None
	bitround_k: int = 0
	dtype: str = "int16"
	chunk_target_bytes: int = 8 * 1024 * 1024
	max_chunk_samples: int = 65536

	def __post_init__(self) -> None:
		if self.dtype not in ("int16", "float16", "float32", "float64"):
			raise ValidationError("codec dtype must be 'int16', 'float16', 'float32' or 'float64'")
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


def _clashes(theirs: Any, mine: Any) -> list[str]:
	"""Paths both sides changed: the same chunk of an array, or any node whose
	metadata either side created, deleted, moved or updated. Both sides creating the
	same group is not a clash -- adding items under one new session creates its
	folders on each side -- but changing that group's metadata on both still is."""
	def nodes(diff: Any) -> set[str]:
		return {*diff.new_arrays, *diff.deleted_arrays, *diff.deleted_groups,
				*diff.updated_arrays, *diff.updated_groups, *(p for move in diff.moved_nodes for p in move)}
	def chunks(diff: Any) -> dict[str, set[tuple[int, ...]]]:
		return {path: {tuple(c) for c in coords} for path, coords in diff.updated_chunks.items()}
	their_nodes, my_nodes = nodes(theirs), nodes(mine)
	# a group one side created where the other created, deleted or wrote under it as an array
	their_nodes |= set(theirs.new_groups) & (my_nodes | set(mine.new_arrays))
	my_nodes |= set(mine.new_groups) & (their_nodes | set(theirs.new_arrays))
	their_chunks, my_chunks = chunks(theirs), chunks(mine)
	clashes = their_nodes & (my_nodes | set(my_chunks)) | my_nodes & set(their_chunks)
	clashes |= {path for path in their_chunks.keys() & my_chunks.keys() if their_chunks[path] & my_chunks[path]}
	return sorted(clashes)


def _join(parent: str, name: str) -> str:
	return f"{parent}/{name}" if parent else name


def _same_layout(group: zarr.Group, shape: tuple[int, ...], dtype: Any, chunks: tuple[int, ...],
				 compressors: Any) -> zarr.Array | None:
	"""The stored recording array, if a replacement can be written into it chunk by chunk:
	same shape, dtype, chunking and codec. (Packing scale and offset are compared by the
	caller's attrs: a different scale changes every stored value anyway.)"""
	if "data" not in group:
		return None
	array = cast(zarr.Array, group["data"])
	if array.shape != shape or array.dtype != np.dtype(dtype) or array.chunks != chunks:
		return None
	if compressors != "auto":  # an explicit codec must match the stored one
		wanted = list(compressors) if isinstance(compressors, (list, tuple)) else [compressors]
		if [c.to_dict() for c in array.compressors] != [c.to_dict() for c in wanted]:  # type: ignore[union-attr]
			return None
	return array


def _annotations_df(raw: "mne.io.BaseRaw") -> pd.DataFrame | None:
	annotations = getattr(raw, "annotations", None)
	if not annotations or len(annotations) == 0:
		return None
	return pd.DataFrame({
		"onset": annotations.onset,
		"duration": annotations.duration,
		"trial_type": [str(d) for d in annotations.description],
	})


class Writer:
	"""Internal engine that places standardized items in the store's tree, which
	is laid out by subject/session/datatype/entities following BIDS naming.
	Not user-facing directly -- use Repo/Subject/Visit, which own a Writer."""

	def __init__(self, session: "icechunk.Session | icechunk.ForkSession", codec: CodecConfig | None = None,
				 label: str = "", repo: icechunk.Repository | None = None, branch: str = "main",
				 base: str | None = None):
		# Staged writes go into a ForkSession: picklable, so it can sit in the index
		# between processes (and travel to a parallel worker), and committed by
		# merging it into a fresh session at the branch tip -- see commit(). `base` is
		# the snapshot it was forked from, for spotting changes made since.
		self._session = session
		self._repo, self._branch, self._base = repo, branch, base
		# where this repository keeps its snapshot files, for a local store (else None):
		# Icechunk's fork() writes a full copy of the tip's snapshot that nothing refers
		# to once the fork is committed, so save() removes it -- half of what an edit
		# added on a large subject (BRAVO sub-001: 2 x 610 KB per commit)
		self.snapshot_dir: Path | None = None
		self.drifted = False  # after save(): the branch moved since this was staged
		# index rows of the items written here, by path -- described from what is in hand,
		# so commit needs no re-read of them (see index.update_subject)
		self.captured: dict[str, dict[str, Any]] = {}
		self._codec = codec or CodecConfig()
		self._label = label
		# set by Repo.ingest for the duration of a reader's ingest; None for the explicit API
		self.plan: CodecPlan | None = None
		# mode="a" (create if missing), never "w" -- "w" means "overwrite if exists",
		# which silently wipes a store you reopened to add more data to.
		self._root = zarr.open_group(store=session.store, mode="a")
		# new nodes not yet in the store, written together by _flush (one call into
		# icechunk for many nodes, not ~0.2 ms + a thread hop each). Anything that reads
		# the store, or hands the session or root out, flushes first.
		self._pending: list[tuple[str, Any]] = []
		# groups this writer has opened, by path: each require_group is a zarr round trip
		# (~1.3 ms on icechunk), and every item in one ses/datatype folder walked the same
		# path again. Cleared by anything here that removes or moves nodes.
		self._groups: dict[tuple[str, ...], zarr.Group] = {(): self._root}
		# the children of groups this writer created in this session, so "does X exist?"
		# needs no store read (each ~1 ms on icechunk, slower as a session grows: ~6,500
		# of them for BRAVO sub-001). Only groups made here are tracked, and every child
		# this writer adds to one goes through _made; anything else asks the store.
		self._known: dict[tuple[str, ...], set[str]] = {}
		# attrs of the group documents replay() wrote, by key: two workers' units can each
		# create the same folder, and the second must not drop the first one's attrs
		self._replayed: dict[str, dict[str, Any]] = {}

	def _reset_paths(self) -> None:
		"""Forget cached groups and contents, after a node is removed or moved."""
		self._groups, self._known = {(): self._root}, {}

	@property
	def root(self) -> zarr.Group:
		self._flush()
		return self._root

	def _flush(self) -> None:
		if self._pending:
			batch, self._pending = self._pending, []
			self._root._sync(util.set_many(self._root.store, batch))

	def _stage(self, key: str, document: Any) -> None:
		self._pending.append((key, document))
		if len(self._pending) >= 512:  # bound what waits in memory
			self._flush()

	def _has(self, path: tuple[str, ...], name: str) -> bool:
		known = self._known.get(path)
		if known is not None:
			return name in known
		group = self._require(path)
		self._flush()
		return name in group

	def _made(self, path: tuple[str, ...], name: str) -> None:
		if path in self._known:
			self._known[path].add(name)

	def _require(self, path: tuple[str, ...]) -> zarr.Group:
		group = self._groups.get(path)
		if group is None:
			parent = self._require(path[:-1])
			if self._has(path[:-1], path[-1]):
				self._flush()  # it may be one of the pending nodes
				group = parent.require_group(path[-1])
			else:
				key, document, created = util.group_node(parent._async_group, path[-1])
				self._stage(key, document)
				group = zarr.Group(created)
				self._made(path[:-1], path[-1])
				self._known[path] = set()
			self._groups[path] = group
		return group

	def _compressors(self, item_type: str, dtype: np.dtype, chunks: tuple[int, ...],
					 sample: Any = None) -> Any:
		"""The codec for one array: CodecConfig.compressors if set (explicit API, or a pin
		that overrides a reader's plan), else the reader's plan, else zarr's default."""
		if self._codec.compressors is not None:
			return self._codec.compressors
		if self.plan is not None:
			return self.plan.compressors(item_type, dtype, chunks, sample)
		return "auto"  # zarr's default -- passing None would mean no compression at all

	def _item_path(self, rel: str) -> str:
		return f"{self._label}/{rel}" if rel else self._label

	def _capture(self, rel: str, kind: str, meta: dict[str, Any], **details: Any) -> None:
		from .entities import parse_entities
		from .index import make_row
		owner = rel if kind == "recording" else rel.rpartition("/")[0]
		path = self._item_path(rel)
		self.captured[path] = make_row(path, kind, parse_entities(owner.rsplit("/", 1)[-1]), meta, **details)

	def _forget(self, rel: str, below: bool = True) -> None:
		"""Drop captured rows a later change made stale; commit re-reads those items."""
		path = self._item_path(rel)
		for key in [k for k in self.captured if k == path or (below and k.startswith(path + "/"))]:
			del self.captured[key]

	def _table_array_options(self, data: np.ndarray) -> dict[str, Any]:
		"""Chunks and codec for one of a table's stored arrays."""
		chunks = util.chunk_shape(data.shape, data.dtype.itemsize, self._codec.chunk_target_bytes,
								  self._codec.max_chunk_samples)
		return {"chunks": chunks, "compressors": self._compressors(
			"Table", data.dtype, chunks, sample=lambda: data[tuple(slice(0, c) for c in chunks)])}

	@staticmethod
	def _path_for(prefix: tuple[str, ...], entities: Entities) -> tuple[str, ...]:
		# a Writer always scopes to one subject's own repo, so the leading
		# sub-XXX from entities.path() is redundant -- the repo already is that subject.
		return (*prefix, *entities.path()[1:])

	def _group_for(self, prefix: tuple[str, ...], entities: Entities) -> zarr.Group:
		return self._require(self._path_for(prefix, entities))

	@staticmethod
	def _should_write(container: zarr.Group, name: str, policy: ExistingPolicy, exists: bool | None = None) -> bool:
		if not (name in container if exists is None else exists):
			return True
		if policy is ExistingPolicy.SKIP:
			return False
		if policy is ExistingPolicy.ERROR:
			raise WriteConflictError(f"an item already exists at {container.path}/{name}; use existing='skip' or 'replace'")
		return True

	def add_recording(self, item: Recording, existing: ExistingPolicy = ExistingPolicy.ERROR) -> bool:
		path = self._path_for(item.prefix, item.entities)
		group = self._require(path)
		if not self._has(path, "data"):
			self._add_new_recording(group, path, item)
			return True
		self._flush()
		if not self._should_write(group, "data", existing):
			return False
		log_progress("Writing recording", f"{group.path}/data", f"{len(item.raw.ch_names)} ch, {item.raw.n_times} samples")
		attrs, layout, shape, chunks, compressors = self._recording_plan(item)
		self._add_annotations(group, item.raw)
		current = group.attrs.asdict()
		if {**current, **attrs} != current:
			util.set_attrs(group, attrs)
		array = _same_layout(group, shape, layout.dtype, chunks, compressors)
		if array is not None:
			# replacing a recording with one laid out the same way (a correction): write
			# only the chunks whose values differ -- a version then stores what changed,
			# not a re-encoded copy of the whole recording (42 MB EDF, one sample: 24 MB)
			for start in range(0, shape[-1], chunks[-1]):
				stop = min(shape[-1], start + chunks[-1])
				encoded = encode_recording(item.raw, start, stop, layout, self._codec)
				if not np.array_equal(array[:, start:stop], encoded):
					array[:, start:stop] = encoded
		else:
			array = group.create_array(
				"data", shape=shape, dtype=layout.dtype, chunks=chunks,
				filters=self._codec.filters, compressors=compressors,
				overwrite=True,  # re-converting a source into an existing store replaces it
			)
			for start in range(0, shape[-1], chunks[-1]):
				stop = min(shape[-1], start + chunks[-1])
				array[:, start:stop] = encode_recording(item.raw, start, stop, layout, self._codec)
		self._capture(group.path, "recording",
					  {k: v for k, v in group.attrs.asdict().items() if k != "_neurozarr_item_type"}, node=array)
		record_write(array.nbytes)
		log_mem()
		return True

	def _recording_plan(self, item: Recording) -> tuple[dict[str, Any], RecordingLayout, tuple[int, int],
															tuple[int, ...], Any]:
		"""(attrs, layout, shape, chunks, compressors) a recording is stored with."""
		attrs = dict(item.meta)
		attrs["_neurozarr_item_type"] = "recording"
		# Keep what the Raw already knows, so a recording ingested without BIDS
		# sidecars (manual API, ManifestReader) still reads back as a real Raw.
		attrs.setdefault("SamplingFrequency", float(item.raw.info["sfreq"]))
		attrs.setdefault("ch_names", list(item.raw.ch_names))
		layout = recording_layout(item.raw, self._codec)
		if layout.scale is not None and layout.offset is not None:
			attrs["data_scale"] = layout.scale.tolist()
			attrs["data_offset"] = layout.offset.tolist()
		attrs["data_note"] = layout.note
		shape = (len(item.raw.ch_names), int(item.raw.n_times))
		chunks = recording_chunks(item.raw, layout, self._codec)
		compressors = self._compressors("Recording", layout.dtype, chunks,
										sample=lambda: middle_chunk(item.raw, self._codec))
		return attrs, layout, shape, chunks, compressors

	def _add_new_recording(self, group: zarr.Group, path: tuple[str, ...], item: Recording) -> None:
		"""A recording whose group holds no data yet (its sidecar tables may already be
		there): attrs, annotations table, array and every chunk in one zarr round trip,
		where the general path took ~16 (each ~1-3 ms on icechunk; most BIDS recordings
		are small)."""
		log_progress("Writing recording", f"{group.path}/data", f"{len(item.raw.ch_names)} ch, {item.raw.n_times} samples")
		attrs, layout, shape, chunks, compressors = self._recording_plan(item)
		events = _annotations_df(item.raw)
		if events is not None and self._has(path, "events"):
			events = None  # a sidecar events table, already written, wins
		metadata = util.array_metadata(shape, layout.dtype, chunks, self._codec.filters, compressors)  # sync: not in the coroutine
		encoder = util._blosc_encoder(metadata)

		pending, self._pending = self._pending, []

		async def write() -> Any:
			target = group._async_group
			if pending:  # its folders (and earlier small nodes), before the recording itself
				await util.set_many(target.store_path.store, pending)
			await target.update_attributes(attrs)
			if events is not None:
				await util.create_table_async(target, "events", events, {"source": "mne.Raw.annotations"},
											  array_options=self._table_array_options, absent=True)
			array = await util.new_array_async(target, "data", metadata)
			for start in range(0, shape[-1], chunks[-1]):
				stop = min(shape[-1], start + chunks[-1])
				await util.write_block_async(array, start, encode_recording(item.raw, start, stop, layout, self._codec),
											 encoder)
			return array

		array = zarr.Array(group._sync(write()))
		self._made(path, "data")
		if events is not None:
			self._made(path, "events")
		self._capture(group.path, "recording",
					  {k: v for k, v in group.attrs.asdict().items() if k != "_neurozarr_item_type"}, node=array)
		record_write(array.nbytes)
		log_mem()

	def _add_annotations(self, group: zarr.Group, raw: "mne.io.BaseRaw") -> None:
		"""An mne.Raw's annotations are events -- store them the way BIDS does, as an
		events table, so they survive a round trip. A recording that already has an
		events table from its BIDS sidecar keeps that one."""
		events = _annotations_df(raw)
		if events is None or "events" in group:
			return
		util.create_table(group, "events", events, {"source": "mne.Raw.annotations"},
						  array_options=self._table_array_options)

	def add_table(self, item: Table, existing: ExistingPolicy = ExistingPolicy.ERROR) -> bool:
		path = self._path_for(item.prefix, item.entities)
		group = self._require(path)
		exists = self._has(path, item.name)
		# _add_annotations's "events" placeholder is only ever a fallback for when a
		# recording has no real BIDS events sidecar -- a genuine sidecar table always
		# supersedes it, regardless of which of the two this ingest processed first
		# (a reader streams a recording and its sidecar as independent items with no
		# ordering guarantee between them).
		if exists:
			self._flush()
		if item.name == "events" and exists \
				and group[item.name].attrs.get("source") == "mne.Raw.annotations":
			existing = ExistingPolicy.REPLACE
		if not self._should_write(group, item.name, existing, exists):
			return False
		log_progress("Writing table", f"{group.path}/{item.name}", f"{len(item.df.columns)} cols, {len(item.df)} rows")
		attributes, writes = util.table_node(item.df, item.meta)
		if not writes and not exists:  # small and new: one node, batched with the others
			key, document, _ = util.group_node(group._async_group, item.name, attributes)
			self._stage(key, document)
		else:
			self._flush()
			group._sync(util.write_table_async(group._async_group, item.name, attributes, writes,
											   self._table_array_options, absent=not exists))
		self._made(path, item.name)
		self._capture(_join(group.path, item.name), "table", dict(item.meta), columns=[str(c) for c in item.df.columns])
		record_write(int(item.df.memory_usage(deep=True).sum()))
		log_mem()
		return True

	def add_attrs(self, item: Attrs) -> None:
		self._flush()
		path = tuple(item.path)
		group = self._require(path[:-1]) if path else self.root
		if not path or path in self._groups:
			target: zarr.Group | zarr.Array = self._groups[path]
		elif self._has(path[:-1], path[-1]):
			target = group[item.path[-1]]  # an existing item -- a plain Array included, not just a Group
		else:
			target = self._require(path)
		log_progress("Writing attrs", "/".join(item.path) or "(dataset root)")
		self._flush()  # target may have just been staged: it must land before its attrs
		util.set_attrs(target, item.attrs)
		self._forget("/".join(item.path), below=False)
		log_mem()

	def _resolve(self, path: tuple[str, ...]) -> tuple[zarr.Group, str]:
		self._flush()
		if not path:
			raise ValueError("path must name an item, not the repo root")
		group = self._require(tuple(path[:-1]))
		if not self._has(tuple(path[:-1]), path[-1]):
			raise KeyError(f"no such item: {'/'.join(path)}")
		return group, path[-1]

	def delete(self, path: tuple[str, ...]) -> None:
		group, name = self._resolve(path)
		del group[name]
		self._reset_paths()
		self._forget("/".join(path))
		log_mem()

	def rename(self, path: tuple[str, ...], new_name: str) -> None:
		validate_segment(new_name, "item name")
		group, old_name = self._resolve(path)
		if new_name in group:
			raise FileExistsError(f"{new_name!r} already exists at this location")
		_copy_node(group[old_name], group, new_name)
		del group[old_name]
		self._reset_paths()
		self._forget("/".join(path))
		log_mem()

	def write_samples(self, path: tuple[str, ...], start: int, values: np.ndarray,
					  channels: list[int] | None = None) -> None:
		"""Overwrite part of a stored recording with samples already encoded as stored
		(``RecordingView.write`` encodes and checks them). zarr rewrites only the
		chunks the samples fall in."""
		group, name = self._resolve(path)
		array = cast(zarr.Array, cast(zarr.Group, group[name])["data"])
		window = slice(start, start + values.shape[1])
		if channels is None:
			array[:, window] = values
		else:
			array.set_orthogonal_selection((np.asarray(channels), window), values)
		self._forget("/".join(path), below=False)
		record_write(values.nbytes)
		log_mem()

	def add_external_file(self, item: ExternalFile, existing: ExistingPolicy = ExistingPolicy.ERROR) -> bool:
		group = self._group_for(item.prefix, item.entities)
		self._flush()
		if not self._should_write(group, item.name, existing):
			return False
		log_progress("Writing external file", f"{group.path}/{item.name}", str(item.uri))
		reference = group.create_group(item.name, overwrite=True)
		self._made(self._path_for(item.prefix, item.entities), item.name)
		util.set_attrs(reference, {
			"_neurozarr_item_type": "external_file",
			"uri": util.safe_uri(str(item.uri)),
			"media_type": item.media_type,
			"reader_hint": item.reader_hint,
			**item.meta,
		})
		self._capture(_join(group.path, item.name), "file", dict(item.meta),
					  uri=util.safe_uri(str(item.uri)), media_type=item.media_type)
		log_mem()
		return True

	def add_array(self, item: Array, existing: ExistingPolicy = ExistingPolicy.ERROR) -> bool:
		group = self._group_for(item.prefix, item.entities)
		self._flush()
		if not self._should_write(group, item.name, existing):
			return False
		log_progress("Writing array", f"{group.path}/{item.name}", f"shape {tuple(int(s) for s in item.data.shape)}")
		shape = tuple(int(size) for size in item.data.shape)
		dtype = np.dtype(item.data.dtype)
		chunks = util.chunk_shape(shape, dtype.itemsize,
								  self._codec.chunk_target_bytes, self._codec.max_chunk_samples)
		def sample() -> np.ndarray:
			start = max(0, (shape[-1] - chunks[-1]) // 2)
			return np.asarray(item.data[(slice(None),) * (len(shape) - 1) + (slice(start, start + chunks[-1]),)])
		self._made(self._path_for(item.prefix, item.entities), item.name)
		array = group.create_array(item.name, shape=shape, dtype=dtype, chunks=chunks, overwrite=True,
								   compressors=self._compressors(
									   "File" if item.meta.get("_neurozarr_decoder") == "binary" else "Array",
									   dtype, chunks, sample=sample))
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
		self._capture(_join(group.path, item.name), "array", dict(item.meta), node=array)
		record_write(array.nbytes)
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

	def replay(self, writes: dict[str, bytes], deleted: set[str]) -> None:
		"""Apply what a parallel-ingest worker's Writer wrote into a
		:class:`~neurozarr.util.OverlayStore` over this writer's session.

		Workers' units never write the same node, but each can create the same shared
		folder (a session's ``ieeg/``): such a group keeps the attrs of every worker
		that wrote it, the last one winning a key -- as consecutive ``add_attrs`` would.
		"""
		self._flush()
		prototype = default_buffer_prototype()
		items = []
		# node documents first, parents before children, then chunks: icechunk takes a
		# chunk only for an array it already has (a key the worker rewrote moved to the
		# end of its dict, so write order alone is no guarantee)
		for key in sorted(writes, key=lambda k: (not k.endswith("zarr.json"), k.count("/"))):
			value = writes[key]
			if key.endswith("zarr.json"):
				document = json.loads(value)
				if document.get("node_type") == "group":
					earlier = self._replayed.get(key)
					attrs = {**(earlier or {}), **document.get("attributes", {})}
					if earlier and attrs != document.get("attributes", {}):
						value = json.dumps({**document, "attributes": attrs}).encode()
					self._replayed[key] = attrs
			items.append((key, prototype.buffer.from_bytes(value)))

		async def apply() -> None:
			store = self._root.store
			for key in deleted:  # a replaced item's old chunks, before its new ones land
				await store.delete(key)
			documents = [item for item in items if item[0].endswith("zarr.json")]
			# two calls: within one, icechunk may land a chunk before its array's document
			for part in (documents, items[len(documents):]):
				for start in range(0, len(part), 512):
					await util.set_many(store, part[start:start + 512])

		self._root._sync(apply())
		self._reset_paths()

	@property
	def session(self) -> "icechunk.Session | icechunk.ForkSession":
		self._flush()
		return self._session

	@property
	def base(self) -> str | None:
		return self._base

	def put_blob(self, path: str, blob: bytes) -> None:
		"""Store bytes as one single-chunk uint8 array -- one object, one read."""
		parent, _, name = path.rpartition("/")
		parent_path = tuple(parent.split("/")) if parent else ()
		group = self._require(parent_path)
		self._flush()
		data = np.frombuffer(blob, dtype=np.uint8)
		array = group.create_array(name, shape=data.shape, chunks=data.shape or (1,), dtype=np.uint8, overwrite=True)
		array[...] = data
		self._made(parent_path, name)

	def drop(self, path: str) -> None:
		"""Delete a node if it exists."""
		parent, _, name = path.rpartition("/")
		try:
			group = self.root[parent] if parent else self.root
		except KeyError:
			return
		if isinstance(group, zarr.Group) and name in group:
			del group[name]
			self._reset_paths()

	def save(self, message: str, metadata: dict[str, Any] | None = None,
			 rebuilt: "Callable[[str], bool] | None" = None) -> str | None:
		"""Commit what this writer staged. Nothing staged is left alone -- icechunk
		refuses empty commits, and touching one subject shouldn't fail on the others.

		A fork is merged into a fresh session at the branch tip. Icechunk's merge
		would silently let the staged change win over anything committed since the
		fork was taken (measured: a concurrent write to the same array vanished), so
		that is checked here first: if the branch moved and touched the same arrays
		(the same chunks, or any node's metadata), nothing is merged.
		"""
		self.drifted = False
		self._flush()
		if not self._session.has_uncommitted_changes:
			return None
		if self._repo is None:  # a plain writable session (store creation, migration)
			snapshot = self._session.commit(message, metadata)
		else:
			tip = self._repo.lookup_branch(self._branch)
			self.drifted = self._base is not None and tip != self._base
			if self.drifted:
				assert self._base is not None
				clashes = _clashes(self._repo.diff(from_snapshot_id=self._base, to_snapshot_id=tip),
								   self._session.status())
				# nodes the caller rebuilds from the new tip anyway (the dataset index and
				# manifest) are not a clash
				clashes = [c for c in clashes if not (rebuilt and rebuilt(c))]
				if clashes:
					raise WriteConflictError(
						f"{self._label or 'store'}: {', '.join(clashes[:5])}{' ...' if len(clashes) > 5 else ''} "
						f"changed on {self._branch!r} since this was staged; reset it and add it again")
			session = self._repo.writable_session(self._branch)
			session.merge(cast(icechunk.ForkSession, self._session))
			snapshot = session.commit(message, metadata)
			if self.snapshot_dir is not None and isinstance(self._session, icechunk.ForkSession):
				# ponytail: local stores only; on object storage the fork's snapshot stays
				# until `gc`, until deletion through the store's own API is added with S3
				(self.snapshot_dir / self._session.snapshot_id).unlink(missing_ok=True)
		log_progress("Committed", self._label or "store", f"snapshot {snapshot}")
		log_mem()
		return snapshot
