import asyncio
import inspect
import json
import queue
import threading
import hashlib
import math
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import icechunk
import numpy as np
import pandas as pd
import zarr

from zarr.codecs import BloscCodec
from zarr.core.array import AsyncArray
from zarr.core.buffer import default_buffer_prototype
from zarr.core.group import AsyncGroup, GroupMetadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.storage import MemoryStore

from .errors import ValidationError


def chunk_shape(shape: tuple[int, ...], itemsize: int, target_bytes: int = 8 * 1024 * 1024,
				max_samples: int = 65536) -> tuple[int, ...]:
	"""Pick a chunk shape, dividing along the last (time) axis only.

	Two limits apply, whichever is smaller: a byte target, which keeps chunks
	reasonable for wide many-channel arrays, and a cap on samples per chunk,
	which is what keeps windowed reads cheap.
	"""
	# The sample cap matters because a byte target alone puts an entire
	# multi-hour 2-channel recording in a single chunk, making a 10 s read cost
	# the whole array. Measured on a 2ch/927k-sample recording: at 65536 samples
	# a 10 s window read 6x faster than a single chunk and full reads ~2x faster,
	# for ~14% more storage; dropping to 16k made full reads 3x slower.
	row_bytes = itemsize
	for dim in shape[:-1]:
		row_bytes *= max(1, dim)
	by_size = max(1, target_bytes // row_bytes)
	chunk_len = max(1, min(shape[-1], by_size, max_samples))
	return tuple(max(1, dim) for dim in shape[:-1]) + (chunk_len,)


def set_attrs(node: zarr.Group | zarr.Array, attrs: dict[str, Any]) -> None:
	"""Merge attrs into a zarr node's existing attributes."""
	node.attrs.put({**node.attrs.asdict(), **attrs})


def _column_arrays(df: pd.DataFrame) -> list[dict[str, Any]]:
	"""Convert every column to its physical numpy representation, unchanged from
	before packing existed -- this is purely the per-column dtype/encoding logic."""
	columns: list[dict[str, Any]] = []
	# one isna() for the whole frame and items() rather than iloc + isna per column:
	# pandas' per-call overhead was most of encoding a typical few-row sidecar table
	masks = df.isna().to_numpy(dtype=bool).reshape(len(df), len(df.columns))
	for position, (_, series) in enumerate(df.items()):
		logical = str(series.dtype)
		mask = masks[:, position]
		categories = None

		if isinstance(series.dtype, pd.CategoricalDtype):
			values = series.cat.codes.to_numpy(dtype=np.int64)
			encoding = "categorical"
			categories = series.cat.categories.tolist()
			try:
				json.dumps(categories, allow_nan=False)
			except (TypeError, ValueError) as exc:
				raise ValidationError(f"table column {series.name!r} has non-JSON categorical labels") from exc
			ordered = bool(series.cat.ordered)
		elif pd.api.types.is_datetime64_any_dtype(series.dtype):
			values = series.to_numpy(dtype="datetime64[ns]").view("int64")
			encoding = "datetime64[ns]"
			ordered = False
		elif pd.api.types.is_timedelta64_dtype(series.dtype):
			values = series.to_numpy(dtype="timedelta64[ns]").view("int64")
			encoding = "timedelta64[ns]"
			ordered = False
		elif pd.api.types.is_integer_dtype(series.dtype):
			integer_dtype: Any = getattr(series.dtype, "numpy_dtype", series.dtype)
			values = series.to_numpy(dtype=integer_dtype, na_value=0)
			encoding = "integer"
			ordered = False
		elif pd.api.types.is_bool_dtype(series.dtype):
			values = series.to_numpy(dtype=bool, na_value=False)
			encoding = "boolean"
			ordered = False
		elif pd.api.types.is_float_dtype(series.dtype):
			float_dtype: Any = getattr(series.dtype, "numpy_dtype", series.dtype)
			values = series.to_numpy(dtype=float_dtype, na_value=np.nan)
			encoding = "float"
			ordered = False
		elif pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
			# infer_dtype checks every non-missing value is a str, in C: a Python
			# isinstance per value was ~40% of encoding BRAVO's ~1,900 sidecar tables
			if pd.api.types.infer_dtype(series, skipna=True) not in ("string", "empty"):
				raise ValidationError(
					f"table column {series.name!r} has unsupported object values; use a concrete scalar dtype"
				)
			# StringDType, not a fixed-width `<U*>` array: zarr maps the latter to
			# FixedLengthUTF32, a Zarr V3 extension with no ratified spec yet (and
			# a real per-write UnstableSpecificationWarning) -- StringDType maps to
			# zarr's core "string" type instead, spec'd and warning-free.
			values = np.array(series.to_numpy(dtype=object, na_value=""), dtype=np.dtypes.StringDType())
			encoding = "string"
			ordered = False
		else:
			raise ValidationError(f"table column {series.name!r} has unsupported dtype {series.dtype}")

		columns.append({
			"name": str(df.columns[position]), "values": np.asarray(values), "mask": mask,
			"logical": logical, "encoding": encoding, "categories": categories, "ordered": ordered,
		})
	return columns


def table_arrays(df: pd.DataFrame) -> tuple[list[tuple[str, np.ndarray]], list[dict[str, Any] | None]]:
	"""The arrays a table is stored as, and its column schema.

	Columns that share a physical dtype and nullability are packed into one
	shared 2D array instead of each getting its own -- icechunk has a real
	per-array bookkeeping cost that dominates writing many small columns
	(measured ~4x faster on a realistic BIDS sidecar table shape). Decoding a
	column only ever looks at its own schema entry and never depends on what,
	if anything, it was packed alongside, so this changes nothing about what
	comes back on read.
	"""
	columns = _column_arrays(df)

	groups: dict[tuple[str, bool], list[int]] = {}
	for i, col in enumerate(columns):
		if col["encoding"] == "categorical":
			continue  # each has its own categories/ordered metadata; never packed
		key = ("string", bool(col["mask"].any())) if col["encoding"] == "string" \
			else (str(col["values"].dtype), bool(col["mask"].any()))
		groups.setdefault(key, []).append(i)

	schema: list[dict[str, Any] | None] = [None] * len(columns)
	writes: list[tuple[str, np.ndarray]] = []
	packed: set[int] = set()
	for group_index, (key, indices) in enumerate(g for g in groups.items() if len(g[1]) >= 2):
		has_mask = key[1]
		packed.update(indices)
		array_name = f"g{group_index:06d}"
		cols = [columns[i] for i in indices]
		writes.append((array_name, np.array([c["values"] for c in cols])))
		if has_mask:
			writes.append((f"{array_name}__mask", np.array([c["mask"] for c in cols])))
		for row, i in enumerate(indices):
			c = columns[i]
			schema[i] = {
				"array": array_name, "row": row, "name": c["name"], "dtype": c["logical"],
				"encoding": c["encoding"], "nullable": bool(c["mask"].any()),
			}

	for i, c in enumerate(columns):
		if i in packed:
			continue
		internal = f"c{i:06d}"
		writes.append((internal, c["values"]))
		if c["mask"].any():
			writes.append((f"{internal}__mask", c["mask"]))
		entry: dict[str, Any] = {
			"id": internal, "name": c["name"], "dtype": c["logical"],
			"encoding": c["encoding"], "nullable": bool(c["mask"].any()),
		}
		if c["encoding"] == "categorical":
			entry["categories"] = c["categories"]
			entry["ordered"] = c["ordered"]
		schema[i] = entry
	return writes, schema


# ---------------------------------------------------------------------------
# Direct node writes. zarr-python spends ~5 ms of its own machinery creating a group
# and a small array (measured on an in-memory store; icechunk adds little), which was
# most of a BIDS ingest: ~16 ms per item on BRAVO, ~2,500 items. A new node is just its
# zarr.json plus its encoded chunks, so these write exactly those objects to the store
# (0.4-0.5 ms per item). The metadata comes from zarr's own classes and an array's from
# a zarr-created template, so what lands is what zarr would have written; only nodes
# known not to exist yet go this way.
# ponytail: builds zarr's AsyncGroup/AsyncArray from metadata directly (public classes,
# undocumented constructors); revisit if zarr adds a bulk or low-overhead create API.
# ---------------------------------------------------------------------------

_PROTO = default_buffer_prototype()
_ARRAY_TEMPLATES: dict[str, dict[str, Any]] = {}


def _codec_key(codecs: Any) -> Any:
	if codecs is None or isinstance(codecs, str):
		return codecs
	return [c.to_dict() for c in (codecs if isinstance(codecs, (list, tuple)) else [codecs])]


def array_metadata(shape: tuple[int, ...], dtype: Any, chunks: tuple[int, ...], filters: Any,
				   compressors: Any) -> ArrayV3Metadata:
	"""The metadata zarr.create_array would write for this array, without its overhead."""
	key = json.dumps([np.dtype(dtype).str, len(shape), _codec_key(filters), _codec_key(compressors)])
	template = _ARRAY_TEMPLATES.get(key)
	if template is None:
		ones = (1,) * len(shape)
		template = zarr.create_array(MemoryStore(), shape=ones, chunks=ones, dtype=dtype, filters=filters,
									 compressors=compressors).metadata.to_dict()
		_ARRAY_TEMPLATES[key] = template
	return ArrayV3Metadata.from_dict({**template, "shape": list(shape),
									  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}}})


def group_node(parent: AsyncGroup, name: str, attributes: dict[str, Any] | None = None) -> tuple[str, Any, AsyncGroup]:
	"""(store key, zarr.json bytes, group) for a new group -- nothing written yet, so a
	writer can batch many new nodes into one store call (see :func:`set_many`)."""
	metadata = GroupMetadata(attributes=attributes or {})
	path = parent.store_path / name
	return f"{path.path}/zarr.json", metadata.to_buffer_dict(_PROTO)["zarr.json"], AsyncGroup(metadata, path)


async def new_group_async(parent: AsyncGroup, name: str, attributes: dict[str, Any] | None = None) -> AsyncGroup:
	"""Create a group the caller knows doesn't exist yet."""
	key, document, group = group_node(parent, name, attributes)
	await parent.store_path.store.set(key, document)
	return group


async def set_many(store: Any, items: list[tuple[str, Any]]) -> None:
	"""Write ``(key, bytes)`` pairs in order: one call into icechunk where it can take a
	batch, else one set each."""
	# ponytail: IcechunkStore._set_many is private (icechunk 2.1); the fallback keeps
	# plain zarr stores and a future icechunk without it working, just slower.
	batch = getattr(store, "_set_many", None)
	if batch is not None:
		result = batch(items)
		if inspect.isawaitable(result):
			await result
		return
	for key, value in items:
		await store.set(key, value)


def _blosc_encoder(metadata: ArrayV3Metadata) -> Any:
	"""numcodecs Blosc giving byte-identical chunks to this array's codec chain, if the
	chain is just bytes + one Blosc codec; else None (zarr's own pipeline encodes)."""
	from numcodecs import Blosc  # type: ignore[import-untyped]
	codecs = metadata.codecs
	if len(codecs) != 2 or not isinstance(codecs[1], BloscCodec):
		return None
	if len(codecs) != 2 or not isinstance(codecs[1], BloscCodec) or metadata.dtype.to_native_dtype().kind not in "biuf":
		return None
	blosc = codecs[1]
	shuffle = {"noshuffle": Blosc.NOSHUFFLE, "shuffle": Blosc.SHUFFLE, "bitshuffle": Blosc.BITSHUFFLE}[
		str(getattr(blosc.shuffle, "value", blosc.shuffle))]
	# numcodecs takes the typesize from the buffer's itemsize; zarr writes the codec's own
	itemsize = metadata.dtype.to_native_dtype().itemsize
	typesize = blosc.typesize or itemsize
	if typesize not in (1, 2, 4, 8) or (math.prod(metadata.chunks) * itemsize) % typesize:
		return None
	return Blosc(cname=str(getattr(blosc.cname, "value", blosc.cname)), clevel=blosc.clevel, shuffle=shuffle,
				 blocksize=blosc.blocksize), np.dtype(f"u{typesize}")


async def new_array_async(group: AsyncGroup, name: str, metadata: ArrayV3Metadata) -> AsyncArray[Any]:
	"""Create an array the caller knows doesn't exist yet; fill it with write_block_async."""
	path = group.store_path / name
	await (path / "zarr.json").set(metadata.to_buffer_dict(_PROTO)["zarr.json"])
	return AsyncArray(metadata, path)


async def write_block_async(array: AsyncArray[Any], start: int, block: np.ndarray, encoder: Any) -> None:
	"""Write ``block`` at ``start`` along the last axis, one whole chunk (the last may be
	short); every other axis spans one chunk. Same stored bytes as array[...] = block:
	padded to the chunk shape with the fill value, and skipped if all fill value."""
	chunks = array.metadata.chunks
	if encoder is None:
		selection = (slice(None),) * (block.ndim - 1) + (slice(start, start + block.shape[-1]),)
		await array.setitem(selection, block)
		return
	fill = array.metadata.fill_value
	if np.all(block == fill):
		return
	if block.shape != tuple(chunks):
		padded = np.full(chunks, fill, dtype=block.dtype)
		padded[tuple(slice(0, n) for n in block.shape)] = block
		block = padded
	key = array.metadata.encode_chunk_key((0,) * (block.ndim - 1) + (start // chunks[-1],))
	blosc, unit = encoder
	raw = np.ascontiguousarray(block, dtype=block.dtype.newbyteorder("<")).reshape(-1).view(np.uint8)
	await (array.store_path / key).set(_PROTO.buffer.from_bytes(blosc.encode(raw.view(unit))))


# Tables whose values fit in this many bytes of JSON are stored inline in the table
# group's own attrs (table_schema_version 4): one object instead of ~5 arrays. zarr-python
# spends ~2-5 ms per array created, so BRAVO's ~1,900 small sidecar/events tables took
# 40 s of a 96 s ingest as arrays; inline they cost ~1 ms each, and a remote table read
# is one GET. Bigger tables keep the packed column arrays (schema 3 layout), since attrs
# load with the group whenever it is opened or listed.
INLINE_TABLE_BYTES = 32 * 1024
INLINE_KEY = "_neurozarr_inline"


def _inline_table(df: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
	"""``({"values": [...], "masks": {...}}, schema)`` for a table small enough to store
	inline, else None. Values are the same physical arrays the column layout stores,
	as JSON lists; a column's mask is listed (as row indices) only if it has any."""
	columns = _column_arrays(df)
	values: list[list[Any]] = []
	masks: dict[str, list[int]] = {}
	schema: list[dict[str, Any]] = []
	for i, c in enumerate(columns):
		data = c["values"]
		if data.dtype.kind == "f":
			if not np.isfinite(data[~np.isnan(data)]).all():
				return None  # +-inf has no JSON form
			data = data.astype(object)
			data[np.isnan(c["values"])] = None
		elif data.dtype.kind not in "biuT":
			return None
		values.append(data.tolist())
		if c["mask"].any():
			masks[str(i)] = np.flatnonzero(c["mask"]).tolist()
		entry: dict[str, Any] = {"inline": i, "physical": "str" if data.dtype.kind == "T" else str(c["values"].dtype),
								 "name": c["name"], "dtype": c["logical"], "encoding": c["encoding"],
								 "nullable": bool(c["mask"].any())}
		if c["encoding"] == "categorical":
			entry["categories"], entry["ordered"] = c["categories"], c["ordered"]
		schema.append(entry)
	inline = {"values": values, "masks": masks}
	if len(json.dumps(inline, allow_nan=False)) > INLINE_TABLE_BYTES:
		return None
	return inline, schema


def create_table(group: zarr.Group, name: str, df: pd.DataFrame,
				 extra_attrs: dict[str, Any] | None = None,
				 array_options: Callable[[np.ndarray], dict[str, Any]] | None = None, absent: bool = False) -> None:
	"""Store a DataFrame with a lossless logical dtype schema (see :func:`table_arrays`).
	``array_options`` gives each stored array its ``create_array`` options (chunks,
	codec) from its data; zarr's defaults without it. ``absent``: the caller knows
	nothing exists at ``name`` yet, so it isn't checked."""
	# ponytail: relies on zarr's private _async_group/_sync (the same thing
	# Group.create_array calls internally) since zarr has no public bulk-create
	# API; switch to one if zarr ever adds it.
	group._sync(create_table_async(group._async_group, name, df, extra_attrs, array_options, absent))


async def create_table_async(group: Any, name: str, df: pd.DataFrame,
							 extra_attrs: dict[str, Any] | None = None,
							 array_options: Callable[[np.ndarray], dict[str, Any]] | None = None,
							 absent: bool = False) -> None:
	""":func:`create_table` on a zarr ``AsyncGroup``, for writers batching several nodes
	into one round trip. The group, its attrs and every array go out together:
	separate create_group, create_array and attrs calls each cost a zarr sync (~1-3 ms
	on icechunk), and a typical sidecar table is a few hundred bytes."""
	attributes, writes = table_node(df, extra_attrs)
	await write_table_async(group, name, attributes, writes, array_options, absent)


def table_node(df: pd.DataFrame, extra_attrs: dict[str, Any] | None = None
			   ) -> tuple[dict[str, Any], list[tuple[str, np.ndarray]]]:
	"""A table's group attrs and the arrays it is stored as; no arrays when it is
	small enough to live inline in the attrs (schema 4)."""
	inline = _inline_table(df)
	if inline is not None:
		writes: list[tuple[str, np.ndarray]] = []
		schema: list[dict[str, Any] | None] = list(inline[1])
	else:
		writes, schema = table_arrays(df)
	attributes = {
		"_neurozarr_item_type": "table",
		"table_schema_version": 4 if inline is not None else 3,
		"columns": [str(c) for c in df.columns],
		"schema": schema,
		**(extra_attrs or {}),
		**({INLINE_KEY: inline[0]} if inline is not None else {}),
	}
	return attributes, writes


async def write_table_async(group: Any, name: str, attributes: dict[str, Any], writes: list[tuple[str, np.ndarray]],
							array_options: Callable[[np.ndarray], dict[str, Any]] | None = None,
							absent: bool = False) -> None:
	if not writes and (absent or not await group.contains(name)):
		await new_group_async(group, name, attributes)
		return
	table = await group.create_group(name, overwrite=True, attributes=attributes)
	await asyncio.gather(*(
		table.create_array(array_name, data=data, **(array_options(data) if array_options else {}))
		for array_name, data in writes
	))


async def _listing(list_keys: Callable[[], Any]) -> Any:
	"""A base store's listing; icechunk refuses to list under a node it does not have
	(a folder so far only in the overlay), which means it has nothing there."""
	try:
		async for key in list_keys():
			yield key
	except icechunk.InvalidInputError:
		return


class OverlayStore(MemoryStore):
	"""A store that keeps every write in memory over a read-only base store, which
	reads fall through to. A parallel-ingest worker writes into one over its subject's
	staged session, so it decides "does this exist?" exactly as a serial ingest would;
	the parent then replays :meth:`written` and :attr:`deleted` (``Writer.replay``)."""

	def __init__(self, base: Any) -> None:
		super().__init__()
		self._base = base
		self.deleted: set[str] = set()  # base keys deleted here

	def __eq__(self, other: object) -> bool:
		return self is other

	__hash__ = object.__hash__

	async def get(self, key: str, prototype: Any = None, byte_range: Any = None) -> Any:
		if key in self._store_dict:
			return await super().get(key, prototype, byte_range)
		if key in self.deleted:
			return None
		return await self._base.get(key, prototype or default_buffer_prototype(), byte_range)

	async def exists(self, key: str) -> bool:
		return key in self._store_dict or (key not in self.deleted and await self._base.exists(key))

	async def set(self, key: str, value: Any, byte_range: Any = None) -> None:
		self.deleted.discard(key)
		await super().set(key, value, byte_range)

	async def set_if_not_exists(self, key: str, value: Any) -> None:
		if not await self.exists(key):
			await self.set(key, value)

	async def delete(self, key: str) -> None:
		await super().delete(key)
		if await self._base.exists(key):
			self.deleted.add(key)

	async def list(self) -> Any:
		async for key in self.list_prefix(""):
			yield key

	async def list_prefix(self, prefix: str) -> Any:
		for key in list(self._store_dict):
			if key.startswith(prefix):
				yield key
		async for key in _listing(lambda: self._base.list_prefix(prefix)):
			if key not in self._store_dict and key not in self.deleted:
				yield key

	async def list_dir(self, prefix: str) -> Any:
		names = {name async for name in super().list_dir(prefix)}
		base = prefix.rstrip("/") + "/" if prefix.rstrip("/") else ""
		# ponytail: a base child counts as deleted when its zarr.json is -- what zarr's
		# node deletes leave behind; a store with stray non-node keys would need a scan
		names |= {name async for name in _listing(lambda: self._base.list_dir(prefix))
				  if f"{base}{name}/zarr.json" not in self.deleted}
		for name in names:
			yield name

	def written(self) -> dict[str, bytes]:
		"""Every key written here."""
		return {key: value.to_bytes() for key, value in self._store_dict.items()}


def prefetch(items: Iterable[Any], depth: int = 8) -> Iterator[Any]:
	"""``items``, produced by a background thread up to ``depth`` ahead of the consumer.

	Ingest alternated between reading a source (parsing EDF headers and sidecars on the
	main thread) and writing it (mostly waiting on zarr/icechunk); reading ahead overlaps
	the two. Order is kept, and an exception from ``items`` is raised here at the point
	it would have been reached. If the consumer stops early, the producer stops too.
	"""
	# ponytail: runs the source's generator on another thread, so a reader holding
	# thread-bound handles (e.g. a sqlite connection made on the caller's thread) would
	# fail here; give prefetch an opt-out if such a reader appears.
	feed: queue.Queue[tuple[str, Any]] = queue.Queue(depth)
	stop = threading.Event()

	def put(entry: tuple[str, Any]) -> bool:
		while not stop.is_set():
			try:
				feed.put(entry, timeout=0.1)
				return True
			except queue.Full:
				continue
		return False

	def produce() -> None:
		try:
			for item in items:
				if not put(("item", item)):
					return
			put(("end", None))
		except BaseException as exc:  # handed to the consumer, raised there
			put(("error", exc))

	thread = threading.Thread(target=produce, name="neurozarr-prefetch", daemon=True)
	thread.start()
	try:
		while True:
			kind, value = feed.get()
			if kind == "end":
				return
			if kind == "error":
				raise value
			yield value
	finally:
		stop.set()
		thread.join()


def safe_uri(uri: str) -> str:
	"""Remove credentials, query strings, and fragments from a stored URI."""
	parts = urlsplit(uri)
	if not parts.scheme:
		return uri
	try:
		port = f":{parts.port}" if parts.port else ""
	except ValueError as exc:
		raise ValidationError(f"invalid source URI {uri!r}: {exc}") from exc
	host = f"{parts.hostname or ''}{port}"
	return urlunsplit((parts.scheme, host, parts.path, "", ""))


def load_json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text()) if path.exists() else {}


def clean_nan(meta_data: dict[str, Any]) -> dict[str, Any]:
	"""NaN isn't valid JSON, and zarr attrs are JSON -- swap them for None."""
	def clean_value(value: Any) -> Any:
		if isinstance(value, np.generic):
			return clean_value(value.item())
		if isinstance(value, dict):
			return {str(key): clean_value(item) for key, item in value.items()}
		if isinstance(value, (list, tuple)):
			return [clean_value(item) for item in value]
		if isinstance(value, float) and math.isnan(value):
			return None
		return value

	return {str(key): clean_value(value) for key, value in meta_data.items()}


def source_provenance(path: Path, adapter: str, checksum: bool = False) -> dict[str, Any]:
	"""Return inexpensive, JSON-safe source provenance without credentials."""
	result: dict[str, Any] = {"source": safe_uri(str(path)), "adapter": adapter}
	try:
		stat = path.stat()
	except OSError:
		return result
	result.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
	if checksum and path.is_file():
		digest = hashlib.sha256()
		with path.open("rb") as stream:
			for block in iter(lambda: stream.read(1024 * 1024), b""):
				digest.update(block)
		result["sha256"] = digest.hexdigest()
	return result
