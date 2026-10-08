"""Codec plans: choose how each group of ingested data is compressed by probing a
handful of real files once, then reuse that choice for every array in the group.

A reader builds the plan (``reader.codec_plan(layout)``) before anything is
written; :meth:`Repo.add` hands it to every writer, serial or parallel. The
writer never decides -- it looks the codec up here.

``probe_level`` sets how finely data is grouped -- one codec decision per:

====  =============================================================
0     nothing: zarr's default codec, no probing
1     item type (Recording, Table, Array, File)
2     item type x stored dtype (the default)
3     item type x stored dtype x chunk shape
4     each array, probed on its own middle chunk as it is written
====  =============================================================

Every level is built only from properties every stored array has, whatever the
data. A group the sample never saw falls back to the next coarser level, then to
zarr's default. Strings and embedded files are never probed: shuffle means
nothing for strings, and embedded files are often already compressed.

Measured on real samples from five datasets (BRAVO, CHB-MIT, RNS, Pediatric iEEG
int16; HBN EEG float32), against each recording compressed whole with its real
chunk shape:

- The shuffle mode is the choice that matters (the wrong one costs 30-80%), and
  its winner differs by dataset -- so it is probed. Compressor and level only set
  the speed/size trade-off, so ``optimize`` fixes them.
- One decision per item type x dtype (level 2) came within 0.1-2.3% of probing
  every recording (level 4). Each dataset has only ~5 such groups.
- A probe on one real chunk picked the true best every time; 256 KiB samples
  misled (CHB-MIT: 6.9% bigger), so probes use a real middle chunk.
- The shuffle winner at zstd level 5 is the winner at level 9 on all five, so
  "size" probes at the cheaper level.
"""

import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Literal

import numpy as np
import zarr
from zarr.codecs import BloscCodec, ZstdCodec
from zarr.storage import MemoryStore

from .errors import ValidationError
from .items import Array, Item, Recording, Table

OPTIMIZE = ("speed", "balanced", "size")
PROBE_LEVELS = (0, 1, 2, 3, 4)
Shuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
_SHUFFLES: tuple[Shuffle, ...] = ("noshuffle", "shuffle", "bitshuffle")
# compressor and level per optimize -- from the 52-config sweep: lz4 writes ~2x and
# reads ~1.5x faster than zstd-1; zstd-5 is within a few % of zstd-9's size at
# 10-40x its write speed; zstd-9 (Blosc's top level) is the smallest core codec.
_WRITE: dict[str, tuple[Literal["lz4", "zstd"], int]] = {"speed": ("lz4", 1), "balanced": ("zstd", 5), "size": ("zstd", 9)}
_PROBE: dict[str, tuple[Literal["lz4", "zstd"], int]] = {"speed": ("lz4", 1), "balanced": ("zstd", 5), "size": ("zstd", 5)}
_ZSTD_LEVEL = {"speed": 1, "balanced": 5, "size": 19}  # strings and embedded files
# ponytail: stop reading a sampled subject after opening this many recordings/arrays
# -- enough to see every group at levels 1-2. A group seen only past it falls back a level.
_ITEMS_PER_SUBJECT = 50
# ...and move on from a subject after this many opened in a row that added nothing
_IDLE_ITEMS = 3


def check_plan_options(optimize: str, probe_level: int) -> None:
	if optimize not in OPTIMIZE:
		raise ValidationError(f"optimize must be one of {OPTIMIZE}, got {optimize!r}")
	if isinstance(probe_level, bool) or probe_level not in PROBE_LEVELS:
		raise ValidationError(f"probe_level must be one of {PROBE_LEVELS}, got {probe_level!r}")


def _blosc(cname: Literal["lz4", "zstd"], level: int, shuffle: Shuffle, itemsize: int) -> list[Any]:
	return [BloscCodec(cname=cname, clevel=level, shuffle=shuffle, typesize=itemsize)]


def _stored_bytes(sample: np.ndarray, chain: list[Any]) -> int:
	array = zarr.create_array(MemoryStore(), shape=sample.shape, chunks=sample.shape or (1,),
							  dtype=sample.dtype, compressors=chain)
	array[...] = sample
	return int(array.nbytes_stored())


def _unprobed(item_type: str, dtype: np.dtype) -> bool:
	return item_type == "File" or dtype.kind in "OSUT"


def probe(sample: np.ndarray, optimize: str = "balanced") -> tuple[list[Any], dict[str, int]]:
	"""The codec for data like ``sample``, and the stored bytes of each shuffle mode tried.

	Numeric data tries no, byte and bit shuffle and keeps the smallest, at the
	compressor and level ``optimize`` names; strings get zstd without probing.
	"""
	check_plan_options(optimize, 2)
	if sample.dtype.kind in "OSUT":
		return [ZstdCodec(level=_ZSTD_LEVEL[optimize])], {}
	cname, level = _PROBE[optimize]
	sizes: dict[str, int] = {mode: _stored_bytes(sample, _blosc(cname, level, mode, sample.dtype.itemsize))
							 for mode in _SHUFFLES}
	cname, level = _WRITE[optimize]
	return _blosc(cname, level, min(_SHUFFLES, key=lambda m: sizes[m]), sample.dtype.itemsize), sizes


def group_keys(item_type: str, dtype: Any, chunks: tuple[int, ...]) -> list[str]:
	"""The plan keys of levels 1, 2 and 3: ``Recording``, ``Recording/int16``,
	``Recording/int16/4x65536``."""
	dtype_key = f"{item_type}/{np.dtype(dtype)}"
	return [item_type, dtype_key, f"{dtype_key}/{'x'.join(str(int(c)) for c in chunks)}"]


@dataclass
class CodecPlan:
	"""Which codec each group of data gets, and the probe results behind each choice.

	Attributes
	----------
	optimize : {"speed", "balanced", "size"}
	probe_level : {0, 1, 2, 3, 4}
		See :mod:`neurozarr.codecs`.
	groups : dict
		Plan key (see :func:`group_keys`) to the chosen shuffle mode.
	evidence : dict
		Plan key to ``{"files": n, "bytes": {shuffle mode: stored bytes}}``.
	"""

	optimize: str = "balanced"
	probe_level: int = 2
	groups: dict[str, Shuffle] = field(default_factory=dict)
	evidence: dict[str, dict[str, Any]] = field(default_factory=dict)

	def compressors(self, item_type: str, dtype: Any, chunks: tuple[int, ...],
					sample: Callable[[], np.ndarray] | None = None) -> Any:
		"""The codec for one array about to be written; zarr's default (``"auto"``)
		at level 0 or for data the plan never saw."""
		dtype = np.dtype(dtype)
		if self.probe_level == 0:
			return "auto"
		if _unprobed(item_type, dtype):
			return [ZstdCodec(level=_ZSTD_LEVEL[self.optimize])]
		if self.probe_level == 4 and sample is not None:
			return probe(sample(), self.optimize)[0]
		cname, level = _WRITE[self.optimize]
		for key in reversed(group_keys(item_type, dtype, chunks)[:min(self.probe_level, 3)]):
			if key in self.groups:
				return _blosc(cname, level, self.groups[key], dtype.itemsize)
		return "auto"

	def as_dict(self) -> dict[str, Any]:
		"""JSON-safe, for the dataset metadata."""
		cname, level = _WRITE[self.optimize]
		return {"optimize": self.optimize, "probe_level": self.probe_level,
				"compressor": {"blosc": cname, "clevel": level}, "shuffle": dict(self.groups),
				"evidence": self.evidence}


def _item_type(item: Item) -> str:
	if isinstance(item, Array) and item.meta.get("_neurozarr_decoder") == "binary":
		return "File"
	return type(item).__name__


def _samples(item: Item, layout: Any) -> Iterator[tuple[str, np.dtype, tuple[int, ...], np.ndarray]]:
	"""(item type, stored dtype, chunk shape, one chunk to probe) for every array an
	item will be stored as -- computed exactly as the writer will."""
	from . import util
	from .writer import middle_chunk, recording_chunks, recording_layout
	if isinstance(item, Recording):
		storage = recording_layout(item.raw, layout)
		yield "Recording", storage.dtype, recording_chunks(item.raw, storage, layout), middle_chunk(item.raw, layout)
	elif isinstance(item, Table):
		for _, data in util.table_arrays(item.df)[0]:
			chunks = util.chunk_shape(data.shape, data.dtype.itemsize, layout.chunk_target_bytes, layout.max_chunk_samples)
			yield "Table", data.dtype, chunks, data[tuple(slice(0, c) for c in chunks)]
	elif isinstance(item, Array):
		shape = tuple(int(s) for s in item.data.shape)
		dtype = np.dtype(item.data.dtype)
		chunks = util.chunk_shape(shape, dtype.itemsize, layout.chunk_target_bytes, layout.max_chunk_samples)
		start = max(0, (shape[-1] - chunks[-1]) // 2)
		window = (slice(None),) * (len(shape) - 1) + (slice(start, start + chunks[-1]),)
		yield _item_type(item), dtype, chunks, np.asarray(item.data[window])


class _Planner:
	"""Accumulates probe results per plan key, up to a quota of arrays per key."""

	def __init__(self, layout: Any, optimize: str, probe_level: int, probe_files: int):
		self.layout, self.optimize, self.level, self.quota = layout, optimize, probe_level, probe_files
		self.totals: dict[str, Counter[str]] = {}
		self.files: Counter[str] = Counter()

	def add(self, item: Item, taken: Counter[str] | None = None, cap: int | None = None) -> bool:
		"""Probe ``item`` for every key still under quota -- and, when ``taken`` and
		``cap`` are given, under ``cap`` probes per key from this source too. True
		if it added anything."""
		try:
			samples = list(_samples(item, self.layout))
		except ValidationError:
			return False  # can't be encoded; the write will say so
		added = False
		for item_type, dtype, chunks, data in samples:
			if _unprobed(item_type, dtype):
				continue
			keys = [k for k in group_keys(item_type, dtype, chunks)[:self.level]
					if self.files[k] < self.quota and (taken is None or cap is None or taken[k] < cap)]
			if not keys:
				continue
			sizes = probe(data, self.optimize)[1]
			for key in keys:
				self.files[key] += 1
				if taken is not None:
					taken[key] += 1
				self.totals.setdefault(key, Counter()).update(sizes)
			added = True
		return added

	def plan(self) -> CodecPlan:
		plan = CodecPlan(optimize=self.optimize, probe_level=self.level)
		for key, sizes in self.totals.items():
			plan.groups[key] = min(_SHUFFLES, key=lambda m: sizes[m])
			plan.evidence[key] = {"files": self.files[key], "bytes": dict(sizes)}
		return plan


def plan_from_items(items: Iterable[Item], layout: Any = None, optimize: str = "balanced",
					probe_level: int = 2, probe_files: int = 8) -> CodecPlan:
	"""Build a plan from sample items -- for custom readers that pick their own sample.

	Probes up to ``probe_files`` arrays per plan key; each key gets the shuffle mode
	with the fewest stored bytes summed over its probes. Items that can't be encoded
	(e.g. out of int16 range) are skipped here and fail later, at write time.
	"""
	from .writer import CodecConfig
	check_plan_options(optimize, probe_level)
	planner = _Planner(layout or CodecConfig(), optimize, probe_level, probe_files)
	if probe_level not in (0, 4):  # 0: zarr's default; 4: every array probed when written
		for item in items:
			planner.add(item)
	return planner.plan()


def plan_codecs(reader: Any, layout: Any = None, optimize: str = "balanced", probe_level: int = 2,
				probe_files: int = 8) -> CodecPlan:
	"""Build a plan for a reader that can list and split by subject.

	Samples up to ``probe_files`` subjects spread evenly across the dataset, each
	giving at most its share of every group's quota, and moves on from a subject
	once a few items in a row add nothing -- opening a file is what costs, not
	probing it. A reader that can't list its subjects is sampled from the start of
	its stream instead.
	"""
	from .writer import CodecConfig
	check_plan_options(optimize, probe_level)
	subjects = reader.subjects() if hasattr(reader, "subjects") else None
	if probe_level in (0, 4) or not subjects:
		items = () if probe_level in (0, 4) else islice(reader.read(), _ITEMS_PER_SUBJECT * probe_files)
		return plan_from_items(items, layout, optimize, probe_level, probe_files)
	planner = _Planner(layout or CodecConfig(), optimize, probe_level, probe_files)
	step = max(1, math.ceil(len(subjects) / probe_files))
	picked = subjects[::step][:probe_files]
	cap = math.ceil(probe_files / len(picked))
	for sub in picked:
		taken: Counter[str] = Counter()
		idle = opened = 0
		for item in reader.for_subject(sub).read():
			added = isinstance(item, (Recording, Table, Array)) and planner.add(item, taken, cap)
			# only opening recordings and arrays costs; tables and metadata come free, and
			# may well precede the recordings (BRAVO lists its beh/ tables first)
			if isinstance(item, (Recording, Array)):
				opened += 1
				idle = 0 if added else idle + 1
				if idle >= _IDLE_ITEMS or opened >= _ITEMS_PER_SUBJECT:
					break
	return planner.plan()
