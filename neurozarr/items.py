"""The standardized item stream every Reader yields and every Writer consumes.

A Reader's only job is to turn its source format into a stream of these three
types; a Writer's only job is to place them correctly in the store's tree.
Write a new Reader by yielding these -- nothing else touches Zarr, so the
output structure can't drift no matter what the source looks like."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import mne  # type: ignore[import-untyped]  # mne ships no type information
import numpy as np
import pandas as pd

from .entities import Entities
from .constraints import validate_metadata, validate_segment


def _validate_prefix(prefix: tuple[str, ...]) -> None:
	for part in prefix:
		validate_segment(part, "prefix segment")


@dataclass
class Recording:
	"""One continuous signal recording (an mne.Raw), e.g. an EDF/BrainVision file.

	- ``entities``: where it belongs in the BIDS tree (sub/ses/datatype + task/run/acq/...).
	- ``raw``: the loaded mne.io.BaseRaw; the Writer reads calibration off it directly.
	- ``meta``: sidecar-style metadata (an ieeg.json's contents, or whatever the
	  source has); merged with what the Writer infers from raw itself.
	- ``prefix``: extra path segments before the entities, e.g. ("derivatives", "my-filter").
	"""
	entities: Entities
	raw: "mne.io.BaseRaw"
	meta: dict[str, Any] = field(default_factory=dict)
	prefix: tuple[str, ...] = ()

	def __post_init__(self) -> None:
		_validate_prefix(self.prefix)
		validate_metadata(self.meta, "recording metadata")
		if not isinstance(self.raw, mne.io.BaseRaw):
			raise TypeError(f"Recording.raw must be an mne BaseRaw, got {type(self.raw)!r}")
		if len(self.raw.ch_names) != int(self.raw.info["nchan"]):
			raise ValueError("recording channel names do not match its channel count")
		if float(self.raw.info["sfreq"]) <= 0:
			raise ValueError("recording sampling frequency must be positive")


@dataclass
class Table:
	"""A row-per-observation table (channels.tsv, events.tsv, a behavioral log, ...).

	- ``name``: the table's name within its group ("channels", "events", "table", ...).
	- ``df``: the data; column dtypes are recorded so reads cast back faithfully.
	"""
	entities: Entities
	name: str
	df: pd.DataFrame
	meta: dict[str, Any] = field(default_factory=dict)
	prefix: tuple[str, ...] = ()

	def __post_init__(self) -> None:
		validate_segment(self.name, "table name")
		_validate_prefix(self.prefix)
		validate_metadata(self.meta, "table metadata")
		if not isinstance(self.df, pd.DataFrame):
			raise TypeError(f"Table.df must be a pandas DataFrame, got {type(self.df)!r}")
		if any(not isinstance(name, str) for name in self.df.columns):
			raise ValueError("table column names must be strings")


@dataclass
class Attrs:
	"""Metadata with no data array of its own -- dataset_description.json,
	participants.tsv rows, a session's sidecar -- attached to the zarr group at
	path (a tuple of path segments, e.g. ("sub-001", "ses-1")). An empty path
	means dataset-wide. A "sub-XXX" segment anywhere in path routes this to
	that subject's own repo (see Repo._route_attrs); no such segment means it
	goes to the shared _dataset repo."""
	path: tuple[str, ...]
	attrs: dict[str, Any]

	def __post_init__(self) -> None:
		for part in self.path:
			validate_segment(part, "attribute path segment")
		validate_metadata(self.attrs, "attributes")


@dataclass
class ExternalFile:
	"""A source file preserved by reference because no decoder is installed."""

	entities: Entities
	name: str
	uri: str | Path
	media_type: str | None = None
	reader_hint: str | None = None
	meta: dict[str, Any] = field(default_factory=dict)
	prefix: tuple[str, ...] = ()

	def __post_init__(self) -> None:
		validate_segment(self.name, "external file name")
		_validate_prefix(self.prefix)
		validate_metadata(self.meta, "external file metadata")
		if not str(self.uri):
			raise ValueError("external file URI cannot be empty")


@dataclass
class Array:
	"""A named N-dimensional neuroscience array with dimensions and coordinates."""

	entities: Entities
	name: str
	data: Any
	dims: tuple[str, ...]
	coords: dict[str, Any] = field(default_factory=dict)
	meta: dict[str, Any] = field(default_factory=dict)
	prefix: tuple[str, ...] = ()

	def __post_init__(self) -> None:
		validate_segment(self.name, "array name")
		_validate_prefix(self.prefix)
		validate_metadata(self.meta, "array metadata")
		validate_metadata(self.coords, "array coordinates")
		shape = getattr(self.data, "shape", None)
		if not isinstance(shape, tuple) or not shape:
			raise TypeError("Array.data must expose a non-empty shape tuple")
		if len(self.dims) != len(shape):
			raise ValueError(f"array has {len(shape)} axes but {len(self.dims)} dimension names")
		for dim in self.dims:
			validate_segment(dim, "dimension name")
		if len(set(self.dims)) != len(self.dims):
			raise ValueError("array dimension names must be unique")
		for dim, coordinate in self.coords.items():
			if dim not in self.dims:
				raise ValueError(f"coordinate {dim!r} is not an array dimension")
			if len(coordinate) != shape[self.dims.index(dim)]:
				raise ValueError(f"coordinate {dim!r} length does not match its dimension")


Item = Recording | Table | Attrs | ExternalFile | Array


class Reader(Protocol):
	"""Anything Repo.ingest() can consume: an object whose read() yields
	Recording/Table/Attrs items in any order, lazily or eagerly. See BidsReader
	and ManifestReader for two implementations."""

	def read(self) -> Iterator[Item]: ...
