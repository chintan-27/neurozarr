"""The standardized item stream every Reader yields and every Writer consumes.

A Reader's only job is to turn its source format into a stream of these three
types; a Writer's only job is to place them correctly in a BIDS-shaped Zarr
tree. Write a new Reader by yielding these -- nothing else touches Zarr, so
the output structure can't drift no matter what the source looks like."""

from dataclasses import dataclass, field
from typing import Iterator, Protocol

import mne
import pandas as pd

from .entities import Entities


@dataclass
class Recording:
	"""One continuous signal recording (an mne.Raw), e.g. an EDF/BrainVision file.

	entities: where it belongs in the BIDS tree (sub/ses/datatype + task/run/acq/...).
	raw:      the loaded mne.io.BaseRaw; the Writer reads calibration off it directly.
	meta:     sidecar-style metadata (an ieeg.json's contents, or whatever the
	          source has); merged with what the Writer infers from raw itself.
	prefix:   extra path segments before the entities, e.g. ("derivatives", "my-filter").
	"""
	entities: Entities
	raw: "mne.io.BaseRaw"
	meta: dict = field(default_factory=dict)
	prefix: tuple = ()


@dataclass
class Table:
	"""A row-per-observation table (channels.tsv, events.tsv, a behavioral log, ...).

	name: the table's name within its group ("channels", "events", "table", ...).
	df:   the data; column dtypes are recorded so reads cast back faithfully.
	"""
	entities: Entities
	name: str
	df: pd.DataFrame
	meta: dict = field(default_factory=dict)
	prefix: tuple = ()


@dataclass
class Attrs:
	"""Metadata with no data array of its own -- dataset_description.json,
	participants.tsv rows, a session's sidecar -- attached to the zarr group at
	path (a tuple of path segments, e.g. ("sub-001", "ses-1")). An empty path
	means dataset-wide. A "sub-XXX" segment anywhere in path routes this to
	that subject's own repo (see Repo._route_attrs); no such segment means it
	goes to the shared _dataset repo."""
	path: tuple
	attrs: dict


class Reader(Protocol):
	"""Anything Repo.ingest() can consume: an object whose read() yields
	Recording/Table/Attrs items in any order, lazily or eagerly. See BidsReader
	and ManifestReader for two implementations."""

	def read(self) -> Iterator[Recording | Table | Attrs]: ...
