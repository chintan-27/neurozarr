"""Per-file format decoders: given one already-located file, decode it into item(s).

Distinct from the reader registry (:mod:`neurozarr.readers.registry`): a *reader*
discovers files across a whole source (a directory tree, a manifest table) and
decides where each one goes; a *format decoder* turns one file that a reader has
already located into item(s). :class:`~neurozarr.ManifestReader` and
:class:`~neurozarr.BidsReader` both consult this registry for a file their own
built-in dispatch (tabular formats, mne-readable signal formats) does not claim,
before falling back to the ``unclaimed`` policy. Adding a decoder here benefits
both, and any other reader that chooses to consult it, without duplicating
file-discovery logic.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..entities import Entities
from ..items import Array, ExternalFile, Recording, Table

# A format decoder turns one file into data, never dataset-wide metadata, so its
# return type excludes Attrs -- unlike a Reader, which may reasonably yield either.
DecodedItem = Recording | Table | ExternalFile | Array
FormatDecoder = Callable[[Path, Entities, str, dict[str, Any]], Iterator[DecodedItem]]

_DECODERS: dict[str, FormatDecoder] = {}


class UnclaimedPolicy(StrEnum):
	"""What to do with a file no format decoder claims.

	``REFERENCE`` (default) preserves today's behavior: the path is recorded as
	an :class:`~neurozarr.ExternalFile`, its contents untouched. ``EMBED`` reads
	the file's raw bytes into the store instead, trading store size for a store
	that no longer depends on the original file staying where it was.
	"""

	REFERENCE = "reference"
	EMBED = "embed"


def register_format(extension: str, decoder: FormatDecoder) -> None:
	"""Register a decoder for one file extension, e.g. ``".nii.gz"``.

	Matched against the full filename, not just its last dot-segment, so a
	compound extension works as given. Registering the same extension again
	replaces the previous decoder -- built-in decoders (NIfTI, when nibabel is
	installed) can be overridden this way.
	"""
	ext = extension.lower()
	if not ext.startswith(".") or len(ext) == 1:
		raise ValueError(f"format extension must include a leading dot, got {extension!r}")
	_DECODERS[ext] = decoder


def decoder_for(path: Path) -> FormatDecoder | None:
	"""The registered decoder claiming this file, or ``None`` if unclaimed."""
	name = path.name.lower()
	for ext, decoder in _DECODERS.items():
		if name.endswith(ext):
			return decoder
	return None


def registered_formats() -> list[str]:
	"""Extensions with a registered decoder."""
	return sorted(_DECODERS)


def decode_embed(path: Path, entities: Entities, name: str, meta: dict[str, Any]) -> Iterator[DecodedItem]:
	"""Read a file's raw bytes into the store as a one-dimensional array of
	bytes, rather than only recording its path. Used when ``unclaimed="embed"``
	is passed to a reader; never applied without that being asked for."""
	data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
	yield Array(entities, name, data, ("byte",),
				meta={**meta, "_neurozarr_decoder": "binary", "original_filename": path.name})


def _seed_builtin_decoders() -> None:
	# find_spec checks importability without importing the (possibly heavy)
	# package, so this costs nothing when nibabel isn't installed.
	if importlib.util.find_spec("nibabel") is not None:
		from .nifti import decode_nifti
		register_format(".nii", decode_nifti)
		register_format(".nii.gz", decode_nifti)


_seed_builtin_decoders()
