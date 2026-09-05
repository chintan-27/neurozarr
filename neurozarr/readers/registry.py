"""Reader discovery without granting plugins direct access to stores."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from ..errors import UnsupportedFormatError
from ..items import Reader
from .bids import BidsReader
from .manifest import ManifestReader

ReaderFactory = Callable[..., Reader]
_REGISTERED: dict[str, tuple[ReaderFactory, frozenset[str]]] = {}


def register_reader(name: str, factory: ReaderFactory, *, extensions: tuple[str, ...] = ()) -> None:
	"""Register an in-process reader factory.

	Plugins intended for distribution should expose the same callable through
	the ``neurozarr.readers`` package entry-point group. A factory may publish an
	``extensions`` attribute to participate in automatic file selection.
	"""
	if not name or name in ("bids", "manifest"):
		raise ValueError(f"reader name {name!r} is reserved or empty")
	if not callable(factory):
		raise TypeError("reader factory must be callable")
	normalized = frozenset(ext.lower() for ext in extensions)
	if any(not ext.startswith(".") or len(ext) == 1 for ext in normalized):
		raise ValueError("reader extensions must include a leading dot, for example '.nwb'")
	_REGISTERED[name] = (factory, normalized)


def available_readers() -> list[str]:
	"""List built-in, registered, and installed entry-point reader names."""
	installed = [entry.name for entry in entry_points(group="neurozarr.readers")]
	return sorted({"bids", "manifest", *_REGISTERED, *installed})


def _factory(name: str) -> ReaderFactory:
	if name == "bids":
		return BidsReader
	if name == "manifest":
		return ManifestReader
	if name in _REGISTERED:
		return _REGISTERED[name][0]
	matching = [entry for entry in entry_points(group="neurozarr.readers") if entry.name == name]
	if not matching:
		raise UnsupportedFormatError(
			f"reader {name!r} is not installed (available: {', '.join(available_readers())})"
		)
	if len(matching) > 1:
		raise UnsupportedFormatError(f"multiple installed reader entry points are named {name!r}")
	loaded = matching[0].load()
	if not callable(loaded):
		raise UnsupportedFormatError(f"reader entry point {name!r} is not callable")
	return loaded


def open_reader(name: str, source: str | Path, **options: Any) -> Reader:
	"""Construct one explicitly selected built-in or plugin reader."""
	return _factory(name)(source, **options)


def reader_for(source: str | Path, name: str | None = None, **options: Any) -> Reader:
	"""Select a reader explicitly or by an unambiguous source extension."""
	if name:
		return open_reader(name, source, **options)
	path = Path(source)
	if path.is_dir() or not path.suffix:
		return BidsReader(path, **options)
	if path.suffix.lower() in (".csv", ".tsv"):
		return ManifestReader(path, **options)

	claiming = [registered_name for registered_name, (_, extensions) in _REGISTERED.items()
				if path.suffix.lower() in extensions]
	for entry in entry_points(group="neurozarr.readers"):
		loaded = entry.load()
		extensions = {str(ext).lower() for ext in getattr(loaded, "extensions", ())}
		if path.suffix.lower() in extensions:
			claiming.append(entry.name)
	if len(claiming) == 1:
		return open_reader(claiming[0], source, **options)
	if len(claiming) > 1:
		raise UnsupportedFormatError(
			f"multiple readers claim {path.suffix!r}: {', '.join(sorted(claiming))}; select one explicitly"
		)
	raise UnsupportedFormatError(f"no reader claims source extension {path.suffix!r}")
