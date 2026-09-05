"""Public exception hierarchy for neurozarr."""


class NeurozarrError(Exception):
	"""Base class for errors raised by neurozarr."""


class ValidationError(NeurozarrError, ValueError):
	"""Input cannot be represented safely in a neurozarr store."""


class SchemaVersionError(NeurozarrError, RuntimeError):
	"""A store schema is unsupported or needs an explicit migration."""


class StoreIntegrityError(NeurozarrError, RuntimeError):
	"""Committed store metadata is missing or internally inconsistent."""


class WriteConflictError(NeurozarrError, RuntimeError):
	"""A concurrent or duplicate write cannot be resolved safely."""


class UnsupportedFormatError(NeurozarrError, ValueError):
	"""No installed reader can decode a source format."""
