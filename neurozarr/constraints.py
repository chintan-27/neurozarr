"""Validation shared by public item and storage APIs."""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import ValidationError

# neurozarr's own dataset-level repository. The leading underscore is what keeps
# it from ever colliding with a BIDS "sub-" label, and is also what the segment
# rules below reject -- so it is the one storage segment exempt from them.
DATASET_REPO = "_dataset"

_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_ENTITY_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
_ENTITY_VALUE = re.compile(r"^[A-Za-z0-9]+$")


def validate_segment(value: str, label: str) -> str:
	"""Require one non-traversing storage path segment."""
	if not isinstance(value, str) or not _SEGMENT.fullmatch(value) or value in (".", ".."):
		raise ValidationError(
			f"invalid {label} {value!r}; use letters, numbers, '.', '_', '+', or '-' without path separators"
		)
	return value


def validate_entity(key: str, value: Any) -> None:
	"""Validate a BIDS-style entity key/value pair."""
	if not isinstance(key, str) or not _ENTITY_KEY.fullmatch(key):
		raise ValidationError(f"invalid entity key {key!r}; entity keys must be alphanumeric and start with a letter")
	if isinstance(value, bool) or not isinstance(value, (str, int)):
		raise ValidationError(f"invalid value for entity {key!r}: {value!r}; expected a string or integer")
	if not _ENTITY_VALUE.fullmatch(str(value)):
		raise ValidationError(f"invalid value for entity {key!r}: {value!r}; BIDS entity values must be alphanumeric")


def validate_metadata(value: dict[str, Any], label: str = "metadata") -> None:
	"""Require JSON-safe metadata and reject NaN/Infinity."""
	if not isinstance(value, dict):
		raise ValidationError(f"{label} must be a dictionary")

	def check_keys(item: Any, path: str) -> None:
		if isinstance(item, dict):
			for key, nested in item.items():
				if not isinstance(key, str):
					raise ValidationError(f"{label} contains a non-string key at {path}: {key!r}")
				check_keys(nested, f"{path}.{key}")
		elif isinstance(item, (list, tuple)):
			for index, nested in enumerate(item):
				check_keys(nested, f"{path}[{index}]")

	check_keys(value, label)
	try:
		json.dumps(value, allow_nan=False)
	except (TypeError, ValueError) as exc:
		raise ValidationError(f"{label} is not valid JSON metadata: {exc}") from exc
