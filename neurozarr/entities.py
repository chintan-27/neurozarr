from dataclasses import dataclass, field
from typing import Any

from .constraints import validate_entity, validate_segment
from .errors import ValidationError

# Canonical BIDS entity key order (per the BIDS spec's entity table).
# Only a subset is ever populated for a given datatype; unrecognized keys
# a caller passes are still accepted and appended after these, in insertion order.
BIDS_ENTITY_ORDER = [
	"task", "acq", "ce", "rec", "dir", "run", "mod", "echo",
	"flip", "inv", "mt", "part", "proc", "hemi", "space",
	"split", "recording", "chunk", "atlas", "res", "den",
	"label", "desc",
]


@dataclass(frozen=True)
class Entities:
	sub: str
	ses: str | None = None  # None for session-less BIDS datasets -- a valid, real layout
	datatype: str | None = None
	extra: dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		if not isinstance(self.sub, str) or not self.sub.startswith("sub-"):
			raise ValidationError(f"subject id must include the 'sub-' prefix, got {self.sub!r}")
		validate_segment(self.sub, "subject id")
		validate_entity("sub", self.sub[4:])
		if self.ses is not None:
			if not isinstance(self.ses, str) or not self.ses.startswith("ses-"):
					raise ValidationError(f"session id must include the 'ses-' prefix, got {self.ses!r}")
			validate_segment(self.ses, "session id")
			validate_entity("ses", self.ses[4:])
		if self.datatype is not None:
			validate_segment(self.datatype, "datatype")
		for key, value in self.extra.items():
			validate_entity(key, value)
		object.__setattr__(self, "extra", dict(self.extra))

	def path(self) -> tuple[str, ...]:
		"""(sub[, ses], datatype[, group_name]) -- the Zarr group path for this entity set."""
		parts = [self.sub]
		if self.ses:
			parts.append(self.ses)
		if self.datatype:
			parts.append(self.datatype)
		name = self.group_name()
		if name:
			parts.append(name)
		return tuple(parts)

	def group_name(self) -> str:
		known = [f"{k}-{self.extra[k]}" for k in BIDS_ENTITY_ORDER if k in self.extra]
		unknown = [f"{k}-{self.extra[k]}" for k in sorted(self.extra) if k not in BIDS_ENTITY_ORDER]
		return "_".join(known + unknown)


def parse_entities(stem: str) -> dict[str, str]:
	"""Split a BIDS filename stem (e.g. 'task-BrainSenseStream_acq-Power_run-3')
	into an entity dict ({'task': 'BrainSenseStream', 'acq': 'Power', 'run': '3'}).
	Parts without a '-' (i.e. the suffix) are silently dropped -- use split_stem
	if you also need the suffix."""
	return split_stem(stem)[0]


def split_stem(stem: str) -> tuple[dict[str, str], str]:
	"""Split a full BIDS filename stem into (entities dict, suffix string).
	E.g. 'sub-001_ses-1_task-rest_run-1_channels' ->
	({'sub': '001', 'ses': '1', 'task': 'rest', 'run': '1'}, 'channels')."""
	entities: dict[str, str] = {}
	suffix_parts: list[str] = []
	for part in stem.split("_"):
		key, sep, value = part.partition("-")
		if sep:
			entities[key] = value
		else:
			suffix_parts.append(part)
	return entities, "_".join(suffix_parts)
