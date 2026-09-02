from dataclasses import dataclass, field

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
	ses: str
	datatype: str
	extra: dict = field(default_factory=dict)

	def group_name(self) -> str:
		known = [f"{k}-{self.extra[k]}" for k in BIDS_ENTITY_ORDER if k in self.extra]
		unknown = [f"{k}-{v}" for k, v in self.extra.items() if k not in BIDS_ENTITY_ORDER]
		return "_".join(known + unknown)


def parse_entities(stem: str) -> dict:
	"""Split a BIDS filename stem (e.g. 'task-BrainSenseStream_acq-Power_run-3')
	into an entity dict ({'task': 'BrainSenseStream', 'acq': 'Power', 'run': '3'})."""
	result = {}
	for part in stem.split("_"):
		key, sep, value = part.partition("-")
		if sep:
			result[key] = value
	return result
