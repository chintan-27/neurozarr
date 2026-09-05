"""Versioned metadata describing a complete neurozarr dataset snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from .constraints import validate_metadata
from .entities import Entities

SCHEMA_VERSION = 2
MANIFEST_ATTR = "_neurozarr"


def utc_now() -> str:
	"""Return a stable, timezone-aware timestamp for stored provenance."""
	return datetime.now(timezone.utc).isoformat()


@dataclass
class StoreManifest:
	"""Dataset-level manifest pinning every subject to one snapshot."""

	schema_version: int = SCHEMA_VERSION
	dataset_id: str = field(default_factory=lambda: str(uuid4()))
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)
	created_with: str = "unknown"
	subject_snapshots: dict[str, str] = field(default_factory=dict)
	catalog: dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
			raise ValueError("store manifest schema_version must be an integer")
		try:
			UUID(self.dataset_id)
		except (TypeError, ValueError) as exc:
			raise ValueError("store manifest dataset_id must be a UUID") from exc
		for label, value in (("created_at", self.created_at), ("updated_at", self.updated_at)):
			try:
				parsed = datetime.fromisoformat(value)
			except (TypeError, ValueError) as exc:
				raise ValueError(f"store manifest {label} must be an ISO-8601 timestamp") from exc
			if parsed.tzinfo is None:
				raise ValueError(f"store manifest {label} must include a timezone")
		if not self.created_with:
			raise ValueError("store manifest created_with cannot be empty")
		for sub_id, snapshot in self.subject_snapshots.items():
			Entities(sub_id)
			if not isinstance(snapshot, str) or not snapshot:
				raise ValueError(f"store manifest snapshot for {sub_id} must be a non-empty string")
		validate_metadata(self.catalog, "store catalog")

	@classmethod
	def from_dict(cls, value: dict[str, Any]) -> "StoreManifest":
		"""Parse and validate a manifest stored in Zarr attributes."""
		version = value.get("schema_version")
		if isinstance(version, bool) or not isinstance(version, int):
			raise ValueError("store manifest has no integer schema_version")
		snapshots = value.get("subject_snapshots", {})
		catalog = value.get("catalog", {})
		if not isinstance(snapshots, dict):
			raise ValueError("store manifest subject_snapshots must be an object")
		if not isinstance(catalog, dict):
			raise ValueError("store manifest catalog must be an object")
		return cls(
			schema_version=version,
			dataset_id=str(value.get("dataset_id", "")),
			created_at=str(value.get("created_at", "")),
			updated_at=str(value.get("updated_at", "")),
			created_with=str(value.get("created_with", "unknown")),
			subject_snapshots={str(k): v for k, v in snapshots.items()},
			catalog=catalog,
		)

	def as_dict(self) -> dict[str, Any]:
		"""Return JSON-compatible data for a Zarr attribute."""
		return {
			"schema_version": self.schema_version,
			"dataset_id": self.dataset_id,
			"created_at": self.created_at,
			"updated_at": self.updated_at,
			"created_with": self.created_with,
			"subject_snapshots": dict(sorted(self.subject_snapshots.items())),
			"catalog": self.catalog,
		}
