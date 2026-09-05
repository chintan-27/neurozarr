"""Structured validation and integrity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterator


class Severity(StrEnum):
	"""Severity of a diagnostic issue."""

	ERROR = "error"
	WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
	"""One machine-readable validation result."""

	code: str
	severity: Severity
	message: str
	location: str = ""
	context: dict[str, Any] = field(default_factory=dict)

	def __str__(self) -> str:
		prefix = f"{self.location}: " if self.location else ""
		return f"{prefix}{self.message}"


@dataclass
class ValidationReport:
	"""Collection of validation issues with convenient severity views."""

	issues: list[ValidationIssue] = field(default_factory=list)

	@property
	def errors(self) -> list[ValidationIssue]:
		return [issue for issue in self.issues if issue.severity is Severity.ERROR]

	@property
	def warnings(self) -> list[ValidationIssue]:
		return [issue for issue in self.issues if issue.severity is Severity.WARNING]

	@property
	def ok(self) -> bool:
		return not self.errors

	def add(self, issue: ValidationIssue) -> None:
		self.issues.append(issue)

	def extend(self, issues: Iterator[ValidationIssue] | list[ValidationIssue]) -> None:
		self.issues.extend(issues)

	def __bool__(self) -> bool:
		return bool(self.issues)

	def __iter__(self) -> Iterator[ValidationIssue]:
		return iter(self.issues)

	def __len__(self) -> int:
		return len(self.issues)
