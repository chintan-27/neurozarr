"""Validate sources without writing, with stable machine-readable diagnostics."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .diagnostics import Severity, ValidationIssue, ValidationReport
from .entities import Entities
from .constraints import validate_metadata, validate_segment
from .errors import NeurozarrError, ValidationError
from .readers.bids import MNE_READABLE_EXTS
from .readers.formats import decoder_for

if TYPE_CHECKING:
	from .items import Reader
	from .readers import ManifestReader

TABLE_EXTS = {".tsv", ".csv"}


def _issue(code: str, severity: Severity, message: str, location: str = "",
			   **context: object) -> ValidationIssue:
	return ValidationIssue(code, severity, message, location, dict(context))


def inspect_manifest(reader: "ManifestReader") -> ValidationReport:
	"""Inspect manifest structure, identifiers, file existence, and formats."""
	report = ValidationReport()
	df = reader.df
	for col, label in ((reader.sub_col, "subject"), (reader.path_col, "file path")):
		if col not in df.columns:
			report.add(_issue("manifest.missing_column", Severity.ERROR,
				f"manifest has no {label} column {col!r}", context_columns=list(map(str, df.columns))))
	if report.errors:
		return report

	for i, row in df.iterrows():
		where = f"row {i}"
		if pd.isna(row[reader.sub_col]):
			report.add(_issue("manifest.empty_subject", Severity.ERROR, "empty subject id", where))
		else:
			try:
				Entities(
					str(row[reader.sub_col]),
					str(row[reader.ses_col]) if pd.notna(row.get(reader.ses_col)) else None,
					str(row[reader.datatype_col]) if pd.notna(row.get(reader.datatype_col)) else None,
					reader._extra_entities(row),
				)
			except ValidationError as exc:
				report.add(_issue("manifest.invalid_entities", Severity.ERROR, str(exc), where))
		if reader.name_col in row and pd.notna(row.get(reader.name_col)):
			try:
				validate_segment(str(row[reader.name_col]), "stored item name")
			except ValidationError as exc:
				report.add(_issue("manifest.invalid_name", Severity.ERROR, str(exc), where))
		try:
			validate_metadata(reader._parse_meta(row), "manifest row metadata")
		except (TypeError, ValueError, json.JSONDecodeError) as exc:
			report.add(_issue("manifest.invalid_metadata", Severity.ERROR, str(exc), where))

		path_value = row[reader.path_col]
		if pd.isna(path_value):
			report.add(_issue("manifest.empty_path", Severity.ERROR, "empty file path", where))
			continue
		path = Path(str(path_value))
		if not path.is_absolute():
			path = reader.base_dir / path
		if not path.exists():
			report.add(_issue("source.not_found", Severity.ERROR, f"file not found: {path}", where,
				path=str(path)))
		elif not path.is_file():
			report.add(_issue("source.not_file", Severity.ERROR, f"not a file: {path}", where, path=str(path)))
		elif path.suffix.lower() not in TABLE_EXTS | MNE_READABLE_EXTS and decoder_for(path) is None:
			report.add(_issue("source.unsupported_format", Severity.WARNING,
				f"no reader for {path.suffix!r} ({path.name}); it will be preserved as an external reference",
				where, path=str(path), extension=path.suffix.lower()))
	return report


def inspect_bids(root_dir: str | Path) -> ValidationReport:
	"""Inspect the structural subset needed before a BIDS conversion."""
	root = Path(root_dir)
	report = ValidationReport()
	if not root.exists():
		report.add(_issue("source.not_found", Severity.ERROR, f"source not found: {root}", str(root)))
		return report
	if not root.is_dir():
		report.add(_issue("source.not_directory", Severity.ERROR, f"not a directory: {root}", str(root)))
		return report

	subjects = [directory for directory in root.glob("sub-*") if directory.is_dir()]
	if not subjects:
		report.add(_issue("bids.no_subjects", Severity.ERROR,
			f"no sub-* directories in {root}; this is not a readable BIDS dataset", str(root)))
	if not (root / "dataset_description.json").exists():
		report.add(_issue("bids.missing_dataset_description", Severity.ERROR,
			"missing required dataset_description.json", str(root)))
	for path in root.rglob("*.json"):
		try:
			value = json.loads(path.read_text())
			validate_metadata(value, f"JSON sidecar {path.name}")
		except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
			report.add(_issue("bids.invalid_json", Severity.ERROR, str(exc), str(path)))

	for subject in subjects:
		try:
			Entities(subject.name)
		except ValidationError as exc:
			report.add(_issue("bids.invalid_subject", Severity.ERROR, str(exc), str(subject)))
		sessions = [directory for directory in subject.glob("ses-*") if directory.is_dir()]
		datatype_dirs = [child for directory in (sessions or [subject])
						 for child in directory.iterdir() if child.is_dir()]
		if not datatype_dirs:
			report.add(_issue("bids.no_datatypes", Severity.ERROR,
				"no datatype directories (ieeg/, beh/, ...) found", subject.name))
		for datatype_dir in datatype_dirs:
			for path in datatype_dir.rglob("*"):
				if (not path.is_file()
						or path.suffix.lower() in ({".json", ".eeg", ".vmrk"} | TABLE_EXTS | MNE_READABLE_EXTS)
						or decoder_for(path) is not None):
					continue
				report.add(_issue("source.unsupported_format", Severity.WARNING,
					f"no reader for {path.suffix!r} ({path.name}); it will be preserved as an external reference",
					str(path), extension=path.suffix.lower()))
	return report


def inspect_source(source: "Reader | str | Path") -> ValidationReport:
	"""Return structured diagnostics for a reader or source path."""
	from .readers import BidsReader, ManifestReader

	if isinstance(source, ManifestReader):
		return inspect_manifest(source)
	if isinstance(source, BidsReader):
		return inspect_bids(source.root_dir)
	if hasattr(source, "read"):
		# Third-party readers validate their own constructor options and yielded
		# item contracts. Consuming the iterator here could perform expensive I/O
		# twice or exhaust a one-shot source.
		return ValidationReport()
	path = Path(str(source))
	if path.suffix.lower() in TABLE_EXTS:
		try:
			reader = ManifestReader(path)
		except (FileNotFoundError, pd.errors.ParserError, ValueError) as exc:
			return ValidationReport([_issue("manifest.unreadable", Severity.ERROR, str(exc), str(path))])
		return inspect_manifest(reader)
	return inspect_bids(path)


def inspect_store(target: object) -> ValidationReport:
	"""Check schema, subject snapshots, and unpublished subject commits."""
	from .repo import Repo
	from .schema import SCHEMA_VERSION

	report = ValidationReport()
	try:
		repo = Repo.open(target, mode="r")  # type: ignore[arg-type]
	except (FileNotFoundError, OSError) as exc:
		report.add(_issue("store.not_found", Severity.ERROR, str(exc), str(target)))
		return report
	except NeurozarrError as exc:
		report.add(_issue("store.invalid_manifest", Severity.ERROR, str(exc), str(target)))
		return report
	try:
		manifest = repo._manifest()
	except Exception as exc:
		report.add(_issue("store.invalid_manifest", Severity.ERROR, str(exc), str(target)))
		return report
	if manifest is None:
		report.add(_issue("store.legacy_schema", Severity.WARNING,
			f"legacy store has no schema manifest; run `neurozarr migrate` to upgrade to schema {SCHEMA_VERSION}",
			str(target)))
		return report
	for sub_id in sorted(manifest.subject_snapshots):
		if sub_id not in manifest.catalog:
			report.add(_issue("store.missing_catalog_entry", Severity.WARNING,
				"subject has no catalog entry; searches will scan its repository", sub_id))
		elif not Repo._catalog_is_current(manifest, sub_id):
			# Not corruption: the summary is simply about a version this dataset no
			# longer publishes, so searches fall back to scanning until the next save.
			report.add(_issue("store.stale_catalog_entry", Severity.WARNING,
				"catalog entry describes a different snapshot than the one published; "
				"searches will scan this repository until it is rewritten", sub_id,
				published_snapshot=manifest.subject_snapshots[sub_id],
				catalog_snapshot=manifest.catalog[sub_id].get("snapshot")))

	for sub_id, snapshot in manifest.subject_snapshots.items():
		if not repo._repo_exists(sub_id):
			report.add(_issue("store.missing_subject_repo", Severity.ERROR,
				f"manifest references missing repository {sub_id}", sub_id, snapshot=snapshot))
			continue
		try:
			subject_repo = repo._icechunk_repo(sub_id)
			subject_repo.lookup_snapshot(snapshot)
			head = subject_repo.lookup_branch("main")
		except Exception as exc:
			report.add(_issue("store.unreadable_snapshot", Severity.ERROR, str(exc), sub_id,
				snapshot=snapshot))
			continue
		if head != snapshot:
			report.add(_issue("store.unpublished_commit", Severity.WARNING,
				"subject branch contains a newer commit not published by the dataset manifest",
				sub_id, published_snapshot=snapshot, branch_head=head))
		try:
			repo.root_of(sub_id)
		except Exception as exc:
			report.add(_issue("store.unreadable_tree", Severity.ERROR, str(exc), sub_id))
	return report


def validate_manifest(reader: "ManifestReader") -> list[str]:
	"""Compatibility wrapper returning manifest diagnostics as strings."""
	return [str(issue) for issue in inspect_manifest(reader)]


def validate_bids(root_dir: str | Path) -> list[str]:
	"""Compatibility wrapper returning BIDS diagnostics as strings."""
	return [str(issue) for issue in inspect_bids(root_dir)]


def validate_source(source: "Reader | str | Path") -> list[str]:
	"""Compatibility wrapper returning source diagnostics as strings."""
	return [str(issue) for issue in inspect_source(source)]
