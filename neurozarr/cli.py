import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from .log import set_verbosity
from .readers import BidsReader, ManifestReader, reader_for
from .repo import Repo
from .validate import inspect_source, inspect_store
from .writer import CodecConfig, ExistingPolicy
from .errors import NeurozarrError

TABLE_EXTS = {".tsv", ".csv"}


def _reader_for(source: str, name: str | None = None) -> BidsReader | ManifestReader | Any:
	"""A BIDS folder or a manifest table -- pick the reader by what the source is."""
	return reader_for(source, name)


def _progress(enabled: bool) -> Callable[[Any], Any] | None:
	"""A tqdm wrapper if tqdm is installed and output is wanted, else None."""
	if not enabled:
		return None
	try:
		from tqdm import tqdm  # type: ignore[import-untyped]  # tqdm ships no type information
	except ImportError:
		return None
	return lambda items: tqdm(items, unit="item")


def cmd_convert(args: argparse.Namespace) -> int:
	reader = _reader_for(args.source, args.reader)
	report = inspect_source(reader)
	if report.errors:
		print(f"{args.source}: cannot convert", file=sys.stderr)
		for issue in report.errors:
			print(f"  - [{issue.code}] {issue}", file=sys.stderr)
		if args.force:
			print("--force cannot bypass malformed input; fix the errors above", file=sys.stderr)
		return 1
	for issue in report.warnings:
		print(f"warning [{issue.code}]: {issue}", file=sys.stderr)

	codec = CodecConfig(dtype=args.dtype) if args.dtype else None

	if args.workers and args.workers != 1:
		if args.reader not in (None, "bids"):
			raise ValueError("parallel conversion currently supports only the BIDS reader")
		from .parallel import convert_parallel
		convert_parallel(args.source, args.dest, workers=args.workers, codec=codec,
						 message=args.message, existing=args.existing)
		repo = Repo.open(args.dest)
	else:
		try:
			repo = Repo.open(args.dest, mode="a", codec=codec)
		except FileNotFoundError:
			repo = Repo.create(args.dest, codec=codec)
		repo.ingest(reader, progress=_progress(not args.quiet),
					existing=args.existing)
		repo.save(args.message)

	if not args.quiet:
		print(f"converted {args.source} -> {args.dest} ({len(repo.subjects())} subjects)")
	return 0


def cmd_info(args: argparse.Namespace) -> int:
	repo = Repo.open(args.store)
	subjects = repo.subjects()
	if not subjects:
		print(f"{args.store}: no subjects found", file=sys.stderr)
		return 1
	print(f"{args.store}: {len(subjects)} subject(s)")
	for sub_id in subjects:
		subject = repo.subject(sub_id)
		visits = subject.visits()
		print(f"  {sub_id}: {len(visits)} visit(s), "
			  f"{len(subject.recordings())} recording(s), {len(subject.tables())} table(s), "
			  f"{len(subject.arrays())} array(s), {len(subject.external_files())} external file(s)")
		if args.verbose:
			for ses in visits:
				print(f"      {ses}")
	return 0


def cmd_history(args: argparse.Namespace) -> int:
	repo = Repo.open(args.store)
	sub_id = args.subject or "_dataset"
	print(f"history of {sub_id}:")
	for snapshot_id, message, written_at in repo.history(sub_id):
		print(f"  {str(snapshot_id)[:12]}  {written_at:%Y-%m-%d %H:%M}  {message}")
	tags = repo.tags(sub_id)
	print(f"tags: {', '.join(tags) if tags else '(none)'}")
	return 0


def cmd_validate(args: argparse.Namespace) -> int:
	report = inspect_source(args.source)
	if not report:
		print(f"{args.source}: OK")
		return 0
	if args.json:
		print(json.dumps([{
			"code": issue.code, "severity": issue.severity.value, "message": issue.message,
			"location": issue.location, "context": issue.context,
		} for issue in report], indent=2))
	else:
		print(f"{args.source}: {len(report)} issue(s)")
		for issue in report:
			print(f"  - {issue.severity.value} [{issue.code}] {issue}")
	return 1 if report.errors else 0


def cmd_verify(args: argparse.Namespace) -> int:
	from .verify import verify

	problems = verify(args.source, args.store, sample_limit=args.limit)
	if not problems:
		print(f"{args.store}: matches {args.source}")
		return 0
	print(f"{args.store}: {len(problems)} mismatch(es) against {args.source}")
	for p in problems[:20]:
		print(f"  - {p}")
	if len(problems) > 20:
		print(f"  ... and {len(problems) - 20} more")
	return 1


def cmd_export(args: argparse.Namespace) -> int:
	from .verify import export_bids

	dest = export_bids(args.store, args.dest, subjects=args.subject)
	print(f"exported {args.store} -> {dest}")
	return 0


def cmd_migrate(args: argparse.Namespace) -> int:
	repo = Repo(args.store)
	result = repo.migrate(dry_run=args.dry_run)
	action = "would migrate" if args.dry_run else "migrated"
	if not result["changed"] and not args.dry_run:
		action = "already current"
	print(f"{args.store}: {action}; schema {result['schema_version']}, {len(result['subjects'])} subject(s)")
	return 0


def cmd_doctor(args: argparse.Namespace) -> int:
	report = inspect_store(args.store)
	if not report:
		print(f"{args.store}: OK")
		return 0
	for issue in report:
		print(f"{issue.severity.value} [{issue.code}]: {issue}")
	return 1 if report.errors else 0


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="neurozarr",
		description="Put recordings into a versioned Zarr/Icechunk store, and inspect one.",
	)
	parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
	sub = parser.add_subparsers(dest="command", required=True)

	convert = sub.add_parser("convert", help="convert a manifest or BIDS dataset into a store")
	convert.add_argument("source", help="a manifest .csv/.tsv, or a BIDS directory")
	convert.add_argument("dest", help="output store: a path or s3://, gs://, az:// URI")
	convert.add_argument("-m", "--message", default="convert", help="commit message")
	convert.add_argument("--dtype", choices=["int16", "float16"], help="how to pack sample data")
	convert.add_argument("--reader", help="explicit built-in or installed reader name")
	convert.add_argument("-q", "--quiet", action="store_true", help="no progress or summary")
	convert.add_argument("--force", action="store_true",
						 help="deprecated; malformed input is always rejected")
	convert.add_argument("-j", "--workers", type=int, metavar="N",
						 help="convert N subjects in parallel (BIDS sources only)")
	convert.add_argument("--existing", choices=[policy.value for policy in ExistingPolicy], default="error",
						 help="what to do when an item path already exists (default: error)")
	convert.set_defaults(func=cmd_convert)

	info = sub.add_parser("info", help="show what's in a store")
	info.add_argument("store")
	info.set_defaults(func=cmd_info)

	history = sub.add_parser("history", help="show a store's versions and tags")
	history.add_argument("store")
	history.add_argument("--subject", help="show one subject's internal history (default: global history)")
	history.set_defaults(func=cmd_history)

	validate = sub.add_parser("validate", help="check a source before converting")
	validate.add_argument("source", help="a manifest .csv/.tsv, or a BIDS directory")
	validate.add_argument("--json", action="store_true", help="emit machine-readable diagnostics")
	validate.set_defaults(func=cmd_validate)

	verify = sub.add_parser("verify", help="check a store faithfully matches its source")
	verify.add_argument("source", help="the BIDS directory it was converted from")
	verify.add_argument("store")
	verify.add_argument("--limit", type=int, metavar="N", help="stop after N items (quick check)")
	verify.set_defaults(func=cmd_verify)

	export = sub.add_parser("export", help="write a store back out as a BIDS folder")
	export.add_argument("store")
	export.add_argument("dest", help="output BIDS directory")
	export.add_argument("--subject", action="append", help="export only this subject (repeatable)")
	export.set_defaults(func=cmd_export)

	migrate = sub.add_parser("migrate", help="upgrade a legacy store to the current schema")
	migrate.add_argument("store")
	migrate.add_argument("--dry-run", action="store_true", help="report the migration without writing")
	migrate.set_defaults(func=cmd_migrate)

	doctor = sub.add_parser("doctor", help="check a store's schema and snapshot integrity")
	doctor.add_argument("store")
	doctor.set_defaults(func=cmd_doctor)
	return parser


def main(argv: list[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	if args.verbose:
		set_verbosity(logging.DEBUG)
	try:
		return args.func(args)
	except (NeurozarrError, ValueError, KeyError, FileNotFoundError, FileExistsError) as e:
		print(f"error: {e}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	sys.exit(main())
