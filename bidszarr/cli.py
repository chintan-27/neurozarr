import argparse
import logging
import sys
from pathlib import Path

from .log import set_verbosity
from .readers import BidsReader, ManifestReader
from .repo import Repo
from .validate import validate_source
from .writer import CodecConfig

TABLE_EXTS = {".tsv", ".csv"}


def _reader_for(source: str):
	"""A BIDS folder or a manifest table -- pick the reader by what the source is."""
	path = Path(source)
	if path.suffix in TABLE_EXTS:
		return ManifestReader(path)
	return BidsReader(path)


def _progress(enabled: bool):
	"""A tqdm wrapper if tqdm is installed and output is wanted, else None."""
	if not enabled:
		return None
	try:
		from tqdm import tqdm
	except ImportError:
		return None
	return lambda items: tqdm(items, unit="item")


def cmd_convert(args) -> int:
	problems = validate_source(args.source)
	fatal = [p for p in problems if "not fatal" not in p]
	if fatal and not args.force:
		print(f"{args.source}: cannot convert", file=sys.stderr)
		for p in fatal:
			print(f"  - {p}", file=sys.stderr)
		print("re-run with --force to convert anyway", file=sys.stderr)
		return 1

	codec = CodecConfig(dtype=args.dtype) if args.dtype else None
	repo = Repo(args.dest, codec=codec)
	repo.ingest(_reader_for(args.source), progress=_progress(not args.quiet))
	repo.save(args.message)

	if not args.quiet:
		print(f"converted {args.source} -> {args.dest} ({len(repo.subjects())} subjects)")
	return 0


def cmd_info(args) -> int:
	repo = Repo(args.store)
	subjects = repo.subjects()
	if not subjects:
		print(f"{args.store}: no subjects found", file=sys.stderr)
		return 1
	print(f"{args.store}: {len(subjects)} subject(s)")
	for sub_id in subjects:
		subject = repo.subject(sub_id)
		visits = subject.visits()
		print(f"  {sub_id}: {len(visits)} visit(s), "
			  f"{len(subject.recordings())} recording(s), {len(subject.tables())} table(s)")
		if args.verbose:
			for ses in visits:
				print(f"      {ses}")
	return 0


def cmd_history(args) -> int:
	repo = Repo(args.store)
	sub_id = args.subject or (repo.subjects() or ["_dataset"])[0]
	print(f"history of {sub_id}:")
	for snapshot_id, message, written_at in repo.history(sub_id):
		print(f"  {str(snapshot_id)[:12]}  {written_at:%Y-%m-%d %H:%M}  {message}")
	tags = repo.tags(sub_id)
	print(f"tags: {', '.join(tags) if tags else '(none)'}")
	return 0


def cmd_validate(args) -> int:
	problems = validate_source(args.source)
	if not problems:
		print(f"{args.source}: OK")
		return 0
	print(f"{args.source}: {len(problems)} problem(s)")
	for p in problems:
		print(f"  - {p}")
	return 1


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="bidszarr",
		description="Convert recordings into a BIDS-structured Zarr/Icechunk store, and inspect one.",
	)
	parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
	sub = parser.add_subparsers(dest="command", required=True)

	convert = sub.add_parser("convert", help="convert a BIDS folder or manifest into a store")
	convert.add_argument("source", help="BIDS directory, or a manifest .csv/.tsv")
	convert.add_argument("dest", help="output store: a path or s3://, gs://, az:// URI")
	convert.add_argument("-m", "--message", default="convert", help="commit message")
	convert.add_argument("--dtype", choices=["int16", "float16"], help="how to pack sample data")
	convert.add_argument("-q", "--quiet", action="store_true", help="no progress or summary")
	convert.add_argument("--force", action="store_true", help="convert despite validation problems")
	convert.set_defaults(func=cmd_convert)

	info = sub.add_parser("info", help="show what's in a store")
	info.add_argument("store")
	info.set_defaults(func=cmd_info)

	history = sub.add_parser("history", help="show a store's versions and tags")
	history.add_argument("store")
	history.add_argument("--subject", help="which subject's repo (default: the first)")
	history.set_defaults(func=cmd_history)

	validate = sub.add_parser("validate", help="check a source before converting")
	validate.add_argument("source", help="BIDS directory, or a manifest .csv/.tsv")
	validate.set_defaults(func=cmd_validate)
	return parser


def main(argv=None) -> int:
	args = build_parser().parse_args(argv)
	if args.verbose:
		set_verbosity(logging.DEBUG)
	try:
		return args.func(args)
	except (ValueError, KeyError, FileNotFoundError) as e:
		print(f"error: {e}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	sys.exit(main())
