"""The ``neurozarr`` command: git's workflow for neural data stores.

Every command works on one store, found like git finds a repository: ``-C <path,
URI or remote name>``, else ``$NEUROZARR_STORE``, else the store the current
directory is in. Each command is a thin shell over the :class:`~neurozarr.Repo`
method of the same name, and shares its staging area (index) and current branch.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

from . import state
from .errors import NeurozarrError
from .log import set_verbosity
from .readers import reader_for
from .repo import Repo
from .validate import inspect_source, inspect_store
from .writer import CodecConfig, ExistingPolicy


# ---- helpers ------------------------------------------------------------------

def _options(pairs: list[str] | None) -> dict[str, Any]:
	"""``-o key=value`` storage options; true/false/numbers parsed, the rest kept as text."""
	options: dict[str, Any] = {}
	for pair in pairs or []:
		key, sep, value = pair.partition("=")
		if not sep:
			raise ValueError(f"storage option must be key=value, got {pair!r}")
		options[key] = json.loads(value) if value in ("true", "false") or value.replace(".", "", 1).isdigit() else value
	return options


def _store(args: argparse.Namespace) -> str:
	if args.store:
		return str(args.store)
	if os.environ.get("NEUROZARR_STORE"):
		return os.environ["NEUROZARR_STORE"]
	found = state.discover()
	if found is None:
		raise ValueError('not a neurozarr store (or any parent directory); use -C <store> or "neurozarr init"')
	return str(found)


def _open(args: argparse.Namespace, mode: str = "r", codec: CodecConfig | None = None) -> Repo:
	return Repo.open(_store(args), mode=mode, codec=codec, **_options(args.option))  # type: ignore[arg-type]


def _emit(args: argparse.Namespace, data: Any, human: Callable[[], Any]) -> None:
	if getattr(args, "json", False):
		print(json.dumps(data, indent=2, default=str))
	else:
		human()


def _plan_options(args: argparse.Namespace) -> dict[str, Any]:
	"""Codec-plan options the user actually gave -- left out otherwise, so a plugin
	reader without them still works."""
	given = {"optimize": args.optimize, "probe_level": args.probe_level, "probe_files": args.probe_files}
	return {key: value for key, value in given.items() if value is not None}


def _progress(enabled: bool, unit: str) -> Callable[[Any], Any] | None:
	"""A tqdm wrapper if tqdm is installed and output is wanted, else None."""
	if not enabled:
		return None
	try:
		from tqdm import tqdm  # type: ignore[import-untyped]  # tqdm ships no type information
	except ImportError:
		return None
	return lambda items: tqdm(items, unit=unit)


def _smart_reader(folders: list[str], answers_file: Path, interactive: bool, **options: Any) -> Any:
	"""Build a SmartReader, asking about its unsure guesses on a terminal and saving the
	answers so the next run doesn't ask again."""
	from .readers import SmartReader
	answers = json.loads(answers_file.read_text()) if answers_file.exists() else {}
	reader = SmartReader(folders, answers=answers, **options)
	r = reader.report
	print(f"smart reader: {r['recordings']} recordings, {r['subjects']} subjects; layouts {r['layouts']}; "
		  f"subject from {r['sub_rules']}; runs from {r['run_rules']}", file=sys.stderr)
	for key in ("unreadable", "duplicates"):
		if r[key]:
			print(f"  {len(r[key])} {key}, e.g. {r[key][0]}", file=sys.stderr)
	if not reader.questions:
		return reader
	if not (interactive and sys.stdin.isatty()):
		for q in reader.questions:
			print(f"  guessing {q['field']}={q['guess']!r} for {q['count']} file(s) under {q['pattern']} "
				  f"({q['why']}); answer in {answers_file}", file=sys.stderr)
		return reader
	changed = False
	for q in reader.questions:
		reply = input(f"{q['why']}: {q['count']} file(s) under {q['pattern']}, e.g. {q['example']}\n"
					  f"  {q['field']} [{q['guess'] if q['guess'] is not None else 'keep as a file reference'}]: ").strip()
		if reply:
			answers.setdefault(q["pattern"], {})[q["field"]] = float(reply) if q["field"] == "sfreq" else reply
			changed = True
	if changed:
		answers_file.write_text(json.dumps(answers, indent=1))
		print(f"answers saved to {answers_file}", file=sys.stderr)
		reader = SmartReader(folders, answers=answers, **options)
	return reader


def _count(summary: dict[str, list[str]]) -> str:
	parts = [f"{len(summary[k])} {k}" for k in ("added", "changed", "removed", "moved") if summary[k]]
	if summary["attrs"]:
		parts.append(f"{len(summary['attrs'])} metadata")
	return ", ".join(parts) or "metadata only"


# ---- store setup ----------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
	target = args.path or "."
	repo = Repo.init(target, codec=CodecConfig(dtype=args.dtype) if args.dtype else None, **_options(args.option))
	print(f"Initialized empty neurozarr store in {repo.target}")
	return 0


def cmd_remote(args: argparse.Namespace) -> int:
	if args.action == "add":
		if not args.name or not args.url:
			raise ValueError("usage: neurozarr remote add NAME URL [-o key=value ...]")
		state.add_remote(args.name, args.url, _options(args.option))
	elif args.action == "remove":
		if not args.name:
			raise ValueError("usage: neurozarr remote remove NAME")
		state.remove_remote(args.name)
	else:
		for name, remote in sorted(state.remotes().items()):
			opts = " ".join(f"{k}={v}" for k, v in remote.get("options", {}).items())
			print(f"{name}\t{remote['url']}{'  ' + opts if opts else ''}")
	return 0


# ---- staging and committing ------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> int:
	options = _plan_options(args)
	if args.reader == "smart" or len(args.sources) > 1:
		reader: Any = _smart_reader(args.sources, Path(args.answers), interactive=not args.quiet, **options)
	else:
		reader = reader_for(args.sources[0], args.reader, **options)
	report = inspect_source(reader)
	for issue in report.warnings:
		print(f"warning [{issue.code}]: {issue}", file=sys.stderr)
	if report.errors:
		print(f"{' '.join(args.sources)}: nothing staged; fix these first", file=sys.stderr)
		for issue in report.errors:
			print(f"  - [{issue.code}] {issue}", file=sys.stderr)
		return 1
	repo = _open(args, "a", CodecConfig(dtype=args.dtype) if args.dtype else None)
	if args.dry_run:
		plan = reader.codec_plan(repo._codec or CodecConfig()) if hasattr(reader, "codec_plan") else None
		print(f"would add {' '.join(args.sources)} with {type(reader).__name__} on {repo.head}")
		if plan is not None:
			print(f"codec plan ({plan.optimize}, probe level {plan.probe_level}): "
				  + (", ".join(f"{k} -> {v}" for k, v in plan.groups.items()) or "zarr default"))
		return 0
	unit = "subject" if args.workers and args.workers > 1 else "item"
	repo.add(reader, progress=_progress(not args.quiet, unit), existing=args.existing, workers=args.workers or 1)
	if not args.quiet:
		staged = repo.status()["staged"]
		subjects = [k for k in staged if k != "_dataset"]
		print(f"staged {sum(len(v['added']) + len(v['changed']) for v in staged.values())} item(s) across "
			  f"{len(subjects)} subject(s) on {repo.head} (use \"neurozarr commit -m <message>\")")
	return 0


def cmd_rm(args: argparse.Namespace) -> int:
	repo = _open(args, "a")
	for path in args.paths:
		repo.rm(path, recursive=args.recursive)
		print(f"rm '{path}'")
	return 0


def cmd_mv(args: argparse.Namespace) -> int:
	_open(args, "a").mv(args.source, args.dest)
	print(f"mv '{args.source}' -> '{args.dest}'")
	return 0


def cmd_status(args: argparse.Namespace) -> int:
	repo = _open(args, "a")
	status = repo.status()

	def human() -> None:
		print(f"On branch {status['branch']}")
		if not status["staged"] and not status["removed_subjects"]:
			print('nothing to commit (use "neurozarr add <source>" to stage data)')
			return
		print("Changes to be committed:")
		for repo_id, summary in status["staged"].items():
			print(f"  {repo_id}: {_count(summary)}")
			for kind in ("added", "changed", "removed", "moved"):
				shown = summary[kind] if args.verbose_items else summary[kind][:5]
				for path in shown:
					print(f"      {kind + ':':9s}{path}")
				if len(summary[kind]) > len(shown):
					print(f"      ... and {len(summary[kind]) - len(shown)} more {kind} (use -l to list all)")
		for sub in status["removed_subjects"]:
			print(f"  {sub}: removed")
		print('(use "neurozarr commit -m <message>" to record, "neurozarr reset" to unstage)')
	_emit(args, status, human)
	return 0


def cmd_reset(args: argparse.Namespace) -> int:
	repo = _open(args, "a")
	if args.hard:
		if args.subjects:
			raise ValueError("reset --hard moves the whole branch; it takes no subjects")
		if not args.yes:
			raise ValueError(f"reset --hard moves {repo.head} to {args.hard} and drops what is staged; "
							 "versions after it stay only on other branches or tags. Add --yes to do it")
		repo.reset_hard(args.hard)
		print(f"HEAD is now at {repo.resolve()[:10]} ({repo.head})")
		return 0
	repo.reset(*args.subjects)
	print(f"unstaged {', '.join(args.subjects) if args.subjects else 'everything'} on {repo.head}")
	return 0


def cmd_commit(args: argparse.Namespace) -> int:
	repo = _open(args, "a")
	status = repo.status()
	version = repo.commit(args.message)
	if version is None:
		print('nothing to commit (use "neurozarr add <source>" to stage data)', file=sys.stderr)
		return 1
	subjects = [k for k in status["staged"] if k != "_dataset"] + status["removed_subjects"]
	print(f"[{repo.head} {version[:10]}] {args.message} -- {len(subjects)} subject(s) changed")
	return 0


# ---- inspecting ------------------------------------------------------------------

def cmd_log(args: argparse.Namespace) -> int:
	repo = _open(args)
	dataset = repo._icechunk_repo("_dataset")
	tags: dict[str, list[str]] = {}
	for name in dataset.list_tags():
		tags.setdefault(dataset.lookup_tag(name), []).append(f"tag: {name}")
	for name in dataset.list_branches():
		tags.setdefault(dataset.lookup_branch(name), []).insert(0, name)
	entries = [{"version": sid, "message": message, "written_at": when.isoformat(), "refs": tags.get(sid, [])}
			   for sid, message, when in repo.log(args.subject)][:args.number or None]

	def human() -> None:
		for e in entries:
			refs = f" ({', '.join(e['refs'])})" if e["refs"] and not args.subject else ""
			if args.oneline:
				print(f"{e['version'][:10]}{refs} {e['message'].splitlines()[0] if e['message'] else ''}")
			else:
				print(f"version {e['version']}{refs}\nDate:   {e['written_at']}\n\n    {e['message']}\n")
	_emit(args, entries, human)
	return 0


def cmd_show(args: argparse.Namespace) -> int:
	shown = _open(args).show(args.version)

	def human() -> None:
		refs = shown["branches"] + [f"tag: {t}" for t in shown["tags"]]
		print(f"version {shown['version']}{' (' + ', '.join(refs) + ')' if refs else ''}")
		print(f"Date:   {shown['written_at']}\n\n    {shown['message']}\n")
		print(f"{shown['subjects']} subject(s)", end="")
		for kind in ("added", "changed", "removed"):
			if shown[f"subjects_{kind}"]:
				print(f"; {kind}: {', '.join(shown[f'subjects_{kind}'])}", end="")
		print()
		plan = shown["codec_plan"]
		if plan:
			detail = plan.get("shuffle") or {"pinned": plan.get("compressors")}
			print(f"codec plan from {plan.get('reader')}: " + ", ".join(f"{k} -> {v}" for k, v in detail.items()))
	_emit(args, shown, human)
	return 0


def cmd_ls(args: argparse.Namespace) -> int:
	entries = _open(args).ls(args.path or "", args.version)
	_emit(args, entries, lambda: [print(f"{e['kind']:13s} {e['path']}") for e in entries])
	return 0


def cmd_find(args: argparse.Namespace) -> int:
	query = dict(term.split("=", 1) for term in args.terms)
	datatype, sub = query.pop("datatype", None), query.pop("sub", None)
	found = [{"path": rec.path, **rec.entities} for rec in _open(args).find(datatype=datatype, sub=sub, **query)]
	_emit(args, found, lambda: [print(f["path"]) for f in found])
	return 0


def cmd_query(args: argparse.Namespace) -> int:
	filters = dict(term.split("=", 1) for term in args.where)
	hits = _open(args).query(args.expression, kind=args.kind, version=args.version, **filters)
	frame = hits.df
	if args.columns:
		frame = frame[[c for c in ["path", *args.columns.split(",")] if c in frame.columns]]
	if args.json:
		print(frame.to_json(orient="records", indent=2))
	elif args.csv:
		print(frame.to_csv(index=False), end="")
	elif args.count:
		print(len(hits))
	else:
		for path in hits.paths:
			print(path)
	return 0


def cmd_tag(args: argparse.Namespace) -> int:
	repo = _open(args, "a" if args.name else "r")
	if args.delete:
		repo._icechunk_repo("_dataset").delete_tag(args.delete)
		print(f"Deleted tag '{args.delete}'")
	elif args.name:
		repo.tag(args.name, version=args.version)
		print(f"tagged {repo.resolve(args.name)[:10]} as '{args.name}'")
	else:
		for name in repo.tags():
			print(name)
	return 0


def _print_diff(result: dict[str, Any], stat: bool) -> None:
	for sub_id, change in result["subjects"].items():
		if change["status"] == "removed":
			print(f"removed  {sub_id}")
			continue
		chunks = f", {change['chunks']} chunk(s)" if change.get("chunks") else ""
		print(f"{change['status']:8s} {sub_id}: {_count(change)}{chunks}")
		if not stat:
			for kind in ("added", "changed", "removed", "moved", "attrs"):
				for path in change[kind]:
					print(f"    {('metadata' if kind == 'attrs' else kind) + ':':10s}{path}")


def cmd_diff(args: argparse.Namespace) -> int:
	result = _open(args, "a" if args.a is None else "r").diff(args.a, args.b, args.subject)
	_emit(args, result, lambda: _print_diff(result, args.stat))
	return 1 if args.exit_code and result["subjects"] else 0


def cmd_branch(args: argparse.Namespace) -> int:
	repo = _open(args, "a")
	if args.delete or args.force_delete:
		name = args.delete or args.force_delete
		if args.force_delete and not args.yes:
			raise ValueError(f"branch -D drops versions only {name!r} has; add --yes to do it")
		repo.delete_branch(name, force=bool(args.force_delete))
		print(f"Deleted branch {name}")
	elif args.name:
		repo.branch(args.name, args.start)
	else:
		for name in repo.branches():
			print(f"{'*' if name == repo.head else ' '} {name}")
	return 0


def cmd_switch(args: argparse.Namespace) -> int:
	repo = _open(args, "a")
	repo.switch(args.name, create=args.create, start=args.start)
	print(f"Switched to {'a new ' if args.create else ''}branch '{args.name}'")
	return 0


def cmd_revert(args: argparse.Namespace) -> int:
	repo = _open(args, "a")
	version = repo.revert(args.version, args.message)
	print(f"[{repo.head} {version[:10]}] {repo.show(version)['message'].splitlines()[0]}")
	return 0


def cmd_merge(args: argparse.Namespace) -> int:
	from .errors import MergeConflictError
	repo = _open(args, "a")
	try:
		version = repo.merge(args.branch, args.message, ours=tuple(args.ours), theirs=tuple(args.theirs))
	except MergeConflictError as conflict:
		print(f"CONFLICT: {conflict}", file=sys.stderr)
		for sub_id, paths in conflict.conflicts.items():
			print(f"  {sub_id}: {', '.join(paths[:5])}{' ...' if len(paths) > 5 else ''}", file=sys.stderr)
		return 1
	print("Already up to date." if version is None else f"[{repo.head} {version[:10]}] merged {args.branch}")
	return 0


def cmd_clone(args: argparse.Namespace) -> int:
	repo = Repo.clone(args.source, args.dest, version=args.version, shallow=args.depth == 1, **_options(args.option))
	print(f"Cloned {args.source} into {repo.target}{' (one version)' if args.depth == 1 else ''}")
	return 0


def cmd_gc(args: argparse.Namespace) -> int:
	import datetime
	repo = _open(args, "a")
	report = repo.gc(datetime.timedelta(days=args.older_than), dry_run=not args.yes)
	total = sum(r["bytes"] for r in report.values())
	objects = sum(r["objects"] for r in report.values())
	verb = "deleted" if args.yes else "would delete"
	print(f"{verb} {objects} unreachable object(s), {total / 1e6:.1f} MB, older than {args.older_than} day(s)"
		  + ("" if args.yes else ' (run "neurozarr gc --yes" to delete)'))
	return 0


# ---- data checks and conversion --------------------------------------------------

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

	store = _store(args)
	problems = verify(args.source, store, sample_limit=args.limit, reader=args.reader)
	if not problems:
		print(f"{store}: matches {args.source}")
		return 0
	print(f"{store}: {len(problems)} mismatch(es) against {args.source}")
	for p in problems[:20]:
		print(f"  - {p}")
	if len(problems) > 20:
		print(f"  ... and {len(problems) - 20} more")
	return 1


def cmd_export(args: argparse.Namespace) -> int:
	from .verify import export_bids

	store = _store(args)
	dest = export_bids(store, args.dest, subjects=args.subject)
	print(f"exported {store} -> {dest}")
	return 0


def cmd_migrate(args: argparse.Namespace) -> int:
	store = _store(args)
	repo = Repo._for_migration(store, **_options(args.option))  # an old schema can't be opened normally
	result = repo.migrate(dry_run=args.dry_run)
	action = "would migrate" if args.dry_run else "migrated"
	if not result["changed"] and not args.dry_run:
		action = "already current"
	print(f"{store}: {action}; schema {result['schema_version']}, {len(result['subjects'])} subject(s)")
	return 0


def cmd_fsck(args: argparse.Namespace) -> int:
	store = _store(args)
	report = inspect_store(store)
	if not report:
		print(f"{store}: OK")
		return 0
	for issue in report:
		print(f"{issue.severity.value} [{issue.code}]: {issue}")
	return 1 if report.errors else 0


# ---- parser ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="neurozarr",
		description="git's workflow for neural data: stage with add, record with commit, "
					"inspect with status/log/show/ls/find.",
	)
	parser.add_argument("-C", dest="store", metavar="STORE",
						help="the store to use: a path, a URI (s3://, gs://, az://, r2://) or a remote name")
	parser.add_argument("-o", "--option", action="append", metavar="KEY=VALUE",
						help="storage option for the store, e.g. region=us-east-1 (repeatable)")
	parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
	sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

	def command(name: str, func: Callable[[argparse.Namespace], int], help: str,
				json_output: bool = False) -> argparse.ArgumentParser:
		p = sub.add_parser(name, help=help, description=help)
		if json_output:
			p.add_argument("--json", action="store_true", help="machine-readable output")
		p.set_defaults(func=func)
		return p

	p = command("init", cmd_init, "create an empty store (default: the current directory)")
	p.add_argument("path", nargs="?", help="a path or URI")
	p.add_argument("--dtype", choices=["int16", "float16", "float32", "float64"], help="how to pack sample data")

	p = command("remote", cmd_remote, "manage named remotes (a URI plus its storage options)")
	p.add_argument("action", nargs="?", choices=["add", "remove", "list"], default="list")
	p.add_argument("name", nargs="?")
	p.add_argument("url", nargs="?")

	p = command("add", cmd_add, "stage data from a source: a BIDS folder, a manifest, or any folders")
	p.add_argument("sources", nargs="+", metavar="SOURCE",
				   help="a BIDS folder or a manifest .csv/.tsv/.json; several folders use the smart reader")
	p.add_argument("--reader", help="bids, manifest, smart, or an installed plugin (default: by source)")
	p.add_argument("--answers", default="neurozarr-answers.json", metavar="FILE",
				   help="smart reader: answers to its questions, read if present and saved after asking")
	p.add_argument("--optimize", choices=["speed", "balanced", "size"], help="codec trade-off (default: balanced)")
	p.add_argument("--probe-level", type=int, choices=[0, 1, 2, 3, 4],
				   help="codec decisions per: 0 none (zarr default), 1 item type, 2 + stored dtype, "
						"3 + chunk shape, 4 each array (default: 2)")
	p.add_argument("--probe-files", type=int, metavar="N", help="arrays probed per group (default: 8)")
	p.add_argument("--dtype", choices=["int16", "float16", "float32", "float64"], help="how to pack sample data")
	p.add_argument("-j", "--workers", type=int, metavar="N", help="stage N subjects in parallel")
	p.add_argument("--existing", choices=[policy.value for policy in ExistingPolicy], default="error",
				   help="when an item already exists (default: error)")
	p.add_argument("-n", "--dry-run", action="store_true", help="show the reader and codec plan; stage nothing")
	p.add_argument("-q", "--quiet", action="store_true", help="no progress or summary")

	p = command("rm", cmd_rm, "stage removing a subject, session or item")
	p.add_argument("paths", nargs="+", metavar="PATH", help="e.g. sub-03, sub-01/ses-2, sub-01/ses-1/ieeg/task-rest")
	p.add_argument("-r", "--recursive", action="store_true", help="needed to remove a subject or session")

	p = command("mv", cmd_mv, "stage renaming a session or item in place")
	p.add_argument("source", metavar="PATH")
	p.add_argument("dest", metavar="NEW_NAME")

	p = command("status", cmd_status, "show what is staged", json_output=True)
	p.add_argument("-l", "--long", dest="verbose_items", action="store_true", help="list every staged item")

	p = command("reset", cmd_reset, "unstage everything or some subjects; --hard moves the branch")
	p.add_argument("subjects", nargs="*", metavar="SUBJECT")
	p.add_argument("--hard", metavar="VERSION", help="move the current branch to VERSION and drop what is staged")
	p.add_argument("--yes", action="store_true", help="confirm --hard")

	p = command("commit", cmd_commit, "record what is staged as a new dataset version")
	p.add_argument("-m", "--message", required=True)

	p = command("log", cmd_log, "show the version history of the current branch", json_output=True)
	p.add_argument("--subject", help="one subject's own history")
	p.add_argument("-n", "--number", type=int, metavar="N", help="show only the last N versions")
	p.add_argument("--oneline", action="store_true")

	p = command("show", cmd_show, "show one version (default: the current branch tip)", json_output=True)
	p.add_argument("version", nargs="?", help="branch, tag, version id or prefix, REF~N")

	p = command("ls", cmd_ls, "list subjects, a subject's sessions, or a session's items", json_output=True)
	p.add_argument("path", nargs="?", help="e.g. sub-01 or sub-01/ses-1")
	p.add_argument("--version", help="list as of this version")

	p = command("find", cmd_find, "find recordings by entity", json_output=True)
	p.add_argument("terms", nargs="*", metavar="KEY=VALUE", help="e.g. task=rest datatype=ieeg sub=sub-01")

	p = command("diff", cmd_diff, "show changes: staged, VERSION vs now, or A vs B", json_output=True)
	p.add_argument("a", nargs="?", metavar="A")
	p.add_argument("b", nargs="?", metavar="B")
	p.add_argument("--subject", help="only this subject")
	p.add_argument("--stat", action="store_true", help="counts only")
	p.add_argument("--exit-code", action="store_true", help="exit 1 if anything differs")

	p = command("branch", cmd_branch, "list, create or delete branches")
	p.add_argument("name", nargs="?")
	p.add_argument("start", nargs="?", help="where the new branch starts (default: the current version)")
	p.add_argument("-d", "--delete", metavar="NAME", help="delete a branch whose versions another branch has")
	p.add_argument("-D", dest="force_delete", metavar="NAME", help="delete a branch even if that drops versions")
	p.add_argument("--yes", action="store_true", help="confirm -D")

	p = command("switch", cmd_switch, "make another branch current")
	p.add_argument("name")
	p.add_argument("start", nargs="?", help="with -c: where the new branch starts")
	p.add_argument("-c", "--create", action="store_true", help="create the branch first")

	p = command("revert", cmd_revert, "undo a version with a new one")
	p.add_argument("version")
	p.add_argument("-m", "--message")

	p = command("merge", cmd_merge, "bring another branch's changes into the current one")
	p.add_argument("branch")
	p.add_argument("-m", "--message")
	p.add_argument("--ours", action="append", default=[], metavar="SUBJECT", help="keep this subject as it is here")
	p.add_argument("--theirs", action="append", default=[], metavar="SUBJECT",
				   help="take this subject from the other branch")

	p = command("clone", cmd_clone, "copy a store: all of it (local paths), or one version anywhere")
	p.add_argument("source", help="a path, URI or remote name")
	p.add_argument("dest", help="a path or URI")
	p.add_argument("--version", help="with --depth 1: which version (default: the current one)")
	p.add_argument("--depth", type=int, choices=[1], help="copy one version only (works between any locations)")

	p = command("gc", cmd_gc, "delete stored data nothing can reach (reports only, unless --yes)")
	p.add_argument("--older-than", type=float, default=14, metavar="DAYS", help="only objects older than this")
	p.add_argument("--yes", action="store_true", help="really delete")

	p = command("query", cmd_query, "find items by identifiers and metadata, from the index", json_output=True)
	p.add_argument("expression", nargs="?", help="e.g. \"datatype == 'ieeg' and sfreq >= 250 and participant.age < 40\"")
	p.add_argument("--where", action="append", default=[], metavar="KEY=VALUE", help="equality filter (repeatable)")
	p.add_argument("--kind", choices=["recording", "table", "array", "file"])
	p.add_argument("--version", help="query this version (default: the current branch)")
	p.add_argument("--columns", help="with --json/--csv: only these columns (comma-separated)")
	p.add_argument("--csv", action="store_true", help="CSV output")
	p.add_argument("--count", action="store_true", help="print only the number of matches")

	p = command("tag", cmd_tag, "list tags, or name a version")
	p.add_argument("name", nargs="?")
	p.add_argument("version", nargs="?", help="what to tag (default: the current branch tip)")
	p.add_argument("-d", "--delete", metavar="NAME", help="delete a tag")

	p = command("validate", cmd_validate, "check a source before adding it", json_output=True)
	p.add_argument("source", help="a BIDS folder or a manifest .csv/.tsv/.json")

	p = command("verify", cmd_verify, "check the store faithfully matches its source")
	p.add_argument("source", help="what it was added from")
	p.add_argument("--limit", type=int, metavar="N", help="stop after N items (quick check)")
	p.add_argument("--reader", help="explicit built-in or installed reader name")

	p = command("export", cmd_export, "write the store back out as a BIDS folder")
	p.add_argument("dest", help="output BIDS directory")
	p.add_argument("--subject", action="append", help="export only this subject (repeatable)")

	p = command("migrate", cmd_migrate, "upgrade a legacy store to the current schema")
	p.add_argument("--dry-run", action="store_true", help="report the migration without writing")

	command("fsck", cmd_fsck, "check the store's schema and snapshot integrity")
	return parser


def main(argv: list[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	if args.verbose:
		set_verbosity(logging.DEBUG)
	try:
		return args.func(args)
	except (NeurozarrError, ValueError, KeyError, FileNotFoundError, FileExistsError, PermissionError) as e:
		print(f"error: {e}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	sys.exit(main())
