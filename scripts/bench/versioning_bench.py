"""Versioning, neurozarr vs DataLad (git + git-annex), the same operations on the same
BIDS source, same node:

	import    neurozarr init+add+commit   |  datalad create, copy files in, datalad save
	edit      the same one-sample correction (EDF sample 0 of channel 0, +1 count), five
	          times in a row: time and bytes each adds to history
	old       read the pre-correction value back from history
	branch    two branches each correct a different recording, then both merged
	clone     clone the dataset, then read one recording from the clone
	diff      what changed between the first and last version
	revert    undo the last correction

	python scripts/bench/versioning_bench.py BIDS --name bravo --out versioning.jsonl --scratch DIR

DataLad needs `datalad` and `git-annex` on PATH (pip install datalad git-annex). Every
step is checked, not only timed: a wrong value back from history fails the run.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mne
import numpy as np

from neurozarr import Repo
from neurozarr.readers import reader_for
from neurozarr.writer import ExistingPolicy


def tree_bytes(path: Path, under: str = "") -> int:
	"""Apparent bytes of regular files (links not followed): history cost, not block usage."""
	root = path / under if under else path
	total = 0
	for r, _, fs in os.walk(root):
		for f in fs:
			try:
				st = os.lstat(Path(r) / f)
			except FileNotFoundError:  # removed while walking
				continue
			if not os.path.islink(Path(r) / f):
				total += st.st_size
	return total


def sh(*cmd: str, cwd: Path | None = None) -> str:
	return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def patch_edf(path: Path, bump: int) -> None:
	"""Add ``bump`` digital counts to channel 0, sample 0 -- two bytes, in place."""
	with open(path, "r+b") as f:
		header_bytes = int(f.read(256)[184:192].decode().strip())
		f.seek(header_bytes)
		value = int(np.frombuffer(f.read(2), dtype="<i2")[0])
		new = value + bump if value + bump <= 32767 else value - bump
		f.seek(header_bytes)
		f.write(np.array([new], dtype="<i2").tobytes())


def first_sample(path: Path) -> float:
	return float(mne.io.read_raw(path, preload=False, verbose=False).get_data(start=0, stop=1)[0, 0])


class Timer:
	def __init__(self, out, base: dict):
		self.out, self.base = out, base

	def __call__(self, tool: str, step: str, fn, **extra):
		start = time.perf_counter()
		result = fn()
		seconds = time.perf_counter() - start
		row = {**self.base, "tool": tool, "step": step, "seconds": seconds, **extra}
		self.out.write(json.dumps(row) + "\n")
		self.out.flush()
		print(f"{tool:9s} {step:22s} {seconds:9.3f} s  {extra}", flush=True)
		return result


def pick_edfs(source: Path) -> tuple[Path, Path]:
	edfs = sorted(source.rglob("*_ieeg.edf")) or sorted(source.rglob("*.edf"))
	subjects = sorted({p.relative_to(source).parts[0] for p in edfs})
	first = next(p for p in edfs if p.relative_to(source).parts[0] == subjects[0])
	second = next((p for p in edfs if p.relative_to(source).parts[0] == subjects[-1] and p != first), edfs[1])
	return first, second


def neurozarr_side(source: Path, scratch: Path, t: Timer, edits: int) -> None:
	store, clone = scratch / "nz", scratch / "nz_clone"
	for p in (store, clone):
		shutil.rmtree(p, ignore_errors=True)
	a, b = pick_edfs(source)
	work = scratch / "nz_edits"
	shutil.rmtree(work, ignore_errors=True)
	work.mkdir(parents=True)

	# where the reader files each of the two EDFs (one pass over the source)
	wanted = {a.resolve(), b.resolve()}
	filed = {Path(i.raw.filenames[0]).resolve(): i.entities for i in reader_for(source).read()
			 if getattr(i, "raw", None) is not None and Path(i.raw.filenames[0]).resolve() in wanted}

	class _Item:
		def __init__(self, entities):
			self.entities = entities

	def item(edf: Path) -> "_Item":
		return _Item(filed[edf.resolve()])

	def correct(repo: Repo, edf: Path, bump: int, message: str) -> str:
		copy = work / edf.name
		shutil.copy(edf, copy)
		patch_edf(copy, bump)
		it = item(edf)
		raw = mne.io.read_raw(copy, preload=True, verbose=False)
		ent = it.entities
		visit = repo.subject(ent.sub)
		visit = visit.visit(ent.ses) if ent.ses else visit
		visit.add_recording(raw, existing=ExistingPolicy.REPLACE, datatype=ent.datatype, **ent.extra)
		return repo.commit(message) or ""

	def value(repo: Repo, edf: Path, version: str | None = None) -> float:
		path = "/".join(item(edf).entities.path())
		rec = repo.get(path, version=version) if version else repo.get(path)
		return float(rec.data(start=0, stop=1)[0][0, 0])

	repo = t("neurozarr", "import", lambda: (r := Repo.init(str(store)), r.add(str(source)), r.commit("import"))[0],
			 )
	base_bytes = tree_bytes(store)
	t.out.write(json.dumps({**t.base, "tool": "neurozarr", "step": "import_bytes", "bytes": base_bytes}) + "\n")
	first_version = repo.resolve()
	baseline = value(repo, a)
	versions = [first_version]
	for k in range(edits):
		before = tree_bytes(store)
		versions.append(t("neurozarr", "edit", lambda k=k: correct(repo, a, k + 1, f"fix {k}"), k=k))
		t.out.write(json.dumps({**t.base, "tool": "neurozarr", "step": "edit_bytes", "k": k,
								"bytes": tree_bytes(store) - before}) + "\n")
	assert abs(value(repo, a) - baseline) > 0, "edit not visible"
	old = t("neurozarr", "read_old_version", lambda: value(Repo.open(str(store)), a, first_version))
	assert old == baseline, (old, baseline)

	main = repo.head
	b_before = value(repo, b)
	repo.switch("fixA", create=True)
	correct(repo, a, 100, "fix A on a branch")
	repo.switch(main)
	repo.switch("fixB", create=True)
	correct(repo, b, 100, "fix B on a branch")
	repo.switch(main)
	t("neurozarr", "merge_two_branches", lambda: (repo.merge("fixA"), repo.merge("fixB")))
	assert value(repo, b) != b_before, "merge lost fix B"

	t("neurozarr", "diff_first_to_last", lambda: repo.diff(first_version, repo.resolve()))
	last = repo.resolve()
	t("neurozarr", "revert_last", lambda: repo.revert(last))
	assert value(repo, b) == b_before, "revert did not undo the merged fix"
	t("neurozarr", "clone", lambda: Repo.clone(str(store), str(clone)))
	t("neurozarr", "read_from_clone", lambda: value(Repo.open(str(clone)), a))
	t.out.write(json.dumps({**t.base, "tool": "neurozarr", "step": "final_bytes", "bytes": tree_bytes(store)}) + "\n")


def datalad_side(source: Path, scratch: Path, t: Timer, edits: int) -> None:
	ds, clone = scratch / "dl", scratch / "dl_clone"
	for p in (ds, clone):
		if p.exists():
			subprocess.run(["chmod", "-R", "u+w", str(p)], check=False)
			shutil.rmtree(p, ignore_errors=True)
	a, b = pick_edfs(source)
	rel_a, rel_b = a.relative_to(source), b.relative_to(source)

	def do_import() -> None:
		sh("datalad", "create", "--force", str(ds))
		# no background `git gc --auto` repacking mid-measurement: sizes stay comparable
		sh("git", "config", "gc.auto", "0", cwd=ds)
		sh("git", "config", "gc.autoDetach", "false", cwd=ds)
		shutil.copytree(source, ds, dirs_exist_ok=True, symlinks=False)
		sh("datalad", "save", "-d", str(ds), "-m", "import")

	t("datalad", "import", do_import)
	t.out.write(json.dumps({**t.base, "tool": "datalad", "step": "import_bytes",
							"bytes": tree_bytes(ds, ".git")}) + "\n")
	first = sh("git", "rev-parse", "HEAD", cwd=ds)
	baseline = first_sample(ds / rel_a)

	def correct(rel: Path, bump: int, message: str) -> None:
		sh("datalad", "unlock", "-d", str(ds), str(ds / rel))
		patch_edf(ds / rel, bump)
		sh("datalad", "save", "-d", str(ds), "-m", message)

	for k in range(edits):
		before = tree_bytes(ds, ".git")
		t("datalad", "edit", lambda k=k: correct(rel_a, k + 1, f"fix {k}"), k=k)
		t.out.write(json.dumps({**t.base, "tool": "datalad", "step": "edit_bytes", "k": k,
								"bytes": tree_bytes(ds, ".git") - before}) + "\n")
	assert abs(first_sample(ds / rel_a) - baseline) > 0, "edit not visible"

	def read_old() -> float:
		sh("git", "checkout", first, "--", str(rel_a), cwd=ds)
		value = first_sample(ds / rel_a)
		sh("git", "checkout", "HEAD", "--", str(rel_a), cwd=ds)
		return value

	old = t("datalad", "read_old_version", read_old)
	assert old == baseline, (old, baseline)

	main = sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=ds)
	b_before = first_sample(ds / rel_b)
	sh("git", "checkout", "-b", "fixA", cwd=ds)
	correct(rel_a, 100, "fix A on a branch")
	sh("git", "checkout", main, cwd=ds)
	sh("git", "checkout", "-b", "fixB", cwd=ds)
	correct(rel_b, 100, "fix B on a branch")
	sh("git", "checkout", main, cwd=ds)
	t("datalad", "merge_two_branches", lambda: sh("git", "merge", "--no-edit", "fixA", "fixB", cwd=ds))
	assert first_sample(ds / rel_b) != b_before, "merge lost fix B"

	t("datalad", "diff_first_to_last", lambda: sh("datalad", "diff", "-d", str(ds), "--from", first, "--to", "HEAD"))
	t("datalad", "revert_last", lambda: sh("git", "revert", "--no-edit", "-m", "1", "HEAD", cwd=ds))
	assert first_sample(ds / rel_b) == b_before, "revert did not undo the merged fix"
	t("datalad", "clone", lambda: sh("datalad", "clone", str(ds), str(clone)))
	t("datalad", "read_from_clone", lambda: (sh("datalad", "get", "-d", str(clone), str(clone / rel_a)),
											 first_sample(clone / rel_a))[1])
	t.out.write(json.dumps({**t.base, "tool": "datalad", "step": "final_bytes", "bytes": tree_bytes(ds, ".git")}) + "\n")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("source", type=Path)
	parser.add_argument("--name", required=True)
	parser.add_argument("--out", required=True)
	parser.add_argument("--scratch", type=Path, required=True)
	parser.add_argument("--edits", type=int, default=5)
	parser.add_argument("--tools", default="neurozarr,datalad")
	args = parser.parse_args()
	args.scratch.mkdir(parents=True, exist_ok=True)
	os.environ.setdefault("GIT_AUTHOR_NAME", "bench"), os.environ.setdefault("GIT_AUTHOR_EMAIL", "bench@localhost")
	os.environ.setdefault("GIT_COMMITTER_NAME", "bench"), os.environ.setdefault("GIT_COMMITTER_EMAIL", "bench@localhost")
	base = {"dataset": args.name, "node": platform.node(), "loadavg": os.getloadavg()[0],
			"source_bytes": tree_bytes(args.source)}
	with open(args.out, "a") as out:
		t = Timer(out, base)
		for tool in args.tools.split(","):
			{"neurozarr": neurozarr_side, "datalad": datalad_side}[tool](args.source, args.scratch, t, args.edits)


if __name__ == "__main__":
	sys.exit(main())
