"""Per-user local state for a store, the way git keeps ``.git/HEAD`` and ``.git/index``
apart from the objects: the current branch, the staging area, and named remotes.

A store is often in the cloud and shared, so this state lives on the user's machine,
never in the store: ``$XDG_STATE_HOME/neurozarr/<store key>/`` holds ``HEAD`` and
``index/<branch>/``; remotes live in ``$XDG_CONFIG_HOME/neurozarr/remotes.json``.

The index holds, per touched repository (a subject, or ``_dataset``), a pickled
Icechunk fork -- the staged changes; chunks themselves were already written to the
store when staged, as git writes objects at ``add`` -- plus the snapshot the fork
was taken from, which ``commit`` needs to detect a conflicting change made since.
"""

import datetime
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import ValidationError

DEFAULT_BRANCH = "main"


def _home(var: str, fallback: str) -> Path:
	return Path(os.environ.get(var) or Path.home() / fallback) / "neurozarr"


def config_dir() -> Path:
	return _home("XDG_CONFIG_HOME", ".config")


def state_root() -> Path:
	return _home("XDG_STATE_HOME", ".local/state")


def persistable(target: Any) -> bool:
	"""Whether other processes can reach this store, so its state can live on disk.
	An in-memory store, or a raw icechunk.Storage object, lives in one process only."""
	return isinstance(target, (str, Path)) and urlparse(str(target)).scheme != "memory"


def normalize(target: str | Path) -> str:
	text = str(target)
	return str(Path(text).expanduser().resolve()) if urlparse(text).scheme in ("", "file") else text.rstrip("/")


# ---- remotes ------------------------------------------------------------------

def _remotes_file() -> Path:
	return config_dir() / "remotes.json"


def remotes() -> dict[str, dict[str, Any]]:
	"""Saved remotes: name -> ``{"url": ..., "options": {...}}``."""
	path = _remotes_file()
	return json.loads(path.read_text()) if path.exists() else {}


def add_remote(name: str, url: str, options: dict[str, Any] | None = None) -> None:
	if not name.replace("-", "").replace("_", "").isalnum():
		raise ValidationError(f"remote name must be letters, digits, '-' or '_', got {name!r}")
	saved = remotes()
	if name in saved:
		raise ValidationError(f"remote {name!r} already exists")
	saved[name] = {"url": url, "options": options or {}}
	_remotes_file().parent.mkdir(parents=True, exist_ok=True)
	_remotes_file().write_text(json.dumps(saved, indent=1))


def remove_remote(name: str) -> None:
	saved = remotes()
	if name not in saved:
		raise ValidationError(f"no such remote: {name!r}")
	del saved[name]
	_remotes_file().write_text(json.dumps(saved, indent=1))


def resolve(target: Any, options: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
	"""A remote name becomes its URL, with its saved options under any given here."""
	if isinstance(target, str) and target in remotes():
		saved = remotes()[target]
		return saved["url"], {**saved.get("options", {}), **options}
	return target, options


def discover(start: Path | None = None) -> Path | None:
	"""The local store the current directory is in, like git finding ``.git``: the
	nearest directory (this one or a parent) holding a ``_dataset`` repository."""
	here = (start or Path.cwd()).resolve()
	for folder in (here, *here.parents):
		if (folder / "_dataset").is_dir():
			return folder
	return None


# ---- HEAD and the index -------------------------------------------------------

class LocalState:
	"""HEAD and the staging area of one store, for this user on this machine.

	Keyed by the store's address *and* its dataset id, so two copies of one dataset
	never share an index. ``None`` as the directory means in-process only (an
	in-memory store): nothing is written, and the index is whatever this process holds.
	"""

	def __init__(self, target: Any, dataset_id: str):
		self.dir: Path | None = None
		if persistable(target):
			key = hashlib.sha256(normalize(target).encode()).hexdigest()[:16]
			self.dir = state_root() / f"{key}-{dataset_id}"

	@property
	def head(self) -> str:
		path = self.dir / "HEAD" if self.dir else None
		return path.read_text().strip() if path and path.exists() else DEFAULT_BRANCH

	@head.setter
	def head(self, branch: str) -> None:
		if self.dir:
			self.dir.mkdir(parents=True, exist_ok=True)
			(self.dir / "HEAD").write_text(branch + "\n")

	def _index(self, branch: str) -> Path | None:
		return self.dir / "index" / branch if self.dir else None

	def stage(self, branch: str, repo_id: str, fork: Any, base: str,
			  rows: dict[str, dict[str, Any]] | None = None) -> None:
		folder = self._index(branch)
		if folder is None:
			return
		folder.mkdir(parents=True, exist_ok=True)
		# write-then-rename, so a crash mid-write never leaves a half fork in the index
		tmp = folder / f".{repo_id}.fork.tmp"
		tmp.write_bytes(pickle.dumps(fork))
		tmp.replace(folder / f"{repo_id}.fork")
		(folder / f"{repo_id}.base").write_text(base)
		(folder / f"{repo_id}.rows.json").write_text(json.dumps(rows or {}))

	def unstage(self, branch: str, repo_id: str) -> None:
		folder = self._index(branch)
		if folder is not None:
			for suffix in (".fork", ".base", ".rows.json"):
				(folder / f"{repo_id}{suffix}").unlink(missing_ok=True)

	def staged(self, branch: str) -> dict[str, tuple[Any, str, dict[str, dict[str, Any]]]]:
		"""repo id -> (fork, base snapshot, index rows the writer captured) for everything
		staged on ``branch``."""
		folder = self._index(branch)
		if folder is None or not folder.exists():
			return {}
		out = {}
		for path in sorted(folder.glob("*.fork")):
			rows = folder / f"{path.stem}.rows.json"
			out[path.stem] = (pickle.loads(path.read_bytes()), (folder / f"{path.stem}.base").read_text(),
							  json.loads(rows.read_text()) if rows.exists() else {})
		return out

	def removed(self, branch: str) -> set[str]:
		folder = self._index(branch)
		path = folder / "removed.json" if folder else None
		return set(json.loads(path.read_text())) if path and path.exists() else set()

	def set_removed(self, branch: str, subjects: set[str]) -> None:
		folder = self._index(branch)
		if folder is None:
			return
		folder.mkdir(parents=True, exist_ok=True)
		if subjects:
			(folder / "removed.json").write_text(json.dumps(sorted(subjects)))
		else:
			(folder / "removed.json").unlink(missing_ok=True)

	def oldest_staged(self) -> "datetime.datetime | None":
		"""When the oldest staged change on any branch was written, for gc's safety check."""
		if self.dir is None or not (self.dir / "index").exists():
			return None
		times = [path.stat().st_mtime for path in (self.dir / "index").rglob("*.fork")]
		return datetime.datetime.fromtimestamp(min(times), datetime.timezone.utc) if times else None

	def clear(self, branch: str) -> None:
		folder = self._index(branch)
		if folder is not None and folder.exists():
			shutil.rmtree(folder)
