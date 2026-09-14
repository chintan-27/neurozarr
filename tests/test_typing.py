"""The package ships py.typed, which promises a downstream type checker can rely
on its annotations. These tests check that promise actually holds."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MYPY = Path(sys.executable).with_name("mypy")

pytestmark = [
	pytest.mark.skipif(not MYPY.exists(), reason="mypy not installed"),
]


def _check(source: str, tmp_path: Path) -> str:
	"""Type check a snippet as a downstream user would, and return the output."""
	script = tmp_path / "downstream.py"
	script.write_text(source)
	result = subprocess.run(
		[str(MYPY), "--no-incremental", "--cache-dir", str(tmp_path / ".cache"), str(script)],
		capture_output=True, text=True, cwd=tmp_path,
		env={"PATH": str(MYPY.parent), "MYPYPATH": str(ROOT), "HOME": str(tmp_path)},
	)
	return result.stdout


def test_package_is_internally_type_clean():
	"""py.typed claims the package is typed, so its own annotations must check out."""
	result = subprocess.run([str(MYPY)], capture_output=True, text=True, cwd=ROOT)
	assert "Success" in result.stdout, result.stdout


def test_optional_returns_are_caught(tmp_path):
	"""duration and sfreq return None when a recording has no sampling frequency.
	Annotating them as plain floats would let this through silently."""
	out = _check(
		"from neurozarr import Repo\n"
		"rec = Repo('s').subject('sub-001').visit('ses-1').recording(task='Rest')\n"
		"half = rec.duration / 2\n"
		"rate: float = rec.sfreq\n",
		tmp_path,
	)
	assert 'Left operand is of type "float | None"' in out, out
	assert 'expression has type "float | None"' in out, out


def test_wrong_argument_types_are_caught(tmp_path):
	out = _check(
		"from neurozarr import Repo\n"
		"rec = Repo('s').subject('sub-001').visit('ses-1').recording(task='Rest')\n"
		"rec.data(tmin='ten')\n",
		tmp_path,
	)
	assert 'has incompatible type "str"; expected "float | None"' in out, out


def test_correct_usage_passes(tmp_path):
	"""The flip side: ordinary correct code must not be flagged.

	Also guards against leaking noise: neurozarr imports mne, which ships no
	types, and those errors must be silenced in this package's own source. A
	config-file override would not help, since a downstream project's mypy
	never reads this repo's pyproject.
	"""
	out = _check(
		"from neurozarr import Repo\n"
		"repo = Repo('s')\n"
		"for sub_id in repo.subjects():\n"
		"    subject = repo.subject(sub_id)\n"
		"    for rec in subject.recordings():\n"
		"        values, meta = rec.data(tmin=1.0, tmax=2.0, picks=['a'])\n"
		"        names: list[str] = rec.ch_names()\n"
		"        rows: int = len(rec.channels())\n",
		tmp_path,
	)
	assert "Success" in out, out
