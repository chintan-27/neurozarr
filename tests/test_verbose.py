"""verbose=True logs progress at info level; the package stays quiet without it."""

import logging

import pytest

from neurozarr import Repo
from neurozarr.log import logger


@pytest.fixture(autouse=True)
def _reset_verbosity():
	"""set_verbosity mutates a module-global logger -- undo it so these tests
	don't leak verbosity into whichever test runs next."""
	yield
	logger.setLevel(logging.WARNING)


def test_verbose_true_logs_writes_and_commits(store, raw, table, caplog):
	with caplog.at_level(logging.INFO, logger="neurozarr"):
		repo = Repo.create(store, verbose=True)
		visit = repo.create_subject("sub-001").add_visit("ses-1")
		visit.add_recording(raw, task="Stream", run=1)
		visit.add_behavioral_table(table, task="Log")
		repo.save("test data")

	messages = "\n".join(caplog.messages)
	assert "Writing recording" in messages
	assert "Writing table" in messages
	assert "Committed" in messages
	assert "MB" in messages


def test_verbose_false_by_default_stays_quiet(store, raw, caplog):
	logger.setLevel(logging.WARNING)  # undo any earlier verbose=True test in this session
	repo = Repo.create(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_recording(raw, task="Stream", run=1)
	repo.save("test data")

	assert caplog.records == []
