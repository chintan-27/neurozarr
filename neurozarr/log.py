import logging
import tracemalloc

logger = logging.getLogger("neurozarr")
logger.addHandler(logging.NullHandler())  # library: never configure root logging

# Logical (uncompressed) bytes actually handed to storage so far this process --
# not a disk scan (expensive, and worse on cloud storage: a full object listing
# per log line) and not the final compressed size (only known after the fact,
# per chunk). Cheap because every write site already knows its own shape/dtype;
# incremented once per add_recording/add_table/add_array call, never decremented.
_bytes_written = 0


def record_write(n_bytes: int) -> None:
	"""Add to the running "bytes written this session" counter. Called once per
	item write, from writer.py, with a size already known at that call site."""
	global _bytes_written
	_bytes_written += n_bytes


def set_verbosity(level: int | str) -> None:
	"""Turn on neurozarr's logging output.

	The package is quiet by default, as a library should be. An application
	that configures logging itself can skip this and set the ``"neurozarr"``
	logger directly.

	Parameters
	----------
	level : int or str
		A logging level, such as ``logging.DEBUG`` or ``"INFO"``.

	Examples
	--------
	>>> import logging
	>>> neurozarr.set_verbosity(logging.INFO)
	"""
	if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
		handler = logging.StreamHandler()
		handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S"))
		logger.addHandler(handler)
	logger.setLevel(level)
	# Raw process RSS is dominated by one-time import cost (pandas/mne/etc.) and
	# by the allocator not handing freed memory back to the OS -- neither means
	# anything about what neurozarr itself is holding. tracemalloc tracks actual
	# Python-level allocations, so "live"/"peak" below reflect real referenced
	# data, not process footprint. Only started here, when verbose output is
	# explicitly requested, since tracking every allocation has a real cost.
	if not tracemalloc.is_tracing():
		tracemalloc.start()


def _fmt_bytes(n: float) -> str:
	"""Human-scaled size -- 0 B, 620 KB, 1.4 MB, 2.1 GB -- never a flat, opaque
	"0.00 MB" that can't tell a reader "zero" apart from "a few hundred KB"."""
	for unit in ("B", "KB", "MB"):
		if abs(n) < 1024:
			return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
		n /= 1024
	return f"{n:.1f} GB"


def log_mem(what: str = "") -> None:
	"""Live traced memory after an operation -- debug level, so a conversion is
	quiet by default (it fires once per write, which is thousands of times)."""
	if logger.isEnabledFor(logging.DEBUG) and tracemalloc.is_tracing():
		current, _peak = tracemalloc.get_traced_memory()
		logger.debug("Live memory: %s%s", _fmt_bytes(current), f" ({what})" if what else "")


def log_progress(verb: str, subject: str, detail: str = "") -> None:
	"""One line per item as the writer touches it -- info level, with real
	memory and data-volume numbers, so `verbose=True` shows actual progress
	instead of nothing for minutes.

	Every call site passes a capitalized verb ("Writing", "Creating", ...) and
	the thing it applies to, so every line reads the same way:
	"<Verb>: <subject> (<detail>)", right-padded so the numeric columns line up
	for anything shorter than a long path.

	Reports two things, each answering a different question: ``ram`` --
	Python-level memory actually referenced right now, with the highest it's
	ever been (in this process, since `set_verbosity`/`verbose=True` started
	tracemalloc) alongside it in parentheses; ``written`` -- cumulative logical
	bytes handed to storage so far this session (not disk usage after
	compression, and not a filesystem scan). Both use human-scaled units
	(B/KB/MB/GB), never a flat unit that goes opaque at either extreme.
	"""
	if logger.isEnabledFor(logging.INFO):
		message = f"{verb}: {subject}"
		if detail:
			message = f"{message} ({detail})"
		written = _fmt_bytes(_bytes_written)
		if tracemalloc.is_tracing():
			current, peak = tracemalloc.get_traced_memory()
			logger.info("%-70s ram %8s (peak %8s)   written %8s",
						message, _fmt_bytes(current), _fmt_bytes(peak), written)
		else:
			logger.info("%-70s written %8s", message, written)
