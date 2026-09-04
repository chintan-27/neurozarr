import logging
import resource

logger = logging.getLogger("bidszarr")
logger.addHandler(logging.NullHandler())  # library: never configure root logging


def set_verbosity(level):
	"""Turn on bidszarr's logging output.

	The package is quiet by default, as a library should be. An application
	that configures logging itself can skip this and set the ``"bidszarr"``
	logger directly.

	Parameters
	----------
	level : int or str
		A logging level, such as ``logging.DEBUG`` or ``"INFO"``.

	Examples
	--------
	>>> import logging
	>>> bidszarr.set_verbosity(logging.INFO)
	"""
	if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
		handler = logging.StreamHandler()
		handler.setFormatter(logging.Formatter("%(message)s"))
		logger.addHandler(handler)
	logger.setLevel(level)


def log_mem(what: str = ""):
	"""Peak process memory after an operation -- debug level, so a conversion is
	quiet by default (it fires once per write, which is thousands of times)."""
	if logger.isEnabledFor(logging.DEBUG):
		kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		logger.debug("peak memory: %.1f MB%s", kb / 1024, f" ({what})" if what else "")
