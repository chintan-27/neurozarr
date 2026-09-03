import logging
import resource

logger = logging.getLogger("bidszarr")
logger.addHandler(logging.NullHandler())  # library: never configure root logging


def set_verbosity(level):
	"""Turn on bidszarr's own logging. level is a logging level (logging.DEBUG,
	logging.INFO, ...) or its name. Applications that already configure logging
	themselves can ignore this and just set the "bidszarr" logger directly."""
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
