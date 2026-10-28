"""util.prefetch reads ahead on a thread but behaves like the plain iterator: same
order, the source's exception raised where it would have been, and an early stop
stops the producer."""

import threading
import time

import pytest

from neurozarr import util


def test_order_kept():
	assert list(util.prefetch(iter(range(1000)), depth=3)) == list(range(1000))


def test_error_raised_after_the_items_before_it():
	def source():
		yield 1
		yield 2
		raise ValueError("bad file")
	seen = []
	with pytest.raises(ValueError, match="bad file"):
		for item in util.prefetch(source()):
			seen.append(item)
	assert seen == [1, 2]


def test_early_stop_stops_the_producer():
	produced = []
	def source():
		for i in range(10_000):
			produced.append(i)
			yield i
	stream = util.prefetch(source(), depth=2)
	assert next(stream) == 0
	stream.close()  # what a for loop breaking out, or an exception in its body, does
	time.sleep(0.3)
	assert len(produced) < 10 and not any(t.name == "neurozarr-prefetch" for t in threading.enumerate())
