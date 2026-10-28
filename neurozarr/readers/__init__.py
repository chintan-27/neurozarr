from .bids import BidsReader
from .formats import UnclaimedPolicy, decoder_for, register_format, registered_formats
from .manifest import ManifestReader
from .smart import SmartReader
from .registry import available_readers, open_reader, reader_for, register_reader


def _join_edf_annotation_records() -> None:
	"""mne turns an EDF+ annotation channel into bytes one data record at a time (a numpy
	round trip per record), which was 60% of opening a small EDF: ~6 of 9.5 ms per BRAVO
	file. Handed the records as one row, the same code yields the same bytes -- each
	record's bytes, concatenated -- in one pass: 9.8 -> 4.3 ms a file."""
	# ponytail: wraps a private mne function (mne 1.12); skipped if it is gone or no longer
	# takes the TAL array first. Upstream fix: concatenate the rows in _read_annotations_edf.
	import inspect
	import numpy as np
	try:
		from mne.io.edf import edf  # type: ignore[import-untyped]
	except ImportError:
		return
	original = getattr(edf, "_read_annotations_edf", None)
	if original is None or getattr(original, "_neurozarr_joined", False) \
			or next(iter(inspect.signature(original).parameters), None) != "annotations":
		return

	def joined(annotations, *args, **kwargs):  # type: ignore[no-untyped-def]
		if isinstance(annotations, np.ndarray) and annotations.ndim == 2 and len(annotations) > 1:
			annotations = annotations.reshape(1, -1)
		return original(annotations, *args, **kwargs)

	joined._neurozarr_joined = True  # type: ignore[attr-defined]
	edf._read_annotations_edf = joined


_join_edf_annotation_records()

__all__ = [
	"BidsReader", "ManifestReader", "SmartReader", "available_readers", "open_reader", "reader_for", "register_reader",
	"UnclaimedPolicy", "register_format", "registered_formats", "decoder_for",
]
