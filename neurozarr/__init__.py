"""Put neural and behavioral recordings into a versioned Zarr/Icechunk store.

Point :class:`Repo` at a destination, add data, and save::

    from neurozarr import Repo

    repo = Repo.create("./study.zarr")
    visit = repo.create_subject("sub-001").add_visit("ses-1")
    visit.add_recording(raw, task="Rest", run=1)
    repo.save("first recording")

To fill a store in bulk instead, hand :meth:`Repo.ingest` a reader:
:class:`ManifestReader` for files described in a table, :class:`BidsReader`
for data already in BIDS form, or one of your own implementing
:class:`Reader`.

Inside the store, data is filed by subject, session, datatype and entities,
following BIDS naming conventions -- a choice about the output, not a
requirement on the input.
"""

from importlib.metadata import PackageNotFoundError, version

from .entities import Entities
from .diagnostics import Severity, ValidationIssue, ValidationReport
from .errors import (
	NeurozarrError, SchemaVersionError, StoreIntegrityError, UnsupportedFormatError,
	ValidationError, WriteConflictError,
)
from .items import Array, Attrs, ExternalFile, Reader, Recording, Table
from .log import set_verbosity
from .read import ArrayView, ExternalFileView, RecordingView, TableView
from .readers import (
	BidsReader, ManifestReader, UnclaimedPolicy, available_readers, open_reader,
	reader_for, register_format, register_reader, registered_formats,
)
from .repo import Repo, Subject, Visit
from .storage import StorageFactory, StorageTarget, storage_from
from .validate import inspect_source, inspect_store, validate_source
from .verify import export_bids, verify
from .writer import CodecConfig, ExistingPolicy, Writer

try:
	__version__ = version("neurozarr")
except PackageNotFoundError:  # running from a source tree that was never installed
	__version__ = "0.0.0.dev0"

__all__ = [
	"__version__",
	"Repo", "Subject", "Visit",
	"Entities", "Attrs", "Recording", "Table", "ExternalFile", "Array", "Reader",
	"BidsReader", "ManifestReader", "available_readers", "open_reader", "reader_for", "register_reader",
	"UnclaimedPolicy", "register_format", "registered_formats",
	"RecordingView", "TableView", "ExternalFileView", "ArrayView",
	"CodecConfig", "ExistingPolicy", "Writer",
	"StorageFactory", "StorageTarget", "storage_from", "set_verbosity",
	"Severity", "ValidationIssue", "ValidationReport", "inspect_source", "inspect_store", "validate_source",
	"NeurozarrError", "ValidationError", "SchemaVersionError", "StoreIntegrityError",
	"WriteConflictError", "UnsupportedFormatError",
	"verify", "export_bids",
]
