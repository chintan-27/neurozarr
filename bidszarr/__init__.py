"""Put neural and behavioral recordings into a versioned Zarr/Icechunk store.

Point :class:`Repo` at a destination, add data, and save::

    from bidszarr import Repo

    repo = Repo("./study.zarr")
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
from .items import Attrs, Reader, Recording, Table
from .log import set_verbosity
from .read import RecordingView, TableView
from .readers import BidsReader, ManifestReader
from .repo import Repo, Subject, Visit
from .storage import storage_from
from .validate import validate_source
from .verify import export_bids, verify
from .writer import CodecConfig, Writer

try:
	__version__ = version("bidszarr")
except PackageNotFoundError:  # running from a source tree that was never installed
	__version__ = "0.0.0.dev0"

__all__ = [
	"__version__",
	"Repo", "Subject", "Visit",
	"Entities", "Attrs", "Recording", "Table", "Reader",
	"BidsReader", "ManifestReader",
	"RecordingView", "TableView",
	"CodecConfig", "Writer",
	"storage_from", "set_verbosity",
	"validate_source", "verify", "export_bids",
]
