"""Convert neural and behavioral recordings into a BIDS-shaped Zarr/Icechunk store.

Point :class:`Repo` at a destination, hand :meth:`Repo.ingest` a reader, and save::

    from bidszarr import Repo, BidsReader

    repo = Repo("./study.zarr")
    repo.ingest(BidsReader("./my_bids_dataset"))
    repo.save("initial conversion")
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
