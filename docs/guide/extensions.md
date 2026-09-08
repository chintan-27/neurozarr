# Arrays, format decoders, and reader extensions

Two different things in this package are called "readers" — see {doc}`ingest`
for the distinction. A **reader** (`Reader`, `BidsReader`, `ManifestReader`,
or one of your own) discovers files across a whole source and decides where
each one goes; a **format decoder** turns one already-located file into
item(s). This page covers extending both, plus arrays.

MNE `Raw` remains the first-class electrophysiology model. Recordings are opened
without preloading and written in time chunks. For other neuroscience arrays,
use `add_array` with explicit dimension names and optional JSON coordinates:

```python
visit.add_array(
    "power",
    power,
    dims=("channel", "frequency"),
    coords={"frequency": frequencies.tolist()},
    datatype="eeg",
    task="Rest",
)

array = visit.arrays()[0]
window = array.data((slice(None), slice(0, 20)))
```

The core writer remains responsible for layout and validation. Custom readers
yield only supported `Recording`, `Table`, `Attrs`, `ExternalFile`, or `Array`
items.

## Reader registration

Writing a {class}`~neurozarr.Reader` (see {doc}`ingest`) is enough to use it —
`repo.ingest(MyReader(...))` works with no registration at all. Register a
reader only when you also want one of:

- **Automatic selection** by file extension, so {func}`~neurozarr.reader_for`
  or `neurozarr convert` can pick it without importing your class.
- **Distribution** as an installable package, so other projects discover it
  without any of your source code.

### Registering for the current process

{func}`~neurozarr.register_reader` adds a reader to the running process. It
is not saved anywhere — call it once, before the first `reader_for()` call,
typically at the top of a script or notebook:

```python
from neurozarr import Attrs, register_reader, reader_for, available_readers

class DicomReader:
    """A whole DICOM series is a directory of files, not one -- unlike
    ManifestReader/BidsReader, this Reader's source is that directory."""

    def __init__(self, source, **options):
        self.source = source

    def read(self):
        yield Attrs((), {"note": f"pretend series read from {self.source}"})

register_reader("dicom", DicomReader, extensions=())

available_readers()                 # ['bids', 'dicom', 'manifest']
reader_for("./series", name="dicom")   # a DicomReader, selected explicitly
```

`extensions` is matched against the full filename, not just its last
dot-segment, so a compound extension like `.nii.gz` or `.tar.gz` works as
given. It is used only for automatic selection — call
{func}`~neurozarr.open_reader` with the name, or construct the class
directly, to use a registered reader without it. If more than one registered
reader claims the same file, `reader_for` raises rather than guessing; pass
`name=` to choose explicitly (`neurozarr convert --reader nifti` does the
same from the command line).

### Distributing a reader as a plugin

A package installed in the same environment is discovered with no
`register_reader` call at all, by declaring the class under the
`neurozarr.readers` entry-point group:

```toml
# in the plugin package's pyproject.toml
[project.entry-points."neurozarr.readers"]
nifti = "my_package:NiftiReader"
```

Once that package is installed, `available_readers()` lists `"nifti"` and
`reader_for()` selects it in any project, with nothing imported and nothing
called at startup. The one difference from local registration is where
`extensions` comes from: with no `register_reader` call to pass it to, it is
read from an `extensions` attribute on the class itself —

```python
class NiftiReader:
    extensions = (".nii.gz",)
    ...
```

— so a reader meant to work both ways should declare it there; a local
registration's own `extensions=` argument then simply confirms it.

Either way, a registered or installed reader never receives a Zarr or
Icechunk writer — only {class}`~neurozarr.Writer` does, reached through
`repo.ingest()`.

## Format decoders

Adding a *reader* means reimplementing file discovery — walking a directory,
or interpreting a manifest table — just to support one more file format.
Registering a **format decoder** instead adds support for one format to both
{class}`~neurozarr.BidsReader` and {class}`~neurozarr.ManifestReader` at
once: given one already-located file, it decodes that file into item(s).

`.nii`/`.nii.gz` is the built-in example: decoded automatically into an
{class}`~neurozarr.Array`, if `nibabel` is installed
(`pip install "neurozarr[imaging]"`), by exactly this mechanism. Add another
format the same way:

```python
from neurozarr import Array, register_format
from neurozarr.readers import registered_formats

def decode_nwb(path, entities, name, meta):
    """(path, entities, name, meta) -> an iterator of items, the same shape a
    Reader yields -- one call per file, decoding it into whatever it becomes."""
    import pynwb
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        data = io.read().acquisition[name].data[:]
    yield Array(entities, name, data, ("channel", "sample"),
                meta={**meta, "_neurozarr_decoder": "pynwb"})

register_format(".nwb", decode_nwb)
registered_formats()   # [..., '.nwb', '.nii', '.nii.gz']
```

`register_format` matches against the full filename, not just its last
dot-segment, so a compound extension like `.nii.gz` works as given — the same
fix as {func}`~neurozarr.reader_for`'s own compound-extension matching. A file
no decoder claims falls to `unclaimed`, a `ManifestReader`/`BidsReader`
constructor argument: `"reference"` (default) records its path as an
{class}`~neurozarr.ExternalFile`, content untouched; `"embed"` reads its raw
bytes into the store as a one-dimensional array instead, for formats you'd
rather have self-contained in the store than depend on the source file
staying put:

```python
BidsReader("./my_dataset", unclaimed="embed")
```

A decoder should stamp `meta["_neurozarr_decoder"]` with its own name, as the
built-in NIfTI decoder and the binary fallback both do — the store then
records not just what an item is, but how it was produced, which matters most
for a binary-embedded item: nothing else records that its bytes are opaque
rather than a real N-dimensional array.

`register_format` and reader registration are independent: a decoder never
receives a source to walk, and a reader never receives a single pre-located
file to decode. Registering one does not register the other.

Two shapes of source don't fit a format decoder at all, and need a full
reader instead: a **DICOM series** is normally many files describing one
volume (many-to-one, the reverse of what a decoder does), and a container
format like **NWB** can itself hold many logically separate recordings in one
file (one-to-many at the *reader* level, before any per-item decoding even
starts) — either can still register decoders for the files it does hand off
one at a time, but the series- or container-level grouping is a reader's job.
