# Arrays and reader extensions

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

class NiftiReader:
    """Reads an imaging format ManifestReader can't parse -- see the note on
    the anat/T1w row in :doc:`ingest`."""

    def __init__(self, source, **options):
        self.source = source

    def read(self):
        yield Attrs((), {"note": f"pretend header read from {self.source}"})

register_reader("nifti", NiftiReader, extensions=(".nii.gz",))

available_readers()                 # ['bids', 'manifest', 'nifti']
reader_for("sub-001_T1w.nii.gz")    # a NiftiReader, chosen automatically
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
