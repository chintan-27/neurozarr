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

Applications can register a factory in-process:

```python
register_reader("vendor-x", VendorReader, extensions=(".vendor",))
reader = open_reader("vendor-x", source)
```

Distributed plugins declare the factory under the `neurozarr.readers` package
entry-point group:

```toml
[project.entry-points."neurozarr.readers"]
vendor-x = "vendor_package:VendorReader"
```

Use `reader_for(source)` for built-in BIDS/manifest selection. A plugin is
selected automatically only when exactly one registered reader claims the file
extension; otherwise select it explicitly. Reader plugins never receive a
Zarr or Icechunk writer.
