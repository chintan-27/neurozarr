# Installation

```bash
pip install -e .
```

Requires Python ≥ 3.11. Pulls in `icechunk`, `zarr`, `mne`, `pandas`, and `numpy`.

```bash
pip install -e ".[imaging]"
```

Adds `nibabel`, which enables automatic NIfTI decoding (`.nii`/`.nii.gz`) in
every reader — see {doc}`guide/ingest`. Without it, NIfTI files are handled like
any other unrecognized format.

```bash
pip install -e ".[formats]"
```

Adds the readers for every other supported format: `neo` (spike and
intracellular systems), `pydicom`, `tifffile` and `imagecodecs` (TIFF, OME-TIFF),
`defusedxml` (EGI MFF), `antio` (ANT), `pymef` (MEF3), `curryreader` (Curry) and
`pymatreader` (FieldTrip). Each is imported only when its format turns up.

## Type checking

The package ships a `py.typed` marker, so mypy, Pyright and other type checkers
use its annotations rather than treating everything as `Any`. No stub package is
needed.

Values that may be absent are typed as optional, which means a checker catches
their misuse rather than leaving it to fail at runtime. A recording stored
without a sampling frequency, for instance, has `sfreq` and `duration` of
`None`:

```python
rec.duration / 2        # error: unsupported operand types for / ("None" and "int")

if rec.duration is not None:
    rec.duration / 2    # fine
```
