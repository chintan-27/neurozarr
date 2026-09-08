# Reading data back

Open a store the same way you created one, then navigate down from it:

```python
from neurozarr import Repo

repo = Repo.open("./study.zarr")

repo.subjects()                       # ['sub-001', 'sub-002', ...]
subject = repo.subject("sub-001")
subject.attrs                         # {'age': 63, 'diagnosis': 'PD', ...}
subject.visits()                      # ['ses-20220908', ...]
```

## Getting to a recording

A {class}`~neurozarr.RecordingView` is a handle, not the data — nothing is read
from storage until you ask for values. Get one by naming its entities within a
session:

```python
rec = subject.visit("ses-20220908").recording(task="Stream", run=1)
```

Or collect them in bulk, from a session, a subject, or the whole store:

```python
subject.visit("ses-20220908").recordings()   # one session
subject.recordings()                          # every session of one subject
repo.find(task="BrainSenseStream", acq="TD")  # across every subject
```

{meth}`~neurozarr.Repo.find` opens one session per subject, so pass `sub=` when
you already know which subject you want.

## Reading the values

```python
values, meta = rec.data()   # ndarray (n_channels, n_samples) + its metadata
raw = rec.raw()             # an mne.io.RawArray, ready to pass to mne
df = rec.channels()         # the channels table as a DataFrame
events = rec.events()       # events and annotations
```

Recordings are stored as integers with a per-channel scale and offset.
{meth}`~neurozarr.RecordingView.data` and {meth}`~neurozarr.RecordingView.raw`
undo that for you and hand back physical units, so values match the source to
floating-point precision. Table columns keep their dtypes, so numbers come back
as numbers rather than strings. Annotations on the source recording are stored
as an events table and put back on the `Raw` when you read it whole.

## Reading part of a recording

Sample data is chunked along the time axis, so a window can be fetched without
reading the whole array — worth doing whenever you want seconds out of a
recording that runs for hours:

```python
values, meta = rec.data(tmin=10, tmax=20)                    # by seconds
values, meta = rec.data(start=1000, stop=2000)               # by sample index
values, meta = rec.data(tmin=10, tmax=20, picks=["LFP_L"])   # one channel
raw = rec.raw(tmin=10, tmax=20)                              # same, as an mne.Raw
```

Reading by seconds needs a stored sampling frequency; if a recording has none,
`tmin`/`tmax` raise `ValueError` and you can use `start`/`stop` instead.

How much a windowed read actually saves depends on chunk size, set by
{class}`~neurozarr.CodecConfig` when the data was written: the window is rounded
out to whole chunks, so a chunk far larger than your typical window means you
fetch more than you asked for.

Some useful properties for sizing a read before making it:

```python
rec.shape       # (n_channels, n_samples)
rec.sfreq       # sampling rate in Hz, or None
rec.duration    # seconds, or None
rec.array       # the underlying zarr array, to slice yourself
```

## Reading tables

Tables that stand on their own — behavioral logs, therapy history — come back as
{class}`~neurozarr.TableView`:

```python
for table in subject.tables():
    print(table.path, table.columns)
    df = table.df()                     # all of it
    df = table.df(columns=["onset"])    # or just some columns
```

Tables belonging to a recording are reached through that recording instead, with
{meth}`~neurozarr.RecordingView.channels` and
{meth}`~neurozarr.RecordingView.events`, rather than appearing in
{meth}`~neurozarr.Subject.tables`.

## Reading arrays and external files

Data that isn't a recording or a table — a NIfTI volume, a time-frequency
decomposition, anything with its own shape and dimensions — comes back as
{class}`~neurozarr.ArrayView`:

```python
array = subject.arrays()[0]
array.dims             # ('channel', 'frequency')
array.coords           # {'frequency': [0, 1, 2, ...]}, if any were stored
array.shape
array.data()                              # the whole array
array.data((slice(None), slice(0, 20)))   # a NumPy-style selection, read lazily
```

A file with no reader or format decoder is preserved by reference rather than
dropped, as {class}`~neurozarr.ExternalFileView`:

```python
ref = subject.external_files()[0]
ref.uri            # the original path or URI, unchanged
ref.media_type      # if the source recorded one
ref.reader_hint     # e.g. "install or register a reader for '.xyz'"
```

See {doc}`ingest` for when a file becomes one of these instead of a
`Recording` or `Table`, and {doc}`extensions` for adding real support for a
format that currently falls back to a reference.
