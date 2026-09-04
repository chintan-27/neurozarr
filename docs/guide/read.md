# Reading data back

Open a store the same way you created one, then navigate down from it:

```python
from neurozarr import Repo

repo = Repo("./study.zarr")

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
