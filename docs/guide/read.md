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

## Exploring what's there

{meth}`~neurozarr.Repo.describe` and {meth}`~neurozarr.Visit.describe` answer
"what's actually in here" as data, without listing each item kind separately
and reading `.path`/`.entities` off every result by hand:

```python
repo.describe()
# [{'sub_id': 'sub-001', 'visits': 1, 'recordings': 3, 'tables': 2, 'arrays': 0, 'external_files': 0}]

subject.visit("ses-20220908").describe()
# [{'kind': 'recording', 'path': 'sub-001/ses-20220908/ieeg/task-Stream_run-1', 'task': 'Stream', 'run': '1'},
#  {'kind': 'table', 'path': 'sub-001/ses-20220908/beh/task-TherapyHistory/table', 'task': 'TherapyHistory'}]
```

Both return a plain list of dicts, sorted by path — hand either straight to
`pandas.DataFrame(...)` for a table view.

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

{meth}`~neurozarr.Repo.find` answers from the store's index (below) and opens
nothing: each view it returns opens its recording only when you read from it.

## Querying the index

Every version of a store carries an index of everything in it, so finding items by
their identifiers *and* their metadata never walks the store:

```python
hits = repo.query("datatype == 'ieeg' and sfreq >= 250 and participant.age < 40")
len(hits)                 # counted from the index -- nothing opened yet
hits.df                   # a DataFrame: path, kind, identifiers, sfreq, n_channels,
                          # duration_s, dtype, codec, meta.*, participant.*
for rec in hits:          # the usual lazy views
    data, times = rec.data(tmin=10, tmax=20)

repo.query(task="rest", sfreq__ge=250, kind="recording")   # the same, as keywords
repo.query("exists(meta.PowerLineFrequency)", version="v1")  # any version
repo.get("sub-01/ses-1/ieeg/task-rest")                    # one item by its path
repo.query(task="rest", include_staged=True)               # with this Repo's staged changes
```

A `Repo` reads one version until told otherwise. {meth}`~neurozarr.Repo.refresh`
catches up with what others committed since -- one small read of the branch tip, then
only the subjects whose index changed are fetched again -- and
{meth}`~neurozarr.Repo.watch` yields each new version as it lands:

```python
for version in repo.watch(interval=2):          # near real time
    print(version, len(repo.query("sfreq >= 250")))
```

Indexes of a remote store are also kept on local disk
(`$XDG_CACHE_HOME/neurozarr/index/`), keyed by their content digest so they never go
stale, and a query fetches the ones it needs concurrently.

An expression uses `== != < <= > >= in`, `not in`, `and` / `or` / `not`,
`exists(col)`, and `col("odd-name")` for a column whose name isn't a plain word. A
missing value never matches a comparison (`!=` included), as in SQL. Columns:

| Column | Holds |
|---|---|
| `path`, `kind`, `sub`, `ses`, `datatype`, `prefix` | where the item is; `kind` is `recording`, `table`, `array` or `file` |
| `task`, `run`, `acq`, ... | every BIDS entity, as text (`run == '01'`) |
| `sfreq`, `n_channels`, `n_samples`, `duration_s` | recordings |
| `dtype`, `shape`, `chunks`, `codec`, `nbytes` | anything stored as an array |
| `meta.<key>` | each scalar in the item's metadata (its JSON sidecar) |
| `participant.<key>` | the subject's attributes (its `participants.tsv` row) |

How it stays fast, borrowing from Iceberg, DuckDB and Parquet: each version holds
one index per subject -- a columnar table with a row per item, dictionary-encoded,
sorted by path, stored as a single object -- and one summary with a row per subject
holding min/max of every numeric column and which values each categorical column
takes (a Bloom filter when there are many). A query first rules out every subject
the summary says can't match, then filters the remaining indexes with numpy, and
opens nothing at all -- a result's data is read only when you ask for it. The index
is updated at each commit from what changed: the writer describes each item as it
writes it, so committing re-reads only items changed some other way (metadata
edits, renames). It is always exactly the version it belongs to, and is cached by
its content digest.

A query that reaches many subjects would otherwise read one index per subject -- one
request each on object storage. So a store with more than 16 subjects also keeps a
packed index: every subject's rows in one object. It is refreshed lazily, at a commit
once more than 16 subjects (or 5% of them) changed since it was written; until then
a wide query reads the packed index plus the few changed subjects' own indexes, so
it is always current. Narrow queries keep reading single subject indexes.

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
