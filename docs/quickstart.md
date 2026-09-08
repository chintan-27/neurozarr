# Quickstart

This converts a folder of recordings into a versioned, cloud-ready store,
using the Python API — neurozarr's primary interface.

## Zarr and Icechunk

Two terms recur below; neither requires direct use.

**Zarr** stores a large array as many small chunks instead of one file, so
reading a chunk does not require reading the rest.

**Icechunk** adds version history on top, the way git does for source code.

Recordings end up stored once, with full history, and reading ten seconds
from an hour-long recording moves about ten seconds of data — on disk or in a
cloud bucket.

## 1. Add one recording

Load a single file and add it to a store before converting a whole dataset:

```python
import mne
from neurozarr import Repo

raw = mne.io.read_raw_fif("messy/patient1/patient1_20220908_stream_raw.fif", verbose=False)

repo = Repo.create("./demo.zarr")
subject = repo.create_subject("sub-001", attrs={"diagnosis": "PD"})
visit = subject.add_visit("ses-20220908")
visit.add_recording(raw, task="Stream")
repo.save("first recording")
```

Load the file with `mne.io.read_raw_fif`, or the `mne.io.read_raw_*` function
for your format (EDF, BrainVision, GDF, and others are supported). A
**subject** is one participant, a **visit** is one of their sessions, and
`add_recording` stores the signal under a `task` label you choose. `attrs`
takes any metadata — age, device, diagnosis.

Read it back:

```python
rec = Repo.open("./demo.zarr").subject("sub-001").visit("ses-20220908").recording(task="Stream")
values, meta = rec.data(tmin=2, tmax=4)      # a numpy array, shape (2, 500)
```

Doing this by hand for every file does not scale. The rest of this guide
converts many files at once.

## 2. Describe multiple files in a table

neurozarr does not infer meaning from filenames — describe your files in a
table, one row per file.

| sub | ses | datatype | task | path |
|---|---|---|---|---|
| sub-001 | ses-20220908 | ieeg | Stream | messy/patient1/patient1_20220908_stream_raw.fif |
| sub-001 | ses-20221103 | ieeg | Stream | messy/patient1/patient1_20221103_stream_raw.fif |
| sub-001 | ses-unknown | beh | TherapyLog | messy/patient1/patient1_therapy_log.csv |
| sub-002 | ses-20230114 | ieeg | Stream | messy/patient2/patient2_20230114_stream_raw.fif |

Five columns are enough:

- **sub** — the participant. Must begin with `sub-`.
- **ses** — the visit. Must begin with `ses-`. A date works well.
- **datatype** — `ieeg` for signal recordings, `beh` for logs and tables.
- **task** — a label for the recording. Your own vocabulary; `Stream` here.
- **path** — where the file is now.

Generate the table with a script rather than typing it by hand:

```python
import csv, pathlib, re

rows = []
for path in sorted(pathlib.Path("messy").rglob("*")):
    if not path.is_file():
        continue
    patient = path.parent.name                          # "patient1" -> "sub-001"
    date = re.search(r"(\d{8})", path.name)             # a date in the name -> a visit
    rows.append({
        "sub": "sub-%03d" % int(re.sub(r"\D", "", patient)),
        "ses": "ses-" + date.group(1) if date else "ses-unknown",
        "datatype": "ieeg" if path.suffix == ".fif" else "beh",
        "task": "Stream" if path.suffix == ".fif" else "TherapyLog",
        "path": str(path),
    })

with open("manifest.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sub", "ses", "datatype", "task", "path"])
    writer.writeheader()
    writer.writerows(rows)
```

Adjust the two derivations — participant id and visit date from your
filenames — for your own data. Everything after this is the same regardless
of source data.

## 3. Check before converting

{func}`~neurozarr.inspect_source` reads the manifest and reports problems
without writing anything:

```python
from neurozarr import inspect_source

report = inspect_source("manifest.csv")
for issue in report:
    print(issue)
print(report.ok)        # True when nothing is fatal
```

Missing files, unreadable formats, and malformed identifiers all show up
here. A warning — an unsupported format stored as a reference, for example —
does not make `ok` false.

## 4. Convert

```python
from neurozarr import Repo, ManifestReader

repo = Repo.create("./study.zarr")
repo.ingest(ManifestReader("manifest.csv"))
version = repo.save("first conversion")
```

{meth}`~neurozarr.Repo.create` makes a new store and fails if one already
exists, so a typo in the path cannot silently create a second store. Use
{meth}`~neurozarr.Repo.open` for a store that already exists.

`save()` writes nothing until it is called, and returns the new version's id.

## 5. Read it back

```python
from neurozarr import Repo

repo = Repo.open("./study.zarr")
repo.subjects()                                  # ['sub-001', 'sub-002']
repo.subject("sub-001").visits()                 # ['ses-20220908', 'ses-20221103', 'ses-unknown']

rec = repo.subject("sub-001").visit("ses-20220908").recording(task="Stream")
rec.ch_names()                                   # ['LFP_L', 'LFP_R']
rec.duration                                     # 10.0
rec.sfreq                                        # 250.0
```

Read a window rather than the whole recording:

```python
values, meta = rec.data(tmin=2, tmax=4)          # a numpy array, shape (2, 500)
raw = rec.raw(tmin=2, tmax=4)                    # the same window as an mne.Raw
```

Only the chunks covering seconds 2 to 4 are read from storage.

Search across every participant at once:

```python
for found in repo.find(task="Stream"):
    print(found.path)
# sub-001/ses-20220908/ieeg/task-Stream
# sub-001/ses-20221103/ieeg/task-Stream
# sub-002/ses-20230114/ieeg/task-Stream
```

Tables come back as pandas DataFrames, with column types preserved:

```python
log = repo.subject("sub-001").visit("ses-unknown").tables()[0]
log.df()
```

## 6. Move it to the cloud

Replace the local path with a bucket URI:

```python
repo = Repo.create("s3://my-bucket/study", region="us-east-1")
repo.ingest(ManifestReader("manifest.csv"))
repo.save("first conversion")

repo = Repo.open("s3://my-bucket/study", region="us-east-1")
rec = repo.subject("sub-001").visit("ses-20220908").recording(task="Stream")
values, meta = rec.data(tmin=2, tmax=4)
```

Credentials resolve the same way other AWS tooling resolves them. Extra
keyword arguments such as `region=` pass through to the storage layer.
`gs://`, `az://`, and `r2://` work the same way — see {doc}`guide/cloud`.

Windowed reads still fetch only the chunks they need, so the read above stays
a small request even from a bucket.

## Start small

Convert one participant first, read it back, and confirm it looks right
before converting the rest. Adding more later is routine:

```python
from neurozarr import Repo, ManifestReader

repo = Repo.open("./study.zarr", mode="a")
repo.ingest(ManifestReader("manifest.csv"), existing="skip")
repo.save("added the rest")
```

`existing="skip"` writes only what the store does not already hold. Without
it, writing over an existing path raises rather than replacing it silently —
pass `existing="replace"` to overwrite deliberately.

## The command line

Each step above has a command-line equivalent:

```bash
neurozarr validate manifest.csv        # step 3
neurozarr convert manifest.csv study.zarr
neurozarr info study.zarr
```

The Python API is primary and does more; the CLI covers common conversions
and inspections. See {doc}`guide/cli`.

## Where to go next

- Already have BIDS-formatted data? The manifest step is unnecessary —
  {doc}`guide/ingest` covers {class}`~neurozarr.BidsReader` and the other
  ways to fill a store.
- {doc}`guide/read` — windowed reads, channel selection, and searching in
  detail.
- {doc}`guide/versioning` — tags, history, and reading a store as it was.
- {doc}`guide/cloud` — buckets, credentials, and custom storage.
- {doc}`guide/layout` — how the store is organized, and why.
