# Quickstart

This guide converts a directory of recordings — files named however they came
off the device — into a versioned store, typically in cloud storage, that
supports reading a short window of a long recording without downloading the
entire file.

It uses the Python API, the primary interface to neurozarr.

## Zarr and Icechunk in brief

Two terms recur throughout the documentation. Neither requires direct
interaction, but each explains part of what the package provides.

**Zarr** stores a large array as many small chunks rather than as a single
file. This is what allows a program to fetch one chunk without reading the
ones before it.

**Icechunk** adds version history on top of Zarr, in the way git tracks
versions of source code.

Together: recordings are stored in one place with a full version history, and
reading ten seconds from an hour-long recording transfers roughly ten seconds
of data rather than the entire file — including when the store is in a cloud
bucket.

## 1. Add one recording

Before converting a full dataset, load a single file and add it to a store
directly. This is the core operation the rest of the package builds on;
everything that follows applies it to many files at once.

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

Load the file with `mne.io.read_raw_fif`, or with the `mne.io.read_raw_*`
function matching your source format (EDF, BrainVision, GDF, and others are
supported). The remaining lines are the neurozarr API: a **subject**
represents one participant, a **visit** represents one of their sessions, and
`add_recording` stores the loaded signal under a `task` label you choose. The
`attrs` argument to `create_subject` and `add_visit` accepts arbitrary
metadata — age, device, diagnosis, or any other field relevant to your study.

Read it back to confirm it was stored correctly:

```python
rec = Repo.open("./demo.zarr").subject("sub-001").visit("ses-20220908").recording(task="Stream")
values, meta = rec.data(tmin=2, tmax=4)      # a numpy array, shape (2, 500)
```

This covers one subject, one visit, and one recording. Repeating it by hand
for every file across every participant and visit does not scale — the
remainder of this guide covers converting many files at once.

## 2. Describe multiple files in a table

neurozarr does not infer meaning from filenames. Instead, you provide a table
with one row per file, and the package handles the rest.

| sub | ses | datatype | task | path |
|---|---|---|---|---|
| sub-001 | ses-20220908 | ieeg | Stream | messy/patient1/patient1_20220908_stream_raw.fif |
| sub-001 | ses-20221103 | ieeg | Stream | messy/patient1/patient1_20221103_stream_raw.fif |
| sub-001 | ses-unknown | beh | TherapyLog | messy/patient1/patient1_therapy_log.csv |
| sub-002 | ses-20230114 | ieeg | Stream | messy/patient2/patient2_20230114_stream_raw.fif |

Five columns are sufficient to begin:

- **sub** — the participant. Must begin with `sub-`.
- **ses** — the visit. Must begin with `ses-`. A date makes a good label.
- **datatype** — `ieeg` for signal recordings, `beh` for logs and tables.
- **task** — a label for the recording. Your own vocabulary; `Stream` here.
- **path** — the file's current location.

Rather than typing this table by hand, write a short script that walks your
directory and derives each column from your own naming convention:

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

Adjust the two derivations — how a participant id and a visit date are
recovered from your filenames — to match your data. This script is the only
dataset-specific step in this guide; everything that follows is the same
regardless of the source data.

## 3. Check before converting

{func}`~neurozarr.inspect_source` reads the manifest and reports any problems
without writing anything. Run it first: correcting a manifest is far less
costly than discovering a missing file partway through a long conversion.

```python
from neurozarr import inspect_source

report = inspect_source("manifest.csv")
for issue in report:
    print(issue)
print(report.ok)        # True when nothing is fatal
```

Missing files, unreadable formats, and malformed identifiers all surface here.
Warnings — an unsupported format that will be stored as a reference, for
example — do not make `ok` false.

## 4. Convert

```python
from neurozarr import Repo, ManifestReader

repo = Repo.create("./study.zarr")
repo.ingest(ManifestReader("manifest.csv"))
version = repo.save("first conversion")
```

{meth}`~neurozarr.Repo.create` makes a new store and refuses to open an
existing one, so a typo in the destination path cannot silently create a
second store. Use {meth}`~neurozarr.Repo.open` for a store that already
exists.

Nothing is written until `save()`, which returns the id of the version just
created.

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

This is windowed reading, the package's main advantage: only the chunks
covering seconds 2 to 4 are read.

Search across every participant at once:

```python
for found in repo.find(task="Stream"):
    print(found.path)
# sub-001/ses-20220908/ieeg/task-Stream
# sub-001/ses-20221103/ieeg/task-Stream
# sub-002/ses-20230114/ieeg/task-Stream
```

Tables are returned as pandas DataFrames, with column types preserved:

```python
log = repo.subject("sub-001").visit("ses-unknown").tables()[0]
log.df()
```

## 6. Move it to the cloud

Replace the local path with a bucket URI. No other code changes:

```python
repo = Repo.create("s3://my-bucket/study", region="us-east-1")
repo.ingest(ManifestReader("manifest.csv"))
repo.save("first conversion")

repo = Repo.open("s3://my-bucket/study", region="us-east-1")
rec = repo.subject("sub-001").visit("ses-20220908").recording(task="Stream")
values, meta = rec.data(tmin=2, tmax=4)
```

Credentials are resolved the same way as other AWS tooling resolves them, and
extra keyword arguments such as `region=` are passed through to the storage
layer. `gs://`, `az://`, and `r2://` work the same way — see {doc}`guide/cloud`.

Windowed reads still fetch only the chunks they need, so the two-second read
above remains a small request even when the recording is stored in a bucket.

## Start small

Convert a single participant, read the result back, and confirm it matches
expectations before converting the full study. Adding more data later is a
routine operation:

```python
from neurozarr import Repo, ManifestReader

repo = Repo.open("./study.zarr", mode="a")
repo.ingest(ManifestReader("manifest.csv"), existing="skip")
repo.save("added the rest")
```

`existing="skip"` writes only what the store does not already hold. Without
it, writing over something that already exists raises an error rather than
silently replacing it — pass `existing="replace"` when overwriting is
intended.

## The command line

Each step above has a command-line equivalent, useful for inspecting a store
or scripting a conversion:

```bash
neurozarr validate manifest.csv        # step 3
neurozarr convert manifest.csv study.zarr
neurozarr info study.zarr
```

The Python API is the primary interface and provides more capability; the CLI
covers common conversions and inspections. See {doc}`guide/cli`.

## Where to go next

- Already have BIDS-formatted data? The manifest step is unnecessary —
  {doc}`guide/ingest` covers {class}`~neurozarr.BidsReader` and the other ways
  to fill a store.
- {doc}`guide/read` — windowed reads, channel selection, and searching in
  detail.
- {doc}`guide/versioning` — tags, history, and reading a store as it was.
- {doc}`guide/cloud` — buckets, credentials, and custom storage.
- {doc}`guide/layout` — how the store is organized, and why.
