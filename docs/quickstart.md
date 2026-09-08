# Start here

You have recordings scattered across folders, named however they were named
when they came off the device. You want them somewhere central — probably in
the cloud — and you want to read a few seconds out of a long recording without
dragging the whole file back down.

This page goes from that pile of files to a working store. It uses the Python
API throughout, which is the way the package is meant to be used.

## The two words in the description

You never have to touch either of these directly, but they explain what you get.

**Zarr** saves a big array as many small pieces instead of one large file. That
is the whole idea. It matters because a program can fetch piece 47 without
reading pieces 1 through 46.

**Icechunk** adds history on top: every save is a version you can go back to,
in the way git holds versions of code.

Put together: your recordings live in one place, every conversion is a version
you can return to, and reading ten seconds out of an hour-long recording moves
roughly ten seconds' worth of data — not the hour. That last part is the reason
to bother, and it is just as true when the store lives in a cloud bucket.

## 1. One recording, by hand

Before automating anything, do the smallest possible version once: load one of
your files and add it to a store yourself. This is the whole shape of the
package — everything later on is this same operation, done for many files at
once instead of one.

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

`mne.io.read_raw_fif` loads a `.fif` file the way you would for any other
purpose — use whichever `mne.io.read_raw_*` function matches your format (EDF,
BrainVision, GDF and others all have one). Everything from `repo =` down is
neurozarr: a **subject** holds one participant, a **visit** holds one session
of theirs, and `add_recording` files the loaded signal under whatever `task`
name you give it. `attrs` on `create_subject` and `add_visit` are free-form
metadata — age, device, diagnosis, whatever you have.

Read it straight back to confirm it landed where you expect:

```python
rec = Repo.open("./demo.zarr").subject("sub-001").visit("ses-20220908").recording(task="Stream")
values, meta = rec.data(tmin=2, tmax=4)      # a numpy array, shape (2, 500)
```

That's it — one subject, one visit, one recording, read back. Doing this by
hand for every file across every participant and every visit does not scale,
which is what the rest of this page is for.

## 2. More than one? Describe your files in a table

The package does not guess what your filenames mean. You hand it a table with
one row per file, and it does the rest.

| sub | ses | datatype | task | path |
|---|---|---|---|---|
| sub-001 | ses-20220908 | ieeg | Stream | messy/patient1/patient1_20220908_stream_raw.fif |
| sub-001 | ses-20221103 | ieeg | Stream | messy/patient1/patient1_20221103_stream_raw.fif |
| sub-001 | ses-unknown | beh | TherapyLog | messy/patient1/patient1_therapy_log.csv |
| sub-002 | ses-20230114 | ieeg | Stream | messy/patient2/patient2_20230114_stream_raw.fif |

Five columns are enough to start:

- **sub** — which participant. Must begin with `sub-`.
- **ses** — which visit. Must begin with `ses-`. A date makes a good label.
- **datatype** — `ieeg` for signal recordings, `beh` for logs and tables.
- **task** — what the recording is. Your own vocabulary; `Stream` here.
- **path** — where the file is right now.

Do not type this by hand. Write a short script that walks your folder and
derives the columns from your own naming scheme:

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

Adjust the two guesses — how a participant id and a visit date are recovered
from your filenames — and you are done. This script is the only real work on
this page; everything after it is the same for everybody.

## 3. Check before converting

{func}`~neurozarr.inspect_source` reads the manifest and reports what it finds
wrong, without writing anything. Run it first: it is far cheaper to fix a
manifest than to discover a missing file partway through a long conversion.

```python
from neurozarr import inspect_source

report = inspect_source("manifest.csv")
for issue in report:
    print(issue)
print(report.ok)        # True when nothing is fatal
```

Missing files, unreadable formats and malformed identifiers all surface here.
Warnings — an unsupported format that will be stored as a reference, say — do
not make `ok` false.

## 4. Convert

```python
from neurozarr import Repo, ManifestReader

repo = Repo.create("./study.zarr")
repo.ingest(ManifestReader("manifest.csv"))
version = repo.save("first conversion")
```

{meth}`~neurozarr.Repo.create` makes a new store and refuses to open an
existing one, so a typo in the path cannot quietly scatter your data into a
second store. Use {meth}`~neurozarr.Repo.open` for a store that already exists.

Nothing is written until `save()`, which returns the id of the version you just
made.

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

Pull out a window rather than the whole recording:

```python
values, meta = rec.data(tmin=2, tmax=4)          # a numpy array, shape (2, 500)
raw = rec.raw(tmin=2, tmax=4)                    # the same window as an mne.Raw
```

Those two lines are the point of the whole exercise. Only the chunks covering
seconds 2 to 4 are read.

Search across every participant at once:

```python
for found in repo.find(task="Stream"):
    print(found.path)
# sub-001/ses-20220908/ieeg/task-Stream
# sub-001/ses-20221103/ieeg/task-Stream
# sub-002/ses-20230114/ieeg/task-Stream
```

Tables come back as pandas DataFrames, with their column types intact:

```python
log = repo.subject("sub-001").visit("ses-unknown").tables()[0]
log.df()
```

## 6. Move it to the cloud

Change the path to a bucket URI. Nothing else about your code changes:

```python
repo = Repo.create("s3://my-bucket/study", region="us-east-1")
repo.ingest(ManifestReader("manifest.csv"))
repo.save("first conversion")

repo = Repo.open("s3://my-bucket/study", region="us-east-1")
rec = repo.subject("sub-001").visit("ses-20220908").recording(task="Stream")
values, meta = rec.data(tmin=2, tmax=4)
```

Credentials are picked up the way other AWS tooling picks them up, and extra
keyword arguments such as `region=` are passed through to the storage layer.
`gs://`, `az://` and `r2://` work the same way — see {doc}`guide/cloud`.

Windowed reads still fetch only the chunks they need, so the two-second read
above stays a small request even when the recording lives in a bucket.

## Start small

Convert one participant, read it back, and confirm it looks the way you expect
before running the whole study. Adding more later is an ordinary operation:

```python
from neurozarr import Repo, ManifestReader

repo = Repo.open("./study.zarr", mode="a")
repo.ingest(ManifestReader("manifest.csv"), existing="skip")
repo.save("added the rest")
```

`existing="skip"` writes only what the store does not already hold. Without it,
writing over something that already exists raises rather than silently
replacing it — pass `existing="replace"` when overwriting is what you mean.

## The command line

Every step above has a command-line equivalent, useful for a quick look at a
store or for a conversion in a shell script:

```bash
neurozarr validate manifest.csv        # step 3
neurozarr convert manifest.csv study.zarr
neurozarr info study.zarr
```

The Python API is the primary interface and does more — the CLI covers the
common conversions and inspections. See {doc}`guide/cli`.

## Where to go next

- Already have BIDS-formatted data? Skip the manifest entirely — {doc}`guide/ingest`
  covers {class}`~neurozarr.BidsReader` and the other ways to fill a store.
- {doc}`guide/read` — windowed reads, channel selection and searching in detail.
- {doc}`guide/versioning` — tags, history, and reading a store as it was.
- {doc}`guide/cloud` — buckets, credentials and custom storage.
- {doc}`guide/layout` — how the store is organized, and why.
