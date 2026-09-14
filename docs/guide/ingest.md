# Getting data in

There are four ways to fill a store. They all produce the same result, so pick
whichever matches the shape your data is already in.

Two of them are backed by a **reader** ({class}`~neurozarr.BidsReader`,
{class}`~neurozarr.ManifestReader`, or one you write): given a whole source —
a directory tree, a table — it discovers many files and decides where each
one goes. Underneath both built-in readers is a second, smaller piece: a
**format decoder** that turns one already-located file into item(s) — the
per-extension dispatch described below, under "How each row is read". A
reader decides *where*; a decoder decides *what a file becomes*. See
{doc}`extensions` to add either.

## From a set of files, via a manifest

The general case: describe your files in a table, one row per file, and
{class}`~neurozarr.ManifestReader` handles the rest. It makes no assumptions
about folder layout or naming, and a row's `datatype` is not tied to any
particular file format — `ieeg`, `beh`, and `anat` below are read by three
different mechanisms, described after the table.

| sub | ses | datatype | task | path |
|---|---|---|---|---|
| sub-001 | ses-1 | ieeg | Stream | /data/sub-001_stream.edf |
| sub-001 | ses-1 | beh | TherapyLog | /data/sub-001_therapy.csv |
| sub-001 | ses-1 | anat | T1w | /data/sub-001_T1w.nii.gz |

```python
from neurozarr import Repo, ManifestReader

repo = Repo.create("./study.zarr")
repo.ingest(ManifestReader("manifest.csv"))
repo.save("initial conversion")
```

Six column names are reserved — `sub`, `ses`, `datatype`, `name`, `meta` and
`path` — and every other column becomes an entity, so a `task` or `run` column
files itself correctly. If your table already uses different names, say so
rather than renaming it:

```python
ManifestReader(df, sub_col="subject", path_col="filepath")
```

### How each row is read

Files are opened by extension, and what comes back depends on what the
extension is:

- **Signal formats** (EDF, BDF, GDF, BrainVision, EEGLAB, FIF, CNT) load
  through `mne.io.read_raw` and become a {class}`~neurozarr.Recording`. This
  is the `ieeg`/Stream row above.
- **`.tsv`/`.csv`** load with pandas and become a {class}`~neurozarr.Table`,
  with column types recorded so they round-trip on read. This is the
  `beh`/TherapyLog row:

  ```python
  log = repo.subject("sub-001").visit("ses-1").tables()[0]
  log.df()
  #    onset  amplitude_mA  pulse_width_us
  # 0      0           2.5              60
  # 1     60           2.5              60
  ```

- **NIfTI** (`.nii`/`.nii.gz`), such as the `anat`/T1w row above, decodes to an
  {class}`~neurozarr.Array` with real voxel data and `x`/`y`/`z` (and `t`, if
  4D) dimensions — automatically, if `nibabel` is installed
  (`pip install "neurozarr[imaging]"`). Without it, a NIfTI file falls through
  to the next rule like any other unrecognized format.
- **Anything else** becomes an explicit {class}`~neurozarr.ExternalFile` by
  default: the path is recorded, not the file's contents.

  ```python
  ref = repo.subject("sub-001").visit("ses-1").external_files()[0]
  ref.uri            # the original path
  ref.reader_hint     # "install or register a reader for '.xyz'"
  ```

  Pass `unclaimed="embed"` to `ManifestReader`/`BidsReader` to read the file's
  raw bytes into the store instead of only referencing it — see
  {doc}`extensions` for that and for adding real support for a format via
  {func}`~neurozarr.register_format`.

A row that can't be represented safely — a malformed identifier, or data that
matches a supported extension but fails to parse — still stops the
conversion; only formats with no reader at all fall back to a reference.

And when a row needs handling the columns can't express, take over entirely:

```python
ManifestReader(df, row_reader=lambda row: my_custom_item(row))
```

## Explicitly

When you are building a dataset programmatically, or only have a few
recordings, skip the readers entirely:

```python
repo = Repo.create("./study.zarr")

subject = repo.create_subject("sub-001", attrs={"age": 63, "diagnosis": "PD"})
visit = subject.add_visit("ses-20220908", attrs={"device": "Percept PC"})

visit.add_recording(my_mne_raw, task="Stream", run=1)
visit.add_behavioral_table(my_dataframe, task="TherapyHistory")

repo.save("added sub-001")
```

Keyword arguments are entities, and decide where each item lands.

## By writing a reader

Reach for this when your data has no standard layout *and* no standard file
format — a lab-specific database export, a directory structure a manifest
can't describe, anything a `path` column and some column-to-entity mapping
won't capture.

Implement {class}`~neurozarr.Reader`: one method, `read()`, yielding any of
{class}`~neurozarr.Recording`, {class}`~neurozarr.Table`,
{class}`~neurozarr.Attrs`, {class}`~neurozarr.ExternalFile`, or
{class}`~neurozarr.Array`. There is no base class to inherit from — `Reader`
is a `Protocol`, so any object with a matching `read()` works:

```python
import mne
import pandas as pd
from neurozarr import Recording, Table, Attrs, Entities

class MyLabReader:
    """Our lab keeps one CSV of session metadata and a folder of raw
    recordings named by an internal session id, not by BIDS convention."""

    def __init__(self, sessions_csv, recordings_dir):
        self.sessions = pd.read_csv(sessions_csv)
        self.recordings_dir = recordings_dir

    def read(self):
        yield Attrs((), {"Name": "my lab's study"})
        for _, row in self.sessions.iterrows():
            entities = Entities(f"sub-{row.patient_id:03d}", f"ses-{row.visit_date}",
                                 "ieeg", {"task": "Stream"})
            path = f"{self.recordings_dir}/{row.internal_session_id}.edf"
            raw = mne.io.read_raw(path, preload=False, verbose=False)
            yield Recording(entities, raw, meta={"device": row.device})

repo.ingest(MyLabReader("sessions.csv", "./raw"))
```

`read()` can be a generator, as above, so nothing is loaded until `ingest()`
asks for it — memory use stays flat regardless of how large the source is.

Since {class}`~neurozarr.Writer` is the only thing that touches Zarr, a reader
cannot produce a malformed store however unusual its source is: it can only
describe items, never place them.

This covers a reader written for your own use. To have it selected
automatically by file extension, or to distribute it so others don't need
your source code to use it, see {doc}`extensions`.

## From a BIDS dataset

If your data already follows BIDS, {class}`~neurozarr.BidsReader` reads it
directly:

```python
from neurozarr import BidsReader

repo.ingest(BidsReader("./my_bids_dataset"))
```

It handles any valid dataset generically: subjects and sessions are discovered
by directory rather than requiring a `sessions.tsv`, datatype directories
(`ieeg`, `eeg`, `meg`, `beh`, `anat`, …) are walked without a fixed list, and
JSON sidecars are resolved through the BIDS **inheritance principle**, merging
applicable metadata from the dataset root toward the data file.

This is a convenience for data that happens to be in that form. The store's own
layout borrows BIDS conventions either way — see {doc}`layout`.

## Existing paths

Writes fail when the same entity path already exists. Make replacement intent
explicit:

```python
repo.ingest(reader, existing="skip")       # incremental import
repo.ingest(reader, existing="replace")    # deliberate rewrite
```
