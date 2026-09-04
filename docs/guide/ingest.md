# Getting data in

There are four ways to fill a store. They all produce the same result, so pick
whichever matches the shape your data is already in.

## From a pile of files, via a manifest

The general case: describe your files in a table, one row per file, and
{class}`~bidszarr.ManifestReader` does the rest. Nothing is assumed about folder
layout or naming.

| sub | ses | datatype | task | path |
|---|---|---|---|---|
| sub-001 | ses-1 | ieeg | Stream | /data/patient1_day1.edf |
| sub-001 | ses-1 | beh | Log | /data/patient1_log.csv |

```python
from bidszarr import Repo, ManifestReader

repo = Repo("./study.zarr")
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

And when a row needs handling the columns can't express, take over entirely:

```python
ManifestReader(df, row_reader=lambda row: my_custom_item(row))
```

Files are opened by extension: `.tsv`/`.csv` with pandas, and signal formats
through `mne.io.read_raw` (EDF, BDF, GDF, BrainVision, EEGLAB, FIF, CNT).
Anything else is recorded as a reference to the file rather than failing the
run.

## By hand

When you are building a dataset programmatically, or only have a few
recordings, skip the readers entirely:

```python
repo = Repo("./study.zarr")

subject = repo.create_subject("sub-001", attrs={"age": 63, "diagnosis": "PD"})
visit = subject.add_visit("ses-20220908", attrs={"device": "Percept PC"})

visit.add_recording(my_mne_raw, task="Stream", run=1)
visit.add_behavioral_table(my_dataframe, task="TherapyHistory")

repo.save("added sub-001")
```

Keyword arguments are entities, and decide where each item lands.

## By writing a reader

For a source format of your own, implement {class}`~bidszarr.Reader`: one
method, `read()`, yielding {class}`~bidszarr.Recording`,
{class}`~bidszarr.Table` and {class}`~bidszarr.Attrs` items.

```python
from bidszarr import Recording, Table, Attrs, Entities

class MyReader:
    def read(self):
        yield Attrs((), {"Name": "my study"})                      # dataset-wide
        yield Recording(Entities("sub-001", "ses-1", "ieeg", {"task": "Rest"}), my_raw)
        yield Table(Entities("sub-001", "ses-1", "beh", {}), "log", my_dataframe)

repo.ingest(MyReader())
```

There is no base class to inherit from — `Reader` is a `Protocol`, so anything
with a matching `read()` works. Since the writer is the only thing that touches
Zarr, a reader cannot produce a malformed store however unusual your source is.

## From a BIDS dataset

If your data already follows BIDS, {class}`~bidszarr.BidsReader` reads it
directly:

```python
from bidszarr import BidsReader

repo.ingest(BidsReader("./my_bids_dataset"))
```

It handles any valid dataset generically: subjects and sessions are discovered
by directory rather than requiring a `sessions.tsv`, datatype directories
(`ieeg`, `eeg`, `meg`, `beh`, `anat`, …) are walked without a fixed list, and
JSON sidecars are resolved through the BIDS **inheritance principle**, where the
nearest matching sidecar wins.

This is a convenience for data that happens to be in that form. The store's own
layout borrows BIDS conventions either way — see {doc}`layout`.
