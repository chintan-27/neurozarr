# bidszarr

Convert neural and behavioral recordings into a **Zarr/Icechunk store that is always BIDS-shaped inside**, whatever shape the source data is in.

Point it at a BIDS folder, describe a pile of loose files in a table, or build a dataset up by hand — the output is always the same clean, versioned, cloud-ready structure.

```python
from bidszarr import Repo, BidsReader

repo = Repo("./study.zarr")
repo.ingest(BidsReader("./my_bids_dataset"))
repo.save("initial conversion")
```

## Install

```bash
pip install -e .          # from a checkout
pip install -e ".[dev]"   # plus pytest and tqdm
```

Requires Python ≥ 3.11. Dependencies: `icechunk`, `zarr`, `mne`, `pandas`, `numpy`.

## Why

Reading a source format and writing a well-structured store are two different problems, so they are two different things here:

- A **Reader** understands *your* data and yields standardized items (`Recording`, `Table`, `Attrs`).
- The **Writer** understands *BIDS structure* and is the only thing that touches Zarr.

Add support for a new input format by writing a Reader — the output structure is guaranteed to stay consistent, because nothing else can write to the store.

## Three ways to get data in

### 1. From a BIDS folder

`BidsReader` handles any valid BIDS dataset generically: subjects and sessions are discovered by directory, datatype folders (`ieeg`, `eeg`, `meg`, `beh`, `anat`, …) are walked without a fixed list, and JSON sidecars are resolved through the BIDS **inheritance principle** (the nearest matching sidecar wins).

```python
repo = Repo("./study.zarr")
repo.ingest(BidsReader("./my_bids_dataset"))
repo.save("initial conversion")
```

### 2. From a pile of files, via a manifest

No standard layout? Describe your files in a table — one row per file — and `ManifestReader` does the rest.

| sub | ses | datatype | task | path |
|---|---|---|---|---|
| sub-001 | ses-1 | ieeg | Stream | /data/patient1_day1.edf |
| sub-001 | ses-1 | beh | Log | /data/patient1_log.csv |

```python
from bidszarr import ManifestReader

repo.ingest(ManifestReader("manifest.csv"))
```

Columns beyond the reserved ones (`sub`, `ses`, `datatype`, `name`, `meta`, `path`) become BIDS entities, so `task`/`run`/`acq` land where they should. Column names are configurable, and a `row_reader` callback handles anything unusual:

```python
ManifestReader(df, sub_col="subject", path_col="filepath")     # your column names
ManifestReader(df, row_reader=lambda row: my_custom_item(row))  # full control per row
```

Files are opened by extension: `.tsv`/`.csv` with pandas, signal formats with `mne.io.read_raw` (EDF, BDF, GDF, BrainVision, EEGLAB, FIF, CNT). Anything else is recorded as a reference rather than crashing the run.

### 3. By writing a Reader

Neither built-in reader fits your source? Implement `bidszarr.Reader` — one method, `read()`, yielding `Recording`/`Table`/`Attrs` (see `bidszarr/items.py`):

```python
from bidszarr import Recording, Table, Attrs, Entities

class MyReader:
    def read(self):
        yield Attrs((), {"Name": "my study"})                      # dataset-wide
        yield Recording(Entities("sub-001", "ses-1", "ieeg", {"task": "Rest"}), my_raw)
        yield Table(Entities("sub-001", "ses-1", "beh", {}), "log", my_dataframe)

repo.ingest(MyReader())
```

The Writer is the only thing that touches Zarr, so however you read your source, the output is guaranteed BIDS-shaped.

### 4. By hand

```python
repo = Repo("./study.zarr")

subject = repo.create_subject("sub-001", attrs={"age": 63, "diagnosis": "PD"})
visit = subject.add_visit("ses-20220908", attrs={"device": "Percept PC"})

visit.add_recording(my_mne_raw, task="Stream", run=1)
visit.add_behavioral_table(my_dataframe, task="TherapyHistory")

repo.save("added sub-001")
```

## Reading data back

```python
repo = Repo("./study.zarr")

repo.subjects()                       # ['sub-001', 'sub-002', ...]
subject = repo.subject("sub-001")
subject.attrs                         # {'age': 63, ...}
subject.visits()                      # ['ses-20220908', ...]

rec = subject.visit("ses-20220908").recording(task="Stream", run=1)
values, meta = rec.data()             # ndarray in physical units + its metadata
raw = rec.raw()                       # a real mne.io.RawArray, ready for mne
df = rec.channels()                   # the channels table as a DataFrame
events = rec.events()                 # events / annotations

# search across every subject in the store
for rec in repo.find(task="BrainSenseStream", acq="TD"):
    print(rec.path, rec.shape)
```

Recordings are stored as int16 with per-channel scale/offset; `.data()` and `.raw()` undo that for you, so what you read back matches the source to floating-point precision. Table columns keep their dtypes — numbers come back as numbers. `mne` annotations are stored as an events table and put back on the `Raw` when you read it.

### Reading only part of a recording

Data is chunked along time, so you can pull a window without fetching the whole array — the point of storing it this way:

```python
values, meta = rec.data(tmin=10, tmax=20)          # ten seconds, by time
values, meta = rec.data(start=1000, stop=2000)     # or by sample index
values, meta = rec.data(tmin=10, tmax=20, picks=["LFP_L"])   # one channel
raw = rec.raw(tmin=10, tmax=20)                    # same window as an mne.Raw

rec.array          # the underlying zarr array, slice it yourself
rec.duration       # seconds
rec.sfreq          # sampling rate
```

On a 3708-second recording, reading a 10-second window this way is ~8× faster than reading the whole thing.

## Versioning

Every store is an Icechunk repository, so history is free:

```python
repo.save("reprocessed with new filter")   # a commit
repo.history()                             # [(snapshot_id, message, timestamp), ...]
repo.tag("v1")                             # name this state
repo.tags()                                # ['v1']

subject.recordings(version="v1")           # read the data as it was at v1
```

## Cloud storage

Anywhere Icechunk can write, this can write — pass a URI instead of a path:

```python
Repo("s3://my-bucket/study", region="us-east-1")
Repo("gs://my-bucket/study")
Repo("az://account/container/study")
Repo("memory://scratch")                   # in-memory, handy for tests
Repo(icechunk.s3_storage(...))             # or a Storage you built yourself
```

Extra keyword arguments (`region=`, `anonymous=`, `from_env=`, …) pass straight through to Icechunk.

## Storing processed results

Analysis outputs go under `derivatives/`, kept separate from raw data the way BIDS does it:

```python
filtered = raw.copy().filter(l_freq=1, h_freq=40)
visit.add_derivative("my-filter", filtered, task="Stream", run=1)
visit.add_derivative("my-stats", stats_dataframe, datatype="beh", task="Stream")
```

They read back like anything else — `subject.recordings()` returns them with `derivatives/my-filter/` in the path.

## Faster and repeat conversions

```python
from bidszarr.parallel import convert_parallel

convert_parallel("./BIDS", "./study.zarr", workers=6)   # one process per subject
repo.ingest(reader, skip_existing=True)                  # only write what's new
```

Subjects are independent repositories, so they convert concurrently — on the reference dataset that's 85s → 55s, bounded by the largest subject. `skip_existing` makes a re-run after new data arrives ~11× faster, since it writes only what isn't stored yet.

## Command line

```bash
bidszarr validate ./my_bids_dataset          # check before converting
bidszarr convert ./my_bids_dataset ./out     # BIDS folder or manifest.csv
bidszarr convert ./BIDS ./out -j 6           # one worker per subject
bidszarr convert ./BIDS ./out --skip-existing   # only what's new
bidszarr info ./out                          # subjects, visits, counts
bidszarr history ./out                       # versions and tags
bidszarr verify ./BIDS ./out                 # confirm the store matches its source
bidszarr export ./out ./bids_again           # write the store back out as BIDS
```

`convert` takes `--dtype {int16,float16}`, `-m` for the commit message, `-q` to quiet it, and `--force` to convert despite validation warnings. `-v` turns on debug logging.

`export` writes BrainVision if `pybv` is installed, EDF if `edfio` is, and otherwise FIF (readable by mne, but not BIDS-conformant for ieeg/eeg).

## How the store is laid out

**One Icechunk repository per subject**, so a single subject can be shared, copied, or versioned on its own without shipping the whole study:

```
study.zarr/
  _dataset/            # dataset-wide metadata (dataset_description, participants info)
  sub-001/             # an independent Icechunk repo
    ses-20220908/
      ieeg/
        task-BrainSenseStream_acq-TD_run-1/
          data         # (channels x samples) int16 + scale/offset
          channels     # the channels table
          events       # if present
      beh/
        task-TherapyHistory/
          table
  sub-002/
  ...
```

Inside a subject's repo the tree starts at its sessions — no redundant `sub-001/` level, because the repository already *is* that subject.

## Development

```bash
pip install -e ".[dev]"
pytest                    # ~30 tests, all in-memory, no files touched
```

Tests run against `memory://` stores, so they are fast and leave nothing behind.

## Docs

This README *is* the docs, rendered as a website with the API reference alongside it:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build
open docs/_build/index.html
```
