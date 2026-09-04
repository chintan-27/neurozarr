# Getting data in

## From a BIDS folder

`BidsReader` handles any valid BIDS dataset generically: subjects and sessions are discovered by directory, datatype folders (`ieeg`, `eeg`, `meg`, `beh`, `anat`, …) are walked without a fixed list, and JSON sidecars are resolved through the BIDS **inheritance principle** (the nearest matching sidecar wins).

```python
repo = Repo("./study.zarr")
repo.ingest(BidsReader("./my_bids_dataset"))
repo.save("initial conversion")
```

## From a pile of files, via a manifest

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

## By writing a Reader

Neither built-in reader fits your source? Implement {class}`~bidszarr.Reader` — one method, `read()`, yielding {class}`~bidszarr.Recording`/{class}`~bidszarr.Table`/{class}`~bidszarr.Attrs`:

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

## By hand

```python
repo = Repo("./study.zarr")

subject = repo.create_subject("sub-001", attrs={"age": 63, "diagnosis": "PD"})
visit = subject.add_visit("ses-20220908", attrs={"device": "Percept PC"})

visit.add_recording(my_mne_raw, task="Stream", run=1)
visit.add_behavioral_table(my_dataframe, task="TherapyHistory")

repo.save("added sub-001")
```
