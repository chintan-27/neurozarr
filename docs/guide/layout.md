# How it works

## Readers and the writer

Reading a source format and writing a well-structured store are two different
problems, so they are two different things here.

A **reader** understands one source format. Its only job is to turn that source
into a stream of three standardized items:

| Item | What it is |
|---|---|
| {class}`~bidszarr.Recording` | One continuous run of sample data, as an mne `Raw`, plus its metadata |
| {class}`~bidszarr.Table` | A row-per-observation table, as a DataFrame |
| {class}`~bidszarr.Attrs` | Metadata with no data of its own, attached to a point in the tree |

The **writer** understands BIDS structure, and is the only code in the package
that touches Zarr. It decides where each item lands, how sample data is packed,
and how tables are stored.

That split is what keeps the output consistent. A reader cannot write a
malformed store because it cannot write to the store at all — it can only
describe what it found. Supporting a new source format means writing a reader
and changing nothing else.

{class}`~bidszarr.Entities` carries the BIDS entities for an item — subject,
session, datatype, and the rest — and decides the path it maps to.

## One repository per subject

A store is not a single repository. It is a directory of Icechunk
repositories, one per subject, plus a `_dataset` repository for metadata
belonging to no single subject:

```text
study.zarr/
  _dataset/            # dataset_description, participants field definitions
  sub-001/             # an independent Icechunk repository
    ses-20220908/
      ieeg/
        task-BrainSenseStream_acq-TD_run-1/
          data         # (channels x samples), int16 + per-channel scale/offset
          channels     # the channels table
          events       # if the recording has any
      beh/
        task-TherapyHistory/
          table
  sub-002/
  ...
```

Splitting by subject means one participant can be copied, shared or versioned
without moving the whole study, and it means subjects can be written
concurrently without coordinating, which is what makes
{func}`~bidszarr.parallel.convert_parallel` straightforward.

The trade-off is that anything spanning subjects — {meth}`~bidszarr.Repo.find`,
{meth}`~bidszarr.Repo.tag`, {meth}`~bidszarr.Repo.subjects` — has to visit every
repository. That is also why a store keeps its own index of which subjects it
holds: object stores cannot be listed like a directory.

Inside a subject's repository the tree starts at that subject's sessions. There
is no redundant `sub-001/` level, because the repository already *is* that
subject.

## How sample data is stored

Sample data is stored as int16 with a per-channel scale and offset, recorded in
the group's metadata. For data that arrived as integers, as EDF does, this is
lossless — the packing recovers exactly what the source held — and it compresses
considerably better than storing floats. Reading undoes it, so
{meth}`~bidszarr.RecordingView.data` hands back physical units either way.

A recording whose source carries no calibration information is stored as
float32 instead, rather than guessing at a scale and offset.

Arrays are chunked along the time axis, which is what allows a window to be read
without fetching the whole recording. Chunk size is a trade-off: larger chunks
compress a little better and make full reads faster, smaller ones make short
windowed reads cheaper. {class}`~bidszarr.CodecConfig` exposes both the byte
target and the sample cap that bound it.

Tables are stored as string arrays with each column's dtype recorded alongside,
so values cast back faithfully on read.
