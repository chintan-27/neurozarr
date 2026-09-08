# Converting faster, and converting again

## Converting subjects in parallel

Each subject is a separate repository, so subjects can be written at the same
time without coordinating — no shared session, no write conflicts.
{func}`~neurozarr.parallel.convert_parallel` runs one worker process per subject:

```python
from neurozarr.parallel import convert_parallel

convert_parallel("./BIDS", "./study.zarr", workers=6)
```

Reading signal files and compressing them are both CPU-bound, which is why this
uses processes rather than threads.

Recordings are opened without preloading. The writer requests one destination
chunk at a time from MNE, so peak memory is governed by `CodecConfig` rather
than by the largest recording.

A subject is the unit of work, so the total time is bounded by the largest
single subject no matter how many workers you give it. If one participant holds
most of a dataset, expect the speedup to flatten out well before `workers`
reaches the subject count.

Dataset-level metadata is written once by the parent process rather than by the
workers, so they never race to write the same shared metadata.

## Controlling how data is packed

{class}`~neurozarr.CodecConfig` trades size, fidelity, and windowed-read cost
against each other. Pass one to {meth}`~neurozarr.Repo.create`:

```python
from neurozarr import Repo, CodecConfig

repo = Repo.create("./study.zarr", codec=CodecConfig(
    bitround_k=4,                        # drop 4 low bits before compressing -- lossy, smaller
    chunk_target_bytes=2 * 1024 * 1024,  # smaller chunks: cheaper short windowed reads
    max_chunk_samples=20_000,
))
```

`dtype="int16"` (the default) stores integers with a per-channel scale and
offset, which is lossless for data that arrived as integers, as EDF does, and
compresses best; `bitround_k` only applies there. `dtype="float16"` stores
physical values rounded to half precision instead — use one or the other, not
both. A recording whose source carries no calibration is always stored as
float32, regardless of `dtype`. `chunk_target_bytes` and `max_chunk_samples`
together bound chunk size: larger chunks compress a little better and make
full reads faster, smaller ones make short windowed reads cheaper — see
{doc}`read` for what a windowed read actually fetches.

## Re-running a conversion

Existing paths fail by default. When new data has arrived and you only want to
add it, choose the skip policy:

```python
repo.ingest(reader, existing="skip")
```

This checks each incoming item against what the store already holds and writes
only the ones that are missing. The saving is roughly proportional to how much
of the dataset is unchanged, so it pays off most when a large store gains a
small amount of new data.

When you specifically want to overwrite what's there instead — reprocessing
with different settings, say — use `existing="replace"` in place of `"skip"`;
{meth}`~neurozarr.Visit.add_recording` and the other `add_*` methods take the
same argument for one item at a time.

On the command line both are flags:

```bash
neurozarr convert ./BIDS ./out -j 6 --existing skip
```
