# Converting faster, and converting again

## Converting subjects in parallel

Each subject is a separate repository, so subjects can be written at the same
time without coordinating — no shared session, no write conflicts. Pass
`workers` to {meth}`~neurozarr.Repo.ingest` to ingest that many subjects at once,
one process each:

```python
from neurozarr import Repo
from neurozarr.readers import BidsReader

repo = Repo.init("./study.zarr")
repo.add(BidsReader("./BIDS"), workers=6)
repo.commit("convert")
```

This is the same ingest as without `workers`, run side by side: each worker
writes its subject into a fork of that subject's session, the forks are merged
back, and `commit()` commits and publishes exactly as it does after a serial
ingest. Any reader with `subjects()` and `for_subject()` works — `BidsReader`,
`ManifestReader` and `SmartReader` all have them; a reader without them, or an
in-memory store, ingests serially. If a subject fails, nothing is published.

The explicit API (`visit.add_recording(...)`) stays serial on purpose: it is for
adding things by hand. To write custom logic at scale, put it in a reader — an
object whose `read()` yields items, plus `subjects()`/`for_subject()` — and pass
that to `ingest(..., workers=N)` rather than managing processes yourself.

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

## Choosing codecs

A reader decides how its data is compressed before anything is written: it probes
a few real files (one chunk each, spread across subjects), picks the shuffle mode
that stores each group of data smallest, and every writer applies that plan. It is
recorded in the dataset metadata as `codec_plan`, with the bytes measured.

```python
BidsReader("./BIDS", optimize="balanced", probe_level=2, probe_files=8)  # the defaults
```

| `optimize` | Codec | Use when |
|---|---|---|
| `speed` | Blosc lz4 | write and read speed matter most |
| `balanced` | Blosc zstd level 5 | default |
| `size` | Blosc zstd level 9 | storage cost matters most; writes are much slower |

`probe_level` sets how finely data is grouped — one codec decision per: `0`
nothing (zarr's default, no probing), `1` item type (Recording, Table, Array,
File), `2` item type × stored dtype (default), `3` item type × stored dtype ×
chunk shape, `4` each array, probed as it is written. Levels 1–3 fall back to the
next coarser level for a group the sample didn't see. On five real datasets level
2 came within 0.1–2.3% of probing every recording, and each dataset had only ~5
groups.

The explicit API (`visit.add_recording(...)`) doesn't probe: it uses
`CodecConfig(compressors=...)` if set, else zarr's default. Setting
`compressors` also overrides a reader's plan during `ingest`, with a warning.

## Controlling how data is packed

{class}`~neurozarr.CodecConfig` trades size, fidelity, and windowed-read cost
against each other. Pass one to {meth}`~neurozarr.Repo.create`:

```python
from neurozarr import Repo, CodecConfig

repo = Repo.init("./study.zarr", codec=CodecConfig(
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
repo.add(reader, existing="skip")
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
neurozarr add ./BIDS -j 6 --existing skip && neurozarr commit -m "new sessions"
```
