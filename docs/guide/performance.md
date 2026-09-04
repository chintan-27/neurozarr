# Converting faster, and converting again

## Converting subjects in parallel

Each subject is a separate repository, so subjects can be written at the same
time without coordinating — no shared session, no write conflicts.
{func}`~bidszarr.parallel.convert_parallel` runs one worker process per subject:

```python
from bidszarr.parallel import convert_parallel

convert_parallel("./BIDS", "./study.zarr", workers=6)
```

Reading signal files and compressing them are both CPU-bound, which is why this
uses processes rather than threads.

A subject is the unit of work, so the total time is bounded by the largest
single subject no matter how many workers you give it. If one participant holds
most of a dataset, expect the speedup to flatten out well before `workers`
reaches the subject count.

Dataset-level metadata is written once by the parent process rather than by the
workers, so they never race to write the same shared metadata.

## Re-running a conversion

Converting is repeatable — running it again over the same destination rewrites
what's there rather than failing or duplicating. When new data has arrived and
you only want to add it, pass `skip_existing`:

```python
repo.ingest(reader, skip_existing=True)
```

This checks each incoming item against what the store already holds and writes
only the ones that are missing. The saving is roughly proportional to how much
of the dataset is unchanged, so it pays off most when a large store gains a
small amount of new data.

On the command line both are flags:

```bash
bidszarr convert ./BIDS ./out -j 6 --skip-existing
```
