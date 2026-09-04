# Faster and repeat conversions

```python
from bidszarr.parallel import convert_parallel

convert_parallel("./BIDS", "./study.zarr", workers=6)   # one process per subject
repo.ingest(reader, skip_existing=True)                  # only write what's new
```

Subjects are independent repositories, so they convert concurrently — on the reference dataset that's 85s → 55s, bounded by the largest subject. `skip_existing` makes a re-run after new data arrives ~11× faster, since it writes only what isn't stored yet.
