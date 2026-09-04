# neurozarr

Put neural and behavioral recordings into a **versioned, cloud-ready Zarr/Icechunk store** — whatever shape your source data is in.

Add recordings one at a time, describe a pile of files in a table, or write a reader for your own format. However the data goes in, it comes out with the same predictable structure, a full version history, and fast reads of any slice of it.

```python
from neurozarr import Repo

repo = Repo("./study.zarr")

visit = repo.create_subject("sub-001").add_visit("ses-1")
visit.add_recording(my_raw, task="Rest", run=1)
repo.save("first recording")

rec = repo.subject("sub-001").visit("ses-1").recording(task="Rest", run=1)
values, meta = rec.data(tmin=10, tmax=20)     # ten seconds, without reading the rest
```

Inside the store, data is filed by subject → session → datatype → entities, following BIDS naming conventions. That is a choice about the *output*, not a requirement on your *input* — see {doc}`guide/layout`. You do not need BIDS-formatted data to use this; if you happen to have some, there is a reader for it.

```{toctree}
:hidden:
:caption: Getting started

installation
guide/ingest
```

```{toctree}
:hidden:
:caption: User guide

guide/read
guide/versioning
guide/derivatives
guide/cloud
guide/performance
guide/cli
guide/layout
```

```{toctree}
:hidden:
:caption: Reference

api
changelog
```
