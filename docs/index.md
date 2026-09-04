# bidszarr

Convert neural and behavioral recordings into a **Zarr/Icechunk store that is always BIDS-shaped inside**, whatever shape the source data is in.

Point it at a BIDS folder, describe a pile of loose files in a table, or build a dataset up by hand — the output is always the same clean, versioned, cloud-ready structure.

```python
from bidszarr import Repo, BidsReader

repo = Repo("./study.zarr")
repo.ingest(BidsReader("./my_bids_dataset"))
repo.save("initial conversion")
```

Reading a source format and writing a well-structured store are two different problems, so they are two different things here: a **Reader** understands *your* data and yields standardized items; the **Writer** understands *BIDS structure* and is the only thing that touches Zarr. Add support for a new input format by writing a Reader — the output structure can't drift, because nothing else can write to the store.

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
guide/cloud
guide/derivatives
guide/performance
guide/cli
guide/layout
```

```{toctree}
:hidden:
:caption: Reference

api
```
