# Cloud storage

A store works the same wherever it lives. Pass a URI instead of a path and
everything else — ingesting, reading, versioning — is unchanged:

```python
Repo("s3://my-bucket/study", region="us-east-1")
Repo("gs://my-bucket/study")
Repo("az://account/container/study")
Repo("r2://my-bucket/study")
Repo("memory://scratch")              # in-memory, useful in tests
Repo("./study.zarr")                  # a local path
```

Any extra keyword arguments go straight through to the underlying Icechunk
storage constructor, which is where credentials and endpoint settings live:

```python
Repo("s3://my-bucket/study", region="us-east-1", from_env=True)
Repo("s3://open-data/study", region="us-east-1", anonymous=True)
```

If you need more control than that, build the storage yourself and hand it over:

```python
import icechunk

Repo(icechunk.s3_storage(bucket="my-bucket", prefix="study", ...))
```

Note that a storage built this way describes a single repository, so it holds
one subject rather than a whole store.

## Things that differ on object storage

Object stores can't be listed like a directory, so a store keeps its own index
of which subjects it contains, written when you call
{meth}`~bidszarr.Repo.save`. {meth}`~bidszarr.Repo.subjects` reads that index.

Because each subject is a separate repository, operations that span subjects —
{meth}`~bidszarr.Repo.find`, {meth}`~bidszarr.Repo.tag` — open a session per
subject. Over a network that cost is per subject rather than per recording, but
it is no longer free the way it is on local disk.

## Using memory:// in tests

`memory://` gives a store that never touches disk, which keeps tests fast and
leaves nothing to clean up:

```python
repo = Repo("memory://test")
repo.ingest(my_reader)
repo.save("fixture")
```

Stores under `memory://` are keyed by name within a process, so reopening the
same URI gets the same store back. They vanish when the process exits.
