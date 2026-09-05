# Cloud storage

A store works the same wherever it lives. Pass a URI instead of a path and
everything else — ingesting, reading, versioning — is unchanged:

```python
Repo.create("s3://my-bucket/study", region="us-east-1")
Repo.open("gs://my-bucket/study")
Repo.open("az://account/container/study")
Repo.open("r2://my-bucket/study")
Repo.create("memory://scratch")       # in-memory, useful in tests
Repo.open("./study.zarr")             # a local path
```

Any extra keyword arguments go straight through to the underlying Icechunk
storage constructor, which is where credentials and endpoint settings live:

```python
Repo.create("s3://my-bucket/study", region="us-east-1", from_env=True)
Repo.open("s3://open-data/study", region="us-east-1", anonymous=True)
```

If you need more control, pass a factory that builds the independent storage
prefix for each subject repository:

```python
import icechunk

def storage_for(sub_id):
    return icechunk.s3_storage(bucket="my-bucket", prefix=f"study/{sub_id}", ...)

Repo.create(storage_for)
```

A single pre-built `icechunk.Storage` is rejected because reusing it would alias
all subjects into one repository. Keep credentials out of URIs; pass them as
storage options so they are never persisted or logged.

## Things that differ on object storage

Object stores can't be listed like a directory, so a store keeps its own index
of which subjects it contains, written when you call
{meth}`~neurozarr.Repo.save`. {meth}`~neurozarr.Repo.subjects` reads that index.

Because each subject is a separate repository, reading matches from
{meth}`~neurozarr.Repo.find` opens the candidate subject repositories. The
dataset catalog prunes subjects that cannot match. Tags touch only the dataset
manifest, whose snapshot map pins the complete state.

## Using memory:// in tests

`memory://` gives a store that never touches disk, which keeps tests fast and
leaves nothing to clean up:

```python
repo = Repo.create("memory://test")
repo.ingest(my_reader)
repo.save("fixture")
```

Stores under `memory://` are keyed by name within a process, so reopening the
same URI gets the same store back. They vanish when the process exits.
