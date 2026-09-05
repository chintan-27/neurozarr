# Versioning

Every store is an Icechunk repository, so it keeps its history rather than
overwriting it. Nothing is written permanently until you commit:

```python
repo.ingest(reader)
dataset_version = repo.save("initial conversion")
```

{meth}`~neurozarr.Repo.save` commits each changed subject, then atomically
publishes a dataset manifest that pins every subject to an exact snapshot. The
returned ID identifies that complete dataset state.

## Looking at history

```python
repo.history()          # [(snapshot_id, message, written_at), ...] newest first
```

By default history is the global dataset history. Pass `sub_id` to inspect the
internal commit history of one subject repository.

## Naming a state

Tags name a state so you can come back to it:

```python
repo.tag("v1")
repo.tags()             # ['v1']
```

A tag is applied only to the dataset manifest; its pinned subject snapshot IDs
make the tag atomic even though subjects use independent repositories.

## Reading an earlier state

Anything that reads takes a `version`, either a tag or a snapshot id:

```python
subject.recordings(version="v1")
subject.visits(version="v1")
repo.root_of("sub-001", version="v1")
```

This reads the data as it was, leaving the current state untouched — useful for
comparing a reprocessing run against what it replaced.

On the command line:

```bash
neurozarr history ./study.zarr
```
