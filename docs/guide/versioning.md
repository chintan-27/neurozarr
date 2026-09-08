# Versioning

Every store is an Icechunk repository, so it keeps its history rather than
overwriting it. Nothing is written permanently until you commit:

```python
repo.ingest(reader)
dataset_version = repo.save("initial conversion")
```

{meth}`~neurozarr.Repo.save` commits each changed subject, then atomically
publishes a dataset manifest that pins every subject to an exact snapshot. The
returned ID identifies that complete dataset state. This applies equally to
ingesting new data, building it up by hand, and the edits and deletions
covered in {doc}`edit` — none of it is permanent until `save()`.

## Transactions

{meth}`~neurozarr.Repo.transaction` commits on a clean exit and discards
everything on an exception, so a batch of changes either all land or none do:

```python
with repo.transaction("add second visit"):
    repo.subject("sub-001").add_visit("ses-2").add_recording(raw, task="Rest")
```

Call {meth}`~neurozarr.Repo.abort` directly for the same discard without a
`with` block — useful after catching an error yourself, or after a
{class}`~neurozarr.WriteConflictError` you plan to retry:

```python
repo.subject("sub-001").add_visit("ses-3")
repo.abort()   # nothing from this Repo instance since the last save() happened
```

Either way, discarding only affects sessions this `Repo` instance opened but
never saved — it cannot undo a version that was already committed. To go back
to an earlier committed state, read it with `version=` below, or start a new
`save()` that writes over it.

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

See {doc}`edit` for changing or removing data that's already there — it
follows the same save-to-commit model as everything above.
