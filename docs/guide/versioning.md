# Versioning

Every store is an Icechunk repository, so it keeps its history rather than
overwriting it. Nothing is written permanently until you commit:

```python
repo.ingest(reader)
repo.save("initial conversion")     # commit
```

{meth}`~neurozarr.Repo.save` commits each subject that changed and skips the
rest, so re-saving after touching one subject doesn't churn the others.

## Looking at history

```python
repo.history()          # [(snapshot_id, message, written_at), ...] newest first
```

A store spans one repository per subject, so history belongs to a subject rather
than the store as a whole. {meth}`~neurozarr.Repo.history` reads the first
subject by default; pass `sub_id` for a particular one.

## Naming a state

Tags name a state so you can come back to it:

```python
repo.tag("v1")
repo.tags()             # ['v1']
```

Because a version of the dataset spans every subject's repository,
{meth}`~neurozarr.Repo.tag` applies the same name to all of them.

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
