# Editing and removing data

Fixing a typo'd attribute, renaming a session, or dropping an item you no
longer want doesn't require rewriting a store. Every level — the dataset, a
subject, a session, and each item — supports this the same way: nothing is
committed until {meth}`~neurozarr.Repo.save`, same as ingesting new data.

## Adding or updating metadata

`set_attrs` merges into whatever attrs already exist, at any level:

```python
repo = Repo.open("./study.zarr", mode="a")

repo.set_attrs({"Name": "my study"})                                    # dataset-wide
repo.subject("sub-001").set_attrs({"handedness": "right"})              # one subject
repo.subject("sub-001").visit("ses-1").set_attrs({"technician": "AB"})  # one session
rec = repo.subject("sub-001").visit("ses-1").recording(task="Rest")
rec.set_attrs({"note": "checked"})                                      # one recording

repo.save("annotated")
```

The same method exists on {class}`~neurozarr.TableView`,
{class}`~neurozarr.ArrayView`, and {class}`~neurozarr.ExternalFileView` —
whatever you can read back, you can annotate. It merges rather than replaces,
so calling it again only adds or overwrites the keys you pass.

## Renaming and deleting an item

{meth}`~neurozarr.RecordingView.rename`, `.rename()` on the other three view
types, and {meth}`~neurozarr.RecordingView.delete` work the same way:

```python
rec = repo.subject("sub-001").visit("ses-1").recording(task="Rest")
rec.rename("renamed-rec")
repo.save("renamed recording")

repo = Repo.open("./study.zarr", mode="a")
repo.subject("sub-001").visit("ses-1").recordings()[0].delete()
repo.save("deleted recording")
```

Neither zarr nor Icechunk has a move operation, so a rename copies the item
to its new key and deletes the old one — cheap within one subject's own
repository, but not instant for a very large recording. Once you've called
`rename()` or `delete()`, stop using that view: it still points at the old
location and now reads stale data. Get a fresh one from `recordings()`,
`tables()`, and so on after the next `save()`.

## Renaming and deleting a session

{meth}`~neurozarr.Subject.rename_visit` and
{meth}`~neurozarr.Subject.delete_visit` work on the whole session at once —
recordings, tables, everything under it:

```python
repo.subject("sub-001").rename_visit("ses-1", "ses-2")
repo.save("renamed session")

repo.subject("sub-001").delete_visit("ses-2")
repo.save("deleted session")
```

A session is just a group inside its subject's own repository, so this is
the same copy-then-delete mechanism as item rename, scoped to everything
under that one session.

## Removing a subject

{meth}`~neurozarr.Repo.delete_subject` removes a subject from the dataset's
published index — it stops appearing in {meth}`~neurozarr.Repo.subjects` and
{meth}`~neurozarr.Repo.find` — but does not touch that subject's own
repository or its version history:

```python
repo.delete_subject("sub-001")
repo.save("removed subject")
```

This is deliberately a soft removal, consistent with everything else here
being versioned rather than erased. There is no `rename_subject`: a subject
id is baked into its repository's storage location, and neither zarr nor
Icechunk can move that cheaply. To move a subject under a new id, read its
data back out, re-ingest it under the new id, and call `delete_subject` on
the old one.
