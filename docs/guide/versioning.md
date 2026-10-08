# Versioning

A store keeps its history the way git does. Changes are first *staged* — added
from a source, or made item by item — and become permanent only when you
*commit* them:

```python
repo.add("./BIDS")                     # stage
repo.subject("sub-001").visit("ses-1").add_recording(raw, task="Rest")  # stage more
repo.status()                          # what is staged
dataset_version = repo.commit("initial conversion")
```

{meth}`~neurozarr.Repo.commit` commits each changed subject, then atomically
publishes a dataset version that pins every subject to an exact snapshot — so one
version id names a complete, consistent dataset even though each subject is its
own repository. This applies equally to adding new data, building it up
explicitly, and the edits and deletions covered in {doc}`edit`.

## The staging area

Staged changes are kept in a local staging area (git's index), shared by Python
and the `neurozarr` command, so they survive the process that made them:

```python
Repo.open("./study.zarr", mode="a").add("./BIDS")   # stage in one script ...
```

```bash
neurozarr -C ./study.zarr status       # ... see it from the command line ...
neurozarr -C ./study.zarr commit -m "import"   # ... and commit it there
```

{meth}`~neurozarr.Repo.reset` unstages everything, or some subjects:

```python
repo.reset("sub-002")
repo.reset()
```

Chunks already written for unstaged data stay in storage, unreferenced, as git
keeps unreachable objects. Unstaging cannot undo a version already committed.

## Transactions

{meth}`~neurozarr.Repo.transaction` commits on a clean exit and unstages
everything on an exception, so a batch of changes either all land or none do:

```python
with repo.transaction("add second visit"):
    repo.subject("sub-001").add_visit("ses-2").add_recording(raw, task="Rest")
```

## When someone else committed first

Each person's staging area starts from the version current when they staged. If
the branch has moved on by the time you commit, neurozarr compares what you
staged with what changed since: changes to different data both land; if you both
changed the same data — the same chunks of an array, or the same item's
metadata — `commit` raises {class}`~neurozarr.WriteConflictError` and commits
nothing. Unstage that subject and add it again on top of the new version.

## Looking at history

```python
repo.log()                  # [(version_id, message, written_at), ...] newest first
repo.log("sub-001")         # one subject's own history
repo.show()                 # the current version: message, subjects changed, codec plan
repo.show("v1")
```

## Naming versions

A version can be named by a branch, a tag, its id or a unique prefix of it (at
least 6 characters), with `~N` for N versions back — {meth}`~neurozarr.Repo.resolve`
turns any of these into a version id:

```python
repo.tag("v1")              # name the current version
repo.tag("v0", "main~3")    # name an earlier one
repo.tags()                 # ['v0', 'v1']
repo.resolve("main~1")
```

A tag is applied only to the dataset manifest; its pinned subject snapshots make
the tag atomic even though subjects use independent repositories.

## Branches

Work that shouldn't touch the main dataset yet — a reprocessing run, a cleanup —
goes on a branch:

```python
repo.switch("cleanup", create=True)     # or repo.branch("cleanup") to create only
repo.rm("sub-03", recursive=True)
repo.commit("drop sub-03")
repo.switch("main")                     # main never saw it
repo.branches()                         # ['cleanup', 'main']
repo.delete_branch("cleanup", force=True)
```

`repo.head` is the current branch; it is shared with the command line, like
git's HEAD.

## Comparing versions

```python
repo.diff()                  # what is staged
repo.diff("v1")              # v1 against the current version
repo.diff("v1", "v2")
```

Each changed subject is reported as added, removed or changed, with the items
added / changed / removed, metadata changes, and the number of chunks rewritten.

## Merging

```python
repo.merge("cleanup")                         # into the current branch
repo.merge("cleanup", theirs=("sub-01",))     # settle a conflict for sub-01
```

A branch the current one hasn't moved away from is fast-forwarded. Otherwise each
subject is merged against the common ancestor: changed on one side, that side is
taken without copying data; changed on both, the changes are combined item by item
when they don't overlap; overlapping changes raise
{class}`~neurozarr.MergeConflictError` listing them per subject, and nothing is
written until each is settled with `ours=` / `theirs=`. The merged-in branch is
recorded with the merge, so merging it again later only brings what is new.

## Copying and cleaning up

```python
Repo.clone("./study.zarr", "./backup.zarr")                           # everything, local
Repo.clone("lab", "./v1.zarr", version="v1", shallow=True)            # one version, anywhere
repo.gc()                       # report unreachable data
repo.gc(dry_run=False)          # delete it
```

{meth}`~neurozarr.Repo.gc` keeps every version on a branch or tag with all the data
it pins, and only deletes objects older than its cutoff (14 days by default).

## Undoing

{meth}`~neurozarr.Repo.revert` undoes a version with a new one, keeping history:
every subject the version changed is re-pinned as it was before it, so no data is
copied. It refuses when a later version changed one of those subjects again.

```python
repo.revert("main~1")
```

{meth}`~neurozarr.Repo.reset_hard` moves the current branch back to a version
and drops what is staged; later versions stay reachable from other branches or
tags.

```python
repo.reset_hard("v1")
```

## Reading an earlier version

Anything that reads takes a `version`:

```python
subject.recordings(version="v1")
subject.visits(version="v1")
repo.ls("sub-001", version="main~2")
repo.root_of("sub-001", version="v1")
```

This reads the data as it was, leaving the current state untouched — useful for
comparing a reprocessing run against what it replaced.

On the command line, see {doc}`cli`: `neurozarr log`, `show`, `ls --version`,
`tag`.
