# Command line

The `neurozarr` command follows git: stage changes with `add`, record them with
`commit`, and inspect with `status`, `log`, `show`, `ls` and `find`. Every
command is a thin shell over the {class}`~neurozarr.Repo` method of the same
name, and shares its staging area and current branch with Python.

```bash
neurozarr init ./study.zarr          # create a store
cd study.zarr                        # commands find the store from here, like git
neurozarr add ~/data/BIDS -j 8       # stage a source
neurozarr status                     # what is staged
neurozarr commit -m "initial import" # record it as a new dataset version
neurozarr log --oneline
```

## Which store

Every command works on one store, found in this order:

1. `-C STORE` — a path, a URI (`s3://`, `gs://`, `az://`, `r2://`) or a remote name.
2. The `NEUROZARR_STORE` environment variable.
3. The store the current directory is in (this directory or a parent holding a
   `_dataset` repository).

`-o key=value` passes a storage option for the store, e.g. `-o region=us-east-1`;
`-v` turns on debug logging.

## Remotes

A remote saves a store's URI with its storage options under a short name, kept in
`~/.config/neurozarr/remotes.json` — credentials are never stored there; use
options like `profile=` or `from_env=true`.

```bash
neurozarr remote add lab s3://lab-bucket/study -o region=us-east-1 -o from_env=true
neurozarr remote                     # list
neurozarr -C lab status
neurozarr remote remove lab
```

## init

```bash
neurozarr init                       # the current directory
neurozarr init s3://lab-bucket/study -o region=us-east-1
```

## add

Stage everything a source holds. The source is validated first; if anything
fatal turns up, nothing is staged.

```bash
neurozarr add ./BIDS                 # a BIDS folder
neurozarr add ./manifest.csv         # a manifest table (.csv/.tsv/.json)
neurozarr add /siteA /siteB          # several folders, any layout: the smart reader
neurozarr add ./BIDS -n              # dry run: show the reader and codec plan only
```

| Option | Effect |
|---|---|
| `--reader NAME` | `bids`, `manifest`, `smart`, or an installed plugin (default: by source) |
| `--answers FILE` | smart reader: its saved answers; asked for interactively and saved |
| `--optimize {speed,balanced,size}` | codec trade-off (see {doc}`performance`) |
| `--probe-level {0..4}`, `--probe-files N` | how codecs are probed |
| `--dtype {int16,float16}` | how sample data is packed |
| `-j N` | stage N subjects in parallel, with any reader |
| `--existing {error,skip,replace}` | when an item already exists (default: error) |
| `-n`, `--dry-run` | stage nothing |
| `-q` | no progress bar or summary |

Repeated `add`s accumulate. Chunks are written to the store when staged (as git
writes objects at `add`); the staging area itself is small and local — see below.

## rm and mv

```bash
neurozarr rm -r sub-03                           # a subject (or session) needs -r
neurozarr rm sub-01/ses-1/ieeg/task-rest_run-1   # one item
neurozarr mv sub-01/ses-1 ses-baseline           # rename in place
```

A subject can't be renamed: its id is its repository's location.

## status, reset, commit

```bash
neurozarr status          # per subject: items added / changed / removed
neurozarr status -l       # list every item
neurozarr reset sub-02    # unstage one subject; `reset` alone unstages everything
neurozarr commit -m "..." # record everything staged as one dataset version
```

If the branch moved since you staged and someone else changed the same data (the
same chunks, or the same item's metadata), `commit` refuses and commits nothing:
`reset` that subject and `add` it again. Changes to different data both land.

## log, show, ls, find

```bash
neurozarr log                        # versions on the current branch
neurozarr log --oneline -n 5
neurozarr log --subject sub-002      # one subject's own history
neurozarr show                       # the current version: message, subjects changed, codec plan
neurozarr show v1                    # any version
neurozarr ls                         # subjects
neurozarr ls sub-01                  # its sessions
neurozarr ls sub-01/ses-1 --version v1
neurozarr find task=rest datatype=ieeg
```

A version is a branch, a tag, a version id or a unique prefix of one (6+
characters), optionally followed by `~N` for N versions back: `main~1`, `v1~2`.
`log`, `show`, `status`, `ls` and `find` take `--json`.

## branch, switch

```bash
neurozarr branch                     # list; * marks the current one
neurozarr branch cleanup             # create at the current version
neurozarr branch v2-pipeline v1      # create at another version
neurozarr switch cleanup             # make it current
neurozarr switch -c trial main~2     # create and switch
neurozarr branch -d cleanup          # delete (its versions are on another branch)
neurozarr branch -D trial --yes      # delete even though that drops versions
```

A branch is cheap: a pointer in the store, plus one in each subject repository the
first time that subject is written on it. Staged changes belong to the branch they
were staged on and come back when you switch back.

## diff

```bash
neurozarr diff                       # what is staged
neurozarr diff v1                    # v1 against the current version
neurozarr diff main~3 main --stat    # counts only
neurozarr diff v1 --subject sub-02 --json
neurozarr diff v1 --exit-code        # exit 1 if anything differs (for scripts)
```

Per subject: added, removed or changed, with the items added / changed / removed,
metadata changed, and how many chunks were rewritten.

## revert, reset --hard

```bash
neurozarr revert main~1              # undo that version with a new one; history kept
neurozarr reset --hard v1 --yes      # move the current branch back to v1
```

`revert` puts every subject the version changed back as it was before it, copying
no data; it refuses if a later version changed one of those subjects again.
`reset --hard` drops later versions from this branch (they stay reachable from
other branches and tags) and drops what is staged, so it needs `--yes`.

## merge

```bash
neurozarr merge cleanup                      # bring cleanup's changes into the current branch
neurozarr merge cleanup --theirs sub-01      # settle a conflict: take sub-01 from cleanup
neurozarr merge cleanup --ours sub-01        # ... or keep this branch's sub-01
```

A branch the current one hasn't moved away from is fast-forwarded. Otherwise each
subject is merged against the two branches' common ancestor: a subject changed on
one side takes that side (no data copied); one changed on both is merged item by
item when the changes don't overlap (the other side's changed chunks are copied);
overlapping changes are a conflict, reported per subject, and nothing is written
until each is settled with `--ours` / `--theirs`.

## clone

```bash
neurozarr clone ./study.zarr ./backup.zarr                    # everything (local paths)
neurozarr clone lab ./snapshot.zarr --depth 1 --version v1    # one version, any locations
```

## gc

```bash
neurozarr gc                         # report what could be deleted
neurozarr gc --yes                   # delete it
neurozarr gc --older-than 30 --yes
```

Deletes stored data nothing can reach any more — versions dropped by
`reset --hard` or a deleted branch, chunks of changes that were unstaged — and
only objects older than `--older-than` days (default 14), so recent staged work is
safe. Every version on a branch or tag is kept, with all the data it pins.

## query

```bash
neurozarr query "datatype == 'ieeg' and sfreq >= 250"      # matching paths
neurozarr query "participant.age < 40" --kind recording --count
neurozarr query --where task=rest --json --columns sfreq,duration_s
neurozarr query "exists(meta.Manufacturer)" --version v1 --csv
```

See {doc}`read` for the expression language and the columns.

## tag

```bash
neurozarr tag                 # list
neurozarr tag v1              # name the current version
neurozarr tag v0 main~3       # name an earlier one
neurozarr tag -d v0
```

## validate, verify, export, migrate, fsck

```bash
neurozarr validate ./BIDS            # check a source before adding it; --json for codes
neurozarr verify ./BIDS              # check the store matches its source (--limit N to spot-check)
neurozarr export ./bids_again        # write the store back out as BIDS (--subject to limit)
neurozarr migrate --dry-run          # preview upgrading a 0.1 store
neurozarr fsck                       # check schema and snapshot integrity
```

`export` writes BrainVision if `pybv` is installed, EDF if `edfio` is, and
otherwise FIF — which mne reads, but which is not BIDS-conformant for ieeg or eeg.

## Local state

Like git's `.git/HEAD` and `.git/index`, the current branch and the staging area
are yours, kept on your machine — never in the store, which may be shared or in
the cloud: `$XDG_STATE_HOME/neurozarr/` (by default `~/.local/state/neurozarr/`).
Staging on one machine and committing on another doesn't work, and two people
staging into one store have separate staging areas — as with git.
