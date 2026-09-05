# Command line

Installing the package puts a `neurozarr` command on your path. Every command
takes `-v` to turn on debug logging.

## convert

Convert a source into a store.

```bash
neurozarr convert <source> <dest>
```

`source` is either a manifest `.csv`/`.tsv` or a BIDS dataset directory, chosen
by extension. `dest` is a path or a cloud URI.

These are the two sources the command line understands. The other ways of
filling a store — building it up by hand, or with a reader of your own — are
Python APIs, since they need code either way.

The source is validated first, and the conversion stops if anything fatal turns
up. Options:

| Option | Effect |
|---|---|
| `-m`, `--message` | Commit message. Defaults to `convert`. |
| `-j N`, `--workers N` | Convert N subjects in parallel. BIDS sources only. |
| `--existing {error,skip,replace}` | Duplicate policy. Defaults to safe failure. |
| `--reader NAME` | Select an installed reader explicitly. |
| `--dtype {int16,float16}` | How sample data is packed. |
| `-q`, `--quiet` | No progress bar or summary. |
| `--force` | Deprecated; malformed inputs cannot be forced through. |

A progress bar appears if `tqdm` is installed.

## validate

Check a source before converting it. Writes nothing.

```bash
neurozarr validate ./manifest.csv
neurozarr validate ./my_bids_dataset
neurozarr validate ./manifest.csv --json
```

Reports stable diagnostic codes, severity, source locations, and context.
Warnings such as preserved unsupported formats exit successfully; errors do not.

## info

Show what a store contains — subjects, and each one's visit, recording and table
counts.

```bash
neurozarr info ./study.zarr
```

## history

Show a store's commits and tags.

```bash
neurozarr history ./study.zarr
neurozarr history ./study.zarr --subject sub-002
```

History defaults to global dataset versions. `--subject` inspects the internal
history of one subject repository.

## doctor and migrate

Check manifest/snapshot integrity, or upgrade a readable 0.1 store explicitly:

```bash
neurozarr doctor ./study.zarr
neurozarr migrate ./old-study.zarr --dry-run
neurozarr migrate ./old-study.zarr
```

Migration writes metadata only; sample chunks are not rewritten.

## verify

Check that a store faithfully matches the source it came from, comparing sample
values and row counts.

```bash
neurozarr verify ./BIDS ./study.zarr
neurozarr verify ./BIDS ./study.zarr --limit 30    # quick spot check
```

Exits non-zero and lists the mismatches if any are found.

## export

Write a store back out as a BIDS dataset on disk, for tools that only read BIDS
from a filesystem.

```bash
neurozarr export ./study.zarr ./bids_again
neurozarr export ./study.zarr ./bids_again --subject sub-001
```

Recordings are written as BrainVision if `pybv` is installed, EDF if `edfio` is,
and otherwise FIF — which mne reads, but which is not BIDS-conformant for ieeg
or eeg data.
