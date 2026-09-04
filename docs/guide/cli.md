# Command line

Installing the package puts a `bidszarr` command on your path. Every command
takes `-v` to turn on debug logging.

## convert

Convert a source into a store.

```bash
bidszarr convert <source> <dest>
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
| `--skip-existing` | Write only what isn't in the store yet. |
| `--dtype {int16,float16}` | How sample data is packed. |
| `-q`, `--quiet` | No progress bar or summary. |
| `--force` | Convert despite validation problems. |

A progress bar appears if `tqdm` is installed.

## validate

Check a source before converting it. Writes nothing.

```bash
bidszarr validate ./manifest.csv
bidszarr validate ./my_bids_dataset
```

Reports missing files, formats with no reader, missing manifest columns, and
sources that aren't shaped like BIDS. Exits non-zero if there are problems.

## info

Show what a store contains — subjects, and each one's visit, recording and table
counts.

```bash
bidszarr info ./study.zarr
```

## history

Show a store's commits and tags.

```bash
bidszarr history ./study.zarr
bidszarr history ./study.zarr --subject sub-002
```

History belongs to a subject, since each subject is a separate repository.
Defaults to the first subject in the store.

## verify

Check that a store faithfully matches the source it came from, comparing sample
values and row counts.

```bash
bidszarr verify ./BIDS ./study.zarr
bidszarr verify ./BIDS ./study.zarr --limit 30    # quick spot check
```

Exits non-zero and lists the mismatches if any are found.

## export

Write a store back out as a BIDS dataset on disk, for tools that only read BIDS
from a filesystem.

```bash
bidszarr export ./study.zarr ./bids_again
bidszarr export ./study.zarr ./bids_again --subject sub-001
```

Recordings are written as BrainVision if `pybv` is installed, EDF if `edfio` is,
and otherwise FIF — which mne reads, but which is not BIDS-conformant for ieeg
or eeg data.
