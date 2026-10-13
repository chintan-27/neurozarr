# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The summary is spliced at commit instead of re-encoded: at 10,000 subjects a commit's
  summary update went from 2.9 s to 34 ms, and a full build from 23 s to 0.8 s (item
  paths no longer go through a Bloom filter; a path's subject is its first segment).
  `Table.paths` no longer copies, which made `paths[i]` in a loop quadratic -- including
  in every query over many subjects.
- int16 recordings unpack in place: one float64 allocation instead of three.
- Reading a recording in blocks (`data(start, stop)` in a loop) decoded the whole chunk
  under each block again. A view now keeps its last few decoded chunks and opens its data
  array once: a 10 s-block scan is 13-17x faster on BRAVO and CHB-MIT, 2.9x on HBN and
  1.5x on Pediatric iEEG (HiPerGator), with identical values; full reads are unchanged.
  On top of that: int16 is unpacked and scaled in one pass, the next chunk is decoded in
  the background while a forward read uses the current one, and a window straddling two
  chunks copies only its two pieces -- the Pediatric scan drops again, 9.8 s to 5.7 s
  (same node), now as fast as reading the whole recording at once. A forward read now
  decodes the next 3 chunks side by side, and `data(dtype=np.float32)` returns half the
  memory for int16 recordings (Pediatric-shaped local test: scan 0.81 to 0.43 s).
- Reading a subset of channels selects a plain range of rows instead of a list (zarr's
  orthogonal selection): one channel had been slower to read than all of them (RNS
  17.6 vs 2.4 ms).
- `find()` and iterating query results build each view without re-decoding its row:
  6,019 hits on RNS 80 to 28 ms.

### Added

- A columnar index in every version, replacing the manifest's JSON `catalog` (schema 3;
  `neurozarr migrate` upgrades a schema-2 store with one walk per subject). Per subject:
  one stored object holding a row per item (path, kind, session, datatype, entities,
  sfreq, n_channels, n_samples, duration, dtype, shape, chunks, codec, `meta.*`),
  dictionary-encoded and sorted by path. Per version: a summary with a row per subject
  (pin, index digest, item counts, min/max zone maps, value sets or Bloom filters,
  `participant.*`). Built at commit from what changed, never by re-scanning: the writer
  captures each item's row as it writes it (rows travel with staged work and back from
  parallel workers), and only items changed otherwise are re-read. Indexes are cached
  by content digest.
- A packed index for stores with more than 16 subjects: every subject's index rows in
  one object, so a query reaching many subjects makes one read instead of one per
  subject (150 subjects, local: 243 to 37 ms cold). Refreshed lazily at commit, once
  more than max(16, 5%) of subjects changed since it was packed; queries read those
  from their own index meanwhile. `fsck` checks it against the subject indexes.
- `repo.refresh()` (catch up with newer versions: one tip read, then only changed
  subject indexes refetched), `repo.watch(interval)` (yield new versions as they land),
  `query(include_staged=True)` (this Repo's staged changes as commit would record them),
  a digest-keyed disk cache for remote stores, and concurrent index fetches.
- Views are lazy: a `RecordingView`/`TableView`/`ArrayView`/`ExternalFileView` from
  `query`, `find` or `get` opens its group only when its data or metadata is read.
- `repo.query(expression, kind=, version=, **filters)` / `neurozarr query`: an expression
  language (`== != < <= > >= in`, `and/or/not`, `exists()`), compiled into a subject-level
  prune over the summary and numpy masks over only the referenced columns; results give
  `len`, `paths` and `.df` without opening anything, and lazy views when iterated.
  `repo.get(path)` finds one item by binary search. `find`, `ls`, `describe` and
  `subjects` now run on the index, and every `version=` takes any version name.
- git's workflow. Changes are staged and then committed: `repo.add(source)` /
  `neurozarr add`, item-level `add_recording`/`add_table`/`add_array`, `rm`, `mv`;
  `status`, `reset` (unstage), `commit -m`. Staged changes live in a local staging area
  (git's index) shared by Python and the CLI, so they survive between processes: each
  touched repository is staged as a pickled Icechunk fork (chunks are written at `add`,
  as git writes objects), under `$XDG_STATE_HOME/neurozarr/`, per user and per store.
  `commit` refuses, and publishes nothing, if the branch moved since staging and touched
  the same chunks or node metadata (Icechunk's own merge would silently drop the other
  change); changes to different data both land.
- Inspection: `log` (with `--oneline`, refs, `--json`), `show` (message, subjects
  changed, codec plan), `ls` (subjects / sessions / items, at any version), `find`
  (`KEY=VALUE` terms), `tag NAME [VERSION]`, `tag -d`. Versions resolve like git: a
  branch, a tag, an id or unique prefix, `REF~N` (`Repo.resolve`).
- Branches and undo: `branch`/`branches`/`delete_branch` (CLI `branch [-d|-D]`),
  `switch [-c]`, `repo.head`; `diff` (staged, VERSION vs now, A vs B; per subject items
  and chunks rewritten; `--stat`, `--json`, `--exit-code`); `revert` (a new version
  re-pinning the subjects a version changed, refused if they changed again since; no
  data copied); `reset_hard` (CLI `reset --hard VERSION --yes`). Branches are pointers
  in `_dataset`; subject repositories get the branch on their first write there.
- `merge` (fast-forward; otherwise per subject against the common ancestor: one-sided
  changes taken without copying, two-sided non-overlapping changes combined item by
  item, overlapping ones a `MergeConflictError` settled with `ours=`/`theirs=`; the
  merged tip is recorded in commit metadata so later merges find the right ancestor),
  `clone` (full for local paths; `shallow`/`--depth 1` copies one version anywhere), and
  `gc` (dry run by default; keeps every version on a branch or tag and all data it pins
  -- subject snapshots only a dataset version reaches get a ref first -- and only
  deletes objects older than the cutoff).
- Stores are found like git repositories: `-C STORE`, then `$NEUROZARR_STORE`, then the
  store the current directory is in. Named remotes (`neurozarr remote add NAME URL -o
  key=value`) save a URI with its storage options; any command takes `-C NAME`.
- `Repo.add(..., workers=N)` (CLI `add -j N`): parallel staging for any reader with
  `subjects()`/`for_subject()` (`BidsReader`, `ManifestReader`, `SmartReader`). Each
  worker writes into its subject's staged fork; a failing subject stages nothing usable
  and nothing is published. In-memory stores and readers without those methods run
  serially.
- `SmartReader` (beta; `neurozarr add /siteA /siteB`): list the folders
  the data is in, in any layout, and it works out the rest.
  - Structure: BIDS, folder-per-subject and flat layouts, alone or mixed; reads
    `dataset_description.json`, `participants.tsv`/`.json`, `*_sessions.tsv`, same-named
    JSON sidecars (merged into the recording) and `<recording>_*.tsv`/`.json` sidecars.
  - Content: opens each file's header; unreadable files are listed instead of failing
    the run, byte-identical copies are ingested once, the header's patient id names the
    subject when the path only gives a filename prefix, and runs are numbered by the
    file's own number, then recording date, then path order.
  - Formats: everything mne opens, plus `.mat` (v5 and v7.3), `.npy`, `.npz`, HDF5 and
    NWB (signal, sampling rate and channel names found in the file), and files with no
    extension, recognized by their first bytes.
  - Asks when unsure: defaulted datatypes, filename-prefix subjects and missing sampling
    rates are grouped into `reader.questions`; answers are path-glob overrides
    (`answers=`), which the CLI asks for on a terminal and saves to `--answers`.
- `verbose=True` on `Repo.init`/`Repo.open`: logs one
  line per item as it's written -- what it is, its size, live + peak
  `tracemalloc`-tracked memory (real Python-level allocations, not raw
  process RSS, which is dominated by one-time import cost and allocator
  retention rather than anything neurozarr itself is holding), and a running
  total of logical bytes handed to storage so far this session -- plus one
  line per commit and per reader/subject-repo step. Shorthand for
  `neurozarr.set_verbosity(logging.INFO)`, which is what starts the
  `tracemalloc` tracking; the package stays quiet by default, as a library
  should. Parallel `add` applies it inside each worker process, since a
  fresh spawned interpreter doesn't inherit the caller's logging setup.
- `Repo.describe()` and `Visit.describe()`: a store's or one session's
  contents summarized as data (a list of dicts, sorted by path) instead of
  printed text -- `neurozarr info` is now a thin wrapper around
  `Repo.describe()` rather than a separate copy of the same counting logic.
- `Subject.add_recording`, `add_behavioral_table`, `add_array`, and
  `add_derivative`, for datasets with no sessions at all: the same methods
  `Visit` has, writing with no `ses-` segment in the path rather than
  requiring a session that doesn't exist for this data. Both classes share
  one internal implementation now, differing only in which session (or none)
  items get filed under.
- `BidsReader(..., include_derivatives=False)`: skip everything under a
  `derivatives/` folder entirely. Previously read unconditionally whenever
  present; that remains the default.

- Explicit `Repo.init`, `Repo.open`, and transactional write APIs.
- A schema-v2 dataset manifest that atomically pins exact subject snapshots,
  plus `fsck` and explicit metadata-only migration commands.
- Structured validation reports and a public neurozarr exception hierarchy.
- Safe duplicate policies, external-file references, typed nullable tables,
  N-dimensional arrays, source provenance, and reader plugin registration.
- A lean Python 3.11 quality workflow with deterministic test timeouts.
- A format decoder registry (`register_format`, `registered_formats`), distinct
  from reader registration: given one already-located file, a decoder turns it
  into item(s), shared automatically by `ManifestReader` and `BidsReader`
  rather than each reimplementing per-format dispatch. `.nii`/`.nii.gz` decode
  to a real `Array` automatically when the new optional `nibabel` extra
  (`pip install "neurozarr[imaging]"`) is installed.
- `unclaimed="embed"` on `ManifestReader`/`BidsReader`: read a file with no
  matching decoder as raw bytes into the store, rather than only referencing
  its path. Opt-in; the default (`"reference"`) is unchanged.
- `Repo.set_attrs`, `Subject.set_attrs`, and `Visit.set_attrs`, for attaching
  metadata to a dataset, subject, or session after it already exists. Merges
  into whatever attrs are already there, same as `create_subject(attrs=)` and
  `add_visit(attrs=)`; not committed until `save`.
- The same `set_attrs` on `RecordingView`, `TableView`, `ArrayView`, and
  `ExternalFileView`, for annotating one already-written item without
  rewriting its data. Views obtained through `Repo`/`Subject`/`Visit` carry a
  route back to their subject's writer; a view built directly from a bare
  zarr group has none and raises `TypeError` on `set_attrs`.
- `rename()` and `delete()` on `RecordingView`, `TableView`, `ArrayView`, and
  `ExternalFileView`, and `Subject.rename_visit`/`delete_visit`. zarr has no
  move primitive; icechunk does (`Repository.rearrange_session().move()`), but
  it commits independently of a `writable_session`, which conflicts with the
  "not committed until `Repo.commit()`" contract every other write here follows
  -- so for now a rename copies the item to its new key and deletes the old
  one, all within the same subject repository. (Thanks to @ianhi for the
  correction -- see [#1](https://github.com/chintan-27/neurozarr/issues/1).)
- `Repo.delete_subject`: removes a subject from schema v2's published index
  (`subjects`, `subject_snapshots`, `catalog`) without touching its own
  repository or version history -- a soft removal, consistent with
  everything else in the store being versioned rather than erased. There is
  no equivalent `rename_subject`: a subject id is baked into its
  repository's storage location, and neither zarr nor icechunk can move that
  cheaply, so renaming a whole subject means reading its data out,
  re-ingesting it under the new id, and calling `delete_subject` on the old
  one.

### Fixed

- `BidsReader` ingest could raise a spurious `WriteConflictError` on a fresh,
  unmodified dataset: a recording's own annotations (e.g. EEGLAB's structural
  `boundary` markers) auto-derive an `events` table as a fallback, but a
  reader streams a recording and its real BIDS `events.tsv` sidecar as two
  independent items with no ordering guarantee between them -- if the
  recording landed first, its placeholder blocked the real sidecar from
  writing at all. The sidecar now always supersedes the placeholder,
  regardless of which is processed first.
- Table string columns are stored as numpy's variable-length `StringDType`
  instead of a fixed-width `<U*>` array -- the latter maps to zarr's
  `FixedLengthUTF32`, a Zarr V3 extension with no ratified spec yet, so every
  string column write raised an `UnstableSpecificationWarning`. `StringDType`
  maps to zarr's core `"string"` type, which is spec'd and warning-free.
- `requires-python` raised to `>=3.12` -- `icechunk>=2.1` dropped Python 3.11
  wheels, so declaring 3.11 support was already false; CI's `test` matrix and
  `quality`/`docs` jobs (previously pinned to 3.11) moved to 3.12 to match.
- Icechunk's own Rust-side warnings (e.g. "LocalFileSystem storage is not safe
  for concurrent commits") no longer print by default -- `RUST_LOG` never
  reached them, so seeing clean output required importing `icechunk` directly
  and calling `set_logs_filter` yourself. Silenced once at import time instead;
  using neurozarr no longer requires knowing icechunk exists.
- `find()` no longer hides subjects whose manifest catalog entry is missing or out
  of date. Catalog entries now record the snapshot they describe, and an entry that
  does not match the published snapshot falls back to scanning that subject instead
  of being read as an authoritative list of matches.
- `doctor` reports a catalog entry that describes an unpublished snapshot as
  `store.stale_catalog_entry`.
- Declared support no longer excludes environments the package is tested on:
  `requires-python` is `>=3.11` again, and the `zarr` upper bound is the next major
  rather than the next minor. Both previously refused to install against the
  interpreter and Zarr version the suite passes on.
- `reader_for()` and `neurozarr convert` now match a registered or installed
  reader's `extensions` against the full filename rather than only its last
  dot-segment, so a reader registered for a compound extension such as
  `.nii.gz` or `.tar.gz` is selected. It previously never matched.
- A group holding a `Recording` could silently hide a sibling `Table`, `Array`,
  or `ExternalFile` filed under the same entities (a recording and an
  unsupported-format sidecar sharing one BIDS task, for instance): the sibling
  never appeared in `visit.tables()`/`arrays()`/`external_files()` or
  `repo.find()`. Both are now found; `RecordingView.channels()`/`.events()`
  remain the only way to reach those two specific facade tables, unchanged.
- `BidsReader` recovered a BIDS suffix like `T1w` as `T1w.nii` for any
  `.nii.gz` file, since `Path.stem` only strips the file's last extension.

### Changed

- Table columns sharing a physical dtype and nullability are now packed into
  one shared zarr array instead of each getting its own (`table_schema_version`
  3). icechunk's per-array bookkeeping cost dominated writing tables with many
  small columns -- measured ~56% of total conversion time on a real BIDS
  dataset, and ~4x faster table writes after packing on a table shaped like a
  typical `channels.tsv`. Reading is unaffected: a column's own schema entry
  is all that's needed to decode it, so `table_schema_version` 2 stores remain
  readable unchanged.
- MNE recordings are opened lazily and written one time chunk at a time.
- BIDS JSON inheritance merges applicable sidecars from root to leaf and
  BrainVision companion files are treated as one bundle.
- Storage URIs reject embedded credentials; custom cloud storage now uses a
  per-repository factory instead of one aliased `Storage` object.
- Existing item paths fail by default; replacement or skipping must be explicit.
- Codecs are now chosen by the reader, once per ingest, instead of being fixed at
  zstd level 19. A reader's `codec_plan()` probes a few real files (one real middle
  chunk each, encoded exactly as stored) with no, byte and bit shuffle and keeps the
  smallest per group; every writer, serial or parallel, applies that plan, and it is
  recorded in the dataset metadata as `codec_plan`. Reader options: `optimize`
  (`speed` = Blosc lz4, `balanced` = Blosc zstd-5, the default, `size` = Blosc
  zstd-9), `probe_level` (one decision per: 0 nothing / zarr default, 1 item type,
  2 item type x stored dtype, the default, 3 + chunk shape, 4 each array) and
  `probe_files`; CLI `--optimize`, `--probe-level`, `--probe-files`. Strings get zstd
  and embedded files are never probed. Measured on BRAVO, CHB-MIT, RNS, Pediatric
  iEEG and HBN EEG: the shuffle mode is what matters and differs by dataset; level 2
  is within 0.1-2.3% of probing every recording, and planning took 1-61 s (opening
  the sampled files), about 1% of a large ingest. Blosc is a core Zarr v3 codec, so
  stores stay readable outside zarr-python; existing stores read as before.
- The explicit API (`visit.add_recording` and friends) never probes: it uses
  `CodecConfig(compressors=...)` if set, else zarr's default. Setting it also
  overrides a reader's plan during `ingest`, with a warning.
- Tables, generic arrays and embedded files get codecs too (previously always
  zarr's default), and table arrays are chunked like other arrays.

### Fixed

- `rename` kept the data but re-encoded the arrays with zarr's default codec; it now
  copies each array's filters, compressors and serializer.

### Changed (renamed outright; nothing was published under the old names)

- Python: `Repo.create` -> `Repo.init`, `ingest` -> `add` (also takes a path, with the
  reader picked as the CLI does), `save` -> `commit`, `abort` -> `reset`, `history` ->
  `log`, `add_behavioral_table` -> `add_table` (with `datatype=`); `add_recording` gains
  `datatype=`; the generic `visit.add(datatype, payload)` is removed.
- CLI: `convert SRC DEST` -> `init` + `add` + `commit`; `info` -> `ls`/`show`; `history`
  -> `log`; `doctor` -> `fsck`; `verify`, `export` and `migrate` use the discovered store.

### Removed

- `neurozarr.parallel.convert_parallel` and the `neurozarr.parallel` module: use
  `repo.add(source, workers=N)`.

### Deprecated

- `Repo(target)` in favor of explicit create/open operations.
- `skip_existing` in favor of `existing="skip"`.

## [0.1.0] — 2026-09-04

First release.

### Added

- `Repo`, `Subject` and `Visit` for building a store, either in bulk from a
  reader or by hand.
- `BidsReader`, reading any valid BIDS dataset: subjects and sessions discovered
  by directory, datatype directories walked without a fixed list, and JSON
  sidecars resolved through the BIDS inheritance principle.
- `ManifestReader`, ingesting data with no standard layout from a table
  describing one file per row, with configurable column names and a
  `row_reader` callback for custom cases.
- `Reader` protocol, so a new source format is supported by writing a reader
  that yields `Recording`, `Table` and `Attrs` items.
- Read-back API: `RecordingView` and `TableView`, with `data()`, `raw()`,
  `channels()`, `events()` and `df()`. Windowed reads by sample index or by
  seconds, and channel selection.
- One Icechunk repository per subject, plus a `_dataset` repository for
  dataset-level metadata, so a single subject can be copied or shared alone.
- Cloud storage targets: `s3://`, `gs://`, `az://`, `r2://` and `memory://`.
- Version history and tags, and reading any earlier state with `version=`.
- Derivatives: `Visit.add_derivative` files processed results under
  `derivatives/<pipeline>/`.
- `convert_parallel`, converting one subject per worker process.
- `skip_existing`, writing only what a store does not already hold.
- `verify`, comparing a store against the source it was converted from, and
  `export_bids`, writing a store back out as a BIDS dataset.
- `validate_source`, checking a source before converting it.
- `neurozarr` command line: `convert`, `validate`, `info`, `history`, `verify`
  and `export`.
- mne annotations stored as an events table and restored on read.
- Sample data packed as int16 with a per-channel scale and offset, chunked along
  time so windowed reads fetch only the chunks they need.
- Type annotations throughout, with `py.typed` so downstream type checkers use
  them. Values that can be absent — `RecordingView.sfreq` and
  `RecordingView.duration` — are typed as optional, so a checker catches
  `rec.duration / 2` rather than letting it fail at runtime.
