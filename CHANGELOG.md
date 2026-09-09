# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Explicit `Repo.create`, `Repo.open`, and transactional write APIs.
- A schema-v2 dataset manifest that atomically pins exact subject snapshots,
  plus `doctor` and explicit metadata-only migration commands.
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
  `ExternalFileView`, and `Subject.rename_visit`/`delete_visit`. Neither zarr
  nor icechunk has a move primitive, so a rename copies the item to its new
  key and deletes the old one, all within the same subject repository.
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
