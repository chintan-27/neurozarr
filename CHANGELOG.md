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

### Changed

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
