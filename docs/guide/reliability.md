# Reliability and store compatibility

Use {meth}`neurozarr.Repo.create` when a destination must be new and
{meth}`neurozarr.Repo.open` for an existing store. Unlike the compatibility
constructor, neither can silently turn a misspelled input path into a new empty
store.

```python
repo = Repo.create("./study.zarr")
repo = Repo.open("./study.zarr", mode="a")
```

## Transactions and versions

Every subject is committed independently. Neurozarr then commits `_dataset`
with a manifest mapping each subject to its exact Icechunk snapshot. The
dataset-manifest snapshot returned by `save()` is therefore a consistent global
version. If publication is interrupted, earlier readers continue to use the
previous manifest.

For automatic abort on exceptions, use:

```python
with repo.transaction("add second visit"):
    repo.subject("sub-001").add_visit("ses-2").add_recording(raw, task="Rest")
```

Different-subject writers can merge at publication. Changes to the same subject
or concurrent dataset metadata raise `WriteConflictError` and must be retried.

## Validation and recovery

`inspect_source()` returns a `ValidationReport` with stable issue codes,
severities, locations, and context. `validate_source()` remains as a string-list
compatibility wrapper. Unsafe path segments, invalid metadata, and malformed
supported data fail; unsupported formats become explicit `ExternalFile`
references with warnings.

Run `neurozarr doctor STORE` after an interrupted job. It detects missing
snapshots and subject commits not yet published by the dataset manifest.

Stores written by 0.1 remain readable. Preview and apply the metadata-only
migration with `neurozarr migrate STORE --dry-run` and then without
`--dry-run`. Newer unknown schemas are never opened optimistically.

## Sensitive data

Neurozarr does not provide authorization or de-identification. Configure access
in the storage backend, avoid PHI in entity labels, and review metadata before
sharing. Credentials, query strings, and fragments are rejected in storage
URIs; external-file URIs are stored without user information or tokens.
