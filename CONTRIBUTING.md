# Contributing

## Setup

```bash
pip install -e ".[dev]"     # pytest, tqdm, mypy
pytest                      # the whole suite
pytest -m "not slow"        # skip the type-checker tests, ~2s
```

Tests build their stores on `memory://`, so they run in a couple of seconds and
leave nothing on disk.

Storage tests have a 30-second per-test timeout. A timeout in a tiny test usually
means an Icechunk/Zarr compatibility problem; reduce it to a minimal upstream
example before changing neurozarr logic.

## Type checking

```bash
mypy
```

The package ships `py.typed`, which promises downstream type checkers that its
annotations can be trusted, so `mypy` must stay clean. Two rules follow from
that promise:

- Annotate what a function really returns. `-> float` on something that returns
  `None` when a value is missing is worse than no annotation at all, because a
  checker will then accept `rec.duration / 2` and let it fail at runtime.
- Silence untyped third-party imports with `# type: ignore[import-untyped]` at
  the import itself, never with an `ignore_missing_imports` override in
  `pyproject.toml`. A downstream project's mypy never reads this repo's config,
  so an override here would leave them seeing errors from our internals.

`tests/test_typing.py` checks both by type checking snippets against the built
package.

## Building the docs

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build
```

The build should stay warning-free. Docstrings are rendered into the API
reference, so they are written as numpydoc — a summary line, then
`Parameters`/`Returns`/`Raises` sections. Keep implementation rationale
(measurements, why an approach was rejected) in `#` comments rather than
docstrings: comments serve maintainers, docstrings serve users of the package.

## Where things go

- `neurozarr/readers/` — one module per source format. A reader turns its source
  into `Recording`, `Table` and `Attrs` items and nothing else.
- `neurozarr/writer.py` — the only code that writes Zarr. Keeping it that way is
  what guarantees the output structure stays consistent across readers.
- `neurozarr/schema.py` — the versioned dataset manifest. Store changes require
  a schema-version decision and a migration fixture.
- `scripts/` — entry points and benchmarks, not part of the package.
- `reference/` — reference-only material, not imported by the package.

## Adding support for a new source format

Write a reader. It needs one method:

```python
class MyReader:
    def read(self):
        yield Attrs((), {"Name": "my study"})
        yield Recording(Entities("sub-001", "ses-1", "ieeg", {"task": "Rest"}), raw)
```

There is no base class to inherit from — `Reader` is a `Protocol`, so anything
with a matching `read()` works. Add tests against a `memory://` store.

Distributed reader plugins use the `neurozarr.readers` entry-point group and
must yield core item types. They never receive a Writer or raw Zarr group.

## Conventions

- Tabs for indentation, matching the existing files.
- Public API is snake_case.
- Deliberate simplifications are marked with a `ponytail:` comment naming the
  ceiling and the upgrade path.
