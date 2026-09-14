# Step-by-step demo of neurozarr, one action per cell, no buttons -- runs
# straight through when opened. Build a store by hand first (no reader),
# read it back and actually use the data, then show the same subject
# converted automatically two different ways: BidsReader and ManifestReader.
# No parallel-conversion section here -- see examples/demo_stepbystep.py
# for that.
#
#     marimo edit examples/demo_notebook.py      # interactive editing
#     marimo run examples/demo_notebook.py       # read-only app view
#
# Runs against the real BRAVO BIDS export at ./BIDS (gitignored, present on
# this machine only). Every write goes to ./zarr/demo_explicit.zarr (also
# gitignored, scratch output).

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Imports, plus plumbing so neurozarr's verbose progress lines (INFO-level
    # logging) show up in this notebook's output instead of the server's own
    # terminal. Nothing icechunk-specific here -- neurozarr silences icechunk's
    # own noisy warnings itself, at import time, so this notebook never needs
    # to import icechunk to get clean output.
    import logging
    import shutil
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import mne  # type: ignore[import-untyped]
    import numpy as np
    import pandas as pd

    from neurozarr import CodecConfig, Repo
    from neurozarr.log import logger as neurozarr_logger, set_verbosity
    from neurozarr.readers import BidsReader, ManifestReader

    set_verbosity(logging.INFO)
    for _handler in neurozarr_logger.handlers:
        if isinstance(_handler, logging.StreamHandler):
            _handler.stream = sys.stdout  # so neurozarr's progress lines land in this notebook's output

    SOURCE = Path("BIDS")
    SUBJECT = "sub-004"
    return (
        BidsReader,
        CodecConfig,
        ManifestReader,
        Path,
        Repo,
        SOURCE,
        SUBJECT,
        mne,
        mo,
        np,
        pd,
        shutil,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # neurozarr

    idea: stop treating a neuroscience dataset as a folder of
    files that gets copied around and mutated in place, and instead put
    it in something that behaves like a versioned database. This
    notebook builds one store by hand -- no automated reader, nothing
    hidden -- then reads it back and uses the data, against real data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Building the store, explicitly -- no reader at all

    Two ideas worth having straight before anything below makes sense:

    **The hierarchy** -- every store follows the same fixed shape, no matter
    how the data got there:

    ```
    Repo  ->  Subject  ->  Visit  ->  Recording / Table / Array / ExternalFile
    (the study)  (one patient)  (one session)  (the actual data)
    ```

    **Entities** are the BIDS-style labels (`task=`, `run=`, `acq=`, ...) that
    say exactly *where inside* a subject's visit one item belongs -- not a
    separate concept from the hierarchy, just its finest level. `task="ChronicLFP", run=2`
    below is what turns into the last segment of this recording's path.

    **The type of thing being stored** is always exactly one of five fixed
    box types -- `Recording`, `Table`, `Array`, `ExternalFile`, or `Attrs`
    (metadata with no data of its own). `add_recording`/`add_behavioral_table`
    don't invent anything new: they build a `Recording`/`Table` out of what
    you hand them and pass it to the one piece of code that actually writes
    to storage. `raw` below is a plain `mne.io.Raw` -- neurozarr never sees
    it as anything else until `add_recording` wraps it into a `Recording`.
    """)
    return


@app.cell
def _(Path, SOURCE, SUBJECT):
    # Find the real files on disk we'll hand to neurozarr ourselves below.
    dest = Path("zarr/demo_explicit.zarr")
    ses_dir = sorted((SOURCE / SUBJECT).glob("ses-*"))[0]
    edf = sorted((ses_dir / "ieeg").glob("*task-ChronicLFP_run-2_ieeg.edf"))[0]
    tsv = sorted((ses_dir / "beh").glob("*task-TherapyHistory_beh.tsv"))[0]
    edf.name, tsv.name
    return dest, edf, ses_dir, tsv


@app.cell
def _(Repo, dest, shutil):
    # Repo.create makes a brand new store -- one call, nothing written yet.
    shutil.rmtree(dest, ignore_errors=True)
    repo = Repo.create(str(dest), verbose=True)
    return (repo,)


@app.cell
def _(SUBJECT, repo):
    # Register the subject. Nothing about a reader here -- this is the same
    # call a reader makes internally, just done by us.
    print(f"Creating Repo for Subject : {SUBJECT}")
    subject = repo.create_subject(SUBJECT, attrs={"note": "added via explicit construction for the demo"})
    return (subject,)


@app.cell
def _(ses_dir, subject):
    # One session under that subject.
    print(f"Creating Session : {ses_dir.name}")
    visit = subject.add_visit(ses_dir.name)
    return (visit,)


@app.cell
def _(edf, mne):
    # We open the EDF ourselves with mne -- neurozarr never touches the file,
    # it only ever sees this Raw object. This is the actual thing about to be
    # handed to add_recording, not a description of it.
    raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
    raw
    return (raw,)


@app.cell
def _(raw):
    # The type neurozarr actually sees here -- an ordinary mne class, nothing
    # neurozarr-specific yet. add_recording is what turns this into a
    # Recording (one of the five fixed item types) a moment from now.
    type(raw)
    return


@app.cell
def _(SUBJECT, visit):
    # Exactly where this is about to land -- subject, session, and every
    # entity -- defined once here, displayed, then unpacked into the actual
    # call below, so there's one source of truth, not two copies to drift.
    recording_entities = {"task": "ChronicLFP", "run": 2}
    {"sub": SUBJECT, "ses": visit.ses_id, **recording_entities}
    return (recording_entities,)


@app.cell
def _(raw, recording_entities, visit):
    # Hand that Raw object straight to neurozarr. existing="replace" makes
    # this cell safe to re-run on its own during a live demo -- the default
    # existing="error" would raise WriteConflictError on a second run (the
    # address is already taken), which permanently poisons this `repo`
    # object for every write after it, not just this one.
    visit.add_recording(raw, **recording_entities, existing="replace")
    recording_added = True
    return (recording_added,)


@app.cell
def _(pd, tsv):
    # Same idea for the table: plain pandas.read_csv. Again, the real
    # DataFrame about to be handed to add_behavioral_table -- not a
    # description of its shape.
    therapy_df = pd.read_csv(tsv, sep="\t")
    therapy_df
    return (therapy_df,)


@app.cell
def _(therapy_df):
    # Same idea -- a plain pandas type until add_behavioral_table wraps it
    # into a Table.
    type(therapy_df)
    return


@app.cell
def _(SUBJECT, visit):
    table_entities = {"task": "TherapyHistory"}
    {"sub": SUBJECT, "ses": visit.ses_id, **table_entities}
    return (table_entities,)


@app.cell
def _(table_entities, therapy_df, visit):
    # Column dtypes survive the round trip through add_behavioral_table.
    # existing="replace" for the same reason as add_recording above.
    visit.add_behavioral_table(therapy_df, **table_entities, existing="replace")
    table_added = True
    return (table_added,)


@app.cell
def _(recording_added, repo, table_added):
    # Nothing is committed until save() -- this is the one real Icechunk
    # commit for everything added above.
    assert recording_added and table_added
    snapshot = repo.save("built sub-004 explicitly")
    snapshot
    return (snapshot,)


@app.cell
def _(SUBJECT, repo, snapshot):
    # How it actually sits in storage now -- queried straight off the real
    # zarr tree (zarr.Group.tree()), not a hand-written description of it.
    # The two recording/table facades from add_recording/add_behavioral_table
    # are visible here as real nested groups and arrays: "data" packed as
    # int16 (decision 5), the events table mne's own annotations produced
    # automatically, and the behavioral table's columns bundled into shared
    # "g######" arrays wherever their dtype and nullability match (decision 4).
    assert snapshot is not None
    repo.subject(SUBJECT).root().tree()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reading it back, and actually using it

    A fresh `Repo.open` -- not the object we just wrote with -- the same way
    you'd open this store in an entirely different script, weeks later. This
    never goes through `Writer`; it's a completely independent path.

    Every store has the same fixed shape, three levels deep, and it's the
    same shape no matter which reader built it:

    ```
    Repo  ->  Subject  ->  Visit  ->  Recording / Table / Array / ExternalFile
    (the study)  (one patient)  (one session)  (the actual data)
    ```

    `Subject` and `Visit` aren't extra data structures of their own -- they're
    just navigation handles over that one fixed shape. Explored one level at a
    time below, as if we didn't already know what's in here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Level 1 -- every subject in the store**
    """)
    return


@app.cell
def _(Repo, dest, snapshot):
    assert snapshot is not None
    explore_repo = Repo.open(str(dest))
    explore_repo.subjects()
    return (explore_repo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Or the whole per-subject overview in one call -- `Repo.describe()`,
    added specifically because there was no single method for this before:
    each subject's own navigation methods (`.visits()`, `.recordings()`,
    `.tables()`, ...) existed, but nothing tied them together into one
    summary. It's real data (a list of dicts), not printed text -- the
    `neurozarr info` CLI command is now a thin wrapper around this same
    method, not a separate copy of the logic.
    """)
    return


@app.cell
def _(explore_repo, pd):
    pd.DataFrame(explore_repo.describe())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Level 2 -- every visit this one subject has**
    """)
    return


@app.cell
def _(SUBJECT, explore_repo):
    explore_subject = explore_repo.subject(SUBJECT)
    explore_subject.visits()
    return (explore_subject,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Level 3 -- everything actually inside one visit**
    """)
    return


@app.cell
def _(explore_subject, ses_dir):
    explore_visit = explore_subject.visit(ses_dir.name)
    explore_recordings = explore_visit.recordings()
    explore_tables = explore_visit.tables()
    explore_recordings, explore_tables
    return explore_tables, explore_visit


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **And what BIDS entity is each one, specifically?** -- `Visit.describe()`,
    the other new method: every item in this session, its kind, its path, and
    every entity it carries, in one call instead of listing each kind
    separately and reading `.path`/`.entities` off every result by hand.
    """)
    return


@app.cell
def _(explore_visit):
    explore_visit.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Now zoom into the one recording we found above, by the entities the
    exploration table just showed it carries.
    """)
    return


@app.cell
def _(explore_visit):
    readback_recording = explore_visit.recording(task="ChronicLFP", run=2)
    readback_recording.path, readback_recording.shape, readback_recording.sfreq
    return (readback_recording,)


@app.cell
def _(readback_recording):
    # Real channel names, read straight back out -- nothing was lost.
    readback_recording.ch_names()
    return


@app.cell
def _(pd, readback_recording):
    # The actual signal -- first 2 seconds, converted back to physical units.
    # Only the storage chunks covering this window are fetched, however long
    # the full recording actually is.
    _values, _meta = readback_recording.data(tmin=0, tmax=2)
    pd.DataFrame(_values.T, columns=readback_recording.ch_names()).round(4)
    return


@app.cell
def _(readback_recording):
    # Reconstructed as a real mne object, ready for any standard analysis tool.
    readback_recording.raw()
    return


@app.cell
def _(explore_tables):
    # The behavioral table found during exploration above, read back with
    # its original column types intact.
    readback_table = explore_tables[0]
    readback_df = readback_table.df()
    f"{readback_table.path}/{readback_table.name}: {readback_df.shape[0]} rows x {readback_df.shape[1]} columns"
    return (readback_df,)


@app.cell
def _(readback_df):
    readback_df[["onset", "Hemisphere", "Contact", "Amplitude", "AmplitudeUnit", "Frequency"]].head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The same subject, automatically, with BidsReader

    Same subject, same files on disk as above -- but this time `BidsReader`
    finds and converts everything for sub-004 on its own: every session,
    every datatype, every sidecar, using BIDS's own naming rules to work out
    where each item belongs. This is what the explicit section above did by
    hand, for two files, done here for the whole subject automatically.
    """)
    return


@app.cell
def _(Path):
    bids_dest = Path("zarr/demo_bids.zarr")
    return (bids_dest,)


@app.cell
def _(Repo, bids_dest, shutil):
    shutil.rmtree(bids_dest, ignore_errors=True)
    bids_repo = Repo.create(str(bids_dest), verbose=True)
    return (bids_repo,)


@app.cell
def _(BidsReader, SOURCE, SUBJECT, bids_repo, time):
    # One call: BidsReader walks every session/datatype/sidecar for this
    # subject and streams it all through ingest.
    bids_t0 = time.perf_counter()
    bids_repo.ingest(BidsReader(str(SOURCE), subjects=[SUBJECT]))
    bids_ingest_seconds = time.perf_counter() - bids_t0
    return (bids_ingest_seconds,)


@app.cell
def _(bids_ingest_seconds, bids_repo):
    bids_snapshot = bids_repo.save("converted sub-004 with BidsReader")
    bids_snapshot, round(bids_ingest_seconds, 1)
    return (bids_snapshot,)


@app.cell
def _(SUBJECT, bids_repo, bids_snapshot):
    # How it actually sits now -- queried straight off the real zarr tree,
    # not a hardcoded description. Compare this to the explicit section's
    # tree above: same subject, same underlying files, vastly more items --
    # every session/datatype/sidecar BidsReader found on its own.
    assert bids_snapshot is not None
    bids_repo.subject(SUBJECT).root().tree()
    return


@app.cell
def _(SUBJECT, bids_repo):
    # Counted programmatically off the real store, not typed in by hand.
    _subject = bids_repo.subject(SUBJECT)
    {
        "recordings": len(_subject.recordings()),
        "tables": len(_subject.tables()),
        "arrays": len(_subject.arrays()),
        "external_files": len(_subject.external_files()),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## A couple of the same files, through ManifestReader instead

    Not every dataset is laid out in BIDS's folder convention. `ManifestReader`
    handles that case: instead of inferring structure from folder names, you
    describe each file in a table -- path, subject, session, task -- and it
    converts exactly what's listed, nothing more.
    """)
    return


@app.cell
def _(SOURCE, SUBJECT):
    _ses_dir = sorted((SOURCE / SUBJECT).glob("ses-*"))[0]
    manifest_edf = sorted((_ses_dir / "ieeg").glob("*task-ChronicLFP_run-4_ieeg.edf"))[0]
    manifest_tsv = sorted((_ses_dir / "beh").glob("*task-Impedance_beh.tsv"))[0]
    manifest_ses = _ses_dir.name
    return manifest_edf, manifest_ses, manifest_tsv


@app.cell
def _(SUBJECT, manifest_edf, manifest_ses, manifest_tsv, pd):
    # One row per file, saying where it goes -- the whole point of
    # ManifestReader is that this table is all it needs. The real table,
    # not a description of it.
    manifest = pd.DataFrame([
        {"path": str(manifest_edf), "sub": SUBJECT, "ses": manifest_ses, "datatype": "ieeg",
         "task": "ChronicLFP", "run": 4},
        {"path": str(manifest_tsv), "sub": SUBJECT, "ses": manifest_ses, "datatype": "beh",
         "task": "Impedance", "name": "impedance"},
    ])
    manifest
    return (manifest,)


@app.cell
def _(Path):
    manifest_dest = Path("zarr/demo_manifest.zarr")
    return (manifest_dest,)


@app.cell
def _(Repo, manifest_dest, shutil):
    shutil.rmtree(manifest_dest, ignore_errors=True)
    manifest_repo = Repo.create(str(manifest_dest), verbose=True)
    return (manifest_repo,)


@app.cell
def _(ManifestReader, manifest, manifest_repo):
    manifest_repo.ingest(ManifestReader(manifest))
    manifest_snapshot = manifest_repo.save("converted 2 files with ManifestReader")
    manifest_snapshot
    return (manifest_snapshot,)


@app.cell
def _(SUBJECT, manifest_repo, manifest_snapshot):
    # Exactly the 2 files from the manifest, nothing more -- unlike
    # BidsReader, ManifestReader only ever touches what you list. Same
    # tree() call as every other section, so the contrast is honest: this
    # one is small because the manifest was small, not because anything
    # was hidden.
    assert manifest_snapshot is not None
    manifest_repo.subject(SUBJECT).root().tree()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Session-less data: adding directly to a subject

    Not every dataset splits into visits. `sub-004` here does, so this part
    uses a small synthetic recording instead -- the point is the shape of
    the call, not the data. `Subject` has the same `add_recording` /
    `add_behavioral_table` / `add_array` / `add_derivative` methods `Visit`
    has (they come from the same shared code internally) -- no `add_visit()`
    call at all, and the resulting path carries no `ses-` segment.
    """)
    return


@app.cell
def _(mne, np):
    sessionless_info = mne.create_info(["ch1", "ch2"], sfreq=100.0, ch_types="eeg")
    sessionless_raw = mne.io.RawArray(np.random.randn(2, 500), sessionless_info, verbose=False)
    sessionless_raw
    return (sessionless_raw,)


@app.cell
def _(Path, Repo, shutil):
    sessionless_dest = Path("zarr/demo_sessionless.zarr")
    shutil.rmtree(sessionless_dest, ignore_errors=True)
    sessionless_repo = Repo.create(str(sessionless_dest), verbose=True)
    sessionless_subject = sessionless_repo.create_subject("sub-001")
    return sessionless_dest, sessionless_repo, sessionless_subject


@app.cell
def _(sessionless_raw, sessionless_subject):
    # No add_visit() anywhere -- straight from Subject to add_recording.
    # existing="replace" for the same re-run safety as the explicit section above.
    sessionless_subject.add_recording(sessionless_raw, task="Rest", existing="replace")
    return


@app.cell
def _(sessionless_repo):
    sessionless_snapshot = sessionless_repo.save("session-less recording")
    sessionless_snapshot
    return (sessionless_snapshot,)


@app.cell
def _(Repo, sessionless_dest, sessionless_snapshot):
    # Read back fresh, same as every other section -- and note .visits() is
    # empty and the recording's own path has no ses- segment in it at all.
    assert sessionless_snapshot is not None
    sessionless_readback = Repo.open(str(sessionless_dest)).subject("sub-001")
    sessionless_readback.visits(), [r.path for r in sessionless_readback.recordings()]
    return


@app.cell
def _(sessionless_repo):
    # How it actually sits -- one "ieeg" group straight under the subject,
    # no session level in between at all.
    sessionless_repo.subject("sub-001").root().tree()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Chunk size: the default, and how to change it

    Chunking only ever splits the *time* axis -- every channel is always kept
    whole in every chunk. The default (`CodecConfig` in `writer.py`) picks
    however many time-samples fit in **8 MiB**, capped at **65,536 samples**,
    whichever is smaller -- the cap exists so a narrow recording (few
    channels) doesn't end up as one giant chunk. `sub-004`'s real recordings
    are too short to show this (only ~1,000 samples, nowhere near either
    limit), so this part uses a larger synthetic one to actually see it.
    """)
    return


@app.cell
def _(mne, np):
    chunk_demo_raw = mne.io.RawArray(
        np.random.randn(8, 5_000_000).astype(np.float32),
        mne.create_info([f"ch{i}" for i in range(8)], sfreq=1000.0, ch_types="eeg"),
        verbose=False,
    )
    chunk_demo_raw
    return (chunk_demo_raw,)


@app.cell
def _(Repo, chunk_demo_raw):
    # Default CodecConfig.
    chunk_default_repo = Repo.create("memory://demo_chunk_default", verbose=True)
    chunk_default_repo.create_subject("sub-001").add_visit("ses-1").add_recording(
        chunk_demo_raw, task="Rest", existing="replace")
    chunk_default_repo.save("default chunking")
    chunk_default_rec = chunk_default_repo.subject("sub-001").visit("ses-1").recording(task="Rest")
    chunk_default_rec.array.chunks
    return (chunk_default_rec,)


@app.cell
def _(CodecConfig, Repo, chunk_demo_raw):
    # Same recording, a smaller target size and a lower sample cap -- real
    # values, not a description of what would happen.
    chunk_custom_repo = Repo.create(
        "memory://demo_chunk_custom",
        codec=CodecConfig(chunk_target_bytes=1 * 1024 * 1024, max_chunk_samples=16384),
        verbose=True,
    )
    chunk_custom_repo.create_subject("sub-001").add_visit("ses-1").add_recording(
        chunk_demo_raw, task="Rest", existing="replace")
    chunk_custom_repo.save("custom chunking")
    chunk_custom_rec = chunk_custom_repo.subject("sub-001").visit("ses-1").recording(task="Rest")
    chunk_custom_rec.array.chunks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Same 5,000,000-sample recording, two different chunk shapes, purely from
    the `CodecConfig` each store was created with -- and a windowed read only
    ever costs the chunks it actually touches, however long the recording is:
    """)
    return


@app.cell
def _(chunk_default_rec):
    _values, _meta = chunk_default_rec.data(tmin=0, tmax=1)   # 1 second = 1,000 samples
    f"read {_values.shape[1]:,} samples out of {chunk_default_rec.shape[1]:,} total"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    Full write-up: `README.md`, `CHANGELOG.md`, `docs/`. Same thing from
    a plain terminal: `python examples/demo_stepbystep.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
