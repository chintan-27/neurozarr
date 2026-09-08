# Storing processed results

Analysis outputs go under `derivatives/`, which is how BIDS keeps derived data
separate from the raw recordings it came from. Name the pipeline that produced
the result and it's filed accordingly:

```python
filtered = raw.copy().filter(l_freq=1, h_freq=40)
visit.add_derivative("my-filter", filtered, task="Stream", run=1)
```

Tables work the same way — pass the datatype they belong under:

```python
visit.add_derivative("my-stats", stats_dataframe, datatype="beh", task="Stream")
```

The pipeline name is recorded as `GeneratedBy` in the stored metadata, so a
result carries a record of where it came from. Record more of that
provenance explicitly when you have it:

```python
visit.add_derivative(
    "my-filter", filtered, task="Stream", run=1,
    pipeline_version="1.2.0",
    parameters={"l_freq": 1, "h_freq": 40},
    inputs=["sub-001/ses-1/ieeg/task-Stream_run-1"],
)
```

`pipeline_version`, `parameters`, and `inputs` are stored alongside
`GeneratedBy` under `meta["_neurozarr_provenance"]`, together with the
dataset id the source recording belonged to at the time — enough to answer
"what produced this, with which settings, from which recording" without
external bookkeeping.

## Reading them back

Derivatives are stored like any other data and read back the same way. They
appear in {meth}`~neurozarr.Subject.recordings` and
{meth}`~neurozarr.Repo.find` alongside raw recordings, distinguished by
`derivatives/<pipeline>/` somewhere in their path (after the subject id, not
at the start of it):

```python
for rec in subject.recordings():
    if "derivatives/" in rec.path:
        print(rec.path, rec.meta["_neurozarr_provenance"])
```

Because they live in the same store as their source data, a derivative is
versioned and tagged along with everything else — a tag names the state of the
raw data and the results computed from it together.
