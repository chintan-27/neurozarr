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
result carries a record of where it came from.

## Reading them back

Derivatives are stored like any other data and read back the same way. They
appear in {meth}`~bidszarr.Subject.recordings` and
{meth}`~bidszarr.Repo.find` alongside raw recordings, distinguished by
`derivatives/<pipeline>/` in their path:

```python
for rec in subject.recordings():
    if rec.path.startswith("derivatives/"):
        print(rec.path)
```

Because they live in the same store as their source data, a derivative is
versioned and tagged along with everything else — a tag names the state of the
raw data and the results computed from it together.
