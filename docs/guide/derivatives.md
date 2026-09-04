# Storing processed results

Analysis outputs go under `derivatives/`, kept separate from raw data the way BIDS does it:

```python
filtered = raw.copy().filter(l_freq=1, h_freq=40)
visit.add_derivative("my-filter", filtered, task="Stream", run=1)
visit.add_derivative("my-stats", stats_dataframe, datatype="beh", task="Stream")
```

They read back like anything else — `subject.recordings()` returns them with `derivatives/my-filter/` in the path.
