# Reading data back

```python
repo = Repo("./study.zarr")

repo.subjects()                       # ['sub-001', 'sub-002', ...]
subject = repo.subject("sub-001")
subject.attrs                         # {'age': 63, ...}
subject.visits()                      # ['ses-20220908', ...]

rec = subject.visit("ses-20220908").recording(task="Stream", run=1)
values, meta = rec.data()             # ndarray in physical units + its metadata
raw = rec.raw()                       # a real mne.io.RawArray, ready for mne
df = rec.channels()                   # the channels table as a DataFrame
events = rec.events()                 # events / annotations

# search across every subject in the store
for rec in repo.find(task="BrainSenseStream", acq="TD"):
    print(rec.path, rec.shape)
```

Recordings are stored as int16 with per-channel scale/offset; `.data()` and `.raw()` undo that for you, so what you read back matches the source to floating-point precision. Table columns keep their dtypes — numbers come back as numbers. `mne` annotations are stored as an events table and put back on the `Raw` when you read it.

## Reading only part of a recording

Data is chunked along time, so you can pull a window without fetching the whole array — the point of storing it this way:

```python
values, meta = rec.data(tmin=10, tmax=20)          # ten seconds, by time
values, meta = rec.data(start=1000, stop=2000)     # or by sample index
values, meta = rec.data(tmin=10, tmax=20, picks=["LFP_L"])   # one channel
raw = rec.raw(tmin=10, tmax=20)                    # same window as an mne.Raw

rec.array          # the underlying zarr array, slice it yourself
rec.duration       # seconds
rec.sfreq          # sampling rate
```

On a 3708-second recording, reading a 10-second window this way is ~8× faster than reading the whole thing.
