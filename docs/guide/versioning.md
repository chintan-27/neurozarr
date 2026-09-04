# Versioning

Every store is an Icechunk repository, so history is free:

```python
repo.save("reprocessed with new filter")   # a commit
repo.history()                             # [(snapshot_id, message, timestamp), ...]
repo.tag("v1")                             # name this state
repo.tags()                                # ['v1']

subject.recordings(version="v1")           # read the data as it was at v1
```
