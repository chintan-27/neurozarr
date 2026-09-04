# Cloud storage

Anywhere Icechunk can write, this can write — pass a URI instead of a path:

```python
Repo("s3://my-bucket/study", region="us-east-1")
Repo("gs://my-bucket/study")
Repo("az://account/container/study")
Repo("memory://scratch")                   # in-memory, handy for tests
Repo(icechunk.s3_storage(...))             # or a Storage you built yourself
```

Extra keyword arguments (`region=`, `anonymous=`, `from_env=`, …) pass straight through to Icechunk.
