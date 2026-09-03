from pathlib import Path
from urllib.parse import urlparse

import icechunk

# One in-memory Storage per logical path, so a memory:// store can be reopened
# within a process (icechunk.in_memory_storage() makes a fresh empty one each call).
_MEMORY_STORES = {}


def storage_from(target, sub_path: str = "", **options) -> icechunk.Storage:
	"""Build an icechunk.Storage for target/sub_path.

	target may be a local path, a URI ("s3://bucket/prefix", "gs://...",
	"az://account/container/prefix", "memory://name"), or an already-built
	icechunk.Storage (returned as-is, sub_path ignored -- the caller owns it).
	Extra keyword options pass straight through to the icechunk constructor,
	e.g. region=/anonymous=/from_env= for s3."""
	if isinstance(target, icechunk.Storage):
		return target

	text = str(target)
	scheme = urlparse(text).scheme

	if scheme in ("", "file"):
		path = Path(text[len("file://"):] if scheme == "file" else text)
		return icechunk.local_filesystem_storage(str(path / sub_path if sub_path else path))

	parsed = urlparse(text)
	prefix = "/".join(p for p in (parsed.path.strip("/"), sub_path) if p)

	if scheme == "memory":
		key = f"{parsed.netloc}/{prefix}"
		if key not in _MEMORY_STORES:
			_MEMORY_STORES[key] = icechunk.in_memory_storage()
		return _MEMORY_STORES[key]
	if scheme in ("s3", "s3a"):
		return icechunk.s3_storage(bucket=parsed.netloc, prefix=prefix or None, **options)
	if scheme == "r2":
		return icechunk.r2_storage(bucket=parsed.netloc, prefix=prefix or None, **options)
	if scheme in ("gs", "gcs"):
		return icechunk.gcs_storage(bucket=parsed.netloc, prefix=prefix or None, **options)
	if scheme in ("az", "abfs", "azure"):
		# az://account/container[/prefix]
		container, _, rest = prefix.partition("/")
		return icechunk.azure_storage(account=parsed.netloc, container=container, prefix=rest, **options)

	raise ValueError(
		f"unsupported storage target {text!r} (scheme {scheme!r}); "
		"use a local path, s3://, r2://, gs://, az://, memory://, or pass an icechunk.Storage"
	)
