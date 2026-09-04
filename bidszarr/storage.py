from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import icechunk

# One in-memory Storage per logical path, so a memory:// store can be reopened
# within a process (icechunk.in_memory_storage() makes a fresh empty one each call).
_MEMORY_STORES: dict[str, icechunk.Storage] = {}


def storage_from(target: "str | Path | icechunk.Storage", sub_path: str = "",
				 **options: Any) -> icechunk.Storage:
	"""Build an :class:`icechunk.Storage` for ``target/sub_path``.

	Parameters
	----------
	target : str or pathlib.Path or icechunk.Storage
		A local path, a URI, or a pre-built storage. Supported URI schemes are
		``s3://bucket/prefix``, ``r2://``, ``gs://``, ``az://account/container/prefix``
		and ``memory://name``, the last being an in-memory store useful for
		tests. A pre-built storage is returned unchanged, ignoring ``sub_path``.
	sub_path : str, optional
		Path segment appended to the target, used to give each subject its own
		repository.
	**options
		Passed to the underlying icechunk storage constructor, e.g.
		``region=``, ``anonymous=``, ``from_env=``.

	Returns
	-------
	icechunk.Storage

	Raises
	------
	ValueError
		If the URI scheme is not one of those listed above.
	"""
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
