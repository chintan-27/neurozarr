from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias
from urllib.parse import urlparse

import icechunk

from .constraints import DATASET_REPO, validate_segment
from .errors import ValidationError

StorageFactory: TypeAlias = Callable[[str], icechunk.Storage]
StorageTarget: TypeAlias = str | Path | icechunk.Storage | StorageFactory

# One in-memory Storage per logical path, so a memory:// store can be reopened
# within a process (icechunk.in_memory_storage() makes a fresh empty one each call).
_MEMORY_STORES: dict[str, icechunk.Storage] = {}


def storage_from(target: StorageTarget, sub_path: str = "",
				 **options: Any) -> icechunk.Storage:
	"""Build an :class:`icechunk.Storage` for ``target/sub_path``.

	Parameters
	----------
	target : str or pathlib.Path or icechunk.Storage or callable
		A local path, a URI, a storage factory, or pre-built storage for a
		single repository. Supported URI schemes are
		``s3://bucket/prefix``, ``r2://``, ``gs://``, ``az://account/container/prefix``
		and ``memory://name``, the last being an in-memory store useful for tests.
		A factory receives ``sub_path`` and must return independent storage.
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
	if sub_path and sub_path != DATASET_REPO:
		validate_segment(sub_path, "storage sub-path")
	if callable(target) and not isinstance(target, type):
		return target(sub_path)
	if isinstance(target, icechunk.Storage):
		if sub_path:
			raise ValidationError(
				"a single icechunk.Storage cannot hold neurozarr's separate subject repositories; "
				"pass a callable like `lambda sub_id: make_storage(sub_id)` instead"
			)
		return target

	text = str(target)
	scheme = urlparse(text).scheme

	if scheme in ("", "file"):
		path = Path(text[len("file://"):] if scheme == "file" else text)
		return icechunk.local_filesystem_storage(str(path / sub_path if sub_path else path))

	parsed = urlparse(text)
	if parsed.username or parsed.password or parsed.query or parsed.fragment:
		raise ValidationError(
			"storage URIs must not contain credentials, query strings, or fragments; pass credentials as options"
		)
	prefix = "/".join(p for p in (parsed.path.strip("/"), sub_path) if p)

	if scheme == "memory":
		key = f"{parsed.netloc}/{prefix}"
		if key not in _MEMORY_STORES:
			_MEMORY_STORES[key] = icechunk.in_memory_storage()
		return _MEMORY_STORES[key]
	if scheme in ("s3", "s3a"):
		if not parsed.netloc:
			raise ValidationError("S3 storage URI must include a bucket")
		return icechunk.s3_storage(bucket=parsed.netloc, prefix=prefix or None, **options)
	if scheme == "r2":
		if not parsed.netloc:
			raise ValidationError("R2 storage URI must include a bucket")
		return icechunk.r2_storage(bucket=parsed.netloc, prefix=prefix or None, **options)
	if scheme in ("gs", "gcs"):
		if not parsed.netloc:
			raise ValidationError("GCS storage URI must include a bucket")
		return icechunk.gcs_storage(bucket=parsed.netloc, prefix=prefix or None, **options)
	if scheme in ("az", "abfs", "azure"):
		# az://account/container[/prefix]
		container, _, rest = prefix.partition("/")
		if not parsed.netloc or not container:
			raise ValidationError("Azure storage URI must include an account and container")
		return icechunk.azure_storage(account=parsed.netloc, container=container, prefix=rest, **options)

	raise ValueError(
		f"unsupported storage target {text!r} (scheme {scheme!r}); "
		"use a local path, s3://, r2://, gs://, az://, memory://, or a storage factory"
	)
