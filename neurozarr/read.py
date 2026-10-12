from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Iterator, NamedTuple, cast

import numpy as np
import pandas as pd
import zarr

from .entities import parse_entities
from .items import Attrs
from .log import logger

if TYPE_CHECKING:
	import mne  # type: ignore[import-untyped]  # mne ships no type information


class WriteBack(NamedTuple):
	"""Bound to a view obtained from a writable Repo (by Subject/Visit, via
	views_in's write= param), so set_attrs()/rename()/delete() can route a
	change back through Repo._route_attrs/_route_rename/_route_delete without
	read.py depending on Repo itself."""
	set_attrs: Callable[[Attrs], None]
	rename: Callable[[tuple[str, ...], str], None]
	delete: Callable[[tuple[str, ...]], None]


_DECODER: ThreadPoolExecutor | None = None


def _decoder() -> ThreadPoolExecutor:
	"""One background thread for chunk prefetch, shared by every view, started on first use."""
	global _DECODER
	if _DECODER is None:
		_DECODER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neurozarr-prefetch")
	return _DECODER


def _array(node: zarr.Group, name: str) -> zarr.Array:
	"""Fetch a child the store's layout guarantees is an array.

	Indexing a zarr Group is typed as returning either an Array or a Group; the
	writer only ever puts arrays at these names, so narrow it here rather than
	at every call site.
	"""
	return cast(zarr.Array, node[name])


def _table_df(node: zarr.Group | zarr.Array, columns: list[str] | None = None) -> pd.DataFrame:
	"""Rebuild a DataFrame from a stored table.

	Tables are groups of typed arrays, one per column or, since schema 3,
	shared by several columns that pack into one array (see
	util.create_table). Stores written before either (a single string array
	for the whole table) are still read, by casting each column back to the
	dtype recorded in attrs.
	"""
	attrs = node.attrs.asdict()
	names = cast("list[str] | None", attrs.get("columns"))

	if isinstance(node, zarr.Group) and attrs.get("table_schema_version") in (2, 3):
		series_list: list[pd.Series] = []
		for spec in cast("list[dict[str, Any]]", attrs.get("schema", [])):
			if "array" in spec:  # schema 3: packed into a shared array, one row per column
				array_name, row = str(spec["array"]), int(spec["row"])
				values = np.asarray(_array(node, array_name)[row])
				mask_name = f"{array_name}__mask"
				mask = np.asarray(_array(node, mask_name)[row]).astype(bool) if mask_name in node \
					else np.zeros(len(values), dtype=bool)
			else:  # schema 2, or a schema-3 column that wasn't packed
				internal = str(spec["id"])
				values = np.asarray(_array(node, internal)[:])
				mask_name = f"{internal}__mask"
				mask = np.asarray(_array(node, mask_name)[:]).astype(bool) if mask_name in node \
					else np.zeros(len(values), dtype=bool)
			encoding = spec.get("encoding", "native")
			logical = str(spec.get("dtype", values.dtype))

			if encoding == "categorical":
				data: Any = pd.Categorical.from_codes(
					values.astype(int), spec.get("categories", []), ordered=bool(spec.get("ordered", False))
				)
			elif encoding == "datetime64[ns]":
				if "," in logical:
					timezone = logical.rsplit(",", 1)[1].rstrip("]").strip()
					data = pd.Series(pd.to_datetime(values.astype("int64"), utc=True).tz_convert(timezone))
				else:
					data = pd.Series(pd.to_datetime(values.astype("int64")))
				data[mask] = pd.NaT
			elif encoding == "timedelta64[ns]":
				data = pd.Series(pd.to_timedelta(values.astype("int64"), unit="ns"))
				data[mask] = pd.NaT
			elif encoding in ("integer", "boolean"):
				uses_extension = logical[:1].isupper() or logical == "boolean"
				if mask.any() or uses_extension:
					try:
						data = pd.Series(pd.array(values, dtype=logical))  # type: ignore[call-overload]
					except (TypeError, ValueError):
						fallback = "Int64" if encoding == "integer" else "boolean"
						data = pd.Series(pd.array(values, dtype=fallback))  # type: ignore[call-overload]
					data[mask] = pd.NA
				else:
					data = pd.Series(values.astype(logical))
			elif encoding == "float":
				if logical[:1].isupper():
					data = pd.Series(pd.array(values, dtype=logical))  # type: ignore[call-overload]
					data[mask] = pd.NA
				else:
					data = pd.Series(values.astype(logical))
			elif encoding == "string":
				# Not .astype(str): values may be numpy's variable-length
				# StringDType (util.py's create_table writes string columns as
				# that, not a fixed-width `<U*>` array), which numpy refuses to
				# cast to a fixed-width str dtype of unspecified length.
				# pandas builds a Series from either array kind directly.
				data = pd.Series(values, dtype="object" if logical == "object" else "string")
				data[mask] = None if logical == "object" else pd.NA
			else:
				data = pd.Series(values)
				if mask.any():
					data[mask] = np.nan
			series_list.append(pd.Series(data, name=str(spec.get("name", spec.get("id", spec.get("array"))))))
		df = pd.concat(series_list, axis=1) if series_list else pd.DataFrame(columns=names or [])
		return df[columns] if columns else df

	if isinstance(node, zarr.Group):  # legacy one-array-per-column layout
		wanted = columns or names or sorted(node.array_keys())
		return pd.DataFrame({name: _array(node, name)[:] for name in wanted})

	df = pd.DataFrame(cast(Any, node[:]), columns=names)  # legacy single string array
	for name, dtype in zip(df.columns, cast("list[str]", attrs.get("dtypes", []))):
		try:
			df[name] = df[name].astype(dtype)  # type: ignore[call-overload]
		except (TypeError, ValueError):
			logger.warning("column %r could not be cast back to %s, left as strings", name, dtype)
	return df[columns] if columns else df


def _target_path(path: str, name: str | None) -> tuple[str, ...]:
	segments = tuple(path.split("/"))
	return (*segments, name) if name is not None else segments


def _require_write(write: "WriteBack | None") -> "WriteBack":
	if write is None:
		raise TypeError("this view has no writable Repo attached")
	return write


class TableView:
	"""A stored table: channels, events, electrodes, a behavioral log, and so on.

	Nothing is read from storage until :meth:`df` is called. Obtained from
	:meth:`Subject.tables`, :meth:`Visit.tables`, or as an attribute of a
	recording — not constructed directly.

	Attributes
	----------
	name : str
		The table's name within its group, e.g. ``"channels"``.
	entities : dict
		BIDS entities parsed from the group name, e.g. ``{"task": "Rest"}``.
	path : str
		Where the table sits in the store, for display and logging.
	"""

	def __init__(self, group: "zarr.Group | Callable[[], Any]", name: str, entities: dict[str, str], path: str,
				 _write: "WriteBack | None" = None):
		self._node: Any = group
		self.name, self.entities, self.path = name, entities, path
		self._write = _write

	@property
	def _group(self) -> zarr.Group:
		# a view may be handed a loader instead of the open group, so listing query
		# results (paths, entities, counts) opens nothing; the group opens on first use
		if callable(self._node):
			self._node = self._node()
		return cast(zarr.Group, self._node)

	@property
	def meta(self) -> dict[str, Any]:
		"""The table's own metadata, without the bookkeeping the writer added."""
		return {k: v for k, v in self._group[self.name].attrs.asdict().items()
				if k not in ("_neurozarr_item_type", "table_schema_version", "columns", "dtypes", "schema")}

	def set_attrs(self, attrs: dict[str, Any]) -> None:
		"""Add or update this table's metadata after creation.

		Merges into whatever attrs already exist rather than replacing them.
		Not committed until :meth:`Repo.save`. Only available on a view obtained
		from a writable :class:`Repo`; this view keeps showing pre-change data
		until the store is reopened.
		"""
		_require_write(self._write).set_attrs(Attrs(_target_path(self.path, self.name), attrs))

	def rename(self, new_name: str) -> None:
		"""Rename this table.

		Not committed until :meth:`Repo.save`. Only available on a view
		obtained from a writable :class:`Repo`. Do not reuse this view
		afterwards -- it still points at the old name; open a fresh one.
		"""
		_require_write(self._write).rename(_target_path(self.path, self.name), new_name)

	def delete(self) -> None:
		"""Delete this table.

		Not committed until :meth:`Repo.save`. Only available on a view
		obtained from a writable :class:`Repo`.
		"""
		_require_write(self._write).delete(_target_path(self.path, self.name))

	@property
	def columns(self) -> list[str]:
		"""Names of the table's columns, without reading any of its values."""
		return cast("list[str]", self._group[self.name].attrs.asdict().get("columns", []))

	def df(self, columns: list[str] | None = None) -> pd.DataFrame:
		"""Read the table into a :class:`pandas.DataFrame`.

		Column dtypes recorded at write time are restored, so numeric columns
		come back as numbers rather than strings.

		Parameters
		----------
		columns : list of str, optional
			Read only these columns. Default reads all of them.

		Returns
		-------
		pandas.DataFrame
		"""
		return _table_df(self._group[self.name], columns)

	def __repr__(self) -> str:
		return f"<TableView {self.path}/{self.name}>"


class RecordingView:
	"""A stored recording — one continuous run of sample data plus its metadata.

	Nothing is read from storage until :meth:`data` or :meth:`raw` is called, so
	holding a view is cheap. Obtained from :meth:`Visit.recording`,
	:meth:`Subject.recordings` or :meth:`Repo.find` — not constructed directly.

	Attributes
	----------
	entities : dict
		BIDS entities parsed from the group name, e.g. ``{"task": "Rest", "run": "1"}``.
	path : str
		Where the recording sits in the store, e.g.
		``"sub-001/ses-1/ieeg/task-Rest_run-1"``.
	"""

	def __init__(self, group: "zarr.Group | Callable[[], zarr.Group]", entities: dict, path: str,
				 _write: "WriteBack | None" = None):
		self._node: Any = group
		self.entities, self.path = entities, path
		self._write = _write
		self._data: zarr.Array | None = None
		# decoded chunks of recent narrow reads, by (channels, chunk index): a loop of
		# small windows (a scan in blocks) decodes each chunk once, not once per block
		self._chunks: "OrderedDict[tuple[Any, int], np.ndarray | Future[np.ndarray]]" = OrderedDict()
		self._last_chunk: tuple[Any, int] | None = None

	@property
	def _group(self) -> zarr.Group:
		# a view may be handed a loader instead of the open group, so listing query
		# results (paths, entities, counts) opens nothing; the group opens on first use
		if callable(self._node):
			self._node = self._node()
		return cast(zarr.Group, self._node)

	def set_attrs(self, attrs: dict[str, Any]) -> None:
		"""Add or update this recording's metadata after creation.

		Merges into whatever attrs already exist rather than replacing them.
		Not committed until :meth:`Repo.save`. Only available on a view obtained
		from a writable :class:`Repo`; this view keeps showing pre-change data
		until the store is reopened.
		"""
		_require_write(self._write).set_attrs(Attrs(_target_path(self.path, None), attrs))

	def rename(self, new_name: str) -> None:
		"""Rename this recording -- changes its raw group key, not its BIDS
		entities, so the new name need not follow BIDS ``key-value`` form.

		Not committed until :meth:`Repo.save`. Only available on a view
		obtained from a writable :class:`Repo`. Do not reuse this view
		afterwards -- it still points at the old name; open a fresh one.
		"""
		_require_write(self._write).rename(_target_path(self.path, None), new_name)

	def delete(self) -> None:
		"""Delete this recording and everything under it (its data, channels,
		and events).

		Not committed until :meth:`Repo.save`. Only available on a view
		obtained from a writable :class:`Repo`.
		"""
		_require_write(self._write).delete(_target_path(self.path, None))

	@property
	def meta(self) -> dict[str, Any]:
		"""The recording's metadata (its BIDS sidecar, plus the data_* keys the
		writer added to describe how the samples are packed)."""
		return {key: value for key, value in self._group.attrs.asdict().items()
				if key != "_neurozarr_item_type"}

	@property
	def array(self) -> zarr.Array:
		"""The underlying zarr array. Slice it directly if you want raw stored
		values without calibration -- only the chunks you touch are fetched."""
		if self._data is None:  # opened once: re-opening it cost a metadata read per data() call
			self._data = _array(self._group, "data")
		return self._data

	@property
	def shape(self) -> tuple[int, ...]:
		"""Shape of the stored data as ``(n_channels, n_samples)``."""
		return self.array.shape

	@property
	def sfreq(self) -> float | None:
		"""Sampling frequency in Hz, or ``None`` if the recording has none stored."""
		return self.meta.get("SamplingFrequency")

	@property
	def duration(self) -> float | None:
		"""Length in seconds, or ``None`` if no sampling frequency is stored."""
		return self.shape[-1] / self.sfreq if self.sfreq else None

	def _picks(self, picks: "str | int | list[str | int] | None") -> list[int] | None:
		"""Channel names or indices -> indices."""
		if picks is None:
			return None
		if isinstance(picks, (str, int)):
			picks = [picks]
		names = self.ch_names()
		return [names.index(p) if isinstance(p, str) else p for p in picks]

	def data(self, start: int | None = None, stop: int | None = None, tmin: float | None = None,
			 tmax: float | None = None,
			 picks: "str | int | list[str | int] | None" = None) -> tuple[np.ndarray, dict[str, Any]]:
		"""Read samples, converted back to physical units.

		Only the requested window is fetched from storage, so reading a few
		seconds out of a long recording costs a few chunks rather than the
		whole array.

		Parameters
		----------
		start, stop : int, optional
			Window bounds as sample indices, following Python slice semantics
			(``stop`` exclusive). Default is the whole recording.
		tmin, tmax : float, optional
			Window bounds in seconds. Converted to sample indices using the
			stored sampling frequency, and may not be combined meaningfully
			with ``start``/``stop`` for the same edge.
		picks : str or int or list, optional
			Channels to read, as names (matched against :meth:`ch_names`) or
			integer indices. Default reads every channel.

		Returns
		-------
		values : numpy.ndarray
			Array of shape ``(n_channels, n_samples)``. When the recording was
			packed as int16, the stored per-channel scale and offset are applied
			so the result is in the channel's native physical unit.
		meta : dict
			The recording's metadata, as stored.

		Raises
		------
		ValueError
			If ``tmin`` or ``tmax`` is given but the recording has no stored
			sampling frequency. Use ``start``/``stop`` instead in that case.

		Examples
		--------
		>>> values, meta = rec.data(start=1000, stop=2000)
		>>> values, meta = rec.data(tmin=10, tmax=20)
		>>> values, meta = rec.data(tmin=10, tmax=20, picks=["LFP_L"])
		"""
		meta = self.meta
		if start is not None and tmin is not None:
			raise ValueError("pass either start or tmin, not both")
		if stop is not None and tmax is not None:
			raise ValueError("pass either stop or tmax, not both")
		if tmin is not None and tmin < 0 or tmax is not None and tmax < 0:
			raise ValueError("time bounds cannot be negative")
		if tmin is not None and tmax is not None and tmax < tmin:
			raise ValueError("tmax cannot be earlier than tmin")
		if tmin is not None or tmax is not None:
			if not self.sfreq:
				raise ValueError(f"{self.path}: tmin/tmax need a stored SamplingFrequency; use start/stop instead")
			start = int(tmin * self.sfreq) if tmin is not None else start
			stop = int(tmax * self.sfreq) if tmax is not None else stop

		indices = self._picks(picks)
		values = self._stored(indices, slice(start, stop))

		raw_scale, raw_offset = meta.get("data_scale"), meta.get("data_offset")
		if raw_scale is not None and raw_offset is not None:
			scale, offset = np.array(raw_scale), np.array(raw_offset)
			if indices is not None:
				scale, offset = scale[indices], offset[indices]
			# one pass that casts and scales into one float64 array (an astype then *= was two
			# passes, and a plain expression three allocations: 13 GB each on a long iEEG recording)
			values = np.multiply(values, scale[:, None], dtype=np.float64)
			if offset.any():
				values += offset[:, None]
		elif values.base is not None:
			values = values.copy()  # a cut of a cached chunk: the caller may write to its result
		return values, meta

	def _stored(self, indices: list[int] | None, window: slice) -> np.ndarray:
		"""Stored values of a window. A window within two chunks along time is cut from
		cached decoded chunks (zarr decodes whole chunks anyway); wider reads go straight
		to zarr, which fetches their chunks concurrently."""
		array = self.array
		n, step = array.shape[-1], array.chunks[-1]
		start, stop, _ = window.indices(n)
		first, last = start // step, (max(stop, start + 1) - 1) // step
		if stop <= start or last - first > 1:
			# zarr types __getitem__ as possibly returning a scalar; selecting along
			# two axes of a 2-D array always gives an array back.
			return cast(np.ndarray, array[:, window] if indices is None
						else array[indices, window])  # type: ignore[index]
		key = None if indices is None else tuple(indices)
		pieces = []
		for k in range(first, last + 1):
			block = self._chunk(array, indices, key, k)
			lo, hi = max(start, k * step) - k * step, min(stop, (k + 1) * step) - k * step
			pieces.append(block[:, lo:hi])
		# reading forward chunk by chunk (a scan): decode the next chunk while this one is used
		previous, self._last_chunk = self._last_chunk, (key, last)
		if previous is not None and previous[0] == key and previous[1] in (last - 1, last) \
				and (last + 1) * step < n and (key, last + 1) not in self._chunks:
			self._chunks[(key, last + 1)] = _decoder().submit(self._read_chunk, array, indices, last + 1)
			self._trim()
		return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=-1)

	@staticmethod
	def _read_chunk(array: zarr.Array, indices: list[int] | None, k: int) -> np.ndarray:
		step = array.chunks[-1]
		span = slice(k * step, min(array.shape[-1], (k + 1) * step))
		return cast(np.ndarray, array[:, span] if indices is None else array[indices, span])  # type: ignore[index]

	def _chunk(self, array: zarr.Array, indices: list[int] | None, key: Any, k: int) -> np.ndarray:
		"""One decoded chunk column: from the cache, from a prefetch in flight, or read now."""
		block = self._chunks.get((key, k))
		if block is None:
			block = self._read_chunk(array, indices, k)
		elif isinstance(block, Future):
			block = block.result()
		self._chunks[(key, k)] = block
		self._chunks.move_to_end((key, k))
		self._trim()
		return block

	def _trim(self) -> None:
		# ponytail: 4 chunk columns per view (~32 MB at the 8 MiB chunk target);
		# size it by bytes if views ever hold many channel subsets at once
		while len(self._chunks) > 4:
			self._chunks.popitem(last=False)

	def channels(self) -> pd.DataFrame:
		"""Read the channels table stored alongside this recording.

		Returns
		-------
		pandas.DataFrame
			The BIDS ``channels.tsv`` contents, or an empty frame if the
			recording has no channels table.
		"""
		if "channels" not in self._group:
			return pd.DataFrame()
		return _table_df(self._group["channels"])

	def events(self) -> pd.DataFrame:
		"""Read the events table stored alongside this recording.

		Returns
		-------
		pandas.DataFrame
			The BIDS ``events.tsv`` contents — which also carry any annotations
			the source recording had — or an empty frame if there are none.
		"""
		if "events" not in self._group:
			return pd.DataFrame()
		return _table_df(self._group["events"])

	def ch_names(self) -> list:
		"""Channel names, taken from the stored channels table when there is one
		and otherwise from the names captured off the source recording.

		Returns
		-------
		list of str
		"""
		channels = self.channels()
		if "name" in channels.columns:
			return list(channels["name"])
		return cast("list[str]", self.meta.get("ch_names", []))

	def raw(self, sfreq: float | None = None, **window: Any) -> "mne.io.RawArray":
		"""Reconstruct this recording as an :class:`mne.io.RawArray`.

		Channel names come from the stored channels table, and any stored events
		are restored as annotations when the whole recording is read.

		Parameters
		----------
		sfreq : float, optional
			Sampling frequency in Hz. Defaults to the recording's stored
			``SamplingFrequency``.
		**window
			Any of the ``start``, ``stop``, ``tmin``, ``tmax`` and ``picks``
			arguments accepted by :meth:`data`.

		Returns
		-------
		mne.io.RawArray
			The recording in physical units, ready to pass to mne.

		Raises
		------
		ValueError
			If no sampling frequency is stored and none was passed.

		See Also
		--------
		data : Read the samples as a plain array.
		"""
		import mne

		values, meta = self.data(**window)
		rate = sfreq or meta.get("SamplingFrequency") or meta.get("sfreq")
		if not rate:
			raise ValueError(
				f"{self.path}: no sampling frequency stored (looked for 'SamplingFrequency' "
				"in this recording's metadata) -- pass raw(sfreq=...) explicitly"
			)
		names = self.ch_names() or [f"ch{i}" for i in range(values.shape[0])]
		picked = self._picks(window.get("picks"))
		if picked is not None:
			names = [names[i] for i in picked]
		info = mne.create_info(names, sfreq=float(rate), ch_types="eeg", verbose=False)
		raw = mne.io.RawArray(values, info, verbose=False)

		# put the events table back on as annotations, so a Raw survives the round trip
		events = self.events()
		time_windowed = any(key in window for key in ("start", "stop", "tmin", "tmax"))
		if not events.empty and "onset" in events and not time_windowed:
			description = events["trial_type"] if "trial_type" in events \
				else events["description"] if "description" in events else [""] * len(events)
			raw.set_annotations(mne.Annotations(
				onset=events["onset"].astype(float),
				duration=events["duration"].astype(float) if "duration" in events else 0.0,
				description=description,
			), verbose=False)
		return raw

	def __repr__(self) -> str:
		return f"<RecordingView {self.path} shape={self.shape}>"


class ArrayView:
	"""A lazily accessed named N-dimensional array."""

	def __init__(self, array: "zarr.Array | Callable[[], Any]", name: str, entities: dict[str, str], path: str,
				 _write: "WriteBack | None" = None):
		self._node: Any = array
		self.name, self.entities, self.path = name, entities, path
		self._write = _write

	@property
	def array(self) -> zarr.Array:
		"""The stored array (opened on first use)."""
		if callable(self._node):
			self._node = self._node()
		return cast(zarr.Array, self._node)

	def set_attrs(self, attrs: dict[str, Any]) -> None:
		"""Add or update this array's metadata after creation.

		Merges into whatever attrs already exist rather than replacing them.
		Not committed until :meth:`Repo.save`. Only available on a view obtained
		from a writable :class:`Repo`; this view keeps showing pre-change data
		until the store is reopened.
		"""
		_require_write(self._write).set_attrs(Attrs(_target_path(self.path, self.name), attrs))

	def rename(self, new_name: str) -> None:
		"""Rename this array.

		Not committed until :meth:`Repo.save`. Only available on a view
		obtained from a writable :class:`Repo`. Do not reuse this view
		afterwards -- it still points at the old name; open a fresh one.
		"""
		_require_write(self._write).rename(_target_path(self.path, self.name), new_name)

	def delete(self) -> None:
		"""Delete this array.

		Not committed until :meth:`Repo.save`. Only available on a view
		obtained from a writable :class:`Repo`.
		"""
		_require_write(self._write).delete(_target_path(self.path, self.name))

	@property
	def dims(self) -> tuple[str, ...]:
		return tuple(cast("list[str]", self.array.attrs.asdict().get("dims", [])))

	@property
	def coords(self) -> dict[str, Any]:
		return cast("dict[str, Any]", self.array.attrs.asdict().get("coords", {}))

	@property
	def meta(self) -> dict[str, Any]:
		return {key: value for key, value in self.array.attrs.asdict().items()
				if key not in ("_neurozarr_item_type", "dims", "coords")}

	@property
	def shape(self) -> tuple[int, ...]:
		return self.array.shape

	def data(self, selection: Any = None) -> np.ndarray:
		"""Read the whole array or a NumPy-style selection."""
		return np.asarray(self.array[:] if selection is None else self.array[selection])

	def __repr__(self) -> str:
		return f"<ArrayView {self.path}/{self.name} shape={self.shape}>"


class ExternalFileView:
	"""A source file preserved by reference because no decoder was available."""

	def __init__(self, group: "zarr.Group | Callable[[], Any]", name: str, entities: dict[str, str], path: str,
				 _write: "WriteBack | None" = None):
		self._node: Any = group
		self.name, self.entities, self.path = name, entities, path
		self._write = _write

	@property
	def _group(self) -> zarr.Group:
		# a view may be handed a loader instead of the open group, so listing query
		# results (paths, entities, counts) opens nothing; the group opens on first use
		if callable(self._node):
			self._node = self._node()
		return cast(zarr.Group, self._node)

	def set_attrs(self, attrs: dict[str, Any]) -> None:
		"""Add or update this external file's metadata after creation.

		Merges into whatever attrs already exist rather than replacing them.
		Not committed until :meth:`Repo.save`. Only available on a view obtained
		from a writable :class:`Repo`; this view keeps showing pre-change data
		until the store is reopened.
		"""
		_require_write(self._write).set_attrs(Attrs(_target_path(self.path, self.name), attrs))

	def rename(self, new_name: str) -> None:
		"""Rename this external-file reference.

		Not committed until :meth:`Repo.save`. Only available on a view
		obtained from a writable :class:`Repo`. Do not reuse this view
		afterwards -- it still points at the old name; open a fresh one.
		"""
		_require_write(self._write).rename(_target_path(self.path, self.name), new_name)

	def delete(self) -> None:
		"""Delete this external-file reference. The referenced source file
		itself is untouched -- only the reference stored in the tree is removed.

		Not committed until :meth:`Repo.save`. Only available on a view
		obtained from a writable :class:`Repo`.
		"""
		_require_write(self._write).delete(_target_path(self.path, self.name))

	@property
	def uri(self) -> str:
		return str(self._group.attrs.asdict()["uri"])

	@property
	def meta(self) -> dict[str, Any]:
		return {key: value for key, value in self._group.attrs.asdict().items()
				if key not in ("_neurozarr_item_type", "uri", "media_type", "reader_hint")}

	@property
	def media_type(self) -> str | None:
		return cast("str | None", self._group.attrs.asdict().get("media_type"))

	@property
	def reader_hint(self) -> str | None:
		return cast("str | None", self._group.attrs.asdict().get("reader_hint"))


def _is_table(node: zarr.Group | zarr.Array) -> bool:
	"""A table is a group of one array per column (or, in older stores, a single
	string array) -- either way it carries a "columns" attr."""
	return "columns" in node.attrs.asdict()


def _is_recording(node: zarr.Group) -> bool:
	if node.attrs.asdict().get("_neurozarr_item_type") == "recording":
		return True
	if "data" not in node:
		return False
	child = node["data"]
	return isinstance(child, zarr.Array) and child.attrs.asdict().get("_neurozarr_item_type") != "array"


# A recording's own facade children -- reachable via RecordingView.channels()/
# .events(), never as independent top-level items, so the sibling scan below
# must not re-yield them.
_RECORDING_OWN_CHILDREN = frozenset({"data", "events", "channels"})


def views_in(root: zarr.Group, base_path: str = "",
			 write: "WriteBack | None" = None) -> Iterator["RecordingView | TableView | ArrayView | ExternalFileView"]:
	"""Walk a subject's tree and yield every item view in it. A group holding a
	"data" array is a recording; anything carrying a "columns" attr is a table
	(channels, events, electrodes, a beh table, ...). A group can be a recording
	and still hold a sibling table, array, or external-file reference under the
	same entities (a BIDS recording next to an unsupported-format sidecar file,
	say); both are yielded, not one at the expense of the other.

	write, when given, is attached to each view so its set_attrs()/rename()/
	delete() can route a change back through Repo -- read.py itself never
	writes and doesn't know about Repo."""
	prefix = f"{base_path}/" if base_path else ""
	for path, node in root.members(max_depth=None):
		if isinstance(node, zarr.Group):
			yield from views_at(node, prefix + path, write)


def owns_items(node: Any) -> bool:
	"""Whether ``node`` is a group whose items :func:`views_at` lists -- not a table's
	own group or an external-file reference, which belong to their parent."""
	return (isinstance(node, zarr.Group) and not _is_table(node)
			and node.attrs.asdict().get("_neurozarr_item_type") != "external_file")


def views_at(node: zarr.Group, path: str,
			 write: "WriteBack | None" = None) -> Iterator["RecordingView | TableView | ArrayView | ExternalFileView"]:
	"""The items one group holds: the recording it is, if it is one, and the tables,
	arrays and external files directly in it. :func:`views_in` is this for every group."""
	if not owns_items(node):
		return  # a table's own group, or an external-file reference, is yielded by its parent
	entities = parse_entities(path.rsplit("/", 1)[-1])
	is_recording = _is_recording(node)
	if is_recording:
		yield RecordingView(node, entities, path, write)
	for name, child in node.members():
		if is_recording and name in _RECORDING_OWN_CHILDREN:
			continue
		if _is_table(child):
			yield TableView(node, name, entities, path, write)
		elif isinstance(child, zarr.Array) and child.attrs.asdict().get("_neurozarr_item_type") == "array":
			yield ArrayView(child, name, entities, path, write)
		elif isinstance(child, zarr.Group) and child.attrs.asdict().get("_neurozarr_item_type") == "external_file":
			yield ExternalFileView(child, name, entities, path, write)
