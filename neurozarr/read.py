from typing import TYPE_CHECKING, Any, Iterator, cast

import numpy as np
import pandas as pd
import zarr

from .entities import parse_entities
from .log import logger

if TYPE_CHECKING:
	import mne  # type: ignore[import-untyped]  # mne ships no type information


def _array(node: zarr.Group, name: str) -> zarr.Array:
	"""Fetch a child the store's layout guarantees is an array.

	Indexing a zarr Group is typed as returning either an Array or a Group; the
	writer only ever puts arrays at these names, so narrow it here rather than
	at every call site.
	"""
	return cast(zarr.Array, node[name])


def _table_df(node: zarr.Group | zarr.Array, columns: list[str] | None = None) -> pd.DataFrame:
	"""Rebuild a DataFrame from a stored table.

	Tables are groups of one typed array per column. Stores written before that
	(a single string array for the whole table) are still read, by casting each
	column back to the dtype recorded in attrs.
	"""
	attrs = node.attrs.asdict()
	names = cast("list[str] | None", attrs.get("columns"))

	if isinstance(node, zarr.Group):  # one array per column
		wanted = columns or names or sorted(node.array_keys())
		return pd.DataFrame({name: _array(node, name)[:] for name in wanted})

	df = pd.DataFrame(cast(Any, node[:]), columns=names)  # legacy single string array
	for name, dtype in zip(df.columns, cast("list[str]", attrs.get("dtypes", []))):
		try:
			df[name] = df[name].astype(dtype)  # type: ignore[call-overload]
		except (TypeError, ValueError):
			logger.warning("column %r could not be cast back to %s, left as strings", name, dtype)
	return df[columns] if columns else df


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

	def __init__(self, group: zarr.Group, name: str, entities: dict[str, str], path: str):
		self._group, self.name, self.entities, self.path = group, name, entities, path

	@property
	def meta(self) -> dict[str, Any]:
		"""The table's own metadata, without the bookkeeping the writer added."""
		return {k: v for k, v in self._group[self.name].attrs.asdict().items()
				if k not in ("columns", "dtypes")}

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

	def __init__(self, group: zarr.Group, entities: dict, path: str):
		self._group, self.entities, self.path = group, entities, path

	@property
	def meta(self) -> dict[str, Any]:
		"""The recording's metadata (its BIDS sidecar, plus the data_* keys the
		writer added to describe how the samples are packed)."""
		return self._group.attrs.asdict()

	@property
	def array(self) -> zarr.Array:
		"""The underlying zarr array. Slice it directly if you want raw stored
		values without calibration -- only the chunks you touch are fetched."""
		return _array(self._group, "data")

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
		if tmin is not None or tmax is not None:
			if not self.sfreq:
				raise ValueError(f"{self.path}: tmin/tmax need a stored SamplingFrequency; use start/stop instead")
			start = int(tmin * self.sfreq) if tmin is not None else start
			stop = int(tmax * self.sfreq) if tmax is not None else stop

		indices = self._picks(picks)
		window = slice(start, stop)
		# zarr types __getitem__ as possibly returning a scalar; selecting along
		# two axes of a 2-D array always gives an array back.
		values = cast(np.ndarray, self.array[:, window] if indices is None
					  else self.array[indices, window])  # type: ignore[index]

		raw_scale, raw_offset = meta.get("data_scale"), meta.get("data_offset")
		if raw_scale is not None and raw_offset is not None:
			scale, offset = np.array(raw_scale), np.array(raw_offset)
			if indices is not None:
				scale, offset = scale[indices], offset[indices]
			values = values.astype(np.float64) * scale[:, None] + offset[:, None]
		return values, meta

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
		if not events.empty and "onset" in events and not window:
			raw.set_annotations(mne.Annotations(
				onset=events["onset"].astype(float),
				duration=events["duration"].astype(float) if "duration" in events else 0.0,
				description=events.get("trial_type", events.get("description", "")),
			), verbose=False)
		return raw

	def __repr__(self) -> str:
		return f"<RecordingView {self.path} shape={self.shape}>"


def _is_table(node: zarr.Group | zarr.Array) -> bool:
	"""A table is a group of one array per column (or, in older stores, a single
	string array) -- either way it carries a "columns" attr."""
	return "columns" in node.attrs.asdict()


def views_in(root: zarr.Group, base_path: str = "") -> Iterator["RecordingView | TableView"]:
	"""Walk a subject's tree and yield every RecordingView/TableView in it. A group
	holding a "data" array is a recording; anything carrying a "columns" attr is a
	table (channels, events, electrodes, a beh table, ...)."""
	prefix = f"{base_path}/" if base_path else ""
	for path, node in root.members(max_depth=None):
		if not isinstance(node, zarr.Group) or _is_table(node):
			continue  # a table's own group is yielded by its parent, not walked into
		entities = parse_entities(path.rsplit("/", 1)[-1])
		if "data" in node:
			yield RecordingView(node, entities, prefix + path)
		else:
			for name, child in node.members():
				if _is_table(child):
					yield TableView(node, name, entities, prefix + path)
