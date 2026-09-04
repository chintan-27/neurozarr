import numpy as np
import pandas as pd
import zarr

from .entities import parse_entities
from .log import logger


def _table_df(node, columns: list = None) -> pd.DataFrame:
	"""Rebuild a DataFrame from a stored table.

	Tables are groups of one typed array per column. Stores written before that
	(a single string array for the whole table) are still read, by casting each
	column back to the dtype recorded in attrs.
	"""
	attrs = node.attrs.asdict()
	names = attrs.get("columns")

	if isinstance(node, zarr.Group):  # one array per column
		wanted = columns or names or sorted(node.array_keys())
		return pd.DataFrame({name: node[name][:] for name in wanted})

	df = pd.DataFrame(node[:], columns=names)  # legacy single string array
	for name, dtype in zip(df.columns, attrs.get("dtypes", [])):
		try:
			df[name] = df[name].astype(dtype)
		except (TypeError, ValueError):
			logger.warning("column %r could not be cast back to %s, left as strings", name, dtype)
	return df[columns] if columns else df


class TableView:
	"""A stored table (channels, events, electrodes, a beh table...). Lazy: the
	array is only read when df() is called."""

	def __init__(self, group: zarr.Group, name: str, entities: dict, path: str):
		self._group, self.name, self.entities, self.path = group, name, entities, path

	@property
	def meta(self) -> dict:
		return {k: v for k, v in self._group[self.name].attrs.asdict().items()
				if k not in ("columns", "dtypes")}

	@property
	def columns(self) -> list:
		return list(self._group[self.name].attrs.asdict().get("columns", []))

	def df(self, columns: list = None) -> pd.DataFrame:
		"""The table as a DataFrame. Pass columns to read only those -- each column
		is its own array, so the rest is never fetched."""
		return _table_df(self._group[self.name], columns)

	def __repr__(self):
		return f"<TableView {self.path}/{self.name}>"


class RecordingView:
	"""A stored recording. Lazy: nothing is read until data() or raw() is called."""

	def __init__(self, group: zarr.Group, entities: dict, path: str):
		self._group, self.entities, self.path = group, entities, path

	@property
	def meta(self) -> dict:
		"""The recording's metadata (its BIDS sidecar, plus the data_* keys the
		writer added to describe how the samples are packed)."""
		return self._group.attrs.asdict()

	@property
	def array(self) -> zarr.Array:
		"""The underlying zarr array. Slice it directly if you want raw stored
		values without calibration -- only the chunks you touch are fetched."""
		return self._group["data"]

	@property
	def shape(self) -> tuple:
		return self._group["data"].shape

	@property
	def sfreq(self) -> float:
		return self.meta.get("SamplingFrequency")

	@property
	def duration(self) -> float:
		"""Length in seconds, or None if no sampling frequency is stored."""
		return self.shape[-1] / self.sfreq if self.sfreq else None

	def _picks(self, picks) -> list:
		"""Channel names or indices -> indices."""
		if picks is None:
			return None
		if isinstance(picks, (str, int)):
			picks = [picks]
		names = self.ch_names()
		return [names.index(p) if isinstance(p, str) else p for p in picks]

	def data(self, start: int = None, stop: int = None, tmin: float = None,
			 tmax: float = None, picks=None):
		"""(ndarray, meta) in physical units -- the stored per-channel scale/offset
		is applied when the samples were packed as int16.

		Reads only the window asked for, so a few seconds out of a long recording
		costs a few chunks rather than the whole array::

		    data(start=1000, stop=2000)      # by sample index
		    data(tmin=10, tmax=20)           # by seconds (needs a stored sfreq)
		    data(picks=["LFP_L"])            # one channel
		"""
		meta = self.meta
		if tmin is not None or tmax is not None:
			if not self.sfreq:
				raise ValueError(f"{self.path}: tmin/tmax need a stored SamplingFrequency; use start/stop instead")
			start = int(tmin * self.sfreq) if tmin is not None else start
			stop = int(tmax * self.sfreq) if tmax is not None else stop

		indices = self._picks(picks)
		window = slice(start, stop)
		values = self.array[:, window] if indices is None else self.array[indices, window]

		scale, offset = meta.get("data_scale"), meta.get("data_offset")
		if scale is not None and offset is not None:
			scale, offset = np.array(scale), np.array(offset)
			if indices is not None:
				scale, offset = scale[indices], offset[indices]
			values = values.astype(np.float64) * scale[:, None] + offset[:, None]
		return values, meta

	def channels(self) -> pd.DataFrame:
		"""The channels table stored next to this recording (empty if absent)."""
		if "channels" not in self._group:
			return pd.DataFrame()
		return _table_df(self._group["channels"])

	def events(self) -> pd.DataFrame:
		"""The events table stored next to this recording (empty if absent)."""
		if "events" not in self._group:
			return pd.DataFrame()
		return _table_df(self._group["events"])

	def ch_names(self) -> list:
		"""Channel names: from the stored channels table (BIDS channels.tsv) if there
		is one, else the names the writer captured off the source Raw."""
		channels = self.channels()
		if "name" in channels.columns:
			return list(channels["name"])
		return list(self.meta.get("ch_names", []))

	def raw(self, sfreq: float = None, **window) -> "mne.io.RawArray":
		"""Reconstruct an mne.io.RawArray in physical units, with channel names from
		the stored channels table. sfreq comes from the recording's metadata
		(SamplingFrequency, the BIDS key) unless passed explicitly. Accepts the same
		start/stop/tmin/tmax/picks window arguments as data()."""
		import mne

		values, meta = self.data(**window)
		sfreq = sfreq or meta.get("SamplingFrequency") or meta.get("sfreq")
		if not sfreq:
			raise ValueError(
				f"{self.path}: no sampling frequency stored (looked for 'SamplingFrequency' "
				"in this recording's metadata) -- pass raw(sfreq=...) explicitly"
			)
		names = self.ch_names() or [f"ch{i}" for i in range(values.shape[0])]
		if window.get("picks") is not None:
			names = [names[i] for i in self._picks(window["picks"])]
		info = mne.create_info(names, sfreq=float(sfreq), ch_types="eeg", verbose=False)
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

	def __repr__(self):
		return f"<RecordingView {self.path} shape={self.shape}>"


def _is_table(node) -> bool:
	"""A table is a group of one array per column (or, in older stores, a single
	string array) -- either way it carries a "columns" attr."""
	return "columns" in node.attrs.asdict()


def views_in(root: zarr.Group, base_path: str = ""):
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
