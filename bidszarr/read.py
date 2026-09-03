import numpy as np
import pandas as pd
import zarr

from .entities import parse_entities
from .log import logger


def _table_df(array: zarr.Array) -> pd.DataFrame:
	"""Rebuild a DataFrame from a stored string table, casting each column back
	to the dtype util.create_table recorded in attrs. A column whose dtype no
	longer round-trips is left as strings rather than failing the whole read."""
	attrs = array.attrs.asdict()
	columns = attrs.get("columns")
	df = pd.DataFrame(array[:], columns=columns)
	for name, dtype in zip(df.columns, attrs.get("dtypes", [])):
		try:
			df[name] = df[name].astype(dtype)
		except (TypeError, ValueError):
			logger.warning("column %r could not be cast back to %s, left as strings", name, dtype)
	return df


class TableView:
	"""A stored table (channels, events, electrodes, a beh table...). Lazy: the
	array is only read when df() is called."""

	def __init__(self, group: zarr.Group, name: str, entities: dict, path: str):
		self._group, self.name, self.entities, self.path = group, name, entities, path

	@property
	def meta(self) -> dict:
		return {k: v for k, v in self._group[self.name].attrs.asdict().items()
				if k not in ("columns", "dtypes")}

	def df(self) -> pd.DataFrame:
		"""The table as a DataFrame, with column dtypes restored."""
		return _table_df(self._group[self.name])

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
	def shape(self) -> tuple:
		return self._group["data"].shape

	def data(self):
		"""(ndarray, meta). Values are physical units: if the writer packed the
		samples as int16 it applies the stored per-channel scale/offset, otherwise
		the samples were already stored as physical floats."""
		meta = self.meta
		values = self._group["data"][:]
		scale, offset = meta.get("data_scale"), meta.get("data_offset")
		if scale is not None and offset is not None:
			values = values.astype(np.float64) * np.array(scale)[:, None] + np.array(offset)[:, None]
		return values, meta

	def channels(self) -> pd.DataFrame:
		"""The channels table stored next to this recording (empty if absent)."""
		if "channels" not in self._group:
			return pd.DataFrame()
		return _table_df(self._group["channels"])

	def ch_names(self) -> list:
		"""Channel names: from the stored channels table (BIDS channels.tsv) if there
		is one, else the names the writer captured off the source Raw."""
		channels = self.channels()
		if "name" in channels.columns:
			return list(channels["name"])
		return list(self.meta.get("ch_names", []))

	def raw(self, sfreq: float = None) -> "mne.io.RawArray":
		"""Reconstruct an mne.io.RawArray in physical units, with channel names from
		the stored channels table. sfreq comes from the recording's metadata
		(SamplingFrequency, the BIDS key) unless passed explicitly."""
		import mne

		values, meta = self.data()
		sfreq = sfreq or meta.get("SamplingFrequency") or meta.get("sfreq")
		if not sfreq:
			raise ValueError(
				f"{self.path}: no sampling frequency stored (looked for 'SamplingFrequency' "
				"in this recording's metadata) -- pass raw(sfreq=...) explicitly"
			)
		names = self.ch_names() or [f"ch{i}" for i in range(values.shape[0])]
		info = mne.create_info(names, sfreq=float(sfreq), ch_types="eeg", verbose=False)
		return mne.io.RawArray(values, info, verbose=False)

	def __repr__(self):
		return f"<RecordingView {self.path} shape={self.shape}>"


def views_in(root: zarr.Group, base_path: str = ""):
	"""Walk a subject's tree and yield every RecordingView/TableView in it. A group
	holding a "data" array is a recording; any other string array is a standalone
	table (a beh table, session-level electrodes, ...)."""
	prefix = f"{base_path}/" if base_path else ""
	for path, node in root.members(max_depth=None):
		if not isinstance(node, zarr.Group):
			continue
		entities = parse_entities(path.rsplit("/", 1)[-1])
		if "data" in node:
			yield RecordingView(node, entities, prefix + path)
		else:
			for name, child in node.members():
				if isinstance(child, zarr.Array):
					yield TableView(node, name, entities, prefix + path)
