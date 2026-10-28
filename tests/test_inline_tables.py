"""Small tables are stored inline in their group's attrs (table_schema_version 4);
large ones, or ones with values JSON can't hold, keep the column arrays. Either way
a read gives back the same DataFrame."""

import numpy as np
import pandas as pd
import zarr

from neurozarr import Repo, util


def _table_group(store) -> zarr.Group:
	return zarr.open_group(store=Repo.open(store)._icechunk_repo("sub-001").readonly_session("main").store,
						   mode="r")["ses-1"]["beh"]["task-Log"]["table"]


def _roundtrip(store, df):
	repo = Repo.init(store)
	repo.create_subject("sub-001").add_visit("ses-1").add_table(df, task="Log", meta={"note": "kept"})
	repo.commit("t")
	table = Repo.open(store).subject("sub-001").visit("ses-1").tables()[0]
	pd.testing.assert_frame_equal(table.df(), df)
	assert table.meta == {"note": "kept"}
	return _table_group(store)


def test_small_table_is_inline_and_lossless(store):
	df = pd.DataFrame({
		"f": [1.5, np.nan, 1e-300], "f32": np.array([0.1, 2, 3], dtype="float32"),
		"i": pd.array([1, None, 2**62], dtype="Int64"), "b": pd.array([True, None, False], dtype="boolean"),
		"s": pd.array(["a", None, "ü"], dtype="string"),
		"c": pd.Categorical(["x", None, "y"], categories=["y", "x"], ordered=True),
		"t": pd.to_datetime(["2024-01-01 00:00", None, "2024-03-01 12:00"]).tz_localize("UTC").astype("datetime64[ns, UTC]"),
		"d": pd.to_timedelta([1, None, 3], unit="s").astype("timedelta64[ns]"),
	})
	group = _roundtrip(store, df)
	assert group.attrs["table_schema_version"] == 4 and not list(group.array_keys())


def test_inf_or_large_tables_keep_column_arrays(store, monkeypatch):
	group = _roundtrip(store, pd.DataFrame({"x": [1.0, np.inf], "y": pd.array(["a", "b"], dtype="string")}))
	assert group.attrs["table_schema_version"] == 3 and list(group.array_keys())
	monkeypatch.setattr(util, "INLINE_TABLE_BYTES", 10)
	group = _roundtrip(f"{store}-large", pd.DataFrame({"x": np.arange(100.0)}))
	assert group.attrs["table_schema_version"] == 3 and list(group.array_keys())
