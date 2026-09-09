"""Table column packing: columns sharing a physical dtype and nullability are
stored in one shared 2D array instead of one array per column (util.create_table),
for far fewer zarr arrays on tables with many small columns. Decoding stays keyed
off each column's own schema entry, so this is purely a storage-layout change."""

import numpy as np
import pandas as pd
import zarr

from neurozarr import Repo


def test_columns_sharing_dtype_and_nullability_are_packed_into_one_array(store):
	df = pd.DataFrame({
		"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0], "d": [10.0, 11.0, 12.0],
		"name": ["x", "yy", "zzz"], "unit": ["mV", "uV", "V"], "kind": ["dbs", "eeg", "seeg"],
	})
	repo = Repo.create(store)
	visit = repo.create_subject("sub-001").add_visit("ses-1")
	visit.add_behavioral_table(df, task="Log")
	repo.save("packed")

	table = Repo.open(store).subject("sub-001").visit("ses-1").tables()[0]
	result = table.df()
	pd.testing.assert_frame_equal(result, df, check_dtype=False)

	group = zarr.open_group(store=Repo.open(store)._icechunk_repo("sub-001")
							 .readonly_session("main").store, mode="r")["ses-1"]["beh"]["task-Log"]["table"]
	assert len(list(group.array_keys())) < len(df.columns), \
		"packing should produce fewer arrays than columns"


def test_lone_dtype_columns_fall_back_to_unpacked(store):
	df = pd.DataFrame({"i": [1, 2], "f": [1.5, 2.5], "b": [True, False]})
	repo = Repo.create(store)
	visit = repo.create_subject("sub-001").add_visit("ses-1")
	visit.add_behavioral_table(df, task="Log")
	repo.save("unpacked")

	table = Repo.open(store).subject("sub-001").visit("ses-1").tables()[0]
	pd.testing.assert_frame_equal(table.df(), df)
	assert all("id" in spec for spec in table._group[table.name].attrs.asdict()["schema"])


def test_legacy_table_schema_version_2_still_reads(store):
	"""Regression: a table written before packing existed (one array per
	column, table_schema_version 2) must still read correctly."""
	repo = Repo.create(store)
	repo.create_subject("sub-001").add_visit("ses-1")
	writer = repo._writer_for("sub-001")
	group = writer.root.require_group("ses-1").require_group("beh").require_group("task-Legacy")
	table = group.create_group("table", overwrite=True)
	table.create_array("c000000", data=np.array([1.5, 2.5, np.nan]))
	table.create_array("c000000__mask", data=np.array([False, False, True]))
	table.create_array("c000001", data=np.array(["a", "b", "c"]))
	table.attrs.put({
		"_neurozarr_item_type": "table",
		"table_schema_version": 2,
		"columns": ["score", "label"],
		"schema": [
			{"id": "c000000", "name": "score", "dtype": "float64", "encoding": "float", "nullable": True},
			{"id": "c000001", "name": "label", "dtype": "string", "encoding": "string", "nullable": False},
		],
	})
	repo.save("legacy table")

	result = Repo.open(store).subject("sub-001").visit("ses-1").tables()[0].df()
	assert result["score"].tolist()[:2] == [1.5, 2.5]
	assert pd.isna(result["score"].iloc[2])
	assert result["label"].tolist() == ["a", "b", "c"]
