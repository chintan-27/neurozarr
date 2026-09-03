import tempfile
from pathlib import Path

import pandas as pd

from bidszarr.items import Attrs, Table
from bidszarr.readers import ManifestReader

tmp = Path(tempfile.mkdtemp())

tsv_path = tmp / "log.tsv"
pd.DataFrame({"onset": [0, 1], "label": ["a", "b"]}).to_csv(tsv_path, sep="\t", index=False)

unsupported_path = tmp / "scan.dcm"
unsupported_path.write_text("not a real dicom")

# default column names, plus an extra "task" column folded into entities
df = pd.DataFrame([
	{"sub": "sub-001", "ses": "ses-1", "datatype": "beh", "task": "Log", "path": str(tsv_path)},
])
items = list(ManifestReader(df).read())
assert len(items) == 1 and isinstance(items[0], Table)
assert items[0].entities.sub == "sub-001" and items[0].entities.extra == {"task": "Log"}
assert list(items[0].df.columns) == ["onset", "label"]

# configurable column mapping (non-default names)
df2 = pd.DataFrame([
	{"subject": "sub-002", "filepath": str(tsv_path)},
])
items2 = list(ManifestReader(df2, sub_col="subject", path_col="filepath").read())
assert isinstance(items2[0], Table) and items2[0].entities.sub == "sub-002"

# unsupported extension -> graceful Attrs fallback, no crash
df3 = pd.DataFrame([{"sub": "sub-003", "path": str(unsupported_path)}])
items3 = list(ManifestReader(df3).read())
assert isinstance(items3[0], Attrs) and items3[0].attrs["unread_file"] == str(unsupported_path)

# meta column as a JSON string
df4 = pd.DataFrame([{"sub": "sub-004", "path": str(tsv_path), "meta": '{"device": "X"}'}])
items4 = list(ManifestReader(df4).read())
assert items4[0].meta == {"device": "X"}

# row_reader escape hatch bypasses all interpretation
sentinel = Attrs(("sub-005",), {"custom": True})
df5 = pd.DataFrame([{"anything": "goes"}])
items5 = list(ManifestReader(df5, row_reader=lambda row: sentinel).read())
assert items5 == [sentinel]

print("test_manifest_reader: all assertions passed")
