import json
import math
from pathlib import Path

import pandas as pd
import zarr


def chunkShape(shape: tuple, itemsize: int, targetBytes: int = 30 * 1024 * 1024) -> tuple:
	rowBytes = itemsize
	for dim in shape[:-1]:
		rowBytes *= dim
	chunkLen = min(shape[-1], max(1, targetBytes // rowBytes))
	return shape[:-1] + (chunkLen,)

def setAttrs(node, attrs: dict):
	node.attrs.put({**node.attrs.asdict(), **attrs})

def createTable(group: zarr.Group, name: str, df: pd.DataFrame, extraAttrs: dict = None):
	table = group.create_array(name, data=df.astype(str).to_numpy().astype(str))
	setAttrs(table, {"columns": list(df.columns), **(extraAttrs or {})})

def loadJson(path: Path) -> dict:
	return json.loads(path.read_text()) if path.exists() else {}

def cleanNan(metaData: dict) -> dict:
	clean = {}
	for key, value in metaData.items():
		if isinstance(value, dict):
			clean[key] = cleanNan(value)
		elif isinstance(value, float) and math.isnan(value):
			clean[key] = None
		else:
			clean[key] = value
	return clean
