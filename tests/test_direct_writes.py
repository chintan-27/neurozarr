"""util's direct node writes must store exactly what zarr-python would: the same
zarr.json and the same chunk bytes, including padded edge chunks and skipped
all-fill-value chunks."""

import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec, ZstdCodec
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from neurozarr import util


def _objects(store: MemoryStore) -> dict[str, bytes]:
	return {k: v.to_bytes() for k, v in store._store_dict.items()}


@pytest.mark.parametrize("dtype", ["int16", "float32", "float64"])
@pytest.mark.parametrize("compressors", [
	[BloscCodec(cname="zstd", clevel=5, shuffle="shuffle", typesize=2)],
	[BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle", typesize=4)],
	[BloscCodec(cname="zstd", clevel=9, shuffle="noshuffle", typesize=8)],
	[ZstdCodec(level=3)],  # not Blosc: zarr's own pipeline encodes
])
def test_direct_array_write_matches_zarr(dtype, compressors):
	rng = np.random.default_rng(0)
	data = np.cumsum(rng.integers(-40, 40, (3, 2500)), axis=1).astype(dtype)
	data[:, 1000:1500] = 0  # one whole chunk of fill value: zarr writes nothing for it
	shape, chunks = data.shape, (3, 500)

	expected = MemoryStore()
	zgroup = zarr.open_group(expected, mode="a").create_group("rec", attributes={"a": 1})
	zgroup.create_array("data", shape=shape, dtype=dtype, chunks=chunks, compressors=compressors)[...] = data

	actual = MemoryStore()
	root = zarr.open_group(actual, mode="a")

	metadata = util.array_metadata(shape, dtype, chunks, [], compressors)

	async def write() -> None:
		group = await util.new_group_async(root._async_group, "rec", {"a": 1})
		array = await util.new_array_async(group, "data", metadata)
		encoder = util._blosc_encoder(metadata)
		for start in range(0, shape[-1], chunks[-1]):
			await util.write_block_async(array, start, data[:, start:start + chunks[-1]], encoder)
	sync(write())

	assert _objects(actual) == _objects(expected)
	assert np.array_equal(zarr.open_array(actual, path="rec/data", mode="r")[:], data)


def test_direct_write_pads_a_short_last_chunk():
	data = np.arange(2 * 1234, dtype="int16").reshape(2, 1234)
	compressors = [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle", typesize=2)]
	expected = MemoryStore()
	zarr.create_array(expected, name="x", shape=data.shape, dtype="int16", chunks=(2, 500), compressors=compressors)[...] = data
	actual = MemoryStore()
	root = zarr.open_group(actual, mode="a")
	metadata = util.array_metadata(data.shape, "int16", (2, 500), [], compressors)

	async def write() -> None:
		array = await util.new_array_async(root._async_group, "x", metadata)
		for start in range(0, 1234, 500):
			await util.write_block_async(array, start, data[:, start:start + 500], util._blosc_encoder(metadata))
	sync(write())
	assert {k: v for k, v in _objects(actual).items() if k != "zarr.json"} == \
		{k: v for k, v in _objects(expected).items() if k != "zarr.json"}
