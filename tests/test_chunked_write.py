"""Tests for the chunked array writer used by the compressed backends."""

import gzip
import io

import numpy as np
import pytest

import mrcfile
from mrcfile.utils import write_array_in_chunks


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<i2", "<u2", "|i1"])
def test_output_is_byte_identical_to_tobytes(dtype):
    arr = np.arange(10_000, dtype=np.int64).astype(dtype).reshape(100, 100)
    stream = io.BytesIO()
    write_array_in_chunks(stream, arr, chunk_bytes=997)  # deliberately unaligned
    assert stream.getvalue() == arr.tobytes()


def test_non_contiguous_input_matches_tobytes():
    arr = np.arange(20_000, dtype=np.float32).reshape(200, 100)[:, ::3]
    assert not arr.flags.c_contiguous
    stream = io.BytesIO()
    write_array_in_chunks(stream, arr, chunk_bytes=1024)
    assert stream.getvalue() == arr.tobytes()


def test_empty_array_writes_nothing():
    stream = io.BytesIO()
    write_array_in_chunks(stream, np.empty(0, dtype=np.float32))
    assert stream.getvalue() == b""


def test_chunk_size_does_not_change_output():
    arr = np.random.default_rng(5).normal(0, 1, 50_000).astype("<f4")
    outputs = []
    for chunk in (64, 4096, 1 << 20):
        stream = io.BytesIO()
        write_array_in_chunks(stream, arr, chunk_bytes=chunk)
        outputs.append(stream.getvalue())
    assert outputs[0] == outputs[1] == outputs[2] == arr.tobytes()


def test_gzip_roundtrip_through_chunked_writer(tmp_path):
    """The compressed backends rely on this producing a valid stream."""
    data = np.random.default_rng(9).normal(0, 1, (5, 16, 16)).astype("<f4")
    path = tmp_path / "chunked.mrc.gz"
    with mrcfile.new(str(path), overwrite=True, compression="gzip") as mrc:
        mrc.set_data(data)
    with gzip.open(path, "rb") as f:
        assert len(f.read()) == 1024 + data.nbytes
    with mrcfile.open(str(path)) as mrc:
        np.testing.assert_array_equal(mrc.data, data)
