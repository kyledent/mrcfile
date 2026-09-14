"""Tests for reading BGZF-compressed MRC files into GPU memory.

Most of these need CuPy, nvCOMP's Python package and a GPU that nvCOMP supports
(compute capability 7.0 or later), and are skipped elsewhere. The checks that
happen before any GPU work run everywhere.
"""

import gzip
import importlib

import numpy as np
import pytest

import mrcfile
from mrcfile import bgzfgpu, bgzfmrcfile
from mrcfile.bgzfmrcfile import EOF_BLOCK, bgzf_blocks


def gpu_ready():
    try:
        cupy = importlib.import_module("cupy")
        importlib.import_module("nvidia.nvcomp")
        return int(cupy.cuda.Device().compute_capability) >= 70
    except (ImportError, OSError, RuntimeError):  # no package, driver or GPU
        return False


needs_gpu = pytest.mark.skipif(
    not gpu_ready(), reason="needs CuPy, nvCOMP and a GPU of compute capability 7.0+"
)


@pytest.fixture
def volume():
    """Forty-one blocks of noise."""
    return np.random.default_rng(3).standard_normal((40, 128, 128)).astype(np.float32)


def write_bgzf(path, data, extended=None):
    with mrcfile.new(path, data, compression="bgzf", threads=4) as mrc:
        if extended is not None:
            mrc.set_extended_header(extended)
    return path


def bgzf_of(raw, path):
    """Write arbitrary bytes as BGZF, as a malformed MRC file needs."""
    blocks = bgzfmrcfile._compress_blocks(
        bgzfmrcfile._block_data(raw, memoryview(b"")), 6, 1
    )
    path.write_bytes(b"".join(blocks) + EOF_BLOCK)
    return path


def plain_file_bytes(tmp_path, data):
    with mrcfile.new(tmp_path / "plain.mrc", data, overwrite=True) as mrc:
        parts = (mrc.header, mrc.extended_header, mrc.data)
        return b"".join(b"" if part is None else part.tobytes() for part in parts)


def to_host(array):
    return importlib.import_module("cupy").asnumpy(array)


@needs_gpu
def test_read_to_gpu_gives_the_data(tmp_path, volume):
    path = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    data = bgzfgpu.read_to_gpu(path)
    assert isinstance(data, importlib.import_module("cupy").ndarray)
    assert data.dtype == volume.dtype
    assert data.shape == volume.shape
    np.testing.assert_array_equal(to_host(data), volume)


@needs_gpu
def test_unaligned_data_block(tmp_path, volume):
    """131 bytes of extended header leave the data block's start unaligned."""
    extended = np.frombuffer(bytes(range(131)), dtype="V1")
    path = write_bgzf(tmp_path / "ext.mrc.gz", volume, extended)
    np.testing.assert_array_equal(to_host(bgzfgpu.read_to_gpu(path)), volume)


@needs_gpu
def test_bad_crc_is_caught(tmp_path, volume):
    path = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    raw = bytearray(path.read_bytes())
    with open(path, "rb") as f:
        offset, size = list(bgzf_blocks(f))[5]
    raw[offset + size - 8] ^= 0xFF  # the first byte of the block's CRC32
    path.write_bytes(bytes(raw))
    with pytest.raises(gzip.BadGzipFile, match="CRC check failed for 1 block"):
        bgzfgpu.read_to_gpu(path)
    # Without the check the damaged CRC32 goes unnoticed; the data are intact
    data = bgzfgpu.read_to_gpu(path, verify=False)
    np.testing.assert_array_equal(to_host(data), volume)


@needs_gpu
def test_trailing_bytes_warn(tmp_path, volume):
    raw = plain_file_bytes(tmp_path, volume)
    path = bgzf_of(raw + b"extra", tmp_path / "long.mrc.gz")
    with pytest.warns(RuntimeWarning, match="5 bytes larger than expected"):
        data = bgzfgpu.read_to_gpu(path)
    np.testing.assert_array_equal(to_host(data), volume)


def test_short_file_is_refused_before_any_gpu_work(tmp_path, volume):
    raw = plain_file_bytes(tmp_path, volume)
    path = bgzf_of(raw[:-100], tmp_path / "short.mrc.gz")
    with pytest.raises(ValueError, match="limit is"):
        bgzfgpu.read_to_gpu(path)


def test_blocks_with_other_gzip_fields_are_refused(tmp_path, volume):
    """A file name in a block's header moves its data, which nvCOMP is not told."""
    path = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    raw = path.read_bytes()
    with open(path, "rb") as f:
        _, size = next(bgzf_blocks(f))
    block = bytearray(raw[:size])
    block[3] |= 0x08  # FNAME
    block[16:18] = (size + 2 - 1).to_bytes(2, "little")  # BSIZE, the block size - 1
    path.write_bytes(bytes(block[:18]) + b"x\0" + bytes(block[18:]) + raw[size:])
    with pytest.raises(ValueError, match="gzip header fields"):
        bgzfgpu.read_to_gpu(path)


@needs_gpu
def test_damaged_compressed_data_are_caught(tmp_path, volume):
    """Damage inside a block's DEFLATE data, rather than in its trailer."""
    path = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    raw = bytearray(path.read_bytes())
    with open(path, "rb") as f:
        offset, size = list(bgzf_blocks(f))[5]
    raw[offset + size // 2] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises((gzip.BadGzipFile, RuntimeError, ValueError)):
        bgzfgpu.read_to_gpu(path)


def test_plain_gzip_is_refused(tmp_path, volume):
    path = tmp_path / "vol.mrc.gz"
    with mrcfile.new(path, volume, compression="gzip"):
        pass
    with pytest.raises(ValueError, match="is not a BGZF file"):
        bgzfgpu.read_to_gpu(path)


def test_big_endian_data_are_refused(tmp_path):
    data = np.arange(12, dtype=">f4").reshape(3, 4)
    path = tmp_path / "be.mrc.gz"
    with mrcfile.new(path, data, compression="bgzf"):
        pass
    with pytest.raises(ValueError, match="byte order"):
        bgzfgpu.read_to_gpu(path)


def test_missing_gpu_packages(tmp_path, volume, monkeypatch):
    path = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    real_import = importlib.import_module

    def without_gpu(module, *args):
        if module in ("cupy", "nvidia.nvcomp"):
            raise ImportError(module)
        return real_import(module, *args)

    monkeypatch.setattr(bgzfgpu.importlib, "import_module", without_gpu)
    with pytest.raises(ImportError, match=r"pip install mrcfile\[gpu\]"):
        bgzfgpu.read_to_gpu(path)
