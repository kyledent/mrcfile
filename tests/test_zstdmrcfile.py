"""Tests for Zstandard-compressed MRC files.

Zstandard support needs Python 3.14, or the ``backports.zstd`` package on earlier
versions, so most of these tests are skipped without it. The tests of what happens
when support is missing run everywhere, by hiding the module.
"""

import struct

import numpy as np
import pytest

import mrcfile
from mrcfile import zstdmrcfile
from mrcfile.zstdmrcfile import ZstdMrcFile

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

needs_zstd = pytest.mark.skipif(
    zstdmrcfile._zstd is None, reason="needs Python 3.14+ or backports.zstd"
)


@pytest.fixture
def volume():
    return np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)


@needs_zstd
def test_new_then_open_round_trip(tmp_path, volume):
    path = tmp_path / "vol.mrc.zst"
    with mrcfile.new(path, volume, compression="zstd") as mrc:
        assert isinstance(mrc, ZstdMrcFile)
    assert path.read_bytes()[:4] == ZSTD_MAGIC
    with mrcfile.open(path) as mrc:
        assert isinstance(mrc, ZstdMrcFile)
        assert repr(mrc) == f"ZstdMrcFile('{path}', mode='r')"
        np.testing.assert_array_equal(mrc.data, volume)


@needs_zstd
def test_write_and_read_choose_zstd_by_extension(tmp_path, volume):
    path = tmp_path / "vol.mrc.zst"
    mrcfile.write(path, volume, voxel_size=1.5)
    assert path.read_bytes()[:4] == ZSTD_MAGIC
    np.testing.assert_array_equal(mrcfile.read(path), volume)
    with mrcfile.open(path) as mrc:
        assert mrc.voxel_size.x == pytest.approx(1.5)


@needs_zstd
@pytest.mark.parametrize("level", [None, 1, 19])
def test_compresslevel_reaches_the_compressor(tmp_path, volume, monkeypatch, level):
    seen = []
    real_zstd_file = zstdmrcfile._zstd.ZstdFile

    def spy(*args, **kwargs):
        if kwargs.get("mode") == "wb":
            seen.append(kwargs.get("level"))
        return real_zstd_file(*args, **kwargs)

    monkeypatch.setattr(zstdmrcfile._zstd, "ZstdFile", spy)
    path = tmp_path / "vol.mrc.zst"
    with mrcfile.new(path, volume, compression="zstd", compresslevel=level):
        pass
    assert seen
    assert set(seen) == {level}


@needs_zstd
@pytest.mark.parametrize("level", [23, -200000])
def test_out_of_range_level_leaves_an_existing_file_untouched(tmp_path, volume, level):
    path = tmp_path / "existing.mrc"
    path.write_bytes(b"precious")
    with pytest.raises(ValueError, match="compresslevel"):
        mrcfile.new(
            path, volume, compression="zstd", compresslevel=level, overwrite=True
        )
    assert path.read_bytes() == b"precious"


@needs_zstd
@pytest.mark.parametrize("level", [3.0, 3.5, True])
def test_a_level_that_is_not_an_integer_leaves_an_existing_file_untouched(
    tmp_path, volume, level
):
    path = tmp_path / "existing.mrc"
    path.write_bytes(b"precious")
    with pytest.raises(TypeError, match="compresslevel must be an integer"):
        mrcfile.new(
            path, volume, compression="zstd", compresslevel=level, overwrite=True
        )
    assert path.read_bytes() == b"precious"


@needs_zstd
def test_concatenated_frames_are_read_as_one_stream(tmp_path, volume):
    path = tmp_path / "vol.mrc.zst"
    mrcfile.write(path, volume)
    raw = zstdmrcfile._zstd.decompress(path.read_bytes())
    half = len(raw) // 2
    compress = zstdmrcfile._zstd.compress
    path.write_bytes(compress(raw[:half]) + compress(raw[half:]))
    with mrcfile.open(path) as mrc:
        np.testing.assert_array_equal(mrc.data, volume)


@needs_zstd
def test_trailing_bytes_are_reported(tmp_path, volume):
    path = tmp_path / "vol.mrc.zst"
    mrcfile.write(path, volume)
    raw = zstdmrcfile._zstd.decompress(path.read_bytes())
    path.write_bytes(zstdmrcfile._zstd.compress(raw + b"extra"))
    with pytest.warns(RuntimeWarning, match="5 bytes larger than expected"):
        mrcfile.open(path).close()


@needs_zstd
def test_a_header_claiming_too_much_data_is_refused_before_reading(tmp_path, volume):
    path = tmp_path / "vol.mrc.zst"
    mrcfile.write(path, volume)
    raw = bytearray(zstdmrcfile._zstd.decompress(path.read_bytes()))
    raw[:12] = struct.pack("<3i", 40000, 40000, 40000)
    path.write_bytes(zstdmrcfile._zstd.compress(bytes(raw)))
    with pytest.raises(ValueError, match="limit is"):
        mrcfile.open(path)
    with (
        pytest.warns(RuntimeWarning, match="limit is"),
        mrcfile.open(path, permissive=True) as mrc,
    ):
        assert mrc.data is None


@needs_zstd
def test_rewriting_in_place(tmp_path, volume):
    path = tmp_path / "vol.mrc.zst"
    mrcfile.write(path, volume)
    with mrcfile.open(path, mode="r+") as mrc:
        mrc.data[0, 0, 0] = -1
    with mrcfile.open(path) as mrc:
        assert mrc.data[0, 0, 0] == -1


@needs_zstd
def test_header_only(tmp_path, volume):
    path = tmp_path / "vol.mrc.zst"
    mrcfile.write(path, volume)
    with mrcfile.open(path, header_only=True) as mrc:
        assert mrc.data is None
        assert mrc.header.nx == 8


def test_new_without_zstd_says_what_is_needed(tmp_path, volume, monkeypatch):
    monkeypatch.setattr(zstdmrcfile, "_zstd", None)
    with pytest.raises(ImportError, match=r"backports\.zstd"):
        mrcfile.new(tmp_path / "vol.mrc.zst", volume, compression="zstd")
    assert not (tmp_path / "vol.mrc.zst").exists()


def test_open_without_zstd_says_what_is_needed(tmp_path, monkeypatch):
    path = tmp_path / "vol.mrc.zst"
    path.write_bytes(ZSTD_MAGIC + bytes(300))
    monkeypatch.setattr(zstdmrcfile, "_zstd", None)
    with pytest.raises(ImportError, match=r"backports\.zstd"):
        mrcfile.open(path)
