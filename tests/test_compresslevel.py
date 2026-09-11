"""Tests for the ``compresslevel`` option of :func:`mrcfile.new` and friends.

The level a file was written with is visible in its header, which allows exact
checks: a gzip header's XFL byte is 4 for level 1 and 2 for level 9, and a bzip2
stream starts with ``BZh`` followed by the level digit.
"""

import numpy as np
import pytest

import mrcfile
from mrcfile.bzip2mrcfile import Bzip2MrcFile
from mrcfile.gzipmrcfile import GzipMrcFile

GZIP_XFL = {1: 4, 9: 2}


@pytest.fixture
def volume():
    return np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)


@pytest.mark.parametrize("level", [1, 9])
def test_gzip_writes_the_requested_level(tmp_path, volume, level):
    path = tmp_path / "vol.mrc.gz"
    with mrcfile.new(path, volume, compression="gzip", compresslevel=level):
        pass
    assert path.read_bytes()[8] == GZIP_XFL[level]
    with mrcfile.open(path) as mrc:
        assert isinstance(mrc, GzipMrcFile)
        np.testing.assert_array_equal(mrc.data, volume)


@pytest.mark.parametrize("level", [1, 9])
def test_bzip2_writes_the_requested_level(tmp_path, volume, level):
    path = tmp_path / "vol.mrc.bz2"
    with mrcfile.new(path, volume, compression="bzip2", compresslevel=level):
        pass
    assert path.read_bytes()[:4] == f"BZh{level}".encode()
    with mrcfile.open(path) as mrc:
        assert isinstance(mrc, Bzip2MrcFile)
        np.testing.assert_array_equal(mrc.data, volume)


def test_default_level_is_still_9(tmp_path, volume):
    gz, bz = tmp_path / "vol.mrc.gz", tmp_path / "vol.mrc.bz2"
    with mrcfile.new(gz, volume, compression="gzip"):
        pass
    with mrcfile.new(bz, volume, compression="bzip2"):
        pass
    assert gz.read_bytes()[8] == GZIP_XFL[9]
    assert bz.read_bytes()[:4] == b"BZh9"


def test_level_applies_when_rewriting_in_place(tmp_path, volume):
    path = tmp_path / "vol.mrc.gz"
    with GzipMrcFile(path, mode="w+", compresslevel=1) as mrc:
        mrc.set_data(volume)
    assert path.read_bytes()[8] == GZIP_XFL[1]
    with GzipMrcFile(path, mode="r+", compresslevel=9) as mrc:
        mrc.data[0, 0, 0] = -1
    assert path.read_bytes()[8] == GZIP_XFL[9]
    with mrcfile.open(path) as mrc:
        assert mrc.data[0, 0, 0] == -1


def test_write_passes_compresslevel_through(tmp_path, volume):
    path = tmp_path / "vol.mrc.gz"
    mrcfile.write(path, volume, compresslevel=1)
    assert path.read_bytes()[8] == GZIP_XFL[1]
    np.testing.assert_array_equal(mrcfile.read(path), volume)


@pytest.mark.parametrize(
    ("compression", "level"), [("gzip", -1), ("gzip", 10), ("bzip2", 0), ("bzip2", 10)]
)
def test_out_of_range_level_leaves_an_existing_file_untouched(
    tmp_path, volume, compression, level
):
    path = tmp_path / "existing.mrc"
    path.write_bytes(b"precious")
    with pytest.raises(ValueError, match="compresslevel"):
        mrcfile.new(
            path, volume, compression=compression, compresslevel=level, overwrite=True
        )
    assert path.read_bytes() == b"precious"


def test_compresslevel_needs_compression(tmp_path, volume):
    with pytest.raises(ValueError, match="compresslevel"):
        mrcfile.new(tmp_path / "vol.mrc", volume, compresslevel=1)
    with pytest.raises(ValueError, match="compresslevel"):
        mrcfile.write(tmp_path / "vol.mrc", volume, compresslevel=1)
    assert not (tmp_path / "vol.mrc").exists()
