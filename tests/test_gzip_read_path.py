"""Tests for the rewritten gzip read path.

Upstream measured the uncompressed length by draining the stream and seeking
back, which cost two extra full decompressions. The fork takes the length from
the gzip ISIZE trailer and counts trailing bytes by reading forwards. ISIZE is
unreliable for multi-member files and for streams of 4 GiB or more, so the
fallback matters as much as the fast path and is tested here directly.
"""

import gzip
import io
import os
import struct
import warnings

import numpy as np
import pytest

import mrcfile
from mrcfile.gzipmrcfile import GzipMrcFile


@pytest.fixture
def volume():
    rng = np.random.default_rng(20260911)
    return rng.normal(0, 1, (4, 8, 8)).astype("<f4")


def write_gz(path, data):
    plain = str(path) + ".plain"
    with mrcfile.new(plain, overwrite=True) as mrc:
        mrc.set_data(data)
    with open(plain, "rb") as fi, gzip.open(path, "wb") as fo:
        fo.write(fi.read())
    return path


def test_roundtrip_matches_uncompressed(tmp_path, volume):
    path = write_gz(tmp_path / "v.mrc.gz", volume)
    with mrcfile.open(str(path)) as mrc:
        assert isinstance(mrc, GzipMrcFile)
        np.testing.assert_array_equal(mrc.data, volume)


def test_trailer_gives_the_true_uncompressed_length(tmp_path, volume):
    path = write_gz(tmp_path / "v.mrc.gz", volume)
    plain_size = os.path.getsize(str(path) + ".plain")
    with mrcfile.open(str(path)) as mrc:
        assert mrc._uncompressed_size_from_trailer() == plain_size


def test_trailing_bytes_are_reported_with_an_exact_count(tmp_path, volume):
    """The warning text is part of the API; upstream asserts on it."""
    plain = tmp_path / "v.mrc"
    with mrcfile.new(str(plain), overwrite=True) as mrc:
        mrc.set_data(volume)
    path = tmp_path / "v.mrc.gz"
    with open(plain, "rb") as fi, gzip.open(path, "wb") as fo:
        fo.write(fi.read())
        fo.write(b"\x00" * 8)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with mrcfile.open(str(path)) as mrc:
            np.testing.assert_array_equal(mrc.data, volume)
    messages = [str(w.message) for w in caught]
    assert any("8 bytes larger than expected" in m for m in messages), messages


def test_multi_member_gzip_is_read_correctly(tmp_path, volume):
    """Concatenated members: ISIZE describes only the last one."""
    plain = tmp_path / "v.mrc"
    with mrcfile.new(str(plain), overwrite=True) as mrc:
        mrc.set_data(volume)
    raw = plain.read_bytes()
    half = len(raw) // 2

    path = tmp_path / "multi.mrc.gz"
    with open(path, "wb") as out:
        out.writelines(gzip.compress(part) for part in (raw[:half], raw[half:]))

    with mrcfile.open(str(path)) as mrc:
        np.testing.assert_array_equal(mrc.data, volume)


def test_multi_member_trailer_is_rejected(tmp_path, volume):
    """The fallback trigger, asserted directly rather than inferred.

    A deliberately corrupted ISIZE cannot be used to test this: CPython's own
    gzip reader validates the trailer and refuses the file. The realistic way
    for ISIZE to under-report is a multi-member stream, where it describes only
    the final member.
    """
    plain = tmp_path / "v.mrc"
    with mrcfile.new(str(plain), overwrite=True) as mrc:
        mrc.set_data(volume)
    raw = plain.read_bytes()
    half = len(raw) // 2

    path = tmp_path / "multi.mrc.gz"
    with open(path, "wb") as out:
        out.writelines(gzip.compress(part) for part in (raw[:half], raw[half:]))

    with mrcfile.open(str(path)) as mrc:
        # Last member's ISIZE is below the total compressed length, so the
        # value must be refused and the file measured exactly instead.
        assert mrc._uncompressed_size_from_trailer() is None
        np.testing.assert_array_equal(mrc.data, volume)


def test_truncated_data_still_raises(tmp_path, volume):
    """Removing the size cap must not silence genuine truncation."""
    plain = tmp_path / "v.mrc"
    with mrcfile.new(str(plain), overwrite=True) as mrc:
        mrc.set_data(volume)
    raw = plain.read_bytes()
    path = tmp_path / "short.mrc.gz"
    with gzip.open(path, "wb") as fo:
        fo.write(raw[: len(raw) - 64])

    with pytest.raises(ValueError, match="Expected"):
        mrcfile.open(str(path))


def test_truncated_data_warns_in_permissive_mode(tmp_path, volume):
    plain = tmp_path / "v.mrc"
    with mrcfile.new(str(plain), overwrite=True) as mrc:
        mrc.set_data(volume)
    raw = plain.read_bytes()
    path = tmp_path / "short.mrc.gz"
    with gzip.open(path, "wb") as fo:
        fo.write(raw[: len(raw) - 64])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with mrcfile.open(str(path), permissive=True) as mrc:
            assert mrc.data is None
    assert any("Expected" in str(w.message) for w in caught)


def test_write_then_read_roundtrip(tmp_path, volume):
    path = tmp_path / "written.mrc.gz"
    with GzipMrcFile(str(path), mode="w+", overwrite=True) as mrc:
        mrc.set_data(volume)
    with mrcfile.open(str(path)) as mrc:
        np.testing.assert_array_equal(mrc.data, volume)
        assert mrc.header.dmax == np.float32(volume.max())


def plain_bytes(tmp_path, data):
    plain = tmp_path / "plain.mrc"
    with mrcfile.new(str(plain), overwrite=True) as mrc:
        mrc.set_data(data)
    return plain.read_bytes()


def test_multi_member_file_whose_last_member_looks_usable(tmp_path):
    """The header and the data as two members, with data that compresses well.

    The last member's ISIZE is then above the compressed length, so it passes
    the trailer check, but it counts only the data, so it is too small to be a
    limit on the data block. The file must still be read.
    """
    data = np.zeros((64, 64, 64), dtype=np.float32)
    raw = plain_bytes(tmp_path, data)
    path = tmp_path / "multi.mrc.gz"
    path.write_bytes(gzip.compress(raw[:1024]) + gzip.compress(raw[1024:]))

    with mrcfile.open(str(path)) as mrc:
        np.testing.assert_array_equal(mrc.data, data)
    np.testing.assert_array_equal(mrcfile.read(str(path)), data)
    assert mrcfile.validate(str(path), print_file=io.StringIO())


@pytest.mark.parametrize("level", [0, 6])
def test_a_header_claiming_too_much_data_is_refused_before_reading(
    tmp_path, volume, level
):
    """Upstream's error, whether or not the trailer can be used as a limit.

    At level 0 the trailer is below the compressed length and is not used; at
    level 6 it is used but the data block does not fit within it.
    """
    raw = bytearray(plain_bytes(tmp_path, volume))
    raw[:12] = struct.pack("<3i", 40000, 40000, 40000)
    path = tmp_path / "huge.mrc.gz"
    path.write_bytes(gzip.compress(bytes(raw), compresslevel=level))

    with pytest.raises(ValueError, match="limit is"):
        mrcfile.open(str(path))
    with (
        pytest.warns(RuntimeWarning, match="limit is"),
        mrcfile.open(str(path), permissive=True) as mrc,
    ):
        assert mrc.data is None
