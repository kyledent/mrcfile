"""Tests for BGZF-compressed MRC files: gzip written as independent blocks."""

import gzip
import io
import zlib

import numpy as np
import pytest

import mrcfile
from mrcfile import bgzfmrcfile
from mrcfile.bgzfmrcfile import (
    BLOCK_DATA_SIZE,
    EOF_BLOCK,
    MAX_BLOCK_SIZE,
    BgzfMrcFile,
    bgzf_blocks,
    is_bgzf,
)
from mrcfile.gzipmrcfile import GzipMrcFile


@pytest.fixture
def volume():
    """About 320 KiB of noise: several blocks, compressing as cryo-EM data does."""
    rng = np.random.default_rng(20260911)
    return rng.standard_normal((5, 128, 128)).astype(np.float32)


def file_bytes(mrc):
    """The uncompressed bytes an open MRC object stands for.

    Taken from the object itself, not from a second file written to compare
    against: a new file's first label records when it was created, so two files
    written a second apart differ.
    """
    parts = (mrc.header, mrc.extended_header, mrc.data)
    return b"".join(b"" if part is None else part.tobytes() for part in parts)


def write_bgzf(path, data=None, **kwargs):
    """Write a BGZF file; return its path and the bytes it must decompress to."""
    with mrcfile.new(path, data, compression="bgzf", **kwargs) as mrc:
        expected = file_bytes(mrc)
    return path, expected


def blocks_of(path):
    with open(path, "rb") as f:
        return list(bgzf_blocks(f))


def test_new_then_open_round_trip(tmp_path, volume):
    path = tmp_path / "vol.mrc.gz"
    with mrcfile.new(path, volume, compression="bgzf") as mrc:
        assert isinstance(mrc, BgzfMrcFile)
    with mrcfile.open(path) as mrc:
        assert type(mrc) is BgzfMrcFile
        assert repr(mrc) == f"BgzfMrcFile('{path}', mode='r')"
        np.testing.assert_array_equal(mrc.data, volume)


def test_other_gzip_readers_see_the_plain_file(tmp_path, volume):
    path, expected = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    assert gzip.decompress(path.read_bytes()) == expected


def test_blocks_are_independent_and_full(tmp_path, volume):
    path, expected = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    raw = path.read_bytes()
    blocks = blocks_of(path)

    assert blocks[0][0] == 0
    for (offset, size), (next_offset, _) in zip(blocks, blocks[1:]):
        assert offset + size == next_offset
    last_offset, last_size = blocks[-1]
    assert last_offset + last_size == len(raw)
    assert raw[last_offset:] == EOF_BLOCK

    # Each block decompresses on its own, as a parallel or GPU reader needs
    pieces = [zlib.decompress(raw[o : o + s], 31) for o, s in blocks[:-1]]
    assert all(len(piece) == BLOCK_DATA_SIZE for piece in pieces[:-1])
    assert 0 < len(pieces[-1]) <= BLOCK_DATA_SIZE
    assert b"".join(pieces) == expected


@pytest.mark.parametrize("level", [0, 1, 9])
def test_incompressible_blocks_still_fit(tmp_path, level):
    """Random bytes are the worst case: DEFLATE cannot shrink them."""
    rng = np.random.default_rng(level)
    data = rng.integers(-128, 128, size=(3, 256, 256), dtype=np.int8)
    path, _ = write_bgzf(tmp_path / "noise.mrc.gz", data, compresslevel=level)
    assert max(size for _, size in blocks_of(path)) <= MAX_BLOCK_SIZE
    with mrcfile.open(path) as mrc:
        np.testing.assert_array_equal(mrc.data, data)


def test_extended_header_longer_than_a_block(tmp_path, volume):
    extended = np.frombuffer(bytes(range(256)) * 400, dtype="V1")
    path = tmp_path / "ext.mrc.gz"
    with mrcfile.new(path, volume, compression="bgzf") as mrc:
        mrc.set_extended_header(extended)
    with mrcfile.open(path) as mrc:
        assert mrc.extended_header.tobytes() == extended.tobytes()
        np.testing.assert_array_equal(mrc.data, volume)


def test_header_only_file(tmp_path):
    path, expected = write_bgzf(tmp_path / "empty.mrc.gz")
    assert len(blocks_of(path)) == 2  # the header, then the end-of-file block
    assert gzip.decompress(path.read_bytes()) == expected


def test_rewriting_keeps_the_blocks(tmp_path, volume):
    path, _ = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    changed = volume * 2
    with mrcfile.open(path, mode="r+") as mrc:
        assert type(mrc) is BgzfMrcFile
        mrc.set_data(changed)
    assert is_bgzf(path.read_bytes())
    assert path.read_bytes().endswith(EOF_BLOCK)
    with mrcfile.open(path) as mrc:
        np.testing.assert_array_equal(mrc.data, changed)


def test_flush_leaves_a_complete_file(tmp_path, volume):
    path = tmp_path / "vol.mrc.gz"
    with mrcfile.new(path, volume, compression="bgzf") as mrc:
        mrc.flush()
        assert gzip.decompress(path.read_bytes()) == file_bytes(mrc)
        mrc.data[0, 0, 0] = 99
    with mrcfile.open(path) as mrc:
        assert mrc.data[0, 0, 0] == 99


def test_reader_does_not_trust_the_last_size_field(tmp_path, volume):
    """The last block is the end-of-file block, whose size field reads 0."""
    path, _ = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    with mrcfile.open(path) as mrc:
        assert mrc._uncompressed_size_from_trailer() is None
        np.testing.assert_array_equal(mrc.data, volume)


def test_compresslevel_reaches_the_compressor(tmp_path, volume, monkeypatch):
    levels = []
    real_compressobj = zlib.compressobj

    def spy(level, *args):
        levels.append(level)
        return real_compressobj(level, *args)

    monkeypatch.setattr(bgzfmrcfile.zlib, "compressobj", spy)
    for given, used in ((None, 9), (3, 3)):
        levels.clear()
        write_bgzf(tmp_path / f"{given}.mrc.gz", volume, compresslevel=given)
        assert levels
        assert set(levels) == {used}


def test_invalid_level_leaves_an_existing_file_alone(tmp_path):
    path = tmp_path / "keep.mrc.gz"
    path.write_bytes(b"precious")
    with pytest.raises(ValueError, match="compresslevel"):
        mrcfile.new(path, compression="bgzf", compresslevel=10, overwrite=True)
    assert path.read_bytes() == b"precious"


def test_plain_gzip_is_not_mistaken_for_bgzf(tmp_path, volume):
    path = tmp_path / "vol.mrc.gz"
    with mrcfile.new(path, volume, compression="gzip"):
        pass
    assert not is_bgzf(path.read_bytes())
    with mrcfile.open(path) as mrc:
        assert type(mrc) is GzipMrcFile


def test_is_bgzf():
    assert is_bgzf(EOF_BLOCK)
    assert not is_bgzf(gzip.compress(b"data"))
    assert not is_bgzf(EOF_BLOCK[:17])
    assert not is_bgzf(EOF_BLOCK[:12] + b"XY" + EOF_BLOCK[14:])


def test_bgzf_blocks_refuses_other_data():
    with pytest.raises(ValueError, match="No BGZF block header at byte 0"):
        list(bgzf_blocks(io.BytesIO(gzip.compress(b"data"))))


def test_bgzf_blocks_detects_truncation(tmp_path, volume):
    path, _ = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    cut = path.read_bytes()[: -len(EOF_BLOCK) - 10]
    with pytest.raises(ValueError, match="runs past the end"):
        list(bgzf_blocks(io.BytesIO(cut)))


def test_validate(tmp_path, volume):
    path, _ = write_bgzf(tmp_path / "vol.mrc.gz", volume)
    assert mrcfile.validate(str(path), print_file=io.StringIO())
