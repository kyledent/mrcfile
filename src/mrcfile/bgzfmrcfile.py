# Copyright (c) 2026, Diamond Light Source
# This software is distributed under a BSD licence. See LICENSE.txt.
"""
bgzfmrcfile
-----------

Module which exports the :class:`BgzfMrcFile` class.

BGZF is gzip written as a series of independent blocks. Each block holds at most
64 KiB of data and records its own compressed size in a gzip extra field. The
format comes from the SAM/BAM specification, where it gives BAM files random
access. Any gzip reader decompresses a BGZF file as ordinary gzip. A reader that
knows the format can also find every block without decompressing anything, and
so decompress the blocks in parallel, on CPU threads or on a GPU. Because the
blocks are independent, a writer can compress them in parallel too.

Classes:
    :class:`BgzfMrcFile`: An object which represents a BGZF-compressed MRC file.

Functions:
    :func:`is_bgzf`: Test whether some bytes begin a BGZF block.
    :func:`bgzf_blocks`: Find the blocks of a BGZF file without decompressing it.

"""

from __future__ import annotations

import builtins
import gzip
import os
import struct
import warnings
import zlib
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import BinaryIO, Literal, cast

import numpy as np

from . import utils
from .gzipmrcfile import GzipMrcFile

#: Uncompressed bytes per block. A compressed block may be at most 64 KiB, and
#: this much input always fits, even when DEFLATE cannot shrink it.
BLOCK_DATA_SIZE = 0xFF00

#: The largest a compressed block may be, in bytes.
MAX_BLOCK_SIZE = 0x10000

#: The empty block that ends a BGZF file, byte for byte as the SAM/BAM
#: specification gives it. Readers use it to tell a complete file from a
#: truncated one.
EOF_BLOCK = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")

# Block header: the gzip fields ID1, ID2, CM, FLG, MTIME, XFL and OS, then XLEN
# and a single extra subfield, "BC", holding the block's total size minus one.
_HEADER = struct.Struct("<4BI2BH2BHH")
# Block trailer: the CRC32 and length of the uncompressed data.
_TRAILER = struct.Struct("<2I")


def is_bgzf(start: bytes) -> bool:
    """Return :data:`True` if ``start`` begins with a BGZF block header.

    This is the test htslib applies: a gzip header with the FEXTRA flag set,
    whose extra field is a single ``BC`` subfield.

    Args:
        start: The first bytes of a file. At least 18 are needed.

    Returns:
        :data:`True` if the bytes begin a BGZF block.
    """
    return (
        len(start) >= _HEADER.size
        and start[:3] == b"\x1f\x8b\x08"
        and (start[3] & 0x04) != 0
        and start[10:16] == b"\x06\x00BC\x02\x00"
    )


def bgzf_blocks(stream: BinaryIO) -> Iterator[tuple[int, int]]:
    """Find the blocks of a BGZF file without decompressing it.

    Every block header records the block's compressed size, so the blocks can
    be found by reading 18 bytes per block. This is what lets a reader
    decompress the blocks independently, and in parallel.

    Args:
        stream: A seekable binary file object, positioned at the start of a
            block.

    Yields:
        ``(offset, size)`` for each block in turn: its position in the file and
        its compressed size in bytes. The end-of-file block is included.

    Raises:
        :exc:`ValueError`: If the stream holds anything other than whole BGZF
            blocks.
    """
    offset = stream.tell()
    end = stream.seek(0, os.SEEK_END)
    while offset < end:
        stream.seek(offset)
        header = stream.read(_HEADER.size)
        if not is_bgzf(header):
            raise ValueError(f"No BGZF block header at byte {offset}")
        size = int(_HEADER.unpack(header)[-1]) + 1
        if offset + size > end:
            raise ValueError(f"The BGZF block at byte {offset} runs past the end")
        yield offset, size
        offset += size


def _block_data(prefix: bytes, data: memoryview) -> Iterator[bytes | memoryview]:
    """Split ``prefix`` followed by ``data`` into pieces of one block each.

    Only the piece that spans the end of ``prefix`` is copied. The others are
    either slices of ``prefix``, which is small, or views of ``data``.
    """
    split = len(prefix)
    total = split + len(data)
    for start in range(0, total, BLOCK_DATA_SIZE):
        end = min(start + BLOCK_DATA_SIZE, total)
        if end <= split:
            yield prefix[start:end]
        elif start >= split:
            yield data[start - split : end - split]
        else:
            yield prefix[start:] + data[: end - split].tobytes()


def _compress_block(data: bytes | memoryview, level: int) -> bytes:
    """Compress at most :data:`BLOCK_DATA_SIZE` bytes as a single BGZF block."""
    compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    deflated = compressor.compress(data) + compressor.flush()
    size = _HEADER.size + len(deflated) + _TRAILER.size
    header = _HEADER.pack(0x1F, 0x8B, 8, 0x04, 0, 0, 0xFF, 6, 0x42, 0x43, 2, size - 1)
    return header + deflated + _TRAILER.pack(zlib.crc32(data), len(data))


def _compress_blocks(
    pieces: Iterable[bytes | memoryview], level: int, threads: int
) -> Iterator[bytes]:
    """Compress each piece as one BGZF block, and yield the blocks in order.

    With more than one thread the pieces are compressed in parallel, because
    zlib releases the GIL while it compresses. At most four blocks per thread
    are in flight at once, so memory use stays small however large the file.
    The blocks are the same whatever the number of threads.
    """
    if threads == 1:
        for piece in pieces:
            yield _compress_block(piece, level)
        return
    with ThreadPoolExecutor(max_workers=threads) as pool:
        pending: deque[Future[bytes]] = deque()
        for piece in pieces:
            pending.append(pool.submit(_compress_block, piece, level))
            if len(pending) >= 4 * threads:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


# Blocks per task when decompressing on several threads, so that the cost of
# handing out work, and of switching between threads, is shared by several blocks
_BLOCKS_PER_TASK = 8


def _inflate_blocks(batch: list[tuple[bytes, int, int, memoryview, int]]) -> None:
    """Decompress a few blocks, each into its place in the output.

    Each entry holds a block's body (the DEFLATE data, then the CRC32 and length
    of the uncompressed data), that CRC32 and length, the slice of the output the
    block fills, and how many of its leading decompressed bytes to skip.

    Raises:
        :exc:`gzip.BadGzipFile`: If a block's CRC or length does not match.
    """
    for body, crc, isize, target, skip in batch:
        data = zlib.decompress(memoryview(body)[: -_TRAILER.size], -zlib.MAX_WBITS)
        if len(data) != isize:
            raise gzip.BadGzipFile("Incorrect length of data produced")
        if zlib.crc32(data) != crc:
            raise gzip.BadGzipFile("CRC check failed")
        target[:] = memoryview(data)[skip : skip + len(target)]


def _read_blocks_into(
    path: str, start: int, out: np.ndarray, threads: int
) -> tuple[int, int] | None:
    """Decompress a BGZF file, from uncompressed offset ``start``, into ``out``.

    The blocks are read in order and decompressed on ``threads`` threads, each
    straight into its place in ``out``, a flat array of bytes. Blocks are handed
    out eight at a time, and at most two such tasks per thread are in flight at
    once, so memory use stays small however large the file.

    Returns:
        ``(filled, total)``: how many bytes of ``out`` were filled, which is
        fewer if the file is too short, and the file's total uncompressed
        length. :data:`None` if the file is not a clean run of BGZF blocks, so
        that the caller can read it some other way.

    Raises:
        :exc:`gzip.BadGzipFile`: If a block's CRC or length does not match its
            trailer.
        :exc:`zlib.error`: If a block's compressed data is corrupt.
    """
    stop = start + len(out)
    # NumPy's stubs before 2.1 do not type ndarray as a buffer for Python 3.9
    view = memoryview(out)  # type: ignore[arg-type]
    total = 0
    with open(path, "rb") as f, ThreadPoolExecutor(max_workers=threads) as pool:
        end = f.seek(0, os.SEEK_END)
        f.seek(0)
        pending: deque[Future[None]] = deque()
        batch: list[tuple[bytes, int, int, memoryview, int]] = []
        offset = 0
        while offset < end:
            header = f.read(_HEADER.size)
            if not is_bgzf(header):
                return None
            size = int(_HEADER.unpack(header)[-1]) + 1
            if size < _HEADER.size + _TRAILER.size or offset + size > end:
                return None
            body = f.read(size - _HEADER.size)
            crc, isize = _TRAILER.unpack(body[-_TRAILER.size :])
            low, high = max(total, start), min(total + isize, stop)
            if low < high:
                target = view[low - start : high - start]
                batch.append((body, crc, isize, target, low - total))
                if len(batch) == _BLOCKS_PER_TASK:
                    pending.append(pool.submit(_inflate_blocks, batch))
                    batch = []
                    if len(pending) >= 2 * threads:
                        pending.popleft().result()
            offset += size
            total += isize
        if batch:
            pending.append(pool.submit(_inflate_blocks, batch))
        for future in pending:
            future.result()
    return max(0, min(stop, total) - start), total


class BgzfMrcFile(GzipMrcFile):
    """:class:`~mrcfile.gzipmrcfile.GzipMrcFile` subclass for BGZF files.

    Writing splits the file into blocks of :data:`BLOCK_DATA_SIZE` bytes,
    compresses each one independently, and ends the file with
    :data:`EOF_BLOCK`. Reading works as for any gzip file, since a BGZF file is
    also a valid gzip file. ``compresslevel`` works as it does for
    :class:`~mrcfile.gzipmrcfile.GzipMrcFile`, with the same default of 9.
    ``threads`` sets how many threads compress the blocks when the file is
    written, and decompress them when the data is read.

    """

    def __init__(  # noqa: PLR0913
        self,
        name: str | os.PathLike[str],
        mode: Literal["r", "r+", "w+"] = "r",
        *,
        overwrite: bool = False,
        permissive: bool = False,
        header_only: bool = False,
        compresslevel: int | None = None,
        threads: int | None = None,
    ) -> None:
        """Initialise a new :class:`BgzfMrcFile` object.

        Takes the same arguments as :class:`~mrcfile.gzipmrcfile.GzipMrcFile`,
        plus ``threads``: the number of threads that compress the blocks when
        the file is written, and decompress them when the data is read. The
        default, :data:`None`, uses one. The file, and the data read from it,
        are the same whatever the number of threads.

        Raises:
            :exc:`TypeError`: If ``threads`` or ``compresslevel`` is not an
                integer.
            :exc:`ValueError`: If ``threads`` is less than 1, or
                ``compresslevel`` is not between 0 and 9. All of these are
                checked before the file is opened, so an invalid value never
                truncates an existing file.
        """
        if threads is not None:
            threads = utils.check_int_argument(threads, "threads", 1)
        self._threads = 1 if threads is None else threads
        # Set by a parallel read: the bytes after the data block, None if unknown
        self._trailing_bytes: int | None = None
        super().__init__(
            name,
            mode,
            overwrite=overwrite,
            permissive=permissive,
            header_only=header_only,
            compresslevel=compresslevel,
        )

    def __repr__(self) -> str:
        """Return a string representation of the BgzfMrcFile object."""
        return f"BgzfMrcFile('{self._fileobj.name}', mode='{self._mode}')"

    def _uncompressed_size_from_trailer(self) -> int | None:
        """Return the uncompressed length: the sum of every block's ISIZE.

        In a BGZF file the last gzip member is the empty end-of-file block, so
        the gzip trailer says nothing about the data. Each block's own ISIZE is
        exact, because no block holds more than 64 KiB, and the blocks can be
        found from their headers, so the total costs two small reads a block.
        :data:`None` if the file is not a clean run of BGZF blocks, or cannot be
        reopened by name.
        """
        name = getattr(self._fileobj, "name", None)
        if not isinstance(name, str):
            return None
        total = 0
        try:
            with builtins.open(name, "rb") as raw:
                for offset, size in bgzf_blocks(raw):
                    raw.seek(offset + size - 4)
                    total += int.from_bytes(raw.read(4), "little")
        except (OSError, ValueError):
            return None
        return total

    def _read_data(self) -> None:
        """Read the data block, decompressing its blocks on several threads.

        With one thread this is the single forward pass of
        :class:`~mrcfile.gzipmrcfile.GzipMrcFile`. With more, the blocks that
        hold the data block are read in order from a separate file handle and
        decompressed in parallel, straight into the new array, and each block's
        CRC and length are checked against its trailer. The data block is
        limited to the length the blocks add up to, so a header that claims
        more data than the file holds is refused before anything is allocated.
        The checks on the result, and the errors and warnings for a data block
        that is too short or too long, are the same as with one thread. A file
        that is not a clean run of BGZF blocks is read on one thread.
        """
        if self._threads == 1:
            super()._read_data()
            return
        total = self._uncompressed_size_from_trailer()
        if self.header is None or total is None:
            super()._read_data()
            return
        header_size = self.header.nbytes + int(self.header.nsymbt)
        self._trailing_bytes = None
        self._read_data_from_stream(max_bytes=total - header_size)
        if self.data is not None:
            extra = self._trailing_bytes
            if extra is None:  # the file was read on one thread after all
                extra = self._count_remaining_bytes()
            if extra > 0:
                warnings.warn(
                    f"MRC file is {extra} bytes larger than expected", RuntimeWarning
                )

    def _read_array_from_stream(
        self, shape: tuple, dtype: np.dtype
    ) -> tuple[np.ndarray, int]:
        """Read the data array, decompressing the blocks in parallel if asked to."""
        name = getattr(self._fileobj, "name", None)
        if self._threads == 1 or self._iostream is None or not isinstance(name, str):
            return super()._read_array_from_stream(shape, dtype)
        array = np.empty(shape, dtype=dtype)
        start = self._iostream.tell()
        flat = array.reshape(-1).view(np.uint8)
        result = _read_blocks_into(name, start, flat, self._threads)
        if result is None:  # not a clean run of BGZF blocks
            return super()._read_array_from_stream(shape, dtype)
        filled, total = result
        self._trailing_bytes = max(0, total - start - array.nbytes)
        return array, filled

    def flush(self) -> None:
        """Write the file as BGZF blocks, rather than as one gzip stream.

        As for gzip, the whole file is rewritten. Unlike gzip, the file on disk
        is complete when this returns, down to the end-of-file block.
        """
        if self._read_only or self._iostream is None:
            return
        self._iostream.close()

        prefix = b""
        if self.header is not None:
            prefix += self.header.tobytes()
        if self.extended_header is not None:
            prefix += self.extended_header.tobytes()
        data = memoryview(b"")
        if self.data is not None:
            contiguous = np.ascontiguousarray(self.data)
            # NumPy's stubs before 2.1 do not type ndarray as a buffer for Python 3.9
            data = memoryview(contiguous.reshape(-1).view(np.uint8))  # type: ignore[arg-type]

        self._fileobj.seek(0)
        pieces = _block_data(prefix, data)
        self._fileobj.writelines(
            _compress_blocks(pieces, self._compresslevel, self._threads)
        )
        self._fileobj.write(EOF_BLOCK)
        self._fileobj.truncate()
        self._fileobj.flush()

        # Leave a reader open, as after opening the file, so that the data can
        # still be read and close() still flushes.
        self._fileobj.seek(0)
        self._iostream = cast(
            BinaryIO, gzip.GzipFile(fileobj=self._fileobj, mode="rb")
        )  # cast needed because of awkward IO types
