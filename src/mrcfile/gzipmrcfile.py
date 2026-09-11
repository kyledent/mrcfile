# Copyright (c) 2016, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.
"""
gzipmrcfile
-----------

Module which exports the :class:`GzipMrcFile` class.

Classes:
    :class:`GzipMrcFile`: An object which represents a gzipped MRC file.

"""

from __future__ import annotations

import builtins
import gzip
import os
import warnings
from typing import BinaryIO, Literal, cast

from . import utils
from .mrcfile import MrcFile


class GzipMrcFile(MrcFile):
    """:class:`~mrcfile.mrcfile.MrcFile` subclass for handling gzipped files.

    Usage is the same as for :class:`~mrcfile.mrcfile.MrcFile`, except that the
    constructor also accepts ``compresslevel``: the gzip compression level, from
    0 to 9, to use when the file is written. The default is 9.

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
    ) -> None:
        """Initialise a new :class:`GzipMrcFile` object.

        Takes the same arguments as :class:`~mrcfile.mrcfile.MrcFile`, plus
        ``compresslevel``, the gzip level (0 to 9) to write with. The default,
        :data:`None`, uses level 9.

        Raises:
            :exc:`ValueError`: If ``compresslevel`` is not between 0 and 9. This
                is checked before the file is opened, so an invalid level never
                truncates an existing file.
        """
        if compresslevel is not None and not 0 <= compresslevel <= 9:
            raise ValueError(
                f"gzip compresslevel must be between 0 and 9, not {compresslevel}"
            )
        self._compresslevel = 9 if compresslevel is None else compresslevel
        super().__init__(
            name,
            mode,
            overwrite=overwrite,
            permissive=permissive,
            header_only=header_only,
        )

    def __repr__(self) -> str:
        """Return a string representation of the GzipMrcFile object."""
        return f"GzipMrcFile('{self._fileobj.name}', mode='{self._mode}')"

    def _open_file(self, name: str | os.PathLike[str]) -> None:
        """Override _open_file() to open both normal and gzip files."""
        self._fileobj = open(name, self._mode + "b")  # noqa: SIM115  # no context manager
        self._iostream = cast(
            BinaryIO, gzip.GzipFile(fileobj=self._fileobj, mode="rb")
        )  # cast needed because of awkward IO types

    def _close_file(self) -> None:
        """Override _close_file() to close both normal and gzip files."""
        if self._iostream is None:
            raise RuntimeError("Cannot close file because no file is set")
        self._iostream.close()
        self._fileobj.close()

    def _read(self, *, header_only: bool = False) -> None:
        """Override _read() to ensure gzip file is in read mode."""
        self._ensure_readable_gzip_stream()
        super()._read(header_only=header_only)

    def _ensure_readable_gzip_stream(self) -> None:
        """Make sure _iostream is a gzip stream that can be read."""
        if self._iostream is None:
            raise RuntimeError("Cannot read file because no file is set")
        if self._iostream.mode != gzip.READ:
            self._iostream.close()
            self._fileobj.seek(0)
            self._iostream = cast(
                BinaryIO, gzip.GzipFile(fileobj=self._fileobj, mode="rb")
            )  # cast needed because of awkward IO types

    def _get_file_size(self) -> int:
        """Override _get_file_size() to avoid seeking from end.

        Kept for API compatibility. This is the expensive measurement: it
        decompresses everything that is left and then seeks backwards, which
        makes :class:`~gzip.GzipFile` decompress the whole stream again from
        the start. :meth:`_read_data` avoids calling it.
        """
        if self._iostream is None:
            raise RuntimeError("Cannot get file size because no file is set")
        self._ensure_readable_gzip_stream()
        pos = self._iostream.tell()
        extra = len(self._iostream.read())
        self._iostream.seek(pos, os.SEEK_SET)
        return pos + extra

    def _uncompressed_size_from_trailer(self) -> int | None:
        """Return the uncompressed length from the gzip ISIZE trailer.

        The last four bytes of a gzip member hold the uncompressed size modulo
        2**32. Reading them costs two syscalls on a separate file handle, so
        the live :class:`~gzip.GzipFile` is never disturbed.

        ISIZE is unreliable in two cases: the counter wraps for streams of 4 GiB
        or more, and for a multi-member file it describes only the last member.
        Both make it an under-estimate, and DEFLATE cannot shrink data, so an
        ISIZE below the compressed length proves the value is untrustworthy.
        :data:`None` is returned in that case, and the caller falls back to
        reading without a cap.
        """
        name = getattr(self._fileobj, "name", None)
        if not isinstance(name, str):
            return None
        try:
            with builtins.open(name, "rb") as raw:
                compressed = raw.seek(0, os.SEEK_END)
                if compressed < 18:  # smaller than an empty gzip member
                    return None
                raw.seek(-4, os.SEEK_END)
                isize = int.from_bytes(raw.read(4), "little")
        except OSError:
            return None
        return isize if isize >= compressed else None

    def _read_data(self) -> None:
        """Read the data block using a single forward decompression pass.

        The inherited implementation calls :meth:`_get_file_size` first, purely
        to work out a sanity limit and to spot trailing bytes. For a compressed
        stream that measurement costs two extra full decompressions. Here the
        limit comes from the ISIZE trailer instead, and trailing bytes are
        counted by continuing forwards after the data block rather than by
        rewinding.
        """
        if self.header is None:
            raise RuntimeError(
                "Cannot read data from an uninitialised or closed MRC object"
            )
        header_size = self.header.nbytes + int(self.header.nsymbt)
        total = self._uncompressed_size_from_trailer()
        # max_bytes of 0 means "no limit" to _read_data_from_stream
        max_bytes = total - header_size if total is not None else 0

        super(MrcFile, self)._read_data_from_stream(max_bytes=max_bytes)

        if self.data is not None:
            extra = self._count_remaining_bytes()
            if extra > 0:
                warnings.warn(
                    f"MRC file is {extra} bytes larger than expected", RuntimeWarning
                )

    def _count_remaining_bytes(self) -> int:
        """Count whatever is left in the stream, reading forwards only."""
        if self._iostream is None:
            return 0
        total = 0
        while True:
            chunk = self._iostream.read(1 << 20)
            if not chunk:
                return total
            total += len(chunk)

    def flush(self) -> None:
        """Override :meth:`~mrcfile.mrcinterpreter.MrcInterpreter.flush` since
        GzipFile objects need special handling.
        """
        if not self._read_only and self._iostream is not None:
            self._iostream.close()
            self._fileobj.seek(0)
            self._iostream = cast(
                BinaryIO,
                gzip.GzipFile(
                    fileobj=self._fileobj, mode="wb", compresslevel=self._compresslevel
                ),
            )  # cast needed because of awkward IO types

            # Header and extended header are small, so a copy is fine. The
            # data block is streamed in chunks: tobytes() on a large volume
            # doubles peak memory for no benefit, and gzip can calculate
            # sizes from memoryview slices just as well.
            if self.header is not None:
                self._iostream.write(self.header.tobytes())
            if self.extended_header is not None:
                self._iostream.write(self.extended_header.tobytes())
            if self.data is not None:
                utils.write_array_in_chunks(self._iostream, self.data)
            self._iostream.flush()
            self._fileobj.truncate()
