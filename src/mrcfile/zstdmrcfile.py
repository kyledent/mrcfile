# Copyright (c) 2016, Science and Technology Facilities Council
# Copyright (c) 2026, Diamond Light Source
# This software is distributed under a BSD licence. See LICENSE.txt.
"""
zstdmrcfile
-----------

Module which exports the :class:`ZstdMrcFile` class.

Zstandard support comes from the standard library's ``compression.zstd`` module on
Python 3.14 and later, or from the ``backports.zstd`` package on earlier versions
(``pip install mrcfile[zstd]`` installs it). Without either, this module still
imports, but :class:`ZstdMrcFile` raises :exc:`ImportError` when used.

Classes:
    :class:`ZstdMrcFile`: An object which represents a Zstandard-compressed MRC
    file.

"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Any, BinaryIO, Literal, cast

from . import utils
from .mrcfile import MrcFile


def _load_zstd() -> Any:
    """Return the Zstandard module: the standard library's, else the backport."""
    try:
        return importlib.import_module("compression.zstd")
    except ImportError:
        try:
            return importlib.import_module("backports.zstd")
        except ImportError:
            return None


#: The Zstandard module in use, or :data:`None` if none is available.
_zstd: Any = _load_zstd()

#: The most a Zstandard file is taken to expand when decompressed. No block holds
#: more than 128 KiB, and the smallest block that decodes to that many bytes takes
#: 4, so data never decompresses to more than 32768 times its compressed size.
#: This allows twice that. It only guards against a header that claims far more
#: data than the file could hold.
_MAX_EXPANSION = 1 << 16


class ZstdMrcFile(MrcFile):
    """:class:`~mrcfile.mrcfile.MrcFile` subclass for handling Zstandard-compressed
    files.

    Usage is the same as for :class:`~mrcfile.mrcfile.MrcFile`, except that the
    constructor also accepts ``compresslevel``: the Zstandard compression level to
    use when the file is written. The default is Zstandard's own default level.

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
        """Initialise a new :class:`ZstdMrcFile` object.

        Takes the same arguments as :class:`~mrcfile.mrcfile.MrcFile`, plus
        ``compresslevel``, the Zstandard level to write with. Levels run up to 22,
        and negative levels are faster still. The default, :data:`None`, uses
        Zstandard's own default level.

        Raises:
            :exc:`ImportError`: If Zstandard support is not available. It needs
                Python 3.14 or later, or the ``backports.zstd`` package.
            :exc:`ValueError`: If ``compresslevel`` is outside the range the
                Zstandard library accepts. This is checked before the file is
                opened, so an invalid level never truncates an existing file.
        """
        if _zstd is None:
            raise ImportError(
                "Zstandard support needs Python 3.14 or later, or the"
                " 'backports.zstd' package (pip install mrcfile[zstd])"
            )
        if compresslevel is not None:
            low, high = _zstd.CompressionParameter.compression_level.bounds()
            if not low <= compresslevel <= high:
                raise ValueError(
                    f"zstd compresslevel must be between {low} and {high},"
                    f" not {compresslevel}"
                )
        self._compresslevel = compresslevel
        super().__init__(
            name,
            mode,
            overwrite=overwrite,
            permissive=permissive,
            header_only=header_only,
        )

    def __repr__(self) -> str:
        """Return a string representation of the ZstdMrcFile object."""
        return f"ZstdMrcFile('{self._fileobj.name}', mode='{self._mode}')"

    def _open_file(self, name: str | os.PathLike[str]) -> None:
        """Override _open_file() to open both the file and a Zstandard stream."""
        self._fileobj = open(name, self._mode + "b")  # noqa: SIM115  # no context manager
        self._iostream = cast(
            BinaryIO, _zstd.ZstdFile(self._fileobj, mode="rb")
        )  # cast needed because of awkward IO types

    def _close_file(self) -> None:
        """Override _close_file() to close both the stream and the file."""
        if self._iostream is None:
            raise RuntimeError("Cannot close file because no file is set")
        self._iostream.close()
        self._fileobj.close()

    def _read(self, *, header_only: bool = False) -> None:
        """Override _read() to ensure the Zstandard stream is in read mode."""
        self._ensure_readable_zstd_stream()
        super()._read(header_only=header_only)

    def _ensure_readable_zstd_stream(self) -> None:
        """Make sure _iostream is a Zstandard stream that can be read."""
        if self._iostream is None:
            raise RuntimeError("Cannot read file because no file is set")
        if not self._iostream.readable():
            self._iostream.close()
            self._fileobj.seek(0)
            self._iostream = cast(
                BinaryIO, _zstd.ZstdFile(self._fileobj, mode="rb")
            )  # cast needed because of awkward IO types

    def _get_file_size(self) -> int:
        """Override _get_file_size() to avoid seeking from the end.

        Kept for API compatibility. It decompresses everything that is left and
        then seeks backwards, which makes the stream decompress again from the
        start. :meth:`_read_data` avoids calling it.
        """
        if self._iostream is None:
            raise RuntimeError("Cannot get file size because no file is set")
        self._ensure_readable_zstd_stream()
        pos = self._iostream.tell()
        extra = len(self._iostream.read())
        self._iostream.seek(pos, os.SEEK_SET)
        return pos + extra

    def _read_data(self) -> None:
        """Read the data block in a single forward decompression pass.

        A Zstandard stream written without a known size records no content
        length, so the data block cannot cheaply be checked against the file's
        true length. It is checked instead against the most the file could
        decompress to, :data:`_MAX_EXPANSION` times its compressed size, which
        refuses a header that claims far more data than the file holds before
        anything is allocated. Such a header is then measured exactly, as
        :class:`~mrcfile.mrcfile.MrcFile` does, so that the error is the same.
        Any bytes after the data block are counted by reading forwards rather
        than by rewinding.
        """
        if self.header is None:
            raise RuntimeError(
                "Cannot read data from an uninitialised or closed MRC object"
            )
        needed = utils.data_block_nbytes(self.header)
        limit = os.fstat(self._fileobj.fileno()).st_size * _MAX_EXPANSION
        if needed is None or needed > limit:
            super()._read_data()
            return

        self._read_data_from_stream(max_bytes=limit)

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
        Zstandard streams need special handling.
        """
        if not self._read_only and self._iostream is not None:
            self._iostream.close()
            self._fileobj.seek(0)
            self._iostream = cast(
                BinaryIO,
                _zstd.ZstdFile(self._fileobj, mode="wb", level=self._compresslevel),
            )  # cast needed because of awkward IO types

            # Header and extended header are small, so a copy is fine. The data
            # block is streamed in chunks to avoid a second copy of the array.
            if self.header is not None:
                self._iostream.write(self.header.tobytes())
            if self.extended_header is not None:
                self._iostream.write(self.extended_header.tobytes())
            if self.data is not None:
                utils.write_array_in_chunks(self._iostream, self.data)
            self._iostream.flush()
            self._fileobj.truncate()
