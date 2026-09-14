# Copyright (c) 2016, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.
"""
bzip2mrcfile
------------

Module which exports the :class:`Bzip2MrcFile` class.

Classes:
    :class:`Bzip2MrcFile`: An object which represents a bzip2-compressed MRC
    file.

"""

from __future__ import annotations

import bz2
import os
from typing import BinaryIO, Literal, cast

from . import utils
from .mrcfile import MrcFile


class Bzip2MrcFile(MrcFile):
    """:class:`~mrcfile.mrcfile.MrcFile` subclass for handling bzip2-compressed
    files.

    Usage is the same as for :class:`~mrcfile.mrcfile.MrcFile`, except that the
    constructor also accepts ``compresslevel``: the bzip2 compression level, from
    1 to 9, to use when the file is written. The default is 9.

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
        """Initialise a new :class:`Bzip2MrcFile` object.

        Takes the same arguments as :class:`~mrcfile.mrcfile.MrcFile`, plus
        ``compresslevel``, the bzip2 level (1 to 9) to write with. The default,
        :data:`None`, uses level 9.

        Raises:
            :exc:`TypeError`: If ``compresslevel`` is not an integer.
            :exc:`ValueError`: If ``compresslevel`` is not between 1 and 9.
                Both are checked before the file is opened, so an invalid level
                never truncates an existing file.
        """
        if compresslevel is not None:
            compresslevel = utils.check_int_argument(
                compresslevel, "bzip2 compresslevel", 1, 9
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
        """Return a string representation of the Bzip2MrcFile object."""
        return f"Bzip2MrcFile('{self._fname}', mode='{self._mode}')"

    def _open_file(self, name: str | os.PathLike[str]) -> None:
        """Override _open_file() to open a bzip2 file."""
        self._fname = name
        if "w" in self._mode and not os.path.exists(name):
            open(name, mode="w").close()
        self._iostream = cast(
            BinaryIO, bz2.BZ2File(name, mode="r")
        )  # cast needed because of awkward IO types

    def _read(self, *, header_only: bool = False) -> None:
        """Override _read() to ensure bzip2 file is in read mode."""
        self._ensure_readable_bzip2_stream()
        super()._read(header_only=header_only)

    def _ensure_readable_bzip2_stream(self) -> None:
        """Make sure _iostream is a bzip2 stream that can be read."""
        if self._iostream is None:
            raise RuntimeError("Cannot read file because no file is set")
        if not self._iostream.readable():
            self._iostream.close()
            self._iostream = cast(
                BinaryIO, bz2.BZ2File(self._fname, mode="r")
            )  # cast needed because of awkward IO types

    def _get_file_size(self) -> int:
        """Override _get_file_size() to ensure stream is readable first."""
        if self._iostream is None:
            raise RuntimeError("Cannot get file size because no file is set")
        self._ensure_readable_bzip2_stream()
        return super()._get_file_size()

    def flush(self) -> None:
        """Override :meth:`~mrcfile.mrcinterpreter.MrcInterpreter.flush` since
        BZ2File objects need special handling.
        """
        if not self._read_only and self._iostream is not None:
            self._iostream.close()
            self._iostream = cast(
                BinaryIO,
                bz2.BZ2File(self._fname, mode="w", compresslevel=self._compresslevel),
            )  # cast needed because of awkward IO types

            # Header and extended header are small, so a copy is fine. The
            # data block is streamed in chunks: tobytes() on a large volume
            # doubles peak memory for no benefit, and bz2 can calculate
            # sizes from memoryview slices just as well.
            if self.header is not None:
                self._iostream.write(self.header.tobytes())
            if self.extended_header is not None:
                self._iostream.write(self.extended_header.tobytes())
            if self.data is not None:
                utils.write_array_in_chunks(self._iostream, self.data)
            # no equivalent for flush() with BZ2File
