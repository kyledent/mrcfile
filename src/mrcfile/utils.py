# Copyright (c) 2016, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.

"""
utils
-----

Utility functions used by the other modules in the mrcfile package.

Functions
---------

* :func:`data_dtype_from_header`: Work out the data :class:`dtype
  <numpy.dtype>` from an MRC header.
* :func:`data_shape_from_header`: Work out the data array shape from an MRC
  header
* :func:`mode_from_dtype`: Convert a :class:`numpy dtype <numpy.dtype>` to an
  MRC mode number.
* :func:`dtype_from_mode`: Convert an MRC mode number to a :class:`numpy dtype
  <numpy.dtype>`.
* :func:`pretty_machine_stamp`: Get a nicely-formatted string from a machine
  stamp.
* :func:`machine_stamp_from_byte_order`: Get a machine stamp from a byte order
  indicator.
* :func:`byte_orders_equal`: Compare two byte order indicators for equal
  endianness.
* :func:`normalise_byte_order`: Convert a byte order indicator to ``<`` or
  ``>``.
* :func:`spacegroup_is_volume_stack`: Identify if a space group number
  represents a volume stack.

"""

from __future__ import annotations

import math
import numbers
import string
import sys
from typing import Any, BinaryIO, Literal

import numpy as np

from .constants import IMAGE_STACK_SPACEGROUP


def data_dtype_from_header(header: np.recarray) -> np.dtype:
    """Return the data dtype indicated by the given header.

    This function calls :func:`dtype_from_mode` to get the basic dtype, and
    then makes sure that the byte order of the new dtype matches the byte order
    of the header's ``mode`` field.

    Args:
        header: An MRC header as a :class:`numpy record array
            <numpy.recarray>`.

    Returns:
        The :class:`numpy dtype <numpy.dtype>` object for the data array
        corresponding to the given header.

    Raises:
        :exc:`ValueError`: If there is no corresponding dtype for the given
            mode.
    """
    mode = header.mode
    return dtype_from_mode(mode).newbyteorder(mode.dtype.byteorder)


def check_int_argument(value: Any, name: str, low: int, high: int | None = None) -> int:
    """Check that ``value`` is an integer from ``low`` to ``high`` inclusive.

    For arguments such as compression levels, which have to be checked before a
    file is opened: a value that the compressor only rejects when it starts
    writing would leave an existing file truncated.

    Args:
        value: The value to check. Any integer type is accepted, :class:`bool`
            is not.
        name: The argument's name, for the error message.
        low: The smallest value allowed.
        high: The largest value allowed, or :data:`None` for no upper limit.

    Returns:
        ``value`` as a Python :class:`int`.

    Raises:
        :exc:`TypeError`: If ``value`` is not an integer.
        :exc:`ValueError`: If ``value`` is out of range.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, not {value!r}")
    value = int(value)
    if value < low or (high is not None and value > high):
        allowed = f"at least {low}" if high is None else f"between {low} and {high}"
        raise ValueError(f"{name} must be {allowed}, not {value}")
    return value


def data_block_nbytes(header: np.recarray) -> int | None:
    """Return the size in bytes of the data block that ``header`` describes.

    Args:
        header: The MRC header, as a :class:`numpy.recarray`.

    Returns:
        The size in bytes, or :data:`None` if the header does not describe a
        valid data block, for example because its mode is not recognised.
    """
    try:
        dtype = data_dtype_from_header(header)
        shape = data_shape_from_header(header)
    except ValueError:
        return None
    return dtype.itemsize * math.prod(int(n) for n in shape)


def data_shape_from_header(
    header: np.recarray,
) -> tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]:
    """Return the data shape indicated by the given header.

    Args:
        header: An MRC header as a :class:`numpy record array
            <numpy.recarray>`.

    Returns:
        The shape tuple for the data array corresponding to the given header.
    """
    nx = int(header.nx)
    ny = int(header.ny)
    nz = int(header.nz)
    mz = int(header.mz)

    shape: tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]
    if spacegroup_is_volume_stack(header.ispg):
        shape = (nz // mz, mz, ny, nx)
    elif header.ispg == IMAGE_STACK_SPACEGROUP and nz == 1:
        # Use a 2D array for a single image
        shape = (ny, nx)
    else:
        shape = (nz, ny, nx)

    return shape


_dtype_to_mode = {"f2": 12, "f4": 2, "i1": 0, "i2": 1, "u1": 6, "u2": 6, "c8": 4}


def mode_from_dtype(dtype: np.dtype) -> int:
    """Return the MRC mode number corresponding to the given :class:`numpy
    dtype <numpy.dtype>`.

    The conversion is as follows:

    * float16   -> mode 12
    * float32   -> mode 2
    * int8      -> mode 0
    * int16     -> mode 1
    * uint8     -> mode 6 (data will be widened to 16 bits in the file)
    * uint16    -> mode 6
    * complex64 -> mode 4

    Note that there is no numpy dtype which corresponds to MRC mode 3.

    Args:
        dtype: A :class:`numpy dtype <numpy.dtype>` object.

    Returns:
        The MRC mode number.

    Raises:
        :exc:`ValueError`: If there is no corresponding MRC mode for the given
            dtype.
    """
    kind_and_size = dtype.kind + str(dtype.itemsize)
    if kind_and_size in _dtype_to_mode:
        return _dtype_to_mode[kind_and_size]
    raise ValueError(f"dtype '{dtype}' cannot be converted to an MRC file mode")


_mode_to_dtype = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    4: np.complex64,
    6: np.uint16,
    12: np.float16,
}


def dtype_from_mode(mode: int) -> np.dtype:
    """Return the :class:`numpy dtype <numpy.dtype>` corresponding to the given
    MRC mode number.

    The mode parameter may be given as a Python scalar, numpy scalar or
    single-item numpy array.

    The conversion is as follows:

    * mode 0 -> int8
    * mode 1 -> int16
    * mode 2 -> float32
    * mode 4 -> complex64
    * mode 6 -> uint16
    * mode 12 -> float16

    Note that modes 3 and 101 are not supported as there is no matching numpy dtype.

    Args:
        mode: The MRC mode number. This may be given as any type which can be
            converted to an int, for example a Python scalar (``int`` or
            ``float``), a numpy scalar or a single-item numpy array.

    Returns:
        The :class:`numpy dtype <numpy.dtype>` object corresponding to the
        given mode.

    Raises:
        :exc:`ValueError`: If there is no corresponding dtype for the given
            mode, or if ``mode`` is an array and does not contain exactly one
            item.
    """
    if isinstance(mode, np.ndarray):
        if mode.size != 1:
            raise ValueError("Mode array should contain exactly one item")
        mode = mode.item()
    if mode in _mode_to_dtype:
        return np.dtype(_mode_to_dtype[mode])
    else:
        raise ValueError(f"Unrecognised mode '{mode}'")


def pretty_machine_stamp(machst: bytes) -> str:
    """Return a human-readable hex string for a machine stamp."""
    return " ".join(f"0x{byte:02x}" for byte in machst)


def byte_order_from_machine_stamp(machst: bytes) -> Literal["<", ">"]:
    """Return the byte order corresponding to the given machine stamp.

    Args:
        machst: The machine stamp, as a :class:`bytearray` or a :class:`numpy
            array <numpy.ndarray>` of bytes.

    Returns:
        ``<`` if the machine stamp represents little-endian data, or ``>`` if
        it represents big-endian.

    Raises:
        :exc:`ValueError`: If the machine stamp is invalid.
    """
    if machst[0] == 0x44 and machst[1] in (0x44, 0x41):
        return "<"
    elif machst[0] == 0x11 and machst[1] == 0x11:
        return ">"
    else:
        pretty_bytes = pretty_machine_stamp(machst)
        raise ValueError("Unrecognised machine stamp: " + pretty_bytes)


_byte_order_to_machine_stamp = {
    "<": bytearray((0x44, 0x44, 0, 0)),
    ">": bytearray((0x11, 0x11, 0, 0)),
}


def machine_stamp_from_byte_order(byte_order: str = "=") -> bytearray:
    """Return the machine stamp corresponding to the given byte order
    indicator.

    Args:
        byte_order: The byte order indicator: one of ``=``, ``<`` or ``>``, as
            defined and used by numpy dtype objects.

    Returns:
        The machine stamp which corresponds to the given byte order, as a
        :class:`bytearray`. This will be either ``(0x44, 0x44, 0, 0)`` for
        little-endian or ``(0x11, 0x11, 0, 0)`` for big-endian. If the given
        byte order indicator is ``=``, the native byte order is used.

    Raises:
        :exc:`ValueError`: If the byte order indicator is unrecognised.
    """
    # If byte order is '=', replace it with the system-native order
    byte_order = normalise_byte_order(byte_order)
    return _byte_order_to_machine_stamp[byte_order]


def byte_orders_equal(a: str, b: str) -> bool:
    """Work out if the byte order indicators represent the same endianness.

    Args:
        a: The first byte order indicator: one of ``=``, ``<`` or ``>``, as
            defined and used by :class:`numpy dtype <numpy.dtype>` objects.
        b: The second byte order indicator.

    Returns:
        :data:`True` if the byte order indicators represent the same
        endianness.

    Raises:
        :exc:`ValueError`: If the byte order indicator is not recognised.
    """
    return normalise_byte_order(a) == normalise_byte_order(b)


def normalise_byte_order(byte_order: str) -> Literal["<", ">"]:
    """Convert a numpy byte order indicator to one of ``<`` or ``>``.

    Args:
        byte_order: One of ``=``, ``<`` or ``>``.

    Returns:
        ``<`` if the byte order indicator represents little-endian data, or
        ``>`` if it represents big-endian. Therefore on a little-endian
        machine, ``=`` will be converted to ``<``, but on a big-endian machine
        it will be converted to ``>``.

    Raises:
        :exc:`ValueError`: If ``byte_order`` is not one of ``=``, ``<`` or
            ``>``.
    """
    if byte_order == "<":
        return "<"
    elif byte_order == ">":
        return ">"
    elif byte_order == "=":
        return "<" if sys.byteorder == "little" else ">"
    else:
        raise ValueError(f"Unrecognised byte order indicator '{byte_order}'")


#: Bytes per write in :func:`write_array_in_chunks`.
WRITE_CHUNK_BYTES = 1 << 24  # 16 MiB


def write_array_in_chunks(
    stream: BinaryIO, array: np.ndarray, chunk_bytes: int = WRITE_CHUNK_BYTES
) -> None:
    """Write an array to a stream without duplicating it in memory.

    ``stream.write(array.tobytes())`` builds a complete copy of the array
    first, so writing a 4 GB volume to a compressed stream needs 8 GB resident.
    Writing :class:`memoryview` slices of a contiguous view of the array
    achieves the same result with a bounded buffer, and compressors do not care
    where the chunk boundaries fall.

    Args:
        stream: A writeable binary stream.
        array: The array to write. Copied only if it is not already contiguous,
            which is the same condition under which ``tobytes()`` would copy.
        chunk_bytes: Maximum bytes per write call.
    """
    contiguous = np.ascontiguousarray(array)
    # NumPy's stubs before 2.1 do not type ndarray as a buffer for Python 3.9
    view = memoryview(contiguous.reshape(-1).view(np.uint8))  # type: ignore[arg-type]
    stream.writelines(
        view[start : start + chunk_bytes]
        for start in range(0, view.nbytes, chunk_bytes)
    )


#: Elements per block in :func:`calculate_stats`. 65536 float32 values is
#: 256 KiB, which stays resident in L2 on typical hardware so that all four
#: reductions read from cache and main memory is traversed exactly once.
STATS_BLOCK_SIZE = 65536


def calculate_stats(
    data: np.ndarray, block_size: int = STATS_BLOCK_SIZE
) -> tuple[Any, Any, float, float]:
    """Calculate min, max, mean and RMS deviation in a single pass.

    This is equivalent to calling :meth:`~numpy.ndarray.min`,
    :meth:`~numpy.ndarray.max`, :meth:`~numpy.ndarray.mean` and
    :meth:`~numpy.ndarray.std` separately, but traverses the array once instead
    of roughly five times, and uses a fixed small amount of scratch memory
    instead of allocating a temporary the same size as the data. The saving is
    largest for memory-mapped arrays, where each avoided pass is an avoided
    read of the whole file.

    Sums are accumulated in double precision about a provisional mean taken
    from the first block. This avoids the catastrophic cancellation that a
    naive sum-of-squares suffers when the mean is large relative to the spread,
    as it is for mode 6 data centred near 30000.

    ``min`` and ``max`` are returned in the array's own dtype so that they
    compare against header values exactly as the equivalent numpy calls would.
    NaN is propagated to both, matching numpy's behaviour.

    Args:
        data: The data array. Must not be empty, and must not be complex.
        block_size: Number of elements to process at a time.

    Returns:
        A 4-tuple of ``(min, max, mean, rms)``.

    Raises:
        :exc:`ValueError`: If the array is empty.
    """
    flat = np.ravel(data)
    n = flat.size
    if n == 0:
        raise ValueError("Cannot calculate statistics for an empty array")

    # Provisional mean from the first block, used as a shift for stability
    shift = float(np.mean(flat[: min(block_size, n)], dtype=np.float64))

    min_: Any = None
    max_: Any = None
    nan_seen = False
    total = 0.0
    total_sq = 0.0

    for start in range(0, n, block_size):
        chunk = flat[start : start + block_size]
        chunk_min = chunk.min()
        chunk_max = chunk.max()
        if np.isnan(chunk_min) or np.isnan(chunk_max):
            nan_seen = True
        elif min_ is None:
            min_, max_ = chunk_min, chunk_max
        else:
            min_ = min(min_, chunk_min)
            max_ = max(max_, chunk_max)

        # Cache-resident widening: the copy never leaves L2
        deviations = chunk.astype(np.float64) - shift
        total += float(deviations.sum())
        total_sq += float(np.dot(deviations, deviations))

    if nan_seen or min_ is None:
        nan = np.array(np.nan, dtype=np.float32)[()]
        min_ = max_ = nan

    offset = total / n
    mean = shift + offset
    variance = total_sq / n - offset * offset
    if math.isnan(variance):
        rms = math.nan
    else:
        # Clamp tiny negatives from rounding on constant data
        rms = math.sqrt(variance) if variance > 0.0 else 0.0
    return min_, max_, mean, rms


def spacegroup_is_volume_stack(ispg: int) -> bool:
    """Identify if the given space group number represents a volume stack.

    Args:
        ispg: The space group number, as an integer, numpy scalar or single-
            element numpy array.

    Returns:
        :data:`True` if the space group number is in the range 401--630.
    """
    return 401 <= ispg <= 630


printable_chars = " " + string.ascii_letters + string.digits + string.punctuation


def is_printable_ascii(string_: str) -> bool:
    """Check if a string is entirely composed of printable ASCII characters."""
    return str.isprintable(string_) and str.isascii(string_)


def printable_string_from_bytes(bytes_: bytes) -> str:
    """Convert bytes into a printable ASCII string.

    Non-printable characters are removed.
    """
    string_ = bytes.decode(bytes_, encoding="ascii", errors="ignore")
    if not is_printable_ascii(string_):
        string_ = "".join([s for s in string_ if is_printable_ascii(s)])
    return string_
