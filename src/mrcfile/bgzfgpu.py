# Copyright (c) 2026, Diamond Light Source
# This software is distributed under a BSD licence. See LICENSE.txt.
"""
bgzfgpu
-------

Decode BGZF-compressed MRC files straight into GPU memory.

A BGZF file's blocks are independent, so a GPU can decode thousands of them at
once. :func:`read_to_gpu` reads the header on the CPU, copies the compressed file
to the GPU in one transfer, and decodes every block there with NVIDIA's nvCOMP
library. It checks each block's length and CRC32, and returns the data as a
CuPy array. The uncompressed data never pass through host memory.

This needs CuPy, nvCOMP's Python package and an NVIDIA GPU that nvCOMP supports.
``pip install mrcfile[gpu]`` installs the CUDA 12 builds of both packages.
Neither is imported until :func:`read_to_gpu` is called, so this module can be
imported without them.

Functions:
    :func:`read_to_gpu`: Read a BGZF-compressed MRC file's data into GPU memory.

"""

from __future__ import annotations

import gzip
import importlib
import io
import os
import warnings
from typing import Any

import numpy as np

from . import utils
from .bgzfmrcfile import _TRAILER, BgzfMrcFile, bgzf_blocks, is_bgzf

# CRC32 of each decoded block, one GPU thread per block. nvCOMP does not check
# the CRC32 in a gzip member's trailer, so this does.
_CRC32_SOURCE = r"""
extern "C" __global__ void crc32_blocks(
    const unsigned long long* blocks, const int* lengths,
    const unsigned int* table, unsigned int* crcs, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned char* p = (const unsigned char*) blocks[i];
    unsigned int c = 0xFFFFFFFFu;
    for (int k = 0; k < lengths[i]; ++k) {
        c = table[(c ^ p[k]) & 0xFFu] ^ (c >> 8);
    }
    crcs[i] = c ^ 0xFFFFFFFFu;
}
"""


def _crc32_table() -> np.ndarray:
    """Return the lookup table for the CRC32 that gzip and zlib use."""
    table = np.zeros(256, dtype=np.uint32)
    for n in range(256):
        c = n
        for _ in range(8):
            c = (c >> 1) ^ 0xEDB88320 if c & 1 else c >> 1
        table[n] = c
    return table


def _load(module: str) -> Any:
    """Import a module that only reading into GPU memory needs."""
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(
            "Reading into GPU memory needs CuPy and nvCOMP's Python package"
            " (pip install mrcfile[gpu])"
        ) from err


def _layout(
    name: str, raw: bytes
) -> tuple[int, int, np.dtype, tuple[int, ...], list[tuple[int, int, int, int]]]:
    """Read the header and list the blocks, checking both before any GPU work.

    Returns:
        The data block's offset and length in the uncompressed file, its dtype
        and shape, and each non-empty block's offset, size, CRC32 and
        uncompressed length.

    Raises:
        :exc:`ValueError`: If the file is not BGZF, if a block has gzip header
            fields besides its size, if its data are not in this machine's byte
            order, or if its data block is shorter than the header says.

    Warns:
        RuntimeWarning: If the file holds more data than the header describes.
    """
    if not is_bgzf(raw):
        raise ValueError(f"'{name}' is not a BGZF file")
    with BgzfMrcFile(name, header_only=True) as mrc:
        header = mrc.header
        if header is None:
            raise RuntimeError(f"Cannot read the header of '{name}'")
        start = header.nbytes + int(header.nsymbt)
        dtype = utils.data_dtype_from_header(header)
        shape = utils.data_shape_from_header(header)
    if not dtype.isnative:
        raise ValueError(
            "The data are not in this machine's byte order, which CuPy cannot"
            " hold; read them with mrcfile.read() instead"
        )
    nbytes = int(np.prod(shape)) * dtype.itemsize

    # Empty blocks, such as the end-of-file block, hold nothing to decode
    blocks: list[tuple[int, int, int, int]] = []
    for offset, size in bgzf_blocks(io.BytesIO(raw)):
        # Any gzip flag but FEXTRA adds fields that move the compressed data
        if raw[offset + 3] != 0x04:
            raise ValueError(
                f"The BGZF block at byte {offset} has gzip header fields besides"
                " its size; read the file with mrcfile.read() instead"
            )
        crc, isize = _TRAILER.unpack_from(raw, offset + size - _TRAILER.size)
        if isize:
            blocks.append((offset, size, crc, isize))
    total = sum(isize for _, _, _, isize in blocks)
    if total < start + nbytes:
        # The same error as mrcfile.open() gives when the blocks hold too little
        raise ValueError(
            f"Expected {nbytes} bytes in data block but limit is"
            f" {max(0, total - start)}"
        )
    if total > start + nbytes:
        extra = total - start - nbytes
        warnings.warn(f"MRC file is {extra} bytes larger than expected", RuntimeWarning)
    return start, nbytes, dtype, shape, blocks


def _decode_blocks(
    cp: Any, nvcomp: Any, raw: bytes, blocks: list[tuple[int, int, int, int]]
) -> Any:
    """Decode every block on the GPU with nvCOMP, and check each one's length.

    Returns:
        The whole uncompressed file, as one CuPy array of bytes.

    Raises:
        :exc:`gzip.BadGzipFile`: If a block's length does not match its trailer.
    """
    compressed = cp.asarray(np.frombuffer(raw, dtype=np.uint8))
    # nvCOMP decodes on CuPy's current device and stream, so it starts only after
    # the upload has finished, and the copies below start only after it is done
    codec = nvcomp.Codec(
        algorithm="Gzip",
        bitstream_kind=nvcomp.BitstreamKind.RAW,
        device_id=cp.cuda.Device().id,
        cuda_stream=cp.cuda.get_current_stream().ptr,
    )
    sources = [nvcomp.as_array(compressed[o : o + s]) for o, s, _, _ in blocks]
    # nvCOMP's arrays own the decoded blocks, and CuPy's views of them do not keep
    # that memory alive, so the blocks are copied into a CuPy array of their own
    # before nvCOMP's arrays are released
    owners = codec.decode(sources)
    views = [cp.from_dlpack(d).view(cp.uint8).ravel() for d in owners]
    cp.cuda.runtime.deviceSynchronize()
    for (offset, _, _, isize), view in zip(blocks, views):
        if view.size != isize:
            raise gzip.BadGzipFile(
                f"Incorrect length of data produced by the block at byte {offset}"
            )
    decoded = cp.concatenate(views)
    # Let the copy finish before nvCOMP's arrays are released
    cp.cuda.runtime.deviceSynchronize()
    return decoded


def _check_crcs(cp: Any, blocks: list[tuple[int, int, int, int]], decoded: Any) -> None:
    """Check each decoded block's CRC32 against its trailer, on the GPU.

    Raises:
        :exc:`gzip.BadGzipFile`: If any block's CRC32 does not match.
    """
    kernel = cp.RawKernel(_CRC32_SOURCE, "crc32_blocks")
    sizes = np.array([isize for _, _, _, isize in blocks], dtype=np.uint64)
    # Where each block starts in the decoded file
    starts = np.cumsum(sizes) - sizes
    pointers = cp.asarray(starts + np.uint64(decoded.data.ptr))
    lengths = cp.asarray(sizes.astype(np.int32))
    table = cp.asarray(_crc32_table())
    crcs = cp.empty(len(blocks), dtype=cp.uint32)
    per_group = 128
    groups = (len(blocks) + per_group - 1) // per_group
    kernel(
        (groups,),
        (per_group,),
        (pointers, lengths, table, crcs, np.int32(len(blocks))),
    )
    expected = np.array([crc for _, _, crc, _ in blocks], dtype=np.uint32)
    bad = np.flatnonzero(cp.asnumpy(crcs) != expected)
    if bad.size:
        raise gzip.BadGzipFile(
            f"CRC check failed for {bad.size} block(s), the first at byte"
            f" {blocks[bad[0]][0]}"
        )


def read_to_gpu(name: str | os.PathLike[str], *, verify: bool = True) -> Any:
    """Read a BGZF-compressed MRC file's data into GPU memory.

    The header is read on the CPU. The compressed file is then copied to the
    current GPU in one transfer, and nvCOMP decodes all of its blocks there at
    once. Each block's length is checked against its trailer, and so is its
    CRC32 unless ``verify`` is :data:`False`. A data block that is too short
    raises the same error as :func:`mrcfile.open`, and one that is too long
    gives the same warning.

    The compressed file is held in host memory, and briefly on the GPU too,
    alongside two copies of the decoded data, so the GPU needs room for the
    compressed file plus twice the data.

    Args:
        name: The file name, as a string or :class:`~pathlib.Path`.
        verify: Check each block's CRC32, on the GPU. nvCOMP does not check it
            itself, and a block whose compressed data are damaged can decode to
            the right length but the wrong bytes, so without this check such
            damage can go unnoticed. The default is :data:`True`.

    Returns:
        The data as a ``cupy.ndarray`` on the current GPU, with the dtype and
        shape that the header describes.

    Raises:
        :exc:`ImportError`: If CuPy or nvCOMP's Python package is not
            installed.
        :exc:`ValueError`: If the file is not BGZF, if a block has gzip header
            fields besides its size, if its data block is shorter than the
            header says, or if its data are not in this machine's byte order.
        :exc:`gzip.BadGzipFile`: If a block's length or CRC32 does not match
            its trailer.

    Warns:
        RuntimeWarning: If the file holds more data than the header describes.
    """
    name = os.fspath(name)
    with open(name, "rb") as f:
        raw = f.read()
    start, nbytes, dtype, shape, blocks = _layout(name, raw)
    cp = _load("cupy")
    nvcomp = _load("nvidia.nvcomp")
    if nbytes == 0:
        return cp.empty(shape, dtype=dtype)
    decoded = _decode_blocks(cp, nvcomp, raw, blocks)
    if verify:
        _check_crcs(cp, blocks, decoded)
    # Copying the data block into a fresh array also aligns it, whatever its
    # offset in the file
    data = decoded[start : start + nbytes].copy()
    return data.view(dtype).reshape(shape)
