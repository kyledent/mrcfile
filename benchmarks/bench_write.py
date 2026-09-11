"""Compressed write path: the cost is peak memory, not wall time."""

import os
import tempfile

import _harness as h
import numpy as np

import mrcfile


def main():
    path = h.fixture("gz_src.mrc")
    h.warm(path)
    with mrcfile.open(path) as m:
        data = np.array(m.data)  # 64 MiB resident

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "bench_write.mrc.gz")

        def write_gz():
            with mrcfile.new(out, overwrite=True, compression="gzip") as m:
                m.set_data(data)
            os.remove(out)

        h.emit("write", {"write_gz_64MiB": h.measure(write_gz, repeats=1)})


main()
