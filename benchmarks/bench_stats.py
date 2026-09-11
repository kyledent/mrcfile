"""update_header_stats() and set_data(), the two statistics hot paths.

set_data is measured against an in-memory stream so we time the copy and the
statistics pass, not the filesystem.
"""

import io

import _harness as h
import numpy as np

import mrcfile
from mrcfile.mrcinterpreter import MrcInterpreter


def main():
    path = h.fixture("vol_f32.mrc")
    h.warm(path)
    mrc = mrcfile.open(path, mode="r+")
    data = np.array(mrc.data)

    ipath = h.fixture("vol_i16.mrc")
    h.warm(ipath)
    mrci = mrcfile.open(ipath, mode="r+")

    def stats_f32():
        mrc.update_header_stats()

    def stats_i16():
        mrci.update_header_stats()

    def set_data():
        m = MrcInterpreter()
        m._create_default_attributes()
        m._iostream = io.BytesIO()
        m.set_data(data)
        m._iostream = None
        m._header = None
        m._data = None

    h.emit(
        "stats",
        {
            "update_header_stats_f32_128MiB": h.measure(stats_f32),
            "update_header_stats_i16_64MiB": h.measure(stats_i16),
            "set_data_128MiB": h.measure(set_data),
        },
    )
    mrc.close()
    mrci.close()


main()
