"""Generate benchmark fixtures. Kept out of the archive - run this first.

Usage:  python benchmarks/make_fixtures.py
"""
import gzip
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXDIR = os.path.join(ROOT, ".fixtures")

# (name, shape, dtype)
SPECS = [
    ("vol_f32.mrc", (128, 512, 512), "<f4"),   # 128 MiB  float32 volume
    ("vol_i16.mrc", (128, 512, 512), "<i2"),   #  64 MiB  int16 volume
    ("gz_src.mrc", (64, 512, 512), "<f4"),     #  64 MiB  source for the .gz
]


def build() -> None:
    os.makedirs(FIXDIR, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "baseline"))
    import mrcfile

    rng = np.random.default_rng(20260911)
    for name, shape, dtype in SPECS:
        path = os.path.join(FIXDIR, name)
        if os.path.exists(path):
            print(f"  exists  {name}")
            continue
        if dtype == "<f4":
            data = rng.normal(0, 1, shape).astype(dtype)
        else:
            data = rng.integers(-2000, 2000, shape).astype(dtype)
        with mrcfile.new(path, overwrite=True) as m:
            m.set_data(data)
        del data
        print(f"  built   {name}  ({os.path.getsize(path) / 2**20:.0f} MiB)")

    gz = os.path.join(FIXDIR, "gz_src.mrc.gz")
    if not os.path.exists(gz):
        with open(os.path.join(FIXDIR, "gz_src.mrc"), "rb") as fi, \
                gzip.open(gz, "wb", compresslevel=1) as fo:
            shutil.copyfileobj(fi, fo, length=1 << 24)
        print(f"  built   gz_src.mrc.gz  ({os.path.getsize(gz) / 2**20:.0f} MiB)")


if __name__ == "__main__":
    print("building fixtures in", FIXDIR)
    build()
