"""Shared benchmark harness.

Each bench_*.py is run as a subprocess with PYTHONPATH pointing at either
baseline/ or fork/, and prints a single JSON object on stdout. This keeps the
two versions of the package completely isolated from each other.
"""

import gc
import json
import os
import sys
import time
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXDIR = os.path.join(ROOT, ".fixtures")


def fixture(name: str) -> str:
    path = os.path.join(FIXDIR, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing fixture {path} - run benchmarks/make_fixtures.py")
    return path


def warm(path: str) -> None:
    """Pull a file into the page cache so we time CPU work, not cold I/O."""
    with open(path, "rb") as f:
        while f.read(1 << 24):
            pass


def measure(fn, repeats: int = 3) -> dict:
    """Best-of-N wall time plus peak transient allocation for one call."""
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return {"ms": best * 1e3, "peak_mib": (peak - base) / 2**20}


def emit(name: str, results: dict) -> None:
    variant = os.environ.get("MRC_VARIANT", "?")
    print(json.dumps({"bench": name, "variant": variant, "results": results}))


def report_env() -> dict:
    return {"python": sys.version.split()[0], "numpy": np.__version__}
