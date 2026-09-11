"""Differential tests for :func:`mrcfile.utils.calculate_stats`.

The kernel replaces four separate numpy reductions with one blocked pass, so
these tests check it against numpy across every dtype an MRC file can hold and
across the edge cases that reduction kernels get wrong: empty and single-element
arrays, constant data, data with a large DC offset, NaN, infinities, block
boundaries, non-contiguous views and both byte orders.
"""

import math

import numpy as np
import pytest

from mrcfile.utils import STATS_BLOCK_SIZE, calculate_stats

# Every dtype reachable through an MRC mode
MRC_DTYPES = ["<i1", "<i2", "<u2", "<f2", "<f4", ">i2", ">f4"]


def reference(arr):
    """numpy's answer, computed in the highest precision available."""
    return (
        arr.min(),
        arr.max(),
        arr.mean(dtype=np.float64),
        arr.std(dtype=np.float64),
    )


def assert_matches(arr, rtol=1e-9):
    got_min, got_max, got_mean, got_rms = calculate_stats(arr)
    exp_min, exp_max, exp_mean, exp_rms = reference(arr)

    assert got_min == exp_min
    assert got_max == exp_max
    # min/max must keep the array's own dtype so header comparisons behave
    assert np.asarray(got_min).dtype == arr.dtype.newbyteorder("=")
    assert got_mean == pytest.approx(exp_mean, rel=rtol, abs=1e-9)
    assert got_rms == pytest.approx(exp_rms, rel=rtol, abs=1e-9)


@pytest.mark.parametrize("dtype", MRC_DTYPES)
def test_matches_numpy_for_every_mrc_dtype(dtype):
    rng = np.random.default_rng(20260911)
    np_dtype = np.dtype(dtype)
    if np_dtype.kind == "f":
        arr = rng.normal(0, 1, 100_003).astype(dtype)
    else:
        info = np.iinfo(np_dtype)
        arr = rng.integers(info.min // 2, info.max // 2, 100_003).astype(dtype)
    # float16 has ~3 decimal digits, so loosen the tolerance for it alone
    assert_matches(arr, rtol=1e-3 if np_dtype.itemsize == 2 and np_dtype.kind == "f" else 1e-9)


@pytest.mark.parametrize(
    "size",
    [1, 2, 1000, STATS_BLOCK_SIZE - 1, STATS_BLOCK_SIZE, STATS_BLOCK_SIZE + 1,
     2 * STATS_BLOCK_SIZE, 2 * STATS_BLOCK_SIZE + 7],
)
def test_block_boundaries(size):
    """Sizes either side of the block stride must not drop or double-count."""
    arr = np.linspace(-5, 5, size, dtype=np.float32)
    assert_matches(arr)


def test_constant_data_gives_exactly_zero_rms():
    arr = np.full(200_000, 7.25, dtype=np.float32)
    min_, max_, mean, rms = calculate_stats(arr)
    assert min_ == max_ == np.float32(7.25)
    assert mean == pytest.approx(7.25)
    assert rms == 0.0  # not a small negative under the sqrt


def test_large_dc_offset_is_stable():
    """The case a naive sum-of-squares gets wrong: mode 6 data near 30000."""
    rng = np.random.default_rng(7)
    arr = (rng.normal(0, 1, 300_000) + 30_000).astype(np.float32)
    _, _, mean, rms = calculate_stats(arr)
    assert mean == pytest.approx(arr.mean(dtype=np.float64), rel=1e-9)
    assert rms == pytest.approx(arr.std(dtype=np.float64), rel=1e-6)
    assert rms > 0.9  # a cancelling implementation collapses this towards 0


def test_nan_propagates_to_min_and_max():
    arr = np.append(np.arange(1000, dtype=np.float32), np.nan).astype(np.float32)
    min_, max_, mean, rms = calculate_stats(arr)
    assert np.isnan(min_) and np.isnan(max_)
    assert math.isnan(mean) and math.isnan(rms)


def test_nan_in_a_later_block_is_still_caught():
    arr = np.zeros(3 * STATS_BLOCK_SIZE, dtype=np.float32)
    arr[-1] = np.nan
    min_, max_, _, _ = calculate_stats(arr)
    assert np.isnan(min_) and np.isnan(max_)


def test_infinities_match_numpy():
    arr = np.array([-np.inf, 0.0, 1.0, np.inf], dtype=np.float32)
    min_, max_, _, _ = calculate_stats(arr)
    assert min_ == -np.inf
    assert max_ == np.inf


def test_non_contiguous_view():
    arr = np.arange(500_000, dtype=np.float32).reshape(1000, 500)[:, ::2]
    assert not arr.flags.c_contiguous
    assert_matches(arr)


def test_multidimensional_input():
    rng = np.random.default_rng(3)
    arr = rng.normal(0, 1, (17, 23, 29)).astype(np.float32)
    assert_matches(arr)


def test_empty_array_raises():
    with pytest.raises(ValueError, match="empty"):
        calculate_stats(np.empty(0, dtype=np.float32))


def test_block_size_does_not_change_the_answer():
    rng = np.random.default_rng(11)
    arr = rng.normal(100, 5, 250_000).astype(np.float32)
    results = [calculate_stats(arr, block_size=b) for b in (1024, 65536, 1 << 20)]
    for other in results[1:]:
        assert other[0] == results[0][0]
        assert other[1] == results[0][1]
        assert other[2] == pytest.approx(results[0][2], rel=1e-12)
        assert other[3] == pytest.approx(results[0][3], rel=1e-9)


def test_is_at_least_as_accurate_as_float32_accumulation():
    """The reason for FORK DEVIATION D1, asserted rather than claimed."""
    from decimal import Decimal, getcontext

    getcontext().prec = 60
    img = np.linspace(-32768, 32767, 90, dtype=np.int16).reshape(9, 10)
    vol = img // np.arange(1, 6, dtype=np.int16).reshape(5, 1, 1)

    values = [Decimal(int(v)) for v in vol.ravel()]
    n = Decimal(len(values))
    exact_mean = sum(values) / n
    exact_rms = (sum((v - exact_mean) ** 2 for v in values) / n).sqrt()

    fork_err = abs(Decimal(calculate_stats(vol)[3]) - exact_rms)
    upstream_err = abs(Decimal(float(vol.std(dtype=np.float32))) - exact_rms)
    assert fork_err <= upstream_err
