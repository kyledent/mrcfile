# Deviations from upstream

Every optimisation in this fork is intended to be behaviour-preserving. Where
behaviour does change, it is recorded here. A maintainer should read this
before merging anything.

---

## D1 — Header statistics are computed in double precision

**Affects:** `MrcObject.update_header_stats()`, `MrcObject.validate()`
**Upstream:** `data.mean(dtype=np.float32)` and `data.std(dtype=np.float32)`
**Fork:** `utils.calculate_stats()`, accumulating in float64 about a shifted mean

The header fields are still float32; only the accumulation changes. **The fork
stores the correctly rounded float32 value of the true mean and standard
deviation. Upstream's float32 accumulation does not, and on real cryo-EM data it
misses by tens of float32 ULPs.**

**On real data.** Each implementation's stored `dmean` and `rms` were compared
with the same statistics computed in extended precision (80-bit `longdouble`),
on nine real float32 files of 0.7 to 690 million values: a reconstructed
tomogram, a subtomogram average, two tilt series, a template-matching score
map, a detector gain reference, a micrograph, a particle stack and a deposited
map.

| | `dmean` error | `rms` error |
|---|---|---|
| Upstream, float32 accumulation | 0 to 35 ULPs | 1 to 17 ULPs |
| Fork, float64 accumulation | 0 ULPs on every file | 0 ULPs on every file |

The fork stores exactly the float32 value nearest the true one, every time. So
the difference between the two, which reached 35 ULPs, is all upstream's error.

**What an ULP count means here.** An ULP (unit in the last place) is the spacing
between adjacent float32 values at the value being stored: about 6e-8 of that
value. So 35 ULPs is a relative error of about 2e-6 in the mean. On a scale that
does not depend on where zero falls, the data's own standard deviation,
upstream's `dmean` error was at most 1.7e-5 of the rms and its `rms` error at
most 1.1e-6 of the rms. Upstream's mean was worst where the mean sits far above
the spread of the values: 1.7e-5 of the rms both for a score map whose mean is
7 times its rms, and for a micrograph in raw counts whose mean is 26 times its
rms. That is too small to matter for display, scaling or any physical use. It
is large enough, though, that headers written by the fork and by upstream for
the same data differ in their last several bits.

**Why upstream errs.** For float32 data, `ndarray.mean()` and `ndarray.std()`
accumulate in float32, pairwise. Every addition rounds to 24 significant bits,
and the rounding errors add up over the tens or hundreds of millions of values in
a real volume, most of all when the values are large compared with how much they
vary. Accumulating in float64 leaves 29 more bits for that error, so the sum is
effectively exact before it is rounded once into the float32 header.

Worked example, using the array from the upstream test
`test_stats_are_updated_for_new_data`:

| | RMS | absolute error |
|---|---|---|
| Exact (50 dp) | 10349.886454251861638608 | — |
| Upstream, float32 accumulation | 10349.8857421875 | 7.12e-04 |
| Fork, float64 accumulation | 10349.886454251862 | 5.94e-13 |

Rounded into the header's float32 field, upstream stores `10349.886` and the
fork stores `10349.887`. The nearest float32 to the exact value is
`10349.887`, so **upstream is one ULP low and the fork is correct.**

The gap widens with array size (float32 accumulation error grows with the
number of terms) and with DC offset. For mode 6 data centred near 30000 a naive
sum-of-squares loses most of its significant digits; the fork shifts by a
provisional mean taken from the first block to avoid this, which is why
`test_large_dc_offset_is_stable` exists.

**Test change required.** `tests/test_mrcobject.py::test_stats_are_updated_for_new_data`
asserted bit-equality against the float32-accumulated value. It now asserts
against the float64 reference, which is what both implementations *should*
round to. The amended test **fails on baseline and passes on the fork** — that
is the intended discrimination, not a regression. The claim is asserted rather
than asserted-by-comment in
`tests/test_calculate_stats.py::test_is_at_least_as_accurate_as_float32_accumulation`,
which compares both implementations against an arbitrary-precision `Decimal`
reference.

**Risk:** Any downstream code comparing a freshly written header's `rms` or
`dmean` for exact equality against its own float32-accumulated calculation will
see a mismatch. On large real arrays it is tens of ULPs, not one. Code using
`np.isclose` with a relative tolerance of 1e-4 or looser, or reading the value
rather than recomputing it, is unaffected. No MRC2014 conformance requirement
is affected, since the standard does not specify an accumulation order.

**To revert** without giving up the speed, cast the accumulators in
`calculate_stats` to float32. Precision goes back to upstream's and the test
change becomes unnecessary; the single-pass saving is unaffected.

---

## D2 — `validate()` no longer allocates a full boolean array

**Affects:** `MrcObject.validate()`

Upstream guards three statistics checks with `len(self.data > 0)`, which
materialises a complete boolean array the size of the data and then takes the
length of its first axis. That value is non-zero for any non-empty array, so
the guard never behaved as written. The fork uses `self.data.size > 0`.

**Behaviour change:** for an empty array whose *first* axis is not zero-length,
such as shape `(5, 0)`. Upstream's guard is then true, so `validate()` goes on to
compute the statistics of an empty array and raises `ValueError: zero-size
array to reduction operation minimum`. The fork's guard is false, so it skips the
statistics checks and reports on the rest of the file instead of raising. For
an array whose first axis is zero-length both spellings agree. No test changed.

---

## Deviations deliberately *not* taken

- `std()`'s return dtype is preserved in `validate()`, so the
  "Data statistics appear to be inaccurate" message formats byte-for-byte as it
  always has. `test_incorrect_rms` asserts the exact string and still passes.
- `min` and `max` are returned in the array's own dtype, not promoted to
  float64, so `header.dmin != real_min` comparisons behave exactly as before.
- NaN propagates to both `min` and `max`, matching numpy, rather than being
  skipped over as a naive running-minimum would.
