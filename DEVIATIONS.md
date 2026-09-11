# Deviations from upstream

Every optimisation in this fork is intended to be behaviour-preserving. Where
behaviour does change, it is recorded here. A maintainer should read this
before merging anything.

---

## D1 — Header statistics are computed in double precision

**Affects:** `MrcObject.update_header_stats()`, `MrcObject.validate()`
**Upstream:** `data.mean(dtype=np.float32)` and `data.std(dtype=np.float32)`
**Fork:** `utils.calculate_stats()`, accumulating in float64 about a shifted mean

The header fields are still float32; only the accumulation changes. The fork's
answer is closer to the true value, sometimes by one ULP of float32.

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

**Test change required.** `tests/upstream/test_mrcobject.py::test_stats_are_updated_for_new_data`
asserted bit-equality against the float32-accumulated value. It now asserts
against the float64 reference, which is what both implementations *should*
round to. The amended test **fails on baseline and passes on the fork** — that
is the intended discrimination, not a regression. The claim is asserted rather
than asserted-by-comment in
`tests/fork/test_calculate_stats.py::test_is_at_least_as_accurate_as_float32_accumulation`,
which compares both implementations against an arbitrary-precision `Decimal`
reference.

**Risk:** Any downstream code comparing a freshly written header's `rms` or
`dmean` for exact equality against its own float32-accumulated calculation will
see a mismatch in the last bit. Code using `np.isclose`, or reading the value
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

**Behaviour change:** for an array whose first axis is length zero but which is
not itself empty, the guard's result changes. Such an array has `size == 0`, so
both spellings agree in practice. No test changed.

---

## Deviations deliberately *not* taken

- `std()`'s return dtype is preserved in `validate()`, so the
  "Data statistics appear to be inaccurate" message formats byte-for-byte as it
  always has. `test_incorrect_rms` asserts the exact string and still passes.
- `min` and `max` are returned in the array's own dtype, not promoted to
  float64, so `header.dmin != real_min` comparisons behave exactly as before.
- NaN propagates to both `min` and `max`, matching numpy, rather than being
  skipped over as a naive running-minimum would.
