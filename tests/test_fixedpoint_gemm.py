"""
The fixed-point GEMM (phase T): its definition, written as an independent
reference (``tests/fixedpoint_gemm_reference.py``), the rules that choose how
a call is computed and what is refused (``mptorch/number/fixedpoint.py``'s
``_fixedpoint_gemm``), and the two ops, ``custom_matmul_fixedpoint`` (a split
mac) and ``custom_matmul_fixedpoint_fma`` (a fused one), on the CPU.

- **The reference rounds as the shipped cast does.** It is held word for
  word to the elementwise fixed-point cast, over values in every region (on
  the grid, ties, below the step, past the range, infinities, NaN), for
  signed, unsigned and symmetric formats and both carriers. The
  deterministic modes are compared with ``fixedpoint_quantize``, and the
  stochastic ones with ``_stochastic_cast`` on explicit random words.
- **Its sums are exact.** A format that never rounds gives the float64
  matmul of on-grid operands. Wrap-around leaves the exact sum's residue, so
  the result is right whenever the final sum fits, however the partial sums
  overflowed; saturation does not.
- **The carrier path is the integer path.** Where the rules say the carrier
  is exact, a step-by-step loop of the shipped casts in the carrier gives
  the reference's words, stochastic rounding included. Just past the
  boundary they differ.
- **The rules:** the carrier's boundary at 24 and 53 bits, fused and
  wrap-around macs on the integer path, each refusal, and when a result is
  wide.
- **The ops compute the definition** (tier 3): word for word with the
  reference, in every deterministic mode, both macs and both carriers, on
  the carrier path and the integer accumulator, saturating and wrapping,
  for narrow, wide, unsigned and symmetric formats, transposed and batched
  operands, and NaN and infinite products. Where the rules choose the
  carrier path, forcing the integer accumulator gives the same words, under
  stochastic rounding too, and stochastic rounding's mean is P3109's. A
  wide result is float64 and exact. The op boundary refuses what the
  integer accumulator cannot hold.
"""

import math
import random
from fractions import Fraction
from typing import NamedTuple

import pytest
import torch

from mptorch import FixedPoint, RoundMode
from mptorch.number.fixedpoint import _fixedpoint_gemm, _fixedpoint_gemm_wide, _fixedpoint_span
from mptorch.quant import fixedpoint_quantize

from . import fixedpoint_gemm_reference as ref

R = RoundMode
DETERMINISTIC = [R.RNE, R.RNA, R.RU, R.RD, R.RZ, R.RO]
STOCHASTIC = [R.SR, R.SR_B, R.SR_C]
CAST = torch.ops.mptorch._stochastic_cast.default
F32 = torch.float32

FORMATS = {
    "s8.4": FixedPoint(8, 4),
    "s8.4 symmetric": FixedPoint(8, 4, symmetric=True),
    "u8.4": FixedPoint(8, 4, is_signed=False),
    "s16.8": FixedPoint(16, 8),
    "s5.-2": FixedPoint(5, -2),  # a step of 4
    "s6.9": FixedPoint(6, 9),  # every value a small fraction
}


def _values(fmt: FixedPoint) -> list[float]:
    """Values in every region of the format: on the grid, at a quarter, a third,
    a half and three quarters of a step, below the step, at and past both ends,
    and the non-finite ones."""
    lo, hi = ref.codes(fmt)
    step = 2.0**-fmt.fl
    out = []
    for c in sorted({lo, lo + 1, -3, -1, 0, 1, 2, 3, hi - 1, hi}):
        for f in (0.0, 0.25, 1 / 3, 0.5, 0.75):
            out += [(c + f) * step, -(c + f) * step]
    out += [step / 8, -step / 8, step / 2, -step / 2]
    out += [(hi + 3) * step, (lo - 3) * step, 1e30, -1e30, math.inf, -math.inf, math.nan]
    return out


def _as_reference(
    x: float, fmt: FixedPoint, mode: RoundMode, word: int, carrier: torch.dtype
) -> float:
    v = ref.value(ref.quantize(x, fmt, mode, word, carrier=carrier), fmt)
    return math.nan if v is None else float(v)


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


# --- the reference's rounding, against the shipped cast --------------------------


@pytest.mark.parametrize("carrier", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", list(FORMATS))
@pytest.mark.parametrize("mode", DETERMINISTIC)
def test_the_reference_rounds_as_the_cast_does(name, mode, carrier):
    fmt = FORMATS[name]
    xs = _values(fmt)
    x = torch.tensor(xs, dtype=carrier)
    got = fixedpoint_quantize(
        x, fmt.wl, fmt.fl, is_signed=fmt.is_signed, symmetric=fmt.symmetric, rounding_mode=mode
    ).tolist()
    for xv, g in zip(x.tolist(), got, strict=True):
        want = _as_reference(xv, fmt, mode, 0, carrier)
        assert _same(g, want), (xv, g, want)


@pytest.mark.parametrize("carrier", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["s8.4", "u8.4", "s8.4 symmetric", "s6.9"])
@pytest.mark.parametrize("n", [0, 3, 8])
@pytest.mark.parametrize("mode", STOCHASTIC)
def test_the_reference_rounds_stochastically_as_the_cast_does(name, n, mode, carrier):
    """Each value under many random words, through the test-only op that
    takes them explicitly: the bits the reference reads are the cast's."""
    base = FORMATS[name]
    fmt = FixedPoint(
        base.wl, base.fl, is_signed=base.is_signed, symmetric=base.symmetric, prng_bits=n
    )
    rng = random.Random(7)
    xs = [v for v in _values(fmt) for _ in range(16)]
    words = [rng.getrandbits(32 if carrier is torch.float32 else 63) for _ in xs]
    x = torch.tensor(xs, dtype=carrier)
    w = torch.tensor(words, dtype=torch.int64)
    fields = [fmt.wl, fmt.fl, n, int(fmt.is_signed), int(fmt.symmetric)]
    got = CAST(x, w, 2, fields, mode.value).tolist()
    for xv, word, g in zip(x.tolist(), words, got, strict=True):
        want = _as_reference(xv, fmt, mode, word, carrier)
        assert _same(g, want), (xv, word, g, want)


def test_the_cast_rounds_ties_and_directions_as_the_definition_says():
    """A few cases by hand, in units of the step: ties to even and away,
    directed modes on negatives, round to odd, saturation, an unsigned
    format's negatives."""
    f = FixedPoint(8, 0)
    q = ref.quantize
    assert [q(Fraction(x), f, R.RNE) for x in (2.5, 3.5, -2.5, -3.5)] == [2, 4, -2, -4]
    assert [q(Fraction(x), f, R.RNA) for x in (2.5, -2.5)] == [3, -3]
    assert [q(Fraction(x), f, R.RU) for x in (2.25, -2.25)] == [3, -2]
    assert [q(Fraction(x), f, R.RD) for x in (2.25, -2.25)] == [2, -3]
    assert [q(Fraction(x), f, R.RZ) for x in (2.75, -2.75)] == [2, -2]
    assert [q(Fraction(x), f, R.RO) for x in (2.25, 3.25, -2.25)] == [3, 3, -3]
    assert [q(Fraction(x), f, R.RNE) for x in (130, -130)] == [127, -128]
    assert q(Fraction(-130), FixedPoint(8, 0, symmetric=True), R.RNE) == -127
    assert q(Fraction(-2), FixedPoint(8, 0, is_signed=False), R.RNE) == 0


# --- the sums ---------------------------------------------------------------------


def _on_grid(shape, fmt: FixedPoint, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    lo, hi = ref.codes(fmt)
    return torch.randint(lo, hi + 1, shape, generator=g).double() * 2.0**-fmt.fl


@pytest.mark.parametrize("fused", [False, True])
def test_a_format_that_never_rounds_gives_the_exact_sum(fused):
    a = _on_grid((5, 33), FixedPoint(8, 4), 1).float()
    b = _on_grid((33, 6), FixedPoint(8, 4), 2).float()
    mul = None if fused else FixedPoint(16, 8)
    got = ref.gemm(a, b, mul=mul, acc=FixedPoint(40, 8))
    assert torch.equal(got, a.double() @ b.double())


def test_wrap_around_keeps_the_residue_and_saturation_does_not():
    """int8 sums of 100, 100, -100 and -60: the partial sums overflow at the
    second step, and the final sum, 40, fits. Wrap-around gives it;
    saturation gives -33."""
    a = torch.tensor([[100.0, 100.0, -100.0, -60.0]])
    b = torch.ones(4, 1)
    int8, wide = FixedPoint(8, 0), FixedPoint(16, 0)
    assert ref.gemm(a, b, mul=wide, acc=int8, wrap=True).item() == 40
    assert ref.gemm(a, b, mul=wide, acc=int8).item() == -33


def test_a_wrapped_sum_is_the_exact_sum_modulo_the_word():
    a = _on_grid((4, 40), FixedPoint(8, 2), 3).float()
    b = _on_grid((40, 5), FixedPoint(8, 2), 4).float()
    acc = FixedPoint(10, 4)
    got = ref.gemm(a, b, mul=FixedPoint(16, 4), acc=acc, wrap=True)
    exact = a.double() @ b.double()
    m = 2 ** (acc.wl - acc.fl)
    assert torch.equal(torch.remainder(got - exact, m), torch.zeros_like(got))


def test_a_nan_product_makes_the_element_nan():
    a = torch.tensor([[1.0, math.nan, 2.0], [1.0, 2.0, 3.0]])
    b = torch.ones(3, 1)
    got = ref.gemm(a, b, mul=FixedPoint(16, 4), acc=FixedPoint(16, 4))
    assert math.isnan(got[0, 0].item()) and got[1, 0].item() == 6


# --- the carrier path gives the integer path's words ------------------------------


def _carrier_loop(a, b, mul, acc, mode, words=None):
    """The split mac computed the carrier's way: the shipped elementwise casts
    in the carrier, a step at a time over the whole [M, N] grid, each sum
    rounded by the carrier. With ``words`` (``[K, 2, M, N]``), the stochastic
    casts take them explicitly."""
    c = torch.float32
    a, b = a.to(c), b.to(c)
    s = torch.zeros(a.shape[0], b.shape[1], dtype=c)

    def q(x, fmt, slot, k):
        if words is None:
            return fixedpoint_quantize(x, fmt.wl, fmt.fl, rounding_mode=mode)
        fields = [fmt.wl, fmt.fl, fmt.prng_bits, int(fmt.is_signed), int(fmt.symmetric)]
        return CAST(x.contiguous(), words[k, slot].contiguous(), 2, fields, mode.value)

    for k in range(a.shape[1]):
        p = q(a[:, k : k + 1] * b[k : k + 1, :], mul, 0, k)
        s = q(s + p, acc, 1, k)
    return s.double()


@pytest.mark.parametrize("mode", DETERMINISTIC)
def test_where_the_rules_say_the_carrier_is_exact_it_gives_the_integer_paths_words(mode):
    mul, acc = FixedPoint(16, 8), FixedPoint(16, 4)
    assert _fixedpoint_gemm(mul, acc, False, torch.float32).in_carrier
    g = torch.Generator().manual_seed(5)
    a = torch.randn(6, 48, generator=g) * 4
    b = torch.randn(48, 7, generator=g) * 4
    assert torch.equal(
        _carrier_loop(a, b, mul, acc, mode), ref.gemm(a, b, mul=mul, acc=acc, mode=mode)
    )


@pytest.mark.parametrize("mode", STOCHASTIC)
def test_the_carrier_path_draws_as_the_integer_path_does(mode):
    """The same random words, read the same way, so stochastic rounding agrees
    too: the condition for the carrier to be the integer accumulator's fast
    path."""
    mul, acc = FixedPoint(16, 8, prng_bits=6), FixedPoint(16, 4, prng_bits=9)
    assert _fixedpoint_gemm(mul, acc, False, torch.float32).in_carrier
    g = torch.Generator().manual_seed(6)
    a = torch.randn(5, 32, generator=g) * 4
    b = torch.randn(32, 6, generator=g) * 4
    words = torch.randint(0, 2**32, (32, 2, 5, 6), generator=g, dtype=torch.int64)
    got = _carrier_loop(a, b, mul, acc, mode, words)
    want = ref.gemm(
        a, b, mul=mul, acc=acc, mode=mode, draw=lambda i, j, k, s: int(words[k, s, i, j])
    )
    assert torch.equal(got, want)


def test_past_the_boundary_the_carrier_rounds_the_sum():
    """The negative control. Four products of 16384 make 65536, where
    binary32's step is 2^-7, and a fifth of 2^-8 is a tie there. The exact sum
    keeps it, and binary32's sum rounds it away: this pair spans 32 bits, so
    the rules send it to the integer path. (The shipped cast refuses a 32-bit
    format in binary32 outright, so the carrier's sums are shown alone.)"""
    mul, acc = FixedPoint(24, 8), FixedPoint(32, 8)
    assert _fixedpoint_span(mul, acc) == 32
    assert not _fixedpoint_gemm(mul, acc, False, torch.float32).in_carrier
    a = torch.tensor([[128.0, 128.0, 128.0, 128.0, 0.0625]])
    b = a.T.clone()
    assert ref.gemm(a, b, mul=mul, acc=acc).item() == 65536 + 2**-8
    s = torch.zeros((), dtype=torch.float32)
    for p in (a * b.T)[0]:  # the carrier's products, each on the multiply grid
        s = s + p  # and the carrier's sums
    assert s.item() == 65536


# --- the rules ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "mul, acc, carrier, in_carrier",
    [
        (FixedPoint(8, 0), FixedPoint(24, 0), torch.float32, True),  # a span of 24 bits
        (FixedPoint(8, 0), FixedPoint(25, 0), torch.float32, False),  # 25
        (FixedPoint(16, 8), FixedPoint(16, 4), torch.float32, True),  # 20
        (FixedPoint(8, 0), FixedPoint(53, 0), torch.float64, True),  # 53
        (FixedPoint(8, 0), FixedPoint(54, 0), torch.float64, False),  # 54
    ],
)
def test_the_carrier_computes_up_to_its_precision(mul, acc, carrier, in_carrier):
    assert _fixedpoint_gemm(mul, acc, False, carrier) == (in_carrier, None)
    assert (_fixedpoint_span(mul, acc) <= (24 if carrier is torch.float32 else 53)) == in_carrier


def test_an_accumulate_format_the_cast_misrounds_takes_the_integer_path():
    """A step of 2^-126 in binary32: the elementwise rule warns that values
    near zero may round incorrectly, so the carrier path, which rounds sums
    with that cast, would not give the definition's bits."""
    mul, acc = FixedPoint(8, 120), FixedPoint(16, 126)
    assert _fixedpoint_span(mul, acc) == 16
    assert _fixedpoint_gemm(mul, acc, False, torch.float32) == (False, None)
    assert _fixedpoint_gemm(mul, acc, False, torch.float64).in_carrier


def test_fused_and_wrapping_macs_take_the_integer_path():
    assert _fixedpoint_gemm(None, FixedPoint(16, 4), False, torch.float32) == (False, None)
    assert _fixedpoint_gemm(FixedPoint(16, 8), FixedPoint(16, 4), True, torch.float32) == (
        False,
        None,
    )


@pytest.mark.parametrize(
    "mul, acc, wrap, carrier, match",
    [
        (
            FixedPoint(32, 8),
            FixedPoint(40, 8),
            False,
            F32,
            "multiply format is rounded in the carrier",
        ),
        (FixedPoint(16, 8), FixedPoint(16, 4, prng_bits=30), False, F32, "draws 30 random bits"),
        (FixedPoint(16, 8), FixedPoint(16, 4, symmetric=True), True, F32, "no modulus to wrap"),
        (FixedPoint(32, 40), FixedPoint(40, -10), False, torch.float64, "needs 90 bits"),
        # a step below what the carrier's cast rounds correctly: elementwise a
        # warning, here a refusal, since both paths round products with it
        (FixedPoint(8, 126), FixedPoint(16, 4), False, F32, "below 2^-125"),
    ],
)
def test_what_the_integer_path_refuses(mul, acc, wrap, carrier, match):
    found = _fixedpoint_gemm(mul, acc, wrap, carrier)
    assert not found.in_carrier and found.error is not None and match in found.error


@pytest.mark.parametrize(
    "acc, dtype, wide",
    [
        (FixedPoint(16, 4), torch.float32, False),
        (FixedPoint(25, 4), torch.float32, False),  # 24 magnitude bits fit
        (FixedPoint(26, 4), torch.float32, True),  # 25 do not
        (FixedPoint(16, 4), torch.bfloat16, True),
        (FixedPoint(9, 4), torch.bfloat16, False),
        (FixedPoint(16, 4), torch.float16, True),
        (FixedPoint(32, 8), torch.float64, False),
    ],
)
def test_a_result_is_wide_past_the_dtypes_precision(acc, dtype, wide):
    assert _fixedpoint_gemm_wide(acc, dtype) == wide


# --- the ops (on the CPU) ------------------------------------------------------------

SPLIT = torch.ops.mptorch.custom_matmul_fixedpoint.default
FUSED = torch.ops.mptorch.custom_matmul_fixedpoint_fma.default
PRECISION = {torch.float64: 53, torch.float32: 24, torch.float16: 11, torch.bfloat16: 8}


def _fields(f: FixedPoint) -> list:
    return [f.wl, f.fl, f.is_signed, f.symmetric]


def _op(
    a, b, *, acc, mul=None, mode=R.RNE, wrap=False, integer=False, trans_a=False, trans_b=False
):
    """One of the two ops: the split mac when ``mul`` is given."""
    if mul is None:
        return FUSED(a, b, trans_a, trans_b, *_fields(acc), mode.value, acc.prng_bits, wrap)
    return SPLIT(
        a, b, trans_a, trans_b, *_fields(mul), *_fields(acc), mode.value, mul.prng_bits,
        acc.prng_bits, wrap, integer,
    )  # fmt: skip


def _operands(shape_a, shape_b, dtype, seed, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(*shape_a, generator=g, dtype=torch.float64) * scale).to(dtype)
    b = (torch.randn(*shape_b, generator=g, dtype=torch.float64) * scale).to(dtype)
    return a, b


def _same_words(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Equal values, NaN where the other has NaN."""
    nan = got.isnan()
    return torch.equal(nan, want.isnan()) and torch.equal(got[~nan].double(), want[~nan].double())


# name: (mul, acc, wrap); mul None is the fused mac
OP_CASES = {
    "carrier path": (FixedPoint(16, 8), FixedPoint(16, 4), False),
    "integer and wide in binary32": (FixedPoint(16, 8), FixedPoint(32, 8), False),
    "unsigned": (FixedPoint(10, 4, is_signed=False), FixedPoint(16, 2, is_signed=False), False),
    "symmetric": (FixedPoint(16, 8, symmetric=True), FixedPoint(14, 6, symmetric=True), False),
    "saturating midway": (FixedPoint(16, 4), FixedPoint(8, 2), False),
    "wrapping": (FixedPoint(16, 4), FixedPoint(8, 2), True),
    "a coarse sum": (FixedPoint(16, 8), FixedPoint(12, -2), False),
    "fused": (None, FixedPoint(16, 4), False),
    "fused unsigned": (None, FixedPoint(12, 6, is_signed=False), False),
    "fused wrapping": (None, FixedPoint(8, 2), True),
    "fused and wide": (None, FixedPoint(40, 12), False),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", list(OP_CASES))
@pytest.mark.parametrize("mode", DETERMINISTIC)
def test_the_op_computes_the_definition(case, mode, dtype):
    mul, acc, wrap = OP_CASES[case]
    a, b = _operands((5, 37), (37, 6), dtype, seed=len(case))
    got = _op(a, b, acc=acc, mul=mul, mode=mode, wrap=wrap)
    assert got.dtype == (torch.float64 if acc.mag_bits > PRECISION[dtype] else dtype)
    assert _same_words(got, ref.gemm(a, b, acc=acc, mul=mul, mode=mode, wrap=wrap))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["carrier path", "saturating midway", "fused", "fused wrapping"])
@pytest.mark.parametrize("mode", [R.RNE, R.RD])
def test_narrow_operands_compute_in_binary32(case, mode, dtype):
    mul, acc, wrap = OP_CASES[case]
    a, b = _operands((5, 37), (37, 6), dtype, seed=11)
    got = _op(a, b, acc=acc, mul=mul, mode=mode, wrap=wrap)
    assert got.dtype == (torch.float64 if acc.mag_bits > PRECISION[dtype] else dtype)
    assert _same_words(got, ref.gemm(a, b, acc=acc, mul=mul, mode=mode, wrap=wrap))


@pytest.mark.parametrize("case", ["carrier path", "wrapping", "fused"])
def test_transposed_and_batched_operands(case):
    mul, acc, wrap = OP_CASES[case]
    a, b = _operands((2, 37, 5), (6, 37), torch.float32, seed=12)
    got = _op(a, b, acc=acc, mul=mul, mode=R.RZ, wrap=wrap, trans_a=True, trans_b=True)
    assert got.shape == (2, 5, 6)
    for i in range(2):
        want = ref.gemm(a[i].T, b.T, acc=acc, mul=mul, mode=R.RZ, wrap=wrap)
        assert _same_words(got[i], want)


@pytest.mark.parametrize("integer", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_nan_and_infinite_products(fused, integer):
    """A NaN operand, or an infinity times zero, makes its element NaN; an
    infinite product saturates the sum on its side."""
    if fused and integer:
        pytest.skip("a fused mac always runs the integer accumulator")
    a, b = _operands((4, 9), (9, 3), torch.float32, seed=13)
    a[0, 2], a[1, 4], a[2, 1] = math.nan, math.inf, -math.inf
    b[1, 0] = 0.0  # inf * 0 in element (2, 0)
    mul = None if fused else FixedPoint(16, 8)
    acc = FixedPoint(16, 4)
    got = _op(a, b, acc=acc, mul=mul, integer=integer)
    assert _same_words(got, ref.gemm(a, b, acc=acc, mul=mul))
    assert got[0].isnan().all() and got[2, 0].isnan()


# Format pairs the rules give the carrier path, with random bits for SR.
CARRIER_PAIRS = {
    torch.float32: (FixedPoint(16, 8, prng_bits=6), FixedPoint(16, 4, prng_bits=6)),
    torch.float64: (FixedPoint(24, 12, prng_bits=9), FixedPoint(40, 8, prng_bits=9)),
    torch.bfloat16: (FixedPoint(16, 8, prng_bits=6), FixedPoint(8, 2, prng_bits=6)),
}


@pytest.mark.parametrize("dtype", list(CARRIER_PAIRS))
@pytest.mark.parametrize("mode", DETERMINISTIC + STOCHASTIC)
def test_the_carrier_path_is_the_integer_path(mode, dtype):
    """The test hook runs the integer accumulator where the rules choose the
    carrier path: the words are the same, stochastic rounding's included,
    since both draw the same words and read them the same way."""
    mul, acc = CARRIER_PAIRS[dtype]
    carrier = torch.float64 if dtype is torch.float64 else torch.float32
    assert _fixedpoint_gemm(mul, acc, False, carrier).in_carrier
    assert acc.mag_bits <= PRECISION[dtype]  # not wide, which runs the integer path
    a, b = _operands((9, 70), (70, 11), dtype, seed=14, scale=6.0)
    results = []
    for integer in (False, True):
        torch.manual_seed(15)
        results.append(_op(a, b, acc=acc, mul=mul, mode=mode, integer=integer))
    assert torch.equal(results[0], results[1])


@pytest.mark.parametrize(
    "mul, acc",
    [
        (FixedPoint(16, 8), FixedPoint(23, 4)),  # 22 magnitude bits: the fast RNE cast
        (FixedPoint(16, 8), FixedPoint(24, 4)),  # 23: the bitwise cast
        (FixedPoint(23, 8), FixedPoint(16, 4)),  # the multiply format at the gate
        (FixedPoint(8, 0), FixedPoint(24, 0)),  # a span of 24: the carrier path
        (FixedPoint(8, 0), FixedPoint(25, 0)),  # 25: the integer accumulator
    ],
)
def test_the_boundaries_compute_the_definition(mul, acc):
    a, b = _operands((4, 40), (40, 5), torch.float32, seed=16, scale=40.0)
    for mode in (R.RNE, R.RU):
        got = _op(a, b, acc=acc, mul=mul, mode=mode)
        assert _same_words(got, ref.gemm(a, b, acc=acc, mul=mul, mode=mode))


def test_a_sum_binary32_cannot_hold_is_exact():
    """The negative control of the carrier's sums, through the op: a pair
    the rules send to the integer accumulator keeps the 2^-8 that binary32's
    sum at 65536 drops, in a float64 result."""
    mul, acc = FixedPoint(24, 8), FixedPoint(32, 8)
    a = torch.tensor([[128.0, 128.0, 128.0, 128.0, 0.0625]])
    got = _op(a, a.T.clone(), acc=acc, mul=mul)
    assert got.dtype == torch.float64 and got.item() == 65536 + 2**-8


@pytest.mark.parametrize("mode", [R.RNE, R.RZ])
def test_past_the_carriers_span_the_sums_stay_exact(mode):
    """A pair the rules send to the integer accumulator although its result
    fits float32 (span 32, 23 magnitude bits): products near the multiply
    format's top, of one sign, make sums whose 8 fractional bits binary32
    cannot keep beside their integer part. The op gives the definition's
    words; the shipped casts looped in the carrier do not."""
    mul, acc = FixedPoint(22, 8), FixedPoint(24, 0)
    assert _fixedpoint_span(mul, acc) == 32 and acc.mag_bits <= 24
    g = torch.Generator().manual_seed(20)
    top = math.sqrt(2.0 ** (mul.mag_bits - mul.fl))
    a = top * (0.5 + torch.randn(8, 96, generator=g).abs() / 8)
    b = top * (0.5 + torch.randn(96, 8, generator=g).abs() / 8)
    want = ref.gemm(a, b, acc=acc, mul=mul, mode=mode)
    got = _op(a, b, acc=acc, mul=mul, mode=mode)
    assert got.dtype == torch.float32 and _same_words(got, want)
    assert not torch.equal(_carrier_loop(a, b, mul, acc, mode), want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_a_wide_result_is_float64_and_exact(dtype):
    a = _on_grid((6, 50), FixedPoint(8, 2), 17).to(dtype)
    b = _on_grid((50, 7), FixedPoint(8, 2), 18).to(dtype)
    for mul in (FixedPoint(16, 4), None):
        got = _op(a, b, acc=FixedPoint(40, 4), mul=mul)
        assert got.dtype == torch.float64
        assert torch.equal(got, a.double() @ b.double())


def _up_probability(eta: Fraction, n: int, mode: RoundMode) -> Fraction:
    """P3109's probability of rounding a fraction ``eta`` away from zero."""
    return Fraction(sum(ref._stochastic_up(eta, r, n, mode) for r in range(2**n)), 2**n)


@pytest.mark.parametrize("integer", [False, True, None])  # None: the fused mac
@pytest.mark.parametrize("mode", STOCHASTIC)
def test_stochastic_rounding_has_p3109s_mean(mode, integer):
    """One product, 23/64 of a step above a code, rounded in 4,096 elements,
    each with its own stream: the count rounded up is within five standard
    deviations of P3109's probability. At 4 random bits that is 5/16 for
    StochasticA and 6/16 for B and C, 256 counts apart, about eight standard
    deviations."""
    n, fl = 4, 4
    fmt = FixedPoint(16, fl, prng_bits=n)
    p = Fraction(52 * 64 + 23, 64 * 2**fl)  # 52 steps and 23/64 of one
    eta = (p * 2**fl) - math.floor(p * 2**fl)
    assert eta == Fraction(23, 64)
    a = torch.full((64, 1), float(p))
    b = torch.ones(1, 64)
    torch.manual_seed(19)
    if integer is None:
        got = _op(a, b, acc=fmt, mode=mode)
    else:
        got = _op(a, b, acc=fmt, mul=fmt, mode=mode, integer=integer)
    lo = math.floor(p * 2**fl) / 2**fl
    ups = int((got.double() > lo).sum())
    assert set(got.double().unique().tolist()) <= {lo, lo + 2**-fl}
    prob = float(_up_probability(eta, n, mode))
    mean, sd = 4096 * prob, math.sqrt(4096 * prob * (1 - prob))
    assert abs(ups - mean) <= 5 * sd, (ups, mean, sd)


def test_an_empty_call_allocates_the_result_and_draws_nothing():
    before = torch.get_rng_state()
    got = _op(
        torch.zeros(0, 5),
        torch.zeros(5, 3),
        acc=FixedPoint(40, 4),
        mul=FixedPoint(16, 4),
        mode=R.SR,
    )
    assert got.shape == (0, 3) and got.dtype == torch.float64
    assert torch.equal(torch.get_rng_state(), before)


class _Raw(NamedTuple):
    """A format's fields as a direct call passes them, for formats that
    `FixedPoint` itself refuses (no carrier holds them)."""

    wl: int
    fl: int
    is_signed: bool = True
    symmetric: bool = False
    prng_bits: int = 0


_FP = FixedPoint
# (mul, acc, wrap, mode, dtype, message); mul None is the fused mac
OP_REFUSALS = {
    "mul width": (
        _FP(40, 8),
        _FP(16, 4),
        False,
        R.RNE,
        F32,
        "39 magnitude bits, but binary32 holds 1 to 24",
    ),
    "mul end": (_FP(16, -120), _FP(16, 4), False, R.RNE, F32, "past binary32's range"),
    "mul step": (_FP(8, 126), _FP(16, 4), False, R.RNE, F32, r"step is 2\^-126, below 2\^-125"),
    "mul prng": (
        _FP(16, 8, prng_bits=12),
        _FP(16, 4),
        False,
        R.SR,
        F32,
        r"prng_bits must be in \[0, 9\]",
    ),
    "acc width": (
        _FP(16, 8),
        _Raw(60, 4),
        False,
        R.RNE,
        F32,
        "59 magnitude bits, and a fixed-point GEMM accumulates 1 to 53",
    ),
    "acc values": (_FP(16, 8), _Raw(16, 1100), False, R.RNE, torch.float64, "not float64 values"),
    "acc prng": (_FP(16, 8), _FP(16, 4, prng_bits=30), False, R.SR, F32, "draws 30 random bits"),
    "wrap symmetric": (
        _FP(16, 8),
        _FP(16, 4, symmetric=True),
        True,
        R.RNE,
        F32,
        "no modulus to wrap",
    ),
    "span": (_FP(16, 60), _FP(40, -10), False, R.RNE, torch.float64, "needs 110 bits"),
    "fused span": (None, _Raw(54, 4, prng_bits=20), False, R.SR, torch.float64, "a fused step"),
}


@pytest.mark.parametrize("case", list(OP_REFUSALS))
def test_the_op_refuses_what_it_cannot_compute(case):
    mul, acc, wrap, mode, dtype, message = OP_REFUSALS[case]
    a, b = torch.ones(2, 3, dtype=dtype), torch.ones(3, 2, dtype=dtype)
    with pytest.raises(RuntimeError, match=message):
        _op(a, b, acc=acc, mul=mul, mode=mode, wrap=wrap)
