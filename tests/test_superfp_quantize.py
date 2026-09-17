"""
superfp_quantize on a format small enough to derive every expected value by hand.

superfp is a binade-coded format with three regions (see
mptorch/csrc/common/cast_superfp.h): the top ``normal_binades`` encoding
binades carry a ``man_bits`` significand, the encoding binades below them are
spent one code per power of two (the "supernormal" region, which trades
subnormals for range), and everything under the smallest supernormal is a
choice between zero and that value. The reference format here has 1 sign, 3
exponent and 2 mantissa bits, 4 normal binades and bias 7, so its 64 code
points are a normal grid from 0.125 to 1.75 in steps of 2**e / 4, fifteen
supernormal levels 2**-18 .. 2**-4, and zero. That is few enough values that
each expectation table below is written out from the rounding mode's
definition rather than from another implementation, which is what makes it
independent of the kernel. Each region has its own tie rule and its own table:
the normal region ties on the significand's parity, the supernormal region on
the exponent's, and the underflow region has one tie, at half the smallest
supernormal. The last tests hold the saturation modes at the top of the range
and stochastic rounding's properties, including in the underflow region,
which binaryK does not have.
"""

import math
from typing import Any

import pytest
import torch

from mptorch.number import RoundMode, SaturationMode
from mptorch.quant import superfp_quantize
from tests.markers import available_devices

# The reference format: 1 sign bit, 3 exponent bits, 2 mantissa bits, 4 normal
# binades, bias 7. Exponent fields 4 .. 7 are the normal binades, exponents -3
# .. 0, a grid of 32 codes from 0.125 to 1.75 in steps of 2**e / 4; fields
# 0 .. 3 hold 16 codes, zero and the fifteen supernormal powers of two 2**-18
# .. 2**-4 (dev/benchmarks/superfp_values.py lists them).
CFG: dict[str, Any] = {"man_bits": 2, "exp_bits": 3, "normal_binades": 4, "bias": 7}
SMALLEST_SUPERNORMAL = 2.0**-18  # the lowest supernormal level
HALF_SMALLEST = 2.0**-19  # the tie between zero and SMALLEST_SUPERNORMAL

ALL_MODES = [
    RoundMode.RNE,
    RoundMode.RNA,
    RoundMode.RU,
    RoundMode.RD,
    RoundMode.RZ,
    RoundMode.RO,
]


def _quantize(
    x_val,
    mode,
    device,
    dtype=torch.float32,
    is_signed=True,
    saturation_mode=SaturationMode.SAT_FINITE,
    cfg=CFG,
):
    """One value through ``superfp_quantize`` in the format ``cfg``, as a
    Python float. SAT_FINITE is the default so the whole top binade is in
    play; the saturation tests name the mode explicitly."""
    x = torch.tensor([x_val], dtype=dtype, device=device)
    return superfp_quantize(
        x,
        is_signed=is_signed,
        rounding_mode=mode,
        saturation_mode=saturation_mode,
        **cfg,
    ).item()


def _same(got: float, want: float) -> bool:
    """Equal, and for a zero equally signed: `-0.0 == 0.0` is true, and superfp's
    only zero is +0.0."""
    return got == want and math.copysign(1.0, got) == math.copysign(1.0, want)


def _negated(value: float) -> float:
    """The rounding of -x, given that of x: the sign flips, except on zero."""
    return -value if value != 0.0 else 0.0


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_nan_passthrough(device, mode):
    """A NaN comes back with its word intact, payload and sign included, in
    every rounding mode: the cast must pass it through, not canonicalize it."""
    import struct

    # A signalling NaN (quiet bit clear, e.g. 0x7FA51234) is left out: torch
    # quiets it in plain tensor construction, before any op runs, which
    # `torch.tensor([nan]).item()` alone shows, so it says nothing about the
    # cast.
    for bits in (0x7FC00000, 0x7FE51234, 0xFFC00000):
        nan = struct.unpack(">f", struct.pack(">I", bits))[0]
        out = _quantize(nan, mode, device)
        assert struct.unpack(">I", struct.pack(">f", out))[0] == bits


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize(
    "saturation_mode",
    [SaturationMode.SAT_FINITE, SaturationMode.SAT_PROPAGATE, SaturationMode.OVF_INF],
)
def test_inf_handling(device, mode, saturation_mode):
    """An infinity is kept under OVF_INF and SAT_PROPAGATE; under SAT_FINITE,
    which promises a finite result for every input, it saturates to the same
    value an overflowing finite input does."""
    for sign in (1.0, -1.0):
        inf = sign * float("inf")
        out = _quantize(inf, mode, device, saturation_mode=saturation_mode)
        if saturation_mode is SaturationMode.SAT_FINITE:
            huge = _quantize(sign * 3.0e38, mode, device, saturation_mode=saturation_mode)
            assert math.isfinite(out)
            assert out == huge
        else:
            assert out == inf


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_zero(device, mode):
    """Both zeros come back as +0.0: superfp's only zero is unsigned."""
    assert _same(_quantize(0.0, mode, device), 0.0)
    assert _same(_quantize(-0.0, mode, device), 0.0)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_unsigned_rejects_negative(device, mode):
    """An unsigned format maps every negative input, -0.0 included, to +0.0."""
    for x in (-3.5, -1.0e-7, -0.0):
        assert _same(_quantize(x, mode, device, is_signed=False), 0.0), x


# ---------------------------------------------------------------------------
# Normal region: man_bits = 2 puts four codes in each binade, at a step of
# 2**e / 4. 0.13 lies between the grid points 0.125 and 0.15625 and is not a
# tie; 0.140625 is their exact midpoint. These are ordinary IEEE 754 roundings
# (RNE breaks the tie on the parity of the mantissa's last bit), so each
# expectation follows from the mode's definition.
NORMAL_CASES = [
    # (value, mode, expected)
    (0.13, RoundMode.RNE, 0.125),
    (0.13, RoundMode.RNA, 0.125),
    (0.13, RoundMode.RU, 0.15625),
    (0.13, RoundMode.RD, 0.125),
    (0.13, RoundMode.RZ, 0.125),
    (0.13, RoundMode.RO, 0.15625),
    (-0.13, RoundMode.RNE, -0.125),
    (-0.13, RoundMode.RNA, -0.125),
    (-0.13, RoundMode.RU, -0.125),  # RU(-x) == -RD(x)
    (-0.13, RoundMode.RD, -0.15625),  # RD(-x) == -RU(x)
    (-0.13, RoundMode.RZ, -0.125),
    (-0.13, RoundMode.RO, -0.15625),
    (0.140625, RoundMode.RNE, 0.125),  # tie -> even mantissa (00)
    (0.140625, RoundMode.RNA, 0.15625),  # tie -> away from zero
    (0.140625, RoundMode.RU, 0.15625),
    (0.140625, RoundMode.RD, 0.125),
    (0.140625, RoundMode.RZ, 0.125),
    (0.140625, RoundMode.RO, 0.15625),  # inexact, LSB forced to 1
    (-0.140625, RoundMode.RNE, -0.125),
    (-0.140625, RoundMode.RNA, -0.15625),
    (-0.140625, RoundMode.RU, -0.125),
    (-0.140625, RoundMode.RD, -0.15625),
    (-0.140625, RoundMode.RZ, -0.125),
    (-0.140625, RoundMode.RO, -0.15625),
]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("value,mode,expected", NORMAL_CASES)
def test_normal_region_rounding(device, value, mode, expected):
    assert _quantize(value, mode, device) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Supernormal region: the codes are bare powers of two, so rounding chooses
# between 2**e and 2**(e + 1). 1.2 * 2**-9 is off the tie; 1.5 * 2**-9 and
# 1.5 * 2**-10 are exact ties. With no significand to be even, RNE's tie rule
# is the even exponent: the zero-argument round_bitwise_nearest_even of
# mptorch/csrc/common/bit_helper.h, which binaryK's P = 1 formats share and
# tests/test_binaryk_quantize.py checks against gfloat. RO's "odd" is the
# parity of the code's index from the bottom of the region (cast_superfp_odd
# in cast_superfp.h), so 2**-10, the ninth level, is already odd.
SUPERNORMAL_CASES = [
    (1.2 * 2**-9, RoundMode.RNE, 2**-9),
    (1.2 * 2**-9, RoundMode.RNA, 2**-9),
    (1.2 * 2**-9, RoundMode.RU, 2**-8),
    (1.2 * 2**-9, RoundMode.RD, 2**-9),
    (1.2 * 2**-9, RoundMode.RZ, 2**-9),
    (1.2 * 2**-9, RoundMode.RO, 2**-8),
    (1.5 * 2**-9, RoundMode.RNE, 2**-8),  # tie: -8 is even -> ceil wins
    (1.5 * 2**-9, RoundMode.RNA, 2**-8),  # tie -> away from zero
    (1.5 * 2**-9, RoundMode.RU, 2**-8),
    (1.5 * 2**-9, RoundMode.RD, 2**-9),
    (1.5 * 2**-9, RoundMode.RZ, 2**-9),
    (1.5 * 2**-9, RoundMode.RO, 2**-8),
    (1.5 * 2**-10, RoundMode.RNE, 2**-10),  # tie: -10 is even -> floor wins
    (1.5 * 2**-10, RoundMode.RNA, 2**-9),
    (1.5 * 2**-10, RoundMode.RU, 2**-9),
    (1.5 * 2**-10, RoundMode.RD, 2**-10),
    (1.5 * 2**-10, RoundMode.RZ, 2**-10),
    (1.5 * 2**-10, RoundMode.RO, 2**-10),  # floor's stored index already odd
]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("value,mode,expected", SUPERNORMAL_CASES)
def test_supernormal_region_rounding(device, value, mode, expected):
    """The table, and its mirror image for the sign-symmetric modes."""
    assert _quantize(value, mode, device) == pytest.approx(expected)
    # The sign-symmetric modes mirror on negation; RU and RD swap roles
    # instead, which test_underflow_boundary checks.
    if mode in (RoundMode.RNE, RoundMode.RNA, RoundMode.RZ, RoundMode.RO):
        assert _quantize(-value, mode, device) == pytest.approx(-expected)


# ---------------------------------------------------------------------------
# Underflow region: below the smallest supernormal there is no finer grid, so
# every mode chooses between 0 and SMALLEST_SUPERNORMAL, with the tie at
# HALF_SMALLEST = 2**-19. RU and RO never flush a nonzero value; RNE sends the
# tie to zero, the "even" side, and RNA away from it.
UNDERFLOW_CASES = [
    # (value, mode, expected)
    (2**-20, RoundMode.RNE, 0.0),  # strictly below half -> always flush
    (2**-20, RoundMode.RNA, 0.0),
    (2**-20, RoundMode.RU, SMALLEST_SUPERNORMAL),  # RU never flushes nonzero
    (2**-20, RoundMode.RD, 0.0),
    (2**-20, RoundMode.RZ, 0.0),
    (2**-20, RoundMode.RO, SMALLEST_SUPERNORMAL),  # RO never flushes nonzero
    (HALF_SMALLEST, RoundMode.RNE, 0.0),  # exact tie -> ties-to-even = zero
    (HALF_SMALLEST, RoundMode.RNA, SMALLEST_SUPERNORMAL),  # tie -> away from zero
    (HALF_SMALLEST, RoundMode.RU, SMALLEST_SUPERNORMAL),
    (HALF_SMALLEST, RoundMode.RD, 0.0),
    (HALF_SMALLEST, RoundMode.RZ, 0.0),
    (HALF_SMALLEST, RoundMode.RO, SMALLEST_SUPERNORMAL),
    (1.5 * HALF_SMALLEST, RoundMode.RNE, SMALLEST_SUPERNORMAL),  # strictly above half
    (1.5 * HALF_SMALLEST, RoundMode.RNA, SMALLEST_SUPERNORMAL),
    (1.5 * HALF_SMALLEST, RoundMode.RU, SMALLEST_SUPERNORMAL),
    (1.5 * HALF_SMALLEST, RoundMode.RD, 0.0),
    (1.5 * HALF_SMALLEST, RoundMode.RZ, 0.0),
    (1.5 * HALF_SMALLEST, RoundMode.RO, SMALLEST_SUPERNORMAL),
]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("value,mode,expected", UNDERFLOW_CASES)
def test_underflow_boundary(device, value, mode, expected):
    """The table for x and for -x, compared with the sign: every value here
    is a power of two or zero, and a zero must come back as +0.0."""
    assert _same(_quantize(value, mode, device), expected)
    if mode in (RoundMode.RNE, RoundMode.RNA, RoundMode.RZ, RoundMode.RO):
        assert _same(_quantize(-value, mode, device), _negated(expected))
    elif mode is RoundMode.RU:
        # RU(-x) == -RD(x), and RD(x) is zero for every case here, so +0.0.
        assert _same(_quantize(-value, mode, device), 0.0)
    elif mode is RoundMode.RD:
        # RD(-x) == -RU(x)
        assert _same(_quantize(-value, mode, device), -_quantize(value, RoundMode.RU, device))


# ---------------------------------------------------------------------------
# RO in the normal region of a man_bits = 0 format is a branch of its own in
# cast_superfp_odd (mptorch/csrc/common/cast_superfp.h): with no significand,
# "odd" is the parity of the target's biased exponent, which is neither the
# significand parity of the man_bits > 0 branch nor the index parity of the
# supernormal branch. exp_bits 4 with bias 10 leaves binades above 8.0, so a
# bump up cannot saturate, and normal_binades 6 keeps 4.0 .. 8.0 normal. 6.0
# is the tie between 4.0 and 8.0: RNE keeps 4.0, whose exponent 2 is even, the
# same rule as the supernormal region since this grid is power-of-two spaced
# too. 5.0 is not a tie, but RO still bumps it to 8.0 because 4.0's biased
# exponent, 2 + 10 = 12, is even.
RO_ZERO_MAN_BITS_CFG: dict[str, Any] = dict(man_bits=0, exp_bits=4, normal_binades=6, bias=10)


@pytest.mark.parametrize("device", available_devices)
def test_ro_normal_region_zero_man_bits(device):
    """RO with no significand bits bumps on the biased exponent's parity."""
    assert _quantize(6.0, RoundMode.RNE, device, cfg=RO_ZERO_MAN_BITS_CFG) == 4.0
    assert _quantize(6.0, RoundMode.RO, device, cfg=RO_ZERO_MAN_BITS_CFG) == 8.0
    assert _quantize(5.0, RoundMode.RO, device, cfg=RO_ZERO_MAN_BITS_CFG) == 8.0


# ---------------------------------------------------------------------------
# The saturation modes at the overflow boundary, in every rounding mode. Under
# OVF_INF the top code of the reference format, 31 (011111), is +Inf, so the
# largest finite value is 1.5 (code 30); SAT_FINITE keeps the whole top binade
# and its largest finite value is 1.75 (code 31).
@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_saturation_mode_differentiation(device, mode):
    """The largest finite value and the fate of an overflow differ between
    SAT_FINITE and OVF_INF, whichever the rounding mode."""
    assert _quantize(1.5, mode, device, saturation_mode=SaturationMode.SAT_FINITE) == 1.5
    assert _quantize(1.5, mode, device, saturation_mode=SaturationMode.OVF_INF) == 1.5

    assert _quantize(1.75, mode, device, saturation_mode=SaturationMode.SAT_FINITE) == 1.75
    assert _quantize(1.75, mode, device, saturation_mode=SaturationMode.OVF_INF) == float("inf")

    assert _quantize(2.0, mode, device, saturation_mode=SaturationMode.SAT_FINITE) == 1.75
    assert _quantize(2.0, mode, device, saturation_mode=SaturationMode.OVF_INF) == float("inf")

    assert _quantize(-5.0, mode, device, saturation_mode=SaturationMode.SAT_FINITE) == -1.75
    assert _quantize(-5.0, mode, device, saturation_mode=SaturationMode.OVF_INF) == float("-inf")


# The same boundary where the reserved code is a large part of the top binade.
# Outside SAT_FINITE the top binade's last code is +Inf, so with man_bits = 0
# (one code per binade) the largest finite value is a binade down and with
# man_bits = 1 it is the binade's first code; SAT_FINITE keeps the whole
# binade. `above` is the first code past `largest`, which has to saturate like
# any other overflow, as does 3e38, which the rounding carries into float32's
# infinity exponent. exp_bits 4 and bias 7 put the top binade at 2**8;
# normal_binades 8 keeps every value here in the normal region.
TOP_CASES = [
    # (man_bits, saturation_mode, largest, above)
    (0, SaturationMode.OVF_INF, 2.0**7, 2.0**8),
    (0, SaturationMode.SAT_PROPAGATE, 2.0**7, 2.0**8),
    (0, SaturationMode.SAT_FINITE, 2.0**8, 2.0**9),
    (1, SaturationMode.OVF_INF, 2.0**8, 1.5 * 2.0**8),
    (1, SaturationMode.SAT_PROPAGATE, 2.0**8, 1.5 * 2.0**8),
    (1, SaturationMode.SAT_FINITE, 1.5 * 2.0**8, 2.0**9),
    (3, SaturationMode.SAT_PROPAGATE, 1.75 * 2.0**8, 1.875 * 2.0**8),
]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("man_bits,saturation_mode,largest,above", TOP_CASES)
def test_top_of_range(device, mode, man_bits, saturation_mode, largest, above):
    """`largest` is kept and everything above it saturates, to the mode's
    value, with the sign, in every rounding mode."""
    cfg = {"man_bits": man_bits, "exp_bits": 4, "normal_binades": 8, "bias": 7}
    kw = dict(saturation_mode=saturation_mode, cfg=cfg)
    saturated = math.inf if saturation_mode is SaturationMode.OVF_INF else largest
    assert _quantize(largest, mode, device, **kw) == largest
    for x in (above, 3.0e38):
        assert _quantize(x, mode, device, **kw) == saturated, x
        assert _quantize(-x, mode, device, **kw) == -saturated, -x


# ---------------------------------------------------------------------------
# Stochastic rounding: the three properties tests/test_binaryk_quantize.py
# checks for binaryK (a grid point is left alone, every result is one of the
# two neighbours, the mean is the input), plus unbiasedness in the underflow
# region, which has no binaryK analogue because binaryK's subnormals keep a
# graded grid all the way down to zero.
GRID_POINTS = (
    [0.0]
    + [2.0**e for e in range(-18, -3)]  # the supernormal levels
    + [(1 + m / 4) * 2.0**e for e in range(-3, 1) for m in range(4)]  # the normal grid
)


@pytest.mark.parametrize("device", available_devices)
def test_superfp_stochastic(device):
    """SR leaves the format's every code point alone, lands on a neighbour of
    every other input, and is unbiased in both the normal and the underflow
    region."""
    # one random bit per significand bit the format drops from binary32's 23
    prng_bits = 23 - CFG["man_bits"]

    grid = torch.tensor(GRID_POINTS + [-v for v in GRID_POINTS], dtype=torch.float32, device=device)
    q_sr_grid = superfp_quantize(
        grid,
        prng_bits=prng_bits,
        is_signed=True,
        rounding_mode=RoundMode.SR,
        saturation_mode=SaturationMode.SAT_FINITE,
        **CFG,
    )
    assert torch.all(q_sr_grid == grid), "SR altered an exactly representable grid point"

    x_rand = (torch.rand(10000, dtype=torch.float32, device=device) * 1.8 - 0.9).requires_grad_(
        False
    )
    q_rd = superfp_quantize(
        x_rand,
        is_signed=True,
        rounding_mode=RoundMode.RD,
        saturation_mode=SaturationMode.SAT_FINITE,
        **CFG,
    )
    q_ru = superfp_quantize(
        x_rand,
        is_signed=True,
        rounding_mode=RoundMode.RU,
        saturation_mode=SaturationMode.SAT_FINITE,
        **CFG,
    )
    q_sr = superfp_quantize(
        x_rand,
        prng_bits=prng_bits,
        is_signed=True,
        rounding_mode=RoundMode.SR,
        saturation_mode=SaturationMode.SAT_FINITE,
        **CFG,
    )
    valid_bounds = (q_sr == q_rd) | (q_sr == q_ru)
    assert torch.all(valid_bounds), "SR produced a value outside the [RD, RU] bounds"

    N_samples = 1_000_000
    tolerance = 0.05

    # Unbiasedness in the normal region: a million draws of 0.14, which lies
    # between 0.125 and 0.15625. 0.05 is loose against the standard error, so
    # only a systematic bias fails it.
    test_val = 0.14
    x_large = torch.full((N_samples,), test_val, dtype=torch.float32, device=device)
    q_sr_large = superfp_quantize(
        x_large,
        prng_bits=prng_bits,
        is_signed=True,
        rounding_mode=RoundMode.SR,
        saturation_mode=SaturationMode.SAT_FINITE,
        **CFG,
    )
    assert abs(q_sr_large.mean().item() - test_val) < tolerance

    # Unbiasedness in the underflow region, where the choice is between zero
    # and the smallest supernormal: the mean of the draws is 0.3 of that
    # level, to a tolerance scaled by it.
    test_val_uf = SMALLEST_SUPERNORMAL * 0.3
    x_uf = torch.full((N_samples,), test_val_uf, dtype=torch.float32, device=device)
    q_sr_uf = superfp_quantize(
        x_uf,
        prng_bits=prng_bits,
        is_signed=True,
        rounding_mode=RoundMode.SR,
        saturation_mode=SaturationMode.SAT_FINITE,
        **CFG,
    )
    assert abs(q_sr_uf.mean().item() - test_val_uf) < tolerance * SMALLEST_SUPERNORMAL
