"""
fixedpoint_quantize on a format small enough to derive every expected value by hand.

A fixed-point format holds the multiples of the step ``2**-fl`` that a ``wl``-bit
integer holds (see mptorch/csrc/common/cast_fixedpoint.h). The reference format
here is signed, ``wl=4, fl=2``: sixteen codes, -2 .. 1.75 in steps of 0.25. That
is few enough values that each expectation table below is written out from the
rounding mode's definition rather than from another implementation, which is
what makes it independent of the kernel. The interior ties on the parity of the
integer code, below the step the two candidates are zero and the step itself,
and past either end the result saturates. The round-to-nearest-even tables
come first; the other deterministic modes follow, one row per mode.

Every table is checked a second way by a sweep against an exact reference, the
rounding done in Python's integers over every float16 value and random float32
and float64 ones, which plays the part gfloat plays for binaryK; and the grid
points and ties at both ends of each format are swept too, in binary32 and in
binary64, along with the formats on binary32's edges.
Stochastic rounding cannot be held to a value, so it is held to its properties
instead, as binaryK's and superfp's are: grid points are left alone, every
result is one of the input's two neighbours, the mean is the input, and
``prng_bits`` sets the resolution. Every test runs on the CPU, CUDA and MPS,
except where it needs float64, which MPS has no tensors of;
tests/test_cast_fast_paths.py and tests/test_mps.py hold the backends to the
CPU's bits. The in-place op, the carrier, strided inputs and the seed are
tested with the other quantizers', in tests/test_quantize_inplace.py,
tests/test_quantize_dispatch.py and tests/test_sr_rng.py.
"""

import math
import random
import warnings

import pytest
import torch

from mptorch.number import FormatRangeWarning, RoundMode
from mptorch.quant import fixedpoint_quantize
from tests.markers import available_devices, float64_devices, has_float64

# Every backend, each skipped where there is no such device. MPS has no float64
# tensors: a test parametrized over a float64 dtype is skipped there by
# conftest.py, and one that needs float64 whatever its parameters takes
# float64_devices.
DEVICES = available_devices

# The reference format: 1 sign bit and 3 magnitude bits, 2 of them fractional.
# Codes -8 .. 7 hold -2 .. 1.75 in steps of 0.25.
CFG = {"wl": 4, "fl": 2}
STEP = 0.25
MAX = 1.75
MIN = -2.0

# The deterministic modes, which the tables and the reference hold to values.
ALL_MODES = [RoundMode.RNE, RoundMode.RNA, RoundMode.RU, RoundMode.RD, RoundMode.RZ, RoundMode.RO]
# And stochastic rounding, for the tests whose expectations hold for every draw.
EVERY_MODE = ALL_MODES + [RoundMode.SR]

# The random bits SR draws where a test does not choose its own: enough to
# reach the round-up as well as the round-down for every input below.
SR_BITS = 8


def _quantize(x_val, device, dtype=torch.float32, cfg=CFG, mode=RoundMode.RNE, **kw):
    """One value through ``fixedpoint_quantize``, as a Python float. Under SR it
    draws ``SR_BITS`` random bits unless the call names ``prng_bits``."""
    if mode is RoundMode.SR:
        kw.setdefault("prng_bits", SR_BITS)
    x = torch.tensor([x_val], dtype=dtype, device=device)
    return fixedpoint_quantize(x, rounding_mode=mode, **cfg, **kw).item()


def _same(got: float, want: float) -> bool:
    """Equal, and for a zero equally signed: `-0.0 == 0.0` is true, and a
    fixed-point format's only zero is +0.0."""
    return got == want and math.copysign(1.0, got) == math.copysign(1.0, want)


# ---------------------------------------------------------------------------
# The interior: two neighbouring codes, and a tie between them goes to the even
# code. 0.375 is 1.5 steps, between codes 1 and 2, so it goes to 2 (0.5); 0.625
# is 2.5 steps and also goes to 2; 0.875 is 3.5 steps and goes to 4 (1.0).
INTERIOR_CASES = [
    # (value, expected)
    (0.25, 0.25),  # a grid point is left alone
    (0.3, 0.25),
    (0.4, 0.5),
    (0.375, 0.5),  # tie -> even code (2)
    (0.625, 0.5),  # tie -> even code (2)
    (0.875, 1.0),  # tie -> even code (4)
    (1.1, 1.0),
    (-0.3, -0.25),
    (-0.375, -0.5),  # tie -> even code (-2)
    (-0.625, -0.5),
    (-0.875, -1.0),
    (-1.1, -1.0),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value,expected", INTERIOR_CASES)
def test_interior_rounding(device, value, expected):
    assert _same(_quantize(value, device), expected)


# ---------------------------------------------------------------------------
# Below the step: the candidates are zero and the step. Half a step is a tie,
# and zero is the even code, so it takes it; anything above half goes up. The
# one zero is unsigned, so a negative value that rounds to it is +0.0.
BELOW_STEP_CASES = [
    (0.0, 0.0),
    (-0.0, 0.0),  # no -0.0 out
    (0.01, 0.0),
    (0.1, 0.0),
    (0.125, 0.0),  # half a step: tie -> zero, the even code
    (0.13, 0.25),
    (0.2, 0.25),
    (-0.1, 0.0),  # rounds to zero: +0.0, not -0.0
    (-0.125, 0.0),
    (-0.13, -0.25),
    (1e-30, 0.0),  # far below the step
    (-1e-30, 0.0),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value,expected", BELOW_STEP_CASES)
def test_below_the_step(device, value, expected):
    assert _same(_quantize(value, device), expected)


# ---------------------------------------------------------------------------
# The ends: a result past the range saturates to its end, whether the input was
# already past it or rounding carried it there. 1.875 is the tie between 1.75
# (code 7) and 2.0 (code 8, past the top), and goes to the even code 8, which
# saturates back to 1.75. The range is two's complement, so the bottom, -2, is
# one step further from zero than the top. An infinity saturates too, since a
# fixed-point word has no code for one.
ENDS_CASES = [
    (1.75, MAX),
    (1.8, MAX),
    (1.875, MAX),  # tie -> code 8, past the top -> saturates
    (1.9, MAX),
    (100.0, MAX),
    (1e30, MAX),
    (float("inf"), MAX),
    (-2.0, MIN),
    (-2.1, MIN),
    (-2.125, MIN),  # tie -> code -8, the bottom itself
    (-2.2, MIN),
    (-100.0, MIN),
    (-1e30, MIN),
    (float("-inf"), MIN),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", EVERY_MODE)
def test_ends_saturate(device, mode):
    """Every mode saturates the same way, SR in every draw: each rounds these
    values onto one end or past it, and past it is the end."""
    for value, expected in ENDS_CASES:
        got = _quantize(value, device, mode=mode)
        assert _same(got, expected), f"{value} -> {got}, expected {expected}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", EVERY_MODE)
def test_symmetric_drops_the_bottom_code(device, mode):
    """``symmetric`` drops code -8, so the bottom is -1.75 like the top."""
    for value in (-2.0, -1.9, -100.0, float("-inf")):
        assert _same(_quantize(value, device, mode=mode, symmetric=True), -MAX)
    assert _same(_quantize(-1.75, device, mode=mode, symmetric=True), -MAX)
    assert _same(_quantize(1.9, device, mode=mode, symmetric=True), MAX)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", EVERY_MODE)
def test_unsigned(device, mode):
    """Unsigned, the four bits are all magnitude: 0 .. 3.75, and every negative
    value is zero in every mode."""
    kw = dict(mode=mode, is_signed=False)
    assert _same(_quantize(3.25, device, **kw), 3.25)
    assert _same(_quantize(3.8, device, **kw), 3.75)
    assert _same(_quantize(100.0, device, **kw), 3.75)
    for value in (-0.3, -2.0, -0.0, -1e-30, float("-inf")):
        assert _same(_quantize(value, device, **kw), 0.0)
    assert _same(_quantize(float("inf"), device, **kw), 3.75)


# Signalling and quiet NaNs of both signs, with small, full and mixed payloads,
# as in tests/test_binaryk_p3109.py. The float64 words add payloads in the low
# 32 bits, which a cast that went through float32 would lose. float16 and
# bfloat16 are left out: storing into them changes a NaN's payload by itself.
NANS32 = torch.tensor(
    [0x7F800001, 0xFF800001, 0x7FC00000, 0xFFC00000, 0x7FFFFFFF, 0xFF812345], dtype=torch.int64
).to(torch.int32)
NANS64 = torch.tensor(
    [
        0x7FF0000000000001,
        0x7FF8000000000000,
        0x7FFFFFFFFFFFFFFF,
        0x7FF0000100000000,
        0x7FF4000000012345,
        0x7FF8000000000001,
    ],
    dtype=torch.int64,
)
NANS64 = torch.cat([NANS64, NANS64 | torch.tensor(-(2**63), dtype=torch.int64)])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("is_signed", [True, False], ids=["signed", "unsigned"])
@pytest.mark.parametrize(
    "nans,wl,fl", [(NANS32, 8, 4), (NANS32, 4, 2), (NANS64, 8, 4), (NANS64, 40, 30)]
)
def test_nan_passes_through_whole(device, is_signed, nans, wl, fl):
    """A NaN comes back bit for bit, in every mode: payload, signalling bit and
    sign.

    The directed modes are where that is not free: a NaN compares false against
    zero, so it takes the arm that works on a magnitude and negates it back, and
    a float negation may canonicalize the payload. An unsigned format turns a
    negative input into zero, but a NaN with its sign bit set is unordered, not
    negative, so it passes through too. Under SR it is returned before any draw
    is used."""
    if nans.dtype is torch.int64 and not has_float64(device):
        pytest.skip("MPS has no float64 tensors.")
    x = nans.view(torch.float32 if nans.dtype is torch.int32 else torch.float64).to(device)
    for mode in EVERY_MODE:
        prng_bits = SR_BITS if mode is RoundMode.SR else 0
        got = fixedpoint_quantize(
            x, wl, fl, prng_bits=prng_bits, is_signed=is_signed, rounding_mode=mode
        )
        assert torch.equal(got.cpu().view(nans.dtype), nans), mode


# ---------------------------------------------------------------------------
# The other deterministic modes, in steps of 0.25: 0.3 is 1.2 steps, 0.375 and
# 0.625 are the ties 1.5 and 2.5, and below the step 0.1 is 0.4 steps and
# 0.125 the tie 0.5. RNA breaks a tie away from zero; RU, RD and RZ take one
# neighbour whatever the distance; RO takes the one with an odd code whenever
# the value is inexact, which under the step is always the step itself (code
# 1). A zero result is +0.0 in every mode.
OTHER_MODE_CASES = [
    # (value, mode, expected)
    (0.3, RoundMode.RNA, 0.25),
    (0.375, RoundMode.RNA, 0.5),  # tie -> away from zero
    (0.625, RoundMode.RNA, 0.75),  # tie -> away from zero, not to even
    (-0.375, RoundMode.RNA, -0.5),
    (0.1, RoundMode.RNA, 0.0),
    (0.125, RoundMode.RNA, 0.25),  # half a step: tie -> the step, not zero
    (-0.125, RoundMode.RNA, -0.25),
    (0.3, RoundMode.RU, 0.5),
    (0.375, RoundMode.RU, 0.5),
    (0.625, RoundMode.RU, 0.75),
    (-0.3, RoundMode.RU, -0.25),  # up is toward zero for a negative value
    (-0.375, RoundMode.RU, -0.25),
    (0.1, RoundMode.RU, 0.25),
    (1e-30, RoundMode.RU, 0.25),  # any positive value leaves zero
    (-0.1, RoundMode.RU, 0.0),  # +0.0, not -0.0
    (-1e-30, RoundMode.RU, 0.0),
    (0.3, RoundMode.RD, 0.25),
    (0.375, RoundMode.RD, 0.25),
    (0.625, RoundMode.RD, 0.5),
    (-0.3, RoundMode.RD, -0.5),  # down is away from zero for a negative value
    (-0.375, RoundMode.RD, -0.5),
    (0.1, RoundMode.RD, 0.0),
    (-0.1, RoundMode.RD, -0.25),
    (-1e-30, RoundMode.RD, -0.25),
    (0.3, RoundMode.RZ, 0.25),
    (0.625, RoundMode.RZ, 0.5),
    (-0.3, RoundMode.RZ, -0.25),
    (-0.625, RoundMode.RZ, -0.5),
    (0.2, RoundMode.RZ, 0.0),
    (-0.2, RoundMode.RZ, 0.0),  # +0.0, not -0.0
    (0.25, RoundMode.RO, 0.25),  # exact: left alone
    (0.5, RoundMode.RO, 0.5),  # exact, even code 2: still left alone
    (0.3, RoundMode.RO, 0.25),  # 1.2 steps: code 1 is odd
    (0.375, RoundMode.RO, 0.25),  # 1.5 steps: code 1 is odd
    (0.625, RoundMode.RO, 0.75),  # 2.5 steps: code 3 is odd
    (-0.3, RoundMode.RO, -0.25),
    (0.1, RoundMode.RO, 0.25),  # under the step: the step, code 1
    (1e-30, RoundMode.RO, 0.25),
    (-0.1, RoundMode.RO, -0.25),
    (0.0, RoundMode.RO, 0.0),  # zero is exact
    (-0.0, RoundMode.RO, 0.0),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value,mode,expected", OTHER_MODE_CASES)
def test_other_modes(device, value, mode, expected):
    assert _same(_quantize(value, device, mode=mode), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "cfg,value,expected",
    [
        ({"wl": 4, "fl": -2}, 13.0, 12.0),  # a step of 4
        ({"wl": 4, "fl": -2}, 14.0, 16.0),  # tie -> even code (4)
        ({"wl": 4, "fl": -2}, 40.0, 28.0),  # saturates at 7 * 4
        ({"wl": 4, "fl": 6}, 0.03, 0.03125),  # pure fractions: steps of 1/64
        ({"wl": 4, "fl": 6}, 0.2, 0.109375),  # saturates at 7/64
        ({"wl": 8, "fl": 0}, 2.5, 2.0),  # integers: tie -> even
        ({"wl": 8, "fl": 0}, 3.5, 4.0),
        ({"wl": 8, "fl": 0}, -128.4, -128.0),
    ],
)
def test_other_steps(device, cfg, value, expected):
    """A step coarser than 1, a format of pure fractions, and plain integers."""
    assert _same(_quantize(value, device, cfg=cfg), expected)


# ---------------------------------------------------------------------------
# The exact reference: the same rounding in Python's integers, where nothing is
# rounded but the rule itself. The kernel's result is stored in the tensor's
# dtype, so the reference's is too, rounded once into it. A result the dtype
# cannot hold comes back changed, which the format's storage check must warn of.


def _reference(
    x: float, wl: int, fl: int, is_signed: bool, symmetric: bool, mode: RoundMode
) -> float:
    """``x`` rounded onto the fixed-point grid in ``mode``, then saturated."""
    if math.isnan(x):
        return x
    mag_bits = wl - (1 if is_signed else 0)
    # the ends as codes, multiples of the step 2**-fl
    top = 2**mag_bits - 1
    bottom = 0 if not is_signed else (-top if symmetric else -(2**mag_bits))
    if math.isinf(x):
        return math.ldexp(top if x > 0 else bottom, -fl)
    if x < 0 and not is_signed:
        return 0.0
    # x / step as num / den, exactly: a float is an integer over a power of two
    n, d = x.as_integer_ratio()
    num, den = (n << fl, d) if fl >= 0 else (n, d << -fl)
    # the code below the value, and how far past it the value is, rem / den
    code, rem = divmod(num, den)
    if rem == 0:
        up = False  # on the grid: every mode leaves it alone
    elif mode is RoundMode.RNE:
        up = 2 * rem > den or (2 * rem == den and code % 2 == 1)
    elif mode is RoundMode.RNA:
        up = 2 * rem > den or (2 * rem == den and x > 0)
    elif mode is RoundMode.RU:
        up = True
    elif mode is RoundMode.RD:
        up = False
    elif mode is RoundMode.RZ:
        up = x < 0
    elif mode is RoundMode.RO:
        up = code % 2 == 0
    else:
        raise ValueError(f"no reference for {mode}")
    code += up
    return math.ldexp(min(max(code, bottom), top), -fl) + 0.0  # + 0.0: no -0.0


def _float16_values() -> list[float]:
    words = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    return words.view(torch.float16).double().tolist()


def _random_values(seed: int) -> list[float]:
    """Values across 60 binades, both signs, and the inputs every cast must handle."""
    rng = random.Random(seed)
    values = [rng.gauss(0, 1) * 2.0 ** rng.randint(-30, 30) for _ in range(5000)]
    return values + [0.0, -0.0, math.inf, -math.inf, math.nan, 1e-30, -1e-30, 1e30, -1e30]


FORMATS = [
    # (wl, fl, is_signed, symmetric)
    (4, 2, True, False),
    (4, 2, True, True),
    (4, 2, False, False),
    (8, 4, True, False),
    (8, -3, True, False),
    (8, 10, True, False),
    (16, 8, True, False),
    (24, 10, True, False),  # 23 magnitude bits: all of binary32's mantissa
    (23, 0, False, False),  # 23 magnitude bits, unsigned
    (2, 0, True, False),  # one magnitude bit
    (1, 3, False, False),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_against_the_exact_reference(device, mode, dtype):
    """In every format of FORMATS, every float16 value (in float16) or 5000
    random ones (in the other dtypes) rounds as the reference rounds it, word
    for word, signed zeros included. Where the dtype cannot hold the exact
    result (the format's end past float16's range, say, or with more bits
    than bfloat16 has), the result is the exact one rounded once into the
    dtype, and only with a FormatRangeWarning saying so."""
    for wl, fl, is_signed, symmetric in FORMATS:
        fmt = f"FixedPoint({wl}, {fl}, is_signed={is_signed}, symmetric={symmetric})"
        values = _float16_values() if dtype is torch.float16 else _random_values(wl * 100 + fl)
        x = torch.tensor(values, dtype=torch.float64).to(dtype).to(device)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FormatRangeWarning)
            got = fixedpoint_quantize(
                x, wl, fl, is_signed=is_signed, symmetric=symmetric, rounding_mode=mode
            )
        exact = torch.tensor(
            [_reference(v, wl, fl, is_signed, symmetric, mode) for v in x.cpu().double().tolist()],
            dtype=torch.float64,
        )
        want = exact.to(dtype)
        got = got.cpu()
        same = (got == want) & (torch.signbit(got) == torch.signbit(want))
        same |= torch.isnan(got) & torch.isnan(want)
        assert bool(same.all()), f"{fmt}: first mismatch at x={x.cpu()[~same][0].item()!r}"
        held = (want.double() == exact) | torch.isnan(exact)
        warned = any(issubclass(w.category, FormatRangeWarning) for w in caught)
        assert warned or bool(held.all()), (
            f"{fmt}: x={x.cpu()[~held][0].item()!r} rounds to {exact[~held][0].item()!r}, "
            f"which {dtype} stores as {want[~held][0].item()!r}, with no FormatRangeWarning"
        )


# Formats only binary64 carries: more than binary32's 23 mantissa bits, a step
# below its floor, or a top past its largest value. Each is inside binary64's
# own bounds (52 mantissa bits, a step no finer than 2**-1021, a top below
# 2**1023).
WIDE_FORMATS = [
    # (wl, fl, is_signed, symmetric)
    (40, 30, True, False),  # 39 magnitude bits
    (40, 30, True, True),
    (53, 20, True, False),  # 52 magnitude bits: all of binary64's mantissa
    (52, 0, False, False),  # 52 magnitude bits, unsigned
    (30, 1000, True, False),  # a step of 2**-1000, a top of 2**-971
    (12, -900, True, False),  # a step of 2**900, a top of 2**910
]


def _wide_values(
    wl: int, fl: int, is_signed: bool, dtype: torch.dtype = torch.float64
) -> list[float]:
    """Values across a format: random significands from three binades under
    the step to one past the top, both signs, and the grid points and
    midpoints at both ends with a ``dtype`` ulp either side of each (a float64
    ulp is lost once the value is stored in a narrower dtype)."""
    rng = random.Random(wl * 10_000 + fl)
    mag_bits = wl - (1 if is_signed else 0)
    step_exp, top_exp = -fl, mag_bits - 1 - fl
    values = [
        math.ldexp(1 + rng.random(), rng.randint(step_exp - 3, top_exp + 1)) for _ in range(4000)
    ]
    for code in (1, 2, 3, 2**mag_bits - 3, 2**mag_bits - 2, 2**mag_bits - 1):
        point = math.ldexp(code, step_exp)
        half = math.ldexp(2 * code + 1, step_exp - 1)  # rounded where dtype cannot hold it
        for v in (point, half):
            t = torch.tensor([v] * 3, dtype=torch.float64).to(dtype)
            towards = torch.tensor([v, math.inf, 0.0], dtype=torch.float64).to(dtype)
            values += torch.nextafter(t, towards).tolist()
    return values + [-v for v in values]


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("wl,fl,is_signed,symmetric", WIDE_FORMATS)
def test_formats_past_binary32(device, wl, fl, is_signed, symmetric):
    """A float64 tensor rounds in binary64 to formats binary32 cannot hold, as
    the reference does, in every mode. Catches a cast constant, mask or bound
    still sized for binary32 (a 32-bit word, a 23-bit mantissa) in the
    ``double`` instantiation."""
    x = torch.tensor(_wide_values(wl, fl, is_signed), dtype=torch.float64, device=device)
    for mode in ALL_MODES:
        got = fixedpoint_quantize(
            x, wl, fl, is_signed=is_signed, symmetric=symmetric, rounding_mode=mode
        ).cpu()
        want = torch.tensor(
            [_reference(v, wl, fl, is_signed, symmetric, mode) for v in x.tolist()],
            dtype=torch.float64,
        )
        same = (got == want) & (torch.signbit(got) == torch.signbit(want))
        assert bool(same.all()), f"{mode.name}: first mismatch at x={x[~same][0].item()!r}"


# Formats on binary32's edges, each the last one tests/test_format_limits.py
# holds silent there: a step at the floor the casts place values by, a top
# whose SR shift is binary32's top binade, and all 23 of its mantissa bits.
EDGE_FORMATS = [
    # (wl, fl, is_signed, symmetric)
    (8, 125, True, False),  # a step of 2**-125
    (8, -120, True, False),  # a top of 127 * 2**120, SR's shift 2**127
    (24, 0, True, False),  # 23 magnitude bits
    (23, 0, False, False),  # 23 magnitude bits, unsigned
]


def _same_as_reference(
    x: torch.Tensor, wl: int, fl: int, is_signed: bool, symmetric: bool, mode: RoundMode
) -> torch.Tensor:
    """Elementwise: ``x`` quantized to the format is the reference's result,
    stored in ``x``'s dtype, signed zeros included."""
    got = fixedpoint_quantize(
        x, wl, fl, is_signed=is_signed, symmetric=symmetric, rounding_mode=mode
    ).cpu()
    want = torch.tensor(
        [_reference(v, wl, fl, is_signed, symmetric, mode) for v in x.cpu().tolist()],
        dtype=torch.float64,
    ).to(x.dtype)
    return (got == want) & (torch.signbit(got) == torch.signbit(want))


@pytest.mark.parametrize("device", DEVICES)
def test_grid_ends_in_binary32(device):
    """In binary32, in every format of FORMATS and EDGE_FORMATS, the grid
    points and ties at both ends, with a binary32 ulp either side of each,
    round as the reference rounds them, in every mode. Random inputs all but
    never land on a tie, and the float16 sweep only reaches ties of up to 11
    significant bits, so this is where a tie of up to 24 is held. It also holds
    the formats on binary32's edges, which tests/test_format_limits.py holds
    silent, to the values they give."""
    for wl, fl, is_signed, symmetric in FORMATS + EDGE_FORMATS:
        fmt = f"FixedPoint({wl}, {fl}, is_signed={is_signed}, symmetric={symmetric})"
        values = _wide_values(wl, fl, is_signed, torch.float32)
        x = torch.tensor(values, dtype=torch.float64).to(torch.float32).to(device)
        for mode in ALL_MODES:
            same = _same_as_reference(x, wl, fl, is_signed, symmetric, mode)
            assert bool(same.all()), (
                f"{fmt} {mode.name}: first mismatch at x={x.cpu()[~same][0].item()!r}"
            )


@pytest.mark.parametrize("device", DEVICES)
def test_one_step_below_binary32s_floor_rounds_wrongly(device):
    """A step of 2**-126, one finer than EDGE_FORMATS' 2**-125, does round some
    values wrongly in binary32, so the warning tests/test_format_limits.py
    expects there marks a real failure rather than a cautious margin. If the
    cast comes to round it correctly, this fails, and the floor should move."""
    values = _wide_values(8, 126, True, torch.float32)
    x = torch.tensor(values, dtype=torch.float64).to(torch.float32).to(device)
    assert not bool(_same_as_reference(x, 8, 126, True, False, RoundMode.RNE).all())


# ---------------------------------------------------------------------------
# Stochastic rounding, held to its properties rather than to values. It adds
# prng_bits random bits below the step and truncates (P3109's StochasticA), so
# a value rounds up with probability equal to how far it sits past the grid
# point below it, to the resolution of the draw.


def _sr_bits(wl: int, is_signed: bool, dtype: torch.dtype) -> int:
    """``SR_BITS``, or as many as fit beside the format's bits in the carrier's
    mantissa (23 in binary32, 52 in binary64), which number.py holds them to."""
    mag_bits = wl - (1 if is_signed else 0)
    return max(0, min(SR_BITS, (52 if dtype is torch.float64 else 23) - mag_bits))


def _sr(x: torch.Tensor, wl: int, fl: int, prng_bits: int = SR_BITS, **kw) -> torch.Tensor:
    return fixedpoint_quantize(x, wl, fl, prng_bits=prng_bits, rounding_mode=RoundMode.SR, **kw)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_sr_lands_on_a_neighbour(device, dtype):
    """In every format of FORMATS, and of WIDE_FORMATS in float64, every draw is
    the RD or the RU result for its input, saturated as they are, and a zero
    result is +0.0. Each input is repeated so that both of its neighbours are
    reached. The inputs' RNE results are inputs too: on a grid point RD and RU
    are the point itself, so every draw must leave it alone."""
    wide = WIDE_FORMATS if dtype is torch.float64 else []
    for wl, fl, is_signed, symmetric in FORMATS + wide:
        kw = dict(is_signed=is_signed, symmetric=symmetric)
        fmt = f"FixedPoint({wl}, {fl}, is_signed={is_signed}, symmetric={symmetric})"
        values = (
            _wide_values(wl, fl, is_signed)
            if (wl, fl, is_signed, symmetric) in wide
            else _random_values(wl * 100 + fl)
        )
        x = torch.tensor(values, dtype=torch.float64).to(dtype).to(device)
        x = torch.cat([x, fixedpoint_quantize(x, wl, fl, **kw)]).repeat(4)
        got = _sr(x, wl, fl, prng_bits=_sr_bits(wl, is_signed, dtype), **kw)
        down = fixedpoint_quantize(x, wl, fl, rounding_mode=RoundMode.RD, **kw)
        up = fixedpoint_quantize(x, wl, fl, rounding_mode=RoundMode.RU, **kw)
        ok = (got == down) | (got == up) | (torch.isnan(got) & torch.isnan(x))
        assert bool(ok.all()), f"{fmt}: first stray result at x={x[~ok][0].item()!r}"
        assert not bool(torch.signbit(got[got == 0]).any()), f"{fmt}: a zero is +0.0"


SR_MEAN_CASES = [
    # (wl, fl, is_signed, symmetric, value)
    # the reference format, above the step and below it, where the two
    # candidates are zero and the step
    *[(4, 2, True, False, v) for v in (0.3, -0.3, 1.6, -1.1, 0.1, -0.05, 0.03)],
    (8, -3, True, False, 13.0),  # a step of 8
    (8, -3, True, False, -3.0),  # under that step
    (12, 4, False, False, 3.3),  # unsigned
    (4, 2, True, True, -1.6),  # symmetric, near its bottom, -1.75
    (40, 30, True, False, 0.3),  # 39 magnitude bits, which only binary64 carries
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sr_is_unbiased(device, dtype):
    """In every case of SR_MEAN_CASES whose magnitude bits fit the carrier's
    mantissa, the mean of the draws is the input: formats of each kind SR
    shifts differently, steps finer and coarser than 1, unsigned, symmetric,
    and past binary32's precision. The fraction of draws that round up is a
    binomial proportion, held to six standard errors plus the draw's
    resolution, 2**-prng_bits, with up to 16 random bits beside the format's in
    the carrier's mantissa."""
    n = 400_000
    man_bits = 52 if dtype is torch.float64 else 23
    for wl, fl, is_signed, symmetric, value in SR_MEAN_CASES:
        mag_bits = wl - (1 if is_signed else 0)
        if mag_bits > man_bits:
            continue
        fmt = f"FixedPoint({wl}, {fl}, is_signed={is_signed}, symmetric={symmetric})"
        prng_bits = min(16, man_bits - mag_bits)
        x = torch.full((n,), value, dtype=dtype, device=device)
        value = x[0].item()  # the value the dtype holds
        got = _sr(x, wl, fl, prng_bits=prng_bits, is_signed=is_signed, symmetric=symmetric)
        got = got.cpu().double()
        step = 2.0**-fl
        below = math.floor(value / step) * step
        assert set(got.unique().tolist()) <= {below, below + step}, f"{fmt} at {value}"
        frac = (value - below) / step
        tolerance = 6 * math.sqrt(frac * (1 - frac) / n) + 2.0**-prng_bits
        mean = (got.mean().item() - below) / step
        assert mean == pytest.approx(frac, abs=tolerance), f"{fmt} at {value}"


@pytest.mark.parametrize("device", DEVICES)
def test_sr_prng_bits_set_the_resolution(device):
    """``prng_bits`` is how finely the draw resolves the distance to the grid.
    With none, nothing is added and SR truncates the magnitude, which is RZ.
    0.3125 sits a quarter step past 0.25: one random bit adds at most half a
    step and never carries it up, and two bits carry it up one draw in four."""
    x = torch.tensor(_random_values(19), device=device)
    rz = fixedpoint_quantize(x, **CFG, rounding_mode=RoundMode.RZ)
    assert torch.equal(_sr(x, **CFG, prng_bits=0).nan_to_num(0.5), rz.nan_to_num(0.5))
    quarter = torch.full((100_000,), STEP + STEP / 4, device=device)
    assert bool((_sr(quarter, **CFG, prng_bits=1) == STEP).all())
    rounded_up = (_sr(quarter, **CFG, prng_bits=2) == 2 * STEP).cpu().double().mean().item()
    assert rounded_up == pytest.approx(0.25, abs=0.01)
