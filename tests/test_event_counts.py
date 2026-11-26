"""The numeric events of a cast, as `mptorch.quant.events` defines them.

Six events are read from a cast's input ``x`` and its result ``q``: every
cast, an overflow (``x`` finite and past the format's largest finite
magnitude), an underflow (``x`` finite and nonzero, ``q`` zero), a subnormal
result, an inexact one, and a NaN result. A quantizer's events are recounted
from its input and output (``_recount``); the GEMM kernels classify their
casts with the same table on the words (``csrc/common/event_counts.h``).

The tables below hold the recount to the definition. Each row is a value at
one event's boundary, in each family, subnormals mode, saturation mode and
directed rounding, and the events written for it come from the definitions,
not from a second classifier: the quantizer rounds the value, and the
recount of that one cast must give exactly the events written. The rows run
in binary32 on every device and in binary64 on float64 tensors.
"""

import math

import pytest
import torch

from mptorch import BinaryK, FixedPoint, RoundMode, SaturationMode, SubnormalsMode, SuperFP
from mptorch.quant import Quant
from mptorch.quant.events import EVENTS, _recount, _thresholds
from tests.markers import available_devices, float64_devices

inf, nan = math.inf, math.nan
OVF, UNF, SUB, INEX, NAN = "overflow", "underflow", "subnormal", "inexact", "nan"

# Binary8p4 (E4M3's layout in P3109's extended domain): the largest finite
# value 224 (240 in the finite domain), the smallest normal 2^-7, the
# smallest subnormal 2^-10.
E4M3 = BinaryK(8, 4)

BINARYK_ROWS = [
    (1.0, set()),
    (1.0625, {INEX}),  # a tie between 1 and 1.125, to even
    (224.0, set()),
    (230.0, {OVF, INEX}),  # past the largest value, rounds back to it
    (236.0, {OVF, INEX}),  # rounds to the infinity
    (300.0, {OVF, INEX}),
    (-300.0, {OVF, INEX}),
    (inf, set()),
    (-inf, set()),
    (2.0**-7, set()),
    (0.99 * 2.0**-7, {INEX}),  # rounds up to the smallest normal
    (2.0**-8, {SUB}),
    (0.7 * 2.0**-7, {SUB, INEX}),
    (2.0**-10, {SUB}),
    (1.5 * 2.0**-11, {SUB, INEX}),  # past half the smallest subnormal
    (2.0**-11, {UNF, INEX}),  # half the smallest subnormal, to even: zero
    (1e-30, {UNF, INEX}),
    (0.0, set()),
    (-0.0, set()),  # the one zero: -0 and +0 are one value
    (nan, {NAN}),
]

# Past the top in each saturation mode: SAT_FINITE is the finite domain
# (largest value 240, infinities onto it); the other three are the extended
# domain (224), of which SAT_NONE overflows to the infinity, SAT_PROPAGATE
# onto the largest value with infinities kept, SAT_FINITE_EXTENDED onto it
# infinities included. An infinite input is never an overflow.
SATURATION_ROWS = {
    SaturationMode.SAT_NONE: [
        (inf, set()),
        (-inf, set()),
        (230.0, {OVF, INEX}),
        (240.0, {OVF, INEX}),
        (300.0, {OVF, INEX}),
    ],
    SaturationMode.SAT_PROPAGATE: [
        (inf, set()),
        (-inf, set()),
        (230.0, {OVF, INEX}),
        (240.0, {OVF, INEX}),
        (300.0, {OVF, INEX}),
    ],
    SaturationMode.SAT_FINITE_EXTENDED: [
        (inf, {INEX}),
        (-inf, {INEX}),
        (230.0, {OVF, INEX}),
        (240.0, {OVF, INEX}),
        (300.0, {OVF, INEX}),
    ],
    SaturationMode.SAT_FINITE: [
        (inf, {INEX}),
        (-inf, {INEX}),
        (230.0, {INEX}),  # below the finite domain's top, 240
        (240.0, set()),
        (300.0, {OVF, INEX}),
    ],
}

# Below the smallest normal in each subnormals mode: NORMALS rounds onto zero
# or 2^-7, EXTENDED_NORMALS has a binade [2^-8, 2^-7) of normals whose
# mantissa-zero code is the zero; neither has subnormal results.
SUBNORMALS_ROWS = {
    SubnormalsMode.SUBNORMALS: [
        (2.0**-7, set()),
        (2.0**-8, {SUB}),
        (0.7 * 2.0**-7, {SUB, INEX}),
        (2.0**-9, {SUB}),
        (2.0**-11, {UNF, INEX}),
    ],
    SubnormalsMode.NORMALS: [
        (2.0**-7, set()),
        (2.0**-8, {UNF, INEX}),
        (0.7 * 2.0**-7, {INEX}),  # onto 2^-7
        (2.0**-9, {UNF, INEX}),
        (2.0**-11, {UNF, INEX}),
    ],
    SubnormalsMode.EXTENDED_NORMALS: [
        (2.0**-7, set()),
        (2.0**-8, {INEX}),  # the floor binade's zero: onto 1.125 * 2^-8
        (1.5 * 2.0**-8, set()),
        (0.7 * 2.0**-7, {INEX}),
        (2.0**-9, {UNF, INEX}),
        (2.0**-11, {UNF, INEX}),
    ],
}

# Directed rounding at the underflow boundary: half the smallest subnormal
# rounds away from zero or onto it, by its sign.
DIRECTED_ROWS = {
    RoundMode.RU: [(2.0**-11, {SUB, INEX}), (-(2.0**-11), {UNF, INEX})],
    RoundMode.RD: [(2.0**-11, {UNF, INEX}), (-(2.0**-11), {SUB, INEX})],
    RoundMode.RZ: [(2.0**-11, {UNF, INEX}), (300.0, {OVF, INEX})],
}

# An unsigned format sends a negative input to zero: an underflow, and an
# overflow too when its magnitude is past the largest value (53248 here).
UNSIGNED_ROWS = [
    (-1.0, {UNF, INEX}),
    (-1e9, {OVF, UNF, INEX}),
    (-inf, {INEX}),
    (1e9, {OVF, INEX}),
    (inf, set()),
    (nan, {NAN}),
]

# SuperFP(3, 4, 2, 7): normals from 128 to 448, supernormal powers of two
# below them down to 2^-104.
SUPERFP_ROWS = [
    (128.0, set()),
    (448.0, set()),
    (500.0, {OVF, INEX}),
    (100.0, {INEX}),  # onto 128, a normal
    (64.0, {SUB}),
    (70.0, {SUB, INEX}),
    (2.0**-104, {SUB}),
    (2.0**-106, {UNF, INEX}),
    (inf, set()),
    (nan, {NAN}),
]

# Fixed point: no subnormal results; a signed format that is not symmetric
# reaches -8, one step further than 7.9375.
FIXEDPOINT_ROWS = {
    FixedPoint(8, 4): [
        (7.9375, set()),
        (8.0, {OVF, INEX}),
        (-8.0, set()),
        (-8.03125, {OVF, INEX}),
        (0.0625, set()),
        (0.04, {INEX}),
        (0.03, {UNF, INEX}),  # below half a step
        (inf, {INEX}),
        (nan, {NAN}),
    ],
    FixedPoint(8, 4, symmetric=True): [
        (-7.9375, set()),
        (-8.0, {OVF, INEX}),
    ],
    FixedPoint(8, 4, is_signed=False): [
        (15.9375, set()),
        (16.0, {OVF, INEX}),
        (-1.0, {UNF, INEX}),
        (-20.0, {OVF, UNF, INEX}),
        (-inf, {INEX}),
    ],
}


def _cases():
    yield "binaryK", E4M3, RoundMode.RNE, BINARYK_ROWS
    for sat, rows in SATURATION_ROWS.items():
        yield f"saturation-{sat.name}", BinaryK(8, 4, saturation=sat), RoundMode.RNE, rows
    for sub, rows in SUBNORMALS_ROWS.items():
        yield f"subnormals-{sub.name}", BinaryK(8, 4, subnormals=sub), RoundMode.RNE, rows
    for mode, rows in DIRECTED_ROWS.items():
        yield f"rounding-{mode.name}", E4M3, mode, rows
    yield "unsigned", BinaryK(8, 4, is_signed=False), RoundMode.RNE, UNSIGNED_ROWS
    yield "superfp", SuperFP(3, 4, 2, 7), RoundMode.RNE, SUPERFP_ROWS
    for fmt, rows in FIXEDPOINT_ROWS.items():
        name = f"fixedpoint-{'s' if fmt.is_signed else 'u'}{'sym' if fmt.symmetric else ''}"
        yield name, fmt, RoundMode.RNE, rows


CASES = list(_cases())


def _events_per_cast(fmt, rounding, values, device, dtype):
    """The quantizer's results, and each cast's recounted events as a set."""
    x = torch.tensor(values, dtype=dtype, device=device)
    q = Quant(fmt, rounding)(x)
    rows = [_recount(x[i : i + 1], q[i : i + 1], fmt).tolist() for i in range(len(values))]
    for row in rows:
        assert row[0] == 1  # one cast
        assert all(n in (0, 1) for n in row[1:])
    # the recount of the whole tensor is the sum of its casts'
    whole = _recount(x, q, fmt).tolist()
    assert whole == [sum(col) for col in zip(*rows, strict=True)]
    return q, [{e for e, n in zip(EVENTS[1:], row[1:], strict=True) if n} for row in rows]


def _check(fmt, rounding, rows, device, dtype):
    values = [v for v, _ in rows]
    q, got = _events_per_cast(fmt, rounding, values, device, dtype)
    for (v, want), have, r in zip(rows, got, q.tolist(), strict=True):
        assert have == want, f"{fmt} under {rounding.name}: {v!r} -> {r!r} gave {have}, not {want}"


@pytest.mark.parametrize("name,fmt,rounding,rows", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("device", available_devices)
def test_events_binary32(name, fmt, rounding, rows, device):
    _check(fmt, rounding, rows, device, torch.float32)


@pytest.mark.parametrize("name,fmt,rounding,rows", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("device", float64_devices)
def test_events_binary64(name, fmt, rounding, rows, device):
    _check(fmt, rounding, rows, device, torch.float64)


@pytest.mark.parametrize("device", available_devices)
def test_events_of_narrow_storage(device):
    """A bfloat16 tensor's casts are counted on its values, widened to the
    carrier: the rows whose values bfloat16 holds give the same events."""
    held = [float(torch.tensor(v, dtype=torch.bfloat16)) for v, _ in BINARYK_ROWS]
    rows = [(v, e) for (v, e), h in zip(BINARYK_ROWS, held, strict=True) if h == v or v != v]
    assert len(rows) > 10
    _check(E4M3, RoundMode.RNE, rows, device, torch.bfloat16)


@pytest.mark.parametrize("device", available_devices)
def test_events_under_stochastic_rounding(device):
    """Under SR the result is either neighbour, so the events of a value
    between two of them are inexact for every draw, and at the underflow
    boundary an underflow or a subnormal, one of the two for every draw."""
    fmt = BinaryK(8, 4, prng_bits=8)
    n = 4096
    sr = Quant(fmt, RoundMode.SR)
    x = torch.full((n,), 1.0625, device=device)
    assert _recount(x, sr(x), fmt).tolist() == [n, 0, 0, 0, n, 0]
    x = torch.full((n,), 2.0**-11, device=device)
    c = dict(zip(EVENTS, _recount(x, sr(x), fmt).tolist(), strict=True))
    assert c["inexact"] == n and c["underflow"] + c["subnormal"] == n
    assert 0 < c["underflow"] < n  # both neighbours drawn


def test_thresholds():
    """The thresholds come from the format's own fields, as the kernels
    build theirs: the largest finite value on each side, and the bottom of
    the normal range where the format has subnormal results."""
    f32, f64 = torch.float32, torch.float64
    assert _thresholds(E4M3, f32) == (224.0, 224.0, 2.0**-7)
    finite = BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE)
    assert _thresholds(finite, f32) == (240.0, 240.0, 2.0**-7)
    assert _thresholds(BinaryK(8, 4, subnormals=SubnormalsMode.NORMALS), f32) == (224.0, 224.0, 0.0)
    assert _thresholds(BinaryK(8, 4, subnormals=SubnormalsMode.EXTENDED_NORMALS), f32)[2] == 0.0
    assert _thresholds(SuperFP(3, 4, 2, 7), f32) == (448.0, 448.0, 128.0)
    assert _thresholds(FixedPoint(8, 4), f32) == (7.9375, 8.0, 0.0)
    assert _thresholds(FixedPoint(8, 4, symmetric=True), f32) == (7.9375, 7.9375, 0.0)
    assert _thresholds(FixedPoint(8, 4, is_signed=False), f32) == (15.9375, 15.9375, 0.0)
    # A range past binary32's: no finite binary32 input overflows it, and a
    # floor below binary32's subnormals leaves no subnormal result there.
    wide = BinaryK(12, 3)  # bias 256: from 2^-255 to about 2^255
    assert _thresholds(wide, f32) == (inf, inf, 0.0)
    assert _thresholds(wide, f64) == (wide.max_finite, wide.max_finite, 2.0**-255)
    # a floor among binary32's subnormals is one of them
    assert _thresholds(BinaryK(8, 4, bias=135), f32)[2] == 2.0**-134
    # a largest value past binary64's
    assert _thresholds(BinaryK(8, 4, bias=-2000), f64)[:2] == (inf, inf)
