"""
binaryK_quantize against IEEE P3109's own definitions, at the two ends of the range.

MPTorch's BinaryK is P3109's binaryK family (arXiv:2606.04028 is the working
group's overview). tests/test_binaryk_quantize.py checks the rounding against
gfloat below the largest finite value; this file checks what happens at and
beyond it, where the two things P3109 adds to a binary float -- the domain and
the saturation mode -- decide the answer. It also checks the other end, where
the same RoundAway decides between zero and the smallest subnormal and a zero
or float32-subnormal input used to slip past the rounded word's zero test.

The reference is transcribed from the paper, not from an implementation:

* the largest finite datum M_hi, from the decoding of Fig. 7 and the code
  points of SVII: a signed format's top non-negative code, 2^(K-1) - 1, is
  +infinity in the extended domain; an unsigned format's top code, 2^K - 1, is
  NaN, and the one below it is +infinity in the extended domain;
* omega-RoundToPrecision (Fig. 1) with RoundAway (Fig. 2);
* omega-Saturate (SIV-B): SatFinite clamps everything into [M_lo, M_hi],
  SatPropagate clamps finite values and keeps infinities, SatNone sends
  anything outside [M_lo, M_hi] to +-infinity. Rounding comes first, so SatNone
  overflows to infinity under every rounding mode -- which is where gfloat,
  following IEEE 754's directed-rounding overflow, parts from the draft, and
  why it is not the oracle here;
* the zero, code point 0: the only one, and unsigned, so a result that rounds
  to zero is +0.0 whatever the sign of the input -- -0.0 included. The
  comparison counts the sign of a zero, since -0.0 == 0.0.

SaturationMode carries the domain as well as the saturation: SAT_FINITE is the
finite domain, the other two the extended one. An unsigned format turns every
negative input into 0, -infinity included; the paper says so for SatNone, and
MPTorch reads the other two modes the same way.
"""

import math

import pytest
import torch

from mptorch.number import RoundMode, SaturationMode
from mptorch.quant import binaryK_quantize
from tests.markers import available_devices

DETERMINISTIC = [rm for rm in RoundMode if rm is not RoundMode.SR]


def _bias(K: int, P: int, signed: bool) -> int:
    return 2 ** (K - P - 1) if signed else 2 ** (K - P)


def _exp_bits(K: int, P: int, signed: bool) -> int:
    return K - P if signed else K - P + 1


def _top_exponent(K: int, P: int, signed: bool) -> int:
    return 2 ** _exp_bits(K, P, signed) - 1 - _bias(K, P, signed)


def _formats(signed: bool) -> list[tuple[int, int]]:
    """Every precision of the 3- to 8-bit formats, and a few wider ones.

    Only formats whose top binade binary32 can hold: a wider one has a largest
    finite value no float32 input reaches, which the paper's M_hi cannot say.
    """
    narrow = [(K, P) for K in range(3, 9) for P in range(1, K if signed else K + 1)]
    wide = [(10, 4), (12, 7), (16, 8), (16, 11)]
    return [(K, P) for K, P in narrow + wide if _top_exponent(K, P, signed) <= 127]


# --- the paper, transcribed ---------------------------------------------------


def _max_finite(K: int, P: int, signed: bool, finite: bool) -> float:
    """M_hi: the value of the highest code point that is neither infinity nor NaN."""
    B = _bias(K, P, signed)
    top = (2 ** (K - 1) - 1) if signed else (2**K - 2)  # the top code that is not NaN
    if not finite:
        top -= 1  # the extended domain's +infinity
    E, T = divmod(top, 2 ** (P - 1))
    significand = T * 2.0 ** (1 - P) if E == 0 else 1 + T * 2.0 ** (1 - P)
    return math.ldexp(significand, max(E, 1) - B)


def _round_to_precision(x: torch.Tensor, P: int, B: int, mode: RoundMode) -> torch.Tensor:
    """omega-RoundToPrecision (Fig. 1) with RoundAway (Fig. 2), on float64 values."""
    ax = x.abs()
    regular = torch.isfinite(x) & (x != 0)
    _, e = torch.frexp(torch.where(regular, ax, torch.ones_like(ax)))
    E_hat = e.to(torch.int64) - 1  # floor(log2 |X|), exactly
    E = torch.clamp(E_hat, min=1 - B) - P + 1
    S = torch.ldexp(ax, -E)
    S_floor = torch.floor(S)
    eta = S - S_floor
    if P > 1:
        code_is_odd = torch.remainder(S_floor, 2) == 1
    else:
        code_is_odd = (torch.remainder(E + B, 2) == 1) & (S_floor != 0)
    away = {
        RoundMode.RZ: torch.zeros_like(regular),
        RoundMode.RU: (eta > 0) & (x > 0),
        RoundMode.RD: (eta > 0) & (x < 0),
        RoundMode.RNA: eta >= 0.5,
        RoundMode.RNE: (eta > 0.5) | ((eta == 0.5) & code_is_odd),
        RoundMode.RO: (eta > 0) & ~code_is_odd,
    }[mode]
    rounded = torch.copysign(torch.ldexp(S_floor + away.to(S_floor.dtype), E), x)
    return torch.where(regular, rounded, x)


def _saturate(z: torch.Tensor, mode: SaturationMode, hi: float, signed: bool) -> torch.Tensor:
    """omega-Saturate (SIV-B) into [M_lo, M_hi]; an unsigned format sends negatives to 0."""
    lo = -hi if signed else 0.0
    if not signed:
        z = torch.where(z < 0, torch.zeros_like(z), z)
    clamped = z.clamp(lo, hi)  # NaN stays NaN
    if mode is SaturationMode.SAT_FINITE:
        return clamped
    if mode is SaturationMode.SAT_PROPAGATE:
        return torch.where(torch.isinf(z), z, clamped)
    return torch.where((z > hi) | (z < lo), torch.copysign(torch.full_like(z, math.inf), z), z)


def _project(x: torch.Tensor, K: int, P: int, signed: bool, rounding, saturation) -> torch.Tensor:
    finite = saturation is SaturationMode.SAT_FINITE
    rounded = _round_to_precision(x, P, _bias(K, P, signed), rounding)
    projected = _saturate(rounded, saturation, _max_finite(K, P, signed, finite), signed)
    # Code point 0 decodes to a zero with no sign.
    return torch.where(projected == 0, torch.zeros_like(projected), projected)


# --- inputs -------------------------------------------------------------------


def _inputs(K: int, P: int, signed: bool) -> torch.Tensor:
    """Eight float32 values per unit in the last place, from three binades below
    the top of the format to three above it, and the float32 extremes."""
    e_top = _top_exponent(K, P, signed)
    step = 1 << max(0, 23 - (P - 1) - 3)
    words = [
        ((e + 127) << 23) | m
        for e in range(max(e_top - 3, -126), min(e_top + 3, 127) + 1)
        for m in range(0, 1 << 23, step)
    ]
    x = torch.tensor(words, dtype=torch.int32).view(torch.float32)
    extremes = torch.tensor([torch.finfo(torch.float32).max, math.inf, math.nan, 1.0, 0.0])
    x = torch.cat([x, extremes])
    return torch.cat([x, -x]) if signed else x


def _mismatches(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    same = (got == want) & (torch.signbit(got) == torch.signbit(want))
    return ~(same | (got.isnan() & want.isnan()))


# --- tests --------------------------------------------------------------------


# Table I of the paper: the largest finite value of every 4-bit signed format,
# in both domains. Pins _max_finite itself before anything is checked against it.
TABLE_I = [(1, 4.0, 8.0), (2, 2.0, 3.0), (3, 1.5, 1.75)]


@pytest.mark.parametrize("P,extended,finite", TABLE_I)
def test_reference_largest_finite_matches_table_i(P, extended, finite):
    assert _max_finite(4, P, signed=True, finite=False) == extended
    assert _max_finite(4, P, signed=True, finite=True) == finite


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "K,P,signed,saturation_mode,largest",
    [
        (8, 4, True, SaturationMode.OVF_INF, 224.0),  # Binary8p4se
        (8, 4, True, SaturationMode.SAT_FINITE, 240.0),  # Binary8p4sf
        (8, 3, True, SaturationMode.SAT_PROPAGATE, 49152.0),  # Binary8p3se
        (8, 4, False, SaturationMode.OVF_INF, 53248.0),  # Binary8p4ue: NaN and +inf on top
        (8, 4, False, SaturationMode.SAT_FINITE, 57344.0),  # Binary8p4uf: NaN on top
        (4, 2, True, SaturationMode.OVF_INF, 2.0),  # Binary4p2se, Table I
        (4, 1, True, SaturationMode.SAT_FINITE, 8.0),  # Binary4p1sf, Table I
    ],
)
def test_largest_finite_value(device, K, P, signed, saturation_mode, largest):
    x = torch.tensor([largest, 3.0e38], device=device)
    for mode in DETERMINISTIC:
        q = binaryK_quantize(
            x, K, P, is_signed=signed, rounding_mode=mode, saturation_mode=saturation_mode
        )
        assert q[0].item() == largest, mode
        expected = math.inf if saturation_mode is SaturationMode.OVF_INF else largest
        assert q[1].item() == expected, mode


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("saturation_mode", list(SaturationMode))
@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
def test_top_of_range_matches_p3109(device, signed, saturation_mode):
    failures = []
    for K, P in _formats(signed):
        x = _inputs(K, P, signed)
        for mode in DETERMINISTIC:
            want = _project(x.double(), K, P, signed, mode, saturation_mode)
            got = binaryK_quantize(
                x.to(device),
                K,
                P,
                is_signed=signed,
                rounding_mode=mode,
                saturation_mode=saturation_mode,
            )
            got = got.cpu().double()
            bad = _mismatches(got, want)
            if bad.any():
                i = int(bad.nonzero()[0])
                failures.append(
                    f"Binary{K}p{P}{'s' if signed else 'u'} {mode.name}: {int(bad.sum())} "
                    f"mismatches, e.g. {x[i].item()!r} -> {got[i].item()!r}, "
                    f"P3109 {want[i].item()!r}"
                )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
def test_zero_and_far_below_the_smallest_subnormal(device, signed):
    # The other end of the range, where the same RoundAway decides between zero
    # and the smallest subnormal: +-0 becomes +0 under every mode, and a nonzero
    # input far below the grid -- a float32 subnormal included -- becomes +0 or
    # the smallest subnormal, never anything else.
    tiny = [0.0, 1.4e-45, 1.0e-40, 2.0**-100, 2.0**-60]
    x = torch.tensor(tiny + [-v for v in tiny] if signed else tiny, dtype=torch.float32)
    failures = []
    for K, P in [(8, 4), (8, 3), (6, 3), (5, 2), (6, 1)]:
        for mode in DETERMINISTIC:
            want = _project(x.double(), K, P, signed, mode, SaturationMode.OVF_INF)
            got = binaryK_quantize(x.to(device), K, P, is_signed=signed, rounding_mode=mode)
            got = got.cpu().double()
            bad = _mismatches(got, want)
            if bad.any():
                i = int(bad.nonzero()[0])
                failures.append(
                    f"Binary{K}p{P}{'s' if signed else 'u'} {mode.name}: {x[i].item()!r} -> "
                    f"{got[i].item()!r}, P3109 {want[i].item()!r}"
                )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("saturation_mode", list(SaturationMode))
@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
def test_stochastic_rounding_saturates_like_its_neighbours(device, signed, saturation_mode):
    # StochasticA picks one of the two rounding directions per input, and the
    # result goes through the same omega-Saturate either way: an SR result is
    # the projection of rounding toward zero or of rounding away from it.
    failures = []
    for K, P in [(8, 4), (8, 3), (6, 3), (5, 2), (5, 1)]:
        x = _inputs(K, P, signed)
        xd = x.double()
        toward = _project(xd, K, P, signed, RoundMode.RZ, saturation_mode)
        up = _project(xd, K, P, signed, RoundMode.RU, saturation_mode)
        down = _project(xd, K, P, signed, RoundMode.RD, saturation_mode)
        away = torch.where(xd > 0, up, down)
        got = binaryK_quantize(
            x.to(device),
            K,
            P,
            prng_bits=8,
            is_signed=signed,
            rounding_mode=RoundMode.SR,
            saturation_mode=saturation_mode,
        )
        got = got.cpu().double()
        bad = _mismatches(got, toward) & _mismatches(got, away)
        if bad.any():
            i = int(bad.nonzero()[0])
            failures.append(
                f"Binary{K}p{P}{'s' if signed else 'u'}: {int(bad.sum())} results outside "
                f"{{toward, away}}, e.g. {x[i].item()!r} -> {got[i].item()!r}"
            )
    assert not failures, "\n".join(failures)
