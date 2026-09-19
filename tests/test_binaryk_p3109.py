"""
``binaryK_quantize`` against IEEE P3109's definitions, at both ends of the range.

MPTorch's ``BinaryK`` is the binaryK family of IEEE P3109 (arXiv:2606.04028 is
the working group's overview). ``tests/test_binaryk_quantize.py`` checks the
rounding against gfloat below the largest finite value. This module guards what
happens at and beyond that value, where the two things P3109 adds to a binary
float (the domain and the saturation mode) decide the answer, and at the other
end, where the same RoundAway chooses between zero and the smallest subnormal.
Without it, a largest finite value counted the IEEE 754 way (one code point off),
a signed zero, or a NaN with a rewritten payload would pass every value
comparison elsewhere in the suite.

The reference is transcribed from the paper, not from an implementation, so it
shares no code with the cast it checks:

* the largest finite datum M_hi, from the decoding of Fig. 7 and the code
  points of section VII: a signed format's top non-negative code, 2^(K-1) - 1,
  is +infinity in the extended domain; an unsigned format's top code, 2^K - 1,
  is NaN, and the one below it is +infinity in the extended domain;
* omega-RoundToPrecision (Fig. 1) with RoundAway (Fig. 2);
* omega-Saturate (section IV-B): SatFinite clamps everything into [M_lo, M_hi],
  SatPropagate clamps finite values and keeps infinities, SatNone sends
  anything outside [M_lo, M_hi] to an infinity of the same sign. Rounding comes
  first, so SatNone overflows to infinity under every rounding mode. gfloat
  overflows the IEEE 754 way instead (a directed mode that rounds toward zero
  returns the largest finite value), which is why it is not the reference here;
* the zero, code point 0: the only one, and unsigned, so a result that rounds
  to zero is +0.0 whatever the sign of the input, -0.0 included. The
  comparison reads the sign bit of a zero, because ``-0.0 == 0.0`` is true.

``SaturationMode`` carries the domain as well as the saturation: ``SAT_FINITE``
is the finite domain, the other two the extended one. An unsigned format turns
every negative input into 0, -infinity included. The paper says so for SatNone,
and MPTorch reads the other two modes the same way.

Each test runs on float32 inputs, rounded in binary32, and again on float64
inputs float32 cannot hold, rounded in binary64, over formats up to K = 64
whose top binade binary64 holds.
"""

import math

import pytest
import torch

from mptorch.number import RoundMode, SaturationMode
from mptorch.quant import binaryK_quantize
from tests.markers import available_devices, has_float64

DETERMINISTIC = [rm for rm in RoundMode if rm is not RoundMode.SR]


def _bias(K: int, P: int, signed: bool) -> int:
    """P3109's exponent bias: half the exponent codes of a ``K``-bit format."""
    return 2 ** (K - P - 1) if signed else 2 ** (K - P)


def _exp_bits(K: int, P: int, signed: bool) -> int:
    """Width of the exponent field: ``K`` less ``P - 1`` and the sign bit, if any."""
    return K - P if signed else K - P + 1


def _top_exponent(K: int, P: int, signed: bool) -> int:
    """Unbiased exponent of the format's top binade (all exponent bits set)."""
    return 2 ** _exp_bits(K, P, signed) - 1 - _bias(K, P, signed)


def _formats(signed: bool) -> list[tuple[int, int]]:
    """Every precision of the 3- to 8-bit formats, and a few wider ones.

    Only formats whose top binade binary32 can hold (exponent at most 127). A
    wider one has a largest finite value no float32 input reaches, so the
    paper's M_hi says nothing a float32 run could observe.
    """
    narrow = [(K, P) for K in range(3, 9) for P in range(1, K if signed else K + 1)]
    wide = [(10, 4), (12, 7), (16, 8), (16, 11)]
    return [(K, P) for K, P in narrow + wide if _top_exponent(K, P, signed) <= 127]


# --- the paper, transcribed ---------------------------------------------------


def _max_finite(K: int, P: int, signed: bool, finite: bool) -> float:
    """M_hi: the value of the highest code point that is neither infinity nor NaN.

    The code splits into an exponent field ``E`` and a ``P - 1`` bit trailing
    significand ``T``, and decodes as a subnormal when ``E == 0`` (Fig. 7).
    """
    B = _bias(K, P, signed)
    top = (2 ** (K - 1) - 1) if signed else (2**K - 2)  # the top code that is not NaN
    if not finite:
        top -= 1  # the extended domain's +infinity
    E, T = divmod(top, 2 ** (P - 1))
    significand = T * 2.0 ** (1 - P) if E == 0 else 1 + T * 2.0 ** (1 - P)
    return math.ldexp(significand, max(E, 1) - B)


def _round_to_precision(x: torch.Tensor, P: int, B: int, mode: RoundMode) -> torch.Tensor:
    """omega-RoundToPrecision (Fig. 1) with RoundAway (Fig. 2), on float64 values.

    ``x`` is scaled to an integer significand ``S`` plus a fraction ``eta`` at the
    exponent the format gives it (clamped to the subnormal exponent ``1 - B``), and
    ``away`` says per mode whether to step from ``floor(S)`` to the next code.
    Parity is the significand's last bit. At ``P = 1`` there is no such bit and
    it is the parity of the biased exponent, with the zero counted as even.
    """
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
    """omega-Saturate (section IV-B) into [M_lo, M_hi].

    An unsigned format sends every negative value to 0 first.
    """
    lo = -hi if signed else 0.0
    if not signed:
        z = torch.where(z < 0, torch.zeros_like(z), z)
    clamped = z.clamp(lo, hi)  # NaN stays NaN
    if mode is SaturationMode.SAT_FINITE:
        return clamped
    if mode is SaturationMode.SAT_PROPAGATE:
        return torch.where(torch.isinf(z), z, clamped)
    return torch.where((z > hi) | (z < lo), torch.copysign(torch.full_like(z, math.inf), z), z)


def _project(
    x: torch.Tensor, K: int, P: int, signed: bool, rounding, saturation, bias: int | None = None
) -> torch.Tensor:
    """omega-Saturate(omega-RoundToPrecision(x)), at P3109's bias or ``bias``.

    Another bias moves the largest finite value by the difference, a power of
    two. Rounding is exact in the paper but done in float64 here, which has no
    2**1024: a finite input near float64's largest value that rounds up comes
    back infinite from ``_round_to_precision``, where the paper has a finite
    result beyond the format's range, which SatPropagate clamps like any
    other. float64's largest value stands in for that result, since it is
    beyond every format these tests round to.
    """
    finite = saturation is SaturationMode.SAT_FINITE
    B = _bias(K, P, signed) if bias is None else bias
    hi = math.ldexp(_max_finite(K, P, signed, finite), _bias(K, P, signed) - B)
    rounded = _round_to_precision(x, P, B, rounding)
    beyond = torch.copysign(torch.full_like(rounded, torch.finfo(torch.float64).max), rounded)
    rounded = torch.where(torch.isinf(rounded) & torch.isfinite(x), beyond, rounded)
    projected = _saturate(rounded, saturation, hi, signed)
    # Code point 0 decodes to a zero with no sign.
    return torch.where(projected == 0, torch.zeros_like(projected), projected)


# --- inputs -------------------------------------------------------------------


def _inputs(K: int, P: int, signed: bool) -> torch.Tensor:
    """Eight float32 values per unit in the last place, from three binades below
    the top of the format to three above it, and the float32 extremes.

    The words are built from the binary32 fields, ``(e + 127) << 23 | m``. The
    mantissa step keeps three bits below the format's last place, which puts
    every tie and both sides of it among the inputs.
    """
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


def _formats64(signed: bool) -> list[tuple[int, int]]:
    """Formats whose top binade binary64 holds, up to K = 64: every precision
    of the 3- to 8-bit formats, and wider ones up to binary64's 53 bits."""
    narrow = [(K, P) for K in range(3, 9) for P in range(1, K if signed else K + 1)]
    wide = [(16, 8), (16, 11), (24, 13), (32, 24), (40, 30), (48, 40), (56, 45), (63, 53), (64, 53)]
    return [(K, P) for K, P in narrow + wide if _top_exponent(K, P, signed) <= 1023]


def _inputs64(K: int, P: int, signed: bool) -> torch.Tensor:
    """float64 inputs around the top of the format, three binades either side.

    Per binade, the lowest and highest 64 codes of the format, each at the
    places where rounding decides (on the grid and one binary64 ulp above it, at
    the midpoint to the next code and one ulp either side of it, and one ulp
    below the next code), plus 256 random mantissas, and binary64's extremes,
    float32's largest value among them. Most of these inputs have bits float32
    cannot hold.
    """
    e_top = _top_exponent(K, P, signed)
    shift = 52 - (P - 1)  # the mantissa bits below the format's last place
    full, half = 1 << shift, (1 << shift) >> 1
    offsets = sorted({o for o in (0, 1, half - 1, half, half + 1, full - 1) if 0 <= o < full})
    n_codes = 1 << (P - 1)
    codes = sorted(set(range(min(64, n_codes))) | set(range(max(0, n_codes - 64), n_codes)))
    mantissas = torch.tensor(
        sorted({(c << shift) + o for c in codes for o in offsets}), dtype=torch.int64
    )
    g = torch.Generator().manual_seed(K * 100 + P)
    words = []
    for e in range(max(e_top - 3, -1022), min(e_top + 3, 1023) + 1):
        man = torch.cat([mantissas, torch.randint(0, 1 << 52, (256,), generator=g)])
        words.append(((e + 1023) << 52) | man)
    x = torch.cat(words).view(torch.float64)
    extremes = torch.tensor(
        [
            torch.finfo(torch.float64).max,
            2.0**1023,
            3.4028234663852886e38,
            math.inf,
            math.nan,
            1.0,
            0.0,
        ],
        dtype=torch.float64,
    )
    x = torch.cat([x, extremes])
    return torch.cat([x, -x]) if signed else x


def _mismatches(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Mask of elements that differ, counting a zero's sign and pairing NaN with NaN."""
    same = (got == want) & (torch.signbit(got) == torch.signbit(want))
    return ~(same | (got.isnan() & want.isnan()))


# --- tests --------------------------------------------------------------------


# Table I of the paper, as (P, extended, finite): the largest finite value of
# every 4-bit signed format in both domains. It pins _max_finite itself before
# anything is checked against it.
TABLE_I = [(1, 4.0, 8.0), (2, 2.0, 3.0), (3, 1.5, 1.75)]


@pytest.mark.parametrize("P,extended,finite", TABLE_I)
def test_reference_largest_finite_matches_table_i(P, extended, finite):
    """The transcribed M_hi reproduces the paper's own table, so a wrong
    reference cannot agree with a wrong kernel unnoticed."""
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
    """The largest finite value of named P3109 formats is a fixed point, and an
    input far above it saturates or overflows as the mode says.

    The literals are hand-decoded from the top code points, so they catch a range
    counted the IEEE 754 way, which reserves a whole exponent code for infinity
    and NaN where P3109 reserves one or two code points.
    """
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
@pytest.mark.parametrize("carrier", ["binary32", "binary64"])
def test_top_of_range_matches_p3109(device, carrier, signed, saturation_mode):
    """Every deterministic mode agrees with the transcription around the top binade.

    It catches a clip that compares before rounding, a mode that saturates where
    SatNone must overflow, and a largest finite value off by a code point.
    """
    failures = []
    wide = carrier == "binary64"
    for K, P in _formats64(signed) if wide else _formats(signed):
        x = _inputs64(K, P, signed) if wide else _inputs(K, P, signed)
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
@pytest.mark.parametrize("carrier", ["binary32", "binary64"])
def test_zero_and_far_below_the_smallest_subnormal(device, carrier, signed):
    """A zero of either sign becomes +0.0, and a nonzero input far below the grid
    becomes +0.0 or the smallest subnormal, as RoundAway decides.

    The inputs include subnormals of the carrier. A cast that tests the rounded
    word for zero, rather than the input, lets a zero or a carrier subnormal
    through with its sign or as a value off the grid.
    """
    if carrier == "binary64":
        tiny = [0.0, 5e-324, 1.0e-310, 2.0**-1000, 2.0**-600, 2.0**-160 * (1 + 2.0**-40)]
        formats = [(8, 4), (8, 3), (6, 1), (16, 8), (32, 24), (40, 30)]
        dtype = torch.float64
    else:
        tiny = [0.0, 1.4e-45, 1.0e-40, 2.0**-100, 2.0**-60]
        formats = [(8, 4), (8, 3), (6, 3), (5, 2), (6, 1)]
        dtype = torch.float32
    x = torch.tensor(tiny + [-v for v in tiny] if signed else tiny, dtype=dtype)
    failures = []
    # Keep the formats whose smallest value, 2^(2 - bias - P), is at or above
    # 2^-1021, the lowest the binary64 cast places a subnormal grid (it derives
    # the grid from the input's exponent field, which every binary64 subnormal
    # shares): bias + P <= 1023. The unsigned 40-bit format, with eleven exponent
    # bits and a bias of 1024, fails it and is skipped.
    for K, P in [(K, P) for K, P in formats if _bias(K, P, signed) + P <= 1023]:
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
    """An SR result is the projection of rounding toward zero or away from it.

    Stochastic rounding picks one of the two directions per input, and the result
    goes through the same omega-Saturate either way. This catches an SR path with
    its own, different, clip at the top of the range.
    """
    failures = []
    for K, P, carrier in [
        (8, 4, "binary32"),
        (8, 3, "binary32"),
        (6, 3, "binary32"),
        (5, 2, "binary32"),
        (5, 1, "binary32"),
        (8, 4, "binary64"),
        (16, 11, "binary64"),
        (40, 30, "binary64"),
        (60, 50, "binary64"),
    ]:
        wide = carrier == "binary64"
        if wide and not has_float64(device):
            continue
        x = _inputs64(K, P, signed) if wide else _inputs(K, P, signed)
        xd = x.double()
        toward = _project(xd, K, P, signed, RoundMode.RZ, saturation_mode)
        up = _project(xd, K, P, signed, RoundMode.RU, saturation_mode)
        down = _project(xd, K, P, signed, RoundMode.RD, saturation_mode)
        away = torch.where(xd > 0, up, down)
        got = binaryK_quantize(
            x.to(device),
            K,
            P,
            # The random bits come out of the carrier's mantissa below the format's
            # last place: binary64 has 52 - (P - 1) of them, two at P = 51.
            prng_bits=min(8, 52 - (P - 1)) if wide else 8,
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


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("saturation_mode", list(SaturationMode))
@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
def test_nan_passes_through_whole(device, signed, saturation_mode):
    """A NaN input comes back bit for bit (payload, signalling bit and sign) under
    every rounding mode and on both backends.

    The cast simulates a value, not an encoding, so P3109's single NaN is
    whichever NaN came in. The directed modes are where that is not free: a NaN
    compares false against zero, so it takes the arm that rounds a magnitude and
    negates it back. A float negation on the device may canonicalize a NaN's
    payload and sign, so a cast that negates with ``-x`` returns ``nan`` as
    ``-nan`` or the reverse, depending on how the compiler inlines it.
    ``bit_helper.h``'s ``flip_sign`` and ``negate_magnitude`` negate on the word.
    """
    # Signalling and quiet NaNs of both signs, with small, full and mixed payloads.
    nans32 = torch.tensor(
        [0x7F800001, 0xFF800001, 0x7FC00000, 0xFFC00000, 0x7FFFFFFF, 0xFF812345],
        dtype=torch.int64,
    ).to(torch.int32)
    # The same in float64, plus payloads in the low 32 bits of the word, which a
    # cast that went through float32 would lose. The sign bit is or-ed in below.
    nans64 = torch.tensor(
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
    nans64 = torch.cat([nans64, nans64 | torch.tensor(-(2**63), dtype=torch.int64)])
    failures = []
    for (K, P), nans in [
        ((8, 4), nans32),
        ((8, 3), nans32),
        ((6, 3), nans32),
        ((5, 1), nans32),
        ((16, 11), nans32),
        ((8, 4), nans64),
        ((40, 30), nans64),
        ((63, 53), nans64),
    ]:
        if nans.dtype is torch.int64 and not has_float64(device):
            continue
        x = nans.view(torch.float32 if nans.dtype is torch.int32 else torch.float64).to(device)
        for mode in DETERMINISTIC + [RoundMode.SR]:
            got = binaryK_quantize(
                x,
                K,
                P,
                prng_bits=min(8, 52 - (P - 1)),
                is_signed=signed,
                rounding_mode=mode,
                saturation_mode=saturation_mode,
            )
            bits = got.cpu().view(nans.dtype)
            # An unsigned format turns every negative input into +0.0. A NaN with
            # its sign bit set is not negative, it is unordered, so it stays.
            bad = bits != nans
            if bad.any():
                i = int(bad.nonzero()[0])
                mask = 0xFFFFFFFF if nans.dtype is torch.int32 else 0xFFFFFFFFFFFFFFFF
                failures.append(
                    f"Binary{K}p{P}{'s' if signed else 'u'} {mode.name}: "
                    f"{int(bad.sum())} of {len(nans)} NaNs changed, e.g. "
                    f"{int(nans[i]) & mask:#x} -> {int(bits[i]) & mask:#x}"
                )
    assert not failures, "\n".join(failures)
