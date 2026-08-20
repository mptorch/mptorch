import pytest
import torch

from mptorch.number import RoundMode, SaturationMode
from mptorch.quant import superfp_quantize
from tests.markers import available_devices

# Reference 6-bit "supernormal" format used throughout this file: 1 sign bit,
# 3 exponent bits, 2 mantissa bits, 4 normal binades, bias 7. All 64
# codepoints of this exact configuration were enumerated and cross-checked
# by hand in temp/show_superfp.cpp during the format's development (see
# walkthrough_superfp.md / walkthrough_superfp2.md): 15 supernormal levels
# (2**-18 .. 2**-4), a 32-codepoint normal grid (0.125 .. 1.75 in steps of
# 2**e / 4), and zero. That makes it a convenient, independently-understood
# oracle for hand-derived expected values below.
CFG = {"man_bits": 2, "exp_bits": 3, "normal_binades": 4, "bias": 7}
SMALLEST_SUPERNORMAL = 2.0**-18  # 3.814697265625e-06
HALF_SMALLEST = 2.0**-19  # 1.9073486328125e-06, the underflow/supernormal tie boundary

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
    x = torch.tensor([x_val], dtype=dtype, device=device)
    return superfp_quantize(
        x,
        is_signed=is_signed,
        rounding_mode=mode,
        saturation_mode=saturation_mode,
        **cfg,
    ).item()


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_nan_passthrough(device, mode):
    # exact bit pattern must survive, including a custom NaN payload and
    # negative-signed NaN, regardless of rounding mode.
    import struct

    # 0x7FA51234 (a *signaling* NaN payload, quiet bit unset) is deliberately
    # avoided here: plain tensor construction alone (no quantization
    # involved) already quiets it on this platform, which is a torch/CPU
    # artifact unrelated to superfp -- verified by round-tripping
    # torch.tensor([nan]).item() with no op applied at all.
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
def test_inf_passthrough(device, mode, saturation_mode):
    for sign in (1.0, -1.0):
        inf = sign * float("inf")
        out = _quantize(inf, mode, device, saturation_mode=saturation_mode)
        assert out == inf


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_zero(device, mode):
    assert _quantize(0.0, mode, device) == 0.0
    assert _quantize(-0.0, mode, device) == 0.0


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_unsigned_rejects_negative(device, mode):
    assert _quantize(-3.5, mode, device, is_signed=False) == 0.0
    assert _quantize(-1.0e-7, mode, device, is_signed=False) == 0.0


# ---------------------------------------------------------------------------
# Normal region (man_bits=2 -> mantissa grid step of 2**e / 4 within each
# binade). 0.13 is a non-tie value between the grid points 0.125 and
# 0.15625; 0.140625 is their exact midpoint (a genuine tie). These are
# ordinary IEEE-754-style roundings (standard mantissa-LSB-parity tie-break
# for RNE), safe to hand-verify directly.
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
# Supernormal region: representable values are bare powers of two (no
# explicit mantissa), so rounding = choosing between 2**e and 2**(e+1).
# 1.2*2**-9 is a non-tie value; 1.5*2**-9 and 1.5*2**-10 are exact ties.
#
# The RNE tie rule here ("of the two bracketing exponents, keep the one
# whose *unbiased* value is even") is a property of round_bitwise_nearest_even's
# bare (0 mantissa bit) overload in bit_helper.h -- an existing primitive
# shared with binaryK's own man_bits==0 case, and already exercised against
# gfloat by test_binaryK_quantize.py's P=1 sweep. It is not something new
# introduced for superfp.
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
    assert _quantize(value, mode, device) == pytest.approx(expected)
    # antisymmetric modes must mirror on negation; RU/RD swap roles instead.
    if mode in (RoundMode.RNE, RoundMode.RNA, RoundMode.RZ, RoundMode.RO):
        assert _quantize(-value, mode, device) == pytest.approx(-expected)


# ---------------------------------------------------------------------------
# Underflow region: no graded precision, so it's a single choice between 0
# and SMALLEST_SUPERNORMAL, split at HALF_SMALLEST = 2**-19.
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
    assert _quantize(value, mode, device) == pytest.approx(expected, abs=1e-30)
    if mode in (RoundMode.RNE, RoundMode.RNA, RoundMode.RZ, RoundMode.RO):
        assert _quantize(-value, mode, device) == pytest.approx(-expected, abs=1e-30)
    elif mode is RoundMode.RU:
        # RU(-x) == -RD(x); RD(x) for these cases is always 0.0.
        assert _quantize(-value, mode, device) == pytest.approx(-0.0, abs=1e-30)
    elif mode is RoundMode.RD:
        # RD(-x) == -RU(x)
        assert _quantize(-value, mode, device) == pytest.approx(
            -_quantize(value, RoundMode.RU, device), abs=1e-30
        )


# ---------------------------------------------------------------------------
# RO's man_bits==0 normal-region branch is a distinct code path (parity of
# the target format's own biased exponent) from both the man_bits>0 normal
# branch above and the supernormal branch's grid-index parity. This config
# (normal_cutoff=0, max target_exp=4) gives enough headroom for a bump not to
# hit saturation. 6.0 is an exact tie between 4.0 and 8.0 (floor exponent 2
# is even -> RNE keeps 4.0, matching the same verified tie rule as the
# supernormal region since man_bits==0 makes this grid power-of-two spaced
# too). 5.0 is not a tie, but RO still forces the bump to 8.0 because 4.0's
# stored exponent parity (target_exp + bias = 2 + 10 = 12, even) isn't odd.
RO_ZERO_MAN_BITS_CFG = {"man_bits": 0, "exp_bits": 4, "normal_binades": 6, "bias": 10}


@pytest.mark.parametrize("device", available_devices)
def test_ro_normal_region_zero_man_bits(device):
    assert _quantize(6.0, RoundMode.RNE, device, cfg=RO_ZERO_MAN_BITS_CFG) == 4.0
    assert _quantize(6.0, RoundMode.RO, device, cfg=RO_ZERO_MAN_BITS_CFG) == 8.0
    assert _quantize(5.0, RoundMode.RO, device, cfg=RO_ZERO_MAN_BITS_CFG) == 8.0


# ---------------------------------------------------------------------------
# Saturation-mode differentiation at the overflow boundary, for every
# rounding mode (not just RNE) -- mirrors temp/test_superfp.cpp's
# test_superfp_6bit_saturation_modes, generalized across modes. Under
# OVF_INF, code 31 (011111) is reserved for +Inf, so the max finite value is
# 1.50 (code 30); under SAT_FINITE the max finite value is 1.75 (code 31).
@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_saturation_mode_differentiation(device, mode):
    assert _quantize(1.5, mode, device, saturation_mode=SaturationMode.SAT_FINITE) == 1.5
    assert _quantize(1.5, mode, device, saturation_mode=SaturationMode.OVF_INF) == 1.5

    assert _quantize(1.75, mode, device, saturation_mode=SaturationMode.SAT_FINITE) == 1.75
    assert _quantize(1.75, mode, device, saturation_mode=SaturationMode.OVF_INF) == float("inf")

    assert _quantize(2.0, mode, device, saturation_mode=SaturationMode.SAT_FINITE) == 1.75
    assert _quantize(2.0, mode, device, saturation_mode=SaturationMode.OVF_INF) == float("inf")

    assert _quantize(-5.0, mode, device, saturation_mode=SaturationMode.SAT_FINITE) == -1.75
    assert _quantize(-5.0, mode, device, saturation_mode=SaturationMode.OVF_INF) == float("-inf")


# ---------------------------------------------------------------------------
# Stochastic rounding: adapted from test_binaryK_stochastic's three
# properties (grid-point identity, RD/RU bounding, statistical
# unbiasedness), plus a superfp-specific unbiasedness check in the
# underflow region (which has no binaryK analogue, since binaryK has no
# single-boundary flush-or-not region).
GRID_POINTS = (
    [0.0]
    + [2.0**e for e in range(-18, -3)]  # supernormal levels
    + [(1 + m / 4) * 2.0**e for e in range(-3, 1) for m in range(4)]  # normal grid
)


@pytest.mark.parametrize("device", available_devices)
def test_superfp_stochastic(device):
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

    # unbiasedness in the normal region
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

    # unbiasedness in the underflow region: no binaryK analogue, since
    # binaryK's subnormal region always has graded precision.
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
