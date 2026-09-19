"""
binaryK_quantize against gfloat's rounding of IEEE P3109's binaryK formats.

gfloat rounds a Python float onto a P3109 format exactly, so it is the oracle
for every rounding mode below the largest finite value: over float32, float16
and bfloat16 inputs rounded in binary32 (K up to 8, every P), and over float64
inputs rounded in binary64 for formats binary32 cannot carry (up to 53 bits of
precision and ten exponent bits). At and past the largest finite value gfloat
overflows the IEEE 754 way under directed rounding where P3109 saturates, so
that end belongs to tests/test_binaryk_p3109.py. Two further tests hold what
a value oracle cannot: stochastic rounding's three properties (a grid point is
left alone, every result is one of the input's two neighbours, the mean is the
input) and the handling of NaN, infinities and the saturation modes.
"""

import math
import random

import pytest
import torch
from gfloat import RoundMode, Signedness, round_float
from gfloat.formats import format_info_p3109

import mptorch
from mptorch.quant import binaryK_quantize
from tests.markers import available_devices, float64_devices
from tests.quant import bits_to_float, float_to_bits


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("K", [4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rounding_mode",
    [
        (RoundMode.TiesToEven, mptorch.number.RoundMode.RNE),
        (RoundMode.TiesToAway, mptorch.number.RoundMode.RNA),
        (RoundMode.TowardPositive, mptorch.number.RoundMode.RU),
        (RoundMode.TowardNegative, mptorch.number.RoundMode.RD),
        (RoundMode.TowardZero, mptorch.number.RoundMode.RZ),
        (RoundMode.ToOdd, mptorch.number.RoundMode.RO),
    ],
)
@pytest.mark.parametrize("signedness", [(Signedness.Signed, True), (Signedness.Unsigned, False)])
def test_binaryK_vs_gfloat(device, K, dtype, rounding_mode, signedness):
    """Every rounding mode agrees with gfloat word for word over a sweep of the
    format's range, in each storage dtype: a wrong tie rule, an off-by-one at
    the subnormal floor or a sign lost through a 16-bit dtype fails here."""
    for P in range(1, K):
        fi = format_info_p3109(K, P, signedness=signedness[0])
        bias = 2 ** (K - P - 1)
        # float32 words at eight per unit in the format's last place (a unit
        # is 2**(24 - P) words, the increment 2**(21 - P)), starting four
        # binades below 2**(1 - bias - (P - 1)), the smallest subnormal at the
        # signed bias, for 2**(K + 2) samples: the subnormals and most of the
        # normal range, short of the top.
        start_value = 2 ** (1 - bias - P - 3)
        istart_value = float_to_bits(start_value)
        increment = 0x7FFFFFFF & (1 << (22 - (P - 1) - 2))

        vals_to_test = [bits_to_float(istart_value + i * increment) for i in range(2 ** (K + 2))]
        if signedness[1]:
            vals_to_test_neg = [-x for x in vals_to_test]
            vals_to_test = [vals_to_test, vals_to_test_neg]

        # Held in the dtype under test, so a float16 or bfloat16 sweep first
        # rounds the words onto that dtype's grid.
        x = torch.tensor(vals_to_test, dtype=dtype).to(device)

        # gfloat rounds a Python float, so the reference runs on the same
        # tensor's float32 values and is stored back in the dtype, as the
        # quantizer's own result is.
        gqx = x.clone().detach().to("cpu").to(torch.float32)
        gqx.apply_(
            lambda x, fi=fi: round_float(
                fi,
                x,
                rnd=rounding_mode[0],
            )
        )
        gqx = gqx.to(device).to(dtype)
        qx = binaryK_quantize(x, K, P, rounding_mode=rounding_mode[1], is_signed=signedness[1])

        assert torch.all(qx == gqx)


# gfloat's rounding modes paired with mptorch's, for the binary64 sweep.
GFLOAT_MODES = [
    (RoundMode.TiesToEven, mptorch.number.RoundMode.RNE),
    (RoundMode.TiesToAway, mptorch.number.RoundMode.RNA),
    (RoundMode.TowardPositive, mptorch.number.RoundMode.RU),
    (RoundMode.TowardNegative, mptorch.number.RoundMode.RD),
    (RoundMode.TowardZero, mptorch.number.RoundMode.RZ),
    (RoundMode.ToOdd, mptorch.number.RoundMode.RO),
]


def _wide_inputs(K: int, P: int, signed: bool) -> list[float]:
    """float64 values across a format binary64 carries: random significands at
    every exponent from three binades under its smallest value to one under
    its largest finite one (above that gfloat overflows the IEEE 754 way,
    which tests/test_binaryk_p3109.py covers instead), and the format's own
    grid points, a float64 ulp either side of them, and the midpoints between
    them, at the bottom and the top of the normals and across the subnormal
    boundary."""
    exp_bits = K - P if signed else K - P + 1
    bias = 2 ** (exp_bits - 1)
    bottom, top = 1 - bias - (P - 1), 2**exp_bits - 1 - bias
    rng = random.Random(K * 100 + P)
    out = [
        math.ldexp(1 + rng.getrandbits(52) / 2**52, rng.randrange(bottom - 3, top))
        for _ in range(2000)
    ]
    for e in (1 - bias, 2 - bias, top - 1):
        step = e - (P - 1)
        for code in list(range(2 ** (P - 1), 2 ** (P - 1) + 16)) + list(range(2**P - 16, 2**P)):
            v = math.ldexp(code, step)
            half = math.ldexp(2 * code + 1, step - 1)
            out += [v, math.nextafter(v, math.inf), math.nextafter(v, 0.0), half]
            out += [math.nextafter(half, math.inf), math.nextafter(half, 0.0)]
    for code in range(1, 64):  # the subnormals and the boundary to the normals
        out.append(math.ldexp(code, 1 - bias - (P - 1)))
    return out + [-v for v in out] if signed else out


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize(
    "K, P, signed",
    # every one inside binary64's bounds at P3109's bias, the last two of each
    # signedness with ten exponent bits, its most
    [(16, 8, True), (40, 30, True), (48, 40, True), (62, 52, True), (63, 53, True)]
    + [(16, 8, False), (39, 30, False), (48, 40, False), (61, 52, False), (62, 53, False)],
)
def test_binaryK_vs_gfloat_in_binary64(device, K, P, signed):
    """A float64 tensor is rounded in binary64, so gfloat, which rounds a
    Python float exactly for formats up to 53 bits of precision, is fed the
    float64 values themselves, over formats binary32 cannot carry."""
    exp_bits = K - P if signed else K - P + 1
    assert 2 ** (exp_bits - 1) + P <= 1023 and 2 ** (exp_bits - 1) - 1 <= 1023
    fi = format_info_p3109(K, P, signedness=Signedness.Signed if signed else Signedness.Unsigned)
    values = _wide_inputs(K, P, signed)
    x = torch.tensor(values, dtype=torch.float64, device=device)
    for gmode, mode in GFLOAT_MODES:
        want = torch.tensor([round_float(fi, v, rnd=gmode) for v in values], dtype=torch.float64)
        got = binaryK_quantize(x, K, P, rounding_mode=mode, is_signed=signed).cpu()
        bad = got != want
        if bad.any():
            i = int(bad.nonzero()[0])
            raise AssertionError(
                f"{mode.name}: {int(bad.sum())} differ, e.g. {values[i]!r} -> "
                f"{got[i].item()!r}, gfloat {want[i].item()!r}"
            )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("K", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("signedness", [(Signedness.Signed, True), (Signedness.Unsigned, False)])
def test_binaryK_stochastic(device, K, dtype, signedness):
    """Stochastic rounding's three properties: a grid point comes back
    unchanged, every result is one of the input's two neighbours, and the
    mean over a million draws of one value is that value."""
    P = K - 2 if K > 4 else K - 1

    # Random inputs inside (-0.9, 0.9), or (0, 0.9) unsigned, so none reaches
    # the format's saturation logic.
    if signedness[1]:
        x_rand = (torch.rand(10000, dtype=dtype, device=device) * 1.8 - 0.9).requires_grad_(False)
    else:
        x_rand = (torch.rand(10000, dtype=dtype, device=device) * 0.9).requires_grad_(False)

    # One random bit per significand bit the format drops from binary32's 23.
    prng_bits = 23 - (P - 1)

    # The two neighbours of every input, by directed rounding.
    q_rd = binaryK_quantize(
        x_rand, K, P, rounding_mode=mptorch.number.RoundMode.RD, is_signed=signedness[1]
    )
    q_ru = binaryK_quantize(
        x_rand, K, P, rounding_mode=mptorch.number.RoundMode.RU, is_signed=signedness[1]
    )

    # RD's results are on the format's grid by construction.
    x_grid = q_rd.clone()
    q_sr_grid = binaryK_quantize(
        x_grid,
        K,
        P,
        prng_bits=prng_bits,
        rounding_mode=mptorch.number.RoundMode.SR,
        is_signed=signedness[1],
    )

    # Property 1: a grid point is exact, so no random bit may move it.
    assert torch.all(q_sr_grid == x_grid), (
        "Stochastic rounding altered an exactly representable grid point!"
    )

    # SR over the inputs themselves, which are generally off the grid.
    q_sr_rand = binaryK_quantize(
        x_rand,
        K,
        P,
        prng_bits=prng_bits,
        rounding_mode=mptorch.number.RoundMode.SR,
        is_signed=signedness[1],
    )

    # Property 2: a draw only ever picks one of the two neighbours.
    valid_bounds = (q_sr_rand == q_rd) | (q_sr_rand == q_ru)
    if not torch.all(valid_bounds):
        idx = (~valid_bounds).nonzero(as_tuple=True)[0]
        print(f"FAILED BOUNDS for {len(idx)} elements. First few:")
        for i in idx[:5]:
            print(
                f"x={x_rand[i].item()}, RD={q_rd[i].item()}, "
                f"RU={q_ru[i].item()}, SR={q_sr_rand[i].item()}"
            )

    assert torch.all(valid_bounds), (
        "Stochastic rounding produced a value outside the [RD, RU] bounds!"
    )

    # Property 3: unbiasedness. A million draws of one value off the grid,
    # inside the format's normal range.
    test_val = 0.333333
    N_samples = 1_000_000
    x_large = torch.full((N_samples,), test_val, dtype=dtype, device=device)

    q_sr_large = binaryK_quantize(
        x_large,
        K,
        P,
        prng_bits=prng_bits,
        rounding_mode=mptorch.number.RoundMode.SR,
        is_signed=signedness[1],
    )

    # The expected value of SR(x) is x.
    mean_val = q_sr_large.mean().item()

    # Loose against the standard error (the grid step is at most 2**-4 here
    # and the sample a million draws, so the error is below 1e-4): only a
    # systematic bias fails it.
    tolerance = 0.05

    assert abs(mean_val - test_val) < tolerance, (
        f"Stochastic rounding expectation biased! Expected {test_val}, got mean {mean_val}"
    )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("K, P", [(8, 4), (8, 3), (6, 3), (24, 8)])
@pytest.mark.parametrize("rounding_mode", list(mptorch.number.RoundMode))
@pytest.mark.parametrize("saturation_mode", list(mptorch.number.SaturationMode))
@pytest.mark.parametrize("is_signed", [True, False])
def test_binaryK_nonfinite_inputs(device, K, P, rounding_mode, saturation_mode, is_signed):
    """NaN always passes through; an infinity does too, except under SAT_FINITE.

    SAT_FINITE's contract is that every value it returns is finite, so there
    an infinity saturates to exactly what an overflowing finite input does.
    The formats sit on both sides of the gate that admits the float-arithmetic
    fast path of mptorch/csrc/common/cast_binaryK.h, so both the float and the
    integer bodies are under test.
    """
    huge = torch.finfo(torch.float32).max
    x = torch.tensor([float("inf"), -float("inf"), float("nan"), huge, -huge], device=device)
    q = binaryK_quantize(
        x,
        K,
        P,
        prng_bits=8 if rounding_mode is mptorch.number.RoundMode.SR else 0,
        rounding_mode=rounding_mode,
        is_signed=is_signed,
        saturation_mode=saturation_mode,
    )
    assert torch.isnan(q[2])
    if not is_signed:
        assert q[1] == 0.0 and q[4] == 0.0
    if saturation_mode is mptorch.number.SaturationMode.SAT_FINITE:
        assert torch.isfinite(q[[0, 1, 3, 4]]).all()
        assert q[0] == q[3] and q[1] == q[4]
        assert q[0] > 0
    else:
        assert q[0] == float("inf")
        assert q[1] == (0.0 if not is_signed else -float("inf"))
