"""
What the elementwise quantize entry points do with an operand's dtype.

A float32, float16 or bfloat16 tensor rounds in binary32, whose values all
three are. A float64 tensor rounds in binary64 (dev/binary64_carrier_plan.md,
phase 3): its kernels are instantiated for ``double`` and compute in it, where
they used to narrow the tensor to float32 first and widen the result back. So
a float64 input is rounded once, directly, and a format that needs more than
binary32 -- 24 bits of precision, seven exponent bits, a floor at 2**-125 --
can be reached.

Four things follow, and each has its tests below.

* **The binary32 image.** A float64 tensor holding float32 values, quantized
  to a format binary32 simulates faithfully, gives exactly the float32
  result: the format's answer does not depend on which carrier held the input.
  ``dev/benchmarks/cast_all_modes_sweep.cu image64`` proves that for the casts
  over every float32 input; these hold the kernels' dispatch, vector loads and
  scalar tail to it.
* **Rounding once.** Where the float64 value is not a float32 value the two
  paths legitimately differ, because narrowing was a rounding of its own.
* **Formats past binary32.** Checked through ``torch.ops`` against
  ``tests/test_binaryk_p3109.py``'s transcription of P3109, which rounds
  float64 values exactly, and through the Python wrapper, which holds a
  float64 tensor's format to binary64's bounds (phase 5) and so must let
  every one of them through without a word.
* **The old arithmetic, on request.** ``carrier="binary32"`` narrows a float64
  tensor, rounds it in binary32 and widens the result, which is exactly what
  every float64 call did before phase 3 -- random draws included.

Stochastic rounding draws a 64-bit word per float64 element -- words
``2 * (j & 1)`` and the next of Philox block ``j >> 1`` (``common/philox.h``)
-- so it is checked by property: every result is one of the input's two
neighbours, and each element's draw is a function of its own index.
"""

import math
import warnings
from typing import Any

import pytest
import torch

from mptorch.number import FormatRangeWarning, RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import binaryK_quantize, superfp_quantize
from tests.markers import available_devices
from tests.test_binaryk_p3109 import _project

# 100_003 is prime: neither the float4 body nor any dtype's vector width
# divides it, so the scalar tail runs on every backend and dtype.
SIZE = 100_003

DETERMINISTIC = [rm for rm in RoundMode if rm is not RoundMode.SR]
SATURATION_MODES = list(SaturationMode)


def _quant_calls(x):
    """Both elementwise quantize entry points, keyed by op name."""
    return {
        "binaryK": lambda **kw: binaryK_quantize(x, K=8, P=4, **kw),
        "superfp": lambda **kw: superfp_quantize(
            x, man_bits=3, exp_bits=4, normal_binades=1, bias=7, **kw
        ),
    }


OP_NAMES = list(_quant_calls(torch.empty(0)).keys())


def _float32_values(device, n=SIZE):
    """float64 tensor of float32 values: x8 so the format's normal,
    supernormal and saturating regions are all hit, and a few specials."""
    torch.manual_seed(7)
    x = torch.randn(n, device=device, dtype=torch.float32) * 8.0
    specials = torch.tensor(
        [0.0, -0.0, math.inf, -math.inf, math.nan, 1.4e-45, -1.4e-45, 3.4e38, 1.0, 1.0625],
        device=device,
    )
    x[: specials.numel()] = specials
    return x.double()


def _assert_same_words(a, b, ctx=""):
    """Raw-word comparison: NaN is not equal to itself and -0.0 is equal to
    0.0, and this is a claim about bits."""
    assert a.dtype == b.dtype, ctx
    bits = torch.int64 if a.dtype is torch.float64 else torch.int32
    same = a.view(bits) == b.view(bits)
    if not bool(same.all()):
        i = int((~same).nonzero()[0])
        raise AssertionError(f"{ctx}: {int((~same).sum())} differ, first {i}: {a[i]!r} vs {b[i]!r}")


# --- the binary32 image -------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("round_mode", DETERMINISTIC)
def test_float64_of_float32_values_matches_float32(device, op, round_mode):
    x64 = _float32_values(device)
    out64 = _quant_calls(x64)[op](rounding_mode=round_mode)
    out32 = _quant_calls(x64.float())[op](rounding_mode=round_mode)
    assert out64.dtype is torch.float64, "float64 in, float64 out"
    assert out32.dtype is torch.float32
    _assert_same_words(out64, out32.double(), f"{op} {round_mode.name}")


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("saturation_mode", SATURATION_MODES)
def test_float64_of_float32_values_across_saturation_modes(device, op, saturation_mode):
    # saturation only decides anything for operands past the format's range,
    # which is what the x8 scale is for
    x64 = _float32_values(device)
    kw = {"saturation_mode": saturation_mode}
    out64 = _quant_calls(x64)[op](**kw)
    out32 = _quant_calls(x64.float())[op](**kw)
    _assert_same_words(out64, out32.double(), f"{op} {saturation_mode.name}")


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("subnormals_mode", list(SubnormalsMode))
def test_float64_of_float32_values_across_subnormals_modes(device, subnormals_mode):
    # /64 as well, so the values below each mode's floor are reached -- divided
    # in float32, or the smallest subnormal's quotient would not be a float32 value
    x32 = _float32_values(device).float()
    x64 = torch.cat([x32, x32 / 64.0]).double()
    for round_mode in DETERMINISTIC:
        out64, out32 = (
            binaryK_quantize(t, K=8, P=4, subnormals_mode=subnormals_mode, rounding_mode=round_mode)
            for t in (x64, x64.float())
        )
        _assert_same_words(out64, out32.double(), f"{subnormals_mode.name} {round_mode.name}")


# --- rounding once -------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    ("round_mode", "x", "via_float32", "direct"),
    [
        # 1.0625 is the tie between 1.0 and 1.125, and float32 has no 2**-30
        # to break it with: narrowed, RNE takes the even 1.0
        (RoundMode.RNE, 1.0625 + 2.0**-30, 1.0, 1.125),
        # narrowed, 1.0 is on the grid and RU has nothing to round
        (RoundMode.RU, 1.0 + 2.0**-30, 1.0, 1.125),
        (RoundMode.RD, -(1.0 + 2.0**-30), -1.0, -1.125),
    ],
)
def test_float64_is_rounded_once(device, round_mode, x, via_float32, direct):
    t = torch.tensor([x], device=device, dtype=torch.float64)
    got = binaryK_quantize(t, K=8, P=4, rounding_mode=round_mode)
    assert got.item() == direct
    assert binaryK_quantize(t.float(), K=8, P=4, rounding_mode=round_mode).item() == via_float32


@pytest.mark.parametrize("device", available_devices)
def test_float64_is_rounded_once_superfp(device):
    # superfp m3e4n1b7's one normal binade is [256, 512) in steps of 32, and
    # 272 is the tie between 256 and 288 that float32 cannot see past
    t = torch.tensor([272.0 + 2.0**-22], device=device, dtype=torch.float64)
    assert superfp_quantize(t, 3, 4, 1, 7).item() == 288.0
    assert superfp_quantize(t.float(), 3, 4, 1, 7).item() == 256.0


# --- formats past binary32 ----------------------------------------------------

# (K, P, bias, signed): each needs binary64 -- precision past 24 bits, an
# exponent field past 7 bits, or a smallest value below 2**-125 -- and each is
# inside binary64's own bounds, two of them on its edges.
WIDE_FORMATS = [
    (40, 30, 512, True),  # 30 bits of precision, ten exponent bits
    (63, 53, 511, True),  # binary64's full precision
    (63, 53, 970, True),  # ... with its smallest value at 2**-1021, binary64's faithful floor
    (11, 1, 1022, True),  # P = 1 on that floor
    (40, 31, 512, False),
    (16, 12, 8, True),  # a format binary32 holds too, beside them
]


def _wide_inputs(K, P, bias, signed, device, n=65_536):
    """float64 values across the format's range and a little past both ends:
    random 53-bit significands at every exponent, the format's own grid points
    and the midpoints between them, and the specials."""
    g = torch.Generator().manual_seed(K * 1000 + P)
    exp_bits = K - P if signed else K - P + 1
    top = 2**exp_bits - 1 - bias
    bottom = 1 - bias - (P - 1)
    exps = torch.randint(max(bottom - 3, -1074), min(top + 2, 1023) + 1, (n,), generator=g)
    sig = torch.randint(2**52, 2**53, (n,), generator=g, dtype=torch.int64).double()
    random = torch.ldexp(sig, exps - 52)
    codes = torch.randint(2 ** (P - 1), 2**P, (n,), generator=g, dtype=torch.int64).double()
    step = torch.clamp(exps, min=1 - bias) - (P - 1)
    on_grid = torch.ldexp(codes, step)
    midpoint = torch.ldexp(2 * codes + 1, step - 1)
    largest = torch.finfo(torch.float64).max
    specials = torch.tensor(
        [0.0, -0.0, math.inf, -math.inf, math.nan, 5e-324, largest], dtype=torch.float64
    )
    x = torch.cat([random, on_grid, midpoint, specials])
    sign = torch.where(torch.rand(x.shape, generator=g) < 0.5, -1.0, 1.0).double()
    return (x * sign).to(device)


def _raw_binaryK(x, K, P, bias, signed, round_mode, saturation, prng_bits=0):
    return torch.ops.mptorch.binaryK_quant.default(
        x,
        K,
        P,
        bias,
        prng_bits,
        signed,
        round_mode.value,
        saturation.value,
        SubnormalsMode.SUBNORMALS.value,
    )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(("K", "P", "bias", "signed"), WIDE_FORMATS)
@pytest.mark.parametrize("saturation", SATURATION_MODES)
def test_float64_reaches_formats_past_binary32(device, K, P, bias, signed, saturation):
    x = _wide_inputs(K, P, bias, signed, device)
    for round_mode in DETERMINISTIC:
        got = _raw_binaryK(x, K, P, bias, signed, round_mode, saturation)
        want = _project(x.cpu(), K, P, signed, round_mode, saturation, bias).to(device)
        ctx = f"K={K} P={P} bias={bias} {round_mode.name} {saturation.name}"
        _assert_same_words(got, want, ctx)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(("K", "P", "bias", "signed"), WIDE_FORMATS)
def test_the_wrapper_holds_float64_to_binary64s_bounds(device, K, P, bias, signed):
    """Every format above is inside binary64's bounds, two on its edges, so the
    wrapper passes a float64 tensor straight to the op -- and a float32 one
    that binary32 cannot carry is refused or warned about, not rounded."""
    x = _wide_inputs(K, P, bias, signed, device)
    sat = SaturationMode.SAT_FINITE
    kw: dict[str, Any] = dict(bias=bias, is_signed=signed, saturation_mode=sat)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FormatRangeWarning)
        got = binaryK_quantize(x, K, P, **kw)
    _assert_same_words(got, _raw_binaryK(x, K, P, bias, signed, RoundMode.RNE, sat), f"K={K} P={P}")
    if P > 24:
        with pytest.raises(ValueError, match="bits of precision"):
            binaryK_quantize(x.float(), K, P, **kw)


# --- the old arithmetic, on request --------------------------------------------


# Values float32 cannot hold, each a hair off a point of both formats' grids
# (binaryK 8p4's 1.0 and the 1.0625 midpoint, superfp m3e4n1b7's 256 and the
# 272 midpoint), so that narrowing moves every rounding mode's answer.
WITNESSES = [
    1.0 + 2.0**-30,  # RU, RO
    1.0 - 2.0**-30,  # RZ
    -(1.0 + 2.0**-30),  # RD
    1.0625 + 2.0**-30,  # RNE
    1.0625 - 2.0**-30,  # RNA
    272.0 + 2.0**-22,  # RNE
    272.0 - 2.0**-22,  # RNA
    256.0 + 2.0**-22,  # RO
]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("round_mode", list(RoundMode))
def test_binary32_carrier_is_the_narrowed_float32_call(device, op, round_mode):
    """Bit for bit, SR included: the same float32 kernel on the same narrowed
    values, drawing the same words, as every float64 call did before phase 3."""
    x64 = torch.randn(SIZE, device=device, dtype=torch.float64) * 8.0
    x64[: len(WITNESSES)] = torch.tensor(WITNESSES, dtype=torch.float64)
    kw = dict(rounding_mode=round_mode, prng_bits=12 if round_mode is RoundMode.SR else 0)
    torch.manual_seed(99)
    got = _quant_calls(x64)[op](carrier="binary32", **kw)
    torch.manual_seed(99)
    want = _quant_calls(x64.float())[op](**kw).double()
    assert got.dtype is torch.float64
    _assert_same_words(got, want, f"{op} {round_mode.name}")
    # ... which is not what a float64 call without it computes
    torch.manual_seed(99)
    assert not torch.equal(_quant_calls(x64)[op](**kw), got)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_binary32_carrier_of_a_strided_float64_tensor(device, op):
    base = torch.randn(2 * 4099, device=device, dtype=torch.float64).reshape(4099, 2) * 8.0
    strided = base[:, 1]
    got = _quant_calls(strided)[op](carrier="binary32")
    _assert_same_words(got, _quant_calls(strided.float())[op]().double(), op)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_a_carrier_a_dtype_already_has_changes_nothing(device, op, dtype):
    x = (torch.randn(4096, device=device) * 8).to(dtype)
    carrier = "binary64" if dtype is torch.float64 else "binary32"
    got = _quant_calls(x)[op](carrier=carrier)
    assert got.dtype is dtype
    _assert_same_words(got, _quant_calls(x)[op](), f"{op} {dtype}")


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_binary64_carrier_needs_a_float64_tensor(device, dtype):
    x = torch.ones(8, device=device, dtype=dtype)
    with pytest.raises(ValueError, match="only a float64 operand"):
        binaryK_quantize(x, 8, 4, carrier="binary64")
    with pytest.raises(ValueError, match="carrier must be"):
        superfp_quantize(x, 3, 4, 1, 7, carrier="float64")


# --- stochastic rounding ------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(("K", "P", "bias", "signed"), WIDE_FORMATS)
def test_float64_sr_lands_on_a_neighbour(device, K, P, bias, signed):
    x = _wide_inputs(K, P, bias, signed, device)
    prng_bits = min(30, 52 - (P - 1))
    sat = SaturationMode.SAT_FINITE
    got = _raw_binaryK(x, K, P, bias, signed, RoundMode.SR, sat, prng_bits)
    xc = x.cpu()
    lo = _project(xc, K, P, signed, RoundMode.RD, sat, bias).to(device)
    hi = _project(xc, K, P, signed, RoundMode.RU, sat, bias).to(device)
    ok = (got == lo) | (got == hi) | (torch.isnan(got) & torch.isnan(x))
    assert bool(ok.all()), f"{int((~ok).sum())} results are neither neighbour"
    assert not bool(torch.signbit(got[got == 0]).any()), "P3109's zero is unsigned"


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_float64_sr_draw_is_keyed_on_the_element_index(device, op):
    """A float64 element's draw depends on its index alone: the first n
    elements of a call come out as a call on those n does, whichever of the
    vector body and the scalar tail each lands in."""
    x = torch.randn(SIZE, device=device, dtype=torch.float64) * 8.0
    n = 4099  # prime, and odd, so the shorter call's tail starts mid-block

    def call(t):
        torch.manual_seed(1234)
        return _quant_calls(t)[op](rounding_mode=RoundMode.SR, prng_bits=12)

    _assert_same_words(call(x)[:n], call(x[:n].clone()), op)


@pytest.mark.parametrize("device", available_devices)
def test_float64_sr_is_unbiased(device):
    """SR's mean is the input, to within 5 sigma, for a value float32 cannot hold."""
    frac = 0.3 + 2.0**-40  # 1 + frac/8 has bits past binary32's reach
    x = torch.full((400_000,), 1.0 + frac * 0.125, device=device, dtype=torch.float64)
    got = binaryK_quantize(x, K=8, P=4, rounding_mode=RoundMode.SR, prng_bits=20)
    assert set(got.unique().tolist()) <= {1.0, 1.125}
    assert (got.mean().item() - 1.0) / 0.125 == pytest.approx(frac, abs=0.004)


# --- the rest of the entry point ----------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_float64_empty_tensor_keeps_its_dtype(device, op):
    empty = torch.empty(0, device=device, dtype=torch.float64)
    out = _quant_calls(empty)[op]()
    assert out.dtype is torch.float64
    assert out.numel() == 0


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_float64_non_contiguous_input(device, op):
    base = torch.randn(2 * SIZE, device=device, dtype=torch.float64).reshape(SIZE, 2) * 8.0
    strided = base[:, 0]
    assert not strided.is_contiguous()
    out = _quant_calls(strided)[op]()
    ref = _quant_calls(strided.contiguous())[op]()
    assert out.dtype is torch.float64
    _assert_same_words(out, ref, op)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_other_dtypes_keep_their_dtype(device, op, dtype):
    x = torch.randn(4096, device=device).to(dtype)
    out = _quant_calls(x)[op]()
    assert out.dtype is dtype


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "raw_op, args",
    [
        ("binaryK_quant", (8, 4, 3, 0, True)),
        ("superfp_quant", (3, 4, 1, 7, 0, True)),
    ],
)
def test_integer_input_is_still_rejected(device, raw_op, args):
    # The dispatch lists the four floating dtypes, and an unsupported one
    # still has to raise rather than fall through. Called through
    # ``torch.ops`` because the Python wrappers reject a non-float dtype
    # earlier, on their own mantissa-width lookup.
    x = torch.ones(64, device=device, dtype=torch.int32)
    fn = getattr(torch.ops.mptorch, raw_op).default
    with pytest.raises(NotImplementedError, match="not implemented for 'Int'"):
        fn(
            x,
            *args,
            RoundMode.RNE.value,
            SaturationMode.OVF_INF.value,
            *([SubnormalsMode.SUBNORMALS.value] if raw_op == "binaryK_quant" else []),
        )
