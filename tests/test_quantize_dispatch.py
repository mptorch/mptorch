"""
Which carrier the elementwise quantizers round an operand in, by its dtype.

Guards ``binaryK_quantize`` and ``superfp_quantize`` (``mptorch/quant/ops.py``)
and the kernels behind them. A float64 tensor rounds in binary64: its kernels
are instantiated for ``double`` and compute in it. A float32, float16 or
bfloat16 tensor rounds in binary32, whose values all three are, unless the call
names ``carrier=torch.float64``. A dispatch that narrowed a float64 tensor to
float32 first would round every input twice, silently, and could not reach a
format that needs more than binary32 offers (24 bits of precision, seven
exponent bits, a smallest value of 2**-125).

Five properties follow, one section each.

* **The binary32 image.** A float64 tensor holding float32 values, quantized
  to a format binary32 simulates faithfully, gives exactly the float32
  result: the answer does not depend on which carrier held the input.
  ``dev/benchmarks/cast_all_modes_sweep.cu image64`` checks that for the casts
  over every float32 input. These tests hold the kernels' dtype dispatch,
  vector loads and scalar tail to the same claim.
* **Rounding once.** Where the float64 value is not a float32 value, the
  result differs from the one a narrowing to float32 would give, because that
  narrowing is a rounding of its own.
* **Formats past binary32.** The raw op is checked against
  ``tests/test_binaryk_p3109.py``'s ``_project``, a transcription of the
  projection in IEEE P3109 (arXiv:2606.04028) that shares no code with the
  kernels and is exact on float64 inputs. The Python wrapper holds a float64
  tensor's format to binary64's bounds rather than binary32's, so it must
  pass every such format without a warning.
* **binary64 for any tensor, on request.** ``carrier=torch.float64`` widens a
  float32, float16 or bfloat16 tensor in Python, rounds it with the binary64
  kernel and narrows the result back once through the ``narrow_float64`` op.
  That is bit-identical to the float64 call on the widened tensor, random
  draws included. A carrier narrower than the tensor is refused.
* **A view that starts mid-storage.** A contiguous view off a 16-byte
  boundary is copied before the CUDA kernel's aligned vector load.

Stochastic rounding (SR) draws one 64-bit word per float64 element: element
``j`` takes Philox block ``j >> 1`` and its 32-bit words ``2 * (j & 1)`` and
the next, low word first (``csrc/common/philox.h``). The draws have no
closed-form reference, so SR is checked by property: every result is one of
the input's two neighbours in the format, each element's draw is a function
of its own index, and the mean is the input.
"""

import math
import warnings
from typing import Any

import pytest
import torch

from mptorch.number import FormatRangeWarning, RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import binaryK_quantize, superfp_quantize
from mptorch.quant.ops import _narrowed
from tests.markers import available_devices
from tests.test_binaryk_p3109 import _project

# 100_003 is prime, so no vector width (4 floats, 2 doubles) divides it and the
# kernels' scalar tail runs on every backend and dtype.
SIZE = 100_003

DETERMINISTIC = [rm for rm in RoundMode if rm is not RoundMode.SR]
SATURATION_MODES = list(SaturationMode)


def _quant_calls(x):
    """Both elementwise quantizers bound to ``x``, keyed by op name.

    Each value takes the remaining keyword arguments. The formats are binaryK
    ``K=8, P=4`` and superfp ``m3e4n1b7``, both of which binary32 carries."""
    return {
        "binaryK": lambda **kw: binaryK_quantize(x, K=8, P=4, **kw),
        "superfp": lambda **kw: superfp_quantize(
            x, man_bits=3, exp_bits=4, normal_binades=1, bias=7, **kw
        ),
    }


OP_NAMES = list(_quant_calls(torch.empty(0)).keys())


def _float32_values(device, n=SIZE):
    """A float64 tensor whose values are all float32 values.

    The normal draws are scaled by 8 so that the formats' normal, supernormal
    and saturating regions are all hit. The first ten elements are specials:
    both zeros, infinities, NaN, float32's smallest subnormal, a value near its
    largest, and 1.0625, a tie of binaryK 8p4."""
    torch.manual_seed(7)
    x = torch.randn(n, device=device, dtype=torch.float32) * 8.0
    specials = torch.tensor(
        [0.0, -0.0, math.inf, -math.inf, math.nan, 1.4e-45, -1.4e-45, 3.4e38, 1.0, 1.0625],
        device=device,
    )
    x[: specials.numel()] = specials
    return x.double()


def _assert_same_words(a, b, ctx=""):
    """Assert that ``a`` and ``b`` hold the same words, naming the first mismatch.

    Bits are compared rather than values because NaN is not equal to itself
    and -0.0 is equal to 0.0, and the claims here include both."""
    assert a.dtype == b.dtype, ctx
    bits = {8: torch.int64, 4: torch.int32, 2: torch.int16}[a.element_size()]
    same = a.view(bits) == b.view(bits)
    if not bool(same.all()):
        i = int((~same).nonzero()[0])
        raise AssertionError(f"{ctx}: {int((~same).sum())} differ, first {i}: {a[i]!r} vs {b[i]!r}")


# --- the binary32 image -------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("round_mode", DETERMINISTIC)
def test_float64_of_float32_values_matches_float32(device, op, round_mode):
    """The binary64 kernel gives float32 values the binary32 kernel's result.

    Catches a float64 dispatch, vector load or scalar tail that rounds
    differently from the float32 one in any deterministic rounding mode."""
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
    """The binary32 image holds in every saturation mode.

    Saturation decides the result only for operands past the format's range,
    which the scale of 8 in ``_float32_values`` provides."""
    x64 = _float32_values(device)
    kw = {"saturation_mode": saturation_mode}
    out64 = _quant_calls(x64)[op](**kw)
    out32 = _quant_calls(x64.float())[op](**kw)
    _assert_same_words(out64, out32.double(), f"{op} {saturation_mode.name}")


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("subnormals_mode", list(SubnormalsMode))
def test_float64_of_float32_values_across_subnormals_modes(device, subnormals_mode):
    """The binary32 image holds in every subnormals mode, below its floor too.

    The inputs are repeated divided by 64 so that values below each mode's
    smallest value are reached. The division is done in float32, because in
    float64 the quotient of float32's smallest subnormal is not a float32
    value."""
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
        # binaryK 8p4 has steps of 0.125 in [1, 2), so 1.0625 is the tie between
        # 1.0 and 1.125. float32's step there is 2**-23 and cannot hold the 2**-30
        # that breaks the tie, so the narrowed input is the tie and RNE takes the
        # even 1.0.
        (RoundMode.RNE, 1.0625 + 2.0**-30, 1.0, 1.125),
        # Narrowed to float32 the input is 1.0, a grid point, and RU and RD have
        # nothing to round.
        (RoundMode.RU, 1.0 + 2.0**-30, 1.0, 1.125),
        (RoundMode.RD, -(1.0 + 2.0**-30), -1.0, -1.125),
    ],
)
def test_float64_is_rounded_once(device, round_mode, x, via_float32, direct):
    """A float64 input a hair off a grid point or a tie rounds by its own value.

    A kernel that narrowed the tensor to float32 first would return
    ``via_float32`` for the float64 call."""
    t = torch.tensor([x], device=device, dtype=torch.float64)
    got = binaryK_quantize(t, K=8, P=4, rounding_mode=round_mode)
    assert got.item() == direct
    assert binaryK_quantize(t.float(), K=8, P=4, rounding_mode=round_mode).item() == via_float32


@pytest.mark.parametrize("device", available_devices)
def test_float64_is_rounded_once_superfp(device):
    """The superfp kernel rounds a float64 input once as well.

    superfp ``m3e4n1b7`` has one normal binade, [256, 512) in steps of 32, so
    272 is the tie between 256 and 288. float32's step there is 2**-15, so the
    2**-22 that breaks the tie is lost by a narrowing to float32."""
    t = torch.tensor([272.0 + 2.0**-22], device=device, dtype=torch.float64)
    assert superfp_quantize(t, 3, 4, 1, 7).item() == 288.0
    assert superfp_quantize(t.float(), 3, 4, 1, 7).item() == 256.0


# --- formats past binary32 ----------------------------------------------------

# (K, P, bias, signed). All but the last need binary64: precision past 24 bits,
# an exponent field past 7 bits, or a smallest value below 2**-125. Each is inside
# binary64's own bounds (53 bits, smallest value 2**(1 - bias - (P - 1)) no lower
# than 2**-1021), and two sit on that floor.
WIDE_FORMATS = [
    (40, 30, 512, True),  # 30 bits of precision, ten exponent bits
    (63, 53, 511, True),  # binary64's full precision
    (63, 53, 970, True),  # smallest value 2**(1 - 970 - 52) = 2**-1021, the floor
    (11, 1, 1022, True),  # P = 1 on that floor: 2**(1 - 1022) = 2**-1021
    (40, 31, 512, False),  # unsigned, so the sign bit is a tenth exponent bit
    (16, 12, 8, True),  # a format binary32 carries too, as a control
]


def _wide_inputs(K, P, bias, signed, device, n=65_536):
    """float64 inputs across the format's range and a little past both ends.

    Exponents run from three below the smallest subnormal to two above the top
    binade, clipped to binary64's. Each gets a random 53-bit significand, a
    point of the format's own grid and the midpoint above it (a tie for the
    nearest modes). Specials are appended and signs are random."""
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
    """The ``binaryK_quant`` op itself, past the Python wrapper's format checks."""
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
    """The binary64 kernel matches the P3109 projection in formats binary32 cannot hold.

    Catches a cast constant or shift that is still sized for binary32 (a
    32-bit word, a 23-bit significand) in the ``double`` instantiation."""
    x = _wide_inputs(K, P, bias, signed, device)
    for round_mode in DETERMINISTIC:
        got = _raw_binaryK(x, K, P, bias, signed, round_mode, saturation)
        want = _project(x.cpu(), K, P, signed, round_mode, saturation, bias).to(device)
        ctx = f"K={K} P={P} bias={bias} {round_mode.name} {saturation.name}"
        _assert_same_words(got, want, ctx)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(("K", "P", "bias", "signed"), WIDE_FORMATS)
def test_the_wrapper_holds_float64_to_binary64s_bounds(device, K, P, bias, signed):
    """The wrapper checks a float64 tensor's format against binary64's bounds.

    Every format in ``WIDE_FORMATS`` is inside them, so the call must be silent
    and equal to the raw op. A wrapper that applied binary32's bounds would
    warn or raise here. The same format on a float32 tensor is refused when
    its precision is past binary32's 24 bits."""
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


# --- binary64 for any tensor, on request ---------------------------------------


# float64 values float32 cannot hold, each a hair off a grid point or a midpoint
# of one of the two formats (binaryK 8p4: 1.0 and the tie 1.0625, superfp
# m3e4n1b7: 256 and the tie 272). The trailing comment names the rounding mode
# whose answer a narrowing to float32 would move.
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


NARROW = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("dtype", NARROW)
@pytest.mark.parametrize("round_mode", list(RoundMode))
def test_binary64_carrier_is_the_widened_float64_call(device, op, dtype, round_mode):
    """``carrier=torch.float64`` equals the float64 call on the widened tensor.

    Bit for bit after ``_narrowed``, in every rounding mode: the same kernel on
    the same values drawing the same random words. Catches a widening that
    changes the SR stream or a result narrowed with more than one rounding."""
    x = (torch.randn(SIZE, device=device) * 8.0).to(dtype)
    kw = dict(rounding_mode=round_mode, prng_bits=12 if round_mode is RoundMode.SR else 0)
    torch.manual_seed(99)
    got = _quant_calls(x)[op](carrier=torch.float64, **kw)
    torch.manual_seed(99)
    want = _narrowed(_quant_calls(x.double())[op](**kw), dtype)
    assert got.dtype is dtype
    _assert_same_words(got, want, f"{op} {dtype} {round_mode.name}")
    # A tensor's own values round the same in either carrier (the binary32 image,
    # for formats binary32 carries), so only SR tells the two apart: it draws 64
    # bits per element in binary64 and 32 in binary32.
    torch.manual_seed(99)
    own = _quant_calls(x)[op](**kw)
    assert torch.equal(own, got) is (round_mode is not RoundMode.SR)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_binary64_carrier_of_a_strided_tensor(device, op):
    """A non-contiguous float32 tensor is widened by its strides, not its storage."""
    base = (torch.randn(2 * 4099, device=device) * 8.0).reshape(4099, 2)
    strided = base[:, 1]
    got = _quant_calls(strided)[op](carrier=torch.float64)
    _assert_same_words(got, _quant_calls(strided.double())[op]().float(), op)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_a_carrier_a_dtype_already_has_changes_nothing(device, op, dtype):
    """Naming the carrier a dtype rounds in by default is the default call."""
    x = (torch.randn(4096, device=device) * 8).to(dtype)
    carrier = torch.float64 if dtype is torch.float64 else torch.float32
    got = _quant_calls(x)[op](carrier=carrier)
    assert got.dtype is dtype
    _assert_same_words(got, _quant_calls(x)[op](), f"{op} {dtype}")


@pytest.mark.parametrize("device", available_devices)
def test_a_carrier_narrower_than_the_tensor_is_refused(device):
    """A carrier below the tensor's dtype, or one that is not a carrier, raises.

    ``torch.float32`` on a float64 tensor would round every input once before
    the format does. Only float32 and float64 name a carrier, and a string is
    a ``TypeError`` rather than a silent default."""
    x = torch.ones(8, device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="narrower than float64"):
        binaryK_quantize(x, 8, 4, carrier=torch.float32)
    with pytest.raises(ValueError, match="narrower than float64"):
        superfp_quantize(x, 3, 4, 1, 7, carrier=torch.float32)
    with pytest.raises(ValueError, match="carrier must be torch.float32"):
        binaryK_quantize(x.half(), 8, 4, carrier=torch.float16)
    with pytest.raises(TypeError, match="carrier must be a torch.dtype"):
        superfp_quantize(x, 3, 4, 1, 7, carrier="binary64")  # ty: ignore[invalid-argument-type]


# --- a view that starts mid-storage -------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("round_mode", [RoundMode.RNE, RoundMode.SR])
def test_a_contiguous_view_off_a_16_byte_boundary(device, op, dtype, round_mode):
    """A contiguous view that starts off a 16-byte boundary quantizes as its copy.

    ``x[k:]`` is contiguous and starts ``k`` elements into its storage, which
    is off a 16-byte boundary for every ``k`` and dtype here except ``k = 2``
    in float64. The CUDA kernels' vector load is aligned, and reading such a
    view in place is a misaligned-address fault that poisons the CUDA context,
    so the entry point must copy it (``csrc/cuda/vector_load.h``). SR draws are
    included: the copy must not move an element's index."""
    kw = dict(rounding_mode=round_mode, prng_bits=4 if round_mode is RoundMode.SR else 0)
    storage = (torch.randn(4099 + 3, device=device) * 8).to(dtype)
    for k in (1, 2, 3):
        view = storage[k:]
        torch.manual_seed(5)
        got = _quant_calls(view)[op](**kw)
        torch.manual_seed(5)
        want = _quant_calls(view.clone())[op](**kw)
        _assert_same_words(got, want, f"{op} {dtype} {round_mode.name} k={k}")
    # A row slice of a matrix starts one row of 3 elements into its storage.
    rows = (torch.randn(64, 3, device=device) * 8).to(dtype)[1:]
    _assert_same_words(_quant_calls(rows)[op](), _quant_calls(rows.clone())[op](), f"{op} rows")


# --- stochastic rounding ------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(("K", "P", "bias", "signed"), WIDE_FORMATS)
def test_float64_sr_lands_on_a_neighbour(device, K, P, bias, signed):
    """SR in binary64 returns the RD or the RU projection of each input.

    ``prng_bits`` is capped so that ``man_bits + prng_bits`` stays within the
    52 significand bits binary64 has for the random tail. The zero check holds
    SR to P3109's unsigned zero, which a value comparison cannot see."""
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
    """A float64 element's draw depends on its index alone.

    The first ``n`` elements of a long call equal a call on those ``n``,
    whichever of the vector body and the scalar tail each lands in. Catches a
    tail that restarts the stream or pairs the block's words differently."""
    x = torch.randn(SIZE, device=device, dtype=torch.float64) * 8.0
    # 4099 is prime, so the shorter call ends in a scalar tail, on a half-used
    # Philox block, where the longer call is still in its vector body.
    n = 4099

    def call(t):
        torch.manual_seed(1234)
        return _quant_calls(t)[op](rounding_mode=RoundMode.SR, prng_bits=12)

    _assert_same_words(call(x)[:n], call(x[:n].clone()), op)


@pytest.mark.parametrize("device", available_devices)
def test_float64_sr_is_unbiased(device):
    """SR's mean is the input, for a value float32 cannot hold.

    The input sits ``frac`` of the way from 1.0 to 1.125, so the fraction of
    results at 1.125 is a binomial proportion with standard deviation
    sqrt(0.3 * 0.7 / 400_000) = 0.00072. The tolerance 0.004 is about 5.5 of
    those."""
    frac = 0.3 + 2.0**-40  # 1 + frac / 8 has bits below float32's 2**-23 step
    x = torch.full((400_000,), 1.0 + frac * 0.125, device=device, dtype=torch.float64)
    got = binaryK_quantize(x, K=8, P=4, rounding_mode=RoundMode.SR, prng_bits=20)
    assert set(got.unique().tolist()) <= {1.0, 1.125}
    assert (got.mean().item() - 1.0) / 0.125 == pytest.approx(frac, abs=0.004)


# --- the rest of the entry point ----------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_float64_empty_tensor_keeps_its_dtype(device, op):
    """The empty early return allocates float64, not the kernel's default dtype."""
    empty = torch.empty(0, device=device, dtype=torch.float64)
    out = _quant_calls(empty)[op]()
    assert out.dtype is torch.float64
    assert out.numel() == 0


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_float64_non_contiguous_input(device, op):
    """A strided float64 tensor quantizes as its contiguous copy does."""
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
    """Rounding in binary32 does not change a float16 or bfloat16 result's dtype."""
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
    """A dtype outside the four floating ones raises instead of falling through.

    Called through ``torch.ops`` because the Python wrappers reject a
    non-float dtype earlier, on their own mantissa-width lookup."""
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
