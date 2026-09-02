"""
Tests for what the elementwise quantize entry points do before any kernel
runs: the float64 narrowing that lets their type dispatch drop its ``double``
instantiation (finding G6 in dev/gemm_perf_audit.md, carried here from the
GEMM -- see tests/test_gemm_dispatch.py for that half).

It may not move a value, and the argument is the same one: ``scalar_t`` is
only ever these kernels' load/store type. ``SIMDTraits<double>`` reinterprets
the 128-bit carrier as a ``double2`` and casts each element to float, and the
CPU twin's loop body is ``static_cast<scalar_t>(cast(static_cast<float>(x)))``
-- so a float64 tensor was already narrowed on load and widened on store, and
doing that in a cast pass instead is the same conversion.

Stochastic rounding is the case that is *not* automatic, and is why these
tests exist separately from the GEMM's. Narrowing changes the vector width
(``SIMDTraits<double>::vec_elems`` is 2 against float's 4) and so the grid,
and a draw that depended on either would move under this change. E1 keys the
draw on the element's own linear index precisely so it does not, and the SR
rows below are what holds it to that.
"""

import pytest
import torch

from mptorch.number import RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import binaryK_quantize, superfp_quantize
from tests.markers import available_devices

# 100_003 is prime: neither the float4 body nor any dtype's vector width
# divides it, so the scalar tail runs on every backend and dtype.
SIZE = 100_003

ROUND_MODES = list(RoundMode)
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


def _operand(device, n=SIZE):
    torch.manual_seed(7)
    # x8 so the format's normal, supernormal and saturating regions are all hit
    return torch.randn(n, device=device, dtype=torch.float64) * 8.0


def _assert_same_words(a, b, ctx=""):
    """Raw-word comparison: NaN is not equal to itself and -0.0 is equal to
    0.0, and this is a claim about bits."""
    assert a.dtype == b.dtype, ctx
    bits = torch.int64 if a.dtype is torch.float64 else torch.int32
    assert torch.equal(a.view(bits), b.view(bits)), ctx


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("round_mode", ROUND_MODES)
def test_float64_matches_float32_bit_for_bit(device, op, round_mode):
    x64 = _operand(device)
    x32 = x64.float()
    kw = {"rounding_mode": round_mode}
    if round_mode is RoundMode.SR:
        kw["prng_bits"] = 12

    torch.manual_seed(99)
    out64 = _quant_calls(x64)[op](**kw)
    torch.manual_seed(99)
    out32 = _quant_calls(x32)[op](**kw)

    assert out64.dtype is torch.float64, "float64 in, float64 out"
    assert out32.dtype is torch.float32
    _assert_same_words(out64, out32.double(), f"{op} {round_mode.name}")


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("saturation_mode", SATURATION_MODES)
def test_float64_matches_float32_across_saturation_modes(device, op, saturation_mode):
    # saturation only decides anything for operands past the format's range,
    # which is what the x8 scale in _operand is for
    x64 = _operand(device)
    kw = {"saturation_mode": saturation_mode}
    out64 = _quant_calls(x64)[op](**kw)
    out32 = _quant_calls(x64.float())[op](**kw)
    assert out64.dtype is torch.float64
    _assert_same_words(out64, out32.double(), f"{op} {saturation_mode.name}")


@pytest.mark.parametrize("device", available_devices)
def test_float64_matches_float32_across_subnormals_modes(device):
    x64 = _operand(device)
    for subnormals_mode in SubnormalsMode:
        out64 = binaryK_quantize(x64, K=8, P=4, subnormals_mode=subnormals_mode)
        out32 = binaryK_quantize(x64.float(), K=8, P=4, subnormals_mode=subnormals_mode)
        assert out64.dtype is torch.float64
        _assert_same_words(out64, out32.double(), subnormals_mode.name)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_float64_empty_tensor_keeps_its_dtype(device, op):
    # the size == 0 short-circuit returns before the dispatch, so it needs the
    # widen of its own -- without it an empty float64 input comes back float32
    empty = torch.empty(0, device=device, dtype=torch.float64)
    out = _quant_calls(empty)[op]()
    assert out.dtype is torch.float64
    assert out.numel() == 0


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_float64_non_contiguous_input(device, op):
    # the narrow now happens before .contiguous(), so a strided float64 view
    # has to survive both in the right order
    base = _operand(device, 2 * SIZE).reshape(SIZE, 2)
    strided = base[:, 0]
    assert not strided.is_contiguous()
    out = _quant_calls(strided)[op]()
    ref = _quant_calls(strided.contiguous())[op]()
    assert out.dtype is torch.float64
    _assert_same_words(out, ref, op)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_narrowing_leaves_the_other_dtypes_alone(device, op, dtype):
    x = _operand(device, 4096).to(dtype)
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
    # The dispatch lists three types now rather than four, so it is worth
    # checking that what it dropped is only ``double``: an unsupported dtype
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
