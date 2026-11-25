"""
``ScaledQuant``: scale into a format's range, round, and scale back.

``mptorch/quant/scaled.py`` computes ``s = target / max|x|`` in the carrier,
multiplies a carrier copy of ``x`` by it, rounds the copy to the format in
place, divides it back and narrows the result to ``x``'s dtype once. The
tests hold it to that statement written out with plain torch operations
(tier 3), to the class the FP8 tutorial used to define for itself, and to its
own granularities computed one slice at a time; then they cover the scale's
options, the inputs that get a scale of 1, the dtypes and carriers, the
errors, and its use in a layer, recomputed in backward and compiled.

On MPS, torch's own elementwise arithmetic flushes binary32 subnormals, so the
data here stay in the normal range, where the GPU computes what the CPU does.
"""

import math
from dataclasses import replace
from typing import Any

import pytest
import torch

from mptorch import E4M3, E5M2, BinaryK, FixedPoint, RoundMode, SaturationMode, SuperFP
from mptorch.number import MXFP8_E4M3
from mptorch.quant import QLinear, Quant, ScaledQuant
from mptorch.quant.ops._checks import _narrowed
from tests.markers import available_devices, cuda_devices, float64_devices, requires_mps
from tests.test_recompute import _equal, _formats, _step

SAT = SaturationMode.SAT_FINITE
SR = RoundMode.SR
# The finite domain's SatFinite, as the tutorial built them: `domain=None`
# lets the mode choose the domain, where `replace` alone would keep E4M3's
# extended one and give SAT_FINITE_EXTENDED.
E4M3_SAT = replace(E4M3, saturation=SAT, domain=None)
E5M2_SAT = replace(E5M2, saturation=SAT, domain=None)

# (format, rounding, target): the OCP layouts as the FP8 recipes scale them,
# a format's own largest value, the other two families, stochastic rounding.
CASES = [
    (E4M3_SAT, RoundMode.RNE, 448.0),
    (E5M2_SAT, RoundMode.RNE, 57344.0),
    (BinaryK(8, 4, prng_bits=8, saturation=SAT), SR, None),
    (BinaryK(8, 3, bias=15, saturation=SAT), RoundMode.RU, None),
    (SuperFP(3, 4, 2, 7, saturation=SAT), RoundMode.RD, None),
    (FixedPoint(8, 4, prng_bits=6), SR, None),
]


def _seeded(device):
    torch.manual_seed(99)
    if device == "cuda":
        torch.cuda.manual_seed_all(99)


def _data(device, dtype=torch.float32, shape=(6, 40)):
    torch.manual_seed(5)
    return (torch.randn(*shape, dtype=torch.float64) * 1e-2).to(dtype).to(device)


def _reference(x, fmt, rounding, target, carrier=None):
    """The statement of ``ScaledQuant``'s ``"tensor"`` granularity, in torch
    operations: widen to the carrier, scale, round, unscale, narrow once."""
    wide = x.dtype is torch.float64 or carrier is torch.float64
    c = torch.float64 if wide else torch.float32
    if target is None:
        target = fmt.max_value if isinstance(fmt, FixedPoint) else fmt.max_finite
    xc = x.to(c)
    s = target / xc.abs().amax().clamp_min(1e-30)
    y = Quant(fmt, rounding)(xc * s) / s
    if y.dtype is x.dtype:
        return y
    return _narrowed(y, x.dtype) if wide else y.to(x.dtype)


def _same_bits(a, b):
    """``torch.equal``, with NaNs equal to NaNs: torch's CUDA arithmetic may
    change a NaN's payload where the quantizers pass it through whole."""
    nan = a.isnan()
    return torch.equal(nan, b.isnan()) and torch.equal(a[~nan], b[~nan])


def _run(device, fn, x):
    _seeded(device)
    return fn(x)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fmt,rounding,target", CASES, ids=str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tier3_the_statement_in_torch(device, fmt, rounding, target, dtype):
    x = _data(device, dtype)
    sq = ScaledQuant(fmt, rounding, target=target)
    got = _run(device, sq, x)
    want = _run(device, lambda t: _reference(t, fmt, rounding, target), x)
    assert got.dtype is dtype and torch.equal(got, want)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("fmt,rounding,target", CASES, ids=str)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_tier3_in_binary64(device, fmt, rounding, target, dtype):
    """A float64 tensor, and a narrower one under ``carrier=torch.float64``,
    round in binary64 and are narrowed once."""
    x = _data(device, dtype)
    sq = ScaledQuant(fmt, rounding, target=target, carrier=torch.float64)
    got = _run(device, sq, x)
    want = _run(device, lambda t: _reference(t, fmt, rounding, target, torch.float64), x)
    assert got.dtype is dtype and torch.equal(got, want)


class _TutorialScaledQuant:
    """The class ``docs/snippets/tutorial/fp8_04_linear.py`` defined before
    the library had one, with the rounding mode added."""

    def __init__(self, fmt, max_finite, rounding=RoundMode.RNE):
        self.q = Quant(BinaryK(fmt.K, fmt.P, bias=fmt.bias, saturation=SAT), rounding)
        self.max_finite = max_finite

    def __call__(self, t):
        scale = self.max_finite / t.abs().amax().clamp_min(1e-30)
        return self.q(t * scale) / scale


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "fmt,target", [(E4M3, 448.0), (E5M2, 57344.0), (BinaryK(8, 4, prng_bits=8), 448.0)]
)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.RZ, RoundMode.RU, SR])
def test_the_tutorials_class(device, fmt, target, rounding):
    g = _data(device) * 0.1
    tutorial = _TutorialScaledQuant(fmt, target, rounding)
    sq = ScaledQuant(BinaryK(fmt.K, fmt.P, bias=fmt.bias, saturation=SAT), rounding, target=target)
    assert torch.equal(_run(device, sq, g), _run(device, tutorial, g))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_a_row_is_a_tensor(device, axis):
    """``"row"`` scales each slice along ``axis`` as ``"tensor"`` scales a
    tensor of that one slice."""
    x = _data(device, shape=(5, 7, 9))
    x[1] *= 1e4  # rows of different sizes
    got = ScaledQuant(E4M3_SAT, granularity="row", axis=axis)(x)
    moved_x, moved_got = x.movedim(axis, -1), got.movedim(axis, -1)
    per = ScaledQuant(E4M3_SAT)
    for idx in torch.cartesian_prod(*[torch.arange(n) for n in moved_x.shape[:-1]]):
        i = tuple(idx.tolist())
        assert torch.equal(moved_got[i], per(moved_x[i]))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("length,block_size", [(32, 8), (30, 8), (5, 8)])
def test_a_block_is_a_tensor(device, length, block_size):
    """``"block"`` scales each block, the last one shorter when the length is
    not a multiple of the block size, as ``"tensor"`` scales that block."""
    x = _data(device, shape=(3, length))
    got = ScaledQuant(E4M3_SAT, granularity="block", block_size=block_size)(x)
    per = ScaledQuant(E4M3_SAT)
    for r in range(x.shape[0]):
        for start in range(0, length, block_size):
            sl = slice(start, start + block_size)
            assert torch.equal(got[r, sl], per(x[r, sl]))
    # along the other axis: blocks of rows 0-1 and row 2 in each column
    down = ScaledQuant(E4M3_SAT, granularity="block", block_size=2, axis=0)(x)
    for c in range(length):
        assert torch.equal(down[0:2, c], per(x[0:2, c]))
        assert torch.equal(down[2:, c], per(x[2:, c]))


@pytest.mark.parametrize("device", available_devices)
def test_a_power_of_two_scale(device):
    """``pow2`` rounds the scale down to a power of two: the largest
    magnitude lands in ``(target / 2, target]``, and the result is the format
    applied to ``x * s``, divided by ``s``, with no other rounding."""
    for amax in (1e-3, 112.0, 0.4375, 3.0e-5):  # 448 / 112 is exactly 4
        x = torch.tensor([amax, -amax / 3, amax / 7, 0.0], device=device)
        s = 2.0 ** math.floor(math.log2(448.0 / amax))
        s = s / 2 if s > 448.0 / amax else s
        got = ScaledQuant(E4M3_SAT, pow2=True)(x)
        assert torch.equal(got, Quant(E4M3_SAT)(x * s) / s)
        assert 224.0 < amax * s <= 448.0


@pytest.mark.parametrize("device", available_devices)
def test_a_margin(device):
    x = _data(device)
    got = ScaledQuant(E4M3_SAT, margin=2)(x)
    assert torch.equal(got, _reference(x, E4M3_SAT, RoundMode.RNE, E4M3_SAT.max_finite / 4))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_element_gives_a_scale_of_1(device, bad):
    x = _data(device)
    x[0, 3] = bad
    assert _same_bits(ScaledQuant(E4M3_SAT)(x), Quant(E4M3_SAT)(x))
    # per row: only the row that holds it is scaled by 1
    rows = ScaledQuant(E4M3_SAT, granularity="row")(x)
    assert _same_bits(rows[0], Quant(E4M3_SAT)(x[0]))
    assert torch.equal(rows[1:], ScaledQuant(E4M3_SAT, granularity="row")(x[1:]))


@pytest.mark.parametrize("device", available_devices)
def test_zeros_and_tiny_tensors(device):
    zeros = torch.zeros(3, 4, device=device)
    assert torch.equal(ScaledQuant(E4M3_SAT)(zeros), zeros)
    rows = torch.stack([torch.zeros(4), torch.full((4,), 0.5)]).to(device)
    assert torch.equal(
        ScaledQuant(E4M3_SAT, granularity="row")(rows)[0], torch.zeros(4, device=device)
    )
    # a target so large that target / eps overflows binary32: scaled by 1
    huge = ScaledQuant(BinaryK(16, 8, saturation=SAT), target=1e30)
    tiny = torch.full((4,), 1e-35, device=device)
    assert torch.equal(huge(tiny), Quant(BinaryK(16, 8, saturation=SAT))(tiny))
    assert ScaledQuant(E4M3_SAT)(torch.empty(0, 3, device=device)).shape == (0, 3)


@pytest.mark.parametrize("device", available_devices)
def test_float16_is_scaled_in_the_carrier(device):
    """E5M2's 57344 over a gradient of size 1e-3 is a scale of 5.7e7, past
    float16's range: computed in binary32, the result is finite."""
    g = _data(device, torch.float16) * 0.1
    out = ScaledQuant(E5M2_SAT, target=57344.0)(g)
    assert out.dtype is torch.float16 and out.isfinite().all()
    assert (out == 0).sum() < (Quant(E5M2)(g) == 0).sum()


@pytest.mark.parametrize("device", available_devices)
def test_the_input_is_not_written(device):
    x = _data(device)
    before = x.clone()
    for sq in (ScaledQuant(E4M3_SAT), ScaledQuant(E4M3_SAT, granularity="row")):
        sq(x)
        sq(x.mT)  # a strided view too
    assert torch.equal(x, before)


def test_carrier_float32_on_float64_raises():
    with pytest.raises(ValueError, match="narrower than float64"):
        ScaledQuant(E4M3_SAT, carrier=torch.float32)(torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (dict(fmt=MXFP8_E4M3), TypeError),
        (dict(fmt="E4M3"), TypeError),
        (dict(target=0.0), ValueError),
        (dict(target=float("inf")), ValueError),
        (dict(eps=0.0), ValueError),
        (dict(granularity="channel"), ValueError),
        (dict(granularity="block"), ValueError),
        (dict(granularity="block", block_size=0), ValueError),
        (dict(block_size=32), ValueError),
        (dict(carrier="float64"), TypeError),
        (dict(carrier=torch.float16), ValueError),
    ],
)
def test_construction_errors(kwargs, error):
    arguments: dict[str, Any] = {"fmt": E4M3_SAT, **kwargs}
    with pytest.raises(error):
        ScaledQuant(**arguments)


def test_defaults():
    assert ScaledQuant(E4M3).target == E4M3.max_finite
    assert ScaledQuant(FixedPoint(8, 4)).target == FixedPoint(8, 4).max_value
    assert ScaledQuant(E4M3) == ScaledQuant(E4M3, target=E4M3.max_finite)
    assert hash(ScaledQuant(E4M3)) == hash(ScaledQuant(E4M3))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", ["linear", "conv2d", "matmul"])
def test_recomputed_in_backward(device, kind):
    """A layer that runs it again in backward gets the gradients of the saved
    copy, bit for bit, under stochastic rounding too."""
    q = ScaledQuant(BinaryK(8, 4, prng_bits=8, saturation=SAT), SR)
    kept = _step(kind, device, _formats(kind, q, flags=(False, False)))
    again = _step(kind, device, _formats(kind, q, flags=(True, True)))
    _equal(kept, again)


@pytest.mark.parametrize("device", available_devices)
def test_compiles(device):
    sq = ScaledQuant(E4M3_SAT, granularity="block", block_size=16, pow2=True)
    x = _data(device, shape=(4, 40))
    torch._dynamo.reset()
    assert torch.equal(torch.compile(sq, backend="aot_eager", fullgraph=True)(x), sq(x))
    layer = QLinear(40, 8).to(device)
    layer.formats.input_quant = layer.formats.weight_quant = ScaledQuant(E4M3_SAT)
    xr = x.clone().requires_grad_(True)
    eager = layer(xr)
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)(xr)
    assert torch.equal(eager, compiled)


# --- the four passes against the five ----------------------------------------------
#
# A call makes four passes over the tensor: the largest magnitudes, read from
# the input; the product, written straight into a carrier tensor; the rounding,
# in place; and the quotient, written straight into the result. It must give
# the bits of the five-pass spelling it replaced, below: a carrier copy,
# scaled, rounded and unscaled in place, then narrowed.

LAYOUTS = [
    ("tensor", None, -1, (6, 40)),
    ("tensor", None, -1, (77,)),
    ("row", None, -1, (6, 40)),
    ("row", None, 0, (3, 5, 9)),
    ("block", 8, -1, (6, 40)),
    ("block", 7, 1, (5, 30, 2)),  # another axis, and a shorter last block
    ("block", 64, -1, (4, 30)),  # a block longer than the row
]


def _five_passes(sq, x, device):
    """The five-pass spelling: a carrier copy of ``x``, scaled, rounded and
    unscaled in place, one view per scale, then narrowed to ``x``'s dtype."""
    wide = x.dtype is torch.float64 or sq.carrier is torch.float64
    carrier = torch.float64 if wide else torch.float32
    y = x.to(carrier, memory_format=torch.contiguous_format, copy=True)
    if sq.granularity == "tensor":
        views = [y]
        scales = [sq._scale(y.abs().amax())]
    else:
        moved = y.movedim(sq.axis, -1)
        step = moved.shape[-1] if sq.granularity == "row" else sq.block_size
        views = [moved[..., i : i + step] for i in range(0, moved.shape[-1], step)]
        scales = [sq._scale(v.abs().amax(-1, keepdim=True)) for v in views]
    for v, sv in zip(views, scales, strict=True):
        v.mul_(sv)
    _seeded(device)
    Quant(sq.fmt, sq.rounding, inplace=True)(y)
    for v, sv in zip(views, scales, strict=True):
        v.div_(sv)
    if x.dtype is carrier:
        return y
    return _narrowed(y, x.dtype) if wide else y.to(x.dtype)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fmt,rounding,target", CASES, ids=str)
@pytest.mark.parametrize("granularity,block_size,axis,shape", LAYOUTS, ids=str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_four_passes_are_the_five(
    device, fmt, rounding, target, granularity, block_size, axis, shape, dtype
):
    x = _data(device, dtype, shape) * torch.logspace(-3, 3, shape[-1], device=device).to(dtype)
    sq = ScaledQuant(
        fmt, rounding, target=target, granularity=granularity, block_size=block_size, axis=axis
    )
    got = _run(device, sq, x)
    assert got.dtype is dtype and got.shape == x.shape and got.is_contiguous()
    assert torch.equal(got, _five_passes(sq, x, device))
    strided = x.mT if x.dim() > 1 else x[::2]
    assert torch.equal(_run(device, sq, strided), _five_passes(sq, strided, device))


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("fmt,rounding,target", CASES, ids=str)
@pytest.mark.parametrize("granularity,block_size,axis,shape", LAYOUTS[::2], ids=str)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_four_passes_are_the_five_in_binary64(
    device, fmt, rounding, target, granularity, block_size, axis, shape, dtype
):
    x = _data(device, dtype, shape)
    sq = ScaledQuant(
        fmt,
        rounding,
        target=target,
        granularity=granularity,
        block_size=block_size,
        axis=axis,
        carrier=torch.float64,
    )
    got = _run(device, sq, x)
    assert got.dtype is dtype and torch.equal(got, _five_passes(sq, x, device))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "granularity,block_size,axis", [("tensor", None, -1), ("row", None, 0), ("block", 6, -1)]
)
def test_compiles_on_every_layout(device, dtype, granularity, block_size, axis):
    """Strided views take a copy and an in-place op under compile, with the same values."""
    sq = ScaledQuant(
        BinaryK(8, 4, prng_bits=8, saturation=SAT),
        SR,
        granularity=granularity,
        block_size=block_size,
        axis=axis,
    )
    x = _data(device, dtype, (4, 20))
    eager = _run(device, sq, x)
    torch._dynamo.reset()  # each case compiles ScaledQuant.__call__ for its own layout
    compiled = _run(device, torch.compile(sq, backend="aot_eager", fullgraph=True), x)
    assert torch.equal(eager, compiled)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("granularity,axis", [("tensor", -1), ("row", 0)])
def test_compiles_on_a_transposed_input(device, dtype, backend, granularity, axis):
    """A transposed input, as simulate passes GPT-2's Conv1D weight: compiled,
    the carrier tensor stays contiguous for the in-place rounding (functionalized,
    an ``out=`` product took the input's strides, and the rounding refused it)."""
    sq = ScaledQuant(BinaryK(8, 4, saturation=SAT), granularity=granularity, axis=axis)
    x = _data(device, dtype, (20, 12)).mT
    eager = _run(device, sq, x)
    torch._dynamo.reset()
    compiled = _run(device, torch.compile(sq, backend=backend, fullgraph=True), x)
    if backend == "inductor" and device == "cuda" and dtype is torch.float32:
        # Inductor's division by the scale can differ from eager's in the last place
        torch.testing.assert_close(compiled, eager, rtol=2**-20, atol=0)
    else:
        assert torch.equal(eager, compiled)


# --- the fused op -------------------------------------------------------------------
#
# With one scale for the tensor or the scaled axis last, a call is one pass of
# the fused op (csrc/common/scaled_quant_host.h) after the largest magnitudes
# are found; anywhere else it is the unfused spelling, which the tests above
# hold to the statement. The fused op gives the unfused spelling's bits on the
# CPU and CUDA. On MPS, PyTorch's own arithmetic in the unfused spelling
# flushes binary32 subnormals and the Metal kernel does not, so MPS's fused op
# is held to the CPU's instead.

FUSED_CASES = CASES + [
    (BinaryK(8, 4, prng_bits=8, saturation=SAT), RoundMode.SR_B, None),
    (BinaryK(8, 3, bias=15, saturation=SAT), RoundMode.RO, None),
    (SuperFP(3, 4, 2, 7, prng_bits=8, saturation=SAT), SR, None),
    (FixedPoint(8, 4), RoundMode.RZ, None),
]
LAYOUTS = [
    ("tensor", None, (6, 40)),
    ("tensor", None, (77,)),
    ("tensor", None, (2, 3, 13)),
    ("row", None, (6, 40)),
    ("row", None, (3, 5, 9)),
    ("block", 8, (6, 40)),
    ("block", 7, (5, 30)),  # a shorter last block
    ("block", 3, (2, 2, 2)),  # rows shorter than a vector
    ("block", 64, (4, 30)),  # a block longer than the row
]
# The devices where torch's own arithmetic, which the unfused call runs, is
# IEEE's. MPS is not one: it flushes binary32 subnormals, so
# test_mps_fused_is_the_cpus holds its fused kernel to the CPU instead. The
# list is spelled out because available_devices holds pytest.param values,
# which no string compares equal to.
UNFUSED_TWIN_DEVICES = ["cpu", *cuda_devices]


def _fused_and_unfused(sq, x, device):
    wide = x.dtype is torch.float64 or sq.carrier is torch.float64
    fused = _run(device, sq, x)
    unfused = _run(device, lambda t: sq._unfused_call(t, wide), x)
    return fused, unfused


def _spread(device, dtype, shape):
    return _data(device, dtype, shape) * torch.logspace(-3, 3, shape[-1], device=device).to(dtype)


@pytest.mark.parametrize("device", UNFUSED_TWIN_DEVICES)
@pytest.mark.parametrize("fmt,rounding,target", FUSED_CASES, ids=str)
@pytest.mark.parametrize("granularity,block_size,shape", LAYOUTS, ids=str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_the_fused_op_is_the_unfused_spelling(
    device, fmt, rounding, target, granularity, block_size, shape, dtype
):
    x = _spread(device, dtype, shape)
    sq = ScaledQuant(fmt, rounding, target=target, granularity=granularity, block_size=block_size)
    fused, unfused = _fused_and_unfused(sq, x, device)
    assert fused.dtype is dtype and fused.shape == x.shape
    assert torch.equal(fused, unfused)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("fmt,rounding,target", FUSED_CASES, ids=str)
@pytest.mark.parametrize("granularity,block_size,shape", LAYOUTS[::2], ids=str)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_the_fused_op_in_binary64(
    device, fmt, rounding, target, granularity, block_size, shape, dtype
):
    x = _data(device, dtype, shape)
    sq = ScaledQuant(
        fmt,
        rounding,
        target=target,
        granularity=granularity,
        block_size=block_size,
        carrier=torch.float64,
    )
    fused, unfused = _fused_and_unfused(sq, x, device)
    assert fused.dtype is dtype and torch.equal(fused, unfused)


@pytest.mark.parametrize("fmt,rounding,target", FUSED_CASES, ids=str)
@pytest.mark.parametrize("granularity,block_size,shape", LAYOUTS, ids=str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", [pytest.param("mps", marks=requires_mps)])
def test_mps_fused_is_the_cpus(
    device, fmt, rounding, target, granularity, block_size, shape, dtype
):
    """The Metal kernel gives the CPU kernel's bits, SR included (it draws its
    seed from the CPU generator and keys the streams as the CPU does), and
    subnormal products and quotients too."""
    x = _spread("cpu", dtype, shape)
    sq = ScaledQuant(fmt, rounding, target=target, granularity=granularity, block_size=block_size)
    want = _run("cpu", sq, x)
    got = _run(device, sq, x.to(device)).cpu()
    assert _same_bits(got, want)


@pytest.mark.parametrize("device", available_devices)
def test_the_fused_op_on_special_values_and_layouts(device):
    """Non-finite elements (a scale of 1, NaN positions kept), a strided
    input, a view off a 16-byte boundary, and a power-of-two scale. The fused
    op is held to the unfused call on the same device, and on MPS to the
    CPU's fused op, as in test_mps_fused_is_the_cpus."""
    x = _data(device, shape=(8, 33))
    x[1, 3], x[2, 5], x[5, 0] = float("inf"), float("nan"), float("-inf")
    for sq in (
        ScaledQuant(E4M3_SAT, granularity="row"),
        ScaledQuant(E4M3_SAT, granularity="block", block_size=4, pow2=True),
        ScaledQuant(E5M2_SAT, pow2=True, margin=1),
    ):
        for t in (x, x.mT, x.flatten()[1:]):
            if device == "mps":
                got, want = _run(device, sq, t).cpu(), _run("cpu", sq, t.cpu())
            else:
                got, want = _fused_and_unfused(sq, t, device)
            assert _same_bits(got, want)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "fmt", [BinaryK(8, 4, saturation=SAT), SuperFP(3, 4, 2, 7), FixedPoint(8, 4)], ids=str
)
def test_the_call_runs_the_fused_op_where_it_can(device, fmt, monkeypatch):
    import mptorch.quant.scaled as scaled

    calls = []
    for name, op in list(scaled._FUSED.items()):
        monkeypatch.setitem(
            scaled._FUSED, name, lambda *a, _op=op, **k: calls.append(_op) or _op(*a, **k)
        )
    sq_tensor, sq_block = ScaledQuant(fmt), ScaledQuant(fmt, granularity="block", block_size=4)
    x = _data(device, shape=(4, 16))
    sq_tensor(x)
    sq_block(x)
    assert len(calls) == 2
    ScaledQuant(fmt, granularity="row", axis=0)(x)  # another axis: the unfused spelling
    assert len(calls) == 2


def _op_args(rows, scale, block_size):
    return (rows, scale, block_size, 8, 4, 8, 0, True, 0, 2, 0)


@pytest.mark.parametrize(
    "rows,scale,block_size,match",
    [
        (torch.ones(4, 8, dtype=torch.int32), torch.ones(4, 1), 0, "float32, float64"),
        (torch.ones(32), torch.ones(1, 1), 0, "2D tensor of rows"),
        (torch.ones(4, 8), torch.ones(4, 1, dtype=torch.float64), 0, "values of the carrier"),
        (
            torch.ones(4, 8, dtype=torch.bfloat16),
            torch.ones(4, 1, dtype=torch.bfloat16),
            0,
            "carrier",
        ),
        (torch.ones(4, 8), torch.ones(4), 0, "2D tensor, one row"),
        (torch.ones(4, 8), torch.ones(4, 1), -1, "block_size is 0"),
        (torch.ones(4, 8), torch.ones(4, 2), 0, "takes 4 x 1 scales"),
        (torch.ones(4, 8), torch.ones(4, 2), 3, "takes 4 x 3 scales"),
        (torch.ones(4, 8), torch.ones(3, 1), 0, "takes 4 x 1 scales"),
    ],
)
def test_the_op_refuses_bad_operands_on_every_backend(rows, scale, block_size, match):
    """The CPU and Meta kernels refuse the same operands with the same message."""
    op = torch.ops.mptorch.binaryK_quant_scaled
    with pytest.raises(RuntimeError, match=match) as cpu:
        op(*_op_args(rows, scale, block_size))
    meta_rows = torch.empty_like(rows, device="meta")
    meta_scale = torch.empty_like(scale, device="meta")
    with pytest.raises(RuntimeError) as meta:
        op(*_op_args(meta_rows, meta_scale, block_size))
    assert str(meta.value).split("\n")[0] == str(cpu.value).split("\n")[0]


@pytest.mark.parametrize("device", available_devices)
def test_the_op_on_its_own(device):
    """A direct call: the scale is applied per row and per block as the
    layout says, and an empty tensor returns an empty result."""
    op = torch.ops.mptorch.binaryK_quant_scaled
    rows = _data(device, shape=(3, 10))
    s = torch.tensor([[2.0, 4.0, 8.0]] * 3, device=device)
    got = op(*_op_args(rows, s, 4))
    q = Quant(BinaryK(8, 4))
    want = torch.cat(
        [q(rows[:, :4] * 2) / 2, q(rows[:, 4:8] * 4) / 4, q(rows[:, 8:] * 8) / 8], dim=1
    )
    assert torch.equal(got, want)
    assert op(*_op_args(rows[:0], s[:0], 4)).shape == (0, 10)
    if device == "cuda":
        with pytest.raises(RuntimeError, match="the scales are on cpu"):
            op(*_op_args(rows, s.cpu(), 4))
