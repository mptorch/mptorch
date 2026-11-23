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
from tests.markers import available_devices, float64_devices
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
