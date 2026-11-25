"""
The fixed-point GEMM's Python API, over the ops that
``tests/test_fixedpoint_gemm.py`` holds to the definition.

- **The flat wrappers** (``fixedpoint_matmul``, ``fixedpoint_matmul_fma``) are
  the ops, with ``torch.matmul``'s operand rules, a named carrier, and a wide
  result kept in float64.
- **The macs** (``SplitMac`` and ``FusedMac`` of ``FixedPoint`` formats)
  resolve to the flat wrappers' specs, refuse what the GEMM does not take,
  and ``Overflow`` chooses between saturation and wrap-around.
- **The layers**, in the three tiers of ``tests/test_qlinear.py``: a
  ``QLinear``, a ``QConv2d`` and a ``QMatmul`` on fixed-point macs give the
  plain layer's result on operands whose sums are exact (tier 1), are close
  to it on others (tier 2), and are the ops called by hand (tier 3), forward
  and backward. The float32 tests run on MPS too; the exact tiers, in
  float64, do not.
- **A wide result**: rounded once by the forward's output rounding, after the
  bias, and refused without one or in a gradient pass, and on MPS, which has
  no float64, in every pass.
- **Recipes**: ``convert`` and ``simulate`` agree word for word, stochastic
  rounding included, and ``profile`` files the products under the mac.
- **``FixedPoint``'s random bits**: a wide stochastic accumulator, which only
  the GEMM's integer accumulator can draw for, can be built.
"""

from typing import Any

import pytest
import torch
from torch import nn

from mptorch import AccumulateAlgorithm, BinaryK, FixedPoint, Overflow, RoundMode
from mptorch.quant import (
    FusedMac,
    GemmFormats,
    QConv2d,
    QLinear,
    Quant,
    Recipe,
    SplitMac,
    conv_formats,
    convert,
    fixedpoint_gemm_formats,
    fixedpoint_gemm_formats_fma,
    fixedpoint_matmul,
    fixedpoint_matmul_fma,
    fixedpoint_quantize,
    profile,
    qmatmul,
    qmm,
    simulate,
)
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops._spec import _wide_refusal
from mptorch.quant.ops.fixedpoint import _fixedpoint_fma_spec, _fixedpoint_spec

from .markers import cuda_devices, requires_mps
from .test_fixedpoint_gemm import FUSED, SPLIT, _conv, _on_grid, _operands

R = RoundMode
# The devices with float64, which the exact tiers and every wide result need,
# and the float32 tests' devices, MPS among them.
DEVICES = ["cpu", *cuda_devices]
FLOAT32_DEVICES = [*DEVICES, pytest.param("mps", marks=requires_mps)]
MUL, ACC = FixedPoint(16, 8), FixedPoint(16, 4)
GRID = FixedPoint(8, 4)  # the operands' grid in the exact tiers
F32, F64 = torch.float32, torch.float64


def _given(t: torch.Tensor | None) -> torch.Tensor:
    """``t``, which the test has set: a layer's bias, or a gradient."""
    assert t is not None
    return t


def _seeded(device: str) -> None:
    torch.manual_seed(5)
    if device == "cuda":
        torch.cuda.manual_seed(5)


# --- the flat wrappers ---------------------------------------------------------------


@pytest.mark.parametrize("device", FLOAT32_DEVICES)
@pytest.mark.parametrize("mode", [R.RNE, R.RD, R.SR, R.SR_C])
def test_the_flat_wrappers_are_the_ops(mode, device):
    a, b = (t.to(device) for t in _operands((5, 37), (37, 6), F32, seed=1))
    _seeded(device)
    got = fixedpoint_matmul(
        a, b, mul_wl=16, mul_fl=8, acc_wl=16, acc_fl=4, mul_prng_bits=5, acc_prng_bits=5,
        rounding_mode=mode,
    )  # fmt: skip
    _seeded(device)
    want = SPLIT(
        a, b, False, False, 16, 8, True, False, 16, 4, True, False, mode.value, 5, 5, False
    )
    assert torch.equal(got, want)
    _seeded(device)
    got = fixedpoint_matmul_fma(
        a, b, fma_wl=12, fma_fl=6, fma_is_signed=False, fma_prng_bits=4, rounding_mode=mode
    )
    _seeded(device)
    want = FUSED(a, b, False, False, 12, 6, False, False, mode.value, 4, False)
    assert torch.equal(got, want)


def test_the_flat_wrappers_take_torch_matmuls_operands():
    """A 1D operand, a batch of rank 4 broadcast against a 2D operand, and a
    transposed view, each equal to the op on the 2D slices."""
    kw: dict[str, Any] = dict(mul_wl=16, mul_fl=8, acc_wl=16, acc_fl=4)
    a, b = _operands((2, 3, 5, 9), (9, 4), F32, seed=2)
    got = fixedpoint_matmul(a, b, **kw)
    assert got.shape == (2, 3, 5, 4)
    for i in range(2):
        for j in range(3):
            assert torch.equal(got[i, j], fixedpoint_matmul(a[i, j], b, **kw))
    v = b[:, 0].contiguous()
    assert torch.equal(
        fixedpoint_matmul(a[0, 0], v, **kw), fixedpoint_matmul(a[0, 0], b, **kw)[:, 0]
    )
    assert torch.equal(
        fixedpoint_matmul(a[0, 0], b.T.contiguous(), trans_b=True, **kw),
        fixedpoint_matmul(a[0, 0], b, **kw),
    )


def test_a_named_carrier_keeps_a_wide_result():
    """carrier=torch.float64 on float32 operands is the call on the widened
    operands: narrowed back when the result fits float32, kept in float64
    when it is wide."""
    a, b = _operands((4, 20), (20, 3), F32, seed=3)
    narrow: dict[str, Any] = dict(mul_wl=16, mul_fl=8, acc_wl=16, acc_fl=4)
    got = fixedpoint_matmul(a, b, **narrow, carrier=F64)
    assert got.dtype == F32
    assert torch.equal(got, fixedpoint_matmul(a.double(), b.double(), **narrow).float())
    wide: dict[str, Any] = dict(mul_wl=16, mul_fl=8, acc_wl=32, acc_fl=8)
    got = fixedpoint_matmul(a, b, **wide, carrier=F64)
    assert got.dtype == F64
    assert torch.equal(got, fixedpoint_matmul(a.double(), b.double(), **wide))


def test_a_flat_wrapper_refuses_what_the_rules_refuse():
    a = torch.ones(2, 3)
    with pytest.raises(ValueError, match="no modulus to wrap"):
        fixedpoint_matmul(
            a, a.T, mul_wl=16, mul_fl=8, acc_wl=16, acc_fl=4, acc_symmetric=True,
            overflow=Overflow.WRAP,
        )  # fmt: skip
    with pytest.raises(ValueError, match="needs 110 bits"):
        fixedpoint_matmul(a, a.T, mul_wl=16, mul_fl=60, acc_wl=40, acc_fl=-10)
    with pytest.raises(TypeError, match="must be an Overflow"):
        fixedpoint_matmul_fma(a, a.T, fma_wl=12, fma_fl=4, overflow=True)  # ty: ignore[invalid-argument-type]


# --- the macs ----------------------------------------------------------------------


@pytest.mark.parametrize("overflow", list(Overflow))
@pytest.mark.parametrize("carrier", [None, F64])
def test_the_macs_resolve_to_the_flat_wrappers_specs(carrier, overflow):
    mul, acc = FixedPoint(16, 8, prng_bits=3), FixedPoint(20, 6, is_signed=False, prng_bits=4)
    mac = SplitMac(mul, acc, rounding=R.SR_B, carrier=carrier, overflow=overflow)
    assert spec_for_mac(mac) == _fixedpoint_spec(
        mul_wl=16, mul_fl=8, mul_prng_bits=3, acc_wl=20, acc_fl=6, acc_is_signed=False,
        acc_prng_bits=4, rounding_mode=R.SR_B, overflow=overflow, carrier=carrier,
    )  # fmt: skip
    fused = FusedMac(acc, rounding=R.RU, carrier=carrier, overflow=overflow)
    assert spec_for_mac(fused) == _fixedpoint_fma_spec(
        fma_wl=20, fma_fl=6, fma_is_signed=False, fma_prng_bits=4, rounding_mode=R.RU,
        overflow=overflow, carrier=carrier,
    )  # fmt: skip


@pytest.mark.parametrize(
    "build, error, match",
    [
        (lambda: SplitMac(MUL), ValueError, "needs an accumulate format"),
        (lambda: SplitMac(MUL, BinaryK(16, 8)), TypeError, "fixed-point formats only"),
        (lambda: SplitMac(BinaryK(8, 4), ACC), TypeError, "fixed-point formats only"),
        (lambda: SplitMac([MUL, FixedPoint(16, 6)], ACC), ValueError, "one format per slot"),
        (lambda: FusedMac([ACC, FixedPoint(16, 6)]), ValueError, "one format per slot"),
        (
            lambda: SplitMac(MUL, ACC, accumulate_algorithm=AccumulateAlgorithm.KAHAN),
            ValueError,
            "NAIVE only",
        ),
        (lambda: FusedMac(ACC, carrier_rounding=R.RO), ValueError, "RoundMode.RNE only"),
        (
            lambda: SplitMac(BinaryK(8, 4), BinaryK(16, 8), overflow=Overflow.WRAP),
            ValueError,
            "needs a fixed-point accumulate format",
        ),
        (
            lambda: SplitMac(MUL, FixedPoint(16, 4, symmetric=True), overflow=Overflow.WRAP),
            ValueError,
            "no modulus",
        ),
        (lambda: FusedMac(ACC, overflow="wrap"), TypeError, "must be an Overflow"),  # ty: ignore[invalid-argument-type]
    ],
)
def test_a_mac_refuses_what_the_fixed_point_gemm_does_not_take(build, error, match):
    with pytest.raises(error, match=match):
        build()


def test_overflow_wraps_around_or_saturates():
    """int8 sums of 100, 100, -100 and -60: the partial sums overflow at the
    second step, and the final sum, 40, fits."""
    a = torch.tensor([[100.0, 100.0, -100.0, -60.0]])
    b = torch.ones(4, 1)
    int8, int16 = FixedPoint(8, 0), FixedPoint(16, 0)
    assert qmm(a, b, SplitMac(int16, int8)).item() == -33
    assert qmm(a, b, SplitMac(int16, int8, overflow=Overflow.WRAP)).item() == 40
    assert qmm(a, b, FusedMac(int8, overflow=Overflow.WRAP)).item() == 40


# --- the layers ----------------------------------------------------------------------


def _grid(shape, seed, device, dtype=F64):
    return _on_grid(shape, GRID, seed).to(device, dtype)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_tier1_a_linear_layer_with_exact_sums_is_nn_linear(fused, device):
    """Values of an 8-bit grid in a float64 layer: every product is a value of
    the multiply format, and every sum fits the accumulate format, so the
    forward and both gradients are nn.Linear's, bit for bit."""
    if fused:
        formats = fixedpoint_gemm_formats_fma(40, 8)
    else:
        formats = fixedpoint_gemm_formats(24, 8, acc_wl=40, acc_fl=8)
    layer = QLinear(16, 6, formats=formats).to(device, F64)
    plain = nn.Linear(16, 6).to(device, F64)
    with torch.no_grad():
        for p, q in ((layer.weight, plain.weight), (_given(layer.bias), _given(plain.bias))):
            p.copy_(_grid(p.shape, p.numel(), device))
            q.copy_(p)
    x = _grid((5, 16), 1, device).requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    g = _grid((5, 6), 2, device)
    y, y2 = layer(x), plain(x2)
    assert torch.equal(y, y2)
    y.backward(g)
    y2.backward(g)
    assert torch.equal(x.grad, x2.grad)
    assert torch.equal(_given(layer.weight.grad), _given(plain.weight.grad))


@pytest.mark.parametrize("device", FLOAT32_DEVICES)
def test_tier2_a_linear_layer_is_close_to_nn_linear(device):
    torch.manual_seed(0)
    formats = fixedpoint_gemm_formats(16, 10, acc_wl=24, acc_fl=10)
    formats.input_quant = formats.weight_quant = Quant(FixedPoint(12, 8))
    layer = QLinear(32, 8, formats=formats).to(device)
    plain = nn.Linear(32, 8).to(device)
    plain.load_state_dict(layer.state_dict())
    x = torch.randn(16, 32, device=device)
    err = (layer(x) - plain(x)).abs().max().item()
    # 32 products each within half a multiply step and two operand steps of
    # their unrounded value, and the sum within half an accumulate step
    assert err < 32 * (2**-11 + 2 * 2**-9) + 2**-11, err


@pytest.mark.parametrize("device", FLOAT32_DEVICES)
@pytest.mark.parametrize("mode", [R.RNE, R.RD])
def test_tier3_a_linear_layer_runs_the_op_in_each_pass(mode, device):
    kw: dict[str, Any] = dict(mul_wl=16, mul_fl=8, acc_wl=16, acc_fl=4, rounding_mode=mode)
    formats = fixedpoint_gemm_formats(16, 8, acc_wl=16, acc_fl=4, rounding_mode=mode)
    layer = QLinear(20, 7, formats=formats).to(device)
    x, g = _operands((6, 20), (6, 7), F32, seed=4)
    x, g = x.to(device).requires_grad_(), g.to(device)
    y = layer(x)
    w, bias = layer.weight.detach(), _given(layer.bias).detach()
    assert torch.equal(y, fixedpoint_matmul(x.detach(), w, trans_b=True, **kw) + bias)
    y.backward(g)
    assert torch.equal(x.grad, fixedpoint_matmul(g, w, **kw))
    assert torch.equal(
        _given(layer.weight.grad), fixedpoint_matmul(g, x.detach(), trans_a=True, **kw)
    )


@pytest.mark.parametrize("device", FLOAT32_DEVICES)
def test_tier3_a_conv_layer_runs_the_conv_op_in_each_pass(device):
    layer = QConv2d(3, 4, 3, stride=2, padding=1, formats=conv_formats(SplitMac(MUL, ACC)))
    layer = layer.to(device)
    x = (torch.randn(2, 3, 9, 8, generator=torch.Generator().manual_seed(6)) * 3).to(device)
    x.requires_grad_()
    y = layer(x)
    w, bias = layer.weight.detach(), _given(layer.bias).detach()
    geo = ([2, 2], [1, 1], [1, 1])
    fwd = _conv(0, w, x.detach(), y.shape[2:], geo, 1, acc=ACC, mul=MUL)
    assert torch.equal(y, fwd + bias.view(-1, 1, 1))
    g = torch.randn(y.shape, generator=torch.Generator().manual_seed(7)).to(device)
    y.backward(g)
    assert torch.equal(_given(x.grad), _conv(1, w, g, x.shape[2:], geo, 1, acc=ACC, mul=MUL))
    assert torch.equal(
        _given(layer.weight.grad), _conv(2, g, x.detach(), w.shape[2:], geo, 1, acc=ACC, mul=MUL)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_tier1_a_conv_layer_with_exact_sums_is_nn_conv2d(device):
    mac = SplitMac(FixedPoint(24, 8), FixedPoint(40, 8))
    layer = QConv2d(2, 3, 3, padding=1, formats=conv_formats(mac)).to(device, F64)
    plain = nn.Conv2d(2, 3, 3, padding=1).to(device, F64)
    with torch.no_grad():
        for p, q in ((layer.weight, plain.weight), (_given(layer.bias), _given(plain.bias))):
            p.copy_(_grid(p.shape, p.numel() + 10, device))
            q.copy_(p)
    x = _grid((2, 2, 6, 5), 11, device).requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    y, y2 = layer(x), plain(x2)
    assert torch.equal(y, y2)
    g = _grid(y.shape, 12, device)
    y.backward(g)
    y2.backward(g)
    assert torch.equal(x.grad, x2.grad)
    assert torch.equal(_given(layer.weight.grad), _given(plain.weight.grad))


@pytest.mark.parametrize("device", FLOAT32_DEVICES)
def test_tier3_a_matmul_runs_the_op_in_each_pass(device):
    kw: dict[str, Any] = dict(mul_wl=16, mul_fl=8, acc_wl=16, acc_fl=4)
    a, b = (t.to(device).requires_grad_() for t in _operands((5, 9), (9, 4), F32, seed=8))
    g = torch.randn(5, 4, generator=torch.Generator().manual_seed(9)).to(device)
    y = qmatmul(a, b, SplitMac(MUL, ACC))
    assert torch.equal(y, fixedpoint_matmul(a.detach(), b.detach(), **kw))
    y.backward(g)
    assert torch.equal(a.grad, fixedpoint_matmul(g, b.detach(), trans_b=True, **kw))
    assert torch.equal(b.grad, fixedpoint_matmul(a.detach(), g, trans_a=True, **kw))


# --- a wide result -------------------------------------------------------------------

WIDE_MAC = SplitMac(MUL, FixedPoint(32, 8))
OUT = FixedPoint(16, 6)


# Each case's sums carry more significant bits than its dtype, so rounding a
# sum once and narrowing it first, then rounding, give different words; each
# test checks that its data tells the two apart. Per dtype: the multiply
# format, the accumulate format, the output format (no more significant bits
# than the dtype: the storage rule refuses one with more), K, and the
# operands' scale.
WIDE_CASES = {
    F32: (FixedPoint(24, 8), FixedPoint(40, 8), FixedPoint(24, 2), 256, 64.0),
    torch.bfloat16: (FixedPoint(16, 8), FixedPoint(32, 8), FixedPoint(8, 3), 24, 4.0),
}


def _scaled(shape, scale, seed, device, dtype):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=gen) * scale).to(device, dtype)


def _rounded_once_and_twice(sum64, out, dtype):
    once = fixedpoint_quantize(sum64, out.wl, out.fl).to(dtype)
    twice = fixedpoint_quantize(sum64.to(dtype), out.wl, out.fl)
    assert not torch.equal(once, twice), "the data does not tell the two orders apart"
    return once


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", list(WIDE_CASES), ids=["float32", "bfloat16"])
def test_a_wide_forward_is_rounded_once_by_its_output_rounding(dtype, device):
    """The product in float64, the bias added in float64, one rounding to the
    output format, then the layer's dtype: through the factory and through a
    recipe's GemmFormats alike."""
    mul, acc, out, k, scale = WIDE_CASES[dtype]
    formats = fixedpoint_gemm_formats(
        mul.wl, mul.fl, acc_wl=acc.wl, acc_fl=acc.fl, output_quant=out
    )
    layer = QLinear(k, 16, formats=formats).to(device, dtype)
    with torch.no_grad():
        layer.weight.copy_(_scaled(layer.weight.shape, scale, 10, device, dtype))
    x = _scaled((32, k), scale, 11, device, dtype)
    w, bias = layer.weight.detach(), _given(layer.bias).detach()
    kw: dict[str, Any] = dict(mul_wl=mul.wl, mul_fl=mul.fl, acc_wl=acc.wl, acc_fl=acc.fl)
    product = fixedpoint_matmul(x, w, trans_b=True, **kw)
    assert product.dtype == F64
    want = _rounded_once_and_twice(product + bias.double(), out, dtype)
    got = layer(x)
    assert got.dtype == dtype and torch.equal(got, want)
    layer.formats = GemmFormats(fwd_mac=SplitMac(mul, acc), output_quant=out).affine()
    assert torch.equal(layer(x), want)


@pytest.mark.parametrize("device", DEVICES)
def test_a_wide_conv_forward_is_rounded_once_by_its_output_rounding(device):
    mul, acc, out, _, scale = WIDE_CASES[F32]
    formats = GemmFormats(fwd_mac=SplitMac(mul, acc), output_quant=out).conv()
    layer = QConv2d(8, 4, 3, formats=formats).to(device)
    with torch.no_grad():
        layer.weight.copy_(_scaled(layer.weight.shape, scale, 12, device, F32))
    x = _scaled((2, 8, 7, 7), scale, 13, device, F32)
    w, bias = layer.weight.detach(), _given(layer.bias).detach()
    product = _conv(0, w, x, (5, 5), ([1, 1], [0, 0], [1, 1]), 1, acc=acc, mul=mul)
    assert product.dtype == F64
    want = _rounded_once_and_twice(product + bias.double().view(-1, 1, 1), out, F32)
    assert torch.equal(layer(x), want)


@pytest.mark.parametrize("device", DEVICES)
def test_a_wide_result_without_an_output_rounding_or_in_a_gradient_raises(device):
    x = torch.randn(3, 8, device=device, requires_grad=True)
    bare = QLinear(8, 2, formats=fixedpoint_gemm_formats(16, 8, acc_wl=32, acc_fl=8)).to(device)
    with pytest.raises(ValueError, match="this forward pass has no output rounding"):
        bare(x)
    rounded = fixedpoint_gemm_formats(16, 8, acc_wl=32, acc_fl=8, output_quant=OUT)
    y = QLinear(8, 2, formats=rounded).to(device)(x)
    with pytest.raises(ValueError, match="a gradient pass stores it in"):
        y.sum().backward()
    with pytest.raises(ValueError, match="this forward pass has no output rounding"):
        qmm(x.detach(), x.detach().T, WIDE_MAC)
    # float64 operands are never wide: the same mac runs there
    assert qmm(x.detach().double(), x.detach().double().T, WIDE_MAC).dtype == F64


@requires_mps
def test_on_mps_a_wide_result_is_refused_in_every_pass():
    """MPS has no float64, so a wide result is refused even where an output
    rounding would store it: by the layer's hook without one, and by the op
    with one."""
    x = torch.randn(3, 8, device="mps")
    bare = QLinear(8, 2, formats=fixedpoint_gemm_formats(16, 8, acc_wl=32, acc_fl=8)).to("mps")
    with pytest.raises(ValueError, match="which MPS does not have"):
        bare(x)
    rounded = fixedpoint_gemm_formats(16, 8, acc_wl=32, acc_fl=8, output_quant=OUT)
    with pytest.raises(RuntimeError, match="which MPS does not have"):
        QLinear(8, 2, formats=rounded).to("mps")(x)
    with pytest.raises(ValueError, match="which MPS does not have"):
        qmm(x, x.T, WIDE_MAC)


@pytest.mark.parametrize("where", ["forward", "gradient"])
def test_the_wide_refusal_on_mps_offers_no_float64(where):
    """The message an MPS operand gets, checked where MPS is absent too."""
    error = str(_wide_refusal(spec_for_mac(WIDE_MAC), F32, where, torch.device("mps")))
    assert "which MPS does not have" in error and "compute on the CPU" in error
    assert "float64 operands" not in error and "float64 gradients" not in error


# --- recipes --------------------------------------------------------------------------


def _mlp() -> nn.Module:
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4))


@pytest.mark.parametrize("mode", [R.RNE, R.SR])
def test_convert_and_simulate_compute_the_same_words(mode):
    mac = SplitMac(FixedPoint(16, 8, prng_bits=5), FixedPoint(20, 8, prng_bits=5), rounding=mode)
    grid = FixedPoint(10, 6)
    recipe = Recipe(
        linear=GemmFormats(
            input_quant=grid, weight_quant=grid, fwd_mac=mac, igrad_mac=mac, wgrad_mac=mac
        )
    )
    x = torch.randn(5, 16, generator=torch.Generator().manual_seed(2))
    results = []
    for run in ("convert", "simulate"):
        model = _mlp()
        torch.manual_seed(3)
        if run == "convert":
            converted, _ = convert(model, recipe)
            y = converted(x)
            y.sum().backward()
            grads = [_given(p.grad) for p in converted.parameters()]
        else:
            with simulate(recipe, model):
                y = model(x)
                y.sum().backward()
            grads = [_given(p.grad) for p in model.parameters()]
        results.append((y, grads))
    (y1, g1), (y2, g2) = results
    assert torch.equal(y1, y2)
    assert all(torch.equal(a, b) for a, b in zip(g1, g2, strict=True))


def test_profile_files_the_products_under_the_mac():
    mac = SplitMac(MUL, FixedPoint(20, 8))
    recipe = Recipe(linear=GemmFormats(fwd_mac=mac))
    p = profile(_mlp(), torch.randn(3, 16), recipe=recipe)
    assert "FixedPoint(wl=16, fl=8)" in p.summary()
    assert sum(r.macs for r in p.records) == 3 * (16 * 12 + 12 * 4)


# --- FixedPoint's random bits ------------------------------------------------------


def test_a_wide_stochastic_accumulator_can_be_built_and_run():
    """mag_bits + prng_bits past binary64's 53 is the elementwise cast's limit,
    not the format's: the GEMM's integer accumulator reads the random bits
    from a draw of its own."""
    acc = FixedPoint(40, 8, prng_bits=20)
    a, b = _operands((4, 30), (30, 3), F64, seed=12)
    out = qmm(a, b, SplitMac(FixedPoint(16, 8, prng_bits=5), acc, rounding=R.SR))
    assert out.dtype == F64 and out.shape == (4, 3)
    with pytest.raises(ValueError, match="stochastic-rounding bits"):
        Quant(acc, R.SR)(a)


# --- compile --------------------------------------------------------------------------


def test_a_fixed_point_layer_compiles_to_its_eager_words():
    formats = fixedpoint_gemm_formats(16, 8, acc_wl=16, acc_fl=4)
    layer = QLinear(12, 5, formats=formats)
    x = torch.randn(4, 12, generator=torch.Generator().manual_seed(13))
    compiled = torch.compile(layer, fullgraph=True, backend="aot_eager")
    assert torch.equal(compiled(x), layer(x))
