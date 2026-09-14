"""A float64 model, end to end, in the arithmetic of its own carrier.

The other float64 tests hold the pieces: tests/test_quantize_dispatch.py and
tests/test_binaryk_p3109.py the elementwise casts, tests/test_gemm_dispatch.py
each GEMM against an exact reference through ``torch.ops``, and
tests/test_format_limits.py the bounds. This file holds the tier a model is
written in -- format objects, ``qmatmul`` and its gradients, ``QLinear`` over a
GEMM factory -- on float64 tensors and formats only binary64 carries, in the
three tiers the layer tests use (CLAUDE.md):

* **Tier 1 (exact):** a 53-bit format rounds every float64 in its normal range
  to itself, so a GEMM in it is float64 arithmetic in the kernel's
  k-ascending order -- forward and both gradients -- bit for bit.
* **Tier 2 (statistical):** a 30-bit layer stays within its own precision of
  the float64 ``nn.Linear``, and nearer to it than the same layer run in
  binary32 with ``carrier="binary32"`` can be.
* **Tier 3 (manual baseline):** a ``SplitMac`` and a ``FusedMac`` in wide
  formats, recomputed step by step from the elementwise quantizer -- with
  ``fractions.Fraction`` for the fused step, since Python 3.12 has no
  ``math.fma`` -- forward and gradients.

And the audited edges of binary64's bounds, where the formats sit exactly: their
values against tests/test_binaryk_p3109.py's projection of the standard.
"""

import math
import warnings
from fractions import Fraction

import pytest
import torch
from torch import nn

from mptorch import BinaryK, FormatRangeWarning, RoundMode, SaturationMode
from mptorch.quant import FusedMac, QLinear, Quant, SplitMac, binaryK_gemm_formats, qmatmul
from tests.markers import available_devices
from tests.test_binaryk_p3109 import _project

F64 = torch.float64


def _seq_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` as the kernel sums it: one K-step at a time, k ascending."""
    s = torch.zeros(a.shape[0], b.shape[1], dtype=F64, device=a.device)
    for k in range(a.shape[1]):
        s = s + a[:, k, None] * b[None, k, :]
    return s


def _silent(call):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FormatRangeWarning)
        return call()


# --- Tier 1 -------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mac", ["split", "fused"])
def test_tier1_53_bit_gemm_is_float64_arithmetic(device, mac):
    torch.manual_seed(3)
    a = torch.randn(9, 33, dtype=F64, device=device, requires_grad=True)
    b = torch.randn(33, 7, dtype=F64, device=device, requires_grad=True)
    fmt = BinaryK(63, 53)  # ten exponent bits at P3109's bias, inside binary64's bounds
    formats = SplitMac(fmt, fmt) if mac == "split" else FusedMac(fmt)
    out = _silent(lambda: qmatmul(a, b, formats))
    ad, bd = a.detach(), b.detach()
    if mac == "split":
        assert torch.equal(out, _seq_matmul(ad, bd))
    else:
        # the fused step is float64's own FMA, rounded to 53 bits: itself
        ref = torch.zeros(9, 7, dtype=torch.float64)
        al, bl = ad.cpu().tolist(), bd.cpu().tolist()
        for i in range(9):
            for j in range(7):
                acc = 0.0
                for k in range(33):
                    acc = float(Fraction(al[i][k]) * Fraction(bl[k][j]) + Fraction(acc))
                ref[i, j] = acc
        assert torch.equal(out.cpu(), ref)
    g = torch.randn_like(out)
    out.backward(g)
    assert a.grad is not None and b.grad is not None
    if mac == "split":
        assert torch.equal(a.grad, _seq_matmul(g, bd.mT))
        assert torch.equal(b.grad, _seq_matmul(ad.mT, g))


@pytest.mark.parametrize("device", available_devices)
def test_tier1_53_bit_qlinear_is_float64_arithmetic(device):
    torch.manual_seed(4)
    layer = QLinear(24, 6, formats=binaryK_gemm_formats(63, 53), device=device, dtype=F64)
    x = torch.randn(5, 24, dtype=F64, device=device, requires_grad=True)
    out = _silent(lambda: layer(x))
    w, bias = layer.weight.detach(), layer.bias
    assert bias is not None
    assert torch.equal(out, _seq_matmul(x.detach(), w.mT) + bias.detach())
    g = torch.randn_like(out)
    out.backward(g)
    assert x.grad is not None and layer.weight.grad is not None
    assert torch.equal(x.grad, _seq_matmul(g, w))
    assert torch.equal(layer.weight.grad, _seq_matmul(g.mT, x.detach()))


# --- Tier 2 -------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_tier2_wide_layer_holds_its_precision(device):
    """A 30-bit accumulator in binary64 is ~2**-29 from float64 per step; the
    same layer run in binary32 is refused the format outright, and with a
    format binary32 does carry it is ~2**-23 from float64 at best."""
    torch.manual_seed(5)
    vanilla = nn.Linear(64, 32, device=device, dtype=F64)
    fmt: dict = dict(mul_K=40, mul_P=30, mul_bias=512, acc_K=40, acc_P=30, acc_bias=512)
    layer = QLinear(64, 32, formats=binaryK_gemm_formats(**fmt), device=device, dtype=F64)
    narrow = QLinear(
        64,
        32,
        formats=binaryK_gemm_formats(31, 24, carrier="binary32"),
        device=device,
        dtype=F64,
    )
    with torch.no_grad():
        for q in (layer, narrow):
            q.weight.copy_(vanilla.weight)
            assert q.bias is not None and vanilla.bias is not None
            q.bias.copy_(vanilla.bias)
    x = torch.randn(16, 64, dtype=F64, device=device)
    ref = vanilla(x)

    def err(out: torch.Tensor) -> float:
        return (torch.linalg.norm(out - ref) / torch.linalg.norm(ref)).item()

    # 64 steps of a product and a sum, each within an ulp of its own magnitude
    wide_err, narrow_err = err(_silent(lambda: layer(x))), err(narrow(x))
    assert 0 < wide_err < 128 * 2.0**-30
    assert 32 * wide_err < narrow_err
    # and binary32 cannot hold the wide format at all
    refuse = QLinear(
        64, 32, formats=binaryK_gemm_formats(**fmt, carrier="binary32"), device=device, dtype=F64
    )
    with pytest.raises(ValueError, match="30 bits of precision"):
        refuse(x)


# --- Tier 3 -------------------------------------------------------------------

MUL, ACC = BinaryK(40, 30, bias=512), BinaryK(48, 40)


def _q(fmt: BinaryK, x: torch.Tensor, rm: RoundMode) -> torch.Tensor:
    return Quant(fmt, rm)(x)


def _split_reference(a: torch.Tensor, b: torch.Tensor, rm: RoundMode) -> torch.Tensor:
    s = torch.zeros(a.shape[0], b.shape[1], dtype=F64, device=a.device)
    for k in range(a.shape[1]):
        s = _q(ACC, s + _q(MUL, a[:, k, None] * b[None, k, :], rm), rm)
    return s


def _fused_reference(a: torch.Tensor, b: torch.Tensor, rm: RoundMode) -> torch.Tensor:
    al, bl = a.cpu().tolist(), b.cpu().tolist()
    rows, cols, K = a.shape[0], b.shape[1], a.shape[1]
    s = torch.zeros(rows, cols, dtype=F64)
    for k in range(K):
        sl = s.tolist()
        exact = [
            [
                float(Fraction(al[i][k]) * Fraction(bl[k][j]) + Fraction(sl[i][j]))
                for j in range(cols)
            ]
            for i in range(rows)
        ]
        s = _q(ACC, torch.tensor(exact, dtype=F64), rm)
    return s.to(a.device)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rm", [RoundMode.RNE, RoundMode.RD, RoundMode.RO])
@pytest.mark.parametrize("mac", ["split", "fused"])
def test_tier3_wide_macs_match_the_manual_baseline(device, rm, mac):
    torch.manual_seed(6)
    # float64 operands with bits float32 lacks, and a product range the 40-bit
    # accumulator rounds visibly
    a = (torch.randn(6, 11, dtype=F64, device=device) * 3).requires_grad_(True)
    b = (torch.randn(11, 5, dtype=F64, device=device) / 3).requires_grad_(True)
    formats = SplitMac(MUL, ACC, rounding=rm) if mac == "split" else FusedMac(ACC, rounding=rm)
    reference = _split_reference if mac == "split" else _fused_reference
    out = _silent(lambda: qmatmul(a, b, formats))
    ad, bd = a.detach(), b.detach()
    assert torch.equal(out, reference(ad, bd, rm))
    g = torch.randn_like(out)
    out.backward(g)
    assert a.grad is not None and b.grad is not None
    assert torch.equal(a.grad, reference(g, bd.mT, rm))
    assert torch.equal(b.grad, reference(ad.mT, g, rm))
    # the reference is not the plain product: the formats did something
    assert not torch.equal(out, _seq_matmul(ad, bd))


# --- binary64's edges ---------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "K,P,bias",
    [
        (14, 4, 1019),  # bias + P = 1023, SUBNORMALS' lowest in binary64
        (63, 53, 970),  # ... at binary64's full precision
        (11, 1, 1022),  # ... at one bit
        (14, 4, 0),  # the top: 2**exp_bits - 1024, binary64's highest
        (63, 53, 511),
    ],
)
def test_formats_on_binary64s_edges_are_exact(device, K, P, bias):
    """Silent on a float64 tensor, and the standard's answer for every input
    around both ends of the format -- including binary64 subnormals."""
    exp_bits = K - P
    bottom, top = 1 - bias - (P - 1), 2**exp_bits - 1 - bias
    g = torch.Generator().manual_seed(K + bias)
    words = []
    for e in sorted({*range(bottom - 3, bottom + 4), *range(top - 2, min(top + 2, 1023) + 1)}):
        if -1075 < e < 1024:
            man = torch.randint(0, 1 << 52, (512,), generator=g)
            field = max(e + 1023, 0)
            words.append((field << 52) | man)
    x = torch.cat(words).view(F64)
    x = torch.cat([x, -x, torch.tensor([0.0, 5e-324, math.ldexp(1.0, bottom)], dtype=F64)])
    for rm in (
        RoundMode.RNE,
        RoundMode.RNA,
        RoundMode.RU,
        RoundMode.RD,
        RoundMode.RZ,
        RoundMode.RO,
    ):
        fmt = BinaryK(K, P, bias=bias)
        got = _silent(lambda fmt=fmt, rm=rm: Quant(fmt, rm)(x.to(device))).cpu()
        want = _project(x, K, P, True, rm, SaturationMode.OVF_INF, bias)
        same = (got == want) & (torch.signbit(got) == torch.signbit(want))
        assert bool((same | (got.isnan() & want.isnan())).all()), rm.name
