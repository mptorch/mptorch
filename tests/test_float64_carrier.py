"""The binary64 carrier, end to end: a float64 model in its own carrier's
arithmetic, and a narrower one that names it.

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
  the float64 ``nn.Linear``, and nearer to it than a float32 layer, rounding
  in binary32, can be.
* **Tier 3 (manual baseline):** a ``SplitMac`` and a ``FusedMac`` in wide
  formats, recomputed step by step from the elementwise quantizer -- with
  ``fractions.Fraction`` for the fused step, since Python 3.12 has no
  ``math.fma`` -- forward and gradients.

And the audited edges of binary64's bounds, where the formats sit exactly: their
values against tests/test_binaryk_p3109.py's projection of the standard.

``carrier=torch.float64`` gives a float32, float16 or bfloat16 tensor the same
arithmetic, and narrows each result back to its dtype with one rounding --
which torch's own ``.to()`` does not give float16 and bfloat16, so the last
section holds that conversion, the ``narrow_float64`` op, to an exact reference
and to a second spelling in torch arithmetic, and a float32 layer to the
float64 one it widens to.
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

F32, F64 = torch.float32, torch.float64


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
    """A 30-bit accumulator in binary64 is ~2**-29 from float64 per step; a
    float32 layer rounding in binary32 is refused the format outright, and with
    a format binary32 does carry it is ~2**-23 from float64 at best."""
    torch.manual_seed(5)
    vanilla = nn.Linear(64, 32, device=device, dtype=F64)
    fmt: dict = dict(mul_K=40, mul_P=30, mul_bias=512, acc_K=40, acc_P=30, acc_bias=512)
    layer = QLinear(64, 32, formats=binaryK_gemm_formats(**fmt), device=device, dtype=F64)
    narrow = QLinear(64, 32, formats=binaryK_gemm_formats(31, 24), device=device, dtype=F32)
    with torch.no_grad():
        for q in (layer, narrow):
            q.weight.copy_(vanilla.weight)
            assert q.bias is not None and vanilla.bias is not None
            q.bias.copy_(vanilla.bias)
    x = torch.randn(16, 64, dtype=F64, device=device)
    ref = vanilla(x)

    def err(out: torch.Tensor) -> float:
        return (torch.linalg.norm(out.double() - ref) / torch.linalg.norm(ref)).item()

    # 64 steps of a product and a sum, each within an ulp of its own magnitude
    wide_err, narrow_err = err(_silent(lambda: layer(x))), err(narrow(x.float()))
    assert 0 < wide_err < 128 * 2.0**-30
    assert 32 * wide_err < narrow_err
    # and binary32 cannot hold the wide format at all, nor a float32 result
    # store it when the layer names binary64
    for carrier in (None, F64):
        refuse = QLinear(
            64, 32, formats=binaryK_gemm_formats(**fmt, carrier=carrier), device=device, dtype=F32
        )
        with pytest.raises(ValueError, match="30 bits of precision"):
            refuse(x.float())


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


# --- a narrower tensor in binary64 --------------------------------------------

# stored significand bits, and the exponents of the smallest normal and largest
# finite value: IEEE binary16's, bfloat16's, which is binary32's range at eight
# bits of precision, and binary32's
_GRIDS = {
    torch.float16: (10, -14, 15),
    torch.bfloat16: (7, -126, 127),
    torch.float32: (23, -126, 127),
}


def _nearest(v: float, dtype: torch.dtype) -> float:
    """``v`` rounded to nearest-even onto ``dtype``'s grid, exactly, by Fraction."""
    if not math.isfinite(v) or v == 0.0:
        return v
    man_bits, min_normal_exp, top_exp = _GRIDS[dtype]
    lead = math.frexp(abs(v))[1] - 1
    step = Fraction(2) ** (max(lead, min_normal_exp) - man_bits)
    q = Fraction(abs(v)) / step
    n = math.floor(q)
    if q - n > Fraction(1, 2) or (q - n == Fraction(1, 2) and n % 2):
        n += 1
    m = n * step
    largest = (2 - Fraction(2) ** -man_bits) * Fraction(2) ** top_exp
    return math.copysign(math.inf if m > largest else float(m), v)


def _hard_cases(dtype: torch.dtype, count: int, gen: torch.Generator) -> torch.Tensor:
    """float64 values at the conversion's hard points: a hair either side of a
    tie on the dtype's grid (the float32 round-trip lands on the tie), both
    overflow boundaries, the subnormal floor, and random words across the range."""
    man_bits, min_normal_exp, top_exp = _GRIDS[dtype]
    lo = min_normal_exp - man_bits
    exps = torch.randint(lo, top_exp - man_bits + 1, (count,), generator=gen).double()
    grid = torch.randint(0, 1 << (man_bits + 1), (count,), generator=gen).double()
    tie = (2 * grid + 1) * torch.pow(2.0, exps - 1)
    hair = torch.pow(2.0, exps - 1 - 30)
    largest = (2 - 2.0**-man_bits) * 2.0**top_exp
    threshold = largest + 2.0 ** (top_exp - man_bits - 1)
    fixed = [
        largest,
        threshold,
        threshold - threshold * 2.0**-50,
        threshold * (1 + 2.0**-50),
        2.0**lo,
        2.0 ** (lo - 1),
        2.0 ** (lo - 1) + 2.0 ** (lo - 30),
        2.0 ** (lo - 1) - 2.0 ** (lo - 30),
        2.0 ** (min_normal_exp - 1) * 3 + 2.0 ** (lo - 20),
        0.0,
        5e-324,
        1e300,
        math.inf,
        math.nan,
    ]
    words = torch.randint(-(1 << 62), 1 << 62, (count,), generator=gen).view(F64)
    x = torch.cat([tie, tie + hair, tie - hair, torch.tensor(fixed, dtype=F64), words])
    return torch.cat([x, -x])


def _same_words(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Elementwise: the same word, or both NaN (whose payloads a device may
    canonicalize)."""
    bits = {4: torch.int32, 2: torch.int16}[got.element_size()]
    return (got.view(bits) == want.view(bits)) | (got.isnan() & want.isnan())


def _narrowed_by_torch(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The narrowing in torch arithmetic, written apart from
    ``csrc/common/narrow_binary64.h`` to hold it to: the grid's step from the
    word's exponent field, clamped to the subnormal step and past the top and
    built back into a word as the power of two, ``x / step`` exact, and
    ``torch.round``'s ties to even. It was ``_narrowed`` itself before the op."""
    man_bits, min_normal_exp, top_exp = _GRIDS[dtype]
    e = (x.view(torch.int64) >> 52).bitwise_and_(0x7FF).sub_(1023 + man_bits)
    e = e.clamp_(min_normal_exp - man_bits, top_exp + 1 - man_bits)
    step = e.add_(1023).bitwise_left_shift_(52).view(F64)
    return torch.round(x / step).mul_(step).to(dtype)


def _every_field(dtype: torch.dtype, per_field: int, gen: torch.Generator) -> torch.Tensor:
    """float64 words with every exponent field, 0 to 2047: random mantissas,
    and at the target's last kept bit -- which moves with the field below the
    target's smallest normal -- a tie on an odd and an even significand, and a
    word either side of each."""
    man_bits, min_normal_exp, _ = _GRIDS[dtype]
    mask = (1 << 52) - 1
    words = []
    for field in range(2048):
        shift = 52 - man_bits + max(min_normal_exp - (field - 1023), 0)
        frac = torch.randint(0, 1 << 52, (per_field,), generator=gen)
        if shift <= 52:
            q = torch.randint(0, 1 << (53 - shift), (per_field,), generator=gen)
            tie = ((q << shift) | (1 << (shift - 1))) & mask
            frac = torch.cat([frac, tie, (tie - 1) & mask, (tie + 1) & mask])
        frac = torch.cat([frac, torch.tensor([0, 1, mask, 1 << 51])])
        words.append((field << 52) | frac)
    x = torch.cat(words)
    return torch.cat([x, x | (-(1 << 63))]).view(F64)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_narrowing_a_binary64_result_rounds_once(device, dtype):
    """`_narrowed` is correct rounding onto the dtype, at every hard point, and
    so is the torch spelling the op is checked against -- and torch's `.to()`,
    which goes through float32, is not, so this is not vacuous."""
    from mptorch.quant.ops import _narrowed

    x = _hard_cases(dtype, 4096, torch.Generator().manual_seed(21))
    want = torch.tensor([_nearest(v, dtype) for v in x.tolist()], dtype=F64).to(dtype)
    got = _narrowed(x.to(device), dtype).cpu()
    assert got.dtype is dtype
    same = _same_words(got, want)
    assert bool(same.all()), x[~same][:4].tolist()
    assert bool(_same_words(_narrowed_by_torch(x.to(device), dtype).cpu(), want).all())
    assert not torch.equal(x.to(dtype), want)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_narrow_op_against_an_independent_spelling(device, dtype):
    """Every float64 exponent field, ties and their neighbours at every shift,
    and random words: float16 and bfloat16 against the torch spelling, float32
    against the hardware conversion, and CPU against CUDA word for word."""
    gen = torch.Generator().manual_seed(22)
    x = torch.cat(
        [
            _every_field(dtype, 8, gen),
            torch.randint(-(1 << 63), (1 << 63) - 1, (1 << 18,), generator=gen).view(F64),
        ]
    )
    op = torch.ops.mptorch.narrow_float64.default
    got = op(x.to(device), dtype)
    assert got.dtype is dtype and got.shape == x.shape
    want = x.to(device).float() if dtype is F32 else _narrowed_by_torch(x.to(device), dtype)
    same = _same_words(got, want)
    assert bool(same.all()), x[~same.cpu()][:4].tolist()
    if device == "cuda":  # NaN included, so words rather than values
        bits = {4: torch.int32, 2: torch.int16}[got.element_size()]
        assert torch.equal(got.cpu().view(bits), op(x, dtype).view(bits))


@pytest.mark.parametrize("device", available_devices)
def test_narrow_op_contract(device):
    op = torch.ops.mptorch.narrow_float64.default
    x = torch.randn(4099, dtype=F64, device=device) * 1e3
    # a strided input, and a contiguous view 8 bytes into its storage -- which a
    # 16-byte vector load cannot read in place -- read what their copies do
    assert torch.equal(op(x[1::2], torch.float16), op(x[1::2].contiguous(), torch.float16))
    assert torch.equal(op(x[1:], torch.float16), op(x[1:].clone(), torch.float16))
    assert op(x[:4098].view(2, 3, 683), torch.bfloat16).shape == (2, 3, 683)
    assert op(x[:0], torch.float16).shape == (0,)
    with pytest.raises(RuntimeError, match="must be float64"):
        op(x.float(), torch.float16)
    with pytest.raises(RuntimeError, match="narrows to float32, float16 or bfloat16"):
        op(x, torch.int32)
    with pytest.raises(RuntimeError, match="narrow_float64 is not differentiable"):
        op(x.clone().requires_grad_(True), torch.float16)


@pytest.mark.parametrize("device", available_devices)
def test_float32_layer_in_binary64_is_the_float64_layer_narrowed(device):
    """Forward and both gradients of a float32 QLinear naming binary64 are the
    float64 layer's on the widened tensors, narrowed -- in formats float32
    stores. No bias: it is added after the GEMM, in each layer's own dtype."""
    torch.manual_seed(7)
    fmt: dict = dict(mul_K=26, mul_P=22, acc_K=27, acc_P=23)
    formats = binaryK_gemm_formats(**fmt, carrier=F64)
    narrow = QLinear(24, 6, bias=False, formats=formats, device=device)
    wide = QLinear(24, 6, bias=False, formats=binaryK_gemm_formats(**fmt), device=device, dtype=F64)
    with torch.no_grad():
        wide.weight.copy_(narrow.weight)
    x = torch.randn(5, 24, device=device, requires_grad=True)
    xw = x.detach().double().requires_grad_(True)
    out, out_w = _silent(lambda: narrow(x)), _silent(lambda: wide(xw))
    assert out.dtype is F32
    assert torch.equal(out, out_w.float())
    g = torch.randn_like(out)
    out.backward(g)
    out_w.backward(g.double())
    assert x.grad is not None and xw.grad is not None
    assert torch.equal(x.grad, xw.grad.float())
    assert narrow.weight.grad is not None and wide.weight.grad is not None
    assert torch.equal(narrow.weight.grad, wide.weight.grad.float())
    # and binary32's arithmetic is not it: the products of 24-bit operands are
    # rounded to 24 bits there before the 22-bit format rounds them
    plain = QLinear(24, 6, bias=False, formats=binaryK_gemm_formats(**fmt), device=device)
    with torch.no_grad():
        plain.weight.copy_(narrow.weight)
        assert not torch.equal(plain(x), out)
