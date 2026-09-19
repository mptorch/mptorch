"""The binary64 carrier, end to end, at the tier a model is written in.

A float64 tensor rounds in binary64, and a float32, float16 or bfloat16 one
does when the call names ``carrier=torch.float64``. Other modules hold the
pieces: ``tests/test_quantize_dispatch.py`` and ``tests/test_binaryk_p3109.py``
the elementwise casts, ``tests/test_gemm_dispatch.py`` each GEMM against an
exact reference through ``torch.ops``, and ``tests/test_format_limits.py`` the
bounds. This module guards what sits above them (format objects, ``qmatmul``
and its gradients, ``QLinear`` over a GEMM factory) on float64 tensors and on
formats only binary64 carries. Without it a wrapper, a factory or a backward
hook that drops to binary32 would still run and return plausible numbers at
24 bits. The sections follow the three tiers of the layer tests:

* **Tier 1 (exact):** a 53-bit format rounds every float64 in its normal range
  to itself, so a GEMM in it is float64 arithmetic in the kernel's
  k-ascending order, forward and both gradients, bit for bit.
* **Tier 2 (statistical):** a 30-bit layer stays within its own precision of
  the float64 ``nn.Linear``, and nearer to it than a float32 layer rounding
  in binary32 can be.
* **Tier 3 (manual baseline):** a ``SplitMac`` and a ``FusedMac`` in wide
  formats, recomputed step by step from the elementwise quantizer, forward and
  gradients. The fused step's exact ``a * b + c`` comes from
  ``fractions.Fraction``, since Python 3.12 has no ``math.fma``.

A fourth section puts formats exactly on the edges of binary64's bounds and
checks their values against ``tests/test_binaryk_p3109.py``'s ``_project``, a
transcription of the projection in IEEE P3109 (arXiv:2606.04028) that shares no
code with the kernels.

The last section covers the narrower tensor. ``carrier=torch.float64`` widens
the operands in Python, runs the binary64 kernel and narrows each result back
to their dtype with one rounding, through the ``narrow_float64`` op. The op
exists because torch's own float64 to float16 or bfloat16 ``.to()`` goes
through float32 and so rounds twice: 65519.999999 becomes float16 infinity
where one rounding gives 65504. The op rounds by integer arithmetic on the
float64 word (``csrc/common/narrow_binary64.h``), and is held to three
references that share no code with it: exact rounding in ``Fraction`` at every
tie plus or minus a hair, a spelling in torch arithmetic over every float64
exponent field, and for float32 the hardware conversion. A float32 layer naming
binary64 is then held to the float64 layer it widens to.
"""

import math
import warnings
from fractions import Fraction

import pytest
import torch
from torch import nn

from mptorch import BinaryK, FormatRangeWarning, RoundMode, SaturationMode
from mptorch.quant import FusedMac, QLinear, Quant, SplitMac, binaryK_gemm_formats, qmatmul
from tests.markers import float64_devices
from tests.test_binaryk_p3109 import _project

F32, F64 = torch.float32, torch.float64


def _seq_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` in float64, summed as the kernel does: one term per step, k ascending.

    The order matters because float64 addition is not associative, and
    ``torch.matmul`` does not promise one."""
    s = torch.zeros(a.shape[0], b.shape[1], dtype=F64, device=a.device)
    for k in range(a.shape[1]):
        s = s + a[:, k, None] * b[None, k, :]
    return s


def _silent(call):
    """Run ``call`` with ``FormatRangeWarning`` as an error and return its result.

    A format inside binary64's bounds must not warn on a float64 tensor."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FormatRangeWarning)
        return call()


# --- Tier 1 -------------------------------------------------------------------


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("mac", ["split", "fused"])
def test_tier1_53_bit_gemm_is_float64_arithmetic(device, mac):
    """``qmatmul`` in a 53-bit format is plain float64 arithmetic, bit for bit.

    Any rounding in binary32 on the way, in the forward or in either gradient's
    GEMM, would lose the low 29 bits and break the equality."""
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
        # The fused step is float64's FMA, and rounding its result to 53 bits
        # changes nothing. float() of the exact Fraction is the FMA's value.
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


@pytest.mark.parametrize("device", float64_devices)
def test_tier1_53_bit_qlinear_is_float64_arithmetic(device):
    """A float64 ``QLinear`` over the 53-bit GEMM factory is float64 arithmetic.

    Holds the factory's Linear contract (which operand each hook transposes,
    the bias add) and its carrier together, forward and both gradients."""
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


@pytest.mark.parametrize("device", float64_devices)
def test_tier2_wide_layer_holds_its_precision(device):
    """A 30-bit float64 layer is as close to ``nn.Linear`` as 30 bits allow.

    Each rounding to 30 bits is within 2**-30 of its operand, relatively. A
    float32 layer rounding in binary32 is about 2**-24 per rounding at best, in
    the widest format binary32 carries, and the 30-bit format is refused to it
    outright."""
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

    # 64 steps, each rounding a product and a sum to 30 bits with a relative error
    # of at most 2**-30, bound the error by 128 * 2**-30. The 24-bit layer is
    # 2**6 coarser per rounding, and half of that margin is asserted.
    wide_err, narrow_err = err(_silent(lambda: layer(x))), err(narrow(x.float()))
    assert 0 < wide_err < 128 * 2.0**-30
    assert 32 * wide_err < narrow_err
    # binary32 cannot round in the 30-bit format (carrier None), and a float32
    # result cannot store it when the layer names binary64: float32 keeps 24
    # bits, and a GEMM's sums reach the format's whole value set.
    for carrier in (None, F64):
        refuse = QLinear(
            64, 32, formats=binaryK_gemm_formats(**fmt, carrier=carrier), device=device, dtype=F32
        )
        with pytest.raises(ValueError, match="30 bits of precision"):
            refuse(x.float())


# --- Tier 3 -------------------------------------------------------------------

# The multiply and accumulate formats, 30 and 40 bits: both past binary32's 24.
MUL, ACC = BinaryK(40, 30, bias=512), BinaryK(48, 40)


def _q(fmt: BinaryK, x: torch.Tensor, rm: RoundMode) -> torch.Tensor:
    """``x`` rounded to ``fmt`` by the elementwise quantizer."""
    return Quant(fmt, rm)(x)


def _split_reference(a: torch.Tensor, b: torch.Tensor, rm: RoundMode) -> torch.Tensor:
    """A ``SplitMac`` GEMM by hand: each float64 product rounded to ``MUL``, added
    to the running sum in float64, and the sum rounded to ``ACC``, k ascending."""
    s = torch.zeros(a.shape[0], b.shape[1], dtype=F64, device=a.device)
    for k in range(a.shape[1]):
        s = _q(ACC, s + _q(MUL, a[:, k, None] * b[None, k, :], rm), rm)
    return s


def _fused_reference(a: torch.Tensor, b: torch.Tensor, rm: RoundMode) -> torch.Tensor:
    """A ``FusedMac`` GEMM by hand: the carrier's FMA, then a rounding to ``ACC``.

    The kernel's fused step is binary64's correctly rounded ``fma(a, b, sum)``
    followed by the format's cast. ``float()`` of the exact ``Fraction`` is that
    same correctly rounded value, computed without an FMA."""
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


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("rm", [RoundMode.RNE, RoundMode.RD, RoundMode.RO])
@pytest.mark.parametrize("mac", ["split", "fused"])
def test_tier3_wide_macs_match_the_manual_baseline(device, rm, mac):
    """``qmatmul`` over wide formats equals the step-by-step recomputation.

    Forward and both gradients, so a backward GEMM wired to the wrong format,
    rounding mode or transpose shows up as a bit difference."""
    torch.manual_seed(6)
    # The operands have bits float32 lacks, and their products have more than the
    # accumulator's 40 bits, so every step rounds.
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
    # The reference differs from the plain product, so the equality above is a
    # statement about the formats and not about float64.
    assert not torch.equal(out, _seq_matmul(ad, bd))


# --- binary64's edges ---------------------------------------------------------


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize(
    "K,P,bias",
    [
        # The smallest value is 2**(2 - bias - P), and binary64 places it
        # faithfully down to 2**-1021, so bias + P = 1023 is the lowest format.
        (14, 4, 1019),
        (63, 53, 970),  # the same floor at binary64's full precision
        (11, 1, 1022),  # and at one bit
        # The top binade is 2**exp_bits - 1 - bias <= 1023, so with ten exponent
        # bits bias = 0 is the highest format.
        (14, 4, 0),
        (63, 53, 511),  # P3109's bias at 53 bits, top binade 512
    ],
)
def test_formats_on_binary64s_edges_are_exact(device, K, P, bias):
    """A format on the edge of binary64's bounds is silent and exactly right.

    Inputs are random words in the binades around both ends of the format,
    binary64 subnormals included (exponent field 0). The bounds are the ones
    ``dev/benchmarks/format_limits.py sweep --carrier binary64 --audit``
    measures, and an off-by-one in them would warn here or misround there."""
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

# Per dtype: stored significand bits, and the exponents of the smallest normal
# and of the largest finite value. bfloat16 is binary32's range at eight bits of
# precision.
_GRIDS = {
    torch.float16: (10, -14, 15),
    torch.bfloat16: (7, -126, 127),
    torch.float32: (23, -126, 127),
}


def _nearest(v: float, dtype: torch.dtype) -> float:
    """``v`` rounded to nearest, ties to even, onto ``dtype``'s grid, in ``Fraction``.

    The grid's step at ``v`` is 2**(max(exponent, smallest normal's) - man_bits).
    Every quantity is an exact rational, so the reference has no rounding of its
    own. A result above the largest finite value is infinity, which is right
    because the step used there is the top binade's."""
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
    """float64 values at the narrowing's hard points, both signs.

    Ties of the dtype's grid and a hair (2**-30 of a step) either side. A
    conversion through float32 rounds the hair away, lands on the tie and then
    goes to even, which is the double rounding. Also the overflow threshold
    (largest plus half a step) and its neighbours, the smallest subnormal and
    the tie at half of it, and random words across float64's range."""
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
    """Elementwise mask: the same word, or both NaN.

    Words, so that the sign of a zero counts. NaN is excepted because a device
    may canonicalize its payload."""
    bits = {4: torch.int32, 2: torch.int16}[got.element_size()]
    return (got.view(bits) == want.view(bits)) | (got.isnan() & want.isnan())


def _narrowed_by_torch(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The narrowing spelled in torch arithmetic, sharing no code with the op.

    The grid's step is a power of two built from the word's exponent field,
    clamped below to the subnormal step and above to the step past the top
    binade. ``x / step`` is then exact, ``torch.round`` rounds ties to even,
    and the final ``.to(dtype)`` converts a value already on the grid (or past
    the largest, to infinity). ``csrc/common/narrow_binary64.h`` does the same
    rounding by integer arithmetic on the word."""
    man_bits, min_normal_exp, top_exp = _GRIDS[dtype]
    e = (x.view(torch.int64) >> 52).bitwise_and_(0x7FF).sub_(1023 + man_bits)
    e = e.clamp_(min_normal_exp - man_bits, top_exp + 1 - man_bits)
    step = e.add_(1023).bitwise_left_shift_(52).view(F64)
    return torch.round(x / step).mul_(step).to(dtype)


def _every_field(dtype: torch.dtype, per_field: int, gen: torch.Generator) -> torch.Tensor:
    """float64 words with every exponent field, 0 to 2047, both signs.

    Each field gets random mantissas and, at the target's last kept bit, ties
    on random (odd and even) significands with the word either side of each.
    ``shift`` is the number of mantissa bits dropped: 52 - man_bits in the
    normal range, and one more per binade below the target's smallest normal."""
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


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_narrowing_a_binary64_result_rounds_once(device, dtype):
    """``_narrowed`` rounds a float64 onto float16 or bfloat16 once, correctly.

    Held to the ``Fraction`` reference at every hard point, as is the torch
    spelling that the next test uses as its reference. torch's ``.to()``, which
    goes through float32, fails the same comparison, so the inputs do
    discriminate."""
    from mptorch.quant.ops import _narrowed

    x = _hard_cases(dtype, 4096, torch.Generator().manual_seed(21))
    want = torch.tensor([_nearest(v, dtype) for v in x.tolist()], dtype=F64).to(dtype)
    got = _narrowed(x.to(device), dtype).cpu()
    assert got.dtype is dtype
    same = _same_words(got, want)
    assert bool(same.all()), x[~same][:4].tolist()
    assert bool(_same_words(_narrowed_by_torch(x.to(device), dtype).cpu(), want).all())
    assert not torch.equal(x.to(dtype), want)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_narrow_op_against_an_independent_spelling(device, dtype):
    """``narrow_float64`` agrees with a second spelling over every exponent field.

    float16 and bfloat16 against ``_narrowed_by_torch``, float32 against the
    hardware conversion ``.float()``, and CUDA against CPU word for word.
    Catches a shift or a sticky-bit error that only one binade range reaches,
    such as the subnormal targets or fields that overflow."""
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
    if device == "cuda":  # words rather than values, so that NaN compares too
        bits = {4: torch.int32, 2: torch.int16}[got.element_size()]
        assert torch.equal(got.cpu().view(bits), op(x, dtype).view(bits))


@pytest.mark.parametrize("device", float64_devices)
def test_narrow_op_contract(device):
    """The op's entry point: layouts, shape, the empty tensor and its refusals."""
    op = torch.ops.mptorch.narrow_float64.default
    x = torch.randn(4099, dtype=F64, device=device) * 1e3
    # A strided input, and a contiguous view 8 bytes into its storage, narrow as
    # their copies do. The CUDA kernel's vector load is 16-byte aligned and
    # faults on such a view read in place, so the entry point copies it.
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


@pytest.mark.parametrize("device", float64_devices)
def test_float32_layer_in_binary64_is_the_float64_layer_narrowed(device):
    """A float32 ``QLinear`` naming binary64 is the float64 layer, narrowed.

    Forward and both gradients, on the widened tensors. The formats (22 and 23
    bits) are ones a float32 result stores, so the storage check is silent.
    There is no bias, because it is added after the GEMM in each layer's own
    dtype and would differ."""
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
    # The default binary32 carrier gives a different result: there the product
    # of two 24-bit operands is rounded to 24 bits before the 22-bit format
    # rounds it, a double rounding the binary64 carrier does not have.
    plain = QLinear(24, 6, bias=False, formats=binaryK_gemm_formats(**fmt), device=device)
    with torch.no_grad():
        plain.weight.copy_(narrow.weight)
        assert not torch.equal(plain(x), out)
