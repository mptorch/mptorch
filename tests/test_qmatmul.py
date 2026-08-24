import numpy as np
import pytest
import torch

from mptorch.quant import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_quantize,
    superfp_matmul,
    superfp_matmul_fma,
    superfp_quantize,
)
from tests.markers import available_devices


def _ref(a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool) -> torch.Tensor:
    op_a = a.t() if trans_a else a
    op_b = b.t() if trans_b else b
    return op_a @ op_b


def _fma32(a: float, b: float, c: float) -> float:
    """
    Single-rounding float32 fused multiply-add, computed via 80-bit
    ``np.longdouble`` (64-bit mantissa -- ample headroom over float32's 24
    bits) so the a*b+c intermediate needs no rounding of its own before the
    one rounding down to float32, matching real hardware FMA semantics.
    """
    la, lb, lc = np.longdouble(a), np.longdouble(b), np.longdouble(c)
    return float(np.float32(la * lb + lc))


TRANS_COMBOS = [(False, False), (False, True), (True, False)]


# ------------------------------------------------------------------------------------
# Tier 1: near-identity format vs. a plain matmul reference.
#
# The kernel's own accumulation order (strictly sequential over K) differs
# from cuBLAS/ATen's internal reduction order, so even a lossless format
# doesn't give bit-exact agreement with `a @ b` -- only a tight numerical
# tolerance. Bit-exact agreement is instead checked in Tier 3 below, against
# a manual reference that reproduces the kernel's own summation order.


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("trans_a,trans_b", TRANS_COMBOS)
@pytest.mark.parametrize("M,K,N", [(5, 7, 9), (65, 9, 63), (1, 5, 1)])
def test_binaryK_matmul_tier1_near_identity(device, trans_a, trans_b, M, K, N):
    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn(*a_shape, device=device)
    b = torch.randn(*b_shape, device=device)

    out = binaryK_matmul(a, b, trans_a=trans_a, trans_b=trans_b, mul_K=33, mul_P=24)
    ref = _ref(a, b, trans_a, trans_b)

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("trans_a,trans_b", TRANS_COMBOS)
def test_superfp_matmul_tier1_near_identity(device, trans_a, trans_b):
    M, K, N = 6, 11, 8
    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn(*a_shape, device=device)
    b = torch.randn(*b_shape, device=device)

    out = superfp_matmul(
        a,
        b,
        trans_a=trans_a,
        trans_b=trans_b,
        mul_man_bits=23,
        mul_exp_bits=8,
        mul_normal_binades=255,
        mul_bias=127,
    )
    ref = _ref(a, b, trans_a, trans_b)

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------------------------------
# Tier 2: statistical -- real low-bit formats stay close to the fp32 reference.


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_tier2_statistical(device):
    M, K, N = 16, 64, 12
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    out = binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4, accumulate_quant=True)
    ref = a @ b.t()

    assert torch.isfinite(out).all()
    cos_sim = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    assert cos_sim > 0.9


@pytest.mark.parametrize("device", available_devices)
def test_superfp_matmul_tier2_statistical(device):
    M, K, N = 16, 64, 12
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    out = superfp_matmul(
        a,
        b,
        trans_b=True,
        mul_man_bits=3,
        mul_exp_bits=4,
        mul_normal_binades=8,
        mul_bias=7,
        accumulate_quant=True,
    )
    ref = a @ b.t()

    assert torch.isfinite(out).all()
    cos_sim = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    assert cos_sim > 0.9


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_accumulate_quant_false_uses_full_precision_sum(device):
    # accumulate_quant=False should only quantize the multiply, leaving the
    # running sum in full precision -- distinct from accumulate_quant=True
    # with the same (low-precision) multiply format.
    M, K, N = 8, 32, 6
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    out_full_acc = binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4, accumulate_quant=False)
    out_quant_acc = binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4, accumulate_quant=True)

    assert torch.isfinite(out_full_acc).all()
    assert not torch.equal(out_full_acc, out_quant_acc)


# ------------------------------------------------------------------------------------
# Tier 3: manual baseline -- reproduce the kernel's own (sequential-over-K)
# summation order in Python, calling the same quantizer once per scalar FMA.
# This is what actually proves the fused kernel's add/mul dispatch is wired
# correctly, not just "close enough".


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_tier3_manual_baseline(device):
    M, K, N = 3, 5, 4
    K_fmt, P_fmt = 8, 4
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    def q(x: torch.Tensor) -> torch.Tensor:
        return binaryK_quantize(x, K=K_fmt, P=P_fmt)

    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            acc = torch.zeros((), device=device)
            for k in range(K):
                term = q(a[i, k] * b[j, k])
                acc = q(acc + term)
            ref[i, j] = acc

    out = binaryK_matmul(a, b, trans_b=True, mul_K=K_fmt, mul_P=P_fmt, accumulate_quant=True)

    assert torch.equal(out, ref)


@pytest.mark.parametrize("device", available_devices)
def test_superfp_matmul_tier3_manual_baseline(device):
    M, K, N = 3, 5, 4
    man_bits, exp_bits, normal_binades, bias = 2, 3, 4, 7
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    def q(x: torch.Tensor) -> torch.Tensor:
        return superfp_quantize(
            x, man_bits=man_bits, exp_bits=exp_bits, normal_binades=normal_binades, bias=bias
        )

    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            acc = torch.zeros((), device=device)
            for k in range(K):
                term = q(a[i, k] * b[j, k])
                acc = q(acc + term)
            ref[i, j] = acc

    out = superfp_matmul(
        a,
        b,
        trans_b=True,
        mul_man_bits=man_bits,
        mul_exp_bits=exp_bits,
        mul_normal_binades=normal_binades,
        mul_bias=bias,
        accumulate_quant=True,
    )

    assert torch.equal(out, ref)


def test_binaryK_matmul_rejects_non_2d():
    a = torch.randn(2, 3, 4)
    b = torch.randn(4, 5)
    with pytest.raises(RuntimeError):
        binaryK_matmul(a, b, mul_K=8, mul_P=4)


# ------------------------------------------------------------------------------------
# FMA (fused multiply-add) variant: each dot-product step is a single
# hardware-style fused multiply-add rounded once, instead of a quantized
# multiply followed by a separately quantized add. Same Tier 1/2/3
# structure as above, plus a case that pins down the single-rounding
# (Fused) vs. double-rounding (Split) distinction directly.


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("trans_a,trans_b", TRANS_COMBOS)
@pytest.mark.parametrize("M,K,N", [(5, 7, 9), (65, 9, 63), (1, 5, 1)])
def test_binaryK_matmul_fma_tier1_near_identity(device, trans_a, trans_b, M, K, N):
    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn(*a_shape, device=device)
    b = torch.randn(*b_shape, device=device)

    out = binaryK_matmul_fma(a, b, trans_a=trans_a, trans_b=trans_b, fma_K=33, fma_P=24)
    ref = _ref(a, b, trans_a, trans_b)

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("trans_a,trans_b", TRANS_COMBOS)
def test_superfp_matmul_fma_tier1_near_identity(device, trans_a, trans_b):
    M, K, N = 6, 11, 8
    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn(*a_shape, device=device)
    b = torch.randn(*b_shape, device=device)

    out = superfp_matmul_fma(
        a,
        b,
        trans_a=trans_a,
        trans_b=trans_b,
        fma_man_bits=23,
        fma_exp_bits=8,
        fma_normal_binades=255,
        fma_bias=127,
    )
    ref = _ref(a, b, trans_a, trans_b)

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_fma_tier2_statistical(device):
    M, K, N = 16, 64, 12
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    out = binaryK_matmul_fma(a, b, trans_b=True, fma_K=8, fma_P=4)
    ref = a @ b.t()

    assert torch.isfinite(out).all()
    cos_sim = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    assert cos_sim > 0.9


@pytest.mark.parametrize("device", available_devices)
def test_superfp_matmul_fma_tier2_statistical(device):
    M, K, N = 16, 64, 12
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    out = superfp_matmul_fma(
        a,
        b,
        trans_b=True,
        fma_man_bits=3,
        fma_exp_bits=4,
        fma_normal_binades=8,
        fma_bias=7,
    )
    ref = a @ b.t()

    assert torch.isfinite(out).all()
    cos_sim = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    assert cos_sim > 0.9


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_fma_quant_false_matches_plain_matmul(device):
    # fma_quant=False runs the fused step in full fp32 precision (a real
    # FMA, no rounding beyond fp32 itself) -- close to a plain matmul, and
    # distinct from fma_quant=True at the same (low-precision) format.
    M, K, N = 8, 32, 6
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    out_full = binaryK_matmul_fma(a, b, trans_b=True, fma_K=8, fma_P=4, fma_quant=False)
    out_quant = binaryK_matmul_fma(a, b, trans_b=True, fma_K=8, fma_P=4, fma_quant=True)
    ref = a @ b.t()

    assert torch.allclose(out_full, ref, atol=1e-4, rtol=1e-4)
    assert not torch.equal(out_full, out_quant)


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_fma_tier3_manual_baseline(device):
    M, K, N = 3, 5, 4
    K_fmt, P_fmt = 8, 4
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    def q(x: float) -> float:
        return binaryK_quantize(
            torch.tensor(x, dtype=torch.float32, device=device), K=K_fmt, P=P_fmt
        ).item()

    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc = q(_fma32(a[i, k].item(), b[j, k].item(), acc))
            ref[i, j] = acc

    out = binaryK_matmul_fma(a, b, trans_b=True, fma_K=K_fmt, fma_P=P_fmt)

    assert torch.equal(out, ref)


@pytest.mark.parametrize("device", available_devices)
def test_superfp_matmul_fma_tier3_manual_baseline(device):
    M, K, N = 3, 5, 4
    man_bits, exp_bits, normal_binades, bias = 2, 3, 4, 7
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    def q(x: float) -> float:
        return superfp_quantize(
            torch.tensor(x, dtype=torch.float32, device=device),
            man_bits=man_bits,
            exp_bits=exp_bits,
            normal_binades=normal_binades,
            bias=bias,
        ).item()

    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc = q(_fma32(a[i, k].item(), b[j, k].item(), acc))
            ref[i, j] = acc

    out = superfp_matmul_fma(
        a,
        b,
        trans_b=True,
        fma_man_bits=man_bits,
        fma_exp_bits=exp_bits,
        fma_normal_binades=normal_binades,
        fma_bias=bias,
    )

    assert torch.equal(out, ref)


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_split_vs_fused_diverge(device):
    # Split (quantized multiply, then quantized add -- two roundings) and
    # Fused (single hardware-style FMA rounding) are different arithmetic,
    # not just different entry points to the same computation: at a
    # low-precision accumulate format they should disagree in general.
    M, K, N = 16, 48, 10
    K_fmt, P_fmt = 8, 4
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)

    out_split = binaryK_matmul(a, b, trans_b=True, mul_K=K_fmt, mul_P=P_fmt, accumulate_quant=True)
    out_fused = binaryK_matmul_fma(a, b, trans_b=True, fma_K=K_fmt, fma_P=P_fmt)

    assert torch.isfinite(out_split).all()
    assert torch.isfinite(out_fused).all()
    assert not torch.equal(out_split, out_fused)


def test_binaryK_matmul_fma_rejects_non_2d():
    a = torch.randn(2, 3, 4)
    b = torch.randn(4, 5)
    with pytest.raises(RuntimeError):
        binaryK_matmul_fma(a, b, fma_K=8, fma_P=4)
