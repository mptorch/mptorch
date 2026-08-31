from typing import Any

import numpy as np
import pytest
import torch

from mptorch.number import RoundMode
from mptorch.quant import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_quantize,
    superfp_matmul,
    superfp_matmul_fma,
    superfp_quantize,
)
from mptorch.quant.ops import mantissa_size_mapping
from tests.markers import available_devices

DETERMINISTIC_ROUND_MODES = [
    RoundMode.RNE,
    RoundMode.RNA,
    RoundMode.RU,
    RoundMode.RD,
    RoundMode.RZ,
    RoundMode.RO,
]

# Storage dtypes for the operand matrices. The GEMM kernel (gemm_policy.h)
# always upcasts each element to float32 before multiplying/accumulating and
# only narrows back to the storage dtype once, on the final output element --
# regardless of whether that storage dtype is float32, float16, or bfloat16.
# Tier 2/3 tests below are parametrized over this list; Tier 1 stays
# float32-only (it already covers many shape/transpose combinations and
# exists to validate wiring, not per-dtype numerics).
MATMUL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# allclose tolerances scale with the storage dtype's own precision -- a
# bfloat16/float16 tensor's *inputs* already carry far more rounding error
# than float32's, so comparisons against a plain (unquantized) reference need
# proportionally looser tolerances, independent of anything the custom GEMM
# kernel itself does.
ALLCLOSE_TOL: dict[torch.dtype, dict[str, Any]] = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=8e-2, rtol=8e-2),
}


def _ref(a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool) -> torch.Tensor:
    op_a = a.t() if trans_a else a
    op_b = b.t() if trans_b else b
    return op_a @ op_b


def _equal_nan_ok(a: torch.Tensor, b: torch.Tensor) -> bool:
    """
    Elementwise equality, treating NaN == NaN as a match (unlike
    ``torch.equal``). A narrow-range format (e.g. superfp with few exponent
    bits) can legitimately overflow to +-inf and then produce NaN (e.g.
    inf + -inf) during accumulation -- the same NaN in both the kernel's
    output and this file's manual baseline is a match, not a divergence.
    """
    return bool(((a == b) | (a.isnan() & b.isnan())).all())


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
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_tier2_statistical(device, dtype):
    M, K, N = 16, 64, 12
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    out = binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4, accumulate_quant=True)
    ref = a @ b.t()

    assert torch.isfinite(out).all()
    cos_sim = torch.nn.functional.cosine_similarity(
        out.flatten().float(), ref.flatten().float(), dim=0
    ).item()
    assert cos_sim > 0.9


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_superfp_matmul_tier2_statistical(device, dtype):
    M, K, N = 16, 64, 12
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

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
    cos_sim = torch.nn.functional.cosine_similarity(
        out.flatten().float(), ref.flatten().float(), dim=0
    ).item()
    assert cos_sim > 0.9


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_accumulate_quant_false_uses_full_precision_sum(device, dtype):
    # accumulate_quant=False should only quantize the multiply, leaving the
    # running sum in full precision -- distinct from accumulate_quant=True
    # with the same (low-precision) multiply format.
    M, K, N = 8, 32, 6
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

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
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_tier3_manual_baseline(device, dtype):
    M, K, N = 3, 5, 4
    K_fmt, P_fmt = 8, 4
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    def q(x: torch.Tensor) -> torch.Tensor:
        return binaryK_quantize(x, K=K_fmt, P=P_fmt)

    # The kernel upcasts each operand to float32 before multiplying and only
    # narrows the running sum back to the storage dtype once, at the very
    # end -- so the reference must widen a[i,k]/b[j,k] to float32 before the
    # multiply too, else a lower-precision storage dtype would round the
    # product an extra time before quantizing it (double rounding the kernel
    # itself never does).
    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            acc = torch.zeros((), device=device)
            for k in range(K):
                term = q(a[i, k].float() * b[j, k].float())
                acc = q(acc + term)
            ref[i, j] = acc

    out = binaryK_matmul(a, b, trans_b=True, mul_K=K_fmt, mul_P=P_fmt, accumulate_quant=True)

    assert _equal_nan_ok(out, ref.to(dtype))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_superfp_matmul_tier3_manual_baseline(device, dtype):
    M, K, N = 3, 5, 4
    man_bits, exp_bits, normal_binades, bias = 2, 3, 4, 7
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    def q(x: torch.Tensor) -> torch.Tensor:
        return superfp_quantize(
            x, man_bits=man_bits, exp_bits=exp_bits, normal_binades=normal_binades, bias=bias
        )

    # See test_binaryK_matmul_tier3_manual_baseline: widen to float32 before
    # the multiply to match the kernel's own upcast-before-multiply order.
    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            acc = torch.zeros((), device=device)
            for k in range(K):
                term = q(a[i, k].float() * b[j, k].float())
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

    assert _equal_nan_ok(out, ref.to(dtype))


# ------------------------------------------------------------------------------------
# Rounding mode selection: every deterministic RoundMode (RNE/RNA/RU/RD/RZ/RO)
# is wired through to the GEMM kernel's Multiplier/Adder, each with its own
# precomputed-parameter fast path (see gemm_policy.h / dev/gemm_core_roadmap.md
# Roadmap item 6). Reuses the Tier 3 manual-baseline pattern above, since bit-
# exact agreement is the whole point -- a near-identity/statistical tolerance
# wouldn't catch a mode being silently ignored or mismatched between the
# kernel's Multiplier and Adder.


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_binaryK_matmul_tier3_manual_baseline_round_modes(device, dtype, round_mode):
    M, K, N = 3, 5, 4
    K_fmt, P_fmt = 8, 4
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    def q(x: torch.Tensor) -> torch.Tensor:
        return binaryK_quantize(x, K=K_fmt, P=P_fmt, rounding_mode=round_mode)

    # See test_binaryK_matmul_tier3_manual_baseline: widen to float32 before
    # the multiply to match the kernel's own upcast-before-multiply order.
    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            acc = torch.zeros((), device=device)
            for k in range(K):
                term = q(a[i, k].float() * b[j, k].float())
                acc = q(acc + term)
            ref[i, j] = acc

    out = binaryK_matmul(
        a,
        b,
        trans_b=True,
        mul_K=K_fmt,
        mul_P=P_fmt,
        accumulate_quant=True,
        rounding_mode=round_mode,
    )

    assert _equal_nan_ok(out, ref.to(dtype))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_superfp_matmul_tier3_manual_baseline_round_modes(device, dtype, round_mode):
    M, K, N = 3, 5, 4
    man_bits, exp_bits, normal_binades, bias = 2, 3, 4, 7
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    def q(x: torch.Tensor) -> torch.Tensor:
        return superfp_quantize(
            x,
            man_bits=man_bits,
            exp_bits=exp_bits,
            normal_binades=normal_binades,
            bias=bias,
            rounding_mode=round_mode,
        )

    # See test_binaryK_matmul_tier3_manual_baseline: widen to float32 before
    # the multiply to match the kernel's own upcast-before-multiply order.
    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            acc = torch.zeros((), device=device)
            for k in range(K):
                term = q(a[i, k].float() * b[j, k].float())
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
        rounding_mode=round_mode,
    )

    assert _equal_nan_ok(out, ref.to(dtype))


# ------------------------------------------------------------------------------------
# Stochastic rounding (RoundMode.SR): each output element seeds its own
# independent random stream (gemm_policy.h's NaiveAccumulator::seed_rng),
# so a single K=1, many-output-elements call gives many independent SR
# draws at once -- mirroring test_binaryK_stochastic's bounding-guarantee
# (SR must land on either the RD or RU neighbor) and statistical-
# unbiasedness (mean over many draws of a fixed value converges to it)
# properties from tests/test_binaryk_quantize.py, applied per-output-
# element instead of per-input-element. accumulate_quant=False isolates
# the multiplier's SR rounding (a single K=1 term needs no accumulate
# step to interpret).


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_stochastic_bounds_and_unbiased(device, dtype):
    K_fmt, P_fmt = 8, 4
    # prng_bits is capped at the storage dtype's own mantissa width minus the
    # target format's mantissa bits (see binaryK_matmul's mantissa_size_mapping
    # assertion) -- fp16/bf16 allow fewer random mantissa bits than fp32.
    prng_bits = mantissa_size_mapping[dtype] - (P_fmt - 1)

    a = torch.rand(500, 1, device=device, dtype=dtype) * 1.8 - 0.9
    b = torch.ones(1, 1, device=device, dtype=dtype)

    q_rd = binaryK_matmul(
        a, b, mul_K=K_fmt, mul_P=P_fmt, accumulate_quant=False, rounding_mode=RoundMode.RD
    )
    q_ru = binaryK_matmul(
        a, b, mul_K=K_fmt, mul_P=P_fmt, accumulate_quant=False, rounding_mode=RoundMode.RU
    )
    q_sr = binaryK_matmul(
        a,
        b,
        mul_K=K_fmt,
        mul_P=P_fmt,
        accumulate_quant=False,
        rounding_mode=RoundMode.SR,
        mul_prng_bits=prng_bits,
    )
    assert torch.all((q_sr == q_rd) | (q_sr == q_ru))

    test_val = 0.333333
    a_const = torch.full((5000, 1), test_val, device=device, dtype=dtype)
    q_sr_const = binaryK_matmul(
        a_const,
        b,
        mul_K=K_fmt,
        mul_P=P_fmt,
        accumulate_quant=False,
        rounding_mode=RoundMode.SR,
        mul_prng_bits=prng_bits,
    )
    assert abs(q_sr_const.float().mean().item() - test_val) < 0.01


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_superfp_matmul_stochastic_bounds_and_unbiased(device, dtype):
    man_bits, exp_bits, normal_binades, bias = 3, 4, 8, 7
    prng_bits = mantissa_size_mapping[dtype] - man_bits

    a = torch.rand(500, 1, device=device, dtype=dtype) * 1.8 - 0.9
    b = torch.ones(1, 1, device=device, dtype=dtype)

    kwargs: dict[str, Any] = dict(
        mul_man_bits=man_bits,
        mul_exp_bits=exp_bits,
        mul_normal_binades=normal_binades,
        mul_bias=bias,
        accumulate_quant=False,
    )
    q_rd = superfp_matmul(a, b, rounding_mode=RoundMode.RD, **kwargs)
    q_ru = superfp_matmul(a, b, rounding_mode=RoundMode.RU, **kwargs)
    q_sr = superfp_matmul(a, b, rounding_mode=RoundMode.SR, mul_prng_bits=prng_bits, **kwargs)
    assert torch.all((q_sr == q_rd) | (q_sr == q_ru))

    test_val = 0.333333
    a_const = torch.full((5000, 1), test_val, device=device, dtype=dtype)
    q_sr_const = superfp_matmul(
        a_const, b, rounding_mode=RoundMode.SR, mul_prng_bits=prng_bits, **kwargs
    )
    assert abs(q_sr_const.float().mean().item() - test_val) < 0.01


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
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_fma_tier2_statistical(device, dtype):
    M, K, N = 16, 64, 12
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    out = binaryK_matmul_fma(a, b, trans_b=True, fma_K=8, fma_P=4)
    ref = a @ b.t()

    assert torch.isfinite(out).all()
    cos_sim = torch.nn.functional.cosine_similarity(
        out.flatten().float(), ref.flatten().float(), dim=0
    ).item()
    assert cos_sim > 0.9


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_superfp_matmul_fma_tier2_statistical(device, dtype):
    M, K, N = 16, 64, 12
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

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
    cos_sim = torch.nn.functional.cosine_similarity(
        out.flatten().float(), ref.flatten().float(), dim=0
    ).item()
    assert cos_sim > 0.9


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_fma_quant_false_matches_plain_matmul(device, dtype):
    # fma_quant=False runs the fused step in full fp32 precision (a real
    # FMA, no rounding beyond fp32 itself) -- close to a plain matmul, and
    # distinct from fma_quant=True at the same (low-precision) format.
    M, K, N = 8, 32, 6
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    out_full = binaryK_matmul_fma(a, b, trans_b=True, fma_K=8, fma_P=4, fma_quant=False)
    out_quant = binaryK_matmul_fma(a, b, trans_b=True, fma_K=8, fma_P=4, fma_quant=True)
    ref = a @ b.t()

    assert torch.allclose(out_full, ref, **ALLCLOSE_TOL[dtype])
    assert not torch.equal(out_full, out_quant)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_fma_tier3_manual_baseline(device, dtype):
    M, K, N = 3, 5, 4
    K_fmt, P_fmt = 8, 4
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    def q(x: float) -> float:
        return binaryK_quantize(
            torch.tensor(x, dtype=torch.float32, device=device), K=K_fmt, P=P_fmt
        ).item()

    # a[i, k].item()/b[j, k].item() already read back the exact (widened)
    # value regardless of storage dtype, and _fma32 itself computes in
    # ``np.longdouble`` before its single float32 rounding -- so this
    # reference already matches the kernel's upcast-before-compute order
    # without further changes; only the final cast to the storage dtype
    # needs to happen explicitly here (the kernel does it once, on write-out).
    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc = q(_fma32(a[i, k].item(), b[j, k].item(), acc))
            ref[i, j] = acc

    out = binaryK_matmul_fma(a, b, trans_b=True, fma_K=K_fmt, fma_P=P_fmt)

    assert _equal_nan_ok(out, ref.to(dtype))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_superfp_matmul_fma_tier3_manual_baseline(device, dtype):
    M, K, N = 3, 5, 4
    man_bits, exp_bits, normal_binades, bias = 2, 3, 4, 7
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

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

    assert _equal_nan_ok(out, ref.to(dtype))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_binaryK_matmul_fma_tier3_manual_baseline_round_modes(device, dtype, round_mode):
    M, K, N = 3, 5, 4
    K_fmt, P_fmt = 8, 4
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    def q(x: float) -> float:
        return binaryK_quantize(
            torch.tensor(x, dtype=torch.float32, device=device),
            K=K_fmt,
            P=P_fmt,
            rounding_mode=round_mode,
        ).item()

    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc = q(_fma32(a[i, k].item(), b[j, k].item(), acc))
            ref[i, j] = acc

    out = binaryK_matmul_fma(a, b, trans_b=True, fma_K=K_fmt, fma_P=P_fmt, rounding_mode=round_mode)

    assert _equal_nan_ok(out, ref.to(dtype))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_superfp_matmul_fma_tier3_manual_baseline_round_modes(device, dtype, round_mode):
    M, K, N = 3, 5, 4
    man_bits, exp_bits, normal_binades, bias = 2, 3, 4, 7
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    def q(x: float) -> float:
        return superfp_quantize(
            torch.tensor(x, dtype=torch.float32, device=device),
            man_bits=man_bits,
            exp_bits=exp_bits,
            normal_binades=normal_binades,
            bias=bias,
            rounding_mode=round_mode,
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
        rounding_mode=round_mode,
    )

    assert _equal_nan_ok(out, ref.to(dtype))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_fma_stochastic_bounds_and_unbiased(device, dtype):
    # FMA analog of test_binaryK_matmul_stochastic_bounds_and_unbiased --
    # with K=1, FusedMac's fma_f32(a, b, 0) reduces to a plain a*b, so the
    # same RD/RU-bounding and unbiasedness properties apply to the single
    # Adder's SR rounding (fma_quant=True is the default -- there's no
    # separate multiply format to leave unquantized here, unlike the Split
    # ops' accumulate_quant=False).
    K_fmt, P_fmt = 8, 4
    prng_bits = mantissa_size_mapping[dtype] - (P_fmt - 1)

    a = torch.rand(500, 1, device=device, dtype=dtype) * 1.8 - 0.9
    b = torch.ones(1, 1, device=device, dtype=dtype)

    q_rd = binaryK_matmul_fma(a, b, fma_K=K_fmt, fma_P=P_fmt, rounding_mode=RoundMode.RD)
    q_ru = binaryK_matmul_fma(a, b, fma_K=K_fmt, fma_P=P_fmt, rounding_mode=RoundMode.RU)
    q_sr = binaryK_matmul_fma(
        a, b, fma_K=K_fmt, fma_P=P_fmt, rounding_mode=RoundMode.SR, fma_prng_bits=prng_bits
    )
    assert torch.all((q_sr == q_rd) | (q_sr == q_ru))

    test_val = 0.333333
    a_const = torch.full((5000, 1), test_val, device=device, dtype=dtype)
    q_sr_const = binaryK_matmul_fma(
        a_const, b, fma_K=K_fmt, fma_P=P_fmt, rounding_mode=RoundMode.SR, fma_prng_bits=prng_bits
    )
    assert abs(q_sr_const.float().mean().item() - test_val) < 0.01


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_superfp_matmul_fma_stochastic_bounds_and_unbiased(device, dtype):
    man_bits, exp_bits, normal_binades, bias = 3, 4, 8, 7
    prng_bits = mantissa_size_mapping[dtype] - man_bits

    a = torch.rand(500, 1, device=device, dtype=dtype) * 1.8 - 0.9
    b = torch.ones(1, 1, device=device, dtype=dtype)

    kwargs: dict[str, Any] = dict(
        fma_man_bits=man_bits,
        fma_exp_bits=exp_bits,
        fma_normal_binades=normal_binades,
        fma_bias=bias,
    )
    q_rd = superfp_matmul_fma(a, b, rounding_mode=RoundMode.RD, **kwargs)
    q_ru = superfp_matmul_fma(a, b, rounding_mode=RoundMode.RU, **kwargs)
    q_sr = superfp_matmul_fma(a, b, rounding_mode=RoundMode.SR, fma_prng_bits=prng_bits, **kwargs)
    assert torch.all((q_sr == q_rd) | (q_sr == q_ru))

    test_val = 0.333333
    a_const = torch.full((5000, 1), test_val, device=device, dtype=dtype)
    q_sr_const = superfp_matmul_fma(
        a_const, b, rounding_mode=RoundMode.SR, fma_prng_bits=prng_bits, **kwargs
    )
    assert abs(q_sr_const.float().mean().item() - test_val) < 0.01


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_split_vs_fused_diverge(device, dtype):
    # Split (quantized multiply, then quantized add -- two roundings) and
    # Fused (single hardware-style FMA rounding) are different arithmetic,
    # not just different entry points to the same computation: at a
    # low-precision accumulate format they should disagree in general.
    M, K, N = 16, 48, 10
    K_fmt, P_fmt = 8, 4
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

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
