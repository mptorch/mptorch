"""
The eight flat GEMM wrappers (``binaryK_matmul``, ``superfp_matmul``, their
``_fma`` variants and the ``_mixed`` palette variants of each) against
references built outside the kernels.

Three tiers per op. Tier 1 runs a near-identity format (24 significand bits,
binary32's range) against ``torch.matmul`` within a tight tolerance, over
several shapes and every transpose combination, to validate the wiring. Tier
2 runs a real low-bit format and asks only for high cosine similarity with
the unquantized result. Tier 3 is the exact check: it reproduces the kernel's
own arithmetic in Python, one sequential pass over K per output element with
the elementwise quantizer applied where the kernel rounds, and demands
bit-for-bit agreement. That reference is trustworthy because the elementwise
quantizers are checked exhaustively over every binary32 input against
independent reference casts (``dev/benchmarks/gemm_cast_*_arith.cu``), so a
tier 3 mismatch points at the GEMM's rounding placement, its summation order
or its operand handling, not at the cast.

On top of the tiers: every deterministic rounding mode through the tier 3
reference, the two properties of stochastic rounding (bounded by the RD and
RU neighbours, unbiased in the mean), the host-side rejections (rank, operand
dtype, palette shape and length), and, for the mixed ops, equivalence of a
one-slot palette with the single-format op and of a broadcast ``prec_idx``
with the dense map it expands to.
"""

from typing import Any

import numpy as np
import pytest
import torch

from mptorch.number import RoundMode
from mptorch.quant import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_fma_mixed,
    binaryK_matmul_mixed,
    binaryK_quantize,
    superfp_matmul,
    superfp_matmul_fma,
    superfp_matmul_fma_mixed,
    superfp_matmul_mixed,
    superfp_quantize,
)
from tests.markers import available_devices

DETERMINISTIC_ROUND_MODES = [
    RoundMode.RNE,
    RoundMode.RNA,
    RoundMode.RU,
    RoundMode.RD,
    RoundMode.RZ,
    RoundMode.RO,
]

# Operand dtypes the binary32 kernels accept. Each element is converted to the
# binary32 carrier on load, every product and sum is computed and rounded
# there, and the result is narrowed to the storage dtype once, on the write,
# whether that dtype is float32, float16 or bfloat16. Tier 2 and 3 tests are
# parametrized over this list; tier 1 stays float32, since it exists to
# validate the wiring across shapes and transposes, not per-dtype numerics.
MATMUL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# Stochastic rounding draws its random bits in the binary32 carrier, whatever
# the operand dtype, so the widest draw a format allows is binary32's 23
# mantissa bits less the format's own.
F32_MAN_BITS = 23

# allclose tolerances against a plain unquantized reference, per storage
# dtype. A float16 or bfloat16 operand already carries far more rounding error
# than a float32 one, independent of anything the GEMM kernel does, so the
# tolerance scales with the dtype's unit roundoff (2^-24, 2^-11, 2^-8) times
# the few tens of terms of the K=32 reduction that uses it, rounded up.
ALLCLOSE_TOL: dict[torch.dtype, dict[str, Any]] = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=8e-2, rtol=8e-2),
}


def _ref(a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool) -> torch.Tensor:
    """Plain ``torch.matmul`` reference honouring the two transpose flags."""
    op_a = a.t() if trans_a else a
    op_b = b.t() if trans_b else b
    return op_a @ op_b


def _equal_nan_ok(a: torch.Tensor, b: torch.Tensor) -> bool:
    """
    Elementwise equality that counts NaN == NaN as a match, unlike
    ``torch.equal``. A narrow-range format (a superfp with few exponent bits,
    say) can legitimately overflow to an infinity and then produce NaN from
    ``inf + -inf`` during accumulation; the same NaN in the kernel's output
    and in this file's manual baseline is agreement, not divergence.
    """
    return bool(((a == b) | (a.isnan() & b.isnan())).all())


def _fma32(a: float, b: float, c: float) -> float:
    """
    Single-rounding float32 fused multiply-add. A hardware FMA forms a*b+c
    exactly and rounds once; here the product of two 24-bit significands (48
    bits, exact) and the addend go through x87's 80-bit ``np.longdouble``,
    whose 64-bit significand leaves the intermediate at most one rounding far
    below float32's 24 bits, before the one rounding to float32 that counts.
    """
    la, lb, lc = np.longdouble(a), np.longdouble(b), np.longdouble(c)
    return float(np.float32(la * lb + lc))


TRANS_COMBOS = [(False, False), (False, True), (True, False)]


# ------------------------------------------------------------------------------------
# Tier 1: near-identity format against a plain matmul reference.
#
# The kernel sums strictly sequentially over K, which is not the reduction
# order cuBLAS or ATen use, so even a lossless format does not agree with
# `a @ b` bit for bit, only within a tight tolerance. Bit-exact agreement is
# checked in tier 3, against a manual reference that reproduces the kernel's
# own summation order.


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
# Tier 2: statistical. Real low-bit formats stay close to the fp32 reference.


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
    """
    ``accumulate_quant=False`` quantizes only the multiply and keeps the
    running sum in the carrier, so it must differ from ``accumulate_quant=True``
    at the same low-precision multiply format; equal outputs would mean the
    flag is ignored.
    """
    M, K, N = 8, 32, 6
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    out_full_acc = binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4, accumulate_quant=False)
    out_quant_acc = binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4, accumulate_quant=True)

    assert torch.isfinite(out_full_acc).all()
    assert not torch.equal(out_full_acc, out_quant_acc)


# ------------------------------------------------------------------------------------
# Tier 3: manual baseline. Reproduce the kernel's own sequential-over-K
# summation in Python, calling the elementwise quantizer once per scalar
# multiply and once per add, and demand bit-exact agreement. This is what
# proves the kernel's multiply and add casts are the intended ones, rather
# than merely close.


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_tier3_manual_baseline(device, dtype):
    M, K, N = 3, 5, 4
    K_fmt, P_fmt = 8, 4
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    def q(x: torch.Tensor) -> torch.Tensor:
        return binaryK_quantize(x, K=K_fmt, P=P_fmt)

    # The kernel converts each operand to binary32 on load and narrows the
    # running sum to the storage dtype once, on the write, so the reference
    # must widen a[i, k] and b[j, k] to float32 before the multiply too.
    # Otherwise a float16 or bfloat16 product would be rounded to the storage
    # dtype before the quantizer saw it, a second rounding the kernel never
    # performs.
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
    # the multiply to match the kernel's own convert-on-load order.
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
# Rounding mode selection: every deterministic RoundMode (RNE, RNA, RU, RD,
# RZ, RO) is a template parameter of the kernel's Multiplier and Adder, each
# of which builds the format's constants once per launch and rounds with them
# in the hot loop (the precomputed-parameter fast path, gemm_policy.h). These
# reuse the tier 3 manual-baseline pattern because bit-exact agreement is the
# point: a near-identity or statistical tolerance would not catch a mode that
# is silently ignored, or one the Multiplier and the Adder disagree on.


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
    # the multiply to match the kernel's own convert-on-load order.
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
    # the multiply to match the kernel's own convert-on-load order.
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
# Philox stream on its global linear index into the output (gemm_policy.h's
# seed_rng), so one K=1 call with many output elements yields that many
# independent draws at once. The two properties are the ones
# tests/test_binaryk_quantize.py checks per input element, applied per output
# element: SR lands on either the RD or the RU neighbour, and the mean of
# many draws of a fixed value converges to that value. accumulate_quant=False
# isolates the multiplier's rounding, since with K=1 there is a single
# product and no accumulate step to interpret.


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_stochastic_bounds_and_unbiased(device, dtype):
    """
    The multiplier's SR result is one of the two directed neighbours, and
    its mean over many draws of a fixed value is that value, for every
    storage dtype.
    """
    K_fmt, P_fmt = 8, 4
    # prng_bits is capped at binary32's 23 mantissa bits minus the format's
    # own: the random bits are drawn in the binary32 carrier, so a float16 or
    # bfloat16 operand allows exactly as many as a float32 one. A result
    # narrower than its carrier is rounded once more when it is stored, but
    # that second rounding is a bound on the format the result holds (the
    # storage check in ops.py), not on the draw.
    prng_bits = F32_MAN_BITS - (P_fmt - 1)

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

    # 0.333333 lies strictly between two P=4 values (0.3125 and 0.34375, a
    # 2^-5 step), so every draw rounds. The 0.01 margin is well above both
    # the sampling noise of 5000 draws over that step (about 2e-4) and the
    # error bfloat16 puts into the constant itself (about 7e-4).
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
    """Superfp twin of the binaryK SR test above (a 3-bit mantissa, same step)."""
    man_bits, exp_bits, normal_binades, bias = 3, 4, 8, 7
    prng_bits = F32_MAN_BITS - man_bits

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


def test_raw_matmul_op_rejects_rank_4():
    """
    The op boundary is rank 2 or 3, strictly. 1D promotion and rank>3
    broadcasting are Python's job (ops.py's ``_matmul_operands``, exercised
    by test_qmatmul_batched.py), so a rank-4 operand reaching the op is a bug
    in that layer and must be reported, not reshaped.
    """
    # Positional arguments follow the op schema in csrc/quant_ops.cpp.
    a = torch.randn(2, 3, 4, 5)
    b = torch.randn(5, 6)
    with pytest.raises(RuntimeError, match="expects 2D or 3D tensors"):
        torch.ops.mptorch.custom_matmul_binaryK.default(
            a, b, False, False, 8, 4, 8, True, True, 8, 4, 8, True, 0, 0, 2, 0, 2, 0, 0, 0
        )


# The GEMM kernels take their operands as `const void *` plus one runtime
# dtype tag, so the storage dtype is checked once on the host rather than by a
# `data_ptr<scalar_t>()` inside a per-dtype dispatch. That host check has to
# keep rejecting a pair the kernel cannot load as one type. A (float64,
# float32) pair is one of those: float64 selects the binary64 kernels and the
# other three dtypes the binary32 ones, so a disagreeing pair is rejected
# rather than computed in a carrier one of its operands did not ask for.
@pytest.mark.parametrize(
    "dtype_a, dtype_b",
    [
        (torch.float32, torch.float16),
        (torch.bfloat16, torch.float16),
        (torch.float64, torch.float32),
    ],
)
@pytest.mark.parametrize("device", available_devices)
def test_matmul_rejects_mismatched_operand_dtypes(device, dtype_a, dtype_b):
    """Operands of different dtypes are rejected on the host, for both families."""
    a = torch.randn(4, 3, device=device).to(dtype_a)
    b = torch.randn(5, 3, device=device).to(dtype_b)
    with pytest.raises(RuntimeError):
        binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4)
    with pytest.raises(RuntimeError):
        superfp_matmul(
            a,
            b,
            trans_b=True,
            mul_man_bits=3,
            mul_exp_bits=4,
            mul_normal_binades=1,
            mul_bias=7,
        )


# ------------------------------------------------------------------------------------
# FMA (fused multiply-add) variant: each dot-product step is one
# hardware-style fused multiply-add rounded once to the format, instead of a
# quantized multiply followed by a separately quantized add. Same tier 1/2/3
# structure as above, plus a case that pins down the single-rounding (fused)
# against double-rounding (split) distinction directly.


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
    """
    ``fma_quant=False`` runs the fused step in the binary32 carrier with no
    format rounding (a real FMA), so it tracks a plain matmul within the
    dtype's tolerance and differs from ``fma_quant=True`` at the same
    low-precision format.
    """
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

    # a[i, k].item() and b[j, k].item() read back the exact value whatever
    # the storage dtype, and _fma32 computes in ``np.longdouble`` before its
    # single float32 rounding, so this reference already matches the kernel's
    # convert-on-load order. Only the final narrowing to the storage dtype
    # has to be spelled out, and the kernel does that once, on the write.
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
    """
    FMA analog of test_binaryK_matmul_stochastic_bounds_and_unbiased. With
    K=1 the fused step fma(a, b, 0) is a plain a*b, so the same bounding and
    unbiasedness properties hold for the fused format's one SR rounding.
    ``fma_quant=True`` is the default: unlike the split ops' ``accumulate_quant``
    there is no separate multiply format to leave unquantized.
    """
    K_fmt, P_fmt = 8, 4
    prng_bits = F32_MAN_BITS - (P_fmt - 1)

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
    """Superfp twin of the binaryK fused SR test above."""
    man_bits, exp_bits, normal_binades, bias = 3, 4, 8, 7
    prng_bits = F32_MAN_BITS - man_bits

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
    """
    Split (quantized multiply, then quantized add: two roundings per step)
    and fused (one rounding per step) are different arithmetic, not two entry
    points to the same computation, so at a low-precision format they must
    disagree somewhere in a K=48 reduction.
    """
    M, K, N = 16, 48, 10
    K_fmt, P_fmt = 8, 4
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)

    out_split = binaryK_matmul(a, b, trans_b=True, mul_K=K_fmt, mul_P=P_fmt, accumulate_quant=True)
    out_fused = binaryK_matmul_fma(a, b, trans_b=True, fma_K=K_fmt, fma_P=P_fmt)

    assert torch.isfinite(out_split).all()
    assert torch.isfinite(out_fused).all()
    assert not torch.equal(out_split, out_fused)


def test_raw_matmul_fma_op_rejects_rank_4():
    """The fused op's boundary is rank 2 or 3 too (see the split case above)."""
    a = torch.randn(2, 3, 4, 5)
    b = torch.randn(5, 6)
    with pytest.raises(RuntimeError, match="expects 2D or 3D tensors"):
        torch.ops.mptorch.custom_matmul_binaryK_fma.default(
            a, b, False, False, True, 8, 4, 8, True, 0, 0, 2, 0, 0
        )


# ------------------------------------------------------------------------------------
# Mixed-format (spatially varying) variant: the multiply and accumulate
# formats each output element's dot product runs in are picked, per element,
# from a small palette by a prec_idx tensor (gemm_policy.h's FormatPalette,
# whose slots are prebuilt Mac policies copied into the accumulator before the
# element's K-loop). Same tier 1/2/3 structure, plus equivalence of a one-slot
# palette with the single-format op, the prec_idx broadcast shapes, and the
# host-side validation of the index map and the palette lists.


def _mixed_manual_baseline(a, b, prec_idx, q_by_slot, device):
    """
    Reproduce the mixed kernel's own sequential-over-K summation for
    ``C = a @ b.T``, quantizing every scalar multiply and add with the
    quantizer of the palette slot ``prec_idx[i, j]`` selects. ``prec_idx``
    may be ``[M, N]``, ``[M, 1]`` or ``[1, N]``, broadcast as the kernel does.
    """
    M, K = a.shape
    N = b.shape[0]
    pidx = prec_idx.expand(M, N)
    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            q = q_by_slot[int(pidx[i, j])]
            acc = torch.zeros((), device=device)
            for k in range(K):
                acc = q(acc + q(a[i, k].float() * b[j, k].float()))
            ref[i, j] = acc
    return ref


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("trans_a,trans_b", TRANS_COMBOS)
@pytest.mark.parametrize("M,K,N", [(5, 7, 9), (64, 9, 63)])
def test_binaryK_matmul_mixed_tier1_near_identity(device, trans_a, trans_b, M, K, N):
    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn(*a_shape, device=device)
    b = torch.randn(*b_shape, device=device)

    # Every palette slot is a few mantissa bits narrower than binary32
    # (P = 21..23, that is 20..22 explicit bits against binary32's 23), so
    # quantization genuinely happens, yet each dot product stays within a
    # tight tolerance of the plain fp32 matmul whichever slot it lands on.
    prec_idx = torch.randint(0, 3, (M, N), dtype=torch.int32, device=device)
    out = binaryK_matmul_mixed(
        a,
        b,
        prec_idx,
        trans_a=trans_a,
        trans_b=trans_b,
        mul_K=[32, 31, 30],
        mul_P=[23, 22, 21],
    )
    ref = _ref(a, b, trans_a, trans_b)
    assert torch.allclose(out, ref, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("accumulate_quant", [True, False])
def test_binaryK_matmul_mixed_single_format_matches_nonmixed(device, dtype, accumulate_quant):
    """
    A one-slot palette reproduces the single-format op bit for bit, which
    shows that binding a palette slot into the accumulator does not perturb
    the arithmetic.
    """
    M, K, N = 12, 20, 10
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    prec_idx = torch.zeros(M, N, dtype=torch.int32, device=device)

    out_mixed = binaryK_matmul_mixed(
        a, b, prec_idx, trans_b=True, mul_K=[8], mul_P=[4], accumulate_quant=accumulate_quant
    )
    out_plain = binaryK_matmul(
        a, b, trans_b=True, mul_K=8, mul_P=4, accumulate_quant=accumulate_quant
    )
    assert torch.equal(out_mixed, out_plain)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_binaryK_matmul_mixed_tier3_manual_baseline(device, dtype, round_mode):
    """Bit-exact against the per-slot manual baseline, in every deterministic mode."""
    M, K, N = 4, 5, 6
    formats = [(8, 4), (10, 5), (6, 3)]  # (K, P) per palette slot
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    prec_idx = torch.randint(0, len(formats), (M, N), dtype=torch.int32, device=device)

    q_by_slot = [
        (lambda x, kf=kf, pf=pf: binaryK_quantize(x, K=kf, P=pf, rounding_mode=round_mode))
        for kf, pf in formats
    ]
    ref = _mixed_manual_baseline(a, b, prec_idx, q_by_slot, device)

    out = binaryK_matmul_mixed(
        a,
        b,
        prec_idx,
        trans_b=True,
        mul_K=[kf for kf, _ in formats],
        mul_P=[pf for _, pf in formats],
        accumulate_quant=True,
        rounding_mode=round_mode,
    )
    assert _equal_nan_ok(out, ref.to(dtype))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_mixed_stochastic_bounds_and_unbiased(device, dtype):
    """
    SR on the mixed path. Binding a palette slot replaces only the
    accumulator's Mac, never its Philox stream, so each output element still
    seeds on its global index and draws a fresh value per rounding. K=1 with
    ``accumulate_quant=False`` isolates the multiplier's draw, and two slots
    alternating row by row exercise SR with the format actually varying
    across outputs. Mirrors test_binaryK_matmul_stochastic_bounds_and_unbiased.
    """
    formats = [(8, 4), (6, 3)]  # (K, P) per palette slot
    prng_bits = F32_MAN_BITS - (max(p for _, p in formats) - 1)

    a = torch.rand(400, 1, device=device, dtype=dtype) * 1.8 - 0.9
    b = torch.ones(1, 1, device=device, dtype=dtype)
    prec_idx = (torch.arange(a.shape[0], device=device).reshape(-1, 1) % 2).to(torch.int32)
    kw: dict[str, Any] = dict(
        trans_b=True,
        mul_K=[k for k, _ in formats],
        mul_P=[p for _, p in formats],
        accumulate_quant=False,
    )

    q_rd = binaryK_matmul_mixed(a, b, prec_idx, rounding_mode=RoundMode.RD, **kw)
    q_ru = binaryK_matmul_mixed(a, b, prec_idx, rounding_mode=RoundMode.RU, **kw)
    q_sr = binaryK_matmul_mixed(
        a, b, prec_idx, rounding_mode=RoundMode.SR, mul_prng_bits=prng_bits, **kw
    )
    assert torch.all((q_sr == q_rd) | (q_sr == q_ru))

    # Unbiasedness: the mean over many independent per-output-element draws of
    # a fixed value converges to it even with the format varying row to row.
    # The P=3 rows step by 2^-4 around 0.333333, so the sampling noise of
    # 4000 draws is about 5e-4, well inside the 0.02 margin.
    test_val = 0.333333
    a_const = torch.full((4000, 1), test_val, device=device, dtype=dtype)
    pidx_const = (torch.arange(4000, device=device).reshape(-1, 1) % 2).to(torch.int32)
    q_sr_const = binaryK_matmul_mixed(
        a_const, b, pidx_const, rounding_mode=RoundMode.SR, mul_prng_bits=prng_bits, **kw
    )
    assert abs(q_sr_const.float().mean().item() - test_val) < 0.02


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_mixed_broadcast_row_and_col(device):
    """
    A ``[M, 1]`` or ``[1, N]`` prec_idx gives the same result as the dense
    ``[M, N]`` map it broadcasts to; the kernel reads both through one strided
    index expression, so a wrong stride would show up here.
    """
    M, K, N = 7, 6, 5
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)
    kw: dict[str, Any] = dict(trans_b=True, mul_K=[8, 28], mul_P=[4, 20], accumulate_quant=True)

    row_idx = torch.randint(0, 2, (M, 1), dtype=torch.int32, device=device)
    out_row = binaryK_matmul_mixed(a, b, row_idx, **kw)
    out_row_dense = binaryK_matmul_mixed(a, b, row_idx.expand(M, N).contiguous(), **kw)
    assert torch.equal(out_row, out_row_dense)

    col_idx = torch.randint(0, 2, (1, N), dtype=torch.int32, device=device)
    out_col = binaryK_matmul_mixed(a, b, col_idx, **kw)
    out_col_dense = binaryK_matmul_mixed(a, b, col_idx.expand(M, N).contiguous(), **kw)
    assert torch.equal(out_col, out_col_dense)


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_mixed_prec_idx_dtype_and_layout(device):
    """
    The host casts prec_idx to int32 on the operand's device, checks its
    values against the palette size, and memoizes that check on the caller's
    tensor (its TensorImpl address and version counter, gemm_host.h), since
    on CUDA the check is a device-to-host copy that drains the stream and the
    device copy is a new tensor every call. Every spelling of the same map,
    int64, non-contiguous, or host-resident while the operands are not, must
    give the same answer as the plain one.
    """
    M, K, N = 7, 6, 5
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)
    kw: dict[str, Any] = dict(trans_b=True, mul_K=[8, 28], mul_P=[4, 20])

    base = torch.randint(0, 2, (M, N), dtype=torch.int32, device=device)
    ref = binaryK_matmul_mixed(a, b, base, **kw)

    assert torch.equal(binaryK_matmul_mixed(a, b, base.to(torch.int64), **kw), ref)
    assert torch.equal(binaryK_matmul_mixed(a, b, base.cpu(), **kw), ref)
    strided = torch.empty(M, 2 * N, dtype=torch.int32, device=device)
    strided[:, ::2] = base
    assert torch.equal(binaryK_matmul_mixed(a, b, strided[:, ::2], **kw), ref)


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_mixed_prec_idx_memo_sees_mutation(device):
    """
    The memo keys on the caller's tensor and its version counter, so an
    in-place edit of a map that already passed the check invalidates the
    entry: an edit that keeps it in range must change the result, and one
    that puts it out of range must still be rejected.
    """
    M, K, N = 7, 6, 5
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)
    kw: dict[str, Any] = dict(trans_b=True, mul_K=[8, 28], mul_P=[4, 20])

    idx = torch.zeros(M, N, dtype=torch.int32, device=device)
    all_zero = binaryK_matmul_mixed(a, b, idx, **kw)
    idx.fill_(1)
    all_one = binaryK_matmul_mixed(a, b, idx, **kw)
    assert not torch.equal(all_zero, all_one)
    assert torch.equal(all_one, binaryK_matmul_mixed(a, b, torch.ones_like(idx), **kw))

    idx.fill_(2)
    with pytest.raises(RuntimeError):
        binaryK_matmul_mixed(a, b, idx, **kw)


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_mixed_rejects_bad_prec_idx(device):
    """An out-of-range slot index or a wrongly shaped map is rejected on the host."""
    M, K, N = 5, 4, 6
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)
    kw: dict[str, Any] = dict(trans_b=True, mul_K=[8, 8], mul_P=[4, 3])

    # out-of-range palette index
    with pytest.raises(RuntimeError):
        bad = torch.full((M, N), 2, dtype=torch.int32, device=device)
        binaryK_matmul_mixed(a, b, bad, **kw)

    # wrong shape (not [M, N] / [M, 1] / [1, N])
    with pytest.raises(RuntimeError):
        bad = torch.zeros(N, M, dtype=torch.int32, device=device)
        binaryK_matmul_mixed(a, b, bad, **kw)


def test_binaryK_matmul_mixed_rejects_bad_palette_lengths():
    """
    The palette lists must agree in length (checked in Python) and fit the
    kernel's fixed slot array (checked on the host).
    """
    a = torch.randn(4, 3)
    b = torch.randn(5, 3)
    prec_idx = torch.zeros(4, 5, dtype=torch.int32)
    with pytest.raises(ValueError):
        binaryK_matmul_mixed(a, b, prec_idx, trans_b=True, mul_K=[8, 8], mul_P=[4])
    with pytest.raises(RuntimeError):
        # More entries than MAX_GEMM_FORMATS (8): the palette is a fixed array
        # passed by value into the kernel (gemm_policy.h).
        binaryK_matmul_mixed(a, b, prec_idx, trans_b=True, mul_K=[8] * 9, mul_P=[4] * 9)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_superfp_matmul_mixed_single_format_matches_nonmixed(device, dtype):
    """A one-slot superfp palette reproduces the single-format op bit for bit."""
    M, K, N = 12, 20, 10
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    prec_idx = torch.zeros(M, N, dtype=torch.int32, device=device)

    out_mixed = superfp_matmul_mixed(
        a,
        b,
        prec_idx,
        trans_b=True,
        mul_man_bits=[3],
        mul_exp_bits=[4],
        mul_normal_binades=[8],
        mul_bias=[7],
    )
    out_plain = superfp_matmul(
        a,
        b,
        trans_b=True,
        mul_man_bits=3,
        mul_exp_bits=4,
        mul_normal_binades=8,
        mul_bias=7,
    )
    assert torch.equal(out_mixed, out_plain)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_superfp_matmul_mixed_tier3_manual_baseline(device, dtype, round_mode):
    """Bit-exact against the per-slot manual baseline, in every deterministic mode."""
    M, K, N = 4, 5, 6
    # (man_bits, exp_bits, normal_binades, bias) per palette slot
    formats = [(2, 3, 4, 7), (3, 4, 8, 7)]
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    prec_idx = torch.randint(0, len(formats), (M, N), dtype=torch.int32, device=device)

    q_by_slot = [
        (
            lambda x, mb=mb, eb=eb, nb=nb, bi=bi: superfp_quantize(
                x, man_bits=mb, exp_bits=eb, normal_binades=nb, bias=bi, rounding_mode=round_mode
            )
        )
        for mb, eb, nb, bi in formats
    ]
    ref = _mixed_manual_baseline(a, b, prec_idx, q_by_slot, device)

    out = superfp_matmul_mixed(
        a,
        b,
        prec_idx,
        trans_b=True,
        mul_man_bits=[f[0] for f in formats],
        mul_exp_bits=[f[1] for f in formats],
        mul_normal_binades=[f[2] for f in formats],
        mul_bias=[f[3] for f in formats],
        accumulate_quant=True,
        rounding_mode=round_mode,
    )
    assert _equal_nan_ok(out, ref.to(dtype))


# ------------------------------------------------------------------------------------
# Mixed-format FMA variant: per-output-element selection of the fused
# multiply-add format from a palette of FusedMac slots, one rounding per
# K-step. Same structure as the split _mixed tests above, plus the
# fma_quant=False rejection: with the fused cast replaced by an identity a
# palette has no format to vary.


def _mixed_fma_manual_baseline(a, b, prec_idx, q_by_slot, device):
    """
    FMA analog of ``_mixed_manual_baseline``: reproduce the mixed FMA kernel's
    own summation for ``C = a @ b.T``, one single-rounding float32 fused
    multiply-add per K-step (via ``_fma32``) quantized by the palette slot
    ``prec_idx[i, j]`` selects. ``q_by_slot`` entries take and return a
    Python float.
    """
    M, K = a.shape
    N = b.shape[0]
    pidx = prec_idx.expand(M, N)
    ref = torch.zeros(M, N, device=device)
    for i in range(M):
        for j in range(N):
            q = q_by_slot[int(pidx[i, j])]
            acc = 0.0
            for k in range(K):
                acc = q(_fma32(a[i, k].item(), b[j, k].item(), acc))
            ref[i, j] = acc
    return ref


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("trans_a,trans_b", TRANS_COMBOS)
@pytest.mark.parametrize("M,K,N", [(5, 7, 9), (64, 9, 63)])
def test_binaryK_matmul_fma_mixed_tier1_near_identity(device, trans_a, trans_b, M, K, N):
    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn(*a_shape, device=device)
    b = torch.randn(*b_shape, device=device)

    # Palette slots a few mantissa bits below binary32 (P = 21..23), so
    # quantization genuinely happens while staying near-identity.
    prec_idx = torch.randint(0, 3, (M, N), dtype=torch.int32, device=device)
    out = binaryK_matmul_fma_mixed(
        a,
        b,
        prec_idx,
        trans_a=trans_a,
        trans_b=trans_b,
        fma_K=[32, 31, 30],
        fma_P=[23, 22, 21],
    )
    ref = _ref(a, b, trans_a, trans_b)
    assert torch.allclose(out, ref, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_fma_mixed_single_format_matches_nonmixed(device, dtype):
    """A one-slot palette reproduces the single-format FMA op bit for bit."""
    M, K, N = 12, 20, 10
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    prec_idx = torch.zeros(M, N, dtype=torch.int32, device=device)

    out_mixed = binaryK_matmul_fma_mixed(a, b, prec_idx, trans_b=True, fma_K=[8], fma_P=[4])
    out_plain = binaryK_matmul_fma(a, b, trans_b=True, fma_K=8, fma_P=4)
    assert torch.equal(out_mixed, out_plain)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_binaryK_matmul_fma_mixed_tier3_manual_baseline(device, dtype, round_mode):
    """Bit-exact against the per-slot fused manual baseline, in every mode."""
    M, K, N = 4, 5, 6
    formats = [(8, 4), (10, 5), (6, 3)]  # (K, P) per palette slot
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    prec_idx = torch.randint(0, len(formats), (M, N), dtype=torch.int32, device=device)

    q_by_slot = [
        (
            lambda x, kf=kf, pf=pf: binaryK_quantize(
                torch.tensor(x, dtype=torch.float32, device=device),
                K=kf,
                P=pf,
                rounding_mode=round_mode,
            ).item()
        )
        for kf, pf in formats
    ]
    ref = _mixed_fma_manual_baseline(a, b, prec_idx, q_by_slot, device)

    out = binaryK_matmul_fma_mixed(
        a,
        b,
        prec_idx,
        trans_b=True,
        fma_K=[kf for kf, _ in formats],
        fma_P=[pf for _, pf in formats],
        rounding_mode=round_mode,
    )
    assert _equal_nan_ok(out, ref.to(dtype))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_binaryK_matmul_fma_mixed_stochastic_bounds_and_unbiased(device, dtype):
    """
    SR on the mixed FMA path. With K=1 the fused step fma(a, b, 0) is a
    plain a*b, so the fused format's one SR draw per step has the bounding
    and unbiasedness properties, here with the format varying row to row.
    Mirrors test_binaryK_matmul_fma_stochastic_bounds_and_unbiased.
    """
    formats = [(8, 4), (6, 3)]  # (K, P) per palette slot
    prng_bits = F32_MAN_BITS - (max(p for _, p in formats) - 1)

    a = torch.rand(400, 1, device=device, dtype=dtype) * 1.8 - 0.9
    b = torch.ones(1, 1, device=device, dtype=dtype)
    prec_idx = (torch.arange(a.shape[0], device=device).reshape(-1, 1) % 2).to(torch.int32)
    kw: dict[str, Any] = dict(
        trans_b=True,
        fma_K=[k for k, _ in formats],
        fma_P=[p for _, p in formats],
    )

    q_rd = binaryK_matmul_fma_mixed(a, b, prec_idx, rounding_mode=RoundMode.RD, **kw)
    q_ru = binaryK_matmul_fma_mixed(a, b, prec_idx, rounding_mode=RoundMode.RU, **kw)
    q_sr = binaryK_matmul_fma_mixed(
        a, b, prec_idx, rounding_mode=RoundMode.SR, fma_prng_bits=prng_bits, **kw
    )
    assert torch.all((q_sr == q_rd) | (q_sr == q_ru))

    test_val = 0.333333
    a_const = torch.full((4000, 1), test_val, device=device, dtype=dtype)
    pidx_const = (torch.arange(4000, device=device).reshape(-1, 1) % 2).to(torch.int32)
    q_sr_const = binaryK_matmul_fma_mixed(
        a_const, b, pidx_const, rounding_mode=RoundMode.SR, fma_prng_bits=prng_bits, **kw
    )
    assert abs(q_sr_const.float().mean().item() - test_val) < 0.02


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_fma_mixed_broadcast_row_and_col(device):
    """
    A ``[M, 1]`` or ``[1, N]`` prec_idx matches the dense ``[M, N]`` map it
    broadcasts to. All mixed ops resolve the map through the same host
    routine, so one fused op is enough here.
    """
    M, K, N = 7, 6, 5
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)
    kw: dict[str, Any] = dict(trans_b=True, fma_K=[8, 28], fma_P=[4, 20])

    row_idx = torch.randint(0, 2, (M, 1), dtype=torch.int32, device=device)
    assert torch.equal(
        binaryK_matmul_fma_mixed(a, b, row_idx, **kw),
        binaryK_matmul_fma_mixed(a, b, row_idx.expand(M, N).contiguous(), **kw),
    )
    col_idx = torch.randint(0, 2, (1, N), dtype=torch.int32, device=device)
    assert torch.equal(
        binaryK_matmul_fma_mixed(a, b, col_idx, **kw),
        binaryK_matmul_fma_mixed(a, b, col_idx.expand(M, N).contiguous(), **kw),
    )


@pytest.mark.parametrize("device", available_devices)
def test_binaryK_matmul_fma_mixed_rejects_fma_quant_false(device):
    """
    ``fma_quant=False`` replaces the fused cast with an identity, leaving a
    palette nothing to vary, so the mixed FMA ops reject it on the host.
    """
    a = torch.randn(4, 3, device=device)
    b = torch.randn(5, 3, device=device)
    prec_idx = torch.zeros(4, 5, dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError):
        binaryK_matmul_fma_mixed(
            a, b, prec_idx, trans_b=True, fma_K=[8, 6], fma_P=[4, 3], fma_quant=False
        )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_superfp_matmul_fma_mixed_single_format_matches_nonmixed(device, dtype):
    """A one-slot superfp palette reproduces the single-format FMA op bit for bit."""
    M, K, N = 12, 20, 10
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    prec_idx = torch.zeros(M, N, dtype=torch.int32, device=device)

    out_mixed = superfp_matmul_fma_mixed(
        a,
        b,
        prec_idx,
        trans_b=True,
        fma_man_bits=[3],
        fma_exp_bits=[4],
        fma_normal_binades=[8],
        fma_bias=[7],
    )
    out_plain = superfp_matmul_fma(
        a, b, trans_b=True, fma_man_bits=3, fma_exp_bits=4, fma_normal_binades=8, fma_bias=7
    )
    assert torch.equal(out_mixed, out_plain)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_superfp_matmul_fma_mixed_tier3_manual_baseline(device, dtype, round_mode):
    """Bit-exact against the per-slot fused manual baseline, in every mode."""
    M, K, N = 4, 5, 6
    # (man_bits, exp_bits, normal_binades, bias) per palette slot
    formats = [(2, 3, 4, 7), (3, 4, 8, 7)]
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    prec_idx = torch.randint(0, len(formats), (M, N), dtype=torch.int32, device=device)

    q_by_slot = [
        (
            lambda x, mb=mb, eb=eb, nb=nb, bi=bi: superfp_quantize(
                torch.tensor(x, dtype=torch.float32, device=device),
                man_bits=mb,
                exp_bits=eb,
                normal_binades=nb,
                bias=bi,
                rounding_mode=round_mode,
            ).item()
        )
        for mb, eb, nb, bi in formats
    ]
    ref = _mixed_fma_manual_baseline(a, b, prec_idx, q_by_slot, device)

    out = superfp_matmul_fma_mixed(
        a,
        b,
        prec_idx,
        trans_b=True,
        fma_man_bits=[f[0] for f in formats],
        fma_exp_bits=[f[1] for f in formats],
        fma_normal_binades=[f[2] for f in formats],
        fma_bias=[f[3] for f in formats],
        rounding_mode=round_mode,
    )
    assert _equal_nan_ok(out, ref.to(dtype))
