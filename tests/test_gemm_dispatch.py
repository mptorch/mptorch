"""
Tests for what the GEMM entry points do around the kernels: which carrier a
float64 operand pair is computed in (dev/binary64_carrier_plan.md, phase 4),
and the validation of a mixed-format op's precision map (finding G5), which is
now memoized per map so that a map reused across calls costs no device
synchronization.

A float64 GEMM used to be narrowed to float32 on the way in and widened on the
way out (finding G6), so it was the float32 GEMM's image by construction. It is
now computed in binary64 -- every product, every sum and every cast -- so the
image holds only where both carriers compute exactly, and the rest is checked
against a reference built from the elementwise quantizers, which
tests/test_quantize_dispatch.py checks against IEEE P3109 in float64. The
reference is Python's own float64 arithmetic, exactly the kernel's steps in
the kernel's order, with the fused step as a correctly-rounded Fraction sum.

G5 may not move a value either: a memo hit has to be indistinguishable from
running the check again, which means it has to be invalidated by anything that
could change the answer -- an in-place edit of the map, or the same map used
against a different palette.
"""

from fractions import Fraction
from typing import Any

import pytest
import torch

from mptorch.number import RoundMode
from mptorch.quant import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_fma_mixed,
    binaryK_matmul_mixed,
    superfp_matmul,
    superfp_matmul_fma,
    superfp_matmul_fma_mixed,
    superfp_matmul_mixed,
)
from tests.markers import available_devices

DETERMINISTIC = [rm for rm in RoundMode if rm != RoundMode.SR]


def _gemm_calls(a, b, prec_idx, rounding_mode=RoundMode.RNE, prng_bits=0):
    """Every GEMM entry point at one fixed format, keyed by op name."""
    rm = rounding_mode
    pb = prng_bits
    return {
        "binaryK": lambda: binaryK_matmul(
            a,
            b,
            trans_b=True,
            mul_K=8,
            mul_P=4,
            acc_K=10,
            acc_P=5,
            rounding_mode=rm,
            mul_prng_bits=pb,
            acc_prng_bits=pb,
        ),
        "superfp": lambda: superfp_matmul(
            a,
            b,
            trans_b=True,
            mul_man_bits=3,
            mul_exp_bits=4,
            mul_normal_binades=8,
            mul_bias=7,
            rounding_mode=rm,
            mul_prng_bits=pb,
            acc_prng_bits=pb,
        ),
        "binaryK_fma": lambda: binaryK_matmul_fma(
            a, b, trans_b=True, fma_K=8, fma_P=4, rounding_mode=rm, fma_prng_bits=pb
        ),
        "superfp_fma": lambda: superfp_matmul_fma(
            a,
            b,
            trans_b=True,
            fma_man_bits=3,
            fma_exp_bits=4,
            fma_normal_binades=8,
            fma_bias=7,
            rounding_mode=rm,
            fma_prng_bits=pb,
        ),
        "binaryK_mixed": lambda: binaryK_matmul_mixed(
            a,
            b,
            prec_idx,
            trans_b=True,
            mul_K=[8, 8],
            mul_P=[4, 3],
            rounding_mode=rm,
            mul_prng_bits=pb,
            acc_prng_bits=pb,
        ),
        "superfp_mixed": lambda: superfp_matmul_mixed(
            a,
            b,
            prec_idx,
            trans_b=True,
            mul_man_bits=[3, 2],
            mul_exp_bits=[4, 4],
            mul_normal_binades=[8, 8],
            mul_bias=[7, 7],
            rounding_mode=rm,
            mul_prng_bits=pb,
            acc_prng_bits=pb,
        ),
        "binaryK_fma_mixed": lambda: binaryK_matmul_fma_mixed(
            a,
            b,
            prec_idx,
            trans_b=True,
            fma_K=[8, 8],
            fma_P=[4, 3],
            rounding_mode=rm,
            fma_prng_bits=pb,
        ),
        "superfp_fma_mixed": lambda: superfp_matmul_fma_mixed(
            a,
            b,
            prec_idx,
            trans_b=True,
            fma_man_bits=[3, 2],
            fma_exp_bits=[4, 4],
            fma_normal_binades=[8, 8],
            fma_bias=[7, 7],
            rounding_mode=rm,
            fma_prng_bits=pb,
        ),
    }


OP_NAMES = list(_gemm_calls(torch.empty(0), torch.empty(0), torch.empty(0)).keys())


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("rounding_mode", DETERMINISTIC)
def test_float64_of_exact_arithmetic_is_the_float32_image(device, op, rounding_mode):
    """
    Where both carriers compute every product and sum exactly, a float64 GEMM
    is the float32 GEMM's image: the casts then see the same values, and the
    elementwise image identity does the rest. Operands on a 2^-4 grid in
    [-4, 4) keep every intermediate a multiple of 2^-8 below 2^16, which
    binary32 holds. This is the plumbing check -- dtype in, dtype out, the
    right kernel, every op.
    """
    M, K, N = 12, 20, 10
    gen = torch.Generator().manual_seed(7)
    a64 = (torch.randint(-64, 64, (M, K), generator=gen) / 16).double().to(device)
    b64 = (torch.randint(-64, 64, (N, K), generator=gen) / 16).double().to(device)
    pidx = torch.randint(0, 2, (M, N), dtype=torch.int32, generator=gen).to(device)

    out64 = _gemm_calls(a64, b64, pidx, rounding_mode)[op]()
    out32 = _gemm_calls(a64.float(), b64.float(), pidx, rounding_mode)[op]()

    assert out64.dtype == torch.float64
    assert out32.dtype == torch.float32
    assert torch.equal(out64, out32.double())


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fused", [False, True])
def test_float64_rounds_the_product_once(device, fused):
    """
    The witness that the carrier moved. (1 + 2^-13)^2 = 1 + 2^-12 + 2^-26 is
    just above the tie between 1 and 1 + 2^-11 on a P = 12 grid. binary32
    rounds the product to 24 bits first, onto the tie itself, and RNE then
    takes it to 1; binary64 holds the product whole and rounds it up. Both
    operands are float32 values, so this is the carrier and nothing else.
    """
    x = 1 + 2**-13
    kw: dict[str, Any] = {}
    for dtype, expected in ((torch.float32, 1.0), (torch.float64, 1 + 2**-11)):
        a = torch.full((1, 1), x, dtype=dtype, device=device)
        b = torch.full((1, 1), x, dtype=dtype, device=device)
        if fused:
            out = binaryK_matmul_fma(a, b, fma_K=16, fma_P=12, **kw)
        else:
            out = binaryK_matmul(a, b, mul_K=16, mul_P=12, acc_K=16, acc_P=12, **kw)
        assert out.dtype == dtype
        assert out.item() == expected


@pytest.mark.parametrize("device", available_devices)
def test_float64_operand_pair_must_agree(device):
    """A mismatched (float64, float32) pair is still rejected, not coerced."""
    a = torch.randn(4, 3, device=device, dtype=torch.float64)
    b = torch.randn(5, 3, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="same dtype"):
        binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4)


# ---------------------------------------------------------------------------
# float64 against a reference, through the raw ops
#
# The typed wrappers still hold every format to binary32's bounds (that moves
# with the carrier into mptorch.number, phase 5), so the formats only binary64
# can carry -- P up to 53, exponents to ten bits -- are reached through
# torch.ops, as test_quantize_dispatch.py reaches them for the quantizers.
# A format is ("binaryK", K, P, bias) or ("superfp", man_bits, exp_bits,
# normal_binades, bias); every format here is signed, OVF_INF, SUBNORMALS.


def _bk(K, P):
    return ("binaryK", K, P, 2 ** (K - P - 1))


def _sfp(man_bits, exp_bits, normal_binades, bias):
    return ("superfp", man_bits, exp_bits, normal_binades, bias)


# (multiply palette, accumulate palette) per family and width. A single-format
# op takes the first slot. Narrow formats are inside binary32's bounds, but
# fine enough (P = 20 to 23, and superfp biased so the operands' magnitudes land
# in its normal region) that rounding a product or a sum to 24 bits first moves
# their results, so they fail against a build that narrows; wide ones need
# binary64's precision or range.
PALETTES = {
    ("binaryK", "narrow"): ([_bk(26, 22), _bk(24, 20)], [_bk(27, 23), _bk(26, 22)]),
    ("binaryK", "wide"): ([_bk(48, 40), _bk(40, 30)], [_bk(60, 50), _bk(63, 53)]),
    ("superfp", "narrow"): (
        [_sfp(21, 4, 8, 11), _sfp(19, 4, 8, 11)],
        [_sfp(22, 5, 16, 23), _sfp(21, 5, 20, 23)],
    ),
    ("superfp", "wide"): (
        [_sfp(40, 8, 250, 127), _sfp(30, 8, 200, 127)],
        [_sfp(50, 9, 500, 255), _sfp(52, 10, 900, 511)],
    ),
}


def _quant(fmt, x, rm, prng_bits=0):
    if fmt[0] == "binaryK":
        _, K, P, bias = fmt
        return torch.ops.mptorch.binaryK_quant.default(
            x, K, P, bias, prng_bits, True, rm.value, 0, 0
        )
    _, man_bits, exp_bits, normal_binades, bias = fmt
    return torch.ops.mptorch.superfp_quant.default(
        x, man_bits, exp_bits, normal_binades, bias, prng_bits, True, rm.value, 0
    )


def _raw_gemm(fused, a, b, prec_idx, mul, acc, rm, *, quant=True, prng_bits=0):
    """One raw GEMM call: split or fused, single-format when prec_idx is None."""
    fam = acc[0][0]
    mixed = prec_idx is not None

    def fields(prefix, fmts):
        names = (
            ["K", "P", "bias"]
            if fam == "binaryK"
            else ["man_bits", "exp_bits", "normal_binades", "bias"]
        )
        return {
            f"{prefix}_{name}": [f[i + 1] for f in fmts] if mixed else fmts[0][i + 1]
            for i, name in enumerate(names)
        }

    kw: dict[str, Any] = dict(
        trans_a=False, trans_b=False, accumulate_algorithm=0, round_mode=rm.value
    )
    slots = ["fma"] if fused else ["mul", "acc"]
    for slot, fmts in zip(slots, [acc] if fused else [mul, acc], strict=True):
        kw |= fields(slot, fmts)
        kw[f"{slot}_is_signed"] = True
        kw[f"{slot}_saturation_mode"] = 0
        kw[f"{slot}_prng_bits"] = prng_bits
        if fam == "binaryK":
            kw[f"{slot}_subnormals_mode"] = 0
    kw["fma_quant" if fused else "accumulate_quant"] = quant

    name = f"custom_matmul_{fam}" + ("_fma" if fused else "") + ("_mixed" if mixed else "")
    op = getattr(torch.ops.mptorch, name).default
    return op(a, b, prec_idx, **kw) if mixed else op(a, b, **kw)


def _reference(fused, a, b, prec_idx, mul, acc, rm, *, quant=True):
    """The kernel's steps, one K-step at a time, in float64 on the host.

    A split step is ``Q_acc(acc + Q_mul(a * b))`` in float64, which torch's
    elementwise ``*`` and ``+`` are; a fused step is ``Q_acc(fma(a, b, acc))``,
    which Python 3.12 has no function for, so it is the correctly-rounded
    conversion of the exact Fraction sum. Each palette slot is computed over
    the whole output and the slot the map names is kept.
    """
    a = a.cpu()
    b = b.cpu()
    M, K = a.shape
    N = b.shape[1]
    slots = range(len(acc)) if prec_idx is not None else range(1)
    out = torch.empty(M, N, dtype=torch.float64)
    for slot in slots:
        s = torch.zeros(M, N, dtype=torch.float64)
        for k in range(K):
            if not fused:
                prod = _quant(mul[slot], a[:, k, None] * b[None, k, :], rm)
                s = s + prod
            else:
                al, bl, sl = a[:, k].tolist(), b[k, :].tolist(), s.tolist()
                s = torch.tensor(
                    [
                        [
                            float(Fraction(al[i]) * Fraction(bl[j]) + Fraction(sl[i][j]))
                            for j in range(N)
                        ]
                        for i in range(M)
                    ],
                    dtype=torch.float64,
                )
            if quant:
                s = _quant(acc[slot], s, rm)
        if prec_idx is None:
            return s
        keep = prec_idx.cpu() == slot
        out[keep] = s[keep]
    return out


def _operands(M, K, N, device, width, seed=11):
    """float64 operands no binary32 value equals, spread over binades.

    A quarter of each operand sits several binades down, so the products
    reach the narrow formats' subnormals and the sums cross binades; the wide
    formats get a wider spread to use their range.
    """
    gen = torch.Generator().manual_seed(seed)
    spread = 6 if width == "narrow" else 40
    a = torch.randn(M, K, dtype=torch.float64, generator=gen)
    b = torch.randn(K, N, dtype=torch.float64, generator=gen)
    a[::2, ::2] *= 2.0 ** -torch.randint(0, spread, (1,), generator=gen).item()
    b[1::2, ::2] *= 2.0 ** -torch.randint(0, spread, (1,), generator=gen).item()
    assert not torch.equal(a, a.float().double())
    return a.to(device), b.to(device)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fam", ["binaryK", "superfp"])
@pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
@pytest.mark.parametrize("mixed", [False, True], ids=["single", "mixed"])
@pytest.mark.parametrize("width", ["narrow", "wide"])
@pytest.mark.parametrize("rounding_mode", DETERMINISTIC)
def test_float64_gemm_matches_its_reference(device, fam, fused, mixed, width, rounding_mode):
    """
    Tier 3: every op, both families, formats binary32 can carry and formats
    only binary64 can, six rounding modes, against the float64 reference.
    """
    M, K, N = 6, 9, 7
    a, b = _operands(M, K, N, device, width)
    mul, acc = PALETTES[(fam, width)]
    prec_idx = None
    if mixed:
        gen = torch.Generator().manual_seed(3)
        prec_idx = torch.randint(0, 2, (M, N), dtype=torch.int32, generator=gen).to(device)

    out = _raw_gemm(fused, a, b, prec_idx, mul, acc, rounding_mode)
    ref = _reference(fused, a, b, prec_idx, mul, acc, rounding_mode)

    assert out.dtype == torch.float64
    assert torch.equal(out.cpu(), ref)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fam", ["binaryK", "superfp"])
@pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
@pytest.mark.parametrize("rounding_mode", [RoundMode.RNE, RoundMode.RZ])
def test_float64_unquantized_accumulator_is_float64_arithmetic(device, fam, fused, rounding_mode):
    """
    accumulate_quant=False / fma_quant=False leave the running sum in the
    carrier, which is now binary64: a split sum is float64 addition, and an
    unquantized fused step is the float64 FMA itself.
    """
    M, K, N = 5, 9, 4
    a, b = _operands(M, K, N, device, "narrow")
    mul, acc = PALETTES[(fam, "narrow")]

    out = _raw_gemm(fused, a, b, None, mul, acc, rounding_mode, quant=False)
    ref = _reference(fused, a, b, None, mul, acc, rounding_mode, quant=False)

    assert torch.equal(out.cpu(), ref)


@pytest.mark.parametrize("device", available_devices)
def test_float64_identity_format_is_the_float64_sum(device):
    """
    Tier 1: a 53-bit format rounds every float64 in its normal range to
    itself, so a GEMM with it is plain float64 arithmetic -- the products and
    the running sums in the kernel's k-ascending order -- bit for bit.
    """
    M, K, N = 17, 33, 9
    gen = torch.Generator().manual_seed(5)
    a = torch.randn(M, K, dtype=torch.float64, generator=gen)
    b = torch.randn(K, N, dtype=torch.float64, generator=gen)
    fmt = [_bk(63, 53)]

    out = _raw_gemm(False, a.to(device), b.to(device), None, fmt, fmt, RoundMode.RNE)
    s = torch.zeros(M, N, dtype=torch.float64)
    for k in range(K):
        s = s + a[:, k, None] * b[None, k, :]

    assert torch.equal(out.cpu(), s)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_float64_sr_stream_is_keyed_on_the_element(device, op):
    """
    A binary64 SR draw is two stream words, but the stream is still the
    element's, keyed on its index into the output: the rows of a 3-row call
    are the first three rows of a 7-row call from the same generator state.
    """
    K, N = 20, 10
    gen = torch.Generator().manual_seed(9)
    a = torch.randn(7, K, dtype=torch.float64, generator=gen).to(device)
    b = torch.randn(N, K, dtype=torch.float64, generator=gen).to(device)
    pidx = torch.randint(0, 2, (7, N), dtype=torch.int32, generator=gen).to(device)

    torch.manual_seed(1234)
    big = _gemm_calls(a, b, pidx, RoundMode.SR, prng_bits=12)[op]()
    torch.manual_seed(1234)
    small = _gemm_calls(a[:3], b, pidx[:3], RoundMode.SR, prng_bits=12)[op]()
    torch.manual_seed(4321)
    other = _gemm_calls(a, b, pidx, RoundMode.SR, prng_bits=12)[op]()

    assert big.dtype == torch.float64
    assert torch.equal(small, big[:3])
    assert not torch.equal(other, big)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
def test_float64_sr_is_unbiased_with_wide_random_bits(device, fused):
    """
    Every element of a K = 1 call rounds the same product x, each from its own
    stream, so the outputs are samples of one SR draw: each is one of x's two
    neighbours, and the fraction that went up estimates (x - lo) / (hi - lo).
    40 random bits -- more than binary32 has below a P = 4 mantissa -- and
    65,536 samples, so the tolerance is ~5.5 standard errors.
    """
    M = N = 256
    x = 1.0375  # between 1 and 1.125 on a P = 4 grid, 0.3 of the way up
    a = torch.full((M, 1), x, dtype=torch.float64, device=device)
    b = torch.ones((1, N), dtype=torch.float64, device=device)
    fmt = [_bk(8, 4)]

    torch.manual_seed(0)
    out = _raw_gemm(fused, a, b, None, fmt, fmt, RoundMode.SR, quant=fused, prng_bits=40)

    assert bool(((out == 1.0) | (out == 1.125)).all())
    p = (out == 1.125).double().mean().item()
    assert p == pytest.approx((x - 1.0) / 0.125, abs=0.01)


@pytest.mark.parametrize("device", available_devices)
def test_mixed_prec_idx_rechecked_after_in_place_edit(device):
    """
    G5: the validation memo is keyed on the map's identity *and* its version,
    so writing an out-of-range index into a map that already passed must
    invalidate the entry rather than ride on it.
    """
    M, K, N = 5, 4, 6
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)
    kw: dict[str, Any] = dict(trans_b=True, mul_K=[8, 8], mul_P=[4, 3])

    pidx = torch.zeros(M, N, dtype=torch.int32, device=device)
    binaryK_matmul_mixed(a, b, pidx, **kw)  # passes, and is remembered

    pidx.fill_(7)
    with pytest.raises(RuntimeError):
        binaryK_matmul_mixed(a, b, pidx, **kw)

    # and back again: the repaired map has to be accepted, not stay poisoned
    pidx.fill_(1)
    binaryK_matmul_mixed(a, b, pidx, **kw)


@pytest.mark.parametrize("device", available_devices)
def test_mixed_prec_idx_rechecked_against_a_smaller_palette(device):
    """
    G5: an index of 2 is in range for a 3-slot palette and out of range for a
    2-slot one, so the memo cannot be keyed on the map alone.
    """
    M, K, N = 5, 4, 6
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)
    pidx = torch.full((M, N), 2, dtype=torch.int32, device=device)

    binaryK_matmul_mixed(a, b, pidx, trans_b=True, mul_K=[8, 8, 8], mul_P=[4, 3, 2])
    with pytest.raises(RuntimeError):
        binaryK_matmul_mixed(a, b, pidx, trans_b=True, mul_K=[8, 8], mul_P=[4, 3])


@pytest.mark.parametrize("device", available_devices)
def test_mixed_repeated_calls_are_identical(device):
    """
    G5: whether the check ran or was skipped must not be observable in the
    output -- the second call through the memo has to match the first.
    """
    M, K, N = 16, 12, 10
    a = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)
    pidx = torch.randint(0, 2, (M, N), dtype=torch.int32, device=device)
    kw: dict[str, Any] = dict(trans_b=True, mul_K=[8, 8], mul_P=[4, 3])

    first = binaryK_matmul_mixed(a, b, pidx, **kw)
    for _ in range(3):
        assert torch.equal(binaryK_matmul_mixed(a, b, pidx, **kw), first)


@pytest.mark.parametrize("device", available_devices)
def test_mixed_prec_idx_checked_under_inference_mode(device):
    """
    G5: a tensor created under inference mode has no version counter to
    invalidate against, so it is never memoized and takes the full check
    every time.
    """
    with torch.inference_mode():
        a = torch.randn(5, 4, device=device)
        b = torch.randn(6, 4, device=device)
        bad = torch.full((5, 6), 3, dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError):
            binaryK_matmul_mixed(a, b, bad, trans_b=True, mul_K=[8, 8], mul_P=[4, 3])


# The raw ops, spelled out because the typed wrappers cannot reach this: they
# take a RoundMode and pass its `.value`, so only a direct
# torch.ops.mptorch.* call can hand the entry point an integer that names no
# mode. One split op and one fma_mixed op are enough -- all sixteen entry
# points share the single check in common/gemm_host.h's check_matmul_inputs.
def _raw_gemm_calls(a, b, prec_idx):
    """Two GEMM ops called through torch.ops, keyed by op name.

    Every schema argument by name: these calls exist to vary one of them, so
    spelling the rest positionally would hide which.
    """
    return {
        "binaryK": lambda rm: torch.ops.mptorch.custom_matmul_binaryK.default(
            a,
            b,
            trans_a=False,
            trans_b=True,
            mul_K=8,
            mul_P=4,
            mul_bias=127,
            mul_is_signed=True,
            accumulate_quant=True,
            acc_K=10,
            acc_P=5,
            acc_bias=127,
            acc_is_signed=True,
            accumulate_algorithm=0,
            round_mode=rm,
            mul_saturation_mode=0,
            mul_subnormals_mode=0,
            acc_saturation_mode=0,
            acc_subnormals_mode=0,
            mul_prng_bits=0,
            acc_prng_bits=0,
        ),
        "superfp_fma_mixed": lambda rm: torch.ops.mptorch.custom_matmul_superfp_fma_mixed.default(
            a,
            b,
            prec_idx,
            trans_a=False,
            trans_b=True,
            fma_quant=True,
            fma_man_bits=[3, 2],
            fma_exp_bits=[4, 4],
            fma_normal_binades=[8, 8],
            fma_bias=[7, 7],
            fma_is_signed=True,
            accumulate_algorithm=0,
            round_mode=rm,
            fma_saturation_mode=0,
            fma_prng_bits=0,
        ),
    }


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", ["binaryK", "superfp_fma_mixed"])
@pytest.mark.parametrize("round_mode", [-1, len(RoundMode), 99, 2**40])
def test_round_mode_outside_the_enum_is_rejected(device, op, round_mode):
    """
    K3: an integer that names no RoundMode used to fall through
    dispatch_round_mode's ``default:`` and round to nearest-even in silence.
    2**40 is in the set because the check has to reject it before anything
    casts it to the enum's underlying int.
    """
    a = torch.randn(16, 12, device=device)
    b = torch.randn(10, 12, device=device)
    pidx = torch.randint(0, 2, (16, 10), dtype=torch.int32, device=device)
    call = _raw_gemm_calls(a, b, pidx)[op]

    with pytest.raises(RuntimeError, match="is not a RoundMode"):
        call(round_mode)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", ["binaryK", "superfp_fma_mixed"])
@pytest.mark.parametrize("round_mode", list(RoundMode))
def test_every_round_mode_is_accepted(device, op, round_mode):
    """The other half of the check above: no mode the enum names is refused."""
    a = torch.randn(16, 12, device=device)
    b = torch.randn(10, 12, device=device)
    pidx = torch.randint(0, 2, (16, 10), dtype=torch.int32, device=device)

    out = _raw_gemm_calls(a, b, pidx)[op](round_mode.value)
    assert out.shape == (16, 10)
