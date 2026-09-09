"""
Tests for what the GEMM entry points do before any kernel runs: the float64
narrowing that lets the type dispatch drop its ``double`` instantiation
(finding G6 in dev/gemm_perf_audit.md), and the validation of a mixed-format
op's precision map (finding G5), which is now memoized per map so that a map
reused across calls costs no device synchronization.

Neither may move a value. G6 because ``scalar_t`` was only ever these
kernels' load/store type -- the tile buffers, the accumulator and the whole
Mac policy chain are float, so a float64 operand was already being narrowed
on load and widened on store, and doing that in a cast pass instead is the
same conversion. G5 because a memo hit has to be indistinguishable from
running the check again, which means it has to be invalidated by anything
that could change the answer: an in-place edit of the map, or the same map
used against a different palette.
"""

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


def _gemm_calls(a, b, prec_idx):
    """Every GEMM entry point at one fixed format, keyed by op name."""
    return {
        "binaryK": lambda: binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4, acc_K=10, acc_P=5),
        "superfp": lambda: superfp_matmul(
            a,
            b,
            trans_b=True,
            mul_man_bits=3,
            mul_exp_bits=4,
            mul_normal_binades=8,
            mul_bias=7,
        ),
        "binaryK_fma": lambda: binaryK_matmul_fma(a, b, trans_b=True, fma_K=8, fma_P=4),
        "superfp_fma": lambda: superfp_matmul_fma(
            a,
            b,
            trans_b=True,
            fma_man_bits=3,
            fma_exp_bits=4,
            fma_normal_binades=8,
            fma_bias=7,
        ),
        "binaryK_mixed": lambda: binaryK_matmul_mixed(
            a, b, prec_idx, trans_b=True, mul_K=[8, 8], mul_P=[4, 3]
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
        ),
        "binaryK_fma_mixed": lambda: binaryK_matmul_fma_mixed(
            a, b, prec_idx, trans_b=True, fma_K=[8, 8], fma_P=[4, 3]
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
        ),
    }


OP_NAMES = list(_gemm_calls(torch.empty(0), torch.empty(0), torch.empty(0)).keys())


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_float64_matches_float32_bit_for_bit(device, op):
    """
    G6: a float64 operand comes back as float64 and holds exactly what the
    float32 call produced, because that is all the double instantiation ever
    computed.
    """
    M, K, N = 12, 20, 10
    a64 = torch.randn(M, K, device=device, dtype=torch.float64)
    b64 = torch.randn(N, K, device=device, dtype=torch.float64)
    pidx = torch.randint(0, 2, (M, N), dtype=torch.int32, device=device)

    out64 = _gemm_calls(a64, b64, pidx)[op]()
    out32 = _gemm_calls(a64.float(), b64.float(), pidx)[op]()

    assert out64.dtype == torch.float64
    assert out32.dtype == torch.float32
    assert torch.equal(out64, out32.double())


@pytest.mark.parametrize("device", available_devices)
def test_float64_operand_pair_must_agree(device):
    """A mismatched (float64, float32) pair is still rejected, not coerced."""
    a = torch.randn(4, 3, device=device, dtype=torch.float64)
    b = torch.randn(5, 3, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4)


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
