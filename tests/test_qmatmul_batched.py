"""Batched and broadcasting GEMM: the ``torch.matmul`` operand contract (X1).

The three tiers of ``test_qmatmul.py``, applied to the operand rules rather
than to the arithmetic -- Tier 3 is the load-bearing one here, because the
claim is not "close to ``a @ b``" but "the same values a loop of 2D calls
produces", which is checkable with ``torch.equal``.
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
    superfp_matmul_mixed,
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

MATMUL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# A near-identity binaryK format: 24 bits of precision in 32, i.e. float32's
# own, so the only difference from `a @ b` is the reduction order.
IDENTITY_BK: dict[str, Any] = dict(mul_K=32, mul_P=24, acc_K=32, acc_P=24)
BK: dict[str, Any] = dict(mul_K=8, mul_P=4)
SFP: dict[str, Any] = dict(mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=8, mul_bias=7)
PALETTE: dict[str, Any] = dict(mul_K=[8, 6], mul_P=[4, 3])

# Every rank combination the contract names, as (a shape, b shape). The last
# two are genuine partial broadcasts, where torch.matmul itself copies.
RANK_CASES = [
    ((5,), (5,)),
    ((5,), (5, 3)),
    ((4, 5), (5,)),
    ((4, 5), (5, 3)),
    ((2, 4, 5), (5, 3)),
    ((2, 4, 5), (2, 5, 3)),
    ((1, 4, 5), (2, 5, 3)),
    ((2, 4, 5), (1, 5, 3)),
    ((5,), (2, 5, 3)),
    ((2, 4, 5), (5,)),
    ((2, 3, 4, 5), (2, 3, 5, 3)),
    ((4, 1, 4, 5), (3, 5, 3)),
]


# ------------------------------------------------------------------------------------
# Tier 1: the operand contract itself -- shapes and values against torch.matmul
# under a format wide enough that only the reduction order differs.


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("a_shape,b_shape", RANK_CASES)
def test_tier1_matches_torch_matmul(device, a_shape, b_shape):
    a = torch.randn(*a_shape, device=device)
    b = torch.randn(*b_shape, device=device)
    ref = a @ b
    out = binaryK_matmul(a, b, **IDENTITY_BK)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (False, True), (True, False)])
def test_tier1_transpose_flags_on_3d(device, trans_a, trans_b):
    a_shape = (3, 5, 4) if trans_a else (3, 4, 5)
    b_shape = (3, 6, 5) if trans_b else (3, 5, 6)
    a = torch.randn(*a_shape, device=device)
    b = torch.randn(*b_shape, device=device)
    ref = (a.mT if trans_a else a) @ (b.mT if trans_b else b)
    out = binaryK_matmul(a, b, trans_a=trans_a, trans_b=trans_b, **IDENTITY_BK)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------------------------
# Tier 2: real formats, within a tolerance appropriate to the storage dtype.


def _cos(out, ref):
    return torch.nn.functional.cosine_similarity(
        out.flatten().float(), ref.flatten().float(), dim=0
    ).item()


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_tier2_batched_statistical(device, dtype):
    a = torch.randn(3, 16, 64, device=device, dtype=dtype)
    b = torch.randn(3, 12, 64, device=device, dtype=dtype)
    out = binaryK_matmul(a, b, trans_b=True, **BK)
    assert torch.isfinite(out).all()
    assert _cos(out, a @ b.mT) > 0.9


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_tier2_superfp_batched_statistical(device, dtype):
    a = torch.randn(3, 16, 64, device=device, dtype=dtype)
    b = torch.randn(3, 12, 64, device=device, dtype=dtype)
    out = superfp_matmul(a, b, trans_b=True, **SFP)
    assert torch.isfinite(out).all()
    assert _cos(out, a @ b.mT) > 0.9


# ------------------------------------------------------------------------------------
# Tier 3: the batched op is a loop of 2D calls, bit for bit.
#
# This is the gate the whole feature rests on. `torch.equal` and not a
# tolerance: batch element b of a batched call runs exactly the K-reduction
# the 2D call on that element runs, in the same order, so anything else is a
# bug rather than an accumulation difference.


def _loop(fn, a, b, n):
    return torch.stack([fn(a[i], b[i]) for i in range(n)])


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_tier3_binaryK_batched_equals_loop(device, round_mode):
    a = torch.randn(4, 33, 40, device=device)
    b = torch.randn(4, 40, 17, device=device)
    fmt: dict[str, Any] = dict(rounding_mode=round_mode, **BK)
    got = binaryK_matmul(a, b, **fmt)
    assert torch.equal(got, _loop(lambda x, y: binaryK_matmul(x, y, **fmt), a, b, 4))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_tier3_superfp_batched_equals_loop(device, round_mode):
    a = torch.randn(4, 33, 40, device=device)
    b = torch.randn(4, 40, 17, device=device)
    fmt: dict[str, Any] = dict(rounding_mode=round_mode, **SFP)
    got = superfp_matmul(a, b, **fmt)
    assert torch.equal(got, _loop(lambda x, y: superfp_matmul(x, y, **fmt), a, b, 4))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("round_mode", DETERMINISTIC_ROUND_MODES)
def test_tier3_fma_batched_equals_loop(device, round_mode):
    a = torch.randn(4, 33, 40, device=device)
    b = torch.randn(4, 40, 17, device=device)
    fmt: dict[str, Any] = dict(rounding_mode=round_mode, fma_K=8, fma_P=4)
    got = binaryK_matmul_fma(a, b, **fmt)
    assert torch.equal(got, _loop(lambda x, y: binaryK_matmul_fma(x, y, **fmt), a, b, 4))


@pytest.mark.parametrize("device", available_devices)
def test_tier3_shared_operand_equals_loop(device):
    """A 2D operand rides at stride 0 -- the same values, never expanded."""
    a = torch.randn(4, 33, 40, device=device)
    w = torch.randn(40, 17, device=device)
    got = binaryK_matmul(a, w, **BK)
    ref = torch.stack([binaryK_matmul(a[i], w, **BK) for i in range(4)])
    assert torch.equal(got, ref)


@pytest.mark.parametrize("device", available_devices)
def test_tier3_fold_into_m_equals_loop(device):
    """`[..., M, K] @ [K, N]` folds its batch into M, and that is a view.

    Bit-identical rather than merely equivalent: an element's SR subsequence
    is `(b*M + row)*N + col` batched and `row' * N + col` folded, with
    `row' = b*M + row`, i.e. the same index.
    """
    a = torch.randn(5, 12, 40, device=device)
    w = torch.randn(40, 17, device=device)
    got = binaryK_matmul(a, w, **BK)
    ref = torch.stack([binaryK_matmul(a[i], w, **BK) for i in range(5)])
    assert torch.equal(got, ref)


@pytest.mark.parametrize("device", available_devices)
def test_tier3_partial_broadcast_equals_loop(device):
    a = torch.randn(3, 1, 8, 10, device=device)
    b = torch.randn(2, 10, 6, device=device)
    got = binaryK_matmul(a, b, **BK)
    assert got.shape == (3, 2, 8, 6)
    for i in range(3):
        for j in range(2):
            assert torch.equal(got[i, j], binaryK_matmul(a[i, 0], b[j], **BK))


@pytest.mark.parametrize("device", available_devices)
def test_tier3_1d_promotion_equals_2d_call(device):
    """The promoted dimension is dropped from the result, not from the math."""
    v = torch.randn(10, device=device)
    m = torch.randn(10, 6, device=device)
    assert torch.equal(
        binaryK_matmul(v, m, **BK), binaryK_matmul(v.unsqueeze(0), m, **BK).squeeze(0)
    )
    m2 = torch.randn(6, 10, device=device)
    assert torch.equal(
        binaryK_matmul(m2, v, **BK), binaryK_matmul(m2, v.unsqueeze(-1), **BK).squeeze(-1)
    )


# --- stochastic rounding -----------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_sr_batch_element_zero_matches_2d_call(device):
    """Batch element 0 of a batched call is the 2D call, under one seed.

    The SR stream is keyed by the output element's global linear index into
    [batch, M, N], so element (0, i, j) draws what the 2D call's (i, j) drew.
    """
    a = torch.randn(3, 17, 20, device=device)
    b = torch.randn(3, 20, 11, device=device)
    fmt: dict[str, Any] = dict(rounding_mode=RoundMode.SR, mul_prng_bits=5, acc_prng_bits=5, **BK)
    torch.manual_seed(1234)
    batched = binaryK_matmul(a, b, **fmt)
    torch.manual_seed(1234)
    single = binaryK_matmul(a[0], b[0], **fmt)
    assert torch.equal(batched[0], single)


@pytest.mark.parametrize("device", available_devices)
def test_sr_every_batch_element_draws_its_own_stream(device):
    """No two batch elements share a subsequence.

    Two identical batch elements under one seed must not produce identical
    output: if they did, the stream would be keyed on (row, col) alone and the
    batch would be correlated noise.
    """
    a = torch.randn(1, 24, 32, device=device).expand(2, 24, 32).contiguous()
    b = torch.randn(1, 32, 24, device=device).expand(2, 32, 24).contiguous()
    torch.manual_seed(7)
    out = binaryK_matmul(a, b, rounding_mode=RoundMode.SR, mul_prng_bits=5, **BK)
    assert not torch.equal(out[0], out[1])
    # ... but they are the same computation under different noise, so the two
    # still agree in the only sense a stochastic format promises.
    assert _cos(out[0], out[1]) > 0.99


# --- the mixed (palette) ops -------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_mixed_shared_map_equals_loop(device):
    """A 2D map is shared across the batch -- the palette is a property of the
    layer, not of the sample."""
    a = torch.randn(3, 20, 24, device=device)
    b = torch.randn(3, 24, 16, device=device)
    idx = torch.randint(0, 2, (20, 16), device=device)
    got = binaryK_matmul_mixed(a, b, idx, **PALETTE)
    ref = torch.stack([binaryK_matmul_mixed(a[i], b[i], idx, **PALETTE) for i in range(3)])
    assert torch.equal(got, ref)


@pytest.mark.parametrize("device", available_devices)
def test_mixed_per_batch_map_equals_loop(device):
    a = torch.randn(3, 20, 24, device=device)
    b = torch.randn(3, 24, 16, device=device)
    idx = torch.randint(0, 2, (3, 20, 16), device=device)
    got = binaryK_matmul_mixed(a, b, idx, **PALETTE)
    ref = torch.stack([binaryK_matmul_mixed(a[i], b[i], idx[i], **PALETTE) for i in range(3)])
    assert torch.equal(got, ref)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("shape", [(20, 16), (20, 1), (1, 16), (3, 20, 16), (1, 20, 16)])
def test_mixed_map_broadcast_shapes(device, shape):
    a = torch.randn(3, 20, 24, device=device)
    b = torch.randn(3, 24, 16, device=device)
    idx = torch.randint(0, 2, shape, device=device)
    assert binaryK_matmul_mixed(a, b, idx, **PALETTE).shape == (3, 20, 16)


@pytest.mark.parametrize("device", available_devices)
def test_mixed_int64_map_is_accepted_batched(device):
    """The G5b regression, on the batched path: an int64 map is what
    torch.randint hands back without an explicit dtype."""
    a = torch.randn(2, 20, 24, device=device)
    b = torch.randn(2, 24, 16, device=device)
    idx64 = torch.randint(0, 2, (2, 20, 16), device=device, dtype=torch.int64)
    idx32 = idx64.to(torch.int32)
    assert torch.equal(
        binaryK_matmul_mixed(a, b, idx64, **PALETTE),
        binaryK_matmul_mixed(a, b, idx32, **PALETTE),
    )


@pytest.mark.parametrize("device", available_devices)
def test_mixed_3d_by_2d_keeps_its_batch(device):
    """The fold into `M` is off for a palette op.

    `prec_idx` is indexed by output element, so a map shaped for the [M, N]
    output of a batched call does not describe the [B*M, N] output of a folded
    one -- the fold would turn a valid call into a shape error.
    """
    a = torch.randn(3, 20, 24, device=device)
    b = torch.randn(24, 16, device=device)
    idx = torch.randint(0, 2, (20, 16), device=device)
    got = binaryK_matmul_mixed(a, b, idx, **PALETTE)
    ref = torch.stack([binaryK_matmul_mixed(a[i], b, idx, **PALETTE) for i in range(3)])
    assert torch.equal(got, ref)


@pytest.mark.parametrize("device", available_devices)
def test_mixed_rank4_map_follows_the_operands(device):
    """A map's leading dims collapse the way the operands' do."""
    a = torch.randn(2, 3, 12, 16, device=device)
    b = torch.randn(2, 3, 16, 10, device=device)
    idx = torch.randint(0, 2, (2, 3, 12, 10), device=device)
    got = binaryK_matmul_mixed(a, b, idx, **PALETTE)
    assert got.shape == (2, 3, 12, 10)
    for i in range(2):
        for j in range(3):
            assert torch.equal(
                got[i, j], binaryK_matmul_mixed(a[i, j], b[i, j], idx[i, j], **PALETTE)
            )


@pytest.mark.parametrize("device", available_devices)
def test_mixed_rejects_wrong_batch_map(device):
    a = torch.randn(3, 20, 24, device=device)
    b = torch.randn(3, 24, 16, device=device)
    idx = torch.randint(0, 2, (2, 20, 16), device=device)
    with pytest.raises(RuntimeError, match="prec_idx batch dimension"):
        binaryK_matmul_mixed(a, b, idx, **PALETTE)


@pytest.mark.parametrize("device", available_devices)
def test_mixed_fma_batched_equals_loop(device):
    a = torch.randn(3, 20, 24, device=device)
    b = torch.randn(3, 24, 16, device=device)
    idx = torch.randint(0, 2, (3, 20, 16), device=device)
    pal: dict[str, Any] = dict(fma_K=[8, 6], fma_P=[4, 3])
    got = binaryK_matmul_fma_mixed(a, b, idx, **pal)
    ref = torch.stack([binaryK_matmul_fma_mixed(a[i], b[i], idx[i], **pal) for i in range(3)])
    assert torch.equal(got, ref)


@pytest.mark.parametrize("device", available_devices)
def test_mixed_superfp_batched_equals_loop(device):
    a = torch.randn(3, 20, 24, device=device)
    b = torch.randn(3, 24, 16, device=device)
    idx = torch.randint(0, 2, (3, 20, 16), device=device)
    pal: dict[str, Any] = dict(
        mul_man_bits=[3, 2], mul_exp_bits=[4, 4], mul_normal_binades=[8, 8], mul_bias=[7, 7]
    )
    got = superfp_matmul_mixed(a, b, idx, **pal)
    ref = torch.stack([superfp_matmul_mixed(a[i], b[i], idx[i], **pal) for i in range(3)])
    assert torch.equal(got, ref)


# --- the two zero-copy paths -------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("shape", [(2, 12, 8), (12, 8)])
def test_transposed_view_is_values_preserving(device, shape):
    """`q @ k.mT` sets the kernel's flag instead of materializing the operand.

    A values-preserving change or a bug: the kernel reads op(b) either way,
    and SR indexes by output element rather than by operand layout.
    """
    q = torch.randn(*shape, device=device)
    k = torch.randn(*shape, device=device)
    assert torch.equal(binaryK_matmul(q, k.mT, **BK), binaryK_matmul(q, k.mT.contiguous(), **BK))


@pytest.mark.parametrize("device", available_devices)
def test_transposed_view_does_not_allocate(device):
    """...and it does not copy, which is the point of doing it at all."""
    k = torch.randn(2, 12, 8, device=device)
    q = torch.randn(2, 12, 8, device=device)
    view = k.mT
    before = view.data_ptr()
    binaryK_matmul(q, view, **BK)
    assert view.data_ptr() == before == k.data_ptr()


@pytest.mark.parametrize("device", available_devices)
def test_fold_into_m_survives_sr(device):
    """The fold needs no gate of its own: it is the same SR index."""
    a = torch.randn(4, 9, 20, device=device)
    w = torch.randn(20, 11, device=device)
    fmt: dict[str, Any] = dict(rounding_mode=RoundMode.SR, mul_prng_bits=5, **BK)
    torch.manual_seed(99)
    folded = binaryK_matmul(a, w, **fmt)
    torch.manual_seed(99)
    flat = binaryK_matmul(a.reshape(-1, 20), w, **fmt)
    assert torch.equal(folded.reshape(-1, 11), flat)


# --- degenerate shapes -------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("shape", [(0, 4, 5), (2, 0, 5), (2, 4, 5)])
def test_empty_batch_or_rows(device, shape):
    """An empty batch, and an empty M, both return early rather than launch."""
    a = torch.randn(*shape, device=device)
    b = torch.randn(5, 3, device=device)
    out = binaryK_matmul(a, b, **BK)
    assert out.shape == (shape[0], shape[1], 3)
    if out.numel():
        assert torch.equal(out, torch.stack([binaryK_matmul(x, b, **BK) for x in a]))


@pytest.mark.parametrize("device", available_devices)
def test_large_batch_is_chunked(device):
    """More batch elements than the 65,535 grid-z limit, on a tiny GEMM."""
    n = 70_000
    a = torch.randn(n, 1, 2, device=device)
    b = torch.randn(n, 2, 1, device=device)
    out = binaryK_matmul(a, b, **IDENTITY_BK)
    assert out.shape == (n, 1, 1)
    assert torch.allclose(out, a @ b, atol=1e-5, rtol=1e-5)
