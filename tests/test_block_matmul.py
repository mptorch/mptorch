"""The block GEMM, custom_matmul_block (mptorch.quant.block_matmul).

Tier 1 holds it to ``torch.matmul`` of the unpacked operands. Tier 3 holds it,
``torch.equal``, to the flat binaryK GEMM over the unpacked operands: the block
GEMM decodes in its tile loads what block_unpack decodes, and its split mac is
the binaryK split mac with a binary32 multiply (``mul_K=32, mul_P=24,
mul_bias=127``, which is the identity on every finite product here), its fused
mac the binaryK fused one, and its accumulate algorithms the ``*_accumulated``
ops'. So every deterministic mode, both macs, every accumulator, every
pairing of formats and every orientation of the operands is checked against
code that shares nothing with the block path but the arithmetic policies.
Under stochastic rounding the block GEMM draws once per step where the flat op
draws twice, so SR is held to its properties instead: reproducible, keyed on
the output element, thread-count independent, and unbiased.
"""

import dataclasses

import pytest
import torch

from mptorch import (
    E2M1,
    E8M0,
    MXFP4_E2M1,
    MXFP6_E2M3,
    MXFP6_E3M2,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    RoundMode,
    ScaleRounding,
    SuperFP,
)
from mptorch.quant import (
    BlockMac,
    BlockPacked,
    QLinear,
    binaryK_matmul,
    binaryK_matmul_fma,
    block_gemm_formats,
    block_matmul,
    block_pack,
    qmatmul,
)

from .markers import float64_devices, requires_mps

devices = float64_devices
DETERMINISTIC = [m for m in RoundMode if m is not RoundMode.SR]
SQ_NVFP4 = dataclasses.replace(NVFP4, block_rows=16)
SQ_MXFP8 = dataclasses.replace(MXFP8_E4M3, block_rows=32)

# superfp elements and scales (tests/test_block_pack.py has their layouts)
SFP4 = BlockFormat(SuperFP(1, 2, 1, 1), E8M0, 32)
SFP8 = BlockFormat(SuperFP(3, 4, 2, 7), E8M0, 32, elem_max=448.0, nan_code=0x7F)
SQ_SFP6 = BlockFormat(SuperFP(2, 3, 2, 3), SuperFP(3, 4, 2, 7), 16, block_rows=16)

PAIRS = {
    "mxfp8": (MXFP8_E4M3, MXFP8_E4M3),
    "mxfp4_mxfp8": (MXFP4_E2M1, MXFP8_E4M3),
    "nvfp4_mxfp6": (NVFP4, MXFP6_E2M3),
    "1d_2d": (MXFP6_E3M2, SQ_NVFP4),
    "e5m2_nvfp4": (MXFP8_E5M2, NVFP4),
    "sfp8_sfp4": (SFP8, SFP4),
    "sfp_scales": (BlockFormat(E2M1, SuperFP(3, 4, 2, 7), 16), SQ_SFP6),
    "mxfp8_sfp_scale": (MXFP8_E4M3, BlockFormat(E2M1, SuperFP(3, 4, 2, 7), 16)),
    # scales chosen by rules other than their defaults: the GEMM reads the codes
    "scale_rules": (
        dataclasses.replace(MXFP4_E2M1, scale_rounding=ScaleRounding.UP),
        dataclasses.replace(NVFP4, scale_rounding=ScaleRounding.SELECTIVE),
    ),
}
ACCS = {"b8p4": BinaryK(8, 4), "b16p11": BinaryK(16, 11), "none": None}


def _ops(M, K, N, fa, fb, device, seed=0, ta=False, tb=False, batch=()):
    """Packed operands of a logically [M, K] @ [K, N] product, A packed along
    K (along M when ``ta``) and B along K (along N when ``tb``)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(*batch, M, K, generator=g).to(device)
    b = torch.randn(*batch, K, N, generator=g).to(device)
    pa = block_pack(a, fa, -2 if ta else -1)
    pb = block_pack(b, fb, -1 if tb else -2)
    return pa, pb


def _flat(pa, pb, mode, fused, acc, **accumulation):
    """The flat binaryK GEMM over the unpacked operands, with the block GEMM's
    arithmetic."""
    a, b = pa.unpack(), pb.unpack()
    if fused:
        if acc is None:
            return binaryK_matmul_fma(
                a, b, fma_K=8, fma_P=4, fma_quant=False, rounding_mode=mode, **accumulation
            )
        return binaryK_matmul_fma(
            a, b, fma_K=acc.K, fma_P=acc.P, fma_bias=acc.bias, rounding_mode=mode, **accumulation
        )
    kw: dict = dict(mul_K=32, mul_P=24, mul_bias=127, rounding_mode=mode)
    if acc is None:
        kw["accumulate_quant"] = False
    else:
        kw.update(acc_K=acc.K, acc_P=acc.P, acc_bias=acc.bias)
    return binaryK_matmul(a, b, **kw, **accumulation)


# --- the tensor scales in the loop or after it ----------------------------------------


def _unit(p: BlockPacked) -> BlockPacked:
    """``p`` with a tensor scale of 1: its elements times their block scales only."""
    return BlockPacked(p.data, p.scales, p.fmt, p.shape, p.axis, 1.0, p.dtype)


def _alpha(pa: BlockPacked, pb: BlockPacked) -> float:
    """The float32 product of the two tensor scales."""
    return torch.tensor(pa.tensor_scale * pb.tensor_scale, dtype=torch.float32).item()


def _scaled_ops(fa, fb, device, ta, tb):
    """Operands packed with the given tensor scales (for the formats that have one)."""
    g = torch.Generator().manual_seed(5)
    a = torch.randn(19, 70, generator=g).to(device)
    b = torch.randn(70, 23, generator=g).to(device)
    pa = block_pack(a, fa, tensor_scale=ta if fa.has_tensor_scale else None)
    pb = block_pack(b, fb, 0, tensor_scale=tb if fb.has_tensor_scale else None)
    return pa, pb


SCALED_PAIRS = {
    "nvfp4": (NVFP4, NVFP4),
    "mxfp8_nvfp4": (MXFP8_E4M3, NVFP4),
    "sfp_scales": PAIRS["sfp_scales"],
}


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", list(RoundMode))
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("pair", list(SCALED_PAIRS))
def test_tensor_scale_epilogue(device, mode, fused, pair):
    """``tensor_scale_epilogue=True`` is the GEMM of the operands decoded with
    their block scales alone, then one float32 multiply by the float32 product
    of the two tensor scales: the unit-scaled block GEMM times alpha, random
    draws included, and so the flat GEMM on the unit-scaled operands times
    alpha."""
    fa, fb = SCALED_PAIRS[pair]
    pa, pb = _scaled_ops(fa, fb, device, 0.3, 0.7)
    acc = BinaryK(12, 7, prng_bits=4)
    torch.manual_seed(1)
    got = block_matmul(pa, pb, acc=acc, fused=fused, rounding=mode, tensor_scale_epilogue=True)
    torch.manual_seed(1)
    unit = block_matmul(_unit(pa), _unit(pb), acc=acc, fused=fused, rounding=mode)
    alpha = _alpha(pa, pb)
    assert alpha != 1.0
    assert torch.equal(got, unit * alpha)
    if mode is not RoundMode.SR:
        assert torch.equal(got, _flat(_unit(pa), _unit(pb), mode, fused, acc) * alpha)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("fused", [False, True])
def test_power_of_two_tensor_scales_commute_with_the_loop(device, fused):
    """With power-of-two tensor scales and a binary32 sum, the two options are
    the same GEMM bit for bit: every decode, product and sum is scaled by a
    power of two, which commutes with rounding in binary32's normal range."""
    pa, pb = _scaled_ops(NVFP4, NVFP4, device, 0.25, 2.0**-7)
    loop = block_matmul(pa, pb, fused=fused)
    assert torch.equal(loop, block_matmul(pa, pb, fused=fused, tensor_scale_epilogue=True))


@pytest.mark.parametrize("device", devices)
def test_tensor_scale_epilogue_leaves_power_of_two_scales_alone(device):
    """A format without a tensor scale (E8M0) has nothing to move: the option
    changes no bit."""
    pa, pb = _ops(19, 70, 23, MXFP8_E4M3, SFP8, device)
    acc = BinaryK(12, 7)
    assert torch.equal(
        block_matmul(pa, pb, acc=acc), block_matmul(pa, pb, acc=acc, tensor_scale_epilogue=True)
    )


@pytest.mark.parametrize("device", devices)
def test_tensor_scale_epilogue_through_qmatmul_and_a_layer(device):
    """The option rides on a BlockMac and on block_gemm_formats: qmatmul's
    forward is block_matmul's with it, and a layer trains with it."""
    g = torch.Generator().manual_seed(2)
    a = torch.randn(8, 64, generator=g).to(device).requires_grad_()
    b = torch.randn(64, 16, generator=g).to(device)
    mac = BlockMac(NVFP4, tensor_scale_epilogue=True)
    got = qmatmul(a, b, mac)
    want = block_matmul(
        block_pack(a.detach(), NVFP4), block_pack(b, NVFP4, 0), tensor_scale_epilogue=True
    )
    assert torch.equal(got.detach(), want)
    got.sum().backward()
    assert a.grad is not None and bool(torch.isfinite(a.grad).all())
    layer = QLinear(64, 16, formats=block_gemm_formats(NVFP4, NVFP4, tensor_scale_epilogue=True))
    layer = layer.to(device)
    x = a.detach()
    y = layer(x)
    w = layer.weight.detach()
    manual = (
        block_matmul(block_pack(x, NVFP4), block_pack(w, NVFP4, 1).mT, tensor_scale_epilogue=True)
        + layer.bias.detach()
    )
    assert torch.equal(y.detach(), manual)
    y.sum().backward()
    assert layer.weight.grad is not None


# --- Tier 1 ------------------------------------------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize(
    "M,K,N", [(1, 1, 1), (17, 33, 9), (64, 128, 48), (5, 300, 7), (40, 64, 70)]
)
def test_matches_torch_matmul_of_the_unpacked_operands(device, M, K, N):
    pa, pb = _ops(M, K, N, MXFP8_E4M3, NVFP4, device)
    want = pa.unpack().cpu().double() @ pb.unpack().cpu().double()
    got = block_matmul(pa, pb)
    assert got.dtype == torch.float32 and got.shape == (M, N)
    torch.testing.assert_close(got.cpu().double(), want, rtol=1e-5, atol=1e-5)


# --- Tier 3 ------------------------------------------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", DETERMINISTIC)
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("acc", list(ACCS))
@pytest.mark.parametrize("pair", list(PAIRS))
def test_equals_the_flat_gemm(device, mode, fused, acc, pair):
    fa, fb = PAIRS[pair]
    pa, pb = _ops(19, 70, 23, fa, fb, device, seed=len(pair))
    got = block_matmul(pa, pb, acc=ACCS[acc], fused=fused, rounding=mode)
    assert torch.equal(got, _flat(pa, pb, mode, fused, ACCS[acc]))


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RZ, RoundMode.RO])
@pytest.mark.parametrize(
    "alg,block_size,outer,fused",
    [
        (AccumulateAlgorithm.KAHAN, None, None, False),
        (AccumulateAlgorithm.KAHAN, None, None, True),
        (AccumulateAlgorithm.BLOCK, 8, BinaryK(16, 11), False),
        (AccumulateAlgorithm.BLOCK, 32, None, True),
        (AccumulateAlgorithm.TREE, 16, BinaryK(16, 11), False),
        (AccumulateAlgorithm.TREE, 4, None, False),
    ],
)
def test_accumulate_algorithms_equal_the_flat_gemm(device, mode, alg, block_size, outer, fused):
    pa, pb = _ops(9, 75, 11, MXFP8_E4M3, MXFP4_E2M1, device, seed=4)
    acc = BinaryK(8, 4)
    got = block_matmul(
        pa,
        pb,
        acc=acc,
        fused=fused,
        rounding=mode,
        accumulate_algorithm=alg,
        block_size=block_size,
        outer=outer,
    )
    accumulation: dict = dict(accumulate_algorithm=alg, block_size=block_size)
    if outer is not None:
        accumulation.update(outer_K=outer.K, outer_P=outer.P)
    assert torch.equal(got, _flat(pa, pb, mode, fused, acc, **accumulation))


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("ta", [False, True])
@pytest.mark.parametrize("tb", [False, True])
@pytest.mark.parametrize(
    "fa,fb",
    [(MXFP8_E4M3, MXFP4_E2M1), (SQ_NVFP4, SQ_MXFP8), (SFP8, SQ_SFP6)],
    ids=["1d", "square", "superfp"],
)
def test_every_orientation(device, ta, tb, fa, fb):
    pa, pb = _ops(21, 96, 13, fa, fb, device, seed=6, ta=ta, tb=tb)
    got = block_matmul(pa, pb, acc=BinaryK(16, 11))
    assert torch.equal(got, _flat(pa, pb, RoundMode.RNE, False, BinaryK(16, 11)))


@pytest.mark.parametrize("device", devices)
def test_mT_reads_the_same_codes(device):
    x = torch.randn(8, 64, device=device)
    w = block_pack(torch.randn(16, 64, device=device), SQ_NVFP4)
    wt = w.mT
    assert wt.data.data_ptr() == w.data.data_ptr() and wt.shape == (64, 16) and wt.axis == 0
    y = block_matmul(block_pack(x, NVFP4), wt)
    assert torch.equal(y, _flat(block_pack(x, NVFP4), wt, RoundMode.RNE, False, None))
    # and the input gradient's orientation: w read along its output features
    g = block_pack(torch.randn(8, 16, device=device), NVFP4)
    assert torch.equal(block_matmul(g, w), _flat(g, w, RoundMode.RNE, False, None))
    assert torch.equal(block_matmul(g, w), block_matmul(g, block_pack(w.unpack(), SQ_NVFP4, 0)))


@pytest.mark.parametrize("device", devices)
def test_batches(device):
    pa, pb = _ops(7, 64, 5, MXFP8_E4M3, MXFP8_E4M3, device, batch=(3,))
    got = block_matmul(pa, pb, acc=BinaryK(16, 11))
    assert got.shape == (3, 7, 5)
    assert torch.equal(got, _flat(pa, pb, RoundMode.RNE, False, BinaryK(16, 11)))
    # a shared right operand, broadcast over the batch
    w = block_pack(torch.randn(64, 5, device=device), MXFP8_E4M3, 0)
    got = block_matmul(pa, w)
    assert torch.equal(got, torch.stack([block_matmul(_first(pa, i), w) for i in range(3)]))
    # rank 4
    pa4, pb4 = _ops(4, 32, 3, MXFP4_E2M1, MXFP4_E2M1, device, batch=(2, 3))
    assert block_matmul(pa4, pb4).shape == (2, 3, 4, 3)


def _first(p: BlockPacked, i: int) -> BlockPacked:
    """Batch element ``i`` of a rank-3 packed operand."""
    return BlockPacked(
        p.data[i], p.scales[i], p.fmt, p.shape[1:], p.axis - 1, p.tensor_scale, p.dtype
    )


# --- stochastic rounding --------------------------------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("fused", [False, True])
def test_sr_properties(device, fused):
    acc = BinaryK(12, 7, prng_bits=10)
    pa, pb = _ops(6, 100, 5, MXFP8_E4M3, MXFP8_E4M3, device, batch=(2,))
    torch.manual_seed(1)
    y = block_matmul(pa, pb, acc=acc, fused=fused, rounding=RoundMode.SR)
    torch.manual_seed(1)
    assert torch.equal(y, block_matmul(pa, pb, acc=acc, fused=fused, rounding=RoundMode.SR))
    # batch element 0 draws what the 2D call draws (the key is the output index)
    torch.manual_seed(1)
    y0 = block_matmul(_first(pa, 0), _first(pb, 0), acc=acc, fused=fused, rounding=RoundMode.SR)
    assert torch.equal(y[0], y0)


def test_sr_thread_count_independent():
    acc = BinaryK(12, 7, prng_bits=10)
    pa, pb = _ops(70, 100, 50, MXFP8_E4M3, MXFP4_E2M1, "cpu")
    n = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(2)
        one = block_matmul(pa, pb, acc=acc, rounding=RoundMode.SR)
        torch.set_num_threads(max(n, 4))
        torch.manual_seed(2)
        many = block_matmul(pa, pb, acc=acc, rounding=RoundMode.SR)
    finally:
        torch.set_num_threads(n)
    assert torch.equal(one, many)


@pytest.mark.parametrize("device", devices)
def test_sr_unbiased(device):
    acc = BinaryK(8, 4, prng_bits=12)
    pa, pb = _ops(4, 64, 4, MXFP8_E4M3, MXFP8_E4M3, device)
    exact = pa.unpack().cpu().double() @ pb.unpack().cpu().double()
    mean = torch.stack(
        [block_matmul(pa, pb, acc=acc, rounding=RoundMode.SR).cpu().double() for _ in range(400)]
    ).mean(0)
    rne = block_matmul(pa, pb, acc=acc).cpu().double()
    assert (mean - exact).abs().mean() < 0.5 * (rne - exact).abs().mean() + 1e-3


# --- what is refused -------------------------------------------------------------


@pytest.mark.parametrize("device", devices)
def test_validation(device):
    pa, pb = _ops(4, 64, 4, MXFP8_E4M3, MXFP8_E4M3, device)
    pc = block_pack(torch.randn(32, 4, device=device), MXFP8_E4M3, 0)
    with pytest.raises(RuntimeError, match="inner dimensions"):
        block_matmul(pa, pc)
    with pytest.raises(ValueError, match="leading dimensions"):
        block_matmul(
            _ops(4, 64, 4, MXFP8_E4M3, MXFP8_E4M3, device, batch=(2,))[0],
            _ops(4, 64, 4, MXFP8_E4M3, MXFP8_E4M3, device, batch=(3,))[1],
        )
    bad = BlockPacked(pa.data[:, :-1], pa.scales, pa.fmt, pa.shape, pa.axis, 1.0, torch.float32)
    with pytest.raises(RuntimeError, match="pack into"):
        block_matmul(bad, pb)
    with pytest.raises(ValueError, match="phase G"):
        block_matmul(pa, pb, carrier=torch.float64)
    with pytest.raises(ValueError, match="sums products pairwise"):
        block_matmul(pa, pb, fused=True, accumulate_algorithm=AccumulateAlgorithm.TREE)
    with pytest.raises(ValueError, match="packed along one of its last two"):
        block_matmul(block_pack(torch.randn(2, 4, 64, device=device), MXFP8_E4M3, 0), pb)


def test_raw_op_refuses_non_uint8_and_grad():
    pa, pb = _ops(4, 64, 4, MXFP8_E4M3, MXFP8_E4M3, "cpu")
    from mptorch.quant.block import _block_gemm_spec, _format_args

    spec = _block_gemm_spec(None, False, RoundMode.RNE, AccumulateAlgorithm.NAIVE, None, None, None)
    args = (
        pa.data.float(),
        pa.scales,
        64,
        False,
        pb.data,
        pb.scales,
        64,
        True,
        *_format_args(pa),
        *_format_args(pb),
        *spec.args,
    )
    with pytest.raises(RuntimeError, match="uint8"):
        torch.ops.mptorch.custom_matmul_block.default(*args)


@requires_mps
def test_mps_raises():
    pa, pb = _ops(4, 64, 4, MXFP8_E4M3, MXFP8_E4M3, "cpu")
    with pytest.raises(RuntimeError, match="phase H"):
        block_matmul(pa.to("mps"), pb.to("mps"))


@pytest.mark.parametrize("device", devices)
def test_nvfp4_split_is_not_fused(device):
    """NVFP4's decoded values carry the E4M3 scale times a float32 tensor
    scale, so their products are not exact in binary32 and the split mac's
    rounded product differs from the fused mac's exact one. Were the split
    step contracted into an FMA (IdentityMultiplier rounds with rn_mul so
    that nvcc cannot), the two would agree: this is the case that shows it."""
    pa, pb = _ops(16, 256, 16, NVFP4, NVFP4, device, seed=12)
    split = block_matmul(pa, pb, acc=None)
    fused = block_matmul(pa, pb, acc=None, fused=True)
    assert not torch.equal(split, fused)
    assert torch.equal(split, _flat(pa, pb, RoundMode.RNE, False, None))


def test_wide_element_codes_are_refused():
    from mptorch import E8M0, BlockFormat

    fmt = BlockFormat(BinaryK(11, 6, bias=7), E8M0, 8)
    pa, pb = _ops(4, 32, 4, fmt, MXFP8_E4M3, "cpu")
    with pytest.raises(RuntimeError, match="at most 8"):
        block_matmul(pa, pb)
