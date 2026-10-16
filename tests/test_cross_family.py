"""The cross-family split mac: binaryK products with superfp sums, and the reverse.

``csrc/common/gemm_cross.h`` composes the policies the single-family split
macs use, a multiplier of one family and an adder of the other, under every
accumulate algorithm. The references are ``test_accumulators``': plain Python
over ``[M, N]`` tensors, one K-step at a time, with the elementwise quantizers
standing in for the roundings, so that ``s = acc(s + mul(a*b))`` is the
statement checked. Tier 3 holds the CPU, CUDA and MPS kernels to them with
``torch.equal`` in every deterministic rounding mode and for each algorithm;
the two API tiers (the flat wrappers and ``SplitMac``) are held to each other,
the layer factories to a hand-written baseline, and stochastic rounding to the
bracket the directed modes put around it. The conv ops are in
``test_qconv_gemm``. float64 operands run the binary64 kernels and are held to
the same references in float64. An MPS call is also held to the CPU op's
words, SR included.
"""

from typing import Any

import pytest
import torch

from mptorch import AccumulateAlgorithm, BinaryK, RoundMode, SaturationMode, SuperFP
from mptorch.quant import (
    QLinear,
    SplitMac,
    binaryK_matmul,
    binaryK_superfp_gemm_formats,
    binaryK_superfp_matmul,
    qmatmul,
    qmm,
    superfp_binaryK_gemm_formats,
    superfp_binaryK_matmul,
)
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops import _narrowed

from .markers import available_devices, float64_devices, requires_mps
from .test_accumulators import (
    BLOCK_SIZES,
    DETERMINISTIC,
    KS,
    TREE_SIZES,
    _operands,
    _Reference,
)

NAIVE, KAHAN, BLOCK, TREE = (
    AccumulateAlgorithm.NAIVE,
    AccumulateAlgorithm.KAHAN,
    AccumulateAlgorithm.BLOCK,
    AccumulateAlgorithm.TREE,
)

# Every backend; the float64 tests take float64_devices.
kernel_devices = available_devices

# Products in a narrow format, sums in a wider one of the other family, and a
# still wider outer format of the accumulate's family.
B8, B12, B16 = BinaryK(8, 4), BinaryK(12, 7), BinaryK(16, 11)
S8, S12, S16 = SuperFP(3, 4, 8, 7), SuperFP(5, 5, 16, 15), SuperFP(8, 6, 32, 31)
DIRECTIONS: dict[str, tuple[Any, Any, Any]] = {
    "binaryK_superfp": (B8, S12, S16),
    "superfp_binaryK": (S8, B12, B16),
}


def _check(device, direction, algorithm, mode, block_size, with_outer, K, dtype=torch.float32):
    mul, acc, outer = DIRECTIONS[direction]
    outer = outer if with_outer else None
    a, b = _operands(5, K, 3, device, dtype)
    mac = SplitMac(
        mul,
        acc,
        rounding=mode,
        accumulate_algorithm=algorithm,
        block_size=block_size,
        outer=outer,
    )
    out = qmm(a, b, mac)
    carrier = torch.float64 if dtype is torch.float64 else torch.float32
    ref = _Reference(a.cpu().to(carrier), b.cpu().to(carrier), mul, acc, outer, mode).run(
        algorithm, block_size
    )
    assert out.dtype is dtype
    assert torch.equal(out.cpu(), ref.to(dtype)), (
        f"{direction} {algorithm.name} {mode.name} block_size={block_size} outer={outer} K={K}"
    )


# --- tier 1: formats that round nothing ----------------------------------------


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
def test_tier1_exact_formats_match_matmul(device, direction):
    """Small integers whose products and sums both formats hold exactly: the
    product is ``torch.matmul``'s, bit for bit, under every algorithm."""
    g = torch.Generator().manual_seed(5)
    a = torch.randint(-3, 4, (9, 24), generator=g).float().to(device)
    b = torch.randint(-3, 4, (24, 7), generator=g).float().to(device)
    # Binary16p11 and an 11-bit superfp whose normal binades reach down to 1/2
    # hold every integer below 2^11; the products are at most 9, the sums 216.
    bk, sf = BinaryK(16, 11), SuperFP(10, 6, 40, 31)
    mul, acc = (bk, sf) if direction == "binaryK_superfp" else (sf, bk)
    ref = a @ b
    for algorithm, block_size in ((NAIVE, None), (KAHAN, None), (BLOCK, 8), (TREE, 4)):
        mac = SplitMac(mul, acc, accumulate_algorithm=algorithm, block_size=block_size)
        assert torch.equal(qmm(a, b, mac), ref), algorithm.name


# --- tier 3: every deterministic mode against the references -------------------


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_tier3_naive(device, direction, mode):
    for K in KS:
        _check(device, direction, NAIVE, mode, None, False, K)


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_tier3_kahan(device, direction, mode):
    for K in KS:
        _check(device, direction, KAHAN, mode, None, False, K)


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
@pytest.mark.parametrize("with_outer", [False, True], ids=["carrier_total", "outer"])
def test_tier3_block(device, direction, mode, with_outer):
    for block_size in BLOCK_SIZES:
        for K in KS:
            _check(device, direction, BLOCK, mode, block_size, with_outer, K)


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
@pytest.mark.parametrize("with_outer", [False, True], ids=["carrier_total", "outer"])
def test_tier3_tree(device, direction, mode, with_outer):
    for block_size in TREE_SIZES:
        for K in KS:
            _check(device, direction, TREE, mode, block_size, with_outer, K)


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
def test_tier3_tree_long_blocks(device, direction):
    """TREE's levels past a 16-step slab, whole blocks and partial ones."""
    for block_size in (64, 256):
        for K in (255, 256, 257, 300):
            _check(device, direction, TREE, RoundMode.RNE, block_size, True, K)


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("algorithm", [NAIVE, KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_tier3_narrow_storage(device, direction, dtype, algorithm):
    """float16 and bfloat16 operands are loaded into binary32, reduced there
    and stored back with one conversion."""
    block_size = 8 if algorithm in (BLOCK, TREE) else None
    _check(device, direction, algorithm, RoundMode.RNE, block_size, False, 40, dtype)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_tier3_binary64(device, direction, mode):
    """float64 operands run the binary64 kernels, every algorithm."""
    f64 = torch.float64
    for K in (17, 40):
        _check(device, direction, NAIVE, mode, None, False, K, f64)
        _check(device, direction, KAHAN, mode, None, False, K, f64)
        for block_size in (4, 32):
            _check(device, direction, BLOCK, mode, block_size, True, K, f64)
            _check(device, direction, TREE, mode, block_size, False, K, f64)


@pytest.mark.parametrize("device", kernel_devices)
def test_tier3_per_slot_saturation(device):
    """Saturation is per format: a multiply that saturates to its top value
    under a sum that overflows to infinity, and the reverse."""
    a = torch.full((2, 3), 300.0, device=device)
    b = torch.full((3, 2), 300.0, device=device)
    for direction, (mul, acc, _) in DIRECTIONS.items():
        mul_sat = type(mul)(**{**_fields(mul), "saturation": SaturationMode.SAT_FINITE})
        mac = SplitMac(mul_sat, acc)
        ref = _Reference(a.cpu(), b.cpu(), mul_sat, acc, None, RoundMode.RNE).naive()
        assert torch.equal(qmm(a, b, mac).cpu(), ref), direction


def _fields(fmt):
    if isinstance(fmt, BinaryK):
        return dict(K=fmt.K, P=fmt.P, bias=fmt.bias, is_signed=fmt.is_signed)
    return dict(
        man_bits=fmt.man_bits,
        exp_bits=fmt.exp_bits,
        normal_binades=fmt.normal_binades,
        bias=fmt.bias,
        is_signed=fmt.is_signed,
    )


# --- the two API tiers, layouts, and the layer factories ------------------------


def _flat(direction, a, b, *, trans_b=False, algorithm=NAIVE, block_size=None, outer=None):
    mul, acc, _ = DIRECTIONS[direction]
    extra: dict[str, Any] = dict(
        trans_b=trans_b, accumulate_algorithm=algorithm, block_size=block_size
    )
    if direction == "binaryK_superfp":
        if outer is not None:
            extra |= dict(
                outer_man_bits=outer.man_bits,
                outer_exp_bits=outer.exp_bits,
                outer_normal_binades=outer.normal_binades,
                outer_bias=outer.bias,
            )
        return binaryK_superfp_matmul(
            a,
            b,
            mul_K=mul.K,
            mul_P=mul.P,
            acc_man_bits=acc.man_bits,
            acc_exp_bits=acc.exp_bits,
            acc_normal_binades=acc.normal_binades,
            acc_bias=acc.bias,
            **extra,
        )
    if outer is not None:
        extra |= dict(outer_K=outer.K, outer_P=outer.P)
    return superfp_binaryK_matmul(
        a,
        b,
        mul_man_bits=mul.man_bits,
        mul_exp_bits=mul.exp_bits,
        mul_normal_binades=mul.normal_binades,
        mul_bias=mul.bias,
        acc_K=acc.K,
        acc_P=acc.P,
        **extra,
    )


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("algorithm", [NAIVE, KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_flat_wrappers_and_layouts(device, direction, algorithm):
    """The schema tier, a transposed operand and a batch, against the mac tier."""
    mul, acc, outer = DIRECTIONS[direction]
    block_size = 4 if algorithm in (BLOCK, TREE) else None
    outer = outer if block_size else None
    a = torch.randn(3, 6, 21, device=device)
    b = torch.randn(5, 21, device=device)
    mac = SplitMac(mul, acc, accumulate_algorithm=algorithm, block_size=block_size, outer=outer)
    flat = _flat(
        direction, a, b, trans_b=True, algorithm=algorithm, block_size=block_size, outer=outer
    )
    assert torch.equal(flat, qmatmul(a, b.mT, mac))
    for i in range(a.shape[0]):
        assert torch.equal(flat[i], qmm(a[i], b.mT, mac))


def test_the_mac_resolves_to_the_flat_wrappers_spec():
    """``spec_for_mac`` and the flat builders resolve to the same ``_GemmSpec``
    (op, arguments, stored format, findings), defaults included."""
    from mptorch.quant.ops import (
        _binaryK_superfp_accumulated_spec,
        _binaryK_superfp_spec,
        _superfp_binaryK_accumulated_spec,
        _superfp_binaryK_spec,
    )

    for rm in (RoundMode.RNE, RoundMode.SR):
        for sat in (SaturationMode.SAT_NONE, SaturationMode.SAT_FINITE):
            mul = BinaryK(8, 4, saturation=sat, prng_bits=3)
            acc = SuperFP(5, 5, 16, 15, is_signed=True, prng_bits=2)
            assert spec_for_mac(SplitMac(mul, acc, rounding=rm)) == _binaryK_superfp_spec(
                mul_K=8,
                mul_P=4,
                mul_prng_bits=3,
                acc_man_bits=5,
                acc_exp_bits=5,
                acc_normal_binades=16,
                acc_bias=15,
                acc_prng_bits=2,
                rounding_mode=rm,
                saturation_mode=sat,
                acc_saturation_mode=SaturationMode.SAT_NONE,
            )
            mul2 = SuperFP(3, 4, 8, 7, saturation=sat)
            acc2 = BinaryK(12, 7)
            assert spec_for_mac(SplitMac(mul2, acc2, rounding=rm)) == _superfp_binaryK_spec(
                mul_man_bits=3,
                mul_exp_bits=4,
                mul_normal_binades=8,
                mul_bias=7,
                acc_K=12,
                acc_P=7,
                rounding_mode=rm,
                saturation_mode=sat,
                acc_saturation_mode=SaturationMode.SAT_NONE,
            )
    blocked = SplitMac(B8, S12, accumulate_algorithm=BLOCK, block_size=8, outer=S16)
    assert spec_for_mac(blocked) == _binaryK_superfp_accumulated_spec(
        mul_K=8,
        mul_P=4,
        acc_man_bits=5,
        acc_exp_bits=5,
        acc_normal_binades=16,
        acc_bias=15,
        accumulate_algorithm=BLOCK,
        block_size=8,
        outer_man_bits=8,
        outer_exp_bits=6,
        outer_normal_binades=32,
        outer_bias=31,
    )
    tree = SplitMac(S8, B12, accumulate_algorithm=TREE, outer=B16)
    assert spec_for_mac(tree) == _superfp_binaryK_accumulated_spec(
        mul_man_bits=3,
        mul_exp_bits=4,
        mul_normal_binades=8,
        mul_bias=7,
        acc_K=12,
        acc_P=7,
        accumulate_algorithm=TREE,
        outer_K=16,
        outer_P=11,
    )
    # The op a spec names, by direction and algorithm.
    ops = torch.ops.mptorch
    assert spec_for_mac(SplitMac(B8, S12)).op is ops.custom_matmul_binaryK_superfp.default
    assert spec_for_mac(SplitMac(S8, B12)).op is ops.custom_matmul_superfp_binaryK.default
    assert spec_for_mac(blocked).op is ops.custom_matmul_binaryK_superfp_accumulated.default
    assert spec_for_mac(tree).op is ops.custom_matmul_superfp_binaryK_accumulated.default
    # The result holds the accumulate format's values, or the outer format's.
    assert spec_for_mac(SplitMac(B8, S12)).stored[0][1] == (5, 5, 16, 15, SaturationMode.SAT_NONE)
    assert spec_for_mac(tree).stored[0][1][:2] == (16, 11)


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
def test_tier3_layer_factories(device, direction):
    """A ``QLinear`` on a cross-family factory runs its forward and both
    gradients as the flat GEMMs, by hand: ``x W^T + b``, ``G W``, ``G^T x``."""
    mul, acc, _ = DIRECTIONS[direction]
    if direction == "binaryK_superfp":
        formats = binaryK_superfp_gemm_formats(
            mul.K, mul.P, acc.man_bits, acc.exp_bits, acc.normal_binades, acc.bias
        )
    else:
        formats = superfp_binaryK_gemm_formats(
            mul.man_bits, mul.exp_bits, mul.normal_binades, mul.bias, acc.K, acc.P
        )
    torch.manual_seed(2)
    layer = QLinear(19, 7, formats=formats).to(device)
    x = torch.randn(2, 5, 19, device=device, requires_grad=True)
    y = layer(x)
    g = torch.randn_like(y)
    y.backward(g)
    w, bias = layer.weight.detach(), layer.bias.detach()
    xd = x.detach().reshape(-1, 19)
    gd = g.reshape(-1, 7)
    assert torch.equal(y.detach(), (_flat(direction, xd, w, trans_b=True) + bias).reshape(2, 5, 7))
    assert x.grad is not None and layer.weight.grad is not None
    assert torch.equal(x.grad, _flat(direction, gd, w).reshape(2, 5, 19))
    assert torch.equal(layer.weight.grad, _flat(direction, gd.T.contiguous(), xd))


# --- stochastic rounding -------------------------------------------------------


def _sr_mac(direction, algorithm=NAIVE):
    mul, acc, outer = DIRECTIONS[direction]
    mul = type(mul)(**_fields(mul), prng_bits=10)
    acc = type(acc)(**_fields(acc), prng_bits=10)
    block_size = 8 if algorithm in (BLOCK, TREE) else None
    return SplitMac(
        mul,
        acc,
        rounding=RoundMode.SR,
        accumulate_algorithm=algorithm,
        block_size=block_size,
        outer=outer if block_size else None,
    )


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("algorithm", [NAIVE, KAHAN, BLOCK, TREE], ids=lambda a: a.name)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_sr_is_seeded_and_keyed_on_the_element(device, direction, algorithm, dtype):
    mac = _sr_mac(direction, algorithm)
    a = torch.randn(3, 33, 45, device=device, dtype=dtype)
    b = torch.randn(45, 17, device=device, dtype=dtype)

    def run(x, seed):
        torch.manual_seed(seed)
        return qmatmul(x, b, mac)

    first = run(a, 7)
    assert torch.equal(first, run(a, 7))
    assert not torch.equal(first, run(a, 8))
    assert torch.equal(first[0], run(a[0], 7))


@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_sr_is_independent_of_the_thread_count(direction, dtype):
    mac = _sr_mac(direction, BLOCK)
    a, b = _operands(70, 45, 66, "cpu", dtype)
    before = torch.get_num_threads()
    try:
        results = []
        for threads in (1, 4):
            torch.set_num_threads(threads)
            torch.manual_seed(3)
            results.append(qmm(a, b, mac))
    finally:
        torch.set_num_threads(before)
    assert torch.equal(results[0], results[1])


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("algorithm", [NAIVE, BLOCK, TREE], ids=lambda a: a.name)
def test_sr_is_bracketed_by_the_directed_modes(device, direction, algorithm):
    """Every SR rounding lands on RD's or RU's neighbour, and these three sums
    are compositions of non-decreasing functions of the products, so each SR
    result lies between the all-RD and the all-RU results, elementwise. (KAHAN
    subtracts its compensation and has no such bound.)"""
    sr = _sr_mac(direction, algorithm)
    a, b = _operands(16, 70, 12, device)
    torch.manual_seed(5)
    out = qmm(a, b, sr)
    lo = qmm(a, b, SplitMac(sr.mul, sr.acc, **_same(sr, RoundMode.RD)))
    hi = qmm(a, b, SplitMac(sr.mul, sr.acc, **_same(sr, RoundMode.RU)))
    assert bool((lo <= out).all()) and bool((out <= hi).all())
    assert bool((lo < hi).any())  # the bracket is not trivially tight


def _same(mac, mode):
    return dict(
        rounding=mode,
        accumulate_algorithm=mac.accumulate_algorithm,
        block_size=mac.block_size,
        outer=mac.outer,
    )


# --- what is refused -----------------------------------------------------------


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("algorithm", [NAIVE, KAHAN, BLOCK, TREE], ids=lambda a: a.name)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.SR], ids=lambda m: m.name)
def test_widened_carrier_is_the_float64_call(device, direction, algorithm, mode):
    """``carrier=torch.float64`` on float32 operands is the float64 call on
    the widened operands, narrowed back once, SR draws included; and the raw
    op runs float64 operands itself."""
    mac = _sr_mac(direction, algorithm)
    extra = _same(mac, mode)
    plain = SplitMac(mac.mul, mac.acc, **extra)
    named = SplitMac(mac.mul, mac.acc, carrier=torch.float64, **extra)
    a, b = _operands(8, 37, 5, device)

    def run(fn):
        torch.manual_seed(9)
        return fn()

    expect = _narrowed(run(lambda: qmm(a.double(), b.double(), plain)), torch.float32)
    assert torch.equal(run(lambda: qmm(a, b, named)), expect)
    spec = spec_for_mac(plain)
    raw = run(lambda: spec.op(a.double(), b.double(), False, False, *spec.args))
    assert torch.equal(_narrowed(raw, torch.float32), expect)


@requires_mps
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RD, RoundMode.SR], ids=lambda m: m.name)
def test_mps_is_the_cpu(mode):
    """An MPS call returns the CPU op's words, SR included."""
    g = torch.Generator().manual_seed(4)
    a = torch.randn(13, 37, generator=g)
    b = torch.randn(37, 11, generator=g)
    b8, s8 = BinaryK(8, 4, prng_bits=8), SuperFP(3, 4, 8, 7, prng_bits=8)
    b12, s12 = BinaryK(12, 7, prng_bits=8), SuperFP(5, 5, 16, 15, prng_bits=8)
    for mac in (
        SplitMac(b8, s12, rounding=mode),
        SplitMac(s8, b12, rounding=mode, accumulate_algorithm=KAHAN),
        SplitMac(b8, s12, rounding=mode, accumulate_algorithm=TREE, block_size=8, outer=S16),
    ):
        torch.manual_seed(2)
        ref = qmm(a, b, mac)
        torch.manual_seed(2)
        got = qmm(a.to("mps"), b.to("mps"), mac)
        assert torch.equal(got.cpu().view(torch.int32), ref.view(torch.int32))


def test_validation():
    with pytest.raises(ValueError, match="different families.*one format per slot"):
        SplitMac([B8, BinaryK(8, 3)], S12)
    with pytest.raises(ValueError, match="different families.*one format per slot"):
        SplitMac(S8, [B12, BinaryK(12, 6)])
    # The outer format is of the accumulate's family, whichever the multiply's is.
    with pytest.raises(TypeError, match="accumulate format's family.*BinaryK over SuperFP"):
        SplitMac(B8, S12, accumulate_algorithm=BLOCK, block_size=4, outer=B16)
    with pytest.raises(TypeError, match="accumulate format's family.*SuperFP over BinaryK"):
        SplitMac(S8, B12, accumulate_algorithm=TREE, outer=S16)
    # An unrounded sum has no second family: the multiply family's op.
    assert spec_for_mac(SplitMac(B8, None)).op is torch.ops.mptorch.custom_matmul_binaryK.default


@pytest.mark.parametrize("device", kernel_devices)
def test_raw_op_validation(device):
    """Each op takes the algorithms it implements: the NAIVE pair NAIVE only,
    the twins everything else."""
    a = torch.randn(3, 5, device=device)
    b = torch.randn(5, 2, device=device)
    naive = spec_for_mac(SplitMac(B8, S12))
    args = list(naive.args)
    args[9] = KAHAN.value
    with pytest.raises(RuntimeError, match="NAIVE only"):
        naive.op(a, b, False, False, *args)
    twin = spec_for_mac(SplitMac(B8, S12, accumulate_algorithm=KAHAN))
    args = list(twin.args)
    args[9] = NAIVE.value
    with pytest.raises(RuntimeError, match="NAIVE is the op without the _accumulated suffix"):
        twin.op(a, b, False, False, *args)
    # The raw ops raise on a requires_grad operand, like every GEMM op.
    with pytest.raises(RuntimeError, match="is not differentiable"):
        naive.op(a.requires_grad_(), b, False, False, *naive.args)


@pytest.mark.parametrize("device", kernel_devices)
def test_not_a_single_family_op_in_disguise(device):
    """The cross-family result differs from both single-family ones on the
    same operands: each family's rounding is where the mac says it is."""
    a, b = _operands(8, 40, 6, device)
    cross = qmm(a, b, SplitMac(B8, S8))
    assert not torch.equal(cross, qmm(a, b, SplitMac(B8, B8)))
    assert not torch.equal(cross, qmm(a, b, SplitMac(S8, S8)))
    ref = _Reference(a.cpu(), b.cpu(), B8, S8, None, RoundMode.RNE).naive()
    assert torch.equal(cross.cpu(), ref)
    # A sum left unrounded is the multiply family's op, and differs again.
    exact_sum = binaryK_matmul(a, b, mul_K=8, mul_P=4, accumulate_quant=False)
    assert torch.equal(exact_sum, qmm(a, b, SplitMac(B8, None)))
    assert not torch.equal(exact_sum, cross)
