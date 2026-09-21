"""The accumulate algorithms past NAIVE: KAHAN, BLOCK and TREE.

``csrc/common/gemm_accumulate.h`` states what each computes. The references
here are written from that statement rather than from the kernels: plain
Python over ``[M, N]`` tensors, one K-step at a time, with the elementwise
quantizers standing in for the three roundings (``mul``, ``add``, ``outer``)
and a Python list standing in for TREE's levels. Tier 3 holds the CPU and CUDA
kernels to them with ``torch.equal`` in every deterministic rounding mode.

The three are binary32-only so far and have no MPS kernel, so a float64 call
and an MPS call are each held to the error that names the follow-up.
"""

from typing import Any, cast

import numpy as np
import pytest
import torch

from mptorch import AccumulateAlgorithm, BinaryK, RoundMode, SuperFP
from mptorch.quant import (
    FusedMac,
    Quant,
    SplitMac,
    binaryK_matmul,
    binaryK_matmul_fma,
    qmatmul,
    qmm,
    superfp_matmul,
    superfp_matmul_fma,
)
from mptorch.quant.mac import spec_for_mac

from .markers import float64_devices, requires_mps

NAIVE, KAHAN, BLOCK, TREE = (
    AccumulateAlgorithm.NAIVE,
    AccumulateAlgorithm.KAHAN,
    AccumulateAlgorithm.BLOCK,
    AccumulateAlgorithm.TREE,
)
DETERMINISTIC = [m for m in RoundMode if m is not RoundMode.SR]

# No GPU kernel on MPS yet: the shared tests run where the kernels are.
kernel_devices = float64_devices

# One narrow and one wider format per family: products in the narrow one,
# sums in the wider, so the accumulate rounding is neither exact nor a no-op.
_BINARYK_FAMILIES = (BinaryK(8, 4), BinaryK(12, 7), BinaryK(16, 11))
_SUPERFP_FAMILIES = (SuperFP(3, 4, 8, 7), SuperFP(5, 5, 16, 15), SuperFP(8, 6, 32, 31))
FAMILIES: dict[str, tuple[Any, ...]] = {
    "binaryK": _BINARYK_FAMILIES,
    "superfp": _SUPERFP_FAMILIES,
}

# K values that leave partial 16-step slabs and partial blocks of every size.
KS = (5, 16, 17, 40, 64, 70)
BLOCK_SIZES = (2, 4, 8, 16, 32, 64)
TREE_SIZES = (2, 4, 16, 32, 128)


def _operands(M, K, N, device, dtype=torch.float32):
    """Operands whose dot products cancel enough for the algorithms to differ."""
    g = torch.Generator().manual_seed(K * 1000 + M)
    a = torch.randn(M, K, generator=g)
    b = torch.randn(K, N, generator=g)
    return a.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


# --- the references ------------------------------------------------------------


def _rounding(fmt, mode):
    """One of the three roundings: a format's quantizer, or the identity."""
    return (lambda x: x) if fmt is None else Quant(fmt, mode)


def _fma32(a, b, c):
    """a*b + c with one rounding to float32, elementwise.

    The product of two 24-bit significands is exact in x87's 80-bit
    ``np.longdouble``, and the sum is at most one rounding far below
    float32's precision before the one that counts.
    """
    wide = np.longdouble(a.numpy()) * np.longdouble(b.numpy()) + np.longdouble(c.numpy())
    return torch.from_numpy(wide.astype(np.float32))


class _Reference:
    """One dot-product reduction over ``[M, N]`` outputs at once.

    ``a`` is ``[M, K]`` and ``b`` ``[K, N]``, float32 on the CPU (the values a
    kernel loads, whatever dtype they were stored in). ``mul`` is ``None`` for
    a fused mac, whose step is then one FMA rounded by ``add``.
    """

    def __init__(self, a, b, mul, add, outer, mode):
        self.a, self.b = a, b
        self.fused = mul is None
        self.mul = _rounding(mul, mode)
        self.add = _rounding(add, mode)
        self.outer = _rounding(outer, mode)
        self.K = a.shape[1]
        self.zero = torch.zeros(a.shape[0], b.shape[1])

    def operands(self, k):
        return self.a[:, k, None].expand_as(self.zero), self.b[None, k, :].expand_as(self.zero)

    def product(self, k):
        x, y = self.operands(k)
        return self.mul(x * y)

    def step(self, k, s):
        """NAIVE's step: the sum ``s`` after product ``k``."""
        if self.fused:
            x, y = self.operands(k)
            return self.add(_fma32(x, y, s))
        return self.add(s + self.product(k))

    def naive(self):
        s = self.zero
        for k in range(self.K):
            s = self.step(k, s)
        return s

    def kahan(self):
        s, c = self.zero, self.zero
        for k in range(self.K):
            if self.fused:
                x, y = self.operands(k)
                y = self.add(_fma32(x, y, -c))
            else:
                y = self.add(self.product(k) - c)
            t = self.add(s + y)
            c = self.add(self.add(t - s) - y)
            s = t
        return s

    def block(self, block_size):
        tot, blk = self.zero, self.zero
        for k in range(self.K):
            blk = self.step(k, blk)
            if (k + 1) % block_size == 0 or k + 1 == self.K:
                tot = self.outer(tot + blk)
                blk = self.zero
        return tot

    def tree(self, block_size):
        assert not self.fused
        tot = self.zero
        for start in range(0, self.K, block_size):
            # A binary counter: level l holds a finished subtree of 2^l
            # products, or nothing.
            levels = {}
            for i, k in enumerate(range(start, min(start + block_size, self.K))):
                v, level = self.product(k), 0
                while i >> level & 1:
                    v = self.add(levels.pop(level) + v)
                    level += 1
                levels[level] = v
            # A whole block has left one subtree, at the top level. A partial
            # one is flushed lowest level first.
            root = None
            for level in sorted(levels):
                root = levels[level] if root is None else self.add(levels[level] + root)
            assert root is not None
            tot = self.outer(tot + root)
        return tot

    def run(self, algorithm, block_size):
        if algorithm is KAHAN:
            return self.kahan()
        if algorithm is BLOCK:
            return self.block(block_size)
        if algorithm is TREE:
            return self.tree(block_size)
        return self.naive()


def _mac(kind, family, algorithm, mode, block_size=None, outer=None):
    """The mac of one of the four kinds, and the reference's (mul, add)."""
    narrow, wide, _ = FAMILIES[family]
    extra: dict[str, Any] = dict(
        rounding=mode, accumulate_algorithm=algorithm, block_size=block_size, outer=outer
    )
    if kind == "split":
        return SplitMac(narrow, wide, **extra), (narrow, wide)
    if kind == "split_exact_sum":
        return SplitMac(narrow, None, **extra), (narrow, None)
    if kind == "fused":
        return FusedMac(wide, **extra), (None, wide)
    return FusedMac(None, **extra), (None, None)


KINDS = ("split", "split_exact_sum", "fused", "fused_unrounded")


def _check(device, kind, family, algorithm, mode, block_size, outer, K, dtype=torch.float32):
    a, b = _operands(5, K, 3, device, dtype)
    mac, (mul, add) = _mac(kind, family, algorithm, mode, block_size, outer)
    out = qmm(a, b, mac)
    ref = _Reference(a.cpu().float(), b.cpu().float(), mul, add, outer, mode).run(
        algorithm, block_size
    )
    assert out.dtype is dtype
    assert torch.equal(out.cpu(), ref.to(dtype)), (
        f"{algorithm.name} {kind} {family} {mode.name} block_size={block_size} outer={outer} K={K}"
    )


# --- tier 1: exact formats against torch.matmul --------------------------------


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("algorithm", [KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_tier1_unrounded_matches_matmul(device, algorithm):
    """With nothing rounded each algorithm is a float32 dot product in its own
    order, within float32's tolerance of ``torch.matmul``."""
    a, b = _operands(37, 203, 19, device)
    ref = a.double().cpu() @ b.double().cpu()
    # TREE needs a multiply format; a 24-bit binaryK leaves float32 products alone.
    exact = BinaryK(32, 24)
    block_size = None if algorithm is KAHAN else 16
    if algorithm is TREE:
        mac = SplitMac(exact, None, accumulate_algorithm=algorithm, block_size=block_size)
    else:
        mac = FusedMac(None, accumulate_algorithm=algorithm, block_size=block_size)
    out = qmm(a, b, mac)
    torch.testing.assert_close(out.cpu().double(), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", kernel_devices)
def test_tier1_kahan_recovers_what_naive_loses(device):
    """The classic demonstration: a sum of terms far below the running sum's
    last bit. NAIVE float32 drops every one; KAHAN carries them."""
    K = 4096
    a = torch.ones(1, K, device=device)
    b = torch.full((K, 1), 1e-8, device=device)
    b[0, 0] = 1.0
    exact = 1.0 + (K - 1) * 1e-8
    naive = qmm(a, b, FusedMac(None)).item()
    kahan = qmm(a, b, FusedMac(None, accumulate_algorithm=KAHAN)).item()
    assert naive == 1.0
    assert abs(kahan - exact) < 2e-7
    assert abs(kahan - exact) < abs(naive - exact) / 100


# --- tier 3: every deterministic mode against the references -------------------


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_tier3_kahan(device, family, kind, mode):
    for K in KS:
        _check(device, kind, family, KAHAN, mode, None, None, K)


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
@pytest.mark.parametrize("with_outer", [False, True], ids=["carrier_total", "outer"])
def test_tier3_block(device, family, kind, mode, with_outer):
    outer = FAMILIES[family][2] if with_outer else None
    for block_size in BLOCK_SIZES:
        for K in KS:
            _check(device, kind, family, BLOCK, mode, block_size, outer, K)


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("kind", ["split", "split_exact_sum"])
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
@pytest.mark.parametrize("with_outer", [False, True], ids=["carrier_total", "outer"])
def test_tier3_tree(device, family, kind, mode, with_outer):
    outer = FAMILIES[family][2] if with_outer else None
    for block_size in TREE_SIZES:
        for K in KS:
            _check(device, kind, family, TREE, mode, block_size, outer, K)


@pytest.mark.parametrize("device", kernel_devices)
def test_tier3_tree_long_blocks(device):
    """Levels 4 to 7 of the tree, across slabs: whole 256-blocks, and every
    partial one a K below and above the block size leaves."""
    outer = FAMILIES["binaryK"][2]
    for block_size in (64, 256):
        for K in (255, 256, 257, 300, 517):
            _check(device, "split", "binaryK", TREE, RoundMode.RNE, block_size, outer, K)


@pytest.mark.parametrize("device", kernel_devices)
def test_tier3_block_sizes_past_a_slab(device):
    """BLOCK sizes that are a multiple of 16 but not a power of two."""
    outer = FAMILIES["binaryK"][2]
    for block_size in (48, 80):
        for K in (47, 48, 49, 100, 161):
            _check(device, "split", "binaryK", BLOCK, RoundMode.RNE, block_size, outer, K)


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("algorithm", [KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_tier3_narrow_storage(device, dtype, algorithm):
    """float16 and bfloat16 operands are loaded into binary32, reduced there
    and stored back with one conversion, as under NAIVE."""
    block_size = None if algorithm is KAHAN else 8
    _check(device, "split", "binaryK", algorithm, RoundMode.RNE, block_size, None, 40, dtype)


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("algorithm", [KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_tier3_flat_wrappers_and_layouts(device, algorithm):
    """The schema tier, a transposed operand and a batch, against the mac tier."""
    narrow, wide, outer = _BINARYK_FAMILIES
    block_size = None if algorithm is KAHAN else 4
    has_outer = algorithm is not KAHAN
    a = torch.randn(3, 6, 21, device=device)
    b = torch.randn(5, 21, device=device)
    mac = SplitMac(
        narrow,
        wide,
        accumulate_algorithm=algorithm,
        block_size=block_size,
        outer=outer if has_outer else None,
    )
    flat = binaryK_matmul(
        a,
        b,
        trans_b=True,
        mul_K=narrow.K,
        mul_P=narrow.P,
        acc_K=wide.K,
        acc_P=wide.P,
        accumulate_algorithm=algorithm,
        block_size=block_size,
        outer_K=outer.K if has_outer else None,
        outer_P=outer.P if has_outer else None,
    )
    assert torch.equal(flat, qmatmul(a, b.mT, mac))
    # Each batch element is the 2D call on it.
    for i in range(a.shape[0]):
        assert torch.equal(flat[i], qmm(a[i], b.mT, mac))


# --- properties ----------------------------------------------------------------


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("kind", KINDS)
def test_one_block_is_naive(device, family, kind):
    """A block at least as long as K, folded into an unrounded total, is the
    NAIVE sum: ``0 + blk``."""
    a, b = _operands(9, 40, 7, device)
    naive, _ = _mac(kind, family, NAIVE, RoundMode.RNE)
    blocked, _ = _mac(kind, family, BLOCK, RoundMode.RNE, block_size=64)
    assert torch.equal(qmm(a, b, blocked), qmm(a, b, naive))


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("family", list(FAMILIES))
def test_tree_of_two_is_block_of_two(device, family):
    """With one format for the products and the sums, a pair summed as a tree
    and a pair summed in order are the same two roundings."""
    a, b = _operands(9, 41, 7, device)
    narrow, _, outer = FAMILIES[family]
    both: dict[str, Any] = dict(block_size=2, outer=outer)
    tree = SplitMac(narrow, narrow, accumulate_algorithm=TREE, **both)
    block = SplitMac(narrow, narrow, accumulate_algorithm=BLOCK, **both)
    assert torch.equal(qmm(a, b, tree), qmm(a, b, block))


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("algorithm", [KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_empty_and_degenerate_shapes(device, algorithm):
    block_size = None if algorithm is KAHAN else 4
    mac = SplitMac(BinaryK(8, 4), None, accumulate_algorithm=algorithm, block_size=block_size)
    assert qmm(torch.zeros(0, 5, device=device), torch.zeros(5, 3, device=device), mac).shape == (
        0,
        3,
    )
    # K = 0: every reduction is empty, and its result is +0.0.
    out = qmm(torch.zeros(2, 0, device=device), torch.zeros(0, 3, device=device), mac)
    assert out.shape == (2, 3)
    assert not out.any() and not torch.signbit(out).any()
    # K = 1: one product, flushed or folded at once.
    a, b = _operands(4, 1, 3, device)
    ref = Quant(BinaryK(8, 4))(a.cpu() @ b.cpu())
    assert torch.equal(qmm(a, b, mac).cpu(), ref)


# --- stochastic rounding -------------------------------------------------------


def _sr_mac(algorithm):
    # prng_bits is the format's: with the default 0, SR has nothing to draw.
    narrow, wide, outer = (BinaryK(f.K, f.P, prng_bits=12) for f in _BINARYK_FAMILIES)
    block_size = None if algorithm is KAHAN else 8
    return SplitMac(
        narrow,
        wide,
        rounding=RoundMode.SR,
        accumulate_algorithm=algorithm,
        block_size=block_size,
        outer=None if algorithm is KAHAN else outer,
    )


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("algorithm", [KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_sr_is_seeded_and_keyed_on_the_element(device, algorithm):
    """One seed, one result; another seed, another; and batch element 0 of a
    batched call is the 2D call, since a stream is keyed on the element's
    index into the whole output."""
    mac = _sr_mac(algorithm)
    a = torch.randn(3, 33, 45, device=device)
    b = torch.randn(45, 17, device=device)

    def run(x, seed):
        torch.manual_seed(seed)
        return qmatmul(x, b, mac)

    first = run(a, 7)
    assert torch.equal(first, run(a, 7))
    assert not torch.equal(first, run(a, 8))
    assert torch.equal(first[0], run(a[0], 7))


@pytest.mark.parametrize("algorithm", [KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_sr_is_independent_of_the_thread_count(algorithm):
    mac = _sr_mac(algorithm)
    a, b = _operands(70, 45, 66, "cpu")
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
@pytest.mark.parametrize("algorithm", [KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_sr_is_bracketed_and_unbiased(device, algorithm):
    """Every SR draw rounds to one of the two neighbours, so the mean over
    many seeds of a one-step reduction is the unrounded value."""
    fmt = BinaryK(8, 4, prng_bits=16)
    block_size = None if algorithm is KAHAN else 4
    mac = SplitMac(
        fmt, None, rounding=RoundMode.SR, accumulate_algorithm=algorithm, block_size=block_size
    )
    n = 20000
    a = torch.full((n, 1), 1.3, device=device)
    b = torch.full((1, 1), 0.7, device=device)
    torch.manual_seed(11)
    out = qmm(a, b, mac).flatten()
    lo = Quant(fmt, RoundMode.RD)(a[:1] * b).item()
    hi = Quant(fmt, RoundMode.RU)(a[:1] * b).item()
    assert set(out.unique().tolist()) == {lo, hi}
    assert abs(out.double().mean().item() - 1.3 * 0.7) < 3 * (hi - lo) / (2 * n**0.5)


# --- what is refused -----------------------------------------------------------


@pytest.mark.parametrize("device", kernel_devices)
@pytest.mark.parametrize("algorithm", [KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_binary64_is_refused(device, algorithm):
    """float64 operands and ``carrier=torch.float64`` name phase G, before
    anything is widened or launched."""
    block_size = None if algorithm is KAHAN else 4
    fmt = BinaryK(8, 4)
    mac = SplitMac(fmt, fmt, accumulate_algorithm=algorithm, block_size=block_size)
    a = torch.randn(3, 5, device=device)
    b = torch.randn(5, 2, device=device)
    with pytest.raises(ValueError, match="binary32 kernels only.*phase G"):
        qmm(a.double(), b.double(), mac)
    wide = SplitMac(
        fmt, fmt, accumulate_algorithm=algorithm, block_size=block_size, carrier=torch.float64
    )
    with pytest.raises(ValueError, match="binary32 kernels only.*phase G"):
        qmm(a, b, wide)
    # The op itself refuses too, for a caller that goes around the wrappers
    # (in a build with no float64 GEMM kernels at all, by naming that flag).
    spec = spec_for_mac(mac)
    with pytest.raises(RuntimeError, match="binary32 kernels only.*phase G|MPTORCH_NO_FP64"):
        spec.op(a.double(), b.double(), False, False, *spec.args)


@requires_mps
@pytest.mark.parametrize("algorithm", [KAHAN, BLOCK, TREE], ids=lambda a: a.name)
def test_mps_is_refused(algorithm):
    block_size = None if algorithm is KAHAN else 4
    fmt = BinaryK(8, 4)
    mac = SplitMac(fmt, fmt, accumulate_algorithm=algorithm, block_size=block_size)
    a = torch.randn(3, 5, device="mps")
    b = torch.randn(5, 2, device="mps")
    with pytest.raises(RuntimeError, match="no MPS kernel yet.*phase H"):
        qmm(a, b, mac)
    # NAIVE is untouched by the new schema arguments.
    assert torch.equal(
        qmm(a, b, SplitMac(fmt, fmt)).cpu(), qmm(a.cpu(), b.cpu(), SplitMac(fmt, fmt))
    )


def test_validation():
    e, w, s = BinaryK(8, 4), BinaryK(16, 11), SuperFP(3, 4, 8, 7)
    with pytest.raises(ValueError, match="BLOCK needs a block_size"):
        SplitMac(e, w, accumulate_algorithm=BLOCK)
    for bad in (0, 3, 5, 24, -16):
        with pytest.raises(ValueError, match="divides 16 or is a multiple of 16"):
            SplitMac(e, w, accumulate_algorithm=BLOCK, block_size=bad)
    for bad in (1, 3, 24, 512):
        with pytest.raises(ValueError, match="power of two in \\[2, 256\\]"):
            SplitMac(e, w, accumulate_algorithm=TREE, block_size=bad)
    tree = spec_for_mac(SplitMac(e, w, accumulate_algorithm=TREE))
    assert tree.args[-9] == 16  # the block size TREE defaults to
    with pytest.raises(ValueError, match="no product term"):
        FusedMac(e, accumulate_algorithm=TREE)
    for algorithm in (NAIVE, KAHAN):
        with pytest.raises(ValueError, match="takes no block_size and no outer format"):
            SplitMac(e, w, accumulate_algorithm=algorithm, block_size=4)
        with pytest.raises(ValueError, match="takes no block_size and no outer format"):
            FusedMac(e, accumulate_algorithm=algorithm, outer=w)
    with pytest.raises(TypeError, match="outer must be of the mac's format family"):
        SplitMac(e, None, accumulate_algorithm=BLOCK, block_size=4, outer=s)
    with pytest.raises(TypeError, match="one BinaryK or SuperFP format"):
        SplitMac(e, w, accumulate_algorithm=BLOCK, block_size=4, outer=cast(Any, [w, w]))
    with pytest.raises(ValueError, match="single format per slot"):
        SplitMac([e, w], None, accumulate_algorithm=KAHAN)
    # An unrounded fused step has no family of its own: the outer format's.
    for outer in (w, s):
        mac = FusedMac(None, accumulate_algorithm=BLOCK, block_size=4, outer=outer)
        assert qmm(torch.ones(2, 8), torch.ones(8, 2), mac).shape == (2, 2)


def test_flat_wrapper_validation():
    a, b = torch.ones(2, 8), torch.ones(8, 2)
    with pytest.raises(ValueError, match="outer_K and outer_P name the outer format together"):
        binaryK_matmul(a, b, mul_K=8, mul_P=4, accumulate_algorithm=BLOCK, block_size=4, outer_K=16)
    with pytest.raises(ValueError, match="name the outer format together"):
        superfp_matmul(
            a,
            b,
            mul_man_bits=3,
            mul_exp_bits=4,
            mul_normal_binades=8,
            mul_bias=7,
            accumulate_algorithm=BLOCK,
            block_size=4,
            outer_man_bits=5,
        )
    with pytest.raises(ValueError, match="takes no block_size"):
        binaryK_matmul_fma(a, b, fma_K=8, fma_P=4, block_size=4)
    with pytest.raises(ValueError, match="takes no block_size"):
        superfp_matmul_fma(
            a, b, fma_man_bits=3, fma_exp_bits=4, fma_normal_binades=8, fma_bias=7, outer_bias=7
        )


def test_raw_op_validation():
    """``torch.ops.mptorch.*`` is a public entry point: what the wrappers
    refuse, the ops refuse too. NAIVE is the four ops it always was, and the
    other three algorithms are their ``*_accumulated`` twins."""
    a, b = torch.ones(2, 8), torch.ones(8, 2)
    head = (a, b, False, False, 8, 4, 8, True, True, 8, 4, 8, True)
    modes = (0, 0, 0, 0, 0, 0, 0)
    no_outer = (False, 0, 0, 0, True, 0, 0, 0)

    naive = torch.ops.mptorch.custom_matmul_binaryK.default
    assert naive(*head, 0, *modes).shape == (2, 2)
    with pytest.raises(RuntimeError, match="is not an AccumulateAlgorithm"):
        naive(*head, 4, *modes)
    with pytest.raises(RuntimeError, match="NAIVE only.*_accumulated"):
        naive(*head, 1, *modes)

    op = torch.ops.mptorch.custom_matmul_binaryK_accumulated.default
    assert op(*head, 1, *modes, 0, *no_outer).shape == (2, 2)
    with pytest.raises(RuntimeError, match="is not an AccumulateAlgorithm"):
        op(*head, 4, *modes, 0, *no_outer)
    with pytest.raises(RuntimeError, match="NAIVE is the op without the _accumulated suffix"):
        op(*head, 0, *modes, 0, *no_outer)
    with pytest.raises(RuntimeError, match="KAHAN takes no block_size"):
        op(*head, 1, *modes, 4, *no_outer)
    with pytest.raises(RuntimeError, match="divides 16 or is a multiple of 16"):
        op(*head, 2, *modes, 5, *no_outer)
    with pytest.raises(RuntimeError, match="power of two in \\[2, 256\\]"):
        op(*head, 3, *modes, 512, *no_outer)
    fused = torch.ops.mptorch.custom_matmul_binaryK_fma_accumulated.default
    with pytest.raises(RuntimeError, match="no product term"):
        fused(a, b, False, False, True, 8, 4, 8, True, 3, 0, 0, 0, 0, 16, *no_outer)
    mixed = torch.ops.mptorch.custom_matmul_binaryK_fma_mixed.default
    idx = torch.zeros(2, 2, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="NAIVE only"):
        mixed(a, b, idx, False, False, True, [8], [4], [8], True, 1, 0, 0, 0, 0)


def test_the_accumulated_ops_need_a_gradient_entry_point_too():
    """They are in the Autograd raise list like every other op."""
    a = torch.ones(2, 8, requires_grad=True)
    b = torch.ones(8, 2)
    with pytest.raises(RuntimeError, match="is not differentiable.*qmatmul"):
        binaryK_matmul(a, b, mul_K=8, mul_P=4, accumulate_algorithm=KAHAN)
    # The differentiable entry point runs them, forward and backward.
    mac = SplitMac(BinaryK(8, 4), None, accumulate_algorithm=KAHAN)
    qmatmul(a, b, mac).sum().backward()
    assert a.grad is not None and a.grad.shape == a.shape
