"""The counted GEMMs: the single-format GEMMs that also count their casts' events.

Each single-format GEMM op has a counted twin (``csrc/common/gemm_counted.h``)
that takes every accumulate algorithm, NAIVE included, and returns its result
with an int64 ``[3, 6]`` tensor: the roles ``mul``, ``acc`` and ``outer`` by
the events of `mptorch.quant.events`. This module holds them to two things:

- The result is the twin's word for word, in every algorithm and rounding
  mode (SR under the same seed), both carriers and the layouts.
- The counts are a step-by-step reference's: `test_accumulators`' reference,
  with each of its roundings recounted by `_recount`, which states the
  events on torch tensors, apart from the kernels' classifier on the words.
  That holds in every deterministic mode; under SR, the casts' totals and the
  events that no draw can change are held instead.

The boundary rows of `test_event_counts` also go through a one-step GEMM
whose product is the value itself, so the kernels' classifier and the recount
are held to each other on every row. A counted conv op's pass is held to its
twin's result and to the counts of the counted GEMMs over the unfolded
operands (`test_qconv_gemm`'s references), one per residue class of the
stride in the input gradient.

The fixed-point GEMMs count on both of their paths, the carrier's and the
integer accumulator's, whose sum is classified on the integers. Their counts
are held to the definition (`fixedpoint_gemm_reference`'s steps) with each
cast classified on exact rationals: the product's as a cast of a carrier
value, the sum's on its exact value.
"""

import math
from contextlib import contextmanager
from fractions import Fraction
from typing import Any
from unittest import mock

import pytest
import torch

from mptorch import AccumulateAlgorithm, BinaryK, FixedPoint, Overflow, RoundMode, SuperFP
from mptorch.quant import FusedMac, Quant, SplitMac
from mptorch.quant.events import _recount
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops._conv import _FWD, _IGRAD, _WGRAD, _ConvOp
from mptorch.quant.ops._counted import _counted_conv_twin, _counted_twin
from mptorch.quant.ops._spec import _run_gemm

from . import fixedpoint_gemm_reference as fxref
from . import test_qconv_gemm as conv_refs
from .markers import HAS_COUNTING, cuda_devices, float64_devices, requires_counting
from .test_accumulators import FAMILIES, FAMILIES64, _operands, _Reference
from .test_event_counts import CASES

NAIVE, KAHAN, BLOCK, TREE = (
    AccumulateAlgorithm.NAIVE,
    AccumulateAlgorithm.KAHAN,
    AccumulateAlgorithm.BLOCK,
    AccumulateAlgorithm.TREE,
)
DETERMINISTIC = [m for m in RoundMode if not m.is_stochastic]
ROLES = ("mul", "acc", "outer")
# The counted GEMMs have CPU and CUDA kernels; MPS has none yet.
counted_devices = ["cpu", *cuda_devices]


class _CountingReference(_Reference):
    """`_Reference` with each rounding's casts recounted, by role."""

    def __init__(self, a, b, mul, add, outer, mode):
        super().__init__(a, b, mul, add, outer, mode)
        self.counts = torch.zeros(3, 6, dtype=torch.int64)
        self.mul = self._counting(mul, mode, 0)
        self.add = self._counting(add, mode, 1)
        self.outer = self._counting(outer, mode, 2)

    def _counting(self, fmt, mode, role):
        if fmt is None:
            return lambda x: x
        quant = Quant(fmt, mode)

        def rounding(x):
            q = quant(x)
            self.counts[role] += _recount(x, q, fmt)
            return q

        return rounding


# The algorithms with their block sizes and outer formats: one in-slab block
# and one past a slab, each tree with and without its outer format.
SCHEDULES = [
    (NAIVE, None, False),
    (KAHAN, None, False),
    (BLOCK, 8, False),
    (BLOCK, 32, True),
    (TREE, 16, False),
    (TREE, 4, True),
]
KINDS = ("split", "split_exact_sum", "fused", "fused_unrounded")


def _mac(kind, formats, algorithm, mode, block_size, with_outer):
    """A mac of one of the four kinds, and the reference's (mul, add, outer)."""
    narrow, wide, outer_fmt = formats
    outer = outer_fmt if with_outer else None
    extra: dict[str, Any] = dict(rounding=mode, accumulate_algorithm=algorithm)
    if algorithm in (BLOCK, TREE):
        extra.update(block_size=block_size, outer=outer)
    if kind == "split":
        return SplitMac(narrow, wide, **extra), (narrow, wide, outer)
    if kind == "split_exact_sum":
        return SplitMac(narrow, None, **extra), (narrow, None, outer)
    if kind == "fused":
        return FusedMac(wide, **extra), (None, wide, outer)
    return FusedMac(None, **extra), (None, None, outer)


def _counted(mac, a, b, trans_a=False, trans_b=False):
    """The counted call of a mac: its twin's spec through the counted op."""
    spec = spec_for_mac(mac)
    twin = _counted_twin(spec)
    assert twin is not None, mac
    op, args = twin
    return op(a, b, trans_a, trans_b, *args), spec


def _schedules(kind):
    for s in SCHEDULES:
        if s[0] is TREE and kind.startswith("fused"):
            continue
        yield s


CASES_32 = [(kind, family, *s) for kind in KINDS for family in FAMILIES for s in _schedules(kind)]


def _id(case):
    kind, family, algorithm, block_size, outer = case
    return f"{kind}-{family}-{algorithm.name}{block_size or ''}{'-outer' if outer else ''}"


def _check(device, case, mode, dtype=torch.float32, families=FAMILIES, K=40):
    kind, family, algorithm, block_size, with_outer = case
    a, b = _operands(5, K, 3, device, dtype)
    mac, (mul, add, outer) = _mac(kind, families[family], algorithm, mode, block_size, with_outer)
    (c, counts), spec = _counted(mac, a, b)
    twin = _run_gemm(spec, a, b, False, False)
    assert torch.equal(c, twin)
    carrier = torch.float64 if dtype is torch.float64 else torch.float32
    ref = _CountingReference(a.cpu().to(carrier), b.cpu().to(carrier), mul, add, outer, mode)
    expected = ref.run(algorithm, block_size)
    assert torch.equal(c.cpu(), expected.to(dtype))
    assert counts.device == c.device and counts.dtype is torch.int64
    assert torch.equal(counts.cpu(), ref.counts), f"\n{counts.cpu()}\n!=\n{ref.counts}"


@requires_counting
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
@pytest.mark.parametrize("case", CASES_32, ids=_id)
@pytest.mark.parametrize("device", counted_devices)
def test_counts_are_the_references(device, case, mode):
    """Every cast of every algorithm, in each role, in binary32."""
    _check(device, case, mode)


BINARY64_CASES = [
    ("split", "binaryK_wide", KAHAN, None, False),
    ("fused", "binaryK_wide", BLOCK, 16, True),
    ("split", "superfp_wide", TREE, 8, True),
    ("split_exact_sum", "superfp_wide", NAIVE, None, False),
]


@requires_counting
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RD, RoundMode.RO], ids=lambda m: m.name)
@pytest.mark.parametrize("case", BINARY64_CASES, ids=_id)
@pytest.mark.parametrize("device", float64_devices)
def test_counts_are_the_references_in_binary64(device, case, mode):
    """The same, on float64 operands, in formats only binary64 holds."""
    _check(device, case, mode, torch.float64, FAMILIES64, K=24)


@requires_counting
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", counted_devices)
def test_narrow_storage(device, dtype):
    """Narrow operands load into binary32, where the steps are counted (the
    result's format, Binary12p7, fits both dtypes)."""
    _check(device, ("split", "binaryK", KAHAN, None, False), RoundMode.RNE, dtype)


@requires_counting
@pytest.mark.parametrize("algorithm", [NAIVE, KAHAN, BLOCK, TREE], ids=lambda a: a.name)
@pytest.mark.parametrize("device", counted_devices)
def test_layouts(device, algorithm):
    """A transposed operand changes no cast, and a batched call counts the
    casts of its batch elements' 2D calls."""
    narrow, wide, outer = FAMILIES["binaryK"]
    extra: dict[str, Any] = {}
    if algorithm in (BLOCK, TREE):
        extra = dict(block_size=16, outer=outer)
    mac = SplitMac(narrow, wide, accumulate_algorithm=algorithm, **extra)
    g = torch.Generator().manual_seed(3)
    a = torch.randn(4, 33, 37, generator=g).to(device)
    b = torch.randn(4, 37, 35, generator=g).to(device)
    (c, counts), _ = _counted(mac, a, b)
    total = torch.zeros(3, 6, dtype=torch.int64)
    for i in range(4):
        (ci, ni), _ = _counted(mac, a[i], b[i])
        assert torch.equal(ci, c[i])
        total += ni.cpu()
    assert torch.equal(counts.cpu(), total)
    (ct, nt), _ = _counted(mac, a.mT.contiguous().mT, b.mT.contiguous().mT)
    assert torch.equal(ct, c) and torch.equal(nt, counts)
    (ct, nt), _ = _counted(mac, a.mT.contiguous(), b.mT.contiguous(), True, True)
    assert torch.equal(ct, c) and torch.equal(nt, counts)
    assert counts[0, 0] == 4 * 33 * 35 * 37  # one product cast per step


@requires_counting
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("device", counted_devices)
def test_the_kernels_classify_as_the_recount(device, case):
    """Each boundary row of `test_event_counts` as a one-step GEMM, ``a @
    [[1]]``, whose product is the value itself: the mul role's counts are the
    recount's of the quantizer on that value."""
    _, fmt, rounding, rows = case
    if isinstance(fmt, BinaryK | SuperFP):
        mac = SplitMac(fmt, None, rounding=rounding)
    else:
        pytest.skip("fixed point has its own counted ops")
    one = torch.ones(1, 1, device=device)
    for v, _ in rows:
        a = torch.tensor([[v]], device=device)
        (c, counts), _ = _counted(mac, a, one)
        expected = _recount(a, Quant(fmt, rounding)(a), fmt)
        assert torch.equal(counts[0].cpu(), expected.cpu()), f"{fmt} {v!r}: {counts[0]}"
        assert not counts[1:].any()


@requires_counting
@pytest.mark.parametrize("algorithm", [NAIVE, KAHAN, BLOCK, TREE], ids=lambda a: a.name)
@pytest.mark.parametrize("device", counted_devices)
def test_stochastic_rounding(device, algorithm):
    """Under SR the result is the twin's under the same seed, a cast's count
    does not depend on the draws, and events certain for every draw are
    counted for every cast: products far past the largest value overflow,
    and exact zeros make no event."""
    narrow, wide, outer = FAMILIES["binaryK"]
    extra: dict[str, Any] = {}
    if algorithm in (BLOCK, TREE):
        extra = dict(block_size=8, outer=outer)
    mac = SplitMac(narrow, wide, rounding=RoundMode.SR, accumulate_algorithm=algorithm, **extra)
    rne = SplitMac(narrow, wide, accumulate_algorithm=algorithm, **extra)
    a, b = _operands(9, 40, 7, device)
    torch.manual_seed(11)
    (c, counts), spec = _counted(mac, a, b)
    torch.manual_seed(11)
    assert torch.equal(c, _run_gemm(spec, a, b, False, False))
    (_, deterministic), _ = _counted(rne, a, b)
    # KAHAN's and TREE's casts are a function of K and the block size alone
    assert torch.equal(counts[:, 0], deterministic[:, 0])
    big = torch.full((4, 24), 1e6, device=device)
    (_, n), _ = _counted(mac, big, big.T.contiguous())
    assert n[0, 1] == n[0, 0] == 4 * 4 * 24  # every product overflows
    zeros = torch.zeros(4, 24, device=device)
    (_, n), _ = _counted(mac, zeros, zeros.T.contiguous())
    assert n[:, 1:].sum() == 0 and n[0, 0] == 4 * 4 * 24


@requires_counting
def test_validation():
    """The counted ops check what their twins check, in the same order and
    words, and NAIVE takes no block size and no outer format."""
    a = torch.ones(2, 3)
    narrow, wide, outer = FAMILIES["binaryK"]
    tree = SplitMac(narrow, wide, accumulate_algorithm=TREE, block_size=4)
    op, args = _counted_twin(spec_for_mac(tree))  # ty: ignore[not-iterable]
    with pytest.raises(RuntimeError, match="expects 2D or 3D tensors"):
        op(a[0], a.T, False, False, *args)
    naive = list(args)
    naive[9] = NAIVE.value  # accumulate_algorithm
    with pytest.raises(RuntimeError, match="NAIVE takes no block_size"):
        op(a, a.T, False, False, *naive)
    fused = FusedMac(wide, accumulate_algorithm=BLOCK, block_size=4)
    fop, fargs = _counted_twin(spec_for_mac(fused))  # ty: ignore[not-iterable]
    tree_args = list(fargs)
    tree_args[5] = TREE.value  # accumulate_algorithm
    with pytest.raises(RuntimeError, match="fused multiply-add has no product term"):
        fop(a, a.T, False, False, *tree_args)
    assert _counted_twin(spec_for_mac(SplitMac([narrow, wide], wide))) is None  # a palette


@contextmanager
def _counted_references():
    """`test_qconv_gemm`'s references, each of their GEMMs counted: yields the
    running total of the counts."""
    total = torch.zeros(3, 6, dtype=torch.int64)

    def gemm(spec, a, b, prec_idx=None):
        assert prec_idx is None
        twin = _counted_twin(spec)
        assert twin is not None
        op, args = twin
        c, n = op(a.contiguous(), b.contiguous(), False, False, *args)
        total.add_(n.cpu())
        return c

    with mock.patch.object(conv_refs, "_gemm", gemm):
        yield total


_BK, _SF = FAMILIES["binaryK"], FAMILIES["superfp"]
CONV_MACS = {
    "split-KAHAN": lambda mode: SplitMac(*_BK[:2], rounding=mode, accumulate_algorithm=KAHAN),
    "split-TREE": lambda mode: SplitMac(
        *_SF[:2], rounding=mode, accumulate_algorithm=TREE, block_size=4, outer=_SF[2],
    ),
    "fused-NAIVE": lambda mode: FusedMac(_BK[1], rounding=mode),
    "fused-BLOCK": lambda mode: FusedMac(
        _SF[1], rounding=mode, accumulate_algorithm=BLOCK, block_size=8, outer=_SF[2],
    ),
}  # fmt: skip


@requires_counting
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RU, RoundMode.SR], ids=lambda m: m.name)
@pytest.mark.parametrize("which", list(CONV_MACS))
@pytest.mark.parametrize("device", counted_devices)
def test_conv_passes(device, which, mode):
    """Each pass of a counted conv op gives its twin's result and the counts
    of the counted GEMMs over the unfolded operands, which under a
    deterministic mode are its result too."""
    mac = CONV_MACS[which](mode)
    spec = spec_for_mac(mac)
    twin = _counted_conv_twin(spec)
    assert twin is not None
    counted, args = twin
    plain = _ConvOp(spec, None)
    size, kernel, groups = (9, 7), (3, 2), 1
    s, p, d = conv_refs._geometry(2, (2, 1), (1, 0), 1)
    x, w = conv_refs._operands(2, 3, 5, groups, size, kernel, device, torch.float32)
    out = conv_refs._out(size, kernel, s, p, d)
    dy = torch.randn(2, 5, *out, generator=torch.Generator().manual_seed(1)).to(device)
    passes = [
        (_FWD, w, x, out, lambda: conv_refs.ref_fwd(spec, x, w, s, p, d, groups, out)),
        (_IGRAD, w, dy, size, lambda: conv_refs.ref_igrad(spec, dy, w, size, s, p, d, groups)),
        (_WGRAD, dy, x, kernel, lambda: conv_refs.ref_wgrad(spec, dy, x, kernel, s, p, d, groups)),
    ]
    for conv_pass, a, b, extent, ref in passes:
        geometry = (tuple(extent), tuple(s), tuple(p), tuple(d), groups)
        torch.manual_seed(5)
        c, n = counted(a, b, conv_pass, *geometry, *args)
        torch.manual_seed(5)
        assert torch.equal(c, plain(conv_pass, a, b, *geometry)), conv_pass
        if mode.is_stochastic:
            continue
        with _counted_references() as total:
            expected = ref()
        assert torch.equal(c, expected), conv_pass
        assert torch.equal(n.cpu(), total), f"pass {conv_pass}:\n{n.cpu()}\n!=\n{total}"
        assert n[0, 0] > 0 or n[1, 0] > 0


# --- fixed point --------------------------------------------------------------

_WORDS = {torch.float32: torch.int32, torch.float64: torch.int64}


def _same(x, y):
    """Word for word, NaNs included, which `torch.equal` never finds equal."""
    return x.dtype == y.dtype and torch.equal(
        x.contiguous().view(_WORDS[x.dtype]), y.contiguous().view(_WORDS[y.dtype])
    )


def _same_values(x, y):
    """The same values, NaNs in the same places whatever their words: the
    carrier path keeps the carrier's NaN (x86's default NaN is negative),
    the integer accumulator writes a positive one."""
    nan = torch.isnan(x)
    return torch.equal(nan, torch.isnan(y)) and torch.equal(
        x.masked_fill(nan, 0), y.masked_fill(nan, 0)
    )


def _fx_ends(fmt):
    """The largest magnitude on each side: an unsigned format's is its top."""
    lo, hi = fxref.codes(fmt)
    top = fxref._exact(hi, fmt)
    return top, (fxref._exact(lo, fmt) if fmt.is_signed else -top)


def _fx_events(x, q, fmt):
    """One cast's events (`mptorch.quant.events`' table) on exact values: ``x``
    a Fraction, a float infinity or None (a NaN), ``q`` the code or None."""
    if x is None or q is None:
        return [1, 0, 0, 0, 0, 1]
    top, bottom = _fx_ends(fmt)
    finite = not isinstance(x, float)
    overflow = finite and (x > top or x < bottom)
    underflow = finite and x != 0 and q == 0
    inexact = not finite or fxref._exact(q, fmt) != x
    return [1, int(overflow), int(underflow), 0, int(inexact), 0]


def _fx_counts(a, b, mul, acc, mode, wrap):
    """The counts of the definition's steps (`fixedpoint_gemm_reference.gemm`),
    in binary32, every cast classified; after a NaN each step's sum is a NaN
    cast, as the kernels make it."""
    counts = torch.zeros(3, 6, dtype=torch.int64)
    products = a.cpu().float()[:, :, None] * b.cpu().float()[None, :, :]
    M, K, N = products.shape
    for i in range(M):
        for j in range(N):
            s, nan = 0, False
            for k in range(K):
                p = products[i, k, j].item()
                value = None if math.isnan(p) else (p if math.isinf(p) else Fraction(p))
                if mul is not None:
                    pq = fxref.quantize(p, mul, mode)
                    counts[0] += torch.tensor(_fx_events(value, pq, mul))
                    if nan or pq is None:
                        nan = True
                        counts[1] += torch.tensor(_fx_events(None, None, acc))
                        continue
                    x = fxref._exact(s, acc) + fxref._exact(pq, mul)
                else:
                    if nan or value is None:
                        nan = True
                        counts[1] += torch.tensor(_fx_events(None, None, acc))
                        continue
                    x = value if isinstance(value, float) else value + fxref._exact(s, acc)
                q = fxref.quantize(x, acc, mode, wrap=wrap)
                assert q is not None  # a sum that is not a NaN rounds to a code
                counts[1] += torch.tensor(_fx_events(x, q, acc))
                s = q
    return counts


FX = {
    # the carrier path: the span of a sum fits binary32
    "carrier": (FixedPoint(12, 6), FixedPoint(16, 6), Overflow.SATURATE),
    "carrier-unsigned": (FixedPoint(8, 4), FixedPoint(10, 4, is_signed=False), Overflow.SATURATE),
    # the integer accumulator: a span past binary32's, a wrap-around, a
    # symmetric format, the fused macs
    "integer": (FixedPoint(16, 12), FixedPoint(20, 4), Overflow.SATURATE),
    "integer-wrap": (FixedPoint(10, 6), FixedPoint(8, 2), Overflow.WRAP),
    "integer-symmetric": (FixedPoint(16, 12), FixedPoint(9, 3, symmetric=True), Overflow.SATURATE),
    "fused": (None, FixedPoint(10, 4), Overflow.SATURATE),
    "fused-wrap": (None, FixedPoint(8, 3), Overflow.WRAP),
    "fused-unsigned": (None, FixedPoint(9, 4, is_signed=False), Overflow.SATURATE),
}


def _fx_mac(which, mode):
    mul, acc, overflow = FX[which]
    if mul is None:
        return FusedMac(acc, rounding=mode, overflow=overflow)
    return SplitMac(mul, acc, rounding=mode, overflow=overflow)


def _fx_operands(device, K=24):
    """Operands whose sums reach both ends of the formats and their steps."""
    g = torch.Generator().manual_seed(K)
    a = torch.randn(4, K, generator=g) * 3
    b = torch.randn(K, 3, generator=g)
    b[0, 0], b[1, 1] = math.inf, 1e-4  # infinite products, and some below a step
    a[0, 0] = 0.0  # and one NaN, 0 * inf, which ends its element's sum
    # a row whose sums saturate (or wrap), and a last step of zero products,
    # so that a saturated sum is cast once more exactly at the format's end
    a[1] *= 8
    b[-1] = 0.0
    return a.to(device), b.to(device)


@requires_counting
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
@pytest.mark.parametrize("which", list(FX))
@pytest.mark.parametrize("device", counted_devices)
def test_fixedpoint_counts_are_the_definitions(device, which, mode):
    """Both paths, split and fused, saturating and wrapping: the twin's
    result, and the counts of the definition's casts."""
    mac = _fx_mac(which, mode)
    a, b = _fx_operands(device)
    (c, counts), spec = _counted(mac, a, b)
    assert _same(c, _run_gemm(spec, a, b, False, False))
    mul, acc, overflow = FX[which]
    expected = _fx_counts(a, b, mul, acc, mode, overflow is Overflow.WRAP)
    assert torch.equal(counts.cpu(), expected), f"\n{counts.cpu()}\n!=\n{expected}"


@requires_counting
@pytest.mark.parametrize("device", counted_devices)
def test_fixedpoint_paths_count_alike(device):
    """The integer accumulator, forced where the carrier path would run
    (the split op's `integer` argument), makes the same casts with the same
    events; and a wide result (float64, an accumulate format wider than the
    operands') is counted as any other."""
    mac = _fx_mac("carrier", RoundMode.RNE)
    a, b = _fx_operands(device)
    (c, counts), spec = _counted(mac, a, b)
    op, args = _counted_twin(spec)  # ty: ignore[not-iterable]
    ci, ni = op(a, b, False, False, *args[:-1], True)  # integer=True
    assert _same_values(ci, c) and torch.equal(ni, counts)
    wide = SplitMac(FixedPoint(16, 8), FixedPoint(40, 8))
    (cw, nw), spec = _counted(wide, a, b)
    assert cw.dtype is torch.float64
    assert _same(cw, _run_gemm(spec, a, b, False, False))
    expected = _fx_counts(a, b, FixedPoint(16, 8), FixedPoint(40, 8), RoundMode.RNE, False)
    assert torch.equal(nw.cpu(), expected)


@requires_counting
@pytest.mark.parametrize("which", list(FX))
@pytest.mark.parametrize("device", counted_devices)
def test_fixedpoint_stochastic_rounding(device, which):
    """Under SR the result is the twin's under the same seed, and the casts'
    count does not depend on the draws."""
    mul, acc, overflow = FX[which]
    sr = FusedMac(
        FixedPoint(acc.wl, acc.fl, is_signed=acc.is_signed, symmetric=acc.symmetric, prng_bits=6),
        rounding=RoundMode.SR,
        overflow=overflow,
    )
    if mul is not None:
        sr = SplitMac(
            FixedPoint(mul.wl, mul.fl, prng_bits=6),
            sr.fma,
            rounding=RoundMode.SR,
            overflow=overflow,
        )
    a, b = _fx_operands(device)
    torch.manual_seed(3)
    (c, counts), spec = _counted(sr, a, b)
    torch.manual_seed(3)
    assert _same(c, _run_gemm(spec, a, b, False, False))
    (_, deterministic), _ = _counted(_fx_mac(which, RoundMode.RNE), a, b)
    assert torch.equal(counts[:, 0], deterministic[:, 0])


@requires_counting
@pytest.mark.parametrize("which", ["carrier", "integer-wrap", "fused"])
@pytest.mark.parametrize("device", counted_devices)
def test_fixedpoint_conv(device, which):
    """A counted fixed-point conv pass gives its twin's result and the
    counted GEMM's counts over the unfolded operands."""
    mac = _fx_mac(which, RoundMode.RNE)
    spec = spec_for_mac(mac)
    twin = _counted_conv_twin(spec)
    assert twin is not None
    counted, args = twin
    size, kernel = (7, 6), (3, 3)
    s, p, d = conv_refs._geometry(2, 1, 1, 1)
    x, w = conv_refs._operands(2, 3, 4, 1, size, kernel, device, torch.float32)
    out = conv_refs._out(size, kernel, s, p, d)
    geometry = (tuple(out), tuple(s), tuple(p), tuple(d), 1)
    c, n = counted(w, x, _FWD, *geometry, *args)
    assert _same(c, _ConvOp(spec, None)(_FWD, w, x, *geometry))
    with _counted_references() as total:
        assert _same(c, conv_refs.ref_fwd(spec, x, w, s, p, d, 1, out))
    assert torch.equal(n.cpu(), total)


@pytest.mark.skipif(HAS_COUNTING, reason="Built with MPTORCH_COUNTING=1.")
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_without_the_flag_the_ops_raise(device):
    """A build without MPTORCH_COUNTING=1 has no counted kernels: the ops
    raise, naming the flag, on the CPU and the meta device alike."""
    narrow, wide, _ = FAMILIES["binaryK"]
    a = torch.ones(2, 3, device=device)
    for mac in (SplitMac(narrow, wide), FusedMac(wide, accumulate_algorithm=KAHAN)):
        op, args = _counted_twin(spec_for_mac(mac))  # ty: ignore[not-iterable]
        with pytest.raises(RuntimeError, match="MPTORCH_COUNTING=1"):
            op(a, a.T, False, False, *args)
