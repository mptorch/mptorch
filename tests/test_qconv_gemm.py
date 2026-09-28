"""The conv ops: a convolution's three passes as gathered GEMMs (R-4).

``mptorch/csrc/common/gemm_gather.h`` states what each pass computes: the
forward and the weight gradient are the GEMM of their mac over the operands
``unfold`` would build, zeros included, and the input gradient is one GEMM per
residue class of the stride over the kernel taps that reach it, in the same K
order and with the same ``[batch, M, N]`` output index. The references here
build those operands explicitly, with gathers written from the convolution's
definition (output position ``o`` reads input position ``o*s - p + i*d``,
zero outside the input), and call the flat GEMM op the mac resolves to. Tier 3
holds the three passes to them with ``torch.equal``, in every rounding mode,
stochastic rounding included (the output index keys the streams and both
calls draw from the same generator state), for every accumulate algorithm and
for a palette. The input gradient is also held to the transposed convolution
that sums the zeros a stride inserts: bit for bit where a zero term changes
nothing (a NAIVE sum, deterministic rounding), and to the same accuracy
against a float64 reference where it does. Tier 1 holds ``QConv*d`` with a
format wide enough to be exact on these inputs to ``nn.Conv*d``.
"""

import itertools
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.grad as tgrad

from mptorch import AccumulateAlgorithm, BinaryK, RoundMode, SuperFP
from mptorch.quant import (
    FusedMac,
    QAffineFormats,
    QConv1d,
    QConv2d,
    QConv3d,
    SplitMac,
    conv_formats,
)
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops import _run_gemm, _run_gemm_mixed

from .markers import float64_devices, requires_cuda, requires_mps

devices = float64_devices  # CPU and CUDA: the conv ops have no MPS kernel yet

NAIVE, KAHAN, BLOCK, TREE = (
    AccumulateAlgorithm.NAIVE,
    AccumulateAlgorithm.KAHAN,
    AccumulateAlgorithm.BLOCK,
    AccumulateAlgorithm.TREE,
)

B8 = BinaryK(8, 4)
B12 = BinaryK(12, 7)
S8 = SuperFP(3, 4, 8, 7)
S12 = SuperFP(5, 5, 16, 15)


# --- the references ------------------------------------------------------------


def _im2col(x, kernel, stride, padding, dilation, out):
    """``x [B, C, *in]`` unfolded to ``[B, C*prod(kernel), prod(out)]``.

    Row ``(c, i)`` and column ``o`` hold ``x[b, c, o*s - p + i*d]``, zero
    where that is outside the input, rows in ``F.unfold``'s order. Written
    with a gather rather than ``F.unfold`` because it has to take any number
    of spatial dimensions and a negative padding (the input gradient's
    reference crops where the forward padded more than the kernel reaches).
    """
    nd = len(kernel)
    B, C = x.shape[:2]
    size = x.shape[2:]
    grids = torch.meshgrid(
        *[torch.arange(k) for k in kernel], *[torch.arange(o) for o in out], indexing="ij"
    )
    flat = torch.zeros(grids[0].shape, dtype=torch.long)
    valid = torch.ones(grids[0].shape, dtype=torch.bool)
    for j in range(nd):
        pos = grids[nd + j] * stride[j] - padding[j] + grids[j] * dilation[j]
        valid &= (pos >= 0) & (pos < size[j])
        flat = flat * size[j] + pos.clamp(0, size[j] - 1)
    flat, valid = flat.to(x.device), valid.to(x.device)
    cols = x.reshape(B, C, -1)[:, :, flat.reshape(-1)]
    cols = torch.where(valid.reshape(-1), cols, torch.zeros((), dtype=x.dtype, device=x.device))
    kk = 1
    for k in kernel:
        kk *= k
    return cols.reshape(B, C * kk, -1)


def _gemm(spec, a, b, prec_idx=None):
    """The flat GEMM op a mac resolves to, batched over the leading dim."""
    if prec_idx is None:
        return _run_gemm(spec, a.contiguous(), b.contiguous(), False, False)
    return _run_gemm_mixed(spec, a.contiguous(), b.contiguous(), prec_idx, False, False)


def _map_rows(prec_idx, batch, groups, rows):
    """A result-shaped palette map as the reference's batched GEMM takes it:
    ``[groups * rows, N]`` (or ``[batch, groups * rows, N]``) to
    ``[batch * groups, rows, N]``."""
    if prec_idx is None:
        return None
    m = prec_idx if prec_idx.dim() == 3 else prec_idx.unsqueeze(0).expand(batch, -1, -1)
    return m.reshape(m.shape[0] * groups, rows, m.shape[-1])


def ref_fwd(spec, x, w, stride, padding, dilation, groups, out, prec_idx=None):
    B, C = x.shape[:2]
    Cout, Cg = w.shape[:2]
    kernel = w.shape[2:]
    cols = _im2col(x, kernel, stride, padding, dilation, out)
    G, Coutg, KK, L = groups, Cout // groups, cols.shape[1] // C, cols.shape[2]
    a = w.reshape(G, Coutg, Cg * KK).expand(B, -1, -1, -1).reshape(B * G, Coutg, Cg * KK)
    b = cols.reshape(B * G, Cg * KK, L)
    y = _gemm(spec, a, b, _map_rows(prec_idx, B, G, Coutg))
    return y.reshape(B, Cout, *out)


def _igrad_classes(size, kernel, stride, padding, dilation):
    """The input gradient's residue classes, as ``(positions, taps)`` per
    dimension: the input positions ``h`` with ``(h + p) % s == r`` and the
    taps ``j`` with ``j*d % s == r``, descending, in row-major order of ``r``;
    a class with no positions is left out."""
    per = []
    for n, k, s, p, d in zip(size, kernel, stride, padding, dilation, strict=True):
        per.append(
            [
                (list(range((r - p) % s, n, s)), [j for j in range(k) if j * d % s == r][::-1])
                for r in range(s)
            ]
        )
    return [c for c in itertools.product(*per) if all(h for h, _ in c)]


def ref_igrad(spec, dy, w, size, stride, padding, dilation, groups, prec_idx=None):
    """The input gradient as one GEMM per residue class of the stride.

    Input position ``h`` takes tap ``j`` of output position
    ``o = (h + p - j*d) / s``, which is whole for every position of a class and
    every tap of it, zero where ``o`` falls outside the output. Each class is a
    GEMM call of its own over ``(co, taps descending)``, whose result is
    scattered into its positions of ``dx``; at stride 1 the one class is the
    transposed convolution, :func:`ref_igrad_zero_inserted`.
    """
    B, Cout = dy.shape[:2]
    Cg = w.shape[1]
    kernel = w.shape[2:]
    nd = len(kernel)
    G, Coutg = groups, Cout // groups
    out = dy.shape[2:]
    IN = 1
    for n in size:
        IN *= n
    dx = dy.new_empty(B, G * Cg, IN)
    for cls in _igrad_classes(size, kernel, stride, padding, dilation):
        hs = [torch.tensor(h) for h, _ in cls]
        taps = [torch.tensor(t, dtype=torch.long) for _, t in cls]
        # The class's positions, flattened into dx's columns.
        hgrid = torch.meshgrid(*hs, indexing="ij")
        hflat = torch.zeros(hgrid[0].shape, dtype=torch.long)
        for j in range(nd):
            hflat = hflat * size[j] + hgrid[j]
        hflat = hflat.reshape(-1).to(dy.device)
        # dy at each (tap, position): cols [B, Cout, T, N].
        grids = torch.meshgrid(*taps, *hs, indexing="ij")
        flat = torch.zeros(grids[0].shape, dtype=torch.long)
        valid = torch.ones(grids[0].shape, dtype=torch.bool)
        for j in range(nd):
            o = (grids[nd + j] + padding[j] - grids[j] * dilation[j]) // stride[j]
            valid &= (o >= 0) & (o < out[j])
            flat = flat * out[j] + o.clamp(0, out[j] - 1)
        T = 1
        for t in taps:
            T *= len(t)
        N = hflat.numel()
        flat, valid = flat.reshape(-1).to(dy.device), valid.reshape(-1).to(dy.device)
        cols = dy.reshape(B, Cout, -1)[:, :, flat]
        cols = torch.where(valid, cols, torch.zeros((), dtype=dy.dtype, device=dy.device))
        b = cols.reshape(B * G, Coutg * T, N)
        # W at the class's taps: a [B*G, Cg, Coutg*T].
        wt = w.reshape(G, Coutg, Cg, *kernel)
        for j in range(nd):
            wt = wt.index_select(3 + j, taps[j].to(w.device))
        a = wt.transpose(1, 2).reshape(G, Cg, Coutg * T).expand(B, -1, -1, -1)
        a = a.reshape(B * G, Cg, Coutg * T)
        m = None
        if prec_idx is not None:
            m = prec_idx[..., hflat] if prec_idx.shape[-1] == IN else prec_idx
            m = _map_rows(m, B, G, Cg)
        dx[:, :, hflat] = _gemm(spec, a, b, m).reshape(B, G * Cg, N)
    return dx.reshape(B, G * Cg, *size)


def ref_igrad_zero_inserted(spec, dy, w, size, stride, padding, dilation, groups):
    """The transposed convolution as a convolution: ``dy`` zero-inserted by the
    stride, correlated with the flipped kernel, padded by ``d(k - 1) - p``.
    It sums the zeros the stride inserts too, which :func:`ref_igrad` leaves
    out; the two agree bit for bit on a NAIVE sum under a deterministic
    rounding, where a zero term changes nothing."""
    B, Cout = dy.shape[:2]
    Cg = w.shape[1]
    kernel = w.shape[2:]
    nd = len(kernel)
    G, Coutg = groups, Cout // groups
    zs = [(o - 1) * s + 1 for o, s in zip(dy.shape[2:], stride, strict=True)]
    dz = dy.new_zeros(B, Cout, *zs)
    dz[(slice(None), slice(None), *[slice(None, None, s) for s in stride])] = dy
    pad = [d * (k - 1) - p for k, d, p in zip(kernel, dilation, padding, strict=True)]
    cols = _im2col(dz, kernel, (1,) * nd, pad, dilation, size)  # [B, Cout*KK, IN]
    KK = cols.shape[1] // Cout
    wt = w.reshape(G, Coutg, Cg, *kernel).transpose(1, 2).flip(list(range(3, 3 + nd)))
    a = wt.reshape(G, Cg, Coutg * KK).expand(B, -1, -1, -1).reshape(B * G, Cg, Coutg * KK)
    b = cols.reshape(B * G, Coutg * KK, -1)
    dx = _gemm(spec, a, b)
    return dx.reshape(B, G * Cg, *size)


def ref_wgrad(spec, dy, x, kernel, stride, padding, dilation, groups, prec_idx=None):
    B, Cout = dy.shape[:2]
    C = x.shape[1]
    G, Coutg, Cg = groups, Cout // groups, C // groups
    out = dy.shape[2:]
    cols = _im2col(x, kernel, stride, padding, dilation, out)  # [B, C*KK, L]
    KK, L = cols.shape[1] // C, cols.shape[2]
    a = dy.reshape(B, G, Coutg, L).permute(1, 2, 0, 3).reshape(G, Coutg, B * L)
    b = cols.reshape(B, G, Cg * KK, L).permute(1, 0, 3, 2).reshape(G, B * L, Cg * KK)
    m = None if prec_idx is None else prec_idx.reshape(G, Coutg, -1)
    dw = _gemm(spec, a, b, m)
    return dw.reshape(Cout, Cg, *kernel)


# --- running the three hooks ---------------------------------------------------


def _hooks(formats: QAffineFormats):
    assert formats.fwd_math and formats.bwd_igrad_math and formats.bwd_wgrad_math
    return formats.fwd_math, formats.bwd_igrad_math, formats.bwd_wgrad_math


def _geometry(nd, stride, padding, dilation):
    t = lambda v: (v,) * nd if isinstance(v, int) else tuple(v)  # noqa: E731
    return t(stride), t(padding), t(dilation)


def _out(size, kernel, stride, padding, dilation):
    return tuple(
        (n + 2 * p - d * (k - 1) - 1) // s + 1
        for n, k, s, p, d in zip(size, kernel, stride, padding, dilation, strict=True)
    )


def _operands(B, C, Cout, groups, size, kernel, device, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, C, *size, generator=g)
    w = torch.randn(Cout, C // groups, *kernel, generator=g)
    return x.to(device, dtype), w.to(device, dtype)


def _seeded(device):
    torch.manual_seed(7)
    if device == "cuda":
        torch.cuda.manual_seed(7)


def _check_three_passes(
    mac,
    device,
    *,
    B=2,
    C=3,
    Cout=5,
    size=(9, 7),
    kernel=(3, 2),
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    dtype=torch.float32,
    maps=(None, None, None),
):
    nd = len(size)
    s, p, d = _geometry(nd, stride, padding, dilation)
    x, w = _operands(B, C, Cout, groups, size, kernel, device, dtype)
    out = _out(size, kernel, s, p, d)
    dy = torch.randn(B, Cout, *out, generator=torch.Generator().manual_seed(1)).to(device, dtype)
    fwd, igrad, wgrad = _hooks(
        conv_formats(mac, prec_idx=maps[0], igrad_prec_idx=maps[1], wgrad_prec_idx=maps[2])
    )
    spec = spec_for_mac(mac)
    geo = dict(stride=stride, padding=padding, dilation=dilation, groups=groups, nd=nd)

    _seeded(device)
    y = fwd(x, w, None, **geo)
    _seeded(device)
    y_ref = ref_fwd(spec, x, w, s, p, d, groups, out, maps[0])
    assert torch.equal(y, y_ref)

    _seeded(device)
    dx = igrad(dy, w, input_size=x.shape, **geo)
    _seeded(device)
    dx_ref = ref_igrad(spec, dy, w, size, s, p, d, groups, maps[1])
    assert torch.equal(dx, dx_ref)

    _seeded(device)
    dw = wgrad(dy, x, weight_size=w.shape, **geo)
    _seeded(device)
    dw_ref = ref_wgrad(spec, dy, x, kernel, s, p, d, groups, maps[2])
    assert torch.equal(dw, dw_ref)
    return y, dx, dw


# --- Tier 3: every pass against the unfolded GEMM ------------------------------


def _macs(mode):
    return {
        "bk_split": SplitMac(B8, B12, rounding=mode),
        "bk_split_noacc": SplitMac(B8, None, rounding=mode),
        "bk_fused": FusedMac(B12, rounding=mode),
        "sf_split": SplitMac(S8, S12, rounding=mode),
        "sf_fused": FusedMac(S12, rounding=mode),
    }


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", list(RoundMode), ids=lambda m: m.name)
@pytest.mark.parametrize("which", list(_macs(RoundMode.RNE)))
def test_tier3_every_mode(device, mode, which):
    _check_three_passes(_macs(mode)[which], device, stride=(2, 1), padding=(1, 0))


GEOMETRIES = {
    # nd, size, kernel, stride, padding, dilation, groups, C, Cout
    "1d": ((17,), (3,), 1, 1, 1, 1, 3, 5),
    "1d_strided": ((20,), (4,), 3, 2, 1, 1, 4, 6),
    "2d_plain": ((8, 8), (3, 3), 1, 0, 1, 1, 3, 4),
    "2d_all": ((11, 9), (3, 2), (2, 3), (1, 2), (2, 1), 1, 4, 6),
    "2d_groups": ((7, 6), (3, 3), 1, 1, 1, 2, 4, 6),
    "2d_depthwise": ((6, 7), (3, 3), 2, 1, 1, 4, 4, 8),
    "2d_big_pad": ((5, 5), (3, 3), 1, 3, 1, 1, 2, 3),
    "2d_1x1": ((6, 5), (1, 1), 1, 0, 1, 1, 7, 18),
    "2d_1x1_strided": ((7, 6), (1, 1), 2, 0, 1, 1, 3, 5),
    "2d_dilation_shares_stride": ((12, 11), (3, 3), 2, 2, 2, 1, 3, 4),
    "2d_wide": ((19, 23), (5, 3), 2, 2, 1, 1, 5, 17),
    "3d": ((5, 6, 4), (3, 2, 2), 1, 1, 1, 1, 2, 3),
    "3d_all": ((7, 5, 6), (2, 3, 2), (2, 1, 2), (0, 1, 1), (1, 2, 2), 2, 4, 2),
}


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RZ, RoundMode.SR], ids=lambda m: m.name)
def test_tier3_geometries(device, geometry, mode):
    size, kernel, stride, padding, dilation, groups, C, Cout = GEOMETRIES[geometry]
    for mac in (SplitMac(B8, B12, rounding=mode), FusedMac(S12, rounding=mode)):
        _check_three_passes(
            mac,
            device,
            C=C,
            Cout=Cout,
            size=size,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )


def _accumulated(mode):
    out = []
    for alg, bs, outer in (
        (KAHAN, None, None),
        (BLOCK, 4, None),
        (BLOCK, 32, B12),
        (TREE, 8, None),
        (TREE, 4, B12),
    ):
        out.append(
            SplitMac(B8, B12, rounding=mode, accumulate_algorithm=alg, block_size=bs, outer=outer)
        )
        if alg is not TREE:
            out.append(
                FusedMac(S12, rounding=mode, accumulate_algorithm=alg, block_size=bs, outer=None)
            )
    return out


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RU, RoundMode.SR], ids=lambda m: m.name)
def test_tier3_accumulate_algorithms(device, mode):
    # K = C*KK = 4*9 = 36 in the forward, 6*9 = 54 in the input gradient and
    # B*L = 2*63 = 126 in the weight gradient: partial slabs and partial
    # blocks in all three.
    for mac in _accumulated(mode):
        _check_three_passes(mac, device, C=4, Cout=6, size=(9, 7), kernel=(3, 3), padding=1)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.SR], ids=lambda m: m.name)
@pytest.mark.parametrize("stride", [1, 2])
def test_tier3_palette(device, mode, stride):
    mac = SplitMac([B8, BinaryK(8, 5)], [B12, BinaryK(12, 8)], rounding=mode)
    B, C, Cout, groups, size, kernel = 2, 4, 6, 2, (7, 6), (3, 3)
    out = _out(size, kernel, (stride,) * 2, (1, 1), (1, 1))
    L = out[0] * out[1]
    g = torch.Generator().manual_seed(3)
    maps = (
        torch.randint(0, 2, (Cout, L), generator=g).to(device),  # per output element
        # per input element: a strided input gradient's classes read the
        # map at their own positions
        torch.randint(0, 2, (C, size[0] * size[1]), generator=g).to(device),
        torch.randint(0, 2, (Cout, C // groups * 9), generator=g).to(device),
    )
    _check_three_passes(
        mac,
        device,
        B=B,
        C=C,
        Cout=Cout,
        size=size,
        kernel=kernel,
        stride=stride,
        padding=1,
        groups=groups,
        maps=maps,
    )
    # A per-channel map too.
    maps = (maps[0], torch.randint(0, 2, (C, 1), generator=g).to(device), maps[2])
    _check_three_passes(
        mac, device, B=B, C=C, Cout=Cout, size=size, kernel=kernel, stride=stride, padding=1,
        groups=groups, maps=maps,
    )  # fmt: skip


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tier3_narrow_storage(device, dtype):
    _check_three_passes(SplitMac(B8, B12), device, padding=1, stride=2, dtype=dtype)


@pytest.mark.parametrize("device", devices)
def test_same_padding_even_kernel(device):
    # "same" with an even kernel pads one more position after than before,
    # which the op reads as zero past the input's end without a copy.
    mac = SplitMac(B8, B12)
    x, w = _operands(2, 3, 4, 1, (8, 9), (4, 2), device)
    fwd, igrad, wgrad = _hooks(conv_formats(mac))
    geo: dict[str, Any] = dict(stride=1, padding="same", dilation=(1, 2), groups=1, nd=2)
    y = fwd(x, w, None, **geo)
    assert y.shape == (2, 4, 8, 9)
    spec = spec_for_mac(mac)
    lead = (3 * 1 // 2, 1 * 2 // 2)  # d*(k-1)//2 per dimension
    assert torch.equal(y, ref_fwd(spec, x, w, (1, 1), lead, (1, 2), 1, (8, 9)))
    dy = torch.randn_like(y)
    dx = igrad(dy, w, input_size=x.shape, **geo)
    assert torch.equal(dx, ref_igrad(spec, dy, w, (8, 9), (1, 1), lead, (1, 2), 1))
    dw = wgrad(dy, x, weight_size=w.shape, **geo)
    assert torch.equal(dw, ref_wgrad(spec, dy, x, (4, 2), (1, 1), lead, (1, 2), 1))


@pytest.mark.parametrize("device", devices)
def test_output_padding_rows_are_zero(device):
    # A stride that does not divide the input leaves its last rows out of
    # every window: their gradient is an empty sum, an unsigned zero.
    mac = SplitMac(B8, B12)
    x, w = _operands(1, 2, 3, 1, (8, 8), (3, 3), device)
    _, igrad, _ = _hooks(conv_formats(mac))
    # (8 - 3) // 3 + 1 = 2 windows per dimension, over rows 0-2 and 3-5.
    dy = torch.randn(1, 3, 2, 2).to(device)
    dx = igrad(dy, w, input_size=x.shape, stride=3, padding=0, dilation=1, groups=1, nd=2)
    assert torch.equal(dx[..., 6:, :], torch.zeros_like(dx[..., 6:, :]))
    assert not torch.signbit(dx[..., 6:, :]).any()
    assert (dx[..., :6, :6] != 0).all()


# --- the strided input gradient against the zero-inserted transposed conv -------

STRIDED = {
    # size, kernel, stride, padding, dilation, groups
    "2d_s2": ((11, 10), (3, 3), 2, 1, 1, 1),
    "2d_s3x2_d2": ((13, 12), (3, 2), (3, 2), (1, 0), (2, 1), 2),
    "1d_s3": ((20,), (4,), 3, 2, 1, 1),
    "3d_s2": ((7, 6, 5), (3, 2, 3), 2, 1, 1, 1),
    "2d_s2_1x1": ((9, 8), (1, 1), 2, 0, 1, 1),
    "2d_s4_k2": ((13, 9), (2, 2), 4, 1, 1, 1),
}


def _strided_operands(geometry, device, seed=0):
    size, kernel, stride, padding, dilation, groups = STRIDED[geometry]
    nd = len(size)
    s, p, d = _geometry(nd, stride, padding, dilation)
    x, w = _operands(2, 4, 6, groups, size, kernel, device, seed=seed)
    out = _out(size, kernel, s, p, d)
    dy = torch.randn(2, 6, *out, generator=torch.Generator().manual_seed(seed + 1)).to(device)
    geo = dict(stride=stride, padding=padding, dilation=dilation, groups=groups, nd=nd)
    return x, w, dy, geo, (size, s, p, d, groups)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("geometry", list(STRIDED))
@pytest.mark.parametrize(
    "mode", [m for m in RoundMode if m is not RoundMode.SR], ids=lambda m: m.name
)
def test_strided_igrad_is_the_zero_inserted_sum_under_naive(device, geometry, mode):
    # Leaving out the zeros a stride inserts changes nothing where adding a
    # zero changes nothing: a NAIVE sum, rounded deterministically. The
    # product of a weight and an inserted zero rounds to zero, and the sum
    # plus zero is the sum, which the accumulate format already holds.
    x, w, dy, geo, (size, s, p, d, groups) = _strided_operands(geometry, device)
    for mac in _macs(mode).values():
        _, igrad, _ = _hooks(conv_formats(mac))
        dx = igrad(dy, w, input_size=x.shape, **geo)
        ref = ref_igrad_zero_inserted(spec_for_mac(mac), dy, w, size, s, p, d, groups)
        assert torch.equal(dx, ref)


def _relative_error(approx, exact):
    return ((approx.cpu().double() - exact).norm() / exact.norm()).item()


# Larger than STRIDED's, so that a mean over a few draws of the data resolves
# a few percent: (size, kernel, stride, padding, channels, stride**nd).
ACCURACY = {
    "2d_s2": ((20, 20), (3, 3), (2, 2), (1, 1), 12, 4),
    "3d_s2": ((8, 8, 8), (3, 3, 3), (2, 2, 2), (1, 1, 1), 6, 8),
}


def _mean_errors(device, geometry, mac, zero_inserted_mac, seeds=6):
    """Mean relative error, against the float64 input gradient, of the conv
    op's and of the zero-inserted transposed convolution's."""
    size, kernel, stride, padding, C, _ = ACCURACY[geometry]
    nd = len(size)
    conv_input = {2: tgrad.conv2d_input, 3: tgrad.conv3d_input}[nd]
    _, igrad, _ = _hooks(conv_formats(mac))
    new = old = 0.0
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(2, C, *size, generator=g)
        w = torch.randn(C, C, *kernel, generator=g)
        out = _out(size, kernel, stride, padding, (1,) * nd)
        dy = torch.randn(2, C, *out, generator=g)
        exact = conv_input(x.shape, w.double(), dy.double(), stride, padding)
        x, w, dy = x.to(device), w.to(device), dy.to(device)
        _seeded(device)
        dx = igrad(dy, w, input_size=x.shape, stride=stride, padding=padding, dilation=1,
                   groups=1, nd=nd)  # fmt: skip
        _seeded(device)
        ref = ref_igrad_zero_inserted(
            spec_for_mac(zero_inserted_mac), dy, w, size, stride, padding, (1,) * nd, 1
        )
        new += _relative_error(dx, exact) / seeds
        old += _relative_error(ref, exact) / seeds
    return new, old


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("geometry", list(ACCURACY))
@pytest.mark.parametrize(
    "name",
    ["naive_rne", "naive_sr", "fma_sr", "kahan", "kahan_sr"],
)
def test_strided_igrad_is_as_accurate_as_the_zero_inserted_sum(device, geometry, name):
    # Where a zero term does change the sum -- a stochastic rounding spends a
    # draw on it, KAHAN applies its pending compensation -- the sum over the
    # terms that exist is another rounding of the same exact value. Against
    # the float64 gradient it is as accurate as the zero-inserted sum.
    sr8, sr12 = BinaryK(8, 4, prng_bits=8), BinaryK(12, 7, prng_bits=8)
    mac = {
        "naive_rne": SplitMac(B8, BinaryK(10, 5)),
        "naive_sr": SplitMac(sr8, sr12, rounding=RoundMode.SR),
        "fma_sr": FusedMac(sr12, rounding=RoundMode.SR),
        "kahan": SplitMac(B8, BinaryK(10, 5), accumulate_algorithm=KAHAN),
        "kahan_sr": SplitMac(sr8, BinaryK(10, 5, prng_bits=8), rounding=RoundMode.SR,
                             accumulate_algorithm=KAHAN),
    }[name]  # fmt: skip
    new, old = _mean_errors(device, geometry, mac, mac)
    assert new <= 1.05 * old, (new, old)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("geometry", list(ACCURACY))
@pytest.mark.parametrize("alg, block_size", [(BLOCK, 2), (BLOCK, 4), (TREE, 4), (TREE, 8)])
def test_strided_igrad_blocks_count_terms_that_exist(device, geometry, alg, block_size):
    # BLOCK and TREE count a block in K steps. In the zero-inserted sum only
    # one step in stride**nd is a term, so a block of b steps held about
    # b / stride**nd terms, and more of the sum was done in the outer format;
    # now a block holds b terms, as it does in the other two passes and in a
    # GEMM. Matched block for block -- b here against b * stride**nd there --
    # the two are as accurate (measured 0.97-0.99 of the zero-inserted error).
    ratio = ACCURACY[geometry][-1]
    mac = SplitMac(B8, BinaryK(10, 5), accumulate_algorithm=alg, block_size=block_size, outer=B12)
    wide = SplitMac(
        B8, BinaryK(10, 5), accumulate_algorithm=alg, block_size=block_size * ratio, outer=B12
    )
    new, old = _mean_errors(device, geometry, mac, wide)
    assert new <= 1.05 * old, (new, old)


# --- Tier 1: the layers against nn.Conv ------------------------------------------


# 24 bits of precision and binary32's range: every product and sum of these
# small-integer inputs is exact, so any order of summation gives torch's.
EXACT = BinaryK(32, 24)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize(
    "nd, kwargs",
    [
        (1, dict(kernel_size=3, stride=2, padding=1)),
        (2, dict(kernel_size=(3, 2), stride=(1, 2), padding=(1, 0), dilation=(2, 1))),
        (2, dict(kernel_size=3, padding="same", groups=2)),
        (2, dict(kernel_size=3, padding=1, padding_mode="reflect")),
        (3, dict(kernel_size=2, stride=1, padding=1, bias=False)),
    ],
)
def test_tier1_layers_match_torch(device, nd, kwargs):
    layer_cls = {1: (QConv1d, nn.Conv1d), 2: (QConv2d, nn.Conv2d), 3: (QConv3d, nn.Conv3d)}[nd]
    torch.manual_seed(0)
    ref = layer_cls[1](4, 6, **kwargs).to(device)
    q = layer_cls[0](4, 6, **kwargs, formats=conv_formats(SplitMac(EXACT, EXACT))).to(device)
    with torch.no_grad():
        # Small integers keep every product and partial sum exact.
        ref.weight.copy_(torch.randint(-3, 4, ref.weight.shape))
        q.weight.copy_(ref.weight)
        if ref.bias is not None and q.bias is not None:
            ref.bias.copy_(torch.randint(-3, 4, ref.bias.shape))
            q.bias.copy_(ref.bias)
    size = (1, 4, *([9] * nd))
    x = torch.randint(-3, 4, size).float().to(device).requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    y, y2 = ref(x), q(x2)
    assert torch.equal(y, y2)
    g = torch.randint(-3, 4, y.shape).float().to(device)
    y.backward(g)
    y2.backward(g)
    assert x.grad is not None and x2.grad is not None
    assert torch.equal(x.grad, x2.grad)
    assert ref.weight.grad is not None and q.weight.grad is not None
    assert torch.equal(ref.weight.grad, q.weight.grad)
    if ref.bias is not None:
        assert q.bias is not None and ref.bias.grad is not None and q.bias.grad is not None
        assert torch.equal(ref.bias.grad, q.bias.grad)


@pytest.mark.parametrize("device", devices)
def test_tier1_unbatched_input(device):
    q = QConv2d(3, 4, 3, padding=1, formats=conv_formats(SplitMac(EXACT, EXACT))).to(device)
    x = torch.randint(-3, 4, (3, 6, 5)).float().to(device).requires_grad_()
    y = q(x)
    assert y.shape == (4, 6, 5)
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


@pytest.mark.parametrize("device", devices)
def test_tier2_close_to_torch(device):
    # A real format: close to nn.Conv2d, not equal to it.
    torch.manual_seed(0)
    ref = nn.Conv2d(8, 16, 3, padding=1).to(device)
    q = QConv2d(8, 16, 3, padding=1, formats=conv_formats(SplitMac(BinaryK(16, 11)))).to(device)
    q.load_state_dict(ref.state_dict())
    x = torch.randn(4, 8, 12, 12, device=device)
    y, yq = ref(x), q(x)
    assert not torch.equal(y, yq)
    assert torch.allclose(y, yq, rtol=0.02, atol=0.05)


# --- partial gradients, shapes and errors --------------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("pattern", list(itertools.product([False, True], repeat=3)))
def test_partial_grad_patterns(device, pattern):
    x_grad, w_grad, b_grad = pattern
    torch.manual_seed(0)
    q = QConv2d(3, 4, 3, padding=1, formats=conv_formats(SplitMac(B8, B12))).to(device)
    assert q.bias is not None
    q.weight.requires_grad_(w_grad)
    q.bias.requires_grad_(b_grad)
    x = torch.randn(2, 3, 6, 6, device=device, requires_grad=x_grad)
    y = q(x)
    if not y.requires_grad:
        return
    y.sum().backward()
    assert (x.grad is not None) == x_grad
    assert (q.weight.grad is not None) == w_grad
    assert (q.bias.grad is not None) == b_grad


def test_empty_batch():
    fwd, igrad, wgrad = _hooks(conv_formats(SplitMac(B8, B12)))
    x, w = _operands(0, 3, 4, 1, (6, 6), (3, 3), "cpu")
    y = fwd(x, w, None, stride=1, padding=0, dilation=1, groups=1, nd=2)
    assert y.shape == (0, 4, 4, 4)
    dy = torch.randn(0, 4, 4, 4)
    dw = wgrad(dy, x, weight_size=w.shape, stride=1, padding=0, dilation=1, groups=1, nd=2)
    assert torch.equal(dw, torch.zeros_like(dw))


@pytest.mark.parametrize("device", devices)
def test_float64_raises_naming_phase_g(device):
    fwd, _, _ = _hooks(conv_formats(SplitMac(B8, B12)))
    x, w = _operands(1, 2, 3, 1, (5, 5), (3, 3), device, torch.float64)
    with pytest.raises(ValueError, match="phase G"):
        fwd(x, w, None, stride=1, padding=0, dilation=1, groups=1, nd=2)
    with pytest.raises(ValueError, match="phase G"):
        conv_formats(SplitMac(B8, B12, carrier=torch.float64))
    # The raw op refuses a float64 call too, with the same pointer.
    with pytest.raises(RuntimeError, match="phase G|MPTORCH_NO_FP64"):
        _raw_fwd(x, w)


def _raw_fwd(x, w, conv_pass=0, out_size=(3, 3), groups=1, **over):
    spec = spec_for_mac(SplitMac(B8, B12))
    tail = (0, False, 0, 0, 0, True, 0, 0, 0)
    args = list(spec.args) + list(tail)
    for name, value in over.items():
        args[{"accumulate_algorithm": 9, "round_mode": 10, "block_size": 17}[name]] = value
    return torch.ops.mptorch.custom_conv_binaryK(
        x if conv_pass else w,
        w if conv_pass else x,
        conv_pass,
        list(out_size),
        [1, 1],
        [0, 0],
        [1, 1],
        groups,
        *args,
    )


def test_raw_op_validation():
    x, w = _operands(1, 4, 6, 1, (5, 5), (3, 3), "cpu")
    ops = torch.ops.mptorch
    with pytest.raises(RuntimeError, match="conv_pass must be"):
        _raw_fwd(x, w, conv_pass=3)
    with pytest.raises(RuntimeError, match="channels per group over 4 groups takes 16"):
        _raw_fwd(x, w, groups=4)
    with pytest.raises(RuntimeError, match="6 output channels do not split into 4"):
        _raw_fwd(x, w[:, :1], groups=4)
    with pytest.raises(RuntimeError, match="is not a RoundMode"):
        _raw_fwd(x, w, round_mode=99)
    with pytest.raises(RuntimeError, match="is not an AccumulateAlgorithm"):
        _raw_fwd(x, w, accumulate_algorithm=99)
    with pytest.raises(RuntimeError, match="NAIVE takes no block_size"):
        _raw_fwd(x, w, block_size=4)
    with pytest.raises(RuntimeError, match="takes operands of rank 4"):
        _raw_fwd(x[0], w)
    spec = spec_for_mac(SplitMac(B8, B12))
    with pytest.raises(RuntimeError, match="stride, padding and dilation"):
        ops.custom_conv_binaryK(w, x, 0, [3, 3], [1], [0, 0], [1, 1], 1, *spec.args,
                                0, False, 0, 0, 0, True, 0, 0, 0)  # fmt: skip
    with pytest.raises(RuntimeError, match="must not be negative"):
        _raw_fwd(x, w, out_size=(-1, 3))


def test_palette_errors():
    mac = SplitMac([B8, BinaryK(8, 5)], B12)
    with pytest.raises(ValueError, match="needs a prec_idx"):
        conv_formats(mac)
    with pytest.raises(ValueError, match="no meaning"):
        conv_formats(SplitMac(B8, B12), prec_idx=torch.zeros(1, 1, dtype=torch.int64))
    f = conv_formats(mac, prec_idx=torch.zeros(3, 1, dtype=torch.int64))
    x, w = _operands(1, 2, 3, 1, (5, 5), (3, 3), "cpu")
    assert f.bwd_igrad_math is not None
    with pytest.raises(ValueError, match="igrad_prec_idx"):
        f.bwd_igrad_math(torch.randn(1, 3, 3, 3), w, input_size=x.shape, stride=1, padding=0,
                         dilation=1, groups=1, nd=2)  # fmt: skip


def test_number_shorthand_is_a_split_mac():
    x, w = _operands(1, 2, 3, 1, (5, 5), (3, 3), "cpu")
    geo: dict[str, Any] = dict(stride=1, padding=0, dilation=1, groups=1, nd=2)
    a, _, _ = _hooks(conv_formats(B8))
    b, _, _ = _hooks(conv_formats(SplitMac(B8, B8)))
    assert torch.equal(a(x, w, None, **geo), b(x, w, None, **geo))


def test_raw_op_raises_on_grad():
    x, w = _operands(1, 4, 6, 1, (5, 5), (3, 3), "cpu")
    with pytest.raises(RuntimeError, match="conv_formats"):
        _raw_fwd(x.requires_grad_(), w)


@requires_mps
def test_mps_raises_naming_phase_h():
    fwd, _, _ = _hooks(conv_formats(SplitMac(B8, B12)))
    x, w = _operands(1, 2, 3, 1, (5, 5), (3, 3), "mps")
    with pytest.raises(RuntimeError, match="phase H"):
        fwd(x, w, None, stride=1, padding=0, dilation=1, groups=1, nd=2)


# --- memory: no unfold buffer ----------------------------------------------------


@requires_cuda
def test_no_unfold_buffer():
    # A (32, 64, 56, 56) 3x3 convolution: its unfold would be 9x the input,
    # 231 MB. The gathered passes allocate their results and nothing else.
    torch.manual_seed(0)
    q = QConv2d(64, 64, 3, padding=1, bias=False, formats=conv_formats(SplitMac(B8, B12)))
    q = q.cuda()
    x = torch.randn(32, 64, 56, 56, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = q(x)
    y.backward(torch.ones_like(y))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    tensor = x.numel() * 4
    unfold = 9 * tensor
    # y, its gradient and x's gradient, plus the weight's: about three
    # tensors of x's size; an unfold of the input would be nine on its own.
    assert peak < 4 * tensor < unfold


@requires_cuda
def test_batch_past_grid_z():
    # The groups ride on the kernels' batch dimension, so a depthwise layer
    # reaches CUDA's 65,535 grid-z limit at a modest batch: 60 samples of
    # 1,100 channels are 66,000 batch elements, which the launcher splits.
    mac = SplitMac(B8, B12, rounding=RoundMode.SR)
    x, w = _operands(60, 1100, 1100, 1100, (4, 3), (2, 2), "cuda")
    fwd, _, _ = _hooks(conv_formats(mac))
    _seeded("cuda")
    y = fwd(x, w, None, stride=1, padding=0, dilation=1, groups=1100, nd=2)
    _seeded("cuda")
    ref = ref_fwd(spec_for_mac(mac), x, w, (1, 1), (0, 0), (1, 1), 1100, (3, 2))
    assert torch.equal(y, ref)
