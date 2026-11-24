"""Sums and means on the GEMM kernels: ``qsum`` and ``qmean``.

Tier 1: with every slot ``None`` they are torch's, gradients included. Tier 3:
a sum is the accumulation ``test_accumulators``' reference computes for one
output per row, step by step in the order of the summed elements, under every
accumulate algorithm, deterministic mode and carrier, round to odd included;
stochastic rounding is the GEMM's own call under the same generator, whatever
the layout of the summed dimensions. A mean is that sum's single rounded
division, and the gradients are the broadcast incoming gradient rounded once.
"""

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from mptorch import AccumulateAlgorithm, BinaryK, RoundMode, SuperFP
from mptorch.number import MXFP8_E4M3
from mptorch.quant import FusedMac, Palette, Quant, ReductionFormats, SplitMac, qmean, qsum
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops._spec import _run_gemm
from tests.markers import HAS_ODD, available_devices, float64_devices, requires_mps
from tests.test_accumulators import _OddReference, _Reference

R = RoundMode
A = AccumulateAlgorithm
P8 = BinaryK(8, 4)
P16 = BinaryK(16, 11)
DETERMINISTIC = [R.RNE, R.RNA, R.RU, R.RD, R.RZ, R.RO]


def _x(device, shape=(3, 4, 5), dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)


def _same(a, b) -> bool:
    """Equal words, NaN where NaN: same dtype, shape and values."""
    if a is None or b is None:
        return a is b
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    nan = a.isnan()
    return torch.equal(nan, b.isnan()) and torch.equal(a[~nan], b[~nan])


def _grad(out, x, seed=1):
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return torch.autograd.grad(out, (x,), g.to(out))[0]


def _rows(x, dims):
    """``x`` as ``[R, K]``: the summed dimensions last and flattened, in
    their order, the others first, the order ``qsum`` sums in."""
    kept = [i for i in range(x.dim()) if i not in dims]
    return x.permute(*kept, *dims).reshape(-1, _count(x, dims))


def _count(x, dims):
    n = 1
    for d in dims:
        n *= x.shape[d]
    return n


def _kept(x, dims, keepdim):
    if keepdim:
        return tuple(1 if i in dims else s for i, s in enumerate(x.shape))
    return tuple(s for i, s in enumerate(x.shape) if i not in dims)


LAYOUTS = [(-1,), (1,), (0,), (1, 2), (0, 1), (0, 2), (0, 1, 2)]


# --- tier 1: torch's ---------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("dim", [None, -1, 1, (0, 2), ()])
@pytest.mark.parametrize("keepdim", [False, True])
def test_without_slots_they_are_torchs(device, dtype, dim, keepdim):
    if device == "mps" and dtype is torch.float64:
        pytest.skip("MPS has no float64")
    x = _x(device, dtype=dtype).requires_grad_()
    for ours, theirs in ((qsum, torch.sum), (qmean, torch.mean)):
        got = ours(x, dim, keepdim=keepdim)
        want = theirs(x, dim=None if dim == () else dim, keepdim=keepdim)
        assert _same(got, want)
        assert _same(_grad(got, x), _grad(want, x))
    assert _same(qsum(x), torch.sum(x)) and _same(qmean(x), torch.mean(x))


def test_integers_sum_as_torch_does():
    x = torch.arange(12).reshape(3, 4)
    assert torch.equal(qsum(x, 1), x.sum(1))


# --- tier 3: the GEMM's accumulation -------------------------------------------------


def _reference(x, dims, mul, add, mode, algorithm=A.NAIVE, block_size=None, outer=None, odd=False):
    """``test_accumulators``' reference for one output per row of ``x``'s
    summed dimensions: ``a`` the rows, ``b`` a ones column, on the CPU in
    the carrier."""
    carrier = torch.float64 if x.dtype is torch.float64 else torch.float32
    rows = _rows(x.detach().cpu().to(carrier), dims)
    ones = torch.ones(rows.shape[1], 1, dtype=carrier)
    ref = (_OddReference if odd else _Reference)(rows, ones, mul, add, outer, mode)
    return ref.run(algorithm, block_size).reshape(-1)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
@pytest.mark.parametrize("dims", LAYOUTS, ids=str)
def test_tier3_naive(device, mode, dims):
    """A fused mac rounds ``s + x_k`` once in the carrier, then to the
    format; a split one rounds each ``x_k`` to its multiply format first."""
    x = _x(device)
    for mac, (mul, add) in (
        (FusedMac(P8, rounding=mode), (None, P8)),
        (SplitMac(P8, P16, rounding=mode), (P8, P16)),
        (SplitMac(P8, None, rounding=mode), (P8, None)),
    ):
        got = qsum(x, dims, mac)
        want = _reference(x, tuple(d % 3 for d in dims), mul, add, mode)
        assert torch.equal(got.reshape(-1).cpu(), want), (mac, dims)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tier3_narrow_storage(device, dtype):
    """A float16 or bfloat16 tensor sums in binary32, stored in its dtype once."""
    x = _x(device, dtype=dtype)
    got = qsum(x, -1, FusedMac(P8))
    want = _reference(x, (2,), None, P8, R.RNE).to(dtype)
    assert got.dtype is dtype and torch.equal(got.reshape(-1).cpu(), want)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("mode", [R.RNE, R.RU], ids=lambda m: m.name)
def test_tier3_binary64(device, mode):
    """A float64 tensor sums in binary64, and so does a float32 one under
    ``carrier=torch.float64``, its result narrowed once."""
    wide = BinaryK(40, 33)
    x = _x(device, dtype=torch.float64)
    got = qsum(x, (0, 2), FusedMac(wide, rounding=mode))
    want = _reference(x, (0, 2), None, wide, mode)
    assert torch.equal(got.reshape(-1).cpu(), want)
    x32 = _x(device)
    got = qsum(x32, -1, FusedMac(P16, rounding=mode, carrier=torch.float64))
    want = _reference(x32.double(), (2,), None, P16, mode).float()
    assert got.dtype is torch.float32 and torch.equal(got.reshape(-1).cpu(), want)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "algorithm,block_size",
    [(A.KAHAN, None), (A.BLOCK, 4), (A.BLOCK, 32), (A.TREE, 4), (A.TREE, 16)],
    ids=lambda v: getattr(v, "name", str(v)),
)
@pytest.mark.parametrize("dims", [(-1,), (1,), (0, 2)], ids=str)
def test_tier3_accumulate_algorithms(device, algorithm, block_size, dims):
    """KAHAN, BLOCK and TREE (a split mac only) sum each row as the GEMM's
    reference does, over 40 elements so that blocks are partial."""
    x = _x(device, shape=(4, 40, 3))
    family = SuperFP(3, 4, 8, 7) if algorithm is A.BLOCK else P8
    outer = P16 if algorithm is not A.KAHAN else None
    macs = [
        (
            SplitMac(
                family, P16, accumulate_algorithm=algorithm, block_size=block_size, outer=outer
            ),
            (family, P16),
        )
    ]
    if algorithm is not A.TREE:
        macs.append(
            (
                FusedMac(P16, accumulate_algorithm=algorithm, block_size=block_size, outer=outer),
                (None, P16),
            )
        )
    for mac, (mul, add) in macs:
        got = qsum(x, dims, mac)
        want = _reference(
            x, tuple(d % 3 for d in dims), mul, add, R.RNE, algorithm, block_size, outer
        )
        assert torch.equal(got.reshape(-1).cpu(), want), (mac, dims)


@pytest.mark.skipif(not HAS_ODD, reason="this build has no odd carrier step (MPTORCH_NO_ODD=1)")
@pytest.mark.parametrize("device", [d for d in available_devices if d != "mps"] or ["cpu"])
@pytest.mark.parametrize("mode", [R.RNE, R.RU, R.RD], ids=lambda m: m.name)
def test_tier3_rounded_to_odd(device, mode):
    """Under ``carrier_rounding=RO`` each step is the format's rounding of the
    exact sum."""
    x = _x(device, shape=(3, 33))
    got = qsum(x, -1, FusedMac(P8, rounding=mode, carrier_rounding=R.RO))
    want = _reference(x, (1,), None, P8, mode, odd=True)
    assert torch.equal(got.reshape(-1).cpu(), want)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", [R.SR, R.SR_B, R.SR_C], ids=lambda m: m.name)
def test_stochastic_rounding_is_the_gemms(device, mode):
    """Under SR a sum is the GEMM over the rows its layout sums, under the
    same generator: the draws are keyed by output index, which a batched
    layout and a flattened one share."""
    fmt = BinaryK(8, 4, prng_bits=8)
    mac = FusedMac(fmt, rounding=mode)
    spec = spec_for_mac(mac)
    for dims in ((-1,), (1,), (0, 2)):
        x = _x(device)
        d = tuple(i % 3 for i in dims)
        rows = _rows(x, d).contiguous()
        torch.manual_seed(5)
        got = qsum(x, dims, mac)
        torch.manual_seed(5)
        want = _run_gemm(spec, torch.ones(1, rows.shape[1], device=device), rows, False, True)
        assert torch.equal(got.reshape(-1), want.reshape(-1)), dims


@pytest.mark.parametrize("device", available_devices)
def test_the_two_orientations_agree(device):
    """``ones @ x.mT``, which ``qsum`` runs, and ``x @ ones`` give the same words."""
    x = _x(device, shape=(7, 37))
    for mac in (FusedMac(P8), SplitMac(P8, P16, accumulate_algorithm=A.KAHAN)):
        spec = spec_for_mac(mac)
        col = _run_gemm(spec, x, torch.ones(37, 1, device=device), False, False)
        assert torch.equal(qsum(x, -1, mac), col.reshape(-1))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dim", [None, -1, (0, 2), [2, 0], ()], ids=str)
def test_shapes(device, keepdim, dim):
    x = _x(device)
    got = qsum(x, dim, P8, keepdim=keepdim)
    want = torch.sum(x, dim=None if dim == () else dim, keepdim=keepdim)
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1.0)


@pytest.mark.parametrize("device", available_devices)
def test_edges(device):
    """A 0-dim tensor, an empty sum (zero) and an empty result."""
    s = qsum(torch.tensor(2.6, device=device), 0, P8)
    assert s.shape == () and s.item() == 2.5
    assert torch.equal(qsum(torch.ones(3, 0, device=device), 1, P8), torch.zeros(3, device=device))
    assert qsum(torch.ones(0, 4, device=device), 1, P8).shape == (0,)


# --- the mean, and the gradients ---------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dims", [(-1,), (0, 2)], ids=str)
def test_a_mean_is_the_sum_divided_once(device, dims):
    x = _x(device)
    q = Quant(P8, R.RU)
    got = qmean(x, dims, P16, quant=q)
    s = qsum(x, dims, P16)
    n = 1
    for d in dims:
        n *= x.shape[d]
    want = q(s / torch.full((), n, dtype=s.dtype, device=device))
    assert torch.equal(got, want)
    # no division slot: torch's division in the tensor's dtype
    assert torch.equal(qmean(x, dims, P16), s / n)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("keepdim", [False, True])
def test_the_gradient_is_broadcast_then_rounded(device, keepdim):
    x = _x(device).requires_grad_()
    q = Quant(P8)
    for op, scale in ((qsum, 1), (qmean, 15)):
        out = op(x, (0, 2), P16, keepdim=keepdim, grad_quant=q)
        g = torch.randn(out.shape, generator=torch.Generator().manual_seed(1)).to(device)
        (gx,) = torch.autograd.grad(out, (x,), g)
        gk = g.reshape(1, 4, 1) if not keepdim else g
        want = (
            gk.expand(x.shape) / torch.full((), scale, device=device)
            if scale > 1
            else gk.expand(x.shape)
        )
        assert torch.equal(gx, q(want.contiguous()))


@pytest.mark.parametrize("device", available_devices)
def test_the_gradient_alone_keeps_torchs_forward(device):
    x = _x(device).requires_grad_()
    out = qsum(x, -1, grad_quant=Quant(P8))
    assert torch.equal(out, x.sum(-1))


# --- the formats, and what is refused ------------------------------------------------------


def test_reduction_formats():
    f = ReductionFormats.uniform(P16, Quant(P8))
    assert f.mac == FusedMac(P16) and f.div == Quant(P8) and f.grad == Quant(P8)
    assert f._spec == spec_for_mac(FusedMac(P16))
    assert (
        f.carrier is None
        and ReductionFormats(mac=FusedMac(P8, carrier=torch.float64)).carrier is torch.float64
    )
    assert "mac FusedMac" in f.label() and ReductionFormats().label() == "torch"
    assert ReductionFormats(div=P8).div == Quant(P8)
    assert ReductionFormats.uniform(P16, Quant(P8), grad=None).grad is None


@pytest.mark.parametrize(
    "call,error,match",
    [
        (lambda x: qsum(x, -1, MXFP8_E4M3), TypeError, "block format"),
        (lambda x: qsum(x, -1, FusedMac(Palette([P8, P16]))), ValueError, "palette"),
        (lambda x: qsum(x, -1, "fp8"), TypeError, "FusedMac"),
        (lambda x: qsum(x.long(), -1, P8), TypeError, "float tensor"),
        (lambda x: qsum(x, 3, P8), IndexError, "out of range"),
        (lambda x: qsum(x, (1, -2), P8), RuntimeError, "more than once"),
        (lambda x: qsum(x, -1, P8, grad_quant=P8), TypeError, "Quant"),
        (
            lambda x: qsum(x, -1, P8, grad_quant=Quant(P8, carrier=torch.float64)),
            ValueError,
            "carrier",
        ),
        (lambda x: ReductionFormats(mac=MXFP8_E4M3), TypeError, "block format"),
    ],
)
def test_what_is_refused(call, error, match):
    with pytest.raises(error, match=match):
        call(_x("cpu"))


# --- compile, and MPS -------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", [R.RNE, R.SR], ids=lambda m: m.name)
def test_compiles_in_one_graph(device, mode):
    fmt = BinaryK(8, 4, prng_bits=8 if mode is R.SR else 0)
    f = ReductionFormats.uniform(FusedMac(fmt, rounding=mode), Quant(fmt, mode))

    def fn(x):
        return qsum(x, (0, 2), f.mac, grad_quant=f.grad) + qmean(x, -1, f.mac, quant=f.div).sum()

    x = _x(device).requires_grad_()
    torch.manual_seed(3)
    want = fn(x)
    gw = _grad(want, x)
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(fn, backend=counter, fullgraph=True)
    torch.manual_seed(3)
    got = compiled(x)
    assert counter.frame_count == 1
    assert _same(got, want) and _same(_grad(got, x), gw)


def test_inductor():
    torch._dynamo.reset()
    x = _x("cpu")
    fn = torch.compile(lambda t: qsum(t, -1, P8), backend="inductor", fullgraph=True)
    assert torch.equal(fn(x), qsum(x, -1, P8))


@requires_mps
@pytest.mark.parametrize("dims", [(-1,), (1,), (0, 2)], ids=str)
def test_mps_is_the_cpu(dims):
    for mac in (FusedMac(P8), SplitMac(P8, P16, accumulate_algorithm=A.KAHAN)):
        x = _x("cpu")
        assert torch.equal(qsum(x.to("mps"), dims, mac).cpu(), qsum(x, dims, mac))
