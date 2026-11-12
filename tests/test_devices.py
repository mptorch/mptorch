"""
Every op on its tensors' device.

Custom operators get no device guard from PyTorch's dispatcher, and mptorch's
CUDA entry points take the current device's stream and generator. Since
cb8cf52 every CUDA op is registered through ``OnOperandDevice``
(``mptorch/csrc/cuda/device_guard.h``), which makes its first CUDA tensor's
device current for the call, and the GEMM driver refuses operands on two
devices, as the conv and block drivers did already. Two kinds of test:

- **Operands on two devices** (one GPU, or the Mac): every GEMM-shaped op
  with one operand on the CPU raises a clean error, and leaves the device's
  context usable. Before cb8cf52 the GEMM ops dispatched to the device kernel
  with a host pointer, which on CUDA was "an illegal memory access", fatal to
  the process's CUDA context.
- **A second GPU** (skipped without two CUDA devices): every CUDA op on
  ``cuda:1`` with ``cuda:0`` current returns its result on ``cuda:1``, equals
  the same call on ``cuda:0`` bit for bit, stochastic rounding included,
  draws from ``cuda:1``'s generator exactly what the ``cuda:0`` call drew from
  ``cuda:0``'s, leaves ``cuda:0``'s alone, and leaves ``cuda:0`` current.
  Each of those cases also runs twice on ``cuda:0`` wherever there is one
  GPU, so a case's own mistakes show up without a second one; the first
  two-GPU run (a Kaggle 2x T4 notebook) found two, a palette conv without its
  maps and cases that reseeded every CUDA generator with ``torch.manual_seed``,
  which the harness now refuses.
"""

import pytest
import torch

from mptorch import MXFP8_E4M3, AccumulateAlgorithm, BinaryK, FixedPoint, RoundMode, SuperFP
from mptorch.quant import (
    BlockQuant,
    FusedMac,
    Palette,
    QAffineFormats,
    QConv2d,
    QLinear,
    Quant,
    SplitMac,
    binaryK_quantize,
    binaryK_quantize_,
    block_matmul,
    block_pack,
    block_quantize,
    block_quantize_,
    block_unpack,
    conv_formats,
    fixedpoint_quantize,
    fixedpoint_quantize_,
    qmatmul,
    superfp_quantize,
    superfp_quantize_,
)
from tests.markers import requires_cuda, requires_mps

SR = RoundMode.SR
BK = BinaryK(8, 4, prng_bits=8)
BK_ACC = BinaryK(16, 11, prng_bits=8)
SFP = SuperFP(3, 4, 2, 7, prng_bits=8)
PAL = Palette([BinaryK(8, 4), BinaryK(8, 3)])
PAL_SFP = Palette([SuperFP(3, 4, 2, 7), SuperFP(2, 4, 2, 7)])
KAHAN = AccumulateAlgorithm.KAHAN

# One mac per GEMM op: NAIVE split and fused in each family, their palette
# (mixed) twins, the accumulated (KAHAN) twins, and the cross-family pairs.
MACS = {
    "binaryK": SplitMac(BK, BK_ACC, rounding=SR),
    "superfp": SplitMac(SFP, SFP, rounding=SR),
    "binaryK_fma": FusedMac(BK_ACC, rounding=SR),
    "superfp_fma": FusedMac(SFP, rounding=SR),
    "binaryK_mixed": SplitMac(PAL, BK_ACC, rounding=SR),
    "superfp_mixed": SplitMac(PAL_SFP, SFP, rounding=SR),
    "binaryK_fma_mixed": FusedMac(PAL, rounding=SR),
    "superfp_fma_mixed": FusedMac(PAL_SFP, rounding=SR),
    "binaryK_accumulated": SplitMac(BK, BK_ACC, rounding=SR, accumulate_algorithm=KAHAN),
    "superfp_accumulated": SplitMac(SFP, SFP, rounding=SR, accumulate_algorithm=KAHAN),
    "binaryK_fma_accumulated": FusedMac(BK_ACC, rounding=SR, accumulate_algorithm=KAHAN),
    "superfp_fma_accumulated": FusedMac(SFP, rounding=SR, accumulate_algorithm=KAHAN),
    "binaryK_superfp": SplitMac(BK, SFP, rounding=SR),
    "superfp_binaryK": SplitMac(SFP, BK_ACC, rounding=SR),
    "binaryK_superfp_accumulated": SplitMac(BK, SFP, rounding=SR, accumulate_algorithm=KAHAN),
    "superfp_binaryK_accumulated": SplitMac(SFP, BK_ACC, rounding=SR, accumulate_algorithm=KAHAN),
}
CONV_MACS = {k: MACS[k] for k in ("binaryK", "binaryK_fma_mixed", "binaryK_superfp")}


def _cpu(*shape, seed=0):
    """Inputs from the CPU generator, so the CUDA generators move only for
    the op under test."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _prec_idx(name):
    """A palette op's precision map (both formats used), or None."""
    return (torch.arange(48) % 2).reshape(8, 6) if "mixed" in name else None


def _gemm(name, device):
    a, b = _cpu(8, 32, seed=1).to(device), _cpu(32, 6, seed=2).to(device)
    prec_idx = _prec_idx(name)
    return qmatmul(a, b, MACS[name], prec_idx=None if prec_idx is None else prec_idx.to(device))


# --- operands on two devices --------------------------------------------------

devices = [pytest.param("cuda", marks=requires_cuda), pytest.param("mps", marks=requires_mps)]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("name", list(MACS))
@pytest.mark.parametrize("host", ["a", "b"])
def test_a_gemm_refuses_operands_on_two_devices(device, name, host):
    """
    Either operand on the CPU and the other on the device raises "must be on
    one device", and the device still runs a GEMM afterwards.
    """
    a, b = _cpu(8, 32), _cpu(32, 6)
    a, b = (a, b.to(device)) if host == "a" else (a.to(device), b)
    with pytest.raises(RuntimeError, match="one device"):
        qmatmul(a, b, MACS[name], prec_idx=_prec_idx(name))
    out = _gemm(name, device)
    assert out.device.type == device and bool(torch.isfinite(out).any())


@pytest.mark.parametrize("device", devices)
def test_a_conv_and_a_block_gemm_refuse_operands_on_two_devices(device):
    """The conv and block drivers' own checks, which cb8cf52 left as they were."""
    conv = QConv2d(3, 4, 3, padding=1, formats=conv_formats(MACS["binaryK"])).to(device)
    with pytest.raises(RuntimeError, match="one device"):
        conv(_cpu(2, 3, 6, 6))
    pa = block_pack(_cpu(4, 64).to(device), MXFP8_E4M3, axis=-1)
    pb = block_pack(_cpu(64, 3), MXFP8_E4M3, axis=-2)
    with pytest.raises(RuntimeError, match="one device"):
        block_matmul(pa, pb)


# --- a second GPU -------------------------------------------------------------

TWO_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
requires_two_gpus = pytest.mark.skipif(not TWO_GPUS, reason="needs two CUDA devices")


def _flat(out):
    if isinstance(out, torch.Tensor):
        return [out]
    return [t for o in out for t in _flat(o)]


def _seed_cpu(seed):
    """Seed the CPU generator alone, for a case's own setup (a layer's
    initial weights): ``torch.manual_seed`` would reseed every CUDA
    generator too, which is what the harness watches."""
    torch.default_generator.manual_seed(seed)


def _states():
    return [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]


def _run(fn, index):
    """``fn`` on ``cuda:index`` with ``cuda:0`` current: its tensors on the
    CPU, and every CUDA generator's state (8 bytes of seed, 8 of offset)
    before and after."""
    torch.cuda.set_device(0)
    torch.manual_seed(1234)  # every generator, CPU and CUDA, the same
    before = _states()
    out = _flat(fn(torch.device("cuda", index)))
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
    assert torch.cuda.current_device() == 0, "the op left another device current"
    assert all(t.device == torch.device("cuda", index) for t in out), "a result on the wrong device"
    after = _states()
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b[:8], a[:8]), "the case reseeded a CUDA generator (use _seed_cpu)"
    return [t.cpu() for t in out], before, after


QUANTIZERS = {
    "binaryK": lambda x: binaryK_quantize(x, 8, 4, prng_bits=8, rounding_mode=SR),
    "superfp": lambda x: superfp_quantize(x, 3, 4, 2, 7, prng_bits=8, rounding_mode=SR),
    "fixedpoint": lambda x: fixedpoint_quantize(x, 8, 5, prng_bits=8, rounding_mode=SR),
    "binaryK_": lambda x: binaryK_quantize_(x, 8, 4, prng_bits=8, rounding_mode=SR),
    "superfp_": lambda x: superfp_quantize_(x, 3, 4, 2, 7, prng_bits=8, rounding_mode=SR),
    "fixedpoint_": lambda x: fixedpoint_quantize_(x, 8, 5, prng_bits=8, rounding_mode=SR),
    "binaryK, binary64 carrier": lambda x: binaryK_quantize(x, 8, 4, carrier=torch.float64),
    "block": lambda x: block_quantize(x, MXFP8_E4M3, rounding=SR),
    "block_": lambda x: block_quantize_(x, MXFP8_E4M3, rounding=SR),
    "block pack, unpack": lambda x: block_unpack(block_pack(x, MXFP8_E4M3, rounding=SR)),
}


def _quantize(name):
    return lambda d: QUANTIZERS[name](_cpu(64, 64).to(d))


def _conv(name):
    """The three passes of a conv layer: forward, input and weight gradient.
    A palette mac takes one map per pass: [Cout, 1], [Cin, 1], [Cout, 1]."""

    def step(d):
        mac = CONV_MACS[name]
        maps = {}
        if "mixed" in name:
            alternate = torch.tensor([[0], [1], [0], [1]], device=d)
            maps = dict(
                prec_idx=alternate,
                igrad_prec_idx=torch.zeros(3, 1, dtype=torch.int64, device=d),
                wgrad_prec_idx=1 - alternate,
            )
        _seed_cpu(7)
        layer = QConv2d(3, 4, 3, padding=1, formats=conv_formats(mac, **maps)).to(d)
        x = _cpu(2, 3, 6, 6).to(d).requires_grad_()
        out = layer(x)
        out.backward(_cpu(*out.shape, seed=3).to(d))
        return out.detach(), x.grad, layer.weight.grad

    return step


def _block_gemm(d):
    pa = block_pack(_cpu(4, 64, seed=1).to(d), MXFP8_E4M3, axis=-1, rounding=SR)
    pb = block_pack(_cpu(64, 3, seed=2).to(d), MXFP8_E4M3, axis=-2, rounding=SR)
    return block_matmul(pa, pb)


def _linear(input_quant, weight_quant, recompute=None):
    """A Linear layer's step: forward and both gradients."""

    def step(d):
        f = QAffineFormats(
            input_quant=input_quant,
            weight_quant=weight_quant,
            igrad_quant=Quant(BK, SR),
            wgrad_quant=Quant(BK, SR),
        )
        f.recompute_input = f.recompute_weight = recompute
        _seed_cpu(7)
        layer = QLinear(32, 16, formats=f).to(d)
        x = _cpu(8, 32).to(d).requires_grad_()
        out = layer(x)
        out.backward(_cpu(8, 16, seed=3).to(d))
        return out.detach(), x.grad, layer.weight.grad

    return step


# Every case: each quantizer, each of the 16 GEMM ops (all under SR), three
# conv macs, the block GEMM, and three layers: the recompute replay (the
# quantizer run again in backward under the generator states its forward call
# saw) with the weight and input recomputed, the same with the copies kept,
# and a fixed-point input with a block-format weight.
CASES = {
    **{f"quantize {n}": _quantize(n) for n in QUANTIZERS},
    **{f"gemm {n}": (lambda d, n=n: _gemm(n, d)) for n in MACS},
    **{f"conv {n}": _conv(n) for n in CONV_MACS},
    "block gemm": _block_gemm,
    "linear, recomputed": _linear(Quant(BK, SR), Quant(BK, SR), recompute=True),
    "linear, copies kept": _linear(Quant(BK, SR), Quant(BK, SR), recompute=False),
    "linear, fixed point and block": _linear(
        Quant(FixedPoint(8, 5, prng_bits=8), SR), BlockQuant(MXFP8_E4M3, SR)
    ),
}


@requires_cuda
@pytest.mark.parametrize("name", list(CASES))
def test_a_two_gpu_case_repeats_on_one(name):
    """
    Each two-GPU case, twice on ``cuda:0``: the same tensors and the same
    generator states both times, and no generator reseeded. The two-GPU tests
    skip on a machine with one GPU; this is what runs their cases there.
    """
    want, _, after = _run(CASES[name], 0)
    got, _, again = _run(CASES[name], 0)
    for w, g in zip(want, got, strict=True):
        assert torch.equal(w, g)
    for a, b in zip(after, again, strict=True):
        assert torch.equal(a, b)


@requires_two_gpus
@pytest.mark.parametrize("name", list(CASES))
def test_a_case_runs_on_its_tensors_gpu(name):
    """
    On ``cuda:1`` with ``cuda:0`` current, a case gives what it gives on
    ``cuda:0``, bit for bit, draws from ``cuda:1``'s generator what the
    ``cuda:0`` run drew from ``cuda:0``'s, and leaves ``cuda:0``'s alone.
    """
    want, _, after0 = _run(CASES[name], 0)
    got, before1, after1 = _run(CASES[name], 1)
    for w, g in zip(want, got, strict=True):
        assert torch.equal(w, g), "cuda:1 computed something else than cuda:0"
    assert torch.equal(after1[0], before1[0]), "an op on cuda:1 drew from cuda:0's generator"
    assert torch.equal(after1[1], after0[0]), "cuda:1's generator moved otherwise than cuda:0's"


@requires_two_gpus
def test_recomputing_agrees_with_the_copies_on_the_second_gpu():
    """The recompute replay rewinds the tensor's own generator on ``cuda:1`` too."""
    recomputed, _, _ = _run(CASES["linear, recomputed"], 1)
    kept, _, _ = _run(CASES["linear, copies kept"], 1)
    for r, k in zip(recomputed, kept, strict=True):
        assert torch.equal(r, k)
