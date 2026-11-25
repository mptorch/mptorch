"""
Fake (Meta) kernels for every op.

Every mptorch op has a Meta kernel (the files under ``mptorch/csrc/meta/``,
over ``common/meta_host.h``): the CPU op's checks, in its
order and with its messages, and a result of the shape, dtype and layout the
CPU op allocates, nothing computed and no RNG state drawn. It is what the
meta device, ``FakeTensorMode``, ``torch.export`` and ``torch.compile`` run in
the op's place. What is tested here:

- **Coverage.** Every op has a Meta and an Autograd kernel. A workload of
  public-API calls on the CPU, recorded by a dispatch mode, reaches every op
  whose kernels the build has: all 70 under ``MPTORCH_ODD_ACCUMULATED=1``.
- **opcheck** on every recorded call: the schema's mutation and aliasing
  claims, the fake result against the real one (sizes, strides, dtype,
  aliasing, storage offset), and AOTAutograd with symbolic shapes against
  eager. That last one seeds both runs, so it holds under SR too.
- **The meta device**: every recorded call again on meta tensors, which must
  give the CPU result's shape, dtype and strides.
- **Error parity**: each driver's refusals raise the CPU op's message on
  meta tensors too.
- **RNG**: a meta SR call leaves the generators as they were.
- **torch.export and torch.compile** of layers over the ops. Compiled results
  equal eager's, under SR as well once Inductor keeps the random ops in
  program order (``fallback_random``). The ``nondeterministic_seeded`` tag is
  what keeps a compiled graph from merging two identical SR calls; without
  it the CSE test below returned one draw twice under ``aot_eager`` and
  ``inductor`` (a build without the tags, 2026-11-11).
"""

import copy
import functools
import sys
import warnings

import pytest
import torch
from torch import nn
from torch._dynamo.decorators import assume_constant_result
from torch._dynamo.exc import Unsupported
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from mptorch import (
    MXFP8_E4M3,
    NVFP4,
    AccumulateAlgorithm,
    BinaryK,
    FixedPoint,
    FormatRangeWarning,
    RoundMode,
    SuperFP,
)
from mptorch.number.base import _warn_per_call
from mptorch.quant import (
    FusedMac,
    Palette,
    QConv2d,
    QLinear,
    QMatmul,
    Quant,
    Quantizer,
    ScaledQuant,
    SplitMac,
    binaryK_gemm_formats,
    binaryK_matmul,
    binaryK_quantize,
    binaryK_quantize_,
    block_gemm_formats,
    block_matmul,
    block_pack,
    block_project,
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
from tests.markers import HAS_ODD, HAS_ODD_ACCUMULATED, requires_cuda

# Dynamo says once per function that it traces through the format checks'
# lru_cache; the checks are pure, so the cache only saves time.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Dynamo detected a call to a `functools.lru_cache`:UserWarning"
)

R = RoundMode
SR = RoundMode.SR
A = AccumulateAlgorithm
BK = BinaryK(8, 4, prng_bits=8)
BK_ACC = BinaryK(16, 11, prng_bits=8)
SFP = SuperFP(3, 4, 2, 7, prng_bits=8)
PAL = Palette([BinaryK(8, 4), BinaryK(8, 3)])
PAL_SFP = Palette([SuperFP(3, 4, 2, 7), SuperFP(2, 4, 2, 7)])

# One mac per GEMM op, all under SR: NAIVE split and fused in each family,
# the palette twins, the accumulated twins, the cross-family pairs and the
# odd carrier steps.
MACS = {
    "binaryK": SplitMac(BK, BK_ACC, rounding=SR),
    "superfp": SplitMac(SFP, SFP, rounding=SR),
    "binaryK_fma": FusedMac(BK_ACC, rounding=SR),
    "superfp_fma": FusedMac(SFP, rounding=SR),
    "binaryK_mixed": SplitMac(PAL, BK_ACC, rounding=SR),
    "superfp_mixed": SplitMac(PAL_SFP, SFP, rounding=SR),
    "binaryK_fma_mixed": FusedMac(PAL, rounding=SR),
    "superfp_fma_mixed": FusedMac(PAL_SFP, rounding=SR),
    "binaryK_accumulated": SplitMac(BK, BK_ACC, rounding=SR, accumulate_algorithm=A.KAHAN),
    "superfp_accumulated": SplitMac(SFP, SFP, rounding=SR, accumulate_algorithm=A.KAHAN),
    "binaryK_fma_accumulated": FusedMac(BK_ACC, rounding=SR, accumulate_algorithm=A.KAHAN),
    "superfp_fma_accumulated": FusedMac(SFP, rounding=SR, accumulate_algorithm=A.KAHAN),
    "binaryK_superfp": SplitMac(BK, SFP, rounding=SR),
    "superfp_binaryK": SplitMac(SFP, BK_ACC, rounding=SR),
    "binaryK_superfp_accumulated": SplitMac(BK, SFP, rounding=SR, accumulate_algorithm=A.KAHAN),
    "superfp_binaryK_accumulated": SplitMac(SFP, BK_ACC, rounding=SR, accumulate_algorithm=A.KAHAN),
    "binaryK_odd": SplitMac(BK, BK_ACC, rounding=SR, carrier_rounding=R.RO),
    "superfp_odd": SplitMac(SFP, SFP, rounding=SR, carrier_rounding=R.RO),
    "binaryK_fma_odd": FusedMac(BK_ACC, rounding=SR, carrier_rounding=R.RO),
    "superfp_fma_odd": FusedMac(SFP, rounding=SR, carrier_rounding=R.RO),
    "binaryK_mixed_odd": SplitMac(PAL, BK_ACC, rounding=SR, carrier_rounding=R.RO),
    "superfp_mixed_odd": SplitMac(PAL_SFP, SFP, rounding=SR, carrier_rounding=R.RO),
    "binaryK_fma_mixed_odd": FusedMac(PAL, rounding=SR, carrier_rounding=R.RO),
    "superfp_fma_mixed_odd": FusedMac(PAL_SFP, rounding=SR, carrier_rounding=R.RO),
    "binaryK_superfp_odd": SplitMac(BK, SFP, rounding=SR, carrier_rounding=R.RO),
    "superfp_binaryK_odd": SplitMac(SFP, BK_ACC, rounding=SR, carrier_rounding=R.RO),
    "binaryK_accumulated_odd": SplitMac(
        BK, BK_ACC, rounding=SR, accumulate_algorithm=A.KAHAN, carrier_rounding=R.RO
    ),
    "superfp_accumulated_odd": SplitMac(
        SFP, SFP, rounding=SR, accumulate_algorithm=A.TREE, block_size=4, carrier_rounding=R.RO
    ),
    "binaryK_fma_accumulated_odd": FusedMac(
        BK_ACC, rounding=SR, accumulate_algorithm=A.BLOCK, block_size=4, carrier_rounding=R.RO
    ),
    "superfp_fma_accumulated_odd": FusedMac(
        SFP, rounding=SR, accumulate_algorithm=A.KAHAN, carrier_rounding=R.RO
    ),
    "binaryK_superfp_accumulated_odd": SplitMac(
        BK, SFP, rounding=SR, accumulate_algorithm=A.KAHAN, carrier_rounding=R.RO
    ),
    "superfp_binaryK_accumulated_odd": SplitMac(
        SFP, BK_ACC, rounding=SR, accumulate_algorithm=A.BLOCK, block_size=8, carrier_rounding=R.RO
    ),
}
CONV_MACS = {
    k: MACS[k]
    for k in (
        "binaryK",
        "superfp",
        "binaryK_fma",
        "superfp_fma",
        "binaryK_mixed",
        "superfp_mixed",
        "binaryK_fma_mixed",
        "superfp_fma_mixed",
        "binaryK_superfp",
        "superfp_binaryK",
        "binaryK_odd",
        "superfp_odd",
        "binaryK_fma_odd",
        "superfp_fma_odd",
        "binaryK_mixed_odd",
        "superfp_mixed_odd",
        "binaryK_fma_mixed_odd",
        "superfp_fma_mixed_odd",
        "binaryK_superfp_odd",
        "superfp_binaryK_odd",
        "binaryK_accumulated_odd",
        "superfp_fma_accumulated_odd",
        "superfp_binaryK_accumulated_odd",
    )
}


def _is_odd(name: str) -> bool:
    """An op of the carrier step rounded to odd, which a build with
    MPTORCH_NO_ODD=1 registers and makes raise."""
    return name.endswith("_odd") or name == "_carrier_odd"


def _is_odd_accumulated(name: str) -> bool:
    """An odd op under KAHAN, BLOCK or TREE, which a build without
    MPTORCH_ODD_ACCUMULATED=1 registers and makes raise."""
    return name.endswith("_accumulated_odd")


def _unbuilt(name: str) -> bool:
    """Whether this build left out the kernels of op (or mac) `name`."""
    return (not HAS_ODD and _is_odd(name)) or (
        not HAS_ODD_ACCUMULATED and _is_odd_accumulated(name)
    )


MACS = {k: v for k, v in MACS.items() if not _unbuilt(k)}
CONV_MACS = {k: v for k, v in CONV_MACS.items() if not _unbuilt(k)}
# A recorded call's tensors are what the workload passed; this many per op
# keep the opcheck cases to a few seconds.
CALLS_PER_OP = 3


def _op_names() -> list[str]:
    return sorted(
        n.split("::")[1] for n in torch._C._dispatch_get_all_op_names() if n.startswith("mptorch::")
    )


def _rnd(gen, *shape):
    return torch.randn(*shape, generator=gen)


def _conv_step(mac, name, gen):
    """A QConv2d's three passes: forward, input and weight gradient. A
    palette mac takes one map per pass: [Cout, 1], [Cin, 1], [Cout, 1]."""
    maps = {}
    if "mixed" in name:
        alternate = torch.tensor([[0], [1], [0], [1]])
        maps = dict(
            prec_idx=alternate,
            igrad_prec_idx=torch.zeros(3, 1, dtype=torch.int64),
            wgrad_prec_idx=1 - alternate,
        )
    layer = QConv2d(3, 4, 3, padding=1, formats=conv_formats(mac, **maps))
    x = _rnd(gen, 2, 3, 6, 6).requires_grad_()
    out = layer(x)
    out.backward(_rnd(gen, *out.shape))


def _workload():
    """Public-API calls that reach every op, on the CPU."""
    gen = torch.Generator().manual_seed(0)
    a, b = _rnd(gen, 8, 32), _rnd(gen, 32, 6)
    for name, mac in MACS.items():
        pidx = (torch.arange(48) % 2).reshape(8, 6) if "mixed" in name else None
        qmatmul(a, b, mac, prec_idx=pidx)
    # a batch, a transposed operand, and a per-row palette map
    qmatmul(_rnd(gen, 2, 8, 32), b, MACS["binaryK"])
    qmatmul(a, _rnd(gen, 6, 32).t(), MACS["binaryK_fma"])
    qmatmul(a, b, MACS["binaryK_mixed"], prec_idx=(torch.arange(8) % 2).reshape(8, 1))
    for name, mac in CONV_MACS.items():
        _conv_step(mac, name, gen)
    x = _rnd(gen, 16, 16)
    binaryK_quantize(x, 8, 4, prng_bits=8, rounding_mode=SR)
    superfp_quantize(x, 3, 4, 2, 7, prng_bits=8, rounding_mode=SR)
    fixedpoint_quantize(x, 8, 5, prng_bits=8, rounding_mode=SR)
    binaryK_quantize_(x.clone(), 8, 4, prng_bits=8, rounding_mode=SR)
    superfp_quantize_(x.clone(), 3, 4, 2, 7, prng_bits=8, rounding_mode=SR)
    fixedpoint_quantize_(x.clone(), 8, 5, prng_bits=8, rounding_mode=SR)
    binaryK_quantize(x.bfloat16(), 8, 4, carrier=torch.float64)  # reaches narrow_float64
    # the scaled quantizers' fused ops: one scale, per row, per block
    ScaledQuant(BinaryK(8, 4, prng_bits=8), SR)(x)
    ScaledQuant(SuperFP(3, 4, 2, 7), granularity="row")(x)
    ScaledQuant(FixedPoint(8, 5), granularity="block", block_size=5)(x)
    xb = _rnd(gen, 4, 64)
    block_quantize(xb, MXFP8_E4M3, rounding=SR)
    block_quantize(xb.t(), MXFP8_E4M3, rounding=SR)  # a strided input keeps its layout
    block_quantize_(xb.clone(), MXFP8_E4M3, rounding=SR)
    block_unpack(block_pack(xb, MXFP8_E4M3, rounding=SR))
    block_unpack(block_pack(_rnd(gen, 2, 4, 64), MXFP8_E4M3))
    pa = block_pack(xb, MXFP8_E4M3, axis=-1, rounding=SR)
    pb = block_pack(_rnd(gen, 64, 3), MXFP8_E4M3, axis=-2, rounding=SR)
    block_matmul(pa, pb, rounding=SR)
    block_matmul(pa, pb, accumulate_algorithm=A.KAHAN)
    # BlockProject: onto the rule's own scales, a batch, and a strided view
    block_project(xb, MXFP8_E4M3, pa.scales, rounding=SR)
    x3 = _rnd(gen, 2, 4, 64)
    block_project(x3, MXFP8_E4M3, block_pack(x3, MXFP8_E4M3).scales)
    xt = _rnd(gen, 32, 4)
    block_project(xt, NVFP4, block_pack(xt, NVFP4, axis=0).scales, axis=0, rounding=SR)
    if HAS_ODD:
        torch.ops.mptorch._carrier_odd(x, x, x, 0)
    # the test-only stochastic cast: binary8p4, StochasticB, N = 3
    torch.ops.mptorch._stochastic_cast(
        x, torch.zeros_like(x, dtype=torch.int64), 0, [8, 4, 7, 3, 1, 2, 0], 7
    )


class _Recorder(TorchDispatchMode):
    """Records every mptorch op the code under it calls, with copies of its
    tensors taken before the call (an in-place op writes its own), and every
    argument positional."""

    def __init__(self):
        super().__init__()
        self.calls: dict[str, list[tuple]] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.namespace == "mptorch":
            schema = func._schema.arguments
            full = list(args) + [
                kwargs[a.name] if a.name in kwargs else a.default_value for a in schema[len(args) :]
            ]
            snap = pytree.tree_map_only(torch.Tensor, lambda t: t.detach().clone(), tuple(full))
            name = func._schema.name.split("::")[1]
            self.calls.setdefault(name, []).append((func, snap))
        return func(*args, **kwargs)


@functools.cache
def _calls() -> dict[str, list[tuple]]:
    with _Recorder() as rec:
        _workload()
    return {k: v[:CALLS_PER_OP] for k, v in rec.calls.items()}


def _fresh(args):
    """A call's arguments with fresh copies of its tensors."""
    return pytree.tree_map_only(torch.Tensor, lambda t: t.clone(), args)


def _on_meta(args):
    """A call's arguments on the meta device, strides kept (``.to()`` would
    make an expanded view dense)."""
    return pytree.tree_map_only(
        torch.Tensor,
        lambda t: torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="meta"),
        args,
    )


# --- coverage -----------------------------------------------------------------


def test_every_op_has_a_meta_and_an_autograd_kernel():
    names = _op_names()
    assert len(names) == 70
    for key in ("Meta", "Autograd"):
        missing = [
            n
            for n in names
            if not torch._C._dispatch_has_kernel_for_dispatch_key(f"mptorch::{n}", key)
        ]
        assert not missing, (key, missing)


def test_the_workload_reaches_every_op():
    assert sorted(_calls()) == [n for n in _op_names() if not _unbuilt(n)]


@pytest.mark.skipif(HAS_ODD, reason="the odd kernels are built")
def test_without_the_odd_kernels_every_odd_op_raises_naming_the_flag():
    """A build with MPTORCH_NO_ODD=1 keeps the odd ops registered, and each
    raises the same message from its CPU and its Meta kernel. A GEMM or conv
    odd op takes its twin's schema, so the twin's recorded call is a valid
    call of it."""
    odd = [n for n in _op_names() if _is_odd(n)]
    assert len(odd) == 27
    x = torch.ones(1)
    for name in odd:
        func = getattr(torch.ops.mptorch, name).default
        if name == "_carrier_odd":
            args: tuple = (x, x, x, 0)
        else:
            args = _calls()[name[: -len("_odd")]][0][1]
        with pytest.raises(RuntimeError, match="MPTORCH_NO_ODD=1") as cpu:
            func(*_fresh(args))
        with pytest.raises(RuntimeError) as meta:
            func(*_on_meta(args))
        assert str(meta.value) == str(cpu.value), name


@pytest.mark.skipif(
    not HAS_ODD or HAS_ODD_ACCUMULATED,
    reason="the odd accumulated kernels are built (or no odd one is)",
)
def test_without_the_flag_every_odd_accumulated_op_raises_naming_it():
    """A build without MPTORCH_ODD_ACCUMULATED=1 keeps the six
    *_accumulated_odd ops registered, and each raises the same message from
    its CPU and its Meta kernel, called on its twin's recorded call (the
    twin's schema)."""
    names = [n for n in _op_names() if _is_odd_accumulated(n)]
    assert len(names) == 6
    for name in names:
        func = getattr(torch.ops.mptorch, name).default
        args = _calls()[name[: -len("_odd")]][0][1]
        with pytest.raises(RuntimeError, match="MPTORCH_ODD_ACCUMULATED=1") as cpu:
            func(*_fresh(args))
        with pytest.raises(RuntimeError) as meta:
            func(*_on_meta(args))
        assert str(meta.value) == str(cpu.value), name


def test_the_tags():
    """Every op passes opcheck (pt2_compliant_tag), and an op that takes a
    round mode can draw, so compile must not merge or fold it
    (nondeterministic_seeded). An op that writes its argument returns nothing,
    so it is not ``inplace``: Inductor reused the memory of a custom op's
    argument while the op's returned alias of it was still live."""
    for name in _op_names():
        op = getattr(torch.ops.mptorch, name).default
        args = [a.name for a in op._schema.arguments]
        want = {torch.Tag.pt2_compliant_tag}
        if "round_mode" in args:
            want.add(torch.Tag.nondeterministic_seeded)
        assert set(op.tags) == want, name
        if op._schema.is_mutable:
            assert not op._schema.returns, name


# --- opcheck and the meta device ------------------------------------------------


@pytest.mark.parametrize("name", _op_names())
def test_opcheck(name):
    """The schema, the fake result against the real one, and AOTAutograd with
    symbolic shapes against eager."""
    if _unbuilt(name):
        pytest.skip("this build has no kernels for it")
    for func, args in _calls()[name]:
        torch.library.opcheck(
            func,
            _fresh(args),
            test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        )


@pytest.mark.parametrize("name", _op_names())
def test_the_meta_device_gives_the_cpu_layout(name):
    if _unbuilt(name):
        pytest.skip("this build has no kernels for it")
    for func, args in _calls()[name]:
        cpu_args = _fresh(args)
        want_out = func(*cpu_args)
        meta_args = _on_meta(args)
        got_out = func(*meta_args)
        if func._schema.is_mutable:
            # An in-place op returns nothing; its result is the argument it wrote.
            assert want_out is None and got_out is None, "an in-place op returns nothing"
            (i,) = [k for k, a in enumerate(func._schema.arguments) if a.is_write]
            want_out, got_out = cpu_args[i], meta_args[i]
        want = pytree.tree_leaves(want_out)
        got = pytree.tree_leaves(got_out)
        assert len(got) == len(want)
        for w, g in zip(want, got, strict=True):
            assert g.device.type == "meta"
            assert (g.shape, g.dtype, g.stride()) == (w.shape, w.dtype, w.stride()), name


def test_a_meta_call_draws_nothing():
    """A meta SR call leaves the CPU generator, and CUDA's, as they were; the
    same call on the CPU moves it."""
    func, args = _calls()["custom_matmul_binaryK"][0]
    names = [a.name for a in func._schema.arguments]
    assert args[names.index("round_mode")] == SR.value
    before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    func(*_on_meta(args))
    assert torch.equal(torch.get_rng_state(), before)
    if cuda_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_before)
    func(*_fresh(args))
    assert not torch.equal(torch.get_rng_state(), before)


# --- error parity -----------------------------------------------------------------


def _with(name, call=0, **changes):
    """Recorded call `call` of op `name`, with arguments replaced by name, or
    None for an op in a build without its kernels."""
    if _unbuilt(name):
        return None
    func, args = _calls()[name][call]
    names = [a.name for a in func._schema.arguments]
    args = list(_fresh(args))
    for k, v in changes.items():
        args[names.index(k)] = v
    return func, tuple(args)


def _in_float64(name, call=0):
    """The operands of recorded call `call` of op `name`, as float64 tensors."""
    if _unbuilt(name):
        return {}
    func, args = _calls()[name][call]
    return {"a": args[0].double(), "b": args[1].double()}


def _gemm_cases():
    x = torch.randn
    i32 = torch.int32
    return {
        "round mode": _with("custom_matmul_binaryK", round_mode=99),
        "algorithm": _with("custom_matmul_binaryK", accumulate_algorithm=99),
        "naive only": _with("custom_matmul_binaryK", accumulate_algorithm=A.KAHAN.value),
        "rank": _with("custom_matmul_binaryK", a=x(32)),
        "inner": _with("custom_matmul_binaryK", b=x(31, 6)),
        "batch": _with("custom_matmul_binaryK", a=x(2, 8, 32), b=x(3, 32, 6)),
        "dtypes": _with("custom_matmul_superfp_fma", b=x(32, 6, dtype=torch.float64)),
        "dtype": _with("custom_matmul_binaryK", a=x(8, 32).int(), b=x(32, 6).int()),
        "accumulated naive": _with("custom_matmul_binaryK_accumulated", accumulate_algorithm=0),
        "kahan block size": _with("custom_matmul_superfp_accumulated", block_size=4),
        "block block size": _with(
            "custom_matmul_binaryK_accumulated", accumulate_algorithm=A.BLOCK.value, block_size=3
        ),
        "block size range": _with("custom_matmul_binaryK_accumulated", block_size=-1),
        "fused tree": _with(
            "custom_matmul_binaryK_fma_accumulated", accumulate_algorithm=A.TREE.value, block_size=4
        ),
        "palette lengths": _with("custom_matmul_binaryK_mixed", mul_P=[4]),
        "prec_idx rank": _with("custom_matmul_binaryK_mixed", prec_idx=torch.zeros(48, dtype=i32)),
        "prec_idx form": _with(
            "custom_matmul_superfp_mixed", prec_idx=torch.zeros(8, 2, dtype=i32)
        ),
        "prec_idx batch": _with(
            "custom_matmul_binaryK_mixed", prec_idx=torch.zeros(3, 8, 6, dtype=i32)
        ),
        "palette naive only": _with("custom_matmul_binaryK_fma_mixed", accumulate_algorithm=1),
        "fma_quant palette": _with("custom_matmul_superfp_fma_mixed", fma_quant=False),
        "fma_quant odd": _with("custom_matmul_binaryK_fma_odd", fma_quant=False),
        "cross inner": _with("custom_matmul_binaryK_superfp", b=x(31, 6)),
        "cross accumulated naive": _with(
            "custom_matmul_superfp_binaryK_accumulated", accumulate_algorithm=0
        ),
        "odd naive only": _with("custom_matmul_superfp_odd", accumulate_algorithm=A.KAHAN.value),
        "odd palette naive only": _with(
            "custom_matmul_binaryK_mixed_odd", accumulate_algorithm=A.KAHAN.value
        ),
        "fma_quant odd palette": _with("custom_matmul_superfp_fma_mixed_odd", fma_quant=False),
        "odd palette lengths": _with("custom_matmul_superfp_mixed_odd", mul_exp_bits=[4]),
        "odd cross naive only": _with(
            "custom_matmul_binaryK_superfp_odd", accumulate_algorithm=A.TREE.value
        ),
        "odd accumulated naive": _with(
            "custom_matmul_binaryK_accumulated_odd", accumulate_algorithm=0
        ),
        "fma_quant odd accumulated": _with(
            "custom_matmul_superfp_fma_accumulated_odd", fma_quant=False
        ),
        "odd accumulated tree size": _with("custom_matmul_superfp_accumulated_odd", block_size=3),
        # The recorded formats have 3-bit mantissas (BK, SFP, the palettes) or
        # 10-bit ones (BK_ACC), so binary32 has room for 20 or 13 random bits.
        "mul prng": _with("custom_matmul_binaryK", mul_prng_bits=21),
        "acc prng": _with("custom_matmul_superfp", acc_prng_bits=21),
        "palette prng": _with("custom_matmul_binaryK_mixed", mul_prng_bits=21),
        "fused palette prng": _with("custom_matmul_superfp_fma_mixed", fma_prng_bits=21),
        "outer prng": _with(
            "custom_matmul_binaryK_accumulated",
            accumulate_algorithm=A.BLOCK.value,
            block_size=16,
            outer_quant=True,
            outer_K=16,
            outer_P=11,
            outer_bias=15,
            outer_prng_bits=14,
        ),
        "cross prng": _with("custom_matmul_superfp_binaryK", acc_prng_bits=14),
        "binary64 prng": _with(
            "custom_matmul_binaryK_fma",
            **_in_float64("custom_matmul_binaryK_fma"),
            fma_prng_bits=43,
        ),
        "odd prng": _with("custom_matmul_superfp_odd", mul_prng_bits=21),
        "mul mantissa width": _with("custom_matmul_binaryK", mul_K=30, mul_P=25),
        "mul exp width": _with("custom_matmul_binaryK", mul_K=35),
        "acc exp width": _with("custom_matmul_superfp", acc_exp_bits=31),
        "palette mantissa width": _with("custom_matmul_binaryK_mixed", mul_P=[4, 0]),
        "fused palette exp width": _with("custom_matmul_superfp_fma_mixed", fma_exp_bits=[4, 31]),
        "outer exp width": _with(
            "custom_matmul_binaryK_accumulated",
            accumulate_algorithm=A.BLOCK.value,
            block_size=16,
            outer_quant=True,
            outer_K=42,
            outer_P=11,
            outer_bias=15,
        ),
        "cross exp width": _with("custom_matmul_superfp_binaryK", acc_K=42),
        "binary64 mantissa width": _with(
            "custom_matmul_binaryK_fma",
            **_in_float64("custom_matmul_binaryK_fma"),
            fma_K=58,
            fma_P=54,
        ),
        "odd exp width": _with("custom_matmul_binaryK_odd", mul_K=35),
        # -(2**31 - 1) is the bias whose `1 - bias` overflowed an int before
        # the op refused it.
        "mul bias": _with("custom_matmul_binaryK", mul_bias=-(2**31 - 1)),
        "acc bias": _with("custom_matmul_superfp", acc_bias=2**31 - 1),
        "acc normal binades": _with("custom_matmul_superfp", acc_normal_binades=0),
        "palette bias": _with("custom_matmul_binaryK_mixed", mul_bias=[8, -(2**31 - 1)]),
        "fused palette normal binades": _with(
            "custom_matmul_superfp_fma_mixed", fma_normal_binades=[2, 16]
        ),
        "outer bias": _with(
            "custom_matmul_binaryK_accumulated",
            accumulate_algorithm=A.BLOCK.value,
            block_size=16,
            outer_quant=True,
            outer_K=16,
            outer_P=11,
            outer_bias=-(2**31 - 1),
        ),
        "cross normal binades": _with("custom_matmul_binaryK_superfp", acc_normal_binades=0),
        "binary64 bias": _with(
            "custom_matmul_binaryK_fma",
            **_in_float64("custom_matmul_binaryK_fma"),
            fma_bias=2**31 - 1,
        ),
        "odd bias": _with("custom_matmul_superfp_odd", mul_bias=-(2**31 - 1)),
        # Values an int cannot hold, which the packers used to wrap (a bias
        # of 2**40 became 0) and now saturate to int's limit.
        "mul bias past int": _with("custom_matmul_binaryK", mul_bias=2**40),
        "palette bias past int": _with("custom_matmul_binaryK_mixed", mul_bias=[8, -(2**40)]),
        "mul mantissa past int": _with("custom_matmul_binaryK", mul_K=2**32 + 8, mul_P=2**32 + 4),
        # K - P overflows an int64 too
        "mul exp past int": _with("custom_matmul_binaryK", mul_K=2**63 - 1, mul_P=-(2**63)),
        "acc normal binades past int": _with("custom_matmul_superfp", acc_normal_binades=2**32 + 2),
        "outer bias past int": _with(
            "custom_matmul_binaryK_accumulated",
            accumulate_algorithm=A.BLOCK.value,
            block_size=16,
            outer_quant=True,
            outer_K=16,
            outer_P=11,
            outer_bias=2**40,
        ),
        "cross bias past int": _with("custom_matmul_superfp_binaryK", acc_bias=2**40),
        "mul prng past int": _with("custom_matmul_binaryK", mul_prng_bits=2**32 + 3),
        # a format the call does not round with: only the byte is checked
        "unused acc prng past int": _with(
            "custom_matmul_binaryK", accumulate_quant=False, acc_prng_bits=2**32 + 3
        ),
        # Integers that name no SaturationMode (0 to 3) or SubnormalsMode (0 to
        # 2), which the casts would run as one of them or as none; the one past
        # int used to wrap to 1, SAT_PROPAGATE.
        "mul saturation mode": _with("custom_matmul_binaryK", mul_saturation_mode=4),
        "acc subnormals mode": _with("custom_matmul_binaryK", acc_subnormals_mode=-1),
        "acc saturation mode": _with("custom_matmul_superfp", acc_saturation_mode=7),
        "mul saturation mode past int": _with(
            "custom_matmul_binaryK", mul_saturation_mode=2**32 + 1
        ),
        "fused palette saturation mode": _with(
            "custom_matmul_superfp_fma_mixed", fma_saturation_mode=9
        ),
        "outer subnormals mode": _with(
            "custom_matmul_binaryK_accumulated",
            accumulate_algorithm=A.BLOCK.value,
            block_size=16,
            outer_quant=True,
            outer_K=16,
            outer_P=11,
            outer_bias=15,
            outer_subnormals_mode=5,
        ),
        "cross subnormals mode": _with("custom_matmul_superfp_binaryK", acc_subnormals_mode=3),
    }


def _conv_cases():
    x = torch.randn
    return {
        "conv pass": _with("custom_conv_binaryK", conv_pass=3),
        "out_size entries": _with("custom_conv_binaryK", out_size=[]),
        "stride entries": _with("custom_conv_binaryK", stride=[1]),
        "rank": _with("custom_conv_binaryK", b=x(2, 3, 6)),
        "groups": _with("custom_conv_superfp", groups=0),
        "channels": _with("custom_conv_binaryK", groups=2),
        "stride": _with("custom_conv_binaryK_fma", stride=[0, 1]),
        "out_size": _with("custom_conv_binaryK", out_size=[-1, 6]),
        "round mode": _with("custom_conv_superfp_fma", round_mode=99),
        "algorithm": _with("custom_conv_binaryK", accumulate_algorithm=99),
        "naive block size": _with("custom_conv_binaryK", block_size=4),
        "kahan block size": _with(
            "custom_conv_binaryK", accumulate_algorithm=A.KAHAN.value, block_size=4
        ),
        "dtypes": _with("custom_conv_binaryK", b=x(2, 3, 6, 6, dtype=torch.float64)),
        "igrad channels": _with("custom_conv_binaryK", call=1, b=x(2, 5, 6, 6)),
        "wgrad samples": _with("custom_conv_binaryK", call=2, b=x(3, 3, 6, 6)),
        "palette naive only": _with("custom_conv_binaryK_mixed", accumulate_algorithm=1),
        "palette lengths": _with("custom_conv_superfp_mixed", mul_exp_bits=[4]),
        "palette map": _with("custom_conv_binaryK_mixed", prec_idx=torch.zeros(5, 1)),
        "fma_quant palette": _with("custom_conv_binaryK_fma_mixed", fma_quant=False),
        "cross groups": _with("custom_conv_superfp_binaryK", groups=0),
        "odd kahan block size": _with(
            "custom_conv_binaryK_odd", accumulate_algorithm=A.KAHAN.value, block_size=4
        ),
        "odd naive block size": _with("custom_conv_superfp_odd", block_size=4),
        "odd algorithm": _with("custom_conv_superfp_binaryK_odd", accumulate_algorithm=99),
        "fma_quant odd": _with("custom_conv_binaryK_fma_odd", fma_quant=False),
        "fma_quant odd palette": _with("custom_conv_superfp_fma_mixed_odd", fma_quant=False),
        "odd palette map": _with("custom_conv_binaryK_mixed_odd", prec_idx=torch.zeros(5, 1)),
        "odd cross groups": _with("custom_conv_binaryK_superfp_odd", groups=0),
        "mul prng": _with("custom_conv_binaryK", mul_prng_bits=21),
        "mul exp width": _with("custom_conv_binaryK", mul_K=35),
        "mul bias": _with("custom_conv_binaryK", mul_bias=-(2**31 - 1)),
        "mul bias past int": _with("custom_conv_binaryK", mul_bias=2**40),
        "mul saturation mode": _with("custom_conv_binaryK", mul_saturation_mode=4),
    }


def _block_cases():
    x = torch.randn
    pack = _calls()["block_pack"][0]
    fmt = pack[1][1]
    unpack_func, unpack_args = _calls()["block_unpack"][0]
    data, scales = unpack_args[0], unpack_args[1]
    return {
        "pack round mode": _with("block_pack", round_mode=99),
        "pack rank": _with("block_pack", x=x(64)),
        "pack dtype": _with("block_pack", x=x(4, 64).int()),
        "pack fmt": _with("block_pack", fmt=list(fmt)[:-1]),
        "unpack scales": _with("block_unpack", scales=scales[:, :1].clone()),
        "unpack data dtype": _with("block_unpack", data=data.to(torch.int8)),
        "unpack cols": _with("block_unpack", cols=-1),
        "unpack dtype": _with("block_unpack", dtype=torch.int32),
        "unpack batch": _with(
            "block_unpack",
            call=1,
            scales=torch.zeros(3, *_calls()["block_unpack"][1][1][1].shape[1:], dtype=torch.uint8),
        ),
        "quant round mode": _with("block_quant", round_mode=99),
        "quant_ overlap": _with("block_quant_", x=x(1, 64).expand(4, 64)),
        "project round mode": _with("block_project", round_mode=99),
        "project rank": _with("block_project", x=x(64)),
        "project dtype": _with("block_project", x=x(4, 64).int()),
        "project fmt": _with("block_project", fmt=list(fmt)[:-1]),
        "project scales dtype": _with("block_project", scales=torch.zeros(4, 2, dtype=torch.int16)),
        "project scales rank": _with(
            "block_project", scales=torch.zeros(1, 4, 2, dtype=torch.uint8)
        ),
        "project scales shape": _with("block_project", scales=torch.zeros(4, 1, dtype=torch.uint8)),
        "project batch": _with(
            "block_project", call=1, scales=torch.zeros(3, 4, 2, dtype=torch.uint8)
        ),
        "gemm a cols": _with("custom_matmul_block", a_cols=32),
        "gemm round mode": _with("custom_matmul_block", round_mode=99),
        "gemm algorithm": _with("custom_matmul_block", accumulate_algorithm=99),
        "gemm data dtype": _with("custom_matmul_block", b_data=x(3, 64).to(torch.int8)),
        "gemm naive block size": _with("custom_matmul_block", block_size=4),
        "gemm kahan block size": _with("custom_matmul_block", call=1, block_size=4),
        "gemm acc prng": _with(
            "custom_matmul_block",
            accumulate_quant=True,
            acc_K=16,
            acc_P=11,
            acc_bias=15,
            acc_prng_bits=14,
        ),
        "gemm acc exp width": _with(
            "custom_matmul_block", accumulate_quant=True, acc_K=42, acc_P=11, acc_bias=15
        ),
        "gemm acc bias": _with(
            "custom_matmul_block", accumulate_quant=True, acc_K=16, acc_P=11, acc_bias=-(2**31 - 1)
        ),
        "gemm acc bias past int": _with(
            "custom_matmul_block", accumulate_quant=True, acc_K=16, acc_P=11, acc_bias=2**40
        ),
        "gemm acc prng past int": _with(
            "custom_matmul_block",
            accumulate_quant=True,
            acc_K=16,
            acc_P=11,
            acc_bias=15,
            acc_prng_bits=2**32 + 3,
        ),
        "gemm acc saturation mode": _with(
            "custom_matmul_block",
            accumulate_quant=True,
            acc_K=16,
            acc_P=11,
            acc_bias=15,
            acc_saturation_mode=4,
        ),
    }


def _other_cases():
    x = torch.randn
    return {
        "narrow input": _with("narrow_float64", a=x(4)),
        "narrow dtype": _with("narrow_float64", dtype=torch.float64),
        "binaryK in place": _with("binaryK_quant_", a=x(4, 6).t()),
        "superfp in place": _with("superfp_quant_", a=x(4, 6).t()),
        "fixedpoint in place": _with("fixedpoint_quant_", a=x(4, 6).t()),
        "carrier_odd op": _with("_carrier_odd", op=3),
        "carrier_odd shape": _with("_carrier_odd", b=x(3)),
        "carrier_odd dtype": _with("_carrier_odd", a=x(16, 16).int()),
        "binaryK prng": _with("binaryK_quant", prng_bits=21),
        "superfp prng": _with("superfp_quant", prng_bits=21),
        "fixedpoint prng": _with("fixedpoint_quant", prng_bits=18),
        "stochastic_cast prng": _with("_stochastic_cast", format=[8, 4, 7, 21, 1, 2, 0]),
        "binaryK exp width": _with("binaryK_quant", K=35),
        "superfp exp width": _with("superfp_quant", exp_bits=31),
        "fixedpoint magnitude width": _with("fixedpoint_quant", wl=26),
        "stochastic_cast exp width": _with("_stochastic_cast", format=[35, 4, 7, 3, 1, 2, 0]),
        "binaryK bias": _with("binaryK_quant", bias=-(2**31 - 1)),
        # an int64 bias reaches the check whole, before any narrowing to int
        "superfp bias": _with("superfp_quant", bias=-(2**40)),
        "superfp normal binades": _with("superfp_quant", normal_binades=0),
        "fixedpoint fl": _with("fixedpoint_quant", fl=2**31 - 1),
        "stochastic_cast bias": _with("_stochastic_cast", format=[8, 4, -(2**31 - 1), 3, 1, 2, 0]),
        "binaryK saturation mode": _with("binaryK_quant", saturation_mode=4),
        "binaryK subnormals mode": _with("binaryK_quant", subnormals_mode=3),
        "superfp saturation mode": _with("superfp_quant", saturation_mode=-1),
        "stochastic_cast saturation mode": _with("_stochastic_cast", format=[8, 4, 7, 3, 1, 7, 0]),
        "stochastic_cast subnormals mode": _with("_stochastic_cast", format=[8, 4, 7, 3, 1, 2, 3]),
        # P3109's (K, P) at int64's limits: K - P overflows an int64
        "binaryK K past int64": _with("binaryK_quant", K=2**63 - 1, P=-(2**63)),
        # The quantizers ran a round mode they do not know as round to odd.
        "binaryK round mode int": _with("binaryK_quant", round_mode=9),
        "superfp round mode int": _with("superfp_quant", round_mode=-1),
        "fixedpoint round mode int": _with("fixedpoint_quant", round_mode=2**40),
    }


def _case_ids():
    # The cases are built from the recorded calls, at run time; their names
    # are listed here so pytest can collect them (a test below holds the two
    # together).
    return (
        [f"gemm: {k}" for k in _GEMM_IDS]
        + [f"conv: {k}" for k in _CONV_IDS]
        + [f"block: {k}" for k in _BLOCK_IDS]
        + [f"other: {k}" for k in _OTHER_IDS]
    )


_GEMM_IDS = [
    "round mode", "algorithm", "naive only", "rank", "inner", "batch", "dtypes", "dtype",
    "accumulated naive", "kahan block size", "block block size", "block size range", "fused tree",
    "palette lengths", "prec_idx rank", "prec_idx form", "prec_idx batch", "palette naive only",
    "fma_quant palette", "fma_quant odd", "cross inner", "cross accumulated naive",
    "odd naive only", "odd palette naive only", "fma_quant odd palette", "odd palette lengths",
    "odd cross naive only", "odd accumulated naive", "fma_quant odd accumulated",
    "odd accumulated tree size", "mul prng", "acc prng", "palette prng", "fused palette prng",
    "outer prng", "cross prng", "binary64 prng", "odd prng", "mul mantissa width",
    "mul exp width", "acc exp width", "palette mantissa width", "fused palette exp width",
    "outer exp width", "cross exp width", "binary64 mantissa width", "odd exp width", "mul bias",
    "acc bias", "acc normal binades", "palette bias", "fused palette normal binades", "outer bias",
    "cross normal binades", "binary64 bias", "odd bias", "mul bias past int",
    "palette bias past int", "mul mantissa past int", "mul exp past int",
    "acc normal binades past int", "outer bias past int", "cross bias past int",
    "mul prng past int", "unused acc prng past int", "mul saturation mode",
    "acc subnormals mode", "acc saturation mode", "mul saturation mode past int",
    "fused palette saturation mode", "outer subnormals mode", "cross subnormals mode",
]  # fmt: skip
_CONV_IDS = [
    "conv pass", "out_size entries", "stride entries", "rank", "groups", "channels", "stride",
    "out_size", "round mode", "algorithm", "naive block size", "kahan block size", "dtypes",
    "igrad channels", "wgrad samples", "palette naive only", "palette lengths", "palette map",
    "fma_quant palette", "cross groups", "odd kahan block size", "odd naive block size",
    "odd algorithm", "fma_quant odd", "fma_quant odd palette", "odd palette map",
    "odd cross groups", "mul prng", "mul exp width", "mul bias", "mul bias past int",
    "mul saturation mode",
]  # fmt: skip
_BLOCK_IDS = [
    "pack round mode", "pack rank", "pack dtype", "pack fmt", "unpack scales",
    "unpack data dtype", "unpack cols", "unpack dtype", "unpack batch", "quant round mode",
    "quant_ overlap", "project round mode", "project rank", "project dtype", "project fmt",
    "project scales dtype", "project scales rank", "project scales shape", "project batch",
    "gemm a cols", "gemm round mode", "gemm algorithm", "gemm data dtype",
    "gemm naive block size", "gemm kahan block size", "gemm acc prng", "gemm acc exp width",
    "gemm acc bias", "gemm acc bias past int", "gemm acc prng past int", "gemm acc saturation mode",
]  # fmt: skip
_OTHER_IDS = [
    "narrow input", "narrow dtype", "binaryK in place", "superfp in place", "fixedpoint in place",
    "carrier_odd op", "carrier_odd shape", "carrier_odd dtype", "binaryK prng", "superfp prng",
    "fixedpoint prng", "stochastic_cast prng", "binaryK exp width", "superfp exp width",
    "fixedpoint magnitude width", "stochastic_cast exp width", "binaryK bias", "superfp bias",
    "superfp normal binades", "fixedpoint fl", "stochastic_cast bias", "binaryK saturation mode",
    "binaryK subnormals mode", "superfp saturation mode", "stochastic_cast saturation mode",
    "stochastic_cast subnormals mode", "binaryK K past int64", "binaryK round mode int",
    "superfp round mode int", "fixedpoint round mode int",
]  # fmt: skip


def test_the_error_table_names_every_case():
    groups = (_gemm_cases(), _conv_cases(), _block_cases(), _other_cases())
    ids = (_GEMM_IDS, _CONV_IDS, _BLOCK_IDS, _OTHER_IDS)
    for g, i in zip(groups, ids, strict=True):
        assert list(g) == i


def _case(case):
    """The (op, arguments) of an error-table case, skipping one whose op this
    build has no kernels for."""
    group, key = case.split(": ", 1)
    table = {"gemm": _gemm_cases, "conv": _conv_cases, "block": _block_cases, "other": _other_cases}
    entry = table[group]()[key]
    if entry is None:
        pytest.skip("this build has no kernels for it")
    return entry


@pytest.mark.parametrize("case", _case_ids())
def test_a_meta_call_raises_the_cpu_message(case):
    func, args = _case(case)
    with pytest.raises(RuntimeError) as cpu:
        func(*args)
    with pytest.raises(RuntimeError) as meta:
        func(*_on_meta(args))
    assert str(meta.value) == str(cpu.value)


_PRNG_CASES = [c for c in _case_ids() if c.endswith(" prng")]
_WIDTH_CASES = [c for c in _case_ids() if c.endswith(" width")]
_EXPONENT_CASES = [c for c in _case_ids() if c.endswith((" bias", " fl", " normal binades"))]
_PAST_INT_CASES = [c for c in _case_ids() if c.endswith((" past int", " past int64"))]
_MODE_CASES = [c for c in _case_ids() if c.endswith((" saturation mode", " subnormals mode"))]
_MODE_CASES += [c for c in _case_ids() if c.endswith(" mode past int")]
_ROUND_MODE_CASES = [c for c in _case_ids() if c.endswith(" round mode int")]


def _raises_in_every_mode(func, args, pattern):
    """Assert that the call raises a RuntimeError matching ``pattern`` under
    RNE and under SR, when the op takes a round mode."""
    names = [a.name for a in func._schema.arguments]
    modes = [R.RNE.value, SR.value] if "round_mode" in names else [None]
    for mode in modes:
        call = list(args)
        if mode is not None:
            call[names.index("round_mode")] = mode
        with pytest.raises(RuntimeError, match=pattern):
            func(*call)


@pytest.mark.parametrize("case", _PRNG_CASES)
def test_a_format_refuses_random_bits_past_the_carrier(case):
    """Each of the table's stochastic-rounding cases is refused for the
    random bits it adds, not for something else: the op names the format's
    width and the carrier it rounds in."""
    func, args = _case(case)
    width = r"(a \d+-bit mantissa|\d+ magnitude bits)"
    carrier = "binary64" if case == "gemm: binary64 prng" else "binary32"
    with pytest.raises(
        RuntimeError, match=rf"prng_bits must be in \[0, \d+\] for {width} rounded in {carrier}"
    ):
        func(*args)


@pytest.mark.parametrize("case", _WIDTH_CASES)
def test_a_format_refuses_widths_the_parameter_builders_cannot_take(case):
    """Each of the table's width cases is refused for the width it changes,
    in a deterministic mode as under SR: the op names the width and the range
    the parameter builders take."""
    func, args = _case(case)
    carrier = "binary64" if case == "gemm: binary64 mantissa width" else "binary32"
    widths = (
        r"-?\d+ exponent bits, but the kernels take 0 to 30",
        rf"-?\d+ stored mantissa bits, but {carrier} holds 0 to \d+",
        rf"-?\d+ magnitude bits, but {carrier} holds 1 to \d+",
    )
    _raises_in_every_mode(func, args, rf"the format has ({'|'.join(widths)})")


@pytest.mark.parametrize("case", _EXPONENT_CASES)
def test_a_format_refuses_what_the_kernels_int_exponents_cannot_take(case):
    """Each of the table's bias, fl and normal_binades cases is refused for
    the parameter it changes, in a deterministic mode as under SR: the op
    names the parameter and the range it takes. Past the bias and fl bounds
    the exponents the parameter builders compute overflow an int."""
    func, args = _case(case)
    # A GEMM's bias of 2**31 - 1 is int's limit, which it prints as "or more"
    # (test_a_gemm_value_past_an_int_is_refused_not_wrapped).
    refusals = (
        r"the format's bias is -?\d+( or more)?, but with \d+ exponent bits the kernels take -?\d+ "
        r"to \d+",
        r"the format's fl is -?\d+, but with \d+ magnitude bits the kernels take -?\d+ to \d+",
        r"the format has -?\d+ normal binades, but a superfp format with \d+ exponent bits has 1",
    )
    _raises_in_every_mode(func, args, "|".join(refusals))


@pytest.mark.parametrize("case", _PAST_INT_CASES)
def test_a_gemm_value_past_an_int_is_refused_not_wrapped(case):
    """A GEMM's packers narrow each format value to int, saturating: a value
    past int's range reaches the checks as int's limit, which every check
    refuses and prints as "or more" or "or less". A plain narrowing wrapped
    it to a value a check could accept instead (a bias of 2**40 became 0)."""
    func, args = _case(case)
    limits = r"(2147483647 or more|-2147483648 or less|9223372036854775807 or more)"
    with pytest.raises(RuntimeError, match=limits):
        func(*args)


@pytest.mark.parametrize("case", _MODE_CASES)
def test_a_format_refuses_a_mode_integer_that_names_no_mode(case):
    """A saturation or subnormals mode that is not a value of its enum in
    modes.h is refused, in a deterministic mode as under SR, rather than run
    as whichever mode the casts' comparisons happen to select."""
    func, args = _case(case)
    mode = r"the format's (saturation|subnormals) mode is -?\d+( or more| or less)?, but a "
    _raises_in_every_mode(func, args, mode + r"(SaturationMode|SubnormalsMode) is 0 to \d")


@pytest.mark.parametrize("case", _ROUND_MODE_CASES)
def test_a_quantizer_refuses_a_round_mode_it_does_not_know(case):
    """The elementwise quantizers refuse a round mode that is not a RoundMode,
    as the GEMMs do; they used to run it as round to odd."""
    func, args = _case(case)
    with pytest.raises(RuntimeError, match=r"-?\d+ is not a RoundMode"):
        func(*args)


def test_a_format_the_call_does_not_round_with_is_not_bounded():
    """An accumulate format under accumulate_quant=False is never built, and
    the Python layer passes it unchecked (with P = 0, and any prng_bits), so
    the op checks neither its widths nor its prng_bits against the carrier,
    only the prng_bits against the byte the spec stores N in. And binary64
    has room for more random bits than binary32."""
    cases = [
        _with("custom_matmul_binaryK", accumulate_quant=False, acc_prng_bits=200),
        _with("custom_matmul_block", acc_prng_bits=200),
        _with(
            "custom_matmul_binaryK_fma",
            **_in_float64("custom_matmul_binaryK_fma"),
            fma_prng_bits=42,
        ),
        _with("custom_matmul_binaryK", accumulate_quant=False, acc_K=0, acc_P=0),
        # not even past an int's range, which it narrows with saturation
        _with("custom_matmul_binaryK", accumulate_quant=False, acc_K=2**40, acc_bias=-(2**40)),
        # nor its modes
        _with("custom_matmul_binaryK", accumulate_quant=False, acc_saturation_mode=9),
    ]
    for func, args in cases:
        func(*args)
        func(*_on_meta(args))


def test_a_quantizer_refuses_another_dtype_on_meta_too():
    """The CPU refuses through AT_DISPATCH, whose NotImplementedError would make
    FakeTensorMode take the op for one without a Meta kernel; the meta kernel
    raises a RuntimeError naming the dtype."""
    x = torch.ones(4, 4, dtype=torch.int32)
    for op, args in (
        (torch.ops.mptorch.binaryK_quant.default, (8, 4, 7, 0, True, 0, 0, 0)),
        (torch.ops.mptorch.superfp_quant.default, (3, 4, 2, 7, 0, True, 0, 0)),
        (torch.ops.mptorch.fixedpoint_quant.default, (8, 5, 0, True, False, 0)),
    ):
        with pytest.raises(NotImplementedError, match="not implemented for 'Int'"):
            op(x, *args)
        with pytest.raises(RuntimeError, match="expected a float32, float64, float16 or bfloat16"):
            op(x.to("meta"), *args)


def test_an_empty_gemm_returns_before_the_dtype_check():
    """The drivers return an empty result before they read the dtypes, on
    meta tensors too."""
    func, args = _with(
        "custom_matmul_binaryK", a=torch.randn(0, 32), b=torch.randn(32, 6, dtype=torch.float64)
    )
    want = func(*args)
    got = func(*_on_meta(args))
    assert (got.shape, got.dtype) == (want.shape, want.dtype) == (torch.Size([0, 6]), torch.float32)


# --- torch.export -------------------------------------------------------------------


def _exported(model, x):
    dims = {0: torch.export.Dim.AUTO}
    ep = torch.export.export(model, (x,), dynamic_shapes=(dims,))
    ops = {
        n.target.name()
        for n in ep.graph.nodes
        if n.op == "call_function" and getattr(n.target, "namespace", None) == "mptorch"
    }
    return ep, ops


EXPORT_CASES = {
    "linear": (
        lambda: QLinear(32, 16, formats=binaryK_gemm_formats(8, 4, acc_K=16, acc_P=11)),
        (8, 32),
        (5, 32),
        "mptorch::custom_matmul_binaryK",
    ),
    "conv": (
        lambda: QConv2d(3, 4, 3, padding=1, formats=conv_formats(SplitMac(BK, BK_ACC))),
        (2, 3, 6, 6),
        (3, 3, 6, 6),
        "mptorch::custom_conv_binaryK",
    ),
    "block": (
        lambda: QLinear(64, 16, formats=block_gemm_formats(MXFP8_E4M3, MXFP8_E4M3)),
        (8, 64),
        (5, 64),
        "mptorch::custom_matmul_block",
    ),
}


@pytest.mark.parametrize("name", list(EXPORT_CASES))
def test_export(name):
    """A layer over the ops exports with a dynamic batch, and the exported
    program gives the layer's result at the traced batch and another. The
    program is the forward alone, the layer's autograd.Function inlined, so it
    runs under no_grad: in grad mode its op meets the weight, which requires
    grad, and refuses it as the op's Autograd kernel refuses any."""
    make, shape, other, op = EXPORT_CASES[name]
    torch.manual_seed(0)
    model = make()
    ep, ops = _exported(model, torch.randn(*shape))
    assert op in ops, ops
    program = ep.module()
    for s in (shape, other):
        x = torch.randn(*s)
        with torch.no_grad():
            assert torch.equal(program(x), model(x))
    with pytest.raises(RuntimeError, match="is not differentiable"):
        program(torch.randn(*shape))


# --- torch.compile --------------------------------------------------------------------

_INDUCTOR_SKIP = pytest.mark.skipif(
    sys.platform != "linux", reason="Inductor's CPU code generation is exercised on Linux"
)
BACKENDS = ["aot_eager", pytest.param("inductor", marks=_INDUCTOR_SKIP)]


def _pipeline(rounding, prng_bits):
    """Two GEMMs and a quantizer through the flat wrappers, which have no
    autograd.Function and so trace into one graph."""

    def mm(a, b, trans_b=False):
        return binaryK_matmul(
            a, b, trans_b=trans_b, mul_K=8, mul_P=4, mul_prng_bits=prng_bits, acc_K=16, acc_P=11,
            acc_prng_bits=prng_bits, rounding_mode=rounding,
        )  # fmt: skip

    def f(a, b):
        q = binaryK_quantize(a, 8, 4, prng_bits=prng_bits, rounding_mode=rounding)
        return mm(torch.relu(mm(q, b)), b, trans_b=True)

    return f


def _mlp(rounding, prng_bits, slots=False):
    f = binaryK_gemm_formats(
        8, 4, mul_prng_bits=prng_bits, acc_K=16, acc_P=11, acc_prng_bits=prng_bits,
        rounding_mode=rounding,
    )  # fmt: skip
    if slots:
        # mptorch's own quantizer, which the weight's default recompute runs
        # again in backward; under SR that replay keeps the copy under compile
        f.input_quant = f.weight_quant = Quant(BinaryK(8, 4, prng_bits=prng_bits), rounding)
    torch.manual_seed(0)
    return nn.Sequential(QLinear(32, 32, formats=f), nn.ReLU(), QLinear(32, 8, formats=f))


def _conv():
    f = conv_formats(SplitMac(BinaryK(8, 4), BinaryK(16, 11)))
    f.input_quant = f.weight_quant = Quant(BinaryK(8, 4))
    torch.manual_seed(0)
    return QConv2d(3, 4, 3, stride=2, formats=f)


class _Matmul(nn.Module):
    """A ``QMatmul`` whose second operand is a parameter."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.mm = QMatmul(SplitMac(BinaryK(8, 4, prng_bits=8), BinaryK(16, 11), rounding=SR))
        self.b = nn.Parameter(torch.randn(32, 8))

    def forward(self, x):
        return self.mm(x, self.b)


def _straight_through():
    torch.manual_seed(0)
    f = binaryK_gemm_formats(8, 4, acc_K=16, acc_P=11)
    return nn.Sequential(QLinear(32, 8, formats=f), Quantizer(BinaryK(8, 4), BinaryK(8, 5)))


# Each of the four autograd Functions, and the paths through _saved: name ->
# (model, input shape).
COMPILE_MODELS = {
    "linear, SR hooks": (lambda: _mlp(SR, 8), (16, 32)),
    "linear, RNE slots": (lambda: _mlp(R.RNE, 0, slots=True), (16, 32)),
    "linear, SR slots": (lambda: _mlp(SR, 8, slots=True), (16, 32)),
    "conv, slots": (_conv, (2, 3, 9, 9)),
    "matmul, SR": (_Matmul, (16, 32)),
    "quantizer": (_straight_through, (16, 32)),
}


def _step(model, x, seed):
    """The output and every parameter's gradient, by name."""
    torch.manual_seed(seed)
    model.zero_grad()
    out = model(x)
    out.square().sum().backward()
    return [("out", out.detach())] + [(n, p.grad.clone()) for n, p in model.named_parameters()]


def test_the_flat_wrappers_trace_into_one_graph():
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(_pipeline(SR, 8))(torch.randn(8, 16), torch.randn(16, 16))
    assert (explanation.graph_count, explanation.graph_break_count) == (1, 0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rounding", [R.RNE, SR])
def test_compiled_ops_equal_eager(backend, rounding):
    """The ops in one compiled graph give eager's results, bit for bit. Under SR
    that needs Inductor to run the random ops in program order, which is what
    ``fallback_random`` asks of it."""
    torch._dynamo.reset()
    f = _pipeline(rounding, 8 if rounding is SR else 0)
    compiled = torch.compile(f, backend=backend)
    a, b = torch.randn(8, 16), torch.randn(16, 16)
    with torch._inductor.config.patch(fallback_random=True):
        torch.manual_seed(3)
        got = compiled(a, b)
    torch.manual_seed(3)
    assert torch.equal(got, f(a, b))


@pytest.mark.parametrize("name", list(COMPILE_MODELS))
def test_the_layers_trace_into_one_graph(name):
    """Every layer's autograd.Function traces: one graph, no break. Each
    layer used to break the graph where its Function read a class attribute
    (Dynamo's gb4841), and a stochastic replay where it read the generator's
    state (``_saved`` keeps the copy under compile instead)."""
    make, shape = COMPILE_MODELS[name]
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(make())(torch.randn(*shape))
    assert (explanation.graph_count, explanation.graph_break_count) == (1, 0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(COMPILE_MODELS))
def test_a_compiled_model_equals_eager(backend, name):
    """A model compiled in one graph gives eager's forward and gradients, bit
    for bit. Inductor compiles the plain torch arithmetic around the
    operators as well, and its bias gradient (a sum over the batch, in the
    order Inductor picks) can differ in the last bits, as a plain
    ``nn.Linear``'s does; every result an mptorch operator rounds is eager's."""
    make, shape = COMPILE_MODELS[name]
    torch._dynamo.reset()
    model = make()
    compiled = torch.compile(copy.deepcopy(model), backend=backend, fullgraph=True)
    x = torch.randn(*shape)
    with torch._inductor.config.patch(fallback_random=True):
        got = _step(compiled, x, seed=1)
    want = _step(model, x, seed=1)
    for (label, w), (_, g) in zip(want, got, strict=True):
        if backend == "inductor" and label.endswith("bias"):
            torch.testing.assert_close(g, w)
        else:
            assert torch.equal(w, g), label


def test_a_format_warning_compiles_and_names_the_caller():
    """A format that warns compiles in one graph and warns while the graph is
    traced, naming the line that called the compiled model, as eager names
    the line that called the layer; the compiled calls after it do not warn
    again. Here the sums reach 2**127, which a float16 result cannot hold."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    layer = QLinear(32, 8, formats=binaryK_gemm_formats(8, 4, acc_K=16, acc_P=8)).half()
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    x = torch.randn(4, 32).half()
    with pytest.warns(FormatRangeWarning, match="float16") as record:
        got = compiled(x)
    assert {w.filename for w in record if w.category is FormatRangeWarning} == {__file__}
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        compiled(x)
    assert not [w for w in again if w.category is FormatRangeWarning]
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", FormatRangeWarning)
        assert torch.equal(got.detach(), layer(x))


def test_a_refused_format_keeps_its_reason_under_compile():
    """A format the call refuses raises its ValueError eagerly, and under a
    compile that may break the graph; under ``fullgraph`` Dynamo raises
    instead, with the ValueError's message in its own."""

    def f(a, b):
        return binaryK_matmul(a, b, mul_K=8, mul_P=4, acc_K=16, acc_P=12)

    a, b = torch.randn(4, 16).half(), torch.randn(16, 16).half()
    with pytest.raises(ValueError, match="12 bits of precision"):
        f(a, b)
    torch._dynamo.reset()
    with pytest.raises(ValueError, match="12 bits of precision"):
        torch.compile(f, backend="aot_eager")(a, b)
    torch._dynamo.reset()
    with pytest.raises(Unsupported, match="12 bits of precision"):
        torch.compile(f, backend="aot_eager", fullgraph=True)(a, b)


def test_the_warning_helper_carries_dynamos_constant_marker():
    """``mptorch.number.base._warn_per_call`` sets the attribute
    ``torch.compiler.assume_constant_result`` sets, without importing
    ``torch._dynamo``. Should torch rename it, a warning would break the
    graph again; this says so first."""
    marked = assume_constant_result(lambda: None)
    assert set(vars(marked)) <= set(vars(_warn_per_call))
    assert all(getattr(_warn_per_call, k) == v for k, v in vars(marked).items())


def test_compile_takes_a_dynamic_batch_without_recompiling():
    """A graph compiled at one batch size serves the next ones: the Meta
    kernels take a symbolic batch."""
    torch._dynamo.reset()
    graphs = []

    def counting(gm, example_inputs):
        graphs.append(gm)
        return torch._dynamo.lookup_backend("aot_eager")(gm, example_inputs)

    f = _pipeline(R.RNE, 0)
    compiled = torch.compile(f, backend=counting, dynamic=True)
    b = torch.randn(16, 16)
    for m in (5, 9, 13):
        a = torch.randn(m, 16)
        assert torch.equal(compiled(a, b), f(a, b))
    assert len(graphs) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_compile_keeps_two_identical_sr_calls_apart(backend):
    """Two identical SR calls in a training graph are two draws, as in eager.
    The ops' ``nondeterministic_seeded`` tag is what stops AOTAutograd's CSE
    from merging them: without it both backends returned one draw twice."""
    torch._dynamo.reset()
    op = torch.ops.mptorch.binaryK_quant.default

    def f(x, w):
        y = op(x, 8, 4, 7, 10, True, SR.value, 0, 0)
        z = op(x, 8, 4, 7, 10, True, SR.value, 0, 0)
        return y * w, z * w

    x = torch.rand(4096) * 0.1 + 1.0
    w = torch.ones(4096, requires_grad=True)
    y, z = f(x, w)
    assert not torch.equal(y, z)
    y, z = torch.compile(f, backend=backend)(x, w)
    assert not torch.equal(y, z)


@requires_cuda
@pytest.mark.parametrize("rounding", [R.RNE, SR])
def test_compile_on_cuda(rounding):
    torch._dynamo.reset()
    f = _pipeline(rounding, 8 if rounding is SR else 0)
    compiled = torch.compile(f)
    a, b = torch.randn(8, 16, device="cuda"), torch.randn(16, 16, device="cuda")
    with torch._inductor.config.patch(fallback_random=True):
        torch.manual_seed(3)
        got = compiled(a, b)
    torch.manual_seed(3)
    assert torch.equal(got, f(a, b))
