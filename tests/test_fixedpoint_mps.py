"""
The fixed-point GEMM and conv ops on MPS: every MPS result
is the CPU op's, word for word, stochastic rounding included, since MPS
draws its seed from the CPU generator and keys each element's Philox stream
as the CPU does. The one exception is the GEMM's documented flush of binary32
subnormal products (``gemm.metal``; ``concepts.rst``'s "On an Apple GPU"),
which the last tests pin. The CPU ops themselves are held to the definition
in ``test_fixedpoint_gemm.py``; this file holds MPS to the CPU.

- **GEMMs:** every case of ``OP_CASES`` whose result MPS can hold, in all
  nine modes, over a ragged shape that runs ``gemm.metal``'s 16 x 16 tiles
  past both edges; float16 and bfloat16 operands; batched and transposed
  operands with NaN and infinite products; and the carrier path against the
  integer accumulator on MPS.
- **Convs:** the three passes of ``CONV_CASES`` on ``2d_all`` in all nine
  modes, and every geometry in RNE, RZ and SR.
- **Refusals:** a wide (float64) result, which MPS does not have, raises by
  name, from the GEMM and the conv ops; a format the shared checks refuse
  raises as on the CPU.
- **The flush:** a product below 2^-126 is zero on MPS before any rounding,
  so under RU it rounds to zero there and to one step on the CPU. The integer
  accumulator's sums and its result's scale are exact, including an
  accumulate format whose step is a binary32 subnormal.
"""

import dataclasses
import math

import pytest
import torch

from mptorch import FixedPoint, RoundMode

from .markers import requires_mps
from .test_fixedpoint_gemm import (
    CARRIER_PAIRS,
    CONV_CASES,
    DETERMINISTIC,
    GEOMETRIES,
    OP_CASES,
    PRECISION,
    STOCHASTIC,
    _conv,
    _conv_operands,
    _op,
    _operands,
    _Raw,
    _same_words,
)

R = RoundMode
MODES = DETERMINISTIC + STOCHASTIC
mps_device = [pytest.param("mps", marks=requires_mps)]


def _wide(acc: FixedPoint, dtype: torch.dtype) -> bool:
    return acc.mag_bits > PRECISION[dtype]


# The cases whose float32 result MPS holds, and the two it refuses.
MPS_CASES = [c for c in OP_CASES if not _wide(OP_CASES[c][1], torch.float32)]
WIDE_CASES = [c for c in OP_CASES if _wide(OP_CASES[c][1], torch.float32)]


def _formats(case, mode):
    """A case's formats, with six random bits each under stochastic rounding."""
    mul, acc, wrap = OP_CASES[case]
    if mode in STOCHASTIC:
        mul = None if mul is None else dataclasses.replace(mul, prng_bits=6)
        acc = dataclasses.replace(acc, prng_bits=6)
    return mul, acc, wrap


def _both(device, a, b, seed, **kw):
    """The op on the CPU and on ``device``, each under the same seed."""
    torch.manual_seed(seed)
    want = _op(a, b, **kw)
    torch.manual_seed(seed)
    got = _op(a.to(device), b.to(device), **kw)
    assert got.device.type == device and got.dtype == want.dtype
    return got.cpu(), want


# --- the GEMMs --------------------------------------------------------------------


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("case", MPS_CASES)
@pytest.mark.parametrize("mode", MODES)
def test_mps_gives_the_cpus_words(mode, case, device):
    """Over several tiles in each direction and a ragged edge."""
    mul, acc, wrap = _formats(case, mode)
    a, b = _operands((37, 70), (70, 45), torch.float32, seed=len(case) + 1)
    got, want = _both(device, a, b, 31, acc=acc, mul=mul, mode=mode, wrap=wrap)
    assert _same_words(got, want)


# (mul, acc, wrap) pairs whose results float16 and bfloat16 hold
NARROW_CASES = {
    "carrier path": (*CARRIER_PAIRS[torch.bfloat16], False),
    "saturating midway": OP_CASES["saturating midway"],
    "wrapping": OP_CASES["wrapping"],
    "fused wrapping": OP_CASES["fused wrapping"],
}


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", list(NARROW_CASES))
@pytest.mark.parametrize("mode", [R.RNE, R.RD, R.SR])
def test_narrow_operands_give_the_cpus_words(mode, case, dtype, device):
    mul, acc, wrap = NARROW_CASES[case]
    if mode in STOCHASTIC:
        mul = None if mul is None else dataclasses.replace(mul, prng_bits=4)
        acc = dataclasses.replace(acc, prng_bits=4)
    assert not _wide(acc, dtype)
    a, b = _operands((37, 70), (70, 45), dtype, seed=32)
    got, want = _both(device, a, b, 33, acc=acc, mul=mul, mode=mode, wrap=wrap)
    assert got.dtype == dtype and _same_words(got, want)


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("case", ["carrier path", "saturating midway", "wrapping", "fused"])
def test_mps_takes_batches_transposes_and_non_finite_products(case, device):
    a, b = _operands((3, 70, 20), (24, 70), torch.float32, seed=21)
    a[1, 5, 2], a[2, 7, 3], a[0, 1, 1] = math.nan, math.inf, -math.inf
    b[3, 7] = 0.0  # inf * 0 in batch 2
    for mode in (R.RNE, R.RU, R.SR):
        mul, acc, wrap = _formats(case, mode)
        got, want = _both(
            device, a, b, 34, acc=acc, mul=mul, mode=mode, wrap=wrap, trans_a=True, trans_b=True
        )
        assert got.shape == (3, 20, 24) and _same_words(got, want), mode
        assert got[1].isnan().any() and got[2].isnan().any(), mode


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", MODES)
def test_on_mps_the_carrier_path_is_the_integer_path(mode, dtype, device):
    """The test hook runs the integer accumulator where the rules choose the
    carrier path: on MPS too, both give the CPU's words."""
    mul, acc = CARRIER_PAIRS[dtype]
    a, b = _operands((40, 70), (70, 33), dtype, seed=22, scale=6.0)
    torch.manual_seed(23)
    want = _op(a, b, acc=acc, mul=mul, mode=mode)
    for integer in (False, True):
        torch.manual_seed(23)
        got = _op(a.to(device), b.to(device), acc=acc, mul=mul, mode=mode, integer=integer)
        assert torch.equal(got.cpu(), want), integer


# --- the conv ops -----------------------------------------------------------------


def _passes_against_the_cpu(geometry, device, case, mode):
    """The forward, input-gradient and weight-gradient passes on ``device``,
    each against the same pass on the CPU, under the same seed."""
    mul, acc, wrap = _formats(case, mode)
    (x, w, dy), geo, size, kernel, out, groups = _conv_operands(geometry, "cpu", torch.float32)
    for conv_pass, a, b, out_size in ((0, w, x, out), (1, w, dy, size), (2, dy, x, kernel)):
        kw = dict(acc=acc, mul=mul, mode=mode, wrap=wrap)
        torch.manual_seed(35)
        want = _conv(conv_pass, a, b, out_size, geo, groups, **kw)
        torch.manual_seed(35)
        got = _conv(conv_pass, a.to(device), b.to(device), out_size, geo, groups, **kw)
        assert got.device.type == device and got.shape == want.shape, conv_pass
        assert got.dtype == want.dtype and _same_words(got.cpu(), want), conv_pass


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("case", CONV_CASES)
@pytest.mark.parametrize("mode", MODES)
def test_each_conv_pass_gives_the_cpus_words(mode, case, device):
    _passes_against_the_cpu("2d_all", device, case, mode)


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
@pytest.mark.parametrize("mode", [R.RNE, R.RZ, R.SR])
def test_the_conv_geometries_give_the_cpus_words(geometry, mode, device):
    for case in ("carrier path", "fused"):
        _passes_against_the_cpu(geometry, device, case, mode)


# --- the refusals -----------------------------------------------------------------

WIDE_MESSAGE = (
    r"has (\d+) magnitude bits, more than float32's 24, so the result is float64, which MPS "
    r"does not have: use an accumulate format of at most 24 magnitude bits, or compute on the CPU"
)


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("mul", [FixedPoint(16, 4), None], ids=["split", "fused"])
def test_a_wide_result_is_refused_by_name(mul, device):
    """A float64 result, which the CPU returns for an accumulate format wider
    than float32's significand, is refused on MPS, by the GEMM and the conv
    ops, with the reason and the fix."""
    acc = FixedPoint(40, 4)
    a, b = torch.ones(2, 3, device=device), torch.ones(3, 2, device=device)
    op = "custom_matmul_fixedpoint" + ("" if mul else "_fma")
    with pytest.raises(RuntimeError, match=f"{op}: the accumulate format {WIDE_MESSAGE}") as e:
        _op(a, b, acc=acc, mul=mul)
    assert "has 39 magnitude bits" in str(e.value)
    x, w = torch.ones(1, 2, 5, 5, device=device), torch.ones(3, 2, 3, 3, device=device)
    geo = ([1, 1], [0, 0], [1, 1])
    op = "custom_conv_fixedpoint" + ("" if mul else "_fma")
    with pytest.raises(RuntimeError, match=f"{op}: the accumulate format {WIDE_MESSAGE}"):
        _conv(0, w, x, [3, 3], geo, 1, acc=acc, mul=mul)
    assert _op(a.cpu(), b.cpu(), acc=acc, mul=mul).dtype == torch.float64


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("case", WIDE_CASES)
def test_the_wide_op_cases_are_refused(case, device):
    mul, acc, wrap = OP_CASES[case]
    a, b = _operands((5, 7), (7, 6), torch.float32, seed=36)
    with pytest.raises(RuntimeError, match="MPS does not have"):
        _op(a.to(device), b.to(device), acc=acc, mul=mul, wrap=wrap)


@pytest.mark.parametrize("device", mps_device)
def test_a_format_refusal_is_the_cpus(device):
    """The shared checks run before anything MPS-specific: the same message."""
    acc, mul = _Raw(60, 4), FixedPoint(16, 8)
    a, b = torch.ones(2, 3, device=device), torch.ones(3, 2, device=device)
    with pytest.raises(RuntimeError, match="accumulates 1 to 53") as on_mps:
        _op(a, b, acc=acc, mul=mul)
    with pytest.raises(RuntimeError) as on_cpu:
        _op(a.cpu(), b.cpu(), acc=acc, mul=mul)
    assert str(on_mps.value).splitlines()[0] == str(on_cpu.value).splitlines()[0]


# --- the flush --------------------------------------------------------------------


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("integer", [False, True, None])  # None: the fused mac
def test_a_subnormal_product_flushes_on_mps(integer, device):
    """2^-70 * 2^-60 = 2^-130, a binary32 subnormal: under RU the CPU rounds
    it to the format's step, and the GPU, which flushes it to zero before any
    rounding, to zero. A normal product beside it rounds the same on both."""
    acc = FixedPoint(16, 4)
    a = torch.tensor([[2.0**-70], [2.0**-10]])
    b = torch.tensor([[2.0**-60]])
    kw = dict(acc=acc, mode=R.RU)
    if integer is not None:
        kw.update(mul=FixedPoint(16, 8), integer=integer)
    got, want = _both(device, a, b, 37, **kw)
    assert want[0, 0].item() == 2.0**-4 and got[0, 0].item() == 0.0
    assert got[1, 0].item() == want[1, 0].item() == 2.0**-4


# Accumulate formats whose step is a binary32 subnormal, held by the integer
# accumulator: its result is code * 2^-fl, and a GPU multiply would flush that
# scale. (12, 140): every value subnormal, and the largest, 2^-129, below every
# normal product, so the fused mac's saturation bound is a subnormal too.
# (24, 135): values up to 2^-112, normal ones on a subnormal step.
SUBNORMAL_STEPS = {
    "s12.140": (FixedPoint(8, 120), FixedPoint(12, 140)),
    "s24.135": (FixedPoint(8, 120), FixedPoint(24, 135)),
}


def _tiny_operands(seed):
    """Operands of magnitude [1, 2) * 2^-60 and random sign, so that every
    product is a normal binary32 value in [2^-120, 2^-118), on the multiply
    format's grid after rounding; one row of A is zero, a row of zero
    products."""
    g = torch.Generator().manual_seed(seed)

    def draw(*shape):
        sign = torch.randint(0, 2, shape, generator=g) * 2 - 1
        return sign * (1 + torch.rand(*shape, generator=g)) * 2.0**-60

    a, b = draw(37, 70), draw(70, 45)
    a[3] = 0.0
    return a, b


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("name", list(SUBNORMAL_STEPS))
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("mode", [R.RNE, R.RU, R.RZ, R.SR])
def test_an_accumulate_step_below_binary32s_normals_is_exact(mode, fused, name, device):
    mul, acc = SUBNORMAL_STEPS[name]
    if mode in STOCHASTIC:
        mul, acc = (dataclasses.replace(f, prng_bits=6) for f in (mul, acc))
    a, b = _tiny_operands(38)
    got, want = _both(device, a, b, 39, acc=acc, mul=None if fused else mul, mode=mode)
    assert want.abs().max() > 0 and torch.equal(want[3], torch.zeros(45))
    assert torch.equal(got, want)


@pytest.mark.parametrize("device", mps_device)
@pytest.mark.parametrize("name", list(SUBNORMAL_STEPS))
@pytest.mark.parametrize("fused", [False, True])
def test_a_conv_onto_a_subnormal_step_is_exact(fused, name, device):
    """The same formats through a conv pass, whose padding gives zero
    products; every other product is normal, in [2^-120, 2^-118)."""
    mul, acc = SUBNORMAL_STEPS[name]
    mul = None if fused else mul
    (x, w, dy), geo, size, kernel, out, groups = _conv_operands("2d_all", "cpu", torch.float32)
    x, w = (t.sign() * (1 + t.abs().frac()) * 2.0**-60 for t in (x, w))
    want = _conv(0, w, x, out, geo, groups, acc=acc, mul=mul, mode=R.RNE)
    got = _conv(0, w.to(device), x.to(device), out, geo, groups, acc=acc, mul=mul, mode=R.RNE)
    assert want.abs().max() > 0 and torch.equal(got.cpu(), want)
