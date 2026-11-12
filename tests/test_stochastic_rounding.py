"""
P3109's three stochastic rounding modes, exactly.

With :math:`\\eta` the fraction of :math:`|x|` between its two neighbouring
values, :math:`R` the draw's :math:`N` random bits (``prng_bits``) and
:math:`X = \\eta 2^N`, P3109 (arXiv:2606.04028, Fig. 2) rounds the magnitude
away from zero iff

- ``RoundMode.SR`` (StochasticA): :math:`\\lfloor X \\rfloor + R \\ge 2^N`;
- ``RoundMode.SR_B`` (StochasticB): :math:`\\lfloor 2X \\rfloor + 2R + 1 \\ge 2^{N+1}`;
- ``RoundMode.SR_C`` (StochasticC): :math:`\\mathrm{RNITE}(X) + R \\ge 2^N`.

So over all :math:`2^N` draws, :math:`\\lfloor X \\rfloor`, :math:`\\lfloor X + 1/2
\\rfloor` and :math:`\\mathrm{RNITE}(X)` of them go up. The test-only op
``_stochastic_cast`` takes each element's random word, which lets these tests
run every draw and count, in every region of every format: binaryK under its
three subnormal modes, superfp's supernormal binades and the region below its
smallest value, fixed point, both carriers, on the CPU and CUDA. The reference
is the figure on ``Fraction``, with the neighbours from the kernels'
deterministic casts; gfloat's three modes are a second oracle for binaryK.
Before cf69807, binaryK's subnormal arm and superfp's arm below its smallest
value rounded ``|x|`` in the carrier first, and three arms ignored
``prng_bits`` (``dev/benchmarks/sweeps/cast_all_modes_sweep.cu sr-mean-old`` is that
negative control).

Then every op that rounds stochastically takes the two new modes, CPU and CUDA
agree bit for bit, and the Python layer knows them.
"""

import math
from fractions import Fraction

import pytest
import torch

from mptorch import MXFP8_E4M3, AccumulateAlgorithm, BinaryK, RoundMode, SubnormalsMode, SuperFP
from mptorch.number import SaturationMode
from mptorch.quant import (
    FusedMac,
    Palette,
    QConv2d,
    QLinear,
    Quant,
    SplitMac,
    binaryK_gemm_formats,
    binaryK_quantize,
    block_matmul,
    block_pack,
    block_quantize,
    block_unpack,
    conv_formats,
    fixedpoint_quantize,
    qmatmul,
    superfp_quantize,
)
from tests.markers import available_devices, float64_devices

# gfloat 0.5's round_float inverts a bool with ~, which Python 3.13 deprecates
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:gfloat")

R = RoundMode
A = AccumulateAlgorithm
STOCHASTIC = [R.SR, R.SR_B, R.SR_C]
S, NORM, EXT = SubnormalsMode.SUBNORMALS, SubnormalsMode.NORMALS, SubnormalsMode.EXTENDED_NORMALS
SAT = SaturationMode.SAT_NONE

CAST = torch.ops.mptorch._stochastic_cast.default
MAN = {torch.float32: 23, torch.float64: 52}


def _expected_up(x: Fraction, lo: Fraction, hi: Fraction, n: int, mode: RoundMode) -> int:
    """How many of the 2^n draws P3109 sends |x| from lo to hi."""
    if hi == lo:
        return 0
    X = (abs(x) - lo) / (hi - lo) * 2**n
    if mode is R.SR:
        return math.floor(X)
    if mode is R.SR_B:
        return math.floor(X + Fraction(1, 2))
    return round(X)  # Fraction's round is to nearest, ties to even: RNITE


# --- the formats, each with its quantizer for the neighbours ------------------


class Fmt:
    """A format, the fields _stochastic_cast takes, where its draw field sits,
    and its deterministic quantizer (for lo and hi)."""

    def __init__(self, family, fields, man_bits, quantize, name):
        self.family, self.fields, self.man_bits, self.quantize, self.name = (
            family,
            fields,
            man_bits,
            quantize,
            name,
        )

    def shift(self, dtype, n):
        """The bit of the draw word that holds R's lowest bit."""
        return MAN[dtype] - n if self.family == 2 else MAN[dtype] - self.man_bits - n


def binaryK(K, P, bias, sub, prng):
    def q(x, mode):
        return binaryK_quantize(x, K, P, bias, rounding_mode=mode, subnormals_mode=sub)

    return Fmt(
        0,
        [K, P, bias, prng, 1, SAT.value, sub.value],
        P - 1,
        q,
        f"binary{K}p{P} b{bias} {sub.name} N{prng}",
    )


def superfp(man, exp, nb, bias, prng):
    def q(x, mode):
        return superfp_quantize(x, man, exp, nb, bias, rounding_mode=mode)

    return Fmt(
        1,
        [man, exp, nb, bias, prng, 1, SAT.value],
        man,
        q,
        f"superfp m{man} e{exp} n{nb} b{bias} N{prng}",
    )


def fixedpoint(wl, fl, prng):
    def q(x, mode):
        return fixedpoint_quantize(x, wl, fl, rounding_mode=mode)

    return Fmt(2, [wl, fl, prng, 1, 0], 0, q, f"fixed wl{wl} fl{fl} N{prng}")


FORMATS = [
    binaryK(8, 4, 7, S, 3),
    binaryK(8, 4, 7, NORM, 3),
    binaryK(8, 4, 7, EXT, 3),
    binaryK(8, 2, 7, S, 2),
    binaryK(8, 2, 7, EXT, 1),
    binaryK(8, 1, 7, NORM, 2),
    binaryK(16, 9, 15, S, 8),
    binaryK(16, 9, 15, EXT, 8),
    binaryK(8, 4, 7, S, 0),
    binaryK(8, 4, 7, EXT, 0),
    superfp(3, 4, 2, 7, 3),
    superfp(0, 4, 2, 7, 2),
    superfp(2, 3, 1, 4, 8),
    superfp(3, 4, 2, 7, 0),
    fixedpoint(8, 5, 3),
    fixedpoint(12, 10, 8),
    fixedpoint(8, 5, 0),
]


def _inputs(fmt: Fmt, dtype) -> torch.Tensor:
    """Inputs in every region of the format: around its smallest value and
    the bottom of its normal binades, a hair either side of each draw level
    there, and ordinary values above."""
    tiny = torch.tensor([torch.finfo(dtype).tiny], dtype=dtype)
    vmin = fmt.quantize(tiny, R.RU).item()  # the smallest positive value
    xs = [vmin * k / 16 for k in range(1, 48)]
    n = fmt.fields[3] if fmt.family == 0 else fmt.fields[4] if fmt.family == 1 else fmt.fields[2]
    for k in range(1, 2**n):
        lvl = vmin * k / 2**n
        xs += [math.nextafter(lvl, 0.0), math.nextafter(lvl, 1.0)]
    xs += [vmin * f for f in (2.5, 3.3, 7.7, 16.6, 33.3)]
    xs += [0.3, 0.7, 1.1, 1.9, 3.14159]
    t = torch.tensor(xs, dtype=dtype)
    return torch.cat([t, -t])


def _counts(fmt, x, n, mode, dtype, device):
    """For each input, how many of the 2^n draws the kernel sends up, and
    whether every result was one of the two neighbours."""
    m = x.numel()
    draws = (torch.arange(2**n, dtype=torch.int64) << fmt.shift(dtype, n)).repeat(m)
    xs = x.repeat_interleave(2**n)
    y = CAST(xs.to(device), draws.to(device), fmt.family, fmt.fields, mode.value).cpu()
    lo = fmt.quantize(x.abs(), R.RZ).abs()
    hi = fmt.quantize(x.abs(), R.RU).abs()
    ya = y.abs().view(m, 2**n)
    up = (ya == hi[:, None]) & (hi != lo)[:, None]
    stray = (ya != hi[:, None]) & (ya != lo[:, None])
    return up.sum(1), stray.any(1), lo, hi


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["binary32", "binary64"])
@pytest.mark.parametrize("mode", STOCHASTIC, ids=lambda m: m.name)
@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_every_draw_counts_as_p3109_says(fmt, mode, dtype, device):
    """Over all 2^N draws, exactly P3109's count goes up, and every result is
    one of the two neighbours, in every region."""
    n = fmt.fields[3] if fmt.family == 0 else fmt.fields[4] if fmt.family == 1 else fmt.fields[2]
    x = _inputs(fmt, dtype)
    # below the largest value: saturation is another test's, and two's
    # complement saturates the two signs differently
    vmax = fmt.quantize(torch.tensor([1e30], dtype=dtype), R.RZ).item()
    x = x[(x.abs() < vmax) & (x != 0)]
    up, stray, lo, hi = _counts(fmt, x, n, mode, dtype, device)
    assert not stray.any(), x[stray]
    for i in range(x.numel()):
        want = _expected_up(
            Fraction(x[i].item()), Fraction(lo[i].item()), Fraction(hi[i].item()), n, mode
        )
        assert int(up[i]) == want, (x[i].item(), int(up[i]), want)


# A superfp whose smallest value 2^c is itself a carrier subnormal: binary32's
# c = -145 (SuperFP(0, 8, 1, 146)) and binary64's c = -1070 (SuperFP(3, 4, 1,
# 966)). A carrier subnormal takes the arm below 2^c, read at the exponent
# 1 - BIAS, with the shift sh = MAN + c - (1 - BIAS) - N - 1 (3 - N for both),
# which was undefined behaviour when negative. The arm's rule
# (stochastic.h's sr_away_below_or_above, reference_casts.h's
# ref_sr_coarse_floor): zero or 2^c, and while sh >= 0 P3109's count on
# |x| / 2^c below it and every draw up at or above it; once sh < 0, every draw
# up, which is the definition at or above 2^c and the user's choice below it
# (the exact answer cost the CUDA SR GEMMs 6-8%). The formats warn that the
# carrier cannot place their bottom; this pins the result they get.
SUBNORMAL_FLOOR = {
    torch.float32: ([0, 8, 1, 146, None, 1, SAT.value], -145, -149),
    torch.float64: ([3, 4, 1, 966, None, 1, SAT.value], -1070, -1074),
}


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["binary32", "binary64"])
@pytest.mark.parametrize("mode", STOCHASTIC, ids=lambda m: m.name)
@pytest.mark.parametrize("n", [1, 2, 8, 12])
def test_a_carrier_subnormal_under_a_subnormal_floor(n, mode, dtype, device):
    fields, c, tiny = SUBNORMAL_FLOOR[dtype]
    fields = [n if f is None else f for f in fields]
    sh = MAN[dtype] + c - (tiny + MAN[dtype]) - n - 1  # 1 - BIAS = tiny + MAN
    step = 2 ** (c - tiny)  # 2^c in units of the carrier's smallest subnormal
    ks = [
        1,
        2,
        3,
        5,
        7,
        step // 2 - 1,
        step // 2,
        step // 2 + 1,
        step - 1,
        step,
        step + 1,
        3 * step,
        1000,
    ]
    x = torch.tensor([k * 2.0**tiny for k in ks], dtype=dtype)
    x = torch.cat([x, -x])
    man = fields[0]
    draws = (torch.arange(2**n, dtype=torch.int64) << (MAN[dtype] - man - n)).repeat(x.numel())
    y = CAST(x.repeat_interleave(2**n).to(device), draws.to(device), 1, fields, mode.value).cpu()
    y = y.view(x.numel(), 2**n)
    v = 2.0**c
    for i in range(x.numel()):
        xi = Fraction(x[i].item())
        row = y[i]
        assert bool(((row == 0) | (row.abs() == v)).all()), (xi, row.unique())
        assert bool((row[row != 0].sign() == (1 if xi > 0 else -1)).all())
        up = int((row != 0).sum())
        if sh < 0 or abs(xi) >= Fraction(v):
            want = 2**n
        else:
            want = _expected_up(xi, Fraction(0), Fraction(v), n, mode)
        assert up == want, (float(xi), sh, up, want)


def test_n0_is_rz_rna_and_half_toward_zero():
    """With no random bits, A is RZ, B is round-half-away and C rounds half
    toward zero (P3109's definitions at N = 0)."""
    fmt = binaryK(8, 4, 7, S, 0)
    x = torch.tensor([1.0, 1.0625, 1.125 + 0.0625, 1.1, -1.0625, 1.0625 + 2**-20])
    zero = torch.zeros_like(x, dtype=torch.int64)
    a = CAST(x, zero, fmt.family, fmt.fields, R.SR.value)
    b = CAST(x, zero, fmt.family, fmt.fields, R.SR_B.value)
    c = CAST(x, zero, fmt.family, fmt.fields, R.SR_C.value)
    assert torch.equal(a, fmt.quantize(x, R.RZ))
    assert torch.equal(b, fmt.quantize(x, R.RNA))
    lo, hi = fmt.quantize(x.abs(), R.RZ), fmt.quantize(x.abs(), R.RU)
    half_toward_zero = torch.where(x.abs() - lo > hi - x.abs(), hi, lo) * x.sign()
    assert torch.equal(c, half_toward_zero)


def test_at_the_bound_b_and_c_are_a():
    """Where no bit of the input lies below the draw's field (man + N = 23),
    B and C read nothing more than A does."""
    fmt = binaryK(16, 9, 15, S, 15)  # 8 mantissa bits + 15 = 23
    x = (torch.rand(4096) + 0.5) * 4 * torch.where(torch.rand(4096) < 0.5, -1.0, 1.0)
    draws = torch.randint(0, 2**31, (4096,), dtype=torch.int64)
    a = CAST(x, draws, fmt.family, fmt.fields, R.SR.value)
    for mode in (R.SR_B, R.SR_C):
        assert torch.equal(CAST(x, draws, fmt.family, fmt.fields, mode.value), a)


@pytest.mark.parametrize("mode", STOCHASTIC, ids=lambda m: m.name)
def test_gfloat_agrees(mode):
    """gfloat's StochasticFastest, StochasticFast and Stochastic are P3109's
    A, B and C: its count of the draws that go up is ours, for binary8p3
    (subnormals, no saturation in reach)."""
    gfloat = pytest.importorskip("gfloat")
    from gfloat.formats import format_info_p3109

    rnd = {
        R.SR: gfloat.RoundMode.StochasticFastest,
        R.SR_B: gfloat.RoundMode.StochasticFast,
        R.SR_C: gfloat.RoundMode.Stochastic,
    }[mode]
    fi = format_info_p3109(8, 3)
    n = 4
    fmt = binaryK(8, 3, 16, S, n)
    xs = [k / 64 * 2**-15 for k in range(1, 128)] + [0.37, 1.3, 2.71, 7.9]
    x = torch.tensor(xs)
    up, stray, lo, hi = _counts(fmt, x, n, mode, torch.float32, "cpu")
    assert not stray.any()
    for i, v in enumerate(xs):
        g = sum(
            gfloat.round_float(fi, v, rnd, srbits=r, srnumbits=n) == hi[i].item()
            for r in range(2**n)
        )
        assert int(up[i]) == (g if hi[i] != lo[i] else 0), (v, int(up[i]), g)


# --- every stochastic op takes the new modes ----------------------------------

BK = BinaryK(8, 4, prng_bits=4)
BK_ACC = BinaryK(16, 11, prng_bits=4)
SFP = SuperFP(3, 4, 2, 7, prng_bits=4)


def _paths():
    """One call of each kind of stochastic op, under a mode, on a device."""

    def quant(mode):
        x = torch.randn(512, generator=torch.Generator().manual_seed(0))
        return lambda d: [
            binaryK_quantize(x.to(d), 8, 4, prng_bits=4, rounding_mode=mode),
            superfp_quantize(x.to(d), 3, 4, 2, 7, prng_bits=4, rounding_mode=mode),
            fixedpoint_quantize(x.to(d), 8, 5, prng_bits=4, rounding_mode=mode),
        ]

    def gemm(mode):
        g = torch.Generator().manual_seed(1)
        a, b = torch.randn(16, 32, generator=g), torch.randn(32, 8, generator=g)
        pal = Palette([BinaryK(8, 4, prng_bits=4), BinaryK(8, 3, prng_bits=4)])
        prec_idx = (torch.arange(16 * 8) % 2).reshape(16, 8)
        calls = [
            (SplitMac(BK, BK_ACC, rounding=mode), None),
            (FusedMac(BK_ACC, rounding=mode), None),
            (SplitMac(SFP, SFP, rounding=mode), None),
            (FusedMac(SFP, rounding=mode), None),
            (SplitMac(BK, SFP, rounding=mode), None),
            (SplitMac(BK, BK_ACC, rounding=mode, accumulate_algorithm=A.KAHAN), None),
            (SplitMac(pal, BK_ACC, rounding=mode), prec_idx),
        ]

        def run(d):
            out = [
                qmatmul(a.to(d), b.to(d), m, prec_idx=None if p is None else p.to(d))
                for m, p in calls
            ]
            if d != "mps":  # the odd carrier step has no MPS kernel
                odd = SplitMac(BK, BK_ACC, rounding=mode, carrier_rounding=R.RO)
                out.append(qmatmul(a.to(d), b.to(d), odd))
            return out

        return run

    def blocks(mode):
        g = torch.Generator().manual_seed(2)
        x, w = torch.randn(4, 64, generator=g), torch.randn(64, 3, generator=g)
        return lambda d: [
            block_quantize(x.to(d), MXFP8_E4M3, rounding=mode),
            block_unpack(block_pack(x.to(d), MXFP8_E4M3, rounding=mode)),
            block_matmul(
                block_pack(x.to(d), MXFP8_E4M3, axis=-1),
                block_pack(w.to(d), MXFP8_E4M3, axis=-2),
                acc=BinaryK(16, 11, prng_bits=4),
                rounding=mode,
            ),
        ]

    def conv(mode):
        def run(d):
            torch.manual_seed(7)
            layer = QConv2d(
                3, 4, 3, padding=1, formats=conv_formats(SplitMac(BK, BK_ACC, rounding=mode))
            ).to(d)
            x = (
                torch.randn(2, 3, 6, 6, generator=torch.Generator().manual_seed(3))
                .to(d)
                .requires_grad_()
            )
            torch.manual_seed(5)
            out = layer(x)
            out.backward(torch.ones_like(out))
            return [out.detach(), x.grad, layer.weight.grad]

        return run

    return {"quantizers": quant, "gemms": gemm, "blocks": blocks, "conv": conv}


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", [R.SR_B, R.SR_C], ids=lambda m: m.name)
@pytest.mark.parametrize("path", ["quantizers", "gemms", "blocks", "conv"])
def test_every_path_takes_the_mode(path, mode, device):
    """Each path runs under SR_B and SR_C, draws as SR does (the same seed
    gives the same result again), and differs from SR somewhere. CUDA draws
    from its own generator, so its results are not the CPU's; MPS draws from
    the CPU's and tests/test_mps.py holds it to the CPU word for word. Which
    way each draw rounds is test_every_draw_counts_as_p3109_says's, on both."""
    run = _paths()[path](mode)
    torch.manual_seed(5)
    first = [t.cpu() for t in run(device)]
    torch.manual_seed(5)
    again = [t.cpu() for t in run(device)]
    for w, g in zip(first, again, strict=True):
        assert torch.equal(w, g)
    torch.manual_seed(5)
    a = [t.cpu() for t in _paths()[path](R.SR)(device)]
    assert any(not torch.equal(x, y) for x, y in zip(a, first, strict=True))


@pytest.mark.parametrize("mode", STOCHASTIC, ids=lambda m: m.name)
def test_unbiased_on_average(mode):
    """A statistical check through the public quantizer, with the generator's
    own draws: 1 + 0.4 * step goes up with P3109's probability at N = 2, where
    X = 1.6: floor(X) = 1 of 4 draws (A), floor(X + 1/2) = 2 (B), RNITE(X) = 2
    (C)."""
    step = 2**-3  # binary8p4 at 1
    x = torch.full((1 << 16,), 1 + 0.4 * step)
    y = binaryK_quantize(x, 8, 4, prng_bits=2, rounding_mode=mode)
    p = (y > 1).double().mean().item()
    want = {R.SR: 1 / 4, R.SR_B: 2 / 4, R.SR_C: 2 / 4}[mode]
    assert abs(p - want) < 5 * math.sqrt(want * (1 - want) / x.numel())


# --- the Python layer and the op boundary --------------------------------------


def test_the_members():
    assert R.SR_A is R.SR
    assert [m for m in R if m.is_stochastic] == STOCHASTIC
    assert not any(m.is_stochastic for m in (R.RNE, R.RNA, R.RU, R.RD, R.RZ, R.RO))


@pytest.mark.parametrize("mode", [R.SR, R.SR_B, R.SR_C], ids=lambda m: m.name)
def test_raw_ops_refuse_prng_bits_past_a_byte(mode):
    """Under a stochastic mode the variant rides beside N in prng_bits, so the
    op boundary refuses an N that would read as one."""
    x = torch.randn(8)
    with pytest.raises(RuntimeError, match="prng_bits must be in"):
        torch.ops.mptorch.binaryK_quant(x, 8, 4, 7, 300, True, mode.value, 0, 0)
    with pytest.raises(RuntimeError, match="prng_bits must be in"):
        torch.ops.mptorch.binaryK_quant(x.to("meta"), 8, 4, 7, 300, True, mode.value, 0, 0)


def test_odd_carrier_bounds_per_variant():
    """The odd carrier step (``carrier_rounding=RO``) holds A for
    P + prng_bits <= 23 in binary32 and B and C for P + prng_bits <= 22."""
    a, b = torch.randn(4, 8), torch.randn(8, 3)
    at_a = FusedMac(BinaryK(16, 11, prng_bits=12), rounding=R.SR, carrier_rounding=R.RO)
    qmatmul(a, b, at_a)
    for mode in (R.SR_B, R.SR_C):
        qmatmul(a, b, FusedMac(BinaryK(16, 11, prng_bits=11), rounding=mode, carrier_rounding=R.RO))
        with pytest.raises(ValueError, match="P \\+ prng_bits"):
            qmatmul(
                a, b, FusedMac(BinaryK(16, 11, prng_bits=12), rounding=mode, carrier_rounding=R.RO)
            )


@pytest.mark.parametrize("mode", [R.SR_B, R.SR_C], ids=lambda m: m.name)
def test_recompute_replays_the_new_modes(mode):
    """A layer that re-quantizes its operands in backward replays the draws of
    the new modes too (the generator states captured before its forward call),
    so its gradients are the copies'."""
    torch.manual_seed(0)
    f = binaryK_gemm_formats(
        8, 4, mul_prng_bits=4, acc_K=16, acc_P=11, acc_prng_bits=4, rounding_mode=mode
    )
    f.input_quant = Quant(BinaryK(8, 4, prng_bits=4), mode)
    f.weight_quant = Quant(BinaryK(8, 4, prng_bits=4), mode)
    grads = []
    for recompute in (True, False):
        f.recompute_input = f.recompute_weight = recompute
        torch.manual_seed(0)
        layer = QLinear(16, 8, formats=f)
        x = torch.randn(4, 16, requires_grad=True)
        torch.manual_seed(9)
        layer(x).square().sum().backward()
        assert x.grad is not None and layer.weight.grad is not None
        grads.append((x.grad.clone(), layer.weight.grad.clone()))
    for r, k in zip(*grads, strict=True):
        assert torch.equal(r, k)
