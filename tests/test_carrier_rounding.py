"""What the carrier rounds in a mac step.

A NAIVE GEMM step rounds twice: the carrier's add or fused multiply-add
rounds to nearest (binary32 for float32 operands, binary64 for float64 ones),
and the format's cast rounds the result again. Under the directed modes that
returns the wrong neighbour whenever the exact sum lies within the carrier's
last place of a format value (in binary32, an exact sum of 1 + 2^-30 rounded
up into binary16p8 gives 1 instead of 1.0078125). ``carrier_rounding=RO``
runs the ``custom_matmul_*_odd`` ops instead, whose carrier steps round to
odd (``csrc/common/carrier_odd.h``), so that the cast rounds the exact value.

The witnesses are one two-term dot product whose exact sum is ``1 ± 2^-g``,
with ``g`` past the carrier's precision (30 in binary32, 61 in binary64) or
inside it (30 in binary64), and whose correct rounding to binary16p8 is
computed from a ``Fraction``: the default ops are pinned at their double
rounding, the odd ops at the correct value. The primitives are held to an
exact ``Fraction`` reference of round-to-odd through ``_carrier_odd``, a
test-only elementwise op, because the casts after them erase zero signs, the
largest finite value and subnormal parity; and the odd GEMMs to a reference
that rounds each step's exact value with the binary64 quantizer, in every
deterministic mode, the CPU and CUDA kernels to each other word for word.
"""

import math
import random
import struct
import zlib
from fractions import Fraction
from typing import Any

import pytest
import torch

from mptorch import AccumulateAlgorithm, BinaryK, RoundMode, SaturationMode, SuperFP
from mptorch.quant import (
    FusedMac,
    Palette,
    QConv2d,
    QLinear,
    QMatmul,
    SplitMac,
    binaryK_gemm_formats_fma,
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_fma_mixed,
    binaryK_matmul_mixed,
    binaryK_quantize,
    binaryK_superfp_matmul,
    conv_formats,
    matmul_formats,
    qmm,
    superfp_matmul,
    superfp_matmul_fma,
    superfp_quantize,
)
from mptorch.quant.mac import spec_for_mac

from .markers import (
    available_devices,
    float64_devices,
    requires_cuda,
    requires_mps,
    requires_odd,
)

# Every test here runs an odd op, or the default op beside one; a build with
# MPTORCH_NO_ODD=1 has none of their kernels (test_fake_kernels.py holds its
# ops to raising, naming the flag).
pytestmark = requires_odd

R = RoundMode
DETERMINISTIC = (R.RNE, R.RNA, R.RU, R.RD, R.RZ, R.RO)
DIRECTED = (R.RU, R.RD, R.RZ, R.RO)
# binary16p8: eight bits of precision, so the neighbours of 1 are 1 - 2^-8 and
# 1 + 2^-7.
K16, P8 = 16, 8
BELOW, ONE, ABOVE = 1 - 2.0**-8, 1.0, 1 + 2.0**-7

odd_op = torch.ops.mptorch._carrier_odd.default
FMA, ADD, MUL = 0, 1, 2


def _correct(exact: Fraction, mode: RoundMode) -> float:
    """binary16p8's rounding of an exact value within one step of 1."""
    if exact == 1:
        return ONE
    up = exact > 1
    if mode in (R.RNE, R.RNA):
        return ONE  # a gap of 2^-30 or less is far from either midpoint
    if mode is R.RU:
        return ABOVE if up else ONE
    if mode in (R.RD, R.RZ):
        return ONE if up else BELOW
    return ABOVE if up else BELOW  # RO: the odd neighbour, never 1 (even)


def _witness(g: int, sign: int, dtype: torch.dtype, device: str):
    """``[1, sign·2^-h] @ [1, 2^-(g-h)]``: products 1 and ±2^-g, each a
    binary16p8 value, so only the sum is inexact."""
    h = g // 2
    a = torch.tensor([[1.0, sign * 2.0**-h]], dtype=dtype, device=device)
    b = torch.tensor([[1.0], [2.0 ** -(g - h)]], dtype=dtype, device=device)
    return a, b, 1 + sign * Fraction(1, 2**g)


# superfp16 with eight bits of precision in its normal binades
SUPERFP: dict[str, Any] = dict(man_bits=7, exp_bits=8, normal_binades=200, bias=127)


def _run(kind: str, a, b, mode: RoundMode, carrier_rounding: RoundMode = R.RNE) -> float:
    kw: dict[str, Any] = dict(rounding_mode=mode, carrier_rounding=carrier_rounding)
    if kind == "binaryK split":
        return binaryK_matmul(a, b, mul_K=K16, mul_P=P8, **kw).item()
    if kind == "binaryK fused":
        return binaryK_matmul_fma(a, b, fma_K=K16, fma_P=P8, **kw).item()
    if kind == "superfp split":
        mul: dict[str, Any] = {f"mul_{k}": v for k, v in SUPERFP.items()}
        return superfp_matmul(a, b, **mul, **kw).item()
    fma: dict[str, Any] = {f"fma_{k}": v for k, v in SUPERFP.items()}
    return superfp_matmul_fma(a, b, **fma, **kw).item()


KINDS = ("binaryK split", "binaryK fused", "superfp split", "superfp fused")
# (gap, dtype, whether the carrier's nearest sum keeps it)
WITNESSES = [
    pytest.param(30, torch.float32, False, id="b32-gap30"),
    pytest.param(30, torch.float64, True, id="b64-gap30"),
    pytest.param(61, torch.float64, False, id="b64-gap61"),
]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("gap, dtype, kept", WITNESSES)
@pytest.mark.parametrize("sign", [1, -1], ids=["above", "below"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", [R.RNE, *DIRECTED], ids=lambda m: m.name)
def test_the_default_step_rounds_in_the_carrier_first(device, gap, dtype, kept, sign, kind, mode):
    """The default ops round the sum to the carrier's nearest value first:
    where the carrier keeps the gap the result is correct, and where it does
    not the format sees exactly 1, so every mode returns 1."""
    a, b, exact = _witness(gap, sign, dtype, device)
    got = _run(kind, a, b, mode)
    assert got == (_correct(exact, mode) if kept else ONE)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("gap, dtype, kept", WITNESSES)
@pytest.mark.parametrize("sign", [1, -1], ids=["above", "below"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", [R.RNE, *DIRECTED], ids=lambda m: m.name)
def test_the_odd_step_rounds_the_exact_sum(device, gap, dtype, kept, sign, kind, mode):
    """With the carrier step rounded to odd, every witness is correct in both
    carriers, whether or not the carrier's nearest sum would have kept it."""
    a, b, exact = _witness(gap, sign, dtype, device)
    assert _run(kind, a, b, mode, carrier_rounding=R.RO) == _correct(exact, mode)


# --- the primitives -----------------------------------------------------------
#
# RO_p(x) is RZ onto the carrier's grid (p bits, gradual underflow below
# 2^emin, the largest finite value Omega on overflow) with the last bit set when
# x is inexact.

CARRIERS = {
    torch.float32: (24, -126, 127, "<f", "<I"),
    torch.float64: (53, -1022, 1023, "<d", "<Q"),
}


def _ro(x: Fraction, dtype: torch.dtype) -> Fraction:
    p, emin, emax, _, _ = CARRIERS[dtype]
    if x == 0:
        return Fraction(0)
    sign, ax = (-1 if x < 0 else 1), abs(x)
    omega = (2**p - 1) * Fraction(2) ** (emax - p + 1)
    if ax > omega:
        return sign * omega
    e = ax.numerator.bit_length() - ax.denominator.bit_length()
    if Fraction(2) ** e > ax:
        e -= 1
    q = Fraction(2) ** max(e - p + 1, emin - p + 1)
    k = math.floor(ax / q)
    if k * q != ax:
        k |= 1
    return sign * k * q


def _word(x: float, dtype: torch.dtype) -> int:
    _, _, _, f, w = CARRIERS[dtype]
    return struct.unpack(w, struct.pack(f, x))[0]


def _value(word: int, dtype: torch.dtype) -> float:
    _, _, _, f, w = CARRIERS[dtype]
    return struct.unpack(f, struct.pack(w, word))[0]


def _expected(a: float, b: float, c: float, op: int, dtype: torch.dtype) -> int | None:
    """The word RO gives for finite operands, or None where IEEE's special
    values decide (an infinity or a NaN operand)."""
    if not all(math.isfinite(v) for v in (a, b, c)):
        return None
    fa, fb, fc = Fraction(a), Fraction(b), Fraction(c)
    x = fa * fb + fc if op == FMA else fa + fb if op == ADD else fa * fb
    if x == 0:
        # +0 unless every term is a -0 (IEEE's sign, which is RZ's)
        if op == MUL:
            negative = math.copysign(1, a) * math.copysign(1, b) < 0
        else:
            first = math.copysign(1, a) * math.copysign(1, b) if op == FMA else math.copysign(1, a)
            second = math.copysign(1, c) if op == FMA else math.copysign(1, b)
            negative = first < 0 and second < 0 and (op != FMA or a * b == 0)
        return _word(-0.0 if negative else 0.0, dtype)
    return _word(float(_ro(x, dtype)), dtype)


def _call(a, b, c, op, dtype, device):
    t = lambda v: torch.tensor(v, dtype=dtype, device=device)  # noqa: E731
    out = odd_op(t(a), t(b), t(c), op).cpu().tolist()
    return [_word(v, dtype) for v in out]


OMEGA32, OMEGA64 = 3.4028234663852886e38, 1.7976931348623157e308
# (a, b, c, op, dtype, the word RO gives): the vectors of the odd step's
# design review (2026-11-10), each group under what it catches.
VECTORS = [
    # double rounding; the binade fix
    (1, 1, 2**-30, FMA, torch.float32, 0x3F800001),
    (1, 1, -(2**-30), FMA, torch.float32, 0x3F7FFFFF),
    # a spurious sticky bit
    (1, 1, 2**-22, FMA, torch.float32, 0x3F800002),
    # RO above x
    (1, 1 + 2**-23, -(2**-40), FMA, torch.float32, 0x3F800001),
    # a grid point, the residual toward zero
    (1, 1 + 2**-22, -(2**-40), FMA, torch.float32, 0x3F800001),
    # an exact midpoint of the carrier, 1 + 2^-24
    (1 + 2**-12, 1 + 2**-12, -(2**-11), FMA, torch.float32, 0x3F800001),
    # Fast2Sum in either fixed order
    (2**-50, 2**-50, 1, FMA, torch.float32, 0x3F800001),
    (1, 1, 2**-100, FMA, torch.float32, 0x3F800001),
    (2**-50, -(2**-50), 1, FMA, torch.float32, 0x3F7FFFFF),
    # never flushed to zero
    (2**-75, 2**-75, 0, FMA, torch.float32, 0x00000001),
    (2**-75, 2**-75, -(2**-149), FMA, torch.float32, 0x80000001),
    # subnormal parity
    (7 * 2**-75, 2**-75, 0, FMA, torch.float32, 0x00000003),
    (9 * 2**-75, 2**-75, 0, FMA, torch.float32, 0x00000005),
    # the normal/subnormal boundary
    (2**-63 * (1 - 2**-24), 2**-63, 0, FMA, torch.float32, 0x007FFFFF),
    # overflow to the largest finite value, not infinity
    (2**64, 2**64, 0, FMA, torch.float32, 0x7F7FFFFF),
    (OMEGA32, 1, 2**103, FMA, torch.float32, 0x7F7FFFFF),
    # a product out of range, the result in it
    (2**64, 2**64, -OMEGA32, FMA, torch.float32, 0x73800000),
    # gaps past p
    (1, 2**100, 2**-100, FMA, torch.float32, 0x71800001),
    (1, -(2**-100), 2**100, FMA, torch.float32, 0x717FFFFF),
    # zero signs
    (1, 1, -1, FMA, torch.float32, 0x00000000),
    (-0.0, 1, -0.0, FMA, torch.float32, 0x80000000),
    (0.0, 1, -0.0, FMA, torch.float32, 0x00000000),
    (-1, 0, -0.0, FMA, torch.float32, 0x80000000),
    # rounding a * b to nearest first would give NaN
    (2**64, 2**64, -math.inf, FMA, torch.float32, 0xFF800000),
    # binary64: double rounding
    (1, 1, 2**-60, FMA, torch.float64, 0x3FF0000000000001),
    (1, 1, -(2**-60), FMA, torch.float64, 0x3FEFFFFFFFFFFFFF),
    # x is TwoProd's low part exactly
    (1 + 2**-52, 1 + 2**-52, -(1 + 2**-51), FMA, torch.float64, 0x3970000000000000),
    # TwoProd's underflow edge
    (2**-484 * (1 + 2**-52), 2**-484 * (1 + 2**-52), 0, FMA, torch.float64, 0x0370000000000003),
    (2**-486 * (1 + 2**-52), 2**-486 * (1 + 2**-52), 0, FMA, torch.float64, 0x0330000000000003),
    # below the subnormals
    (2**-540, 2**-540, 0, FMA, torch.float64, 0x0000000000000001),
    (2**-540, 2**-540, -5e-324, FMA, torch.float64, 0x8000000000000001),
    # c dominates
    (2**-1000, 1, 1, FMA, torch.float64, 0x3FF0000000000001),
    (-(2**-1000), 1, 1, FMA, torch.float64, 0x3FEFFFFFFFFFFFFF),
    # the scaled-down path, and overflow
    (2**512, 2**512, 0, FMA, torch.float64, 0x7FEFFFFFFFFFFFFF),
    (2**512, 2**512, -OMEGA64, FMA, torch.float64, 0x7CA0000000000000),
    (2**600, 2**600, -OMEGA64, FMA, torch.float64, 0x7FEFFFFFFFFFFFFF),
    (OMEGA64, 1, 2**970, FMA, torch.float64, 0x7FEFFFFFFFFFFFFF),
    # rounding a * b to nearest first would give NaN
    (2**600, 2**600, -math.inf, FMA, torch.float64, 0xFFF0000000000000),
    # 2Sum's spurious overflow
    (OMEGA64, -1.5 * 2**971, 0, ADD, torch.float64, 0x7FEFFFFFFFFFFFFD),
    # the zero-sign fix
    (-0.0, 5, -0.0, FMA, torch.float64, 0x8000000000000000),
    (0.0, -0.0, -0.0, FMA, torch.float64, 0x8000000000000000),
]


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize(
    "a, b, c, op, dtype, want", VECTORS, ids=[f"v{i}" for i in range(len(VECTORS))]
)
def test_the_primitives_on_the_review_vectors(device, a, b, c, op, dtype, want):
    assert _call([a], [b], [c], op, dtype, device) == [want]
    expected = _expected(float(a), float(b), float(c), op, dtype)
    assert expected is None or expected == want  # the reference agrees with the table


def _operands(regime: str, n: int, dtype: torch.dtype, rng: random.Random):
    """``n`` operand triples of one regime, as Python floats the dtype holds."""
    wide = dtype is torch.float64
    p = 53 if wide else 24

    def value(lo: int, hi: int) -> float:
        m = rng.getrandbits(p) | (1 << (p - 1))
        x = math.ldexp(m, rng.randint(lo, hi) - p + 1) * rng.choice((1, -1))
        return x if wide else struct.unpack("<f", struct.pack("<f", x))[0]

    out = []
    for _ in range(n):
        if regime == "normal":
            a, b, c = value(-30, 30), value(-30, 30), value(-90, 60)
        elif regime == "cancel":  # c within a few carrier ulps of -a*b
            a, b = value(-20, 20), value(-20, 20)
            ab = a * b if wide else struct.unpack("<f", struct.pack("<f", a * b))[0]
            c = -ab + rng.choice((0, 1, -1, 3)) * math.ulp(ab) * (1 if wide else 2**29)
            c = c if wide else struct.unpack("<f", struct.pack("<f", c))[0]
        elif regime == "tiny":  # products at and below the carrier's subnormals
            lo, hi = (-560, -470) if wide else (-80, -60)
            a, b = value(lo, hi), value(lo, hi)
            c = value(*((-1074, -1000) if wide else (-149, -120))) * rng.choice((0, 1))
        else:  # "huge": products and sums at the top of the range
            lo, hi = (500, 512) if wide else (60, 64)
            a, b = value(lo, hi), value(lo, hi)
            c = value(*((1000, 1023) if wide else (120, 127)))
        out.append((a, b, c))
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["b32", "b64"])
@pytest.mark.parametrize("regime", ["normal", "cancel", "tiny", "huge"])
@pytest.mark.parametrize("op", [FMA, ADD, MUL], ids=["fma", "add", "mul"])
def test_the_host_primitives_match_the_exact_reference(dtype, regime, op):
    rng = random.Random(f"{dtype}{regime}{op}")
    triples = _operands(regime, 400, dtype, rng)
    a, b, c = (list(t) for t in zip(*triples, strict=True))
    got = _call(a, b, c, op, dtype, "cpu")
    for i, (x, y, z) in enumerate(triples):
        want = _expected(x, y, z, op, dtype)
        assert got[i] == want, (x, y, z, hex(got[i]), want and hex(want))


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["b32", "b64"])
@pytest.mark.parametrize("regime", ["normal", "cancel", "tiny", "huge"])
@pytest.mark.parametrize("op", [FMA, ADD, MUL], ids=["fma", "add", "mul"])
def test_the_cpu_and_cuda_primitives_agree_word_for_word(dtype, regime, op):
    rng = random.Random(f"cuda{dtype}{regime}{op}")
    triples = _operands(regime, 4000, dtype, rng)
    specials = [math.inf, -math.inf, 0.0, -0.0, 1.0, -1.0]
    triples += [(x, y, z) for x in specials for y in specials for z in specials]
    a, b, c = (list(t) for t in zip(*triples, strict=True))
    cpu, cuda = _call(a, b, c, op, dtype, "cpu"), _call(a, b, c, op, dtype, "cuda")
    nan = [math.isnan(_value(w, dtype)) for w in cpu]
    assert [w for w, n in zip(cpu, nan, strict=True) if not n] == [
        w for w, n in zip(cuda, nan, strict=True) if not n
    ]
    assert nan == [math.isnan(_value(w, dtype)) for w in cuda]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["b32", "b64"])
@pytest.mark.parametrize("op", [FMA, ADD, MUL], ids=["fma", "add", "mul"])
def test_a_nan_passes_through_the_host_primitives_whole(dtype, op):
    """On the host a NaN takes today's expression, payload included."""
    payload = 0x7FC12340 if dtype is torch.float32 else 0x7FF8123456789AB0
    nan = _value(payload, dtype)
    got = _call([nan], [1.0], [1.0], op, dtype, "cpu")
    assert got == [payload]


@requires_mps
def test_the_odd_ops_have_no_mps_kernel():
    a = torch.ones(2, 2, device="mps")
    with pytest.raises(RuntimeError, match="no MPS kernel"):
        odd_op(a, a, a, FMA)
    with pytest.raises(RuntimeError, match="no MPS kernel"):
        binaryK_matmul_fma(a, a, fma_K=8, fma_P=4, carrier_rounding=R.RO)
    idx = torch.zeros(2, 2, dtype=torch.int32, device="mps")
    with pytest.raises(RuntimeError, match="no MPS kernel"):
        binaryK_matmul_mixed(a, a, idx, mul_K=[8, 8], mul_P=[4, 3], carrier_rounding=R.RO)
    with pytest.raises(RuntimeError, match="no MPS kernel"):
        binaryK_superfp_matmul(
            a, a, mul_K=8, mul_P=4, acc_man_bits=3, acc_exp_bits=4, acc_normal_binades=8,
            acc_bias=7, carrier_rounding=R.RO,
        )  # fmt: skip
    layer = QConv2d(1, 1, 1, formats=conv_formats(FusedMac(BinaryK(8, 4), carrier_rounding=R.RO)))
    with pytest.raises(RuntimeError, match="no MPS kernel"):
        layer.to("mps")(torch.ones(1, 1, 2, 2, device="mps"))


# --- the GEMMs ----------------------------------------------------------------

# Formats inside both carriers' bounds for every deterministic mode
# (P <= 22), wide enough that the carrier's last bits matter, and with a range
# binary32 holds.
BK = BinaryK(28, 22)
SF = SuperFP(man_bits=19, exp_bits=6, normal_binades=40, bias=31)


def _quantize64(x: torch.Tensor, fmt, mode: RoundMode) -> torch.Tensor:
    if isinstance(fmt, BinaryK):
        return binaryK_quantize(
            x,
            fmt.K,
            fmt.P,
            fmt.bias,
            rounding_mode=mode,
            saturation_mode=fmt.saturation,
            subnormals_mode=fmt.subnormals,
        )
    return superfp_quantize(
        x,
        fmt.man_bits,
        fmt.exp_bits,
        fmt.normal_binades,
        fmt.bias,
        rounding_mode=mode,
        saturation_mode=fmt.saturation,
    )


def _ro53(values: list[Fraction]) -> torch.Tensor:
    return torch.tensor([float(_ro(v, torch.float64)) for v in values], dtype=torch.float64)


def _reference(
    a: torch.Tensor, b: torch.Tensor, mac: str, fmt, mode: RoundMode, acc=None
) -> torch.Tensor:
    """Each step's exact value, rounded to odd at 53 bits and then to the
    format by the binary64 quantizer: the format's rounding of the exact
    value, for a format inside binary64's bound. A split mac's sums are
    rounded to ``acc`` when it is given (a cross-family mac), else to
    ``fmt``."""
    A = [[Fraction(v) for v in row] for row in a.double().tolist()]
    B = [[Fraction(v) for v in row] for row in b.double().tolist()]
    M, K, N = len(A), len(B), len(B[0])
    s = [Fraction(0)] * (M * N)
    for k in range(K):
        prods = [A[i][k] * B[k][j] for i in range(M) for j in range(N)]
        if mac == "fused":
            q = _quantize64(_ro53([p + t for p, t in zip(prods, s, strict=True)]), fmt, mode)
        else:
            qp = _quantize64(_ro53(prods), fmt, mode).tolist()
            sums = [t + Fraction(v) for t, v in zip(s, qp, strict=True)]
            q = _quantize64(_ro53(sums), fmt if acc is None else acc, mode)
        s = [Fraction(v) for v in q.tolist()]
    return torch.tensor([float(v) for v in s], dtype=torch.float64).reshape(M, N)


def _odd_gemm(a, b, mac: str, fmt, mode: RoundMode) -> torch.Tensor:
    if mac == "fused":
        m: FusedMac | SplitMac = FusedMac(fmt, rounding=mode, carrier_rounding=R.RO)
    else:
        m = SplitMac(fmt, fmt, rounding=mode, carrier_rounding=R.RO)
    return qmm(a, b, m)


def _gemm_operands(dtype: torch.dtype, seed: int):
    """Small operands built to make the carrier's last bits matter: values of
    widely spread exponents, products that nearly cancel in pairs, and some
    operands the format holds exactly."""
    g = torch.Generator().manual_seed(seed)
    M, K, N = 6, 24, 5
    exps = torch.randint(-12, 13, (M, K), generator=g)
    a = torch.randn(M, K, generator=g, dtype=torch.float64) * torch.pow(2.0, exps.double())
    b = torch.randn(K, N, generator=g, dtype=torch.float64)
    for k in range(0, 8, 2):  # a[:, k] b[k] + a[:, k] b[k + 1] nearly cancels
        a[:, k + 1] = a[:, k]
        b[k + 1] = -b[k] * (1 + 2.0 ** -(10 + 3 * k))
    a, b = a.to(dtype), b.to(dtype)
    a[:3] = _quantize64(a[:3].double(), BK, R.RNE).to(dtype)
    return a, b


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["b32", "b64"])
@pytest.mark.parametrize("mac", ["fused", "split"])
@pytest.mark.parametrize("fmt", [BK, SF], ids=["binaryK", "superfp"])
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_the_odd_gemm_rounds_each_exact_step(device, dtype, mac, fmt, mode):
    a, b = _gemm_operands(dtype, seed=zlib.crc32(f"{mac}{fmt}{mode.name}".encode()))
    want = _reference(a, b, mac, fmt, mode).to(dtype)
    got = _odd_gemm(a.to(device), b.to(device), mac, fmt, mode).cpu()
    assert torch.equal(got, want)


# SR is not compared: the CPU and CUDA kernels draw from different generators.
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["b32", "b64"])
@pytest.mark.parametrize("mac", ["fused", "split", "split unrounded sum"])
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_the_cpu_and_cuda_odd_gemms_agree_word_for_word(dtype, mac, mode):
    torch.manual_seed(5)
    a = torch.randn(33, 70, dtype=dtype) * torch.pow(2.0, torch.randint(-8, 9, (33, 70)))
    b = torch.randn(70, 17, dtype=dtype)
    fmt = BinaryK(16, 11, prng_bits=6)
    if mac == "fused":
        m: FusedMac | SplitMac = FusedMac(fmt, rounding=mode, carrier_rounding=R.RO)
    else:
        acc = None if mac == "split unrounded sum" else fmt
        m = SplitMac(fmt, acc, rounding=mode, carrier_rounding=R.RO)
    cpu, cuda = qmm(a, b, m), qmm(a.cuda(), b.cuda(), m).cpu()
    assert torch.equal(
        cpu.view(torch.int32 if dtype is torch.float32 else torch.int64),
        cuda.view(torch.int32 if dtype is torch.float32 else torch.int64),
    )


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["b32", "b64"])
def test_an_unrounded_sum_stays_the_carriers_nearest_add(device, dtype):
    """``SplitMac(mul, None)``: only the product is rounded to odd, then
    rounded by the multiply format; the sum is the carrier's own add."""
    torch.manual_seed(2)
    a = torch.randn(4, 9, dtype=dtype, device=device)
    b = torch.randn(9, 3, dtype=dtype, device=device)
    fmt = BinaryK(12, 7)
    got = qmm(a, b, SplitMac(fmt, None, rounding=R.RU, carrier_rounding=R.RO))
    A, B = a.double().cpu().tolist(), b.double().cpu().tolist()
    exact = [
        Fraction(A[i][k]) * Fraction(B[k][j]) for i in range(4) for k in range(9) for j in range(3)
    ]
    q = _quantize64(_ro53(exact).reshape(4, 9, 3), fmt, R.RU).to(dtype)
    want = torch.zeros(4, 3, dtype=dtype)
    for k in range(9):
        want = want + q[:, k, :]  # the carrier's nearest add, in order
    assert torch.equal(got.cpu(), want)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["b32", "b64"])
@pytest.mark.parametrize("mac", ["fused", "split"])
def test_exact_cancellation_is_an_unsigned_zero(device, dtype, mac):
    a = torch.tensor([[1.5, -1.5]], dtype=dtype, device=device)
    b = torch.tensor([[1.0], [1.0]], dtype=dtype, device=device)
    for mode in DIRECTED:
        out = _odd_gemm(a, b, mac, BinaryK(8, 4), mode)
        assert out.item() == 0 and not math.copysign(1, out.item()) < 0


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("mac", ["fused", "split"])
def test_infinities_and_nans_pass_through(device, mac):
    a = torch.tensor([[math.inf, 1.0], [math.nan, 1.0]], device=device)
    b = torch.tensor([[1.0], [1.0]], device=device)
    out = _odd_gemm(a, b, mac, BinaryK(8, 4), R.RU).cpu()
    assert out[0, 0].item() == math.inf and math.isnan(out[1, 0].item())


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize(
    "saturation, default, odd",
    [
        (SaturationMode.SAT_FINITE, "max", "max"),
        (SaturationMode.SAT_NONE, "inf", "inf"),
        (SaturationMode.SAT_PROPAGATE, "inf", "max"),
    ],
    ids=lambda v: v.name if isinstance(v, SaturationMode) else v,
)
def test_a_finite_sum_past_the_carrier_saturates_as_a_finite_value(
    device, saturation, default, odd
):
    """The exact 2^127·1.5 + 2^127 is finite and past binary32's largest
    value: the nearest carrier step makes it infinity, which SAT_PROPAGATE
    keeps; the odd step makes it the largest finite value, which every
    saturating mode saturates, as P3109 says a finite value should."""
    a = torch.tensor([[2.0**127, 2.0**126]], device=device)
    b = torch.tensor([[1.5], [2.0]], device=device)
    fmt = BinaryK(16, 8, saturation=saturation)
    top = binaryK_quantize(  # the format's largest finite value
        torch.tensor([3.4e38]), 16, 8, rounding_mode=R.RZ, saturation_mode=saturation
    )
    words = {"max": top.item(), "inf": math.inf}
    plain = qmm(a, b, FusedMac(fmt)).item()
    assert plain == words[default]
    assert qmm(a, b, FusedMac(fmt, carrier_rounding=R.RO)).item() == words[odd]


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype, g", [(torch.float32, 30), (torch.float64, 61)], ids=["b32", "b64"])
def test_stochastic_rounding_sees_the_bits_the_nearest_carrier_drops(device, dtype, g):
    """The exact sum 1 + 2^-7 - 2^-g is just below binary16p8's 1 + 2^-7. The
    nearest carrier step rounds it up onto that value, which SR then always
    keeps; the odd step keeps it below, and SR with 8 random bits rounds it
    down with probability 2^-8 (P + prng_bits = 16, inside the bound)."""
    h = g // 2
    fmt = BinaryK(16, 8, prng_bits=8)
    a = torch.tensor([[1 + 2.0**-7, -(2.0**-h)]], dtype=dtype, device=device).expand(256, 2)
    b = torch.tensor([[1.0], [2.0 ** -(g - h)]], dtype=dtype, device=device).expand(2, 256)
    plain = qmm(a, b, FusedMac(fmt, rounding=R.SR))
    odd = qmm(a, b, FusedMac(fmt, rounding=R.SR, carrier_rounding=R.RO))
    assert (plain == 1 + 2.0**-7).all()
    down = int((odd == 1.0).sum())
    assert set(odd.unique().tolist()) <= {1.0, 1 + 2.0**-7}
    assert 150 < down < 400  # 65536 draws at 2^-8: mean 256, sd 16


@pytest.mark.parametrize(
    "dtype, P, mode, raises",
    [
        (torch.float32, 22, R.RNE, False),
        (torch.float32, 23, R.RNE, True),
        (torch.float32, 23, R.RU, False),
        (torch.float32, 24, R.RZ, True),
        (torch.float64, 24, R.RNE, False),
    ],
    ids=lambda v: v.name if isinstance(v, RoundMode) else str(v),
)
def test_a_format_outside_the_bound_raises(dtype, P, mode, raises):
    a = torch.ones(2, 2, dtype=dtype)
    call = lambda: binaryK_matmul_fma(  # noqa: E731
        a, a, fma_K=P + 6, fma_P=P, rounding_mode=mode, carrier_rounding=R.RO
    )
    if raises:
        with pytest.raises(ValueError, match="carrier=torch.float64"):
            call()
    else:
        call()


def test_sr_holds_precision_and_random_bits_together():
    a = torch.ones(2, 2)
    fma: dict[str, Any] = dict(fma_K=16, fma_P=8, rounding_mode=R.SR, carrier_rounding=R.RO)
    binaryK_matmul_fma(a, a, fma_prng_bits=15, **fma)  # P + r = 23
    with pytest.raises(ValueError, match="P \\+ prng_bits is 24"):
        binaryK_matmul_fma(a, a, fma_prng_bits=16, **fma)
    # binary64 holds the 24 the first call could not: and a split mac's
    # accumulate format is held as well as its multiply format.
    binaryK_matmul_fma(a.double(), a.double(), fma_prng_bits=16, **fma)
    with pytest.raises(ValueError, match="accumulate format"):
        binaryK_matmul(
            a,
            a,
            mul_K=8,
            mul_P=4,
            acc_K=30,
            acc_P=23,
            rounding_mode=R.RNE,
            carrier_rounding=R.RO,
        )


@pytest.mark.parametrize(
    "build",
    [
        lambda: FusedMac(
            BinaryK(8, 4), accumulate_algorithm=AccumulateAlgorithm.KAHAN, carrier_rounding=R.RO
        ),
        lambda: SplitMac(
            BinaryK(8, 4),
            SuperFP(3, 4, 8, 7),
            accumulate_algorithm=AccumulateAlgorithm.TREE,
            block_size=4,
            carrier_rounding=R.RO,
        ),
        lambda: FusedMac(None, carrier_rounding=R.RO),
        lambda: FusedMac(BinaryK(8, 4), carrier_rounding=R.RD),
        lambda: binaryK_superfp_matmul(
            torch.ones(2, 2),
            torch.ones(2, 2),
            mul_K=8,
            mul_P=4,
            acc_man_bits=3,
            acc_exp_bits=4,
            acc_normal_binades=8,
            acc_bias=7,
            block_size=4,
            accumulate_algorithm=AccumulateAlgorithm.BLOCK,
            carrier_rounding=R.RO,
        ),
        lambda: binaryK_matmul_mixed(
            torch.ones(2, 2),
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int32),
            mul_K=[8, 8],
            mul_P=[4, 3],
            accumulate_algorithm=AccumulateAlgorithm.KAHAN,
            carrier_rounding=R.RO,
        ),
        lambda: binaryK_matmul_fma_mixed(
            torch.ones(2, 2),
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int32),
            fma_K=[8, 8],
            fma_P=[4, 3],
            fma_quant=False,
            carrier_rounding=R.RO,
        ),
        lambda: binaryK_matmul_fma(
            torch.ones(2, 2),
            torch.ones(2, 2),
            fma_K=8,
            fma_P=4,
            block_size=16,
            carrier_rounding=R.RO,
            accumulate_algorithm=AccumulateAlgorithm.BLOCK,
        ),
        lambda: binaryK_matmul_fma(
            torch.ones(2, 2),
            torch.ones(2, 2),
            fma_K=8,
            fma_P=4,
            fma_quant=False,
            carrier_rounding=R.RO,
        ),
    ],
    ids=[
        "kahan",
        "cross-family TREE",
        "unrounded fma",
        "not RNE or RO",
        "flat cross-family BLOCK",
        "flat palette KAHAN",
        "flat palette fma_quant=False",
        "flat BLOCK",
        "fma_quant=False",
    ],
)
def test_what_the_odd_step_does_not_implement_raises(build):
    with pytest.raises(ValueError):
        build()


def test_the_op_refuses_an_unrounded_fused_step():
    a = torch.ones(2, 2)
    with pytest.raises(RuntimeError, match="fma_quant=false"):
        torch.ops.mptorch.custom_matmul_binaryK_fma_odd(
            a, a, False, False, False, 8, 4, 7, True, 0, 0, 0, 0, 0
        )


@pytest.mark.parametrize("device", float64_devices)
def test_a_qlinear_runs_the_odd_step_in_every_pass(device):
    torch.manual_seed(3)
    formats = binaryK_gemm_formats_fma(12, 7, rounding_mode=R.RU, carrier_rounding=R.RO)
    layer = QLinear(16, 8, bias=False, formats=formats).to(device)
    x = torch.randn(5, 16, device=device, requires_grad=True)
    out = layer(x)
    mac = FusedMac(BinaryK(12, 7), rounding=R.RU, carrier_rounding=R.RO)
    with torch.no_grad():
        assert torch.equal(out, qmm(x, layer.weight.T, mac))
    out.sum().backward()
    g = torch.ones_like(out)
    assert x.grad is not None and torch.equal(x.grad, qmm(g, layer.weight.detach(), mac))


# --- the conv passes, the cross-family macs and the palettes ------------------
#
# Each is an existing op's twin with the carrier steps rounded to odd: the
# cross-family GEMMs are held to the exact reference with the accumulate
# format in the other family, the palette GEMMs entry by entry to the
# single-format odd GEMMs, and every conv pass to the odd GEMM over the
# unfolded operands (tests/test_qconv_gemm.py's references, which run the
# spec they are given).

CROSS = [(BK, SF), (SF, BK)]


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["b32", "b64"])
@pytest.mark.parametrize("pair", CROSS, ids=["binaryK_superfp", "superfp_binaryK"])
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_the_odd_cross_family_gemm_rounds_each_exact_step(device, dtype, pair, mode):
    mul, acc = pair
    a, b = _gemm_operands(dtype, seed=zlib.crc32(f"cross{mul}{mode.name}".encode()))
    want = _reference(a, b, "split", mul, mode, acc=acc).to(dtype)
    mac = SplitMac(mul, acc, rounding=mode, carrier_rounding=R.RO)
    assert torch.equal(qmm(a.to(device), b.to(device), mac).cpu(), want)


# Two entries per palette, both inside every deterministic mode's bound.
PALETTE_ENTRIES = {
    "binaryK": [BK, BinaryK(22, 15)],
    "superfp": [SF, SuperFP(man_bits=12, exp_bits=6, normal_binades=40, bias=31)],
}


def _odd_mac(kind: str, fmt, mode: RoundMode) -> FusedMac | SplitMac:
    if kind == "fused":
        return FusedMac(fmt, rounding=mode, carrier_rounding=R.RO)
    acc = None if kind == "split unrounded sum" else fmt
    return SplitMac(fmt, acc, rounding=mode, carrier_rounding=R.RO)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["b32", "b64"])
@pytest.mark.parametrize("family", list(PALETTE_ENTRIES))
@pytest.mark.parametrize("kind", ["fused", "split", "split unrounded sum"])
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_each_odd_palette_entry_is_its_formats_odd_gemm(device, dtype, family, kind, mode):
    entries = PALETTE_ENTRIES[family]
    a, b = _gemm_operands(dtype, seed=zlib.crc32(f"pal{family}{kind}{mode.name}".encode()))
    a, b = a.to(device), b.to(device)
    M, N = a.shape[0], b.shape[1]
    idx = (torch.arange(M * N, device=device) % len(entries)).to(torch.int32).view(M, N)
    got = qmm(a, b, _odd_mac(kind, Palette(entries), mode), prec_idx=idx)
    for i, fmt in enumerate(entries):
        want = qmm(a, b, _odd_mac(kind, fmt, mode))
        assert torch.equal(got[idx == i], want[idx == i])


def test_a_palette_entry_outside_the_bound_raises():
    a = torch.ones(2, 2)
    idx = torch.zeros(2, 2, dtype=torch.int32)
    kw: dict[str, Any] = dict(rounding_mode=R.RNE, carrier_rounding=R.RO)
    binaryK_matmul_mixed(a, a, idx, mul_K=[28, 28], mul_P=[22, 21], **kw)
    with pytest.raises(ValueError, match="multiply \\(entry 1\\) format's P is 23"):
        binaryK_matmul_mixed(a, a, idx, mul_K=[28, 28], mul_P=[22, 23], **kw)
    # binary64 holds it, and the accumulate entries are held as well
    binaryK_matmul_mixed(a.double(), a.double(), idx, mul_K=[28, 28], mul_P=[22, 23], **kw)
    with pytest.raises(ValueError, match="accumulate \\(entry 0\\)"):
        binaryK_matmul_mixed(
            a, a, idx, mul_K=[8, 8], mul_P=[4, 3], acc_K=[30, 30], acc_P=[23, 4], **kw
        )


def test_a_cross_family_format_outside_the_bound_raises():
    a = torch.ones(2, 2)
    kw: dict[str, Any] = dict(mul_K=8, mul_P=4, acc_exp_bits=6, acc_normal_binades=40)
    kw |= dict(acc_bias=31, rounding_mode=R.RNE, carrier_rounding=R.RO)
    binaryK_superfp_matmul(a, a, acc_man_bits=21, **kw)  # P = 22
    with pytest.raises(ValueError, match="accumulate format's P is 23"):
        binaryK_superfp_matmul(a, a, acc_man_bits=22, **kw)


@pytest.mark.parametrize("device", float64_devices)
def test_the_mac_tier_and_the_flat_tier_run_the_same_odd_palette(device):
    a, b = _gemm_operands(torch.float32, seed=11)
    a, b = a.to(device), b.to(device)
    idx = (torch.arange(30, device=device) % 2).to(torch.int32).view(6, 5)
    mac = SplitMac(Palette(PALETTE_ENTRIES["binaryK"]), None, rounding=R.RD, carrier_rounding=R.RO)
    flat = binaryK_matmul_mixed(
        a, b, idx, mul_K=[28, 22], mul_P=[22, 15], accumulate_quant=False,
        rounding_mode=R.RD, carrier_rounding=R.RO,
    )  # fmt: skip
    assert torch.equal(qmm(a, b, mac, prec_idx=idx), flat)


# SR is not compared: the CPU and CUDA kernels draw from different generators.
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["b32", "b64"])
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_the_cpu_and_cuda_odd_palette_and_cross_gemms_agree_word_for_word(dtype, mode):
    torch.manual_seed(6)
    a = torch.randn(33, 70, dtype=dtype) * torch.pow(2.0, torch.randint(-8, 9, (33, 70)))
    b = torch.randn(70, 17, dtype=dtype)
    idx = torch.randint(0, 2, (33, 17), dtype=torch.int32)
    word = torch.int32 if dtype is torch.float32 else torch.int64
    bk = Palette([BinaryK(16, 11), BinaryK(16, 9)])
    sf = Palette([SuperFP(9, 5, 16, 15), SuperFP(7, 5, 16, 15)])
    for mac, kw in (
        (FusedMac(bk, rounding=mode, carrier_rounding=R.RO), {"prec_idx": idx}),
        (SplitMac(bk, bk, rounding=mode, carrier_rounding=R.RO), {"prec_idx": idx}),
        (SplitMac(sf, None, rounding=mode, carrier_rounding=R.RO), {"prec_idx": idx}),
        (FusedMac(sf, rounding=mode, carrier_rounding=R.RO), {"prec_idx": idx}),
        (
            SplitMac(BinaryK(16, 11), SuperFP(9, 5, 16, 15), rounding=mode, carrier_rounding=R.RO),
            {},
        ),
        (
            SplitMac(SuperFP(9, 5, 16, 15), BinaryK(16, 11), rounding=mode, carrier_rounding=R.RO),
            {},
        ),
    ):
        cpu = qmm(a, b, mac, **kw)
        cuda = qmm(a.cuda(), b.cuda(), mac, **{k: v.cuda() for k, v in kw.items()}).cpu()
        assert torch.equal(cpu.view(word), cuda.view(word)), mac


# --- conv ------------------------------------------------------------------------

B12 = BinaryK(16, 11)
S12 = SuperFP(9, 5, 16, 15)


def _odd_conv_macs(mode: RoundMode) -> dict[str, FusedMac | SplitMac]:
    return {
        "bk_split": SplitMac(BinaryK(12, 7), B12, rounding=mode, carrier_rounding=R.RO),
        "bk_split_noacc": SplitMac(BinaryK(12, 7), None, rounding=mode, carrier_rounding=R.RO),
        "bk_fused": FusedMac(B12, rounding=mode, carrier_rounding=R.RO),
        "sf_split": SplitMac(SuperFP(5, 5, 16, 15), S12, rounding=mode, carrier_rounding=R.RO),
        "sf_fused": FusedMac(S12, rounding=mode, carrier_rounding=R.RO),
        "bk_sf_split": SplitMac(BinaryK(12, 7), S12, rounding=mode, carrier_rounding=R.RO),
        "sf_bk_split": SplitMac(SuperFP(5, 5, 16, 15), B12, rounding=mode, carrier_rounding=R.RO),
    }


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("mode", (*DETERMINISTIC, R.SR), ids=lambda m: m.name)
@pytest.mark.parametrize("which", list(_odd_conv_macs(R.RNE)))
def test_every_odd_conv_pass_is_the_odd_gemm_over_the_unfold(device, mode, which):
    from .test_qconv_gemm import _check_three_passes

    mac = _odd_conv_macs(mode)[which]
    assert str(spec_for_mac(mac).op).endswith("_odd.default")
    _check_three_passes(mac, device, stride=(2, 1), padding=(1, 0))


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("which", ["bk_split", "sf_fused", "bk_sf_split"])
def test_the_odd_conv_ops_compute_in_binary64(device, which):
    from .test_qconv_gemm import _check_three_passes

    mac = _odd_conv_macs(R.RU)[which]
    _check_three_passes(mac, device, stride=2, padding=1, dtype=torch.float64)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize(
    "geometry", ["1d_strided", "2d_all", "2d_groups", "2d_depthwise", "3d_all"]
)
def test_the_odd_conv_ops_hold_over_the_geometries(device, geometry):
    from .test_qconv_gemm import GEOMETRIES, _check_three_passes

    size, kernel, stride, padding, dilation, groups, C, Cout = GEOMETRIES[geometry]
    for mac in (
        _odd_conv_macs(R.RU)["bk_split"],
        _odd_conv_macs(R.RD)["sf_fused"],
        _odd_conv_macs(R.RZ)["sf_bk_split"],
    ):
        _check_three_passes(
            mac, device, C=C, Cout=Cout, size=size, kernel=kernel, stride=stride,
            padding=padding, dilation=dilation, groups=groups,
        )  # fmt: skip


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("mode", [R.RU, R.SR], ids=lambda m: m.name)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("kind", ["split", "fused"])
def test_the_odd_palette_conv_passes(device, mode, stride, kind):
    from .test_qconv_gemm import _check_three_passes, _out

    pal = Palette([B12, BinaryK(16, 8)])
    mac = _odd_mac(kind, pal, mode)
    B, C, Cout, groups, size, kernel = 2, 4, 6, 2, (7, 6), (3, 3)
    out = _out(size, kernel, (stride,) * 2, (1, 1), (1, 1))
    g = torch.Generator().manual_seed(3)
    maps = (
        torch.randint(0, 2, (Cout, out[0] * out[1]), generator=g).to(device),
        torch.randint(0, 2, (C, size[0] * size[1]), generator=g).to(device),
        torch.randint(0, 2, (Cout, C // groups * 9), generator=g).to(device),
    )
    _check_three_passes(
        mac, device, B=B, C=C, Cout=Cout, size=size, kernel=kernel, stride=stride, padding=1,
        groups=groups, maps=maps,
    )  # fmt: skip


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_the_odd_strided_igrad_is_the_zero_inserted_sum_under_naive(device, mode):
    """The input gradient's residue classes leave out the products of the
    zeros a stride inserts. The odd step keeps a zero term exact
    (fma_odd(0, b, s) = s, add_odd(s, 0) = s), so the equality R-4 states
    for the nearest step holds for it too."""
    from .test_qconv_gemm import _hooks, _strided_operands, ref_igrad_zero_inserted

    x, w, dy, geo, (size, s, p, d, groups) = _strided_operands("2d_s2", device)
    for mac in _odd_conv_macs(mode).values():
        _, igrad, _ = _hooks(conv_formats(mac))
        dx = igrad(dy, w, input_size=x.shape, **geo)
        ref = ref_igrad_zero_inserted(spec_for_mac(mac), dy, w, size, s, p, d, groups)
        assert torch.equal(dx, ref)


@requires_cuda
@pytest.mark.parametrize("mode", DETERMINISTIC, ids=lambda m: m.name)
def test_the_cpu_and_cuda_odd_conv_passes_agree_word_for_word(mode):
    from .test_qconv_gemm import _check_three_passes

    g = torch.Generator().manual_seed(4)
    palette_maps = tuple(torch.randint(0, 2, n, generator=g) for n in ((5, 1), (3, 1), (5, 1)))
    none: tuple[torch.Tensor | None, ...] = (None, None, None)
    cases: list[tuple[FusedMac | SplitMac, tuple[torch.Tensor | None, ...]]] = [
        (m, none) for m in _odd_conv_macs(mode).values()
    ]
    cases.append((_odd_mac("fused", Palette([B12, BinaryK(16, 8)]), mode), palette_maps))
    for mac, maps in cases:
        cpu = _check_three_passes(mac, "cpu", stride=2, padding=1, maps=maps)
        cuda_maps = tuple(None if m is None else m.cuda() for m in maps)
        cuda = _check_three_passes(mac, "cuda", stride=2, padding=1, maps=cuda_maps)
        for c, d in zip(cpu, cuda, strict=True):
            assert torch.equal(c.view(torch.int32), d.cpu().view(torch.int32)), mac


@pytest.mark.parametrize("device", float64_devices)
def test_a_qconv2d_runs_the_odd_step_in_every_pass(device):
    from .test_qconv_gemm import ref_fwd, ref_igrad, ref_wgrad

    torch.manual_seed(9)
    mac = FusedMac(B12, rounding=R.RU, carrier_rounding=R.RO)
    layer = QConv2d(3, 4, 3, stride=2, padding=1, bias=False, formats=conv_formats(mac))
    layer = layer.to(device)
    x = torch.randn(2, 3, 9, 8, device=device, requires_grad=True)
    y = layer(x)
    spec = spec_for_mac(mac)
    w = layer.weight.detach()
    geo = ((2, 2), (1, 1), (1, 1), 1)
    with torch.no_grad():
        assert torch.equal(y, ref_fwd(spec, x, w, *geo, tuple(y.shape[2:])))
    dy = torch.randn_like(y)
    y.backward(dy)
    assert x.grad is not None and layer.weight.grad is not None
    assert torch.equal(x.grad, ref_igrad(spec, dy, w, (9, 8), *geo))
    assert torch.equal(layer.weight.grad, ref_wgrad(spec, dy, x.detach(), (3, 3), *geo))


@pytest.mark.parametrize("device", float64_devices)
def test_a_palette_qmatmul_runs_the_odd_step(device):
    torch.manual_seed(10)
    a = torch.randn(6, 9, device=device, requires_grad=True)
    b = torch.randn(9, 5, device=device)
    maps = {
        "prec_idx": torch.randint(0, 2, (6, 5), dtype=torch.int32, device=device),
        "agrad_prec_idx": torch.randint(0, 2, (6, 9), dtype=torch.int32, device=device),
        "bgrad_prec_idx": torch.randint(0, 2, (9, 5), dtype=torch.int32, device=device),
    }
    mac = FusedMac(Palette([B12, BinaryK(16, 8)]), rounding=R.RD, carrier_rounding=R.RO)
    layer = QMatmul(formats=matmul_formats(mac, **maps))
    out = layer(a, b)
    with torch.no_grad():
        assert torch.equal(out, qmm(a, b, mac, prec_idx=maps["prec_idx"]))
    g = torch.ones_like(out)
    out.backward(g)
    assert a.grad is not None
    assert torch.equal(a.grad, qmm(g, b.T, mac, prec_idx=maps["agrad_prec_idx"]))
