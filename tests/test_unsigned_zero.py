"""
No quantizer and no GEMM returns -0.0.

IEEE P3109 has a single zero, code point 0, and it is unsigned: in a signed
binaryK the encoding that would hold -0.0 is the format's NaN, and superfp
spends no code on a negative zero either. The kernels simulate a format's
values in a carrier, binary32 or binary64, that does have -0.0 and produces it
readily: an underflow that keeps its sign, a directed mode negating a magnitude
that rounded away, a -0.0 input passed through. So every zero a cast returns
has to be made +0.0, and the casts do it where the zero is made rather than on
the way out: the clip functions of mptorch/csrc/common/bit_helper.h drop the
sign with the magnitude, superfp's region arms return a bare zero, the
float-arithmetic fast paths select +0.0, and the one signed zero that can still
arise afterwards, a directed mode's negation of a magnitude that rounded to
zero, is handled by bit_helper.h's negate_magnitude. The checks here are on
the sign bit, because -0.0 == 0.0 is true and a value comparison, which is what
the rest of the suite makes, lets a -0.0 through.

A float64 tensor is rounded in binary64 by separate instantiations of the same
casts, so the elementwise and GEMM checks run again on float64 inputs down to
binary64's own subnormals, over formats binary32 cannot carry.

The GEMMs get their own tests because a -0.0 reaches their output by another
route. The running sum starts at +0.0, which absorbs a -0.0 product, so what
survives is an accumulate step rounding a negative partial sum to zero. On
CUDA a K that is not a multiple of 16 hides that: the kernel pads its last
tile with 0 * 0 steps, whose +0.0 absorbs the zero's sign as well, while the
CPU does not pad and returns what the accumulate step made. So K takes both
kinds of value, and the two backends are held to the same words.
"""

import itertools

import pytest
import torch

from mptorch.number import RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_fma_mixed,
    binaryK_matmul_mixed,
    binaryK_quantize,
    superfp_matmul,
    superfp_matmul_fma,
    superfp_matmul_fma_mixed,
    superfp_matmul_mixed,
    superfp_quantize,
)
from tests.markers import available_devices, requires_cuda

# (K, P, bias) and (man_bits, exp_bits, normal_binades, bias): formats on both
# sides of the gates that admit the float-arithmetic fast paths of
# cast_binaryK.h and cast_superfp.h, one binaryK without significand bits
# (P = 1), one with an exponent range wider than binary32's (24p8), and one
# superfp format whose underflow region binary32 cannot reach (m5e5n1b15,
# whose supernormals run down to 2**-976).
BINARYK_FORMATS = [
    (8, 4, None),
    (8, 3, None),
    (6, 3, None),
    (5, 2, None),
    (6, 1, None),
    (8, 4, 7),
    (11, 5, None),
    (24, 8, None),
]
SUPERFP_FORMATS = [
    (3, 4, 1, 7),
    (4, 4, 2, 7),
    (2, 3, 1, 3),
    (0, 4, 1, 7),
    (5, 5, 1, 15),
    (3, 4, 8, 7),
]

# SR repeats the probe so each input draws several times, which is what
# reaches the rounding-up and the rounding-down arm of every value.
SR_REPEATS = 16
DETERMINISTIC = [m for m in RoundMode if m is not RoundMode.SR]


def _probe() -> torch.Tensor:
    """+-0, and four significands in every binade from float32's smallest
    subnormal up to 2, with their negatives, so that whatever a format's range,
    some of these round to zero from below it."""
    mags = torch.tensor([m * 2.0**e for e in range(-149, 2) for m in (1.0, 1.25, 1.5, 1.75)])
    mags = mags[mags > 0].unique()
    zero = torch.zeros(1)
    return torch.cat([zero, -zero, mags, -mags])


def _probe64() -> torch.Tensor:
    """`_probe` for binary64: +-0, and in every binade from its smallest
    subnormal up to 2, two significands float32 could hold and two it could
    not."""
    sigs = (1.0, 1.25, 1.5 + 2.0**-40, 2.0 - 2.0**-52)
    mags = torch.tensor([m * 2.0**e for e in range(-1074, 2) for m in sigs], dtype=torch.float64)
    mags = mags[mags > 0].unique()
    zero = torch.zeros(1, dtype=torch.float64)
    return torch.cat([zero, -zero, mags, -mags])


# The binary64 formats: the binary32 ones again, two past binary32's precision
# and exponent range, and two whose floor sits at binary64's own subnormals.
BINARYK_FORMATS64 = [
    (8, 4, None),
    (6, 1, None),
    (24, 8, None),
    (40, 30, None),
    (63, 53, None),
    (14, 4, 1019),
]
SUPERFP_FORMATS64 = [(3, 4, 1, 7), (0, 4, 1, 7), (20, 10, 1020, 511), (3, 4, 1, 917)]


def _negative_zeros(t: torch.Tensor) -> torch.Tensor:
    """Elementwise: a zero whose sign bit is set, which `t == -0.0` cannot see."""
    return (t == 0) & torch.signbit(t)


def _describe(x: torch.Tensor, out: torch.Tensor) -> str:
    """How many -0.0 `out` holds, and the input of the first one."""
    bad = _negative_zeros(out)
    first = int(bad.nonzero()[0, 0])
    return f"{int(bad.sum())} x -0.0, e.g. from {x[first].item()!r}"


# --- elementwise ----------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("subnormals_mode", list(SubnormalsMode))
@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
@pytest.mark.parametrize("carrier", ["binary32", "binary64"])
def test_binaryK_quantize(device, carrier, signed, subnormals_mode):
    """No binaryK format, saturation mode and rounding mode returns a -0.0 for
    any probe value, in either carrier."""
    wide = carrier == "binary64"
    x = _probe64() if wide else _probe()
    failures = []
    for (K, P, bias), saturation_mode, mode in itertools.product(
        BINARYK_FORMATS64 if wide else BINARYK_FORMATS, SaturationMode, RoundMode
    ):
        xs = x.repeat(SR_REPEATS) if mode is RoundMode.SR else x
        out = binaryK_quantize(
            xs.to(device),
            K,
            P,
            bias=bias,
            # binary64 draws from its 52 significand bits, which P = 53 fills
            prng_bits=(min(8, 52 - (P - 1)) if wide else 8) if mode is RoundMode.SR else 0,
            is_signed=signed,
            rounding_mode=mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
        ).cpu()
        if _negative_zeros(out).any():
            fmt = f"Binary{K}p{P} bias={bias}"
            failures.append(f"{fmt} {saturation_mode.name} {mode.name}: {_describe(xs, out)}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
@pytest.mark.parametrize("carrier", ["binary32", "binary64"])
def test_superfp_quantize(device, carrier, signed):
    """The same for superfp, whose underflow region flushes to a zero of its
    own making."""
    wide = carrier == "binary64"
    x = _probe64() if wide else _probe()
    failures = []
    for (man_bits, exp_bits, normal_binades, bias), saturation_mode, mode in itertools.product(
        SUPERFP_FORMATS64 if wide else SUPERFP_FORMATS, SaturationMode, RoundMode
    ):
        xs = x.repeat(SR_REPEATS) if mode is RoundMode.SR else x
        out = superfp_quantize(
            xs.to(device),
            man_bits,
            exp_bits,
            normal_binades,
            bias,
            prng_bits=8 if mode is RoundMode.SR else 0,
            is_signed=signed,
            rounding_mode=mode,
            saturation_mode=saturation_mode,
        ).cpu()
        if _negative_zeros(out).any():
            failures.append(
                f"m{man_bits}e{exp_bits}n{normal_binades}b{bias} {saturation_mode.name} "
                f"{mode.name}: {_describe(xs, out)}"
            )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64], ids=str)
def test_storage_dtypes(device, dtype):
    """The other storage dtypes: float16 and bfloat16 load into the same
    binary32 cast and store its result back, and -0.0 survives both
    conversions, so the sign has to come off inside the cast; float64 is
    rounded in binary64, whose casts have to take it off just the same."""
    x = _probe().to(dtype)
    failures = []
    for mode in RoundMode:
        prng_bits = 4 if mode is RoundMode.SR else 0
        for subnormals_mode in (SubnormalsMode.SUBNORMALS, SubnormalsMode.NORMALS):
            out = binaryK_quantize(
                x.to(device),
                8,
                4,
                prng_bits=prng_bits,
                rounding_mode=mode,
                subnormals_mode=subnormals_mode,
            ).cpu()
            if _negative_zeros(out).any():
                failures.append(
                    f"Binary8p4 {subnormals_mode.name} {mode.name}: {_describe(x, out)}"
                )
        out = superfp_quantize(
            x.to(device), 3, 4, 1, 7, prng_bits=prng_bits, rounding_mode=mode
        ).cpu()
        if _negative_zeros(out).any():
            failures.append(f"m3e4n1b7 {mode.name}: {_describe(x, out)}")
    assert not failures, "\n".join(failures)


# --- GEMMs ----------------------------------------------------------------------

M, N = 3, 5
# Ragged against both tilings (2), a whole number of CUDA tiles (16), and past
# one CPU tile (34).
K_VALUES = [2, 16, 34]
GEMM_OPS = [
    "binaryK split",
    "binaryK fma",
    "binaryK mixed split",
    "binaryK mixed fma",
    "superfp split",
    "superfp fma",
    "superfp mixed split",
    "superfp mixed fma",
]


def _operands(op: str, K: int, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Operands whose every output element is s^2 (1 - 1 + 1 - ... + 1 - (1 + d)):
    exactly +0.0 before the last pair and -s^2 d after it, which is below the
    accumulator format's smallest magnitude, so the last accumulate step rounds
    a negative partial sum to zero. The multiply format holds 1 + d, so the
    products reach the accumulator unrounded. K must be even."""
    s, d = (1.0, 2.0**-12) if op.startswith("binaryK") else (2.0**-55, 2.0**-8)
    col = torch.full((K,), s)
    col[1 : K - 2 : 2] = -s
    col[K - 1] = -s * (1 + d)
    a = torch.full((M, K), s)
    b = col[:, None].expand(K, N).contiguous()
    return a.to(device, dtype), b.to(device, dtype)


def _gemm(op: str, a, b, mode: RoundMode, subnormals_mode: SubnormalsMode) -> torch.Tensor:
    """The GEMM `op` on `a @ b` in the formats `_operands` was built for.
    binaryK: multiply in Binary16p13 (holds 1 + 2^-12), accumulate in Binary8p4
    (smallest subnormal 2^-10). superfp: multiply in m12e8n254b127 (normal down
    to 2^-125), accumulate in m3e4n1b7 (underflows below 2^-112). The mixed ops
    put every element on slot 0, which is the same pair."""
    prng = 4 if mode is RoundMode.SR else 0
    idx = torch.zeros(M, N, dtype=torch.int32, device=a.device)
    if op == "binaryK split":
        return binaryK_matmul(
            a,
            b,
            mul_K=16,
            mul_P=13,
            acc_K=8,
            acc_P=4,
            mul_prng_bits=prng,
            acc_prng_bits=prng,
            rounding_mode=mode,
            subnormals_mode=subnormals_mode,
        )
    if op == "binaryK fma":
        return binaryK_matmul_fma(
            a,
            b,
            fma_K=8,
            fma_P=4,
            fma_prng_bits=prng,
            rounding_mode=mode,
            subnormals_mode=subnormals_mode,
        )
    if op == "binaryK mixed split":
        return binaryK_matmul_mixed(
            a,
            b,
            idx,
            mul_K=[16, 8],
            mul_P=[13, 4],
            acc_K=[8, 8],
            acc_P=[4, 4],
            mul_prng_bits=prng,
            acc_prng_bits=prng,
            rounding_mode=mode,
            subnormals_mode=subnormals_mode,
        )
    if op == "binaryK mixed fma":
        return binaryK_matmul_fma_mixed(
            a,
            b,
            idx,
            fma_K=[8, 8],
            fma_P=[4, 3],
            fma_prng_bits=prng,
            rounding_mode=mode,
            subnormals_mode=subnormals_mode,
        )
    if op == "superfp split":
        return superfp_matmul(
            a,
            b,
            mul_man_bits=12,
            mul_exp_bits=8,
            mul_normal_binades=254,
            mul_bias=127,
            acc_man_bits=3,
            acc_exp_bits=4,
            acc_normal_binades=1,
            acc_bias=7,
            mul_prng_bits=prng,
            acc_prng_bits=prng,
            rounding_mode=mode,
        )
    if op == "superfp fma":
        return superfp_matmul_fma(
            a,
            b,
            fma_man_bits=3,
            fma_exp_bits=4,
            fma_normal_binades=1,
            fma_bias=7,
            fma_prng_bits=prng,
            rounding_mode=mode,
        )
    if op == "superfp mixed split":
        return superfp_matmul_mixed(
            a,
            b,
            idx,
            mul_man_bits=[12, 3],
            mul_exp_bits=[8, 4],
            mul_normal_binades=[254, 1],
            mul_bias=[127, 7],
            acc_man_bits=[3, 3],
            acc_exp_bits=[4, 4],
            acc_normal_binades=[1, 1],
            acc_bias=[7, 7],
            mul_prng_bits=prng,
            acc_prng_bits=prng,
            rounding_mode=mode,
        )
    assert op == "superfp mixed fma", op
    return superfp_matmul_fma_mixed(
        a,
        b,
        idx,
        fma_man_bits=[3, 2],
        fma_exp_bits=[4, 3],
        fma_normal_binades=[1, 1],
        fma_bias=[7, 3],
        fma_prng_bits=prng,
        rounding_mode=mode,
    )


def _gemm_configs(op: str, modes):
    """(rounding mode, subnormals mode, K) for every configuration `op` takes.
    superfp has no subnormals mode, so one pass stands for it."""
    subs = list(SubnormalsMode) if op.startswith("binaryK") else [SubnormalsMode.SUBNORMALS]
    return itertools.product(modes, subs, K_VALUES)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", GEMM_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_gemm(device, op, dtype):
    """An accumulate step that rounds a negative partial sum to zero writes
    +0.0, in every GEMM op, rounding mode, subnormals mode and K, on operands
    of either carrier."""
    failures = []
    for mode, subnormals_mode, K in _gemm_configs(op, RoundMode):
        out = _gemm(op, *_operands(op, K, device, dtype), mode, subnormals_mode).cpu()
        assert out.dtype is dtype
        if mode is RoundMode.RZ:
            # Toward zero every element's sum is a zero, which is the case
            # under test, so this guards the operands' construction.
            assert (out == 0).all(), (subnormals_mode.name, K)
        if _negative_zeros(out).any():
            failures.append(
                f"{mode.name} {subnormals_mode.name} K={K}: "
                f"{int(_negative_zeros(out).sum())} of {out.numel()} are -0.0"
            )
    assert not failures, "\n".join(failures)


@requires_cuda
@pytest.mark.parametrize("op", GEMM_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_gemm_backends_agree_on_zero(op, dtype):
    """CPU and CUDA return the same words, sign of zero included. Both
    accumulate in the same order, so the only thing that could tell them apart
    is what the CUDA kernel's tile padding adds, which is +0.0, and so a matter
    of the zero's sign alone."""
    failures = []
    words = torch.int64 if dtype is torch.float64 else torch.int32
    for mode, subnormals_mode, K in _gemm_configs(op, DETERMINISTIC):
        cpu = _gemm(op, *_operands(op, K, "cpu", dtype), mode, subnormals_mode)
        gpu = _gemm(op, *_operands(op, K, "cuda", dtype), mode, subnormals_mode).cpu()
        if not torch.equal(cpu.view(words), gpu.view(words)):
            failures.append(f"{mode.name} {subnormals_mode.name} K={K}")
    assert not failures, "\n".join(failures)
