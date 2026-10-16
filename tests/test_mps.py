"""
The MPS backend (``csrc/mps/``) held to the CPU backend, word for word.

Every op on an Apple GPU tensor runs Metal kernels compiled from the casts
and GEMM policies the CPU runs, and its result is compared with the CPU op's
on the same values: the words themselves, so a NaN's payload and a zero's
sign count, under every rounding mode. Stochastic rounding included: both
backends draw one seed from the CPU generator and key every element's Philox
stream on it the same way, so the same ``torch.manual_seed`` gives the same
draws. The inputs reach the corners: every word of the 16-bit dtypes, and for
float32 a spread of random words (every exponent, infinities, NaNs and
subnormals among them) next to values inside the formats' ranges.

Every op is here: the elementwise quantizers and their in-place twins, the
eight GEMMs, their KAHAN, BLOCK and TREE twins and the cross-family ones, the
conv ops' three passes, and the block-format ops, packing, unpacking,
quantizing and the block GEMM.

What the backend cannot match is the GPU's flush of binary32 subnormals in
its arithmetic (``csrc/mps/gemm.metal``), and the last tests pin both what
that flush changes and what it does not.

No tier convention: there is no MPS arithmetic of its own to test, only the
CPU's compiled for another device, which the CPU backend's tests hold to the
formats.
"""

import dataclasses
import math
from typing import Any

import pytest
import torch

from mptorch import (
    E8M0,
    MXFP4_E2M1,
    MXFP6_E2M3,
    MXFP8_E4M3,
    NVFP4,
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    ScaleRounding,
    SuperFP,
)
from mptorch.number import RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import (
    FusedMac,
    QLinear,
    SplitMac,
    binaryK_gemm_formats,
    binaryK_quantize,
    binaryK_quantize_,
    block_matmul,
    block_pack,
    block_quantize,
    block_quantize_,
    conv_formats,
    superfp_quantize,
    superfp_quantize_,
)
from mptorch.quant.ops import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_fma_mixed,
    binaryK_matmul_mixed,
    binaryK_superfp_matmul,
    superfp_binaryK_matmul,
    superfp_matmul,
    superfp_matmul_fma,
    superfp_matmul_fma_mixed,
    superfp_matmul_mixed,
)
from tests.markers import requires_mps

pytestmark = requires_mps

_WORD = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
MODES = list(RoundMode)


def _assert_same_words(ref: torch.Tensor, got: torch.Tensor, *, nan_payloads: bool = True) -> None:
    """``got`` is on MPS and holds exactly ``ref``'s words.

    ``nan_payloads=False`` accepts any NaN where ``ref`` has one, for the
    GEMMs: a NaN reaching a product or a sum comes out of the GPU's float unit
    as its canonical NaN, where the CPU's keeps the operand's payload. That is
    the float unit's doing, not the casts', which hand a NaN through whole on
    both (the quantizers are held to it), and CUDA's arithmetic canonicalizes
    the same way (tests/test_cast_fast_paths.py)."""
    assert got.device.type == "mps"
    got = got.cpu()
    assert got.dtype == ref.dtype and got.shape == ref.shape
    word = _WORD[ref.dtype]
    differ = ref.reshape(-1).view(word) != got.reshape(-1).view(word)
    if not nan_payloads:
        differ &= ~(ref.reshape(-1).isnan() & got.reshape(-1).isnan())
    if differ.any():
        i = int(differ.nonzero()[0])
        r, g = ref.reshape(-1)[i], got.reshape(-1)[i]
        raise AssertionError(
            f"{int(differ.sum())} of {differ.numel()} words differ; the first, at {i}: "
            f"cpu {r.item()!r} ({int(r.view(word)):#x}), mps {g.item()!r} ({int(g.view(word)):#x})"
        )


def _on_both(fn, *tensors, seed: int = 7):
    """``fn`` on the CPU tensors and on their MPS copies, each from ``seed``."""
    torch.manual_seed(seed)
    ref = fn(*tensors)
    torch.manual_seed(seed)
    got = fn(*(t.to("mps") for t in tensors))
    return ref, got


def _words(dtype: torch.dtype) -> torch.Tensor:
    """Inputs for a quantizer: every word of a 16-bit dtype; for float32,
    random words, values inside an 8-bit format's range, and the specials."""
    if dtype != torch.float32:
        return torch.arange(-(2**15), 2**15, dtype=torch.int16).view(dtype)
    g = torch.Generator().manual_seed(0)
    random_words = torch.randint(-(2**31), 2**31, (2**17,), generator=g).to(torch.int32)
    scales = torch.pow(2.0, torch.randint(-24, 12, (2**16,), generator=g).float())
    in_range = torch.randn(2**16, generator=g) * scales
    specials = torch.tensor(
        [
            0x00000000, 0x80000000, 0x7F800000, 0xFF800000,  # zeros, infinities
            0x7FC00000, 0xFFC00123, 0x7F800001, 0xFF812345,  # NaNs, quiet and signalling
            0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,  # subnormals
            0x00400000, 0x80400000, 0x00800000, 0x80800000,  # 2^-127, 2^-126
            0x7F7FFFFF, 0xFF7FFFFF,  # the largest finite values
        ],
        dtype=torch.int64,
    ).to(torch.int32)  # fmt: skip
    return torch.cat([random_words.view(torch.float32), in_range, specials.view(torch.float32)])


# (K, P, bias, is_signed, saturation, subnormals). P3109's binary8p4 and its
# neighbours, an unsigned format, P = 1 (no stored mantissa), the two
# non-P3109 subnormal modes, and a 12-bit precision finer than float16's, whose
# float16 results the store has to round.
BINARYK = [
    (8, 4, None, True, SaturationMode.SAT_NONE, SubnormalsMode.SUBNORMALS),
    (8, 3, None, True, SaturationMode.SAT_FINITE, SubnormalsMode.SUBNORMALS),
    (8, 5, None, False, SaturationMode.SAT_PROPAGATE, SubnormalsMode.SUBNORMALS),
    (8, 1, None, True, SaturationMode.SAT_NONE, SubnormalsMode.SUBNORMALS),
    (8, 4, None, True, SaturationMode.SAT_NONE, SubnormalsMode.NORMALS),
    (8, 4, None, True, SaturationMode.SAT_FINITE, SubnormalsMode.EXTENDED_NORMALS),
    (16, 12, None, True, SaturationMode.SAT_NONE, SubnormalsMode.SUBNORMALS),
    (8, 4, None, True, SaturationMode.SAT_FINITE_EXTENDED, SubnormalsMode.SUBNORMALS),
    (8, 3, None, False, SaturationMode.SAT_FINITE_EXTENDED, SubnormalsMode.NORMALS),
]

# (man_bits, exp_bits, normal_binades, bias, is_signed, saturation). The last
# has so many supernormal binades that nothing underflows at all.
SUPERFP = [
    (3, 4, 1, 7, True, SaturationMode.SAT_NONE),
    (2, 5, 2, 15, True, SaturationMode.SAT_FINITE),
    (0, 4, 1, 7, True, SaturationMode.SAT_NONE),
    (4, 3, 1, 3, False, SaturationMode.SAT_PROPAGATE),
    (9, 3, 1, 3, True, SaturationMode.SAT_NONE),
    (3, 4, 1, 7, True, SaturationMode.SAT_FINITE_EXTENDED),
]


def _ids(fmt) -> str:
    return "-".join(str(getattr(v, "name", v)) for v in fmt)


def _cases(formats):
    """Every format in float32, and the first and the widest in float16 and
    bfloat16, whose every word the input holds. Each format meets every
    rounding mode."""
    wide = [formats[0], max(formats, key=lambda f: f[0])]
    return [
        pytest.param(fmt, dtype, id=f"{_ids(fmt)}-{str(dtype).removeprefix('torch.')}")
        for dtype in DTYPES
        for fmt in (formats if dtype == torch.float32 else dict.fromkeys(wide))
    ]


# --------------------------------------------------------------------------
# The elementwise quantizers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rounding_mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("fmt, dtype", _cases(BINARYK))
def test_binaryK_quantize(fmt, dtype, rounding_mode):
    K, P, bias, is_signed, sat, subn = fmt
    x = _words(dtype)
    ref, got = _on_both(
        lambda t: binaryK_quantize(
            t, K, P, bias=bias, prng_bits=10, is_signed=is_signed, rounding_mode=rounding_mode,
            saturation_mode=sat, subnormals_mode=subn,
        ),
        x,
    )  # fmt: skip
    _assert_same_words(ref, got)


@pytest.mark.parametrize("rounding_mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("fmt, dtype", _cases(SUPERFP))
def test_superfp_quantize(fmt, dtype, rounding_mode):
    man, exp, binades, bias, is_signed, sat = fmt
    x = _words(dtype)
    ref, got = _on_both(
        lambda t: superfp_quantize(
            t, man, exp, binades, bias, prng_bits=10, is_signed=is_signed,
            rounding_mode=rounding_mode, saturation_mode=sat,
        ),
        x,
    )  # fmt: skip
    _assert_same_words(ref, got)


def test_quantize_strided_and_empty():
    """A strided view is read in logical order, and an empty tensor comes back
    empty, having drawn the SR seed the CPU draws for it too."""
    x = torch.randn(64, 48)
    ref, got = _on_both(lambda t: binaryK_quantize(t.t()[3:], 8, 4), x)
    _assert_same_words(ref, got)

    torch.manual_seed(1)
    binaryK_quantize(torch.empty(0), 8, 4, rounding_mode=RoundMode.SR)
    after_cpu = torch.rand(4)
    torch.manual_seed(1)
    out = binaryK_quantize(torch.empty(0, device="mps"), 8, 4, rounding_mode=RoundMode.SR)
    assert out.device.type == "mps" and out.numel() == 0
    assert torch.equal(torch.rand(4), after_cpu)


def test_sr_keeps_the_generator_in_step():
    """An SR call consumes what the CPU call consumes from the CPU generator,
    so what torch draws next is the same on both devices."""
    x = torch.randn(1000)
    torch.manual_seed(3)
    binaryK_quantize(x, 8, 4, rounding_mode=RoundMode.SR, prng_bits=8)
    binaryK_matmul(x.view(10, 100), x.view(100, 10), mul_K=8, mul_P=4, rounding_mode=RoundMode.SR)
    after_cpu = torch.rand(4)
    torch.manual_seed(3)
    xm = x.to("mps")
    binaryK_quantize(xm, 8, 4, rounding_mode=RoundMode.SR, prng_bits=8)
    binaryK_matmul(xm.view(10, 100), xm.view(100, 10), mul_K=8, mul_P=4, rounding_mode=RoundMode.SR)
    assert torch.equal(torch.rand(4), after_cpu)


def test_quantize_rejects_what_mps_cannot_hold():
    """An integer tensor raises the NotImplementedError AT_DISPATCH raises on
    the other backends, and a float64 carrier cannot be had on MPS at all."""
    x = torch.zeros(4, dtype=torch.int32, device="mps")
    with pytest.raises(NotImplementedError, match="not implemented for 'Int'"):
        torch.ops.mptorch.binaryK_quant(x, 8, 4, 8, 0, True, 0, 2, 0)
    with pytest.raises((TypeError, RuntimeError), match="float64"):
        binaryK_quantize(torch.randn(4, device="mps"), 8, 4, carrier=torch.float64)


@pytest.mark.parametrize("rounding_mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("op", ["binaryK", "superfp"])
def test_quantize_in_place(op, dtype, rounding_mode):
    """binaryK_quant_ and superfp_quant_ write the CPU's out-of-place words
    into the MPS tensor itself (one buffer bound as the kernel's input and its
    output), a view off the start of its storage included."""
    if op == "binaryK":
        fmt: dict[str, Any] = dict(K=8, P=4, prng_bits=10)
        fn, fn_ = binaryK_quantize, binaryK_quantize_
    else:
        fmt = dict(man_bits=3, exp_bits=4, normal_binades=1, bias=7, prng_bits=10)
        fn, fn_ = superfp_quantize, superfp_quantize_
    x = _words(dtype)
    torch.manual_seed(7)
    ref = fn(x[3:], rounding_mode=rounding_mode, **fmt)
    y = x.to("mps")
    torch.manual_seed(7)
    assert fn_(y[3:], rounding_mode=rounding_mode, **fmt).data_ptr() == y[3:].data_ptr()
    _assert_same_words(ref, y[3:])
    _assert_same_words(x[:3], y[:3])


# --------------------------------------------------------------------------
# The GEMMs
# --------------------------------------------------------------------------

# One call per op, each with its own formats; `rm` is the rounding mode.
_B: dict[str, Any] = dict(mul_K=8, mul_P=4, acc_K=12, acc_P=6, mul_prng_bits=8, acc_prng_bits=8)
_S: dict[str, Any] = dict(
    mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=1, mul_bias=7,
    acc_man_bits=5, acc_exp_bits=5, acc_normal_binades=2, acc_bias=15,
    mul_prng_bits=8, acc_prng_bits=8,
)  # fmt: skip
_BF: dict[str, Any] = dict(fma_K=10, fma_P=5, fma_prng_bits=8)
_SF: dict[str, Any] = dict(
    fma_man_bits=4, fma_exp_bits=5, fma_normal_binades=1, fma_bias=15, fma_prng_bits=8
)
_BM: dict[str, Any] = dict(
    mul_K=[8, 8, 6], mul_P=[4, 3, 3], acc_K=[12, 10, 8], acc_P=[6, 5, 4], mul_prng_bits=8
)
_SM: dict[str, Any] = dict(
    mul_man_bits=[3, 2, 1], mul_exp_bits=[4, 5, 4], mul_normal_binades=1, mul_bias=[7, 15, 7],
    acc_man_bits=[5, 4, 3], acc_exp_bits=[5, 5, 5], acc_normal_binades=2, acc_bias=[15, 15, 15],
    mul_prng_bits=8,
)  # fmt: skip
_BFM: dict[str, Any] = dict(fma_K=[10, 8], fma_P=[5, 4], fma_prng_bits=8)
_SFM: dict[str, Any] = dict(
    fma_man_bits=[4, 2], fma_exp_bits=[5, 5], fma_normal_binades=1, fma_bias=[15, 15]
)

GEMMS = {
    "binaryK": lambda a, b, p, rm, **kw: binaryK_matmul(a, b, rounding_mode=rm, **_B, **kw),
    "superfp": lambda a, b, p, rm, **kw: superfp_matmul(a, b, rounding_mode=rm, **_S, **kw),
    "binaryK_fma": lambda a, b, p, rm, **kw: binaryK_matmul_fma(
        a, b, rounding_mode=rm, **_BF, **kw
    ),
    "superfp_fma": lambda a, b, p, rm, **kw: superfp_matmul_fma(
        a, b, rounding_mode=rm, **_SF, **kw
    ),
    "binaryK_mixed": lambda a, b, p, rm, **kw: binaryK_matmul_mixed(
        a, b, p % 3, rounding_mode=rm, **_BM, **kw
    ),
    "superfp_mixed": lambda a, b, p, rm, **kw: superfp_matmul_mixed(
        a, b, p % 3, rounding_mode=rm, **_SM, **kw
    ),
    "binaryK_fma_mixed": lambda a, b, p, rm, **kw: binaryK_matmul_fma_mixed(
        a, b, p % 2, rounding_mode=rm, **_BFM, **kw
    ),
    "superfp_fma_mixed": lambda a, b, p, rm, **kw: superfp_matmul_fma_mixed(
        a, b, p % 2, rounding_mode=rm, **_SFM, **kw
    ),
}


def _operands(M: int, K: int, N: int, dtype=torch.float32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(M, K, generator=g) * 2).to(dtype)
    b = torch.randn(K, N, generator=g).to(dtype)
    prec = torch.randint(0, 6, (M, N), generator=g, dtype=torch.int32)
    return a, b, prec


@pytest.mark.parametrize("rounding_mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("op", list(GEMMS))
def test_gemm_every_mode(op, rounding_mode):
    a, b, prec = _operands(37, 70, 29)
    ref, got = _on_both(lambda x, y, p: GEMMS[op](x, y, p, rounding_mode), a, b, prec)
    _assert_same_words(ref, got)


@pytest.mark.parametrize("rounding_mode", [RoundMode.RNE, RoundMode.SR], ids=lambda m: m.name)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("op", list(GEMMS))
def test_gemm_narrow_dtypes(op, dtype, rounding_mode):
    """float16 and bfloat16 operands convert on load and the result on store
    exactly as the CPU converts them."""
    a, b, prec = _operands(21, 33, 17, dtype)
    ref, got = _on_both(lambda x, y, p: GEMMS[op](x, y, p, rounding_mode), a, b, prec)
    _assert_same_words(ref, got)


@pytest.mark.parametrize("op", ["binaryK", "binaryK_fma", "superfp", "superfp_fma"])
def test_gemm_unquantized_sum(op):
    """accumulate_quant / fma_quant false: the sum stays in binary32."""
    a, b, prec = _operands(19, 41, 23)
    flag = "fma_quant" if "fma" in op else "accumulate_quant"
    ref, got = _on_both(
        lambda x, y, p: GEMMS[op](x, y, p, RoundMode.RNE, **{flag: False}), a, b, prec
    )
    _assert_same_words(ref, got)


@pytest.mark.parametrize(
    "shapes",
    [
        ((5, 13, 7), (5, 7, 9)),  # batched on both sides
        ((5, 13, 7), (7, 9)),  # a shared weight, read at batch stride 0
        ((13, 7), (5, 7, 9)),  # a shared left operand
        ((1, 13, 7), (5, 7, 9)),  # batch 1 broadcast
        ((1, 1), (1, 1)),
        ((3, 0), (0, 4)),  # K = 0: every sum is the zero it starts at
        ((0, 4), (4, 3)),  # empty output
    ],
    ids=str,
)
@pytest.mark.parametrize(
    "trans", [(False, False), (True, False), (False, True), (True, True)], ids=str
)
def test_gemm_shapes(shapes, trans):
    sa, sb = shapes
    ta, tb = trans
    g = torch.Generator().manual_seed(1)
    a = torch.randn(*sa, generator=g)
    b = torch.randn(*sb, generator=g)
    if ta:
        a = a.transpose(-1, -2).contiguous()
    if tb:
        b = b.transpose(-1, -2).contiguous()
    ref, got = _on_both(
        lambda x, y: binaryK_matmul(x, y, trans_a=ta, trans_b=tb, rounding_mode=RoundMode.SR, **_B),
        a,
        b,
    )
    _assert_same_words(ref, got)


@pytest.mark.parametrize("rounding_mode", [RoundMode.RNE, RoundMode.SR], ids=lambda m: m.name)
@pytest.mark.parametrize(
    "shapes",
    [
        ((16, 16), (16, 16)),  # one whole tile
        ((17, 33), (33, 15)),  # a partial tile on every side, and a partial last K-step
        ((31, 16), (16, 33)),
        ((1, 48), (48, 1)),  # three whole K-steps into one element
        ((3, 17, 33), (3, 33, 18)),  # batched
        ((3, 17, 33), (33, 18)),  # a shared weight
    ],
    ids=str,
)
@pytest.mark.parametrize(
    "trans", [(False, False), (True, False), (False, True), (True, True)], ids=str
)
def test_gemm_tile_edges(shapes, trans, rounding_mode):
    """Shapes on and off gemm.metal's 16 x 16 tiles and 16-step K stages, in
    both of its K-loops: RNE takes the fixed-length loop for whole stages and
    a shorter one for the last, SR the one variable-length loop."""
    sa, sb = shapes
    ta, tb = trans
    g = torch.Generator().manual_seed(3)
    a = torch.randn(*sa, generator=g)
    b = torch.randn(*sb, generator=g)
    if ta:
        a = a.transpose(-1, -2).contiguous()
    if tb:
        b = b.transpose(-1, -2).contiguous()
    ref, got = _on_both(
        lambda x, y: binaryK_matmul(
            x, y, trans_a=ta, trans_b=tb, rounding_mode=rounding_mode, **_B
        ),
        a,
        b,
    )
    _assert_same_words(ref, got)


def test_gemm_strided_views():
    """Transposed views are read in place (the Python side flips the flag),
    and other strided views are copied, on MPS as on the CPU."""
    g = torch.Generator().manual_seed(2)
    a = torch.randn(40, 30, generator=g)
    b = torch.randn(40, 50, generator=g)
    ref, got = _on_both(lambda x, y: binaryK_matmul(x.t(), y[:, ::2], **_B), a, b)
    _assert_same_words(ref, got)


@pytest.mark.parametrize("layout", ["dense", "rows", "columns", "batched", "int64", "on_cpu"])
def test_gemm_palette_maps(layout):
    """Every shape a precision map comes in, and a map left on the CPU."""
    g = torch.Generator().manual_seed(3)
    a = torch.randn(4, 11, 19, generator=g)
    b = torch.randn(19, 13, generator=g)
    maps = {
        "dense": torch.randint(0, 3, (11, 13), generator=g, dtype=torch.int32),
        "rows": torch.randint(0, 3, (11, 1), generator=g, dtype=torch.int32),
        "columns": torch.randint(0, 3, (1, 13), generator=g, dtype=torch.int32),
        "batched": torch.randint(0, 3, (4, 11, 13), generator=g, dtype=torch.int32),
        "int64": torch.randint(0, 3, (11, 13), generator=g),
        "on_cpu": torch.randint(0, 3, (11, 13), generator=g, dtype=torch.int32),
    }
    p = maps[layout]
    torch.manual_seed(5)
    ref = binaryK_matmul_mixed(a, b, p, rounding_mode=RoundMode.SR, **_BM)
    torch.manual_seed(5)
    p_dev = p if layout == "on_cpu" else p.to("mps")
    got = binaryK_matmul_mixed(a.to("mps"), b.to("mps"), p_dev, rounding_mode=RoundMode.SR, **_BM)
    _assert_same_words(ref, got)


def test_gemm_palette_index_out_of_range():
    a, b, prec = (t.to("mps") for t in _operands(6, 5, 4))
    with pytest.raises(RuntimeError, match="prec_idx entries must be in"):
        binaryK_matmul_mixed(a, b, prec + 3, **_BM)


def test_gemm_special_operands():
    """Infinities, NaNs and overflowing products."""
    a, b, _ = _operands(9, 17, 11)
    a[0, 3] = math.inf
    a[1, 5] = -math.inf
    a[2].view(torch.int32)[7] = 0x7FC01234
    b[4, 2] = 1e30
    b[6] = -3e38
    for op in ["binaryK", "binaryK_fma", "superfp", "superfp_fma"]:
        ref, got = _on_both(lambda x, y, op=op: GEMMS[op](x, y, None, RoundMode.RU), a, b)
        _assert_same_words(ref, got, nan_payloads=False)
        assert ref.isnan().any()


KAHAN, BLOCK, TREE = (
    AccumulateAlgorithm.KAHAN,
    AccumulateAlgorithm.BLOCK,
    AccumulateAlgorithm.TREE,
)

# Each algorithm, BLOCK with a block inside the 16-step slab and one past it,
# TREE with levels inside the slab only and across slabs.
ACCUMULATIONS: dict[str, dict[str, Any]] = {
    "kahan": dict(accumulate_algorithm=KAHAN),
    "block4": dict(accumulate_algorithm=BLOCK, block_size=4),
    "block32": dict(accumulate_algorithm=BLOCK, block_size=32),
    "tree8": dict(accumulate_algorithm=TREE, block_size=8),
    "tree64": dict(accumulate_algorithm=TREE, block_size=64),
}
_BOUTER: dict[str, Any] = dict(outer_K=16, outer_P=11, outer_prng_bits=8)
_SOUTER: dict[str, Any] = dict(
    outer_man_bits=7, outer_exp_bits=5, outer_normal_binades=2, outer_bias=15, outer_prng_bits=8
)
_BS: dict[str, Any] = dict(
    mul_K=8, mul_P=4, acc_man_bits=5, acc_exp_bits=5, acc_normal_binades=2, acc_bias=15,
    mul_prng_bits=8, acc_prng_bits=8,
)  # fmt: skip
_SB: dict[str, Any] = dict(
    mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=1, mul_bias=7, acc_K=12, acc_P=6,
    mul_prng_bits=8, acc_prng_bits=8,
)  # fmt: skip

# The ops each algorithm runs on, and the outer format of their family: the
# four *_accumulated GEMMs and the two cross-family ones (the outer format is
# the accumulate's family).
ACC_GEMMS = {
    "binaryK": (lambda a, b, rm, **kw: binaryK_matmul(a, b, rounding_mode=rm, **_B, **kw), _BOUTER),
    "binaryK_fma": (
        lambda a, b, rm, **kw: binaryK_matmul_fma(a, b, rounding_mode=rm, **_BF, **kw),
        _BOUTER,
    ),
    "superfp": (lambda a, b, rm, **kw: superfp_matmul(a, b, rounding_mode=rm, **_S, **kw), _SOUTER),
    "superfp_fma": (
        lambda a, b, rm, **kw: superfp_matmul_fma(a, b, rounding_mode=rm, **_SF, **kw),
        _SOUTER,
    ),
    "binaryK_superfp": (
        lambda a, b, rm, **kw: binaryK_superfp_matmul(a, b, rounding_mode=rm, **_BS, **kw),
        _SOUTER,
    ),
    "superfp_binaryK": (
        lambda a, b, rm, **kw: superfp_binaryK_matmul(a, b, rounding_mode=rm, **_SB, **kw),
        _BOUTER,
    ),
}


@pytest.mark.parametrize("rounding_mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("acc", list(ACCUMULATIONS))
@pytest.mark.parametrize("op", list(ACC_GEMMS))
def test_accumulated_gemm_every_mode(op, acc, rounding_mode):
    """KAHAN, BLOCK and TREE on every op that takes them, in every mode: K = 70
    leaves a partial slab and partial blocks of every size, and nothing is
    padded, on the GPU as on the CPU."""
    fn, outer = ACC_GEMMS[op]
    kw = dict(ACCUMULATIONS[acc])
    if kw["accumulate_algorithm"] is TREE and "fma" in op:
        pytest.skip("TREE sums products pairwise; a fused mac has none")
    if kw["accumulate_algorithm"] is not KAHAN:
        kw.update(outer)
    a, b, _ = _operands(37, 70, 29)
    ref, got = _on_both(lambda x, y: fn(x, y, rounding_mode, **kw), a, b)
    _assert_same_words(ref, got)


@pytest.mark.parametrize("rounding_mode", [RoundMode.RNE, RoundMode.SR], ids=lambda m: m.name)
@pytest.mark.parametrize("op", ["binaryK", "superfp_fma", "binaryK_superfp"])
def test_accumulated_gemm_shapes(op, rounding_mode):
    """Batched, broadcast, transposed and narrow-dtype operands under an
    accumulate algorithm."""
    fn, outer = ACC_GEMMS[op]
    kw = dict(accumulate_algorithm=BLOCK, block_size=16, **outer)
    g = torch.Generator().manual_seed(4)
    a = torch.randn(3, 40, 17, generator=g).transpose(-1, -2)  # read in place, transposed
    b = torch.randn(40, 18, generator=g)
    # bfloat16 does not store the outer formats' 11 bits
    for dtype in (torch.float32, torch.float16):
        ref, got = _on_both(lambda x, y: fn(x, y, rounding_mode, **kw), a.to(dtype), b.to(dtype))
        _assert_same_words(ref, got)


@pytest.mark.parametrize("rounding_mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("op", ["binaryK_superfp", "superfp_binaryK"])
def test_cross_family_gemm_every_mode(op, rounding_mode):
    fn, _ = ACC_GEMMS[op]
    a, b, _ = _operands(37, 70, 29)
    ref, got = _on_both(lambda x, y: fn(x, y, rounding_mode), a, b)
    _assert_same_words(ref, got)


# --------------------------------------------------------------------------
# The conv ops
# --------------------------------------------------------------------------

_CB8, _CB12 = BinaryK(8, 4, prng_bits=8), BinaryK(12, 7, prng_bits=8)
_CS8, _CS12 = SuperFP(3, 4, 8, 7, prng_bits=8), SuperFP(5, 5, 16, 15, prng_bits=8)

# One mac per conv op: the four single-format ones, their palette twins and
# the two cross-family ones.
CONV_MACS = {
    "binaryK": lambda rm, **kw: SplitMac(_CB8, _CB12, rounding=rm, **kw),
    "binaryK_fma": lambda rm, **kw: FusedMac(_CB12, rounding=rm, **kw),
    "superfp": lambda rm, **kw: SplitMac(_CS8, _CS12, rounding=rm, **kw),
    "superfp_fma": lambda rm, **kw: FusedMac(_CS12, rounding=rm, **kw),
    "binaryK_mixed": lambda rm: SplitMac(
        [_CB8, BinaryK(8, 5, prng_bits=8)], [_CB12, BinaryK(12, 8, prng_bits=8)], rounding=rm
    ),
    "binaryK_fma_mixed": lambda rm: FusedMac([_CB12, BinaryK(10, 6, prng_bits=8)], rounding=rm),
    "superfp_mixed": lambda rm: SplitMac(
        [_CS8, SuperFP(2, 4, 8, 7, prng_bits=8)],
        [_CS12, SuperFP(4, 5, 16, 15, prng_bits=8)],
        rounding=rm,
    ),
    "superfp_fma_mixed": lambda rm: FusedMac(
        [_CS12, SuperFP(4, 5, 16, 15, prng_bits=8)], rounding=rm
    ),
    "binaryK_superfp": lambda rm, **kw: SplitMac(_CB8, _CS12, rounding=rm, **kw),
    "superfp_binaryK": lambda rm, **kw: SplitMac(_CS8, _CB12, rounding=rm, **kw),
}

# (size, kernel, stride, padding, dilation, groups): 1D, 2D and 3D, a stride
# whose input gradient has several residue classes, a dilation, and groups.
CONV_GEOMETRIES = {
    "1d": ((17,), (3,), (2,), (1,), (1,), 1),
    "2d": ((9, 8), (3, 3), (2, 2), (1, 1), (1, 1), 2),
    "2d_dilated": ((11, 9), (3, 2), (1, 2), (2, 0), (2, 1), 1),
    "3d": ((5, 6, 4), (2, 3, 2), (1, 2, 1), (0, 1, 1), (1, 1, 1), 2),
}


def _conv_passes(mac, geometry, device, palette):
    """The three passes of `mac` over one geometry's operands on `device`,
    each from the same seed, through conv_formats' hooks."""
    size, kernel, stride, padding, dilation, groups = CONV_GEOMETRIES[geometry]
    nd = len(size)
    B, C, Cout = 2, 4, 6
    out = tuple(
        (n + 2 * p - d * (k - 1) - 1) // s + 1
        for n, k, s, p, d in zip(size, kernel, stride, padding, dilation, strict=True)
    )
    g = torch.Generator().manual_seed(3)
    x = torch.randn(B, C, *size, generator=g)
    w = torch.randn(Cout, C // groups, *kernel, generator=g)
    dy = torch.randn(B, Cout, *out, generator=g)
    maps: list[Any] = [None, None, None]
    if palette:
        L, IN, KK = math.prod(out), math.prod(size), math.prod(kernel)
        maps = [
            torch.randint(0, 2, (Cout, L), generator=g),
            torch.randint(0, 2, (C, IN), generator=g),
            torch.randint(0, 2, (Cout, C // groups * KK), generator=g),
        ]
    formats = conv_formats(
        mac, **{
            k: None if m is None else m.to(device)
            for k, m in zip(("prec_idx", "igrad_prec_idx", "wgrad_prec_idx"), maps, strict=True)
        },
    )  # fmt: skip
    assert formats.fwd_math and formats.bwd_igrad_math and formats.bwd_wgrad_math
    geo: dict[str, Any] = dict(
        stride=stride, padding=padding, dilation=dilation, groups=groups, nd=nd
    )
    xd, wd, dyd = x.to(device), w.to(device), dy.to(device)
    torch.manual_seed(5)
    y = formats.fwd_math(xd, wd, None, **geo)
    dx = formats.bwd_igrad_math(dyd, wd, input_size=x.shape, **geo)
    dw = formats.bwd_wgrad_math(dyd, xd, weight_size=w.shape, **geo)
    return [t.cpu() for t in (y, dx, dw)]


@pytest.mark.parametrize(
    "rounding_mode", [RoundMode.RNE, RoundMode.RU, RoundMode.SR], ids=lambda m: m.name
)
@pytest.mark.parametrize("geometry", list(CONV_GEOMETRIES))
@pytest.mark.parametrize("op", list(CONV_MACS))
def test_conv_passes(op, geometry, rounding_mode):
    """Each conv op's forward, input gradient and weight gradient: the
    operand tiles gathered from the convolution's tensors on the GPU are the
    CPU's, every residue class of a strided input gradient, the groups and the
    palette maps (read at the result's own positions) included."""
    mac = CONV_MACS[op](rounding_mode)
    palette = "mixed" in op
    for ref, got in zip(
        _conv_passes(mac, geometry, "cpu", palette),
        _conv_passes(mac, geometry, "mps", palette),
        strict=True,
    ):
        assert torch.equal(got.view(torch.int32), ref.view(torch.int32))


@pytest.mark.parametrize("rounding_mode", [RoundMode.RNE, RoundMode.SR], ids=lambda m: m.name)
@pytest.mark.parametrize("acc", ["kahan", "block4", "tree8"])
@pytest.mark.parametrize("op", ["binaryK", "superfp_fma", "binaryK_superfp"])
def test_conv_accumulated(op, acc, rounding_mode):
    kw = dict(ACCUMULATIONS[acc])
    if kw["accumulate_algorithm"] is TREE and "fma" in op:
        pytest.skip("TREE sums products pairwise; a fused mac has none")
    mac = CONV_MACS[op](rounding_mode, **kw)
    for ref, got in zip(
        _conv_passes(mac, "2d", "cpu", False), _conv_passes(mac, "2d", "mps", False), strict=True
    ):
        assert torch.equal(got.view(torch.int32), ref.view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_conv_narrow_dtypes(dtype):
    """float16 and bfloat16 operands and results, through a QConv2d."""
    from mptorch.quant import QConv2d

    torch.manual_seed(0)
    q = QConv2d(4, 6, 3, stride=2, padding=1, groups=2, formats=conv_formats(SplitMac(_CB8, _CB12)))
    x = torch.randn(2, 4, 9, 8).to(dtype)
    cpu = q.to(dtype)
    ref = cpu(x)
    mps = q.to("mps")
    got = mps(x.to("mps"))
    _assert_same_words(ref.detach(), got.detach())


# --------------------------------------------------------------------------
# The block formats
# --------------------------------------------------------------------------

_SQ_NVFP4 = dataclasses.replace(NVFP4, block_rows=16)
BLOCK_FORMATS = {
    "mxfp8": MXFP8_E4M3,
    "mxfp4": MXFP4_E2M1,
    "mxfp6": MXFP6_E2M3,
    "nvfp4": NVFP4,
    "nvfp4_16x16": _SQ_NVFP4,
    "nvfp4_selective": dataclasses.replace(NVFP4, scale_rounding=ScaleRounding.SELECTIVE),
    "mxfp4_up": dataclasses.replace(MXFP4_E2M1, scale_rounding=ScaleRounding.UP),
    "fos_mse": dataclasses.replace(NVFP4, scale_rounding=ScaleRounding.FOUR_OVER_SIX_MSE),
    "fos_abs_max": dataclasses.replace(NVFP4, scale_rounding=ScaleRounding.FOUR_OVER_SIX_ABS_MAX),
    "fos_16x16_mae": dataclasses.replace(_SQ_NVFP4, scale_rounding=ScaleRounding.FOUR_OVER_SIX_MAE),
    "sfp4_elem": BlockFormat(SuperFP(1, 2, 1, 1), E8M0, 32),
    "sfp_scale": BlockFormat(MXFP4_E2M1.elem, SuperFP(3, 4, 2, 7), 16),
}


def _block_input(fmt: BlockFormat, dtype: torch.dtype) -> torch.Tensor:
    """Random words (NaNs, infinities and subnormals among them; for a cast
    scale, whose tensor scale comes from amax, exponents below 2^1), a block
    of binary32 subnormals, and blocks far enough down that their E8M0 scale
    is 2^-127 and their decoded values are subnormals of the carrier."""
    g = torch.Generator().manual_seed(8)
    w = torch.randint(-(2**31), 2**31, (3, 32, 96), generator=g).to(torch.int32)
    if fmt.has_tensor_scale:
        w = w & int(0xBFFFFFFF - (1 << 32))
    x = w.view(torch.float32)
    x[0] = torch.randn(32, 96, generator=g) * 2.0**-131
    x[1, :8] = torch.randn(8, 96, generator=g) * 2.0**-117
    return x.to(dtype)


@pytest.mark.parametrize("rounding_mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("fmt", list(BLOCK_FORMATS))
def test_block_elementwise(fmt, dtype, rounding_mode):
    """block_pack's codes and scales, block_unpack's values, block_quantize's
    and block_quantize_'s: the CPU's words, SR and the Four Over Six rules
    included, along either axis (the second read through the tensor's
    strides, as on the CPU)."""
    f = BLOCK_FORMATS[fmt]
    x = _block_input(f, dtype)
    for axis in (-1, -2):

        def ops(t, axis=axis):
            torch.manual_seed(9)
            p = block_pack(t, f, axis, rounding=rounding_mode)
            torch.manual_seed(9)
            q = block_quantize(t, f, axis, rounding=rounding_mode)
            y = t.clone()
            torch.manual_seed(9)
            block_quantize_(y, f, axis, rounding=rounding_mode)
            return [p.data, p.scales, p.unpack(), q, y]

        for ref, got in zip(ops(x), ops(x.to("mps")), strict=True):
            assert got.device.type == "mps"
            got = got.cpu().contiguous().view(torch.uint8)
            assert torch.equal(got, ref.contiguous().view(torch.uint8))


@pytest.mark.parametrize(
    "rounding_mode", [RoundMode.RNE, RoundMode.RD, RoundMode.SR], ids=lambda m: m.name
)
@pytest.mark.parametrize("kind", ["split", "fused", "unquantized", "kahan", "tree"])
@pytest.mark.parametrize(
    "pair",
    [("mxfp8", "mxfp8"), ("mxfp4", "mxfp6"), ("nvfp4_16x16", "nvfp4"), ("sfp4_elem", "sfp_scale")],
)
def test_block_gemm(pair, kind, rounding_mode):
    """custom_matmul_block on operands packed on the GPU: the CPU's words, in
    both orientations of each operand, batched, under every mac and
    accumulate algorithm."""
    fa, fb = (BLOCK_FORMATS[p] for p in pair)
    kw: dict[str, Any] = dict(acc=BinaryK(12, 7, prng_bits=8), rounding=rounding_mode)
    if kind == "fused":
        kw["fused"] = True
    if kind == "unquantized":
        kw["acc"] = None
    if kind == "kahan":
        kw["accumulate_algorithm"] = KAHAN
    if kind == "tree":
        kw.update(accumulate_algorithm=TREE, block_size=8, outer=BinaryK(16, 11, prng_bits=8))
    g = torch.Generator().manual_seed(2)
    a = torch.randn(2, 21, 80, generator=g)
    b = torch.randn(80, 19, generator=g)
    for ta, tb in [(False, False), (True, True)]:

        def run(device, ta=ta, tb=tb):
            pa = block_pack(a.to(device), fa, -2 if ta else -1)
            pb = block_pack(b.to(device), fb, -1 if tb else -2)
            torch.manual_seed(4)
            return block_matmul(pa, pb, **kw)

        ref, got = run("cpu"), run("mps")
        _assert_same_words(ref, got)


def test_block_ops_refuse_float64():
    """MPS has no float64, so binary64 has no block kernel there."""
    p = block_pack(torch.randn(4, 32, device="mps"), MXFP8_E4M3)
    with pytest.raises((RuntimeError, TypeError), match="float64"):
        p.unpack(dtype=torch.float64)
    with pytest.raises((RuntimeError, TypeError), match="float64"):
        block_matmul(p, p.mT, carrier=torch.float64)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


def test_qlinear_trains_on_mps():
    torch.manual_seed(0)
    cpu = QLinear(32, 16, formats=binaryK_gemm_formats(8, 4))
    mps = QLinear(32, 16, formats=binaryK_gemm_formats(8, 4), device="mps")
    with torch.no_grad():
        mps.weight.copy_(cpu.weight)
        mps.bias.copy_(cpu.bias)
    x = torch.randn(8, 32)
    x_cpu = x.clone().requires_grad_(True)
    x_mps = x.to("mps").requires_grad_(True)

    cpu(x_cpu).square().sum().backward()
    out = mps(x_mps)
    out.square().sum().backward()

    _assert_same_words(cpu(x_cpu).detach(), out.detach())
    assert x_cpu.grad is not None and x_mps.grad is not None
    _assert_same_words(x_cpu.grad, x_mps.grad)
    assert cpu.weight.grad is not None and mps.weight.grad is not None
    _assert_same_words(cpu.weight.grad, mps.weight.grad)


def test_not_differentiable_still_raises():
    """The Autograd kernel runs above the MPS key, so the raw ops keep refusing
    an operand that requires grad on the device too."""
    x = torch.randn(4, 4, device="mps", requires_grad=True)
    with pytest.raises(RuntimeError, match="not differentiable"):
        binaryK_quantize(x, 8, 4)


# --------------------------------------------------------------------------
# The GPU's flush of binary32 subnormals (csrc/mps/gemm.metal)
# --------------------------------------------------------------------------


def _subnormal_words() -> torch.Tensor:
    """A spread of binary32 subnormals of both signs, and the two zeros."""
    mantissas = torch.arange(0, 2**23, 1021, dtype=torch.int32)
    sign = torch.tensor(-(2**31), dtype=torch.int32)
    return torch.cat([mantissas, mantissas | sign]).view(torch.float32)


# Formats for the subnormal inputs below: binary8p4; its NORMALS floor; a
# format whose subnormal step is 2^-126, which keeps it off the binaryK fast
# path (whose magic add would see a flushed zero); one whose smallest normal,
# 2^-122, is close enough to the subnormals for SR's add to see them; a
# NORMALS floor at 2^-124, where SR's bound u * 2^-124 is mostly a
# subnormal itself; and superfp formats whose supernormal floor is 2^-111 and 2^-126.
_SUBNORMAL_CASES = {
    "binary8p4": lambda t, **kw: binaryK_quantize(t, 8, 4, **kw),
    "binary8p4-NORMALS": lambda t, **kw: binaryK_quantize(
        t, 8, 4, subnormals_mode=SubnormalsMode.NORMALS, **kw
    ),
    "binaryK-12-5-bias123": lambda t, **kw: binaryK_quantize(t, 12, 5, bias=123, **kw),
    "binaryK-12-5-bias125-NORMALS": lambda t, **kw: binaryK_quantize(
        t, 12, 5, bias=125, subnormals_mode=SubnormalsMode.NORMALS, **kw
    ),
    "superfp-3-4-1-bias7": lambda t, **kw: superfp_quantize(t, 3, 4, 1, 7, **kw),
    "superfp-3-4-1-bias22": lambda t, **kw: superfp_quantize(t, 3, 4, 1, 22, **kw),
}


@pytest.mark.parametrize("rounding_mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("case", list(_SUBNORMAL_CASES))
def test_subnormal_inputs_round_exactly(case, rounding_mode):
    """A binary32 subnormal *input* rounds as on the CPU, in every mode: the
    casts round on the word, and the few places they compare or compute on
    the value itself go through bit_helper.h's Apple-GPU spellings (the sign
    tests, SR's add into the floor's binade, SR's bound on a NORMALS floor)."""
    ref, got = _on_both(
        lambda t: _SUBNORMAL_CASES[case](t, prng_bits=18, rounding_mode=rounding_mode),
        _subnormal_words(),
    )
    _assert_same_words(ref, got)


def test_subnormal_products_flush():
    """A product whose exact value is a binary32 subnormal is zero on the GPU
    before it is rounded. Under RNE, RNA and RZ that is the CPU's answer as
    well; under RU the CPU rounds the tiny positive product up to the format's
    smallest value, and the GPU gives the zero it rounded."""
    a = torch.full((1, 1), 2.0**-70)
    b = torch.full((1, 1), 2.0**-70)  # 2^-140

    def gemm(rm):
        return _on_both(lambda x, y: binaryK_matmul(x, y, mul_K=8, mul_P=4, rounding_mode=rm), a, b)

    for rm in [RoundMode.RNE, RoundMode.RNA, RoundMode.RZ]:
        ref, got = gemm(rm)
        _assert_same_words(ref, got)
        assert ref.item() == 0.0
    ref, got = gemm(RoundMode.RU)
    assert ref.item() == 2.0**-10  # binary8p4's smallest subnormal
    assert got.item() == 0.0


def test_block_decode_of_subnormals_is_exact():
    """A decoded block value below 2^-126 (an E8M0 scale of 2^-127 times an
    element) is the CPU's subnormal, in float32 and in bfloat16, whose word
    the backend stores as a word: the GPU flushes a bfloat16 subnormal it
    forms as a value. The products are block_decode.h's flush-proof
    MPTORCH_BLOCK_MUL."""
    x = torch.linspace(-1, 1, 64).reshape(2, 32) * 2.0**-128
    p = block_pack(x, MXFP8_E4M3)
    assert (p.scales == 0).all()  # 2^-127
    for dtype in DTYPES:
        ref = p.unpack(dtype=dtype)
        got = p.to("mps").unpack(dtype=dtype)
        _assert_same_words(ref, got)
        if dtype is not torch.float16:
            assert (ref != 0).sum() > 32 and (ref.abs() < 2.0**-126).all()


def test_block_gemm_products_flush():
    """The block GEMM's products and sums are the GEMM's arithmetic, which
    the GPU flushes below 2^-126 (csrc/mps/gemm.metal): its decode is exact,
    but a product of two decoded values of 2^-70 is a zero there, which RU
    then keeps, where the CPU rounds it up to the format's smallest value."""
    a = torch.full((1, 32), 2.0**-70)
    pa, pb = block_pack(a, MXFP8_E4M3), block_pack(a.t().contiguous(), MXFP8_E4M3, 0)
    acc = BinaryK(8, 4)
    ref = block_matmul(pa, pb, acc=acc, rounding=RoundMode.RU)
    got = block_matmul(pa.to("mps"), pb.to("mps"), acc=acc, rounding=RoundMode.RU).cpu()
    assert ref.item() == 2.0**-10  # binary8p4's smallest subnormal
    assert got.item() == 0.0
