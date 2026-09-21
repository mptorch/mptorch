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

What the backend cannot match is the GPU's flush of binary32 subnormals in
its arithmetic (``csrc/mps/gemm.metal``), and the last tests pin both what
that flush changes and what it does not.

No tier convention: there is no MPS arithmetic of its own to test, only the
CPU's compiled for another device, which the CPU backend's tests hold to the
formats.
"""

import math
from typing import Any

import pytest
import torch

from mptorch.number import RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import QLinear, binaryK_gemm_formats, binaryK_quantize, superfp_quantize
from mptorch.quant.ops import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_fma_mixed,
    binaryK_matmul_mixed,
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
    (8, 4, None, True, SaturationMode.OVF_INF, SubnormalsMode.SUBNORMALS),
    (8, 3, None, True, SaturationMode.SAT_FINITE, SubnormalsMode.SUBNORMALS),
    (8, 5, None, False, SaturationMode.SAT_PROPAGATE, SubnormalsMode.SUBNORMALS),
    (8, 1, None, True, SaturationMode.OVF_INF, SubnormalsMode.SUBNORMALS),
    (8, 4, None, True, SaturationMode.OVF_INF, SubnormalsMode.NORMALS),
    (8, 4, None, True, SaturationMode.SAT_FINITE, SubnormalsMode.EXTENDED_NORMALS),
    (16, 12, None, True, SaturationMode.OVF_INF, SubnormalsMode.SUBNORMALS),
]

# (man_bits, exp_bits, normal_binades, bias, is_signed, saturation). The last
# has so many supernormal binades that nothing underflows at all.
SUPERFP = [
    (3, 4, 1, 7, True, SaturationMode.OVF_INF),
    (2, 5, 2, 15, True, SaturationMode.SAT_FINITE),
    (0, 4, 1, 7, True, SaturationMode.OVF_INF),
    (4, 3, 1, 3, False, SaturationMode.SAT_PROPAGATE),
    (9, 3, 1, 3, True, SaturationMode.OVF_INF),
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
