"""Block formats in binary64: the elementwise ops and the block GEMM.

A float64 tensor rounds in binary64, and so does a call that names
``carrier=torch.float64`` on a narrower one, which is widened first and
narrowed back once. Tier 1 holds the binary64 kernels to
``tests/block_reference.py`` run in float64 (``F=np.float64``): every code and
scale decodes to it, and packed bytes, scales and decoded values equal it in
every deterministic mode, for the formats ``test_block_pack`` holds in
binary32 and for formats only binary64 holds (an element with eight to ten
exponent bits, supernormals below binary32's range, a power-of-two scale
biased past 2^-127, a cast scale below binary32's normals). The block GEMM is
held, ``torch.equal``, to ``test_accumulators``' references over the operands
unpacked in binary64: products in the carrier, sums in the accumulate format.
Then the carrier's rules: the widened call is the float64 call, SR included;
a format only binary64 holds is refused in binary32 with the finding that
says so; a carrier is never narrower than what it reads.
"""

import dataclasses

import numpy as np
import pytest
import torch

from mptorch import (
    E2M1,
    E8M0,
    MXFP4_E2M1,
    MXFP8_E4M3,
    NVFP4,
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    RoundMode,
    ScaleRounding,
    SubnormalsMode,
    SuperFP,
)
from mptorch.quant import (
    BlockMac,
    BlockPacked,
    BlockQuant,
    QLinear,
    block_gemm_formats,
    block_matmul,
    block_pack,
    block_quantize,
    block_quantize_,
    block_unpack,
    qmatmul,
)
from mptorch.quant.ops import _narrowed

from . import block_reference as ref
from .markers import float64_devices
from .test_accumulators import _Reference
from .test_block_pack import FORMATS, FOUR_OVER_SIX_TILES, PRESETS, SUPERFP, _as3, _inputs

devices = float64_devices
DETERMINISTIC = [m for m in RoundMode if m is not RoundMode.SR]
F64 = torch.float64
_EXT = SubnormalsMode.EXTENDED_NORMALS

# Formats only binary64 holds. Elements with eight, nine and ten exponent bits
# (binary32 holds seven), an E8M0-style element (powers of two from 2^-127),
# a superfp element whose supernormals reach 2^-240, a power-of-two scale
# biased one past binary32's 2^-127 and one far below it, and an E4M3-shaped
# cast scale whose values lie near 2^-290, below binary32's normals.
WIDE = {
    "e8m3_elem": BlockFormat(BinaryK(12, 4), E8M0, 32),
    "e9m2_elem": BlockFormat(BinaryK(12, 3), E8M0, 32),
    "e10m5_elem": BlockFormat(BinaryK(16, 6), E8M0, 16),
    "e8m0_elem": BlockFormat(BinaryK(8, 1, bias=127, is_signed=False, subnormals=_EXT), E8M0, 32),
    "sfp_low_elem": BlockFormat(SuperFP(3, 5, 1, 7), E8M0, 32),
    "pow2_scale_past": BlockFormat(
        E2M1, BinaryK(8, 1, bias=128, is_signed=False, subnormals=_EXT), 32
    ),
    "pow2_scale_far": BlockFormat(
        E2M1, BinaryK(8, 1, bias=600, is_signed=False, subnormals=_EXT), 32
    ),
    "cast_scale_low": BlockFormat(E2M1, BinaryK(8, 4, bias=300), 16),
}
# where each wide format's inputs sit, so that its blocks reach its range
WIDE_SPREAD = {"pow2_scale_far": 2.0**-450}
# the wide formats the block GEMM takes (element codes of at most 8 bits)
WIDE_GEMM = ["e8m0_elem", "pow2_scale_past", "pow2_scale_far", "cast_scale_low"]


def _inputs64(shape, device, seed=0, scale=1.0):
    """``_inputs``' values with their low 29 bits filled, so that they are
    float64 values binary32 does not hold, times ``scale``."""
    x = _inputs(shape, "cpu", seed).double()
    g = torch.Generator().manual_seed(seed + 1)
    x = x * (1 + torch.rand(shape, generator=g, dtype=F64) * 2.0**-30)
    return (x * scale).to(device)


# --- Tier 1: the decode -----------------------------------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("name", list(PRESETS | SUPERFP) + ["cast_scale_low", "pow2_scale_past"])
def test_every_code_and_scale_decodes_to_the_reference(device, name):
    """Every (element code, scale code) pair, decoded in binary64, with a
    tensor scale that is no float32 value."""
    fmt = (FORMATS | WIDE)[name]
    bits, bs = fmt.elem_bits, fmt.block_size
    n_codes = 1 << bits
    blocks = -(-n_codes // bs)
    codes = np.zeros((blocks * bs,), dtype=np.int64)
    codes[:n_codes] = np.arange(n_codes)
    assert fmt.scale is not None
    s_bits = ref.Layout(fmt.scale).e + ref.Layout(fmt.scale).m
    s_codes = list(range(1 << s_bits))
    ts = 0.25 * (1 + 2.0**-40) if fmt.has_tensor_scale else 1.0
    rows = np.tile(codes, (len(s_codes), 1))
    data = torch.from_numpy(ref.pack_bytes(rows[None], fmt)[0]).to(device)
    scales = torch.tensor(s_codes, dtype=torch.uint8).view(-1, 1).expand(-1, blocks).contiguous()
    p = BlockPacked(data, scales.to(device), fmt, (len(s_codes), n_codes), 1, ts, F64)
    assert p.carrier is F64
    got = block_unpack(p).cpu().numpy()
    assert got.dtype == np.float64
    elem = ref.decode_elem(fmt, np.arange(n_codes))
    for i, sc in enumerate(s_codes):
        with np.errstate(over="ignore"):
            want = elem * ref.scale_value(fmt, sc, ts, np.float64)
        want = np.where(want == 0, 0.0, want)
        np.testing.assert_array_equal(got[i], want)
        assert not np.any(np.signbit(got[i]) & (got[i] == 0)), "a decode returned -0.0"


# --- Tier 1: packing --------------------------------------------------------------


def _check_pack(fmt, x, mode, axis=-1):
    p = block_pack(x, fmt, axis, rounding=mode)
    assert p.carrier is F64
    codes, scales, values = ref.reference(x, fmt, axis, mode, F=np.float64)
    np.testing.assert_array_equal(_as3(p.data).cpu().numpy(), ref.pack_bytes(codes, fmt))
    np.testing.assert_array_equal(_as3(p.scales).cpu().numpy(), scales)
    unpacked = block_unpack(p).cpu()
    assert unpacked.dtype is F64
    assert torch.equal(unpacked, values)
    assert torch.equal(block_quantize(x, fmt, axis, rounding=mode).cpu(), values)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", DETERMINISTIC)
@pytest.mark.parametrize("name", list(FORMATS))
def test_pack_equals_the_reference(device, mode, name):
    """The formats binary32 holds, packed from float64 tensors in binary64."""
    x = _inputs64((6, 70), device, seed=sum(map(ord, name)))
    _check_pack(FORMATS[name], x, mode)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", DETERMINISTIC)
@pytest.mark.parametrize("name", list(WIDE))
def test_wide_formats_equal_the_reference(device, mode, name):
    """Formats only binary64 holds."""
    x = _inputs64((6, 70), device, seed=len(name), scale=WIDE_SPREAD.get(name, 1.0))
    _check_pack(WIDE[name], x, mode)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RU, RoundMode.RZ])
@pytest.mark.parametrize(
    "fmt",
    [
        dataclasses.replace(MXFP8_E4M3, block_rows=32),
        dataclasses.replace(NVFP4, block_rows=16),
        dataclasses.replace(WIDE["cast_scale_low"], block_rows=4),
        *FOUR_OVER_SIX_TILES.values(),
    ],
    ids=lambda f: f"{f.elem}_{f.block_rows}x{f.block_size}_{f.scale_rule.name}",
)
def test_tiles_equal_the_reference(device, mode, fmt):
    x = _inputs64((2, 37, 70), device, seed=fmt.block_rows)
    _check_pack(fmt, x, mode)


# --- the carrier's rules ------------------------------------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RD, RoundMode.SR])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "name", ["mxfp8_e4m3", "nvfp4", "sfp8", "nvfp4_fos_mse", "e2m1_e4m0_scale"]
)
def test_widened_carrier_is_the_float64_call(device, mode, dtype, name):
    """``carrier=torch.float64`` on a narrower tensor packs, unpacks and
    quantizes as the float64 call on the widened tensor, SR draws included,
    and narrows a decoded result back once."""
    fmt = FORMATS[name]
    x = _inputs((5, 64), device, seed=3).to(dtype)

    def seeded(fn):
        torch.manual_seed(13)
        return fn()

    wide = seeded(lambda: block_pack(x, fmt, rounding=mode, carrier=F64))
    plain = seeded(lambda: block_pack(x.double(), fmt, rounding=mode))
    assert wide.carrier is F64 and wide.dtype is dtype
    assert torch.equal(wide.data, plain.data) and torch.equal(wide.scales, plain.scales)
    assert wide.tensor_scale == plain.tensor_scale
    if fmt.has_tensor_scale or dtype is torch.float32:
        # a decoded value is a binary64 product, narrowed once
        assert torch.equal(block_unpack(wide), _narrowed(block_unpack(plain), dtype))
        got = seeded(lambda: block_quantize(x, fmt, rounding=mode, carrier=F64))
        want = _narrowed(seeded(lambda: block_quantize(x.double(), fmt, rounding=mode)), dtype)
        assert got.dtype is dtype and torch.equal(got, want)


def test_tensor_scale_in_binary64():
    """NVFP4's default tensor scale is a float64 quotient in binary64, and a
    static one keeps all of its bits there."""
    x = _inputs64((4, 64), "cpu", seed=2)
    p = block_pack(x, NVFP4)
    amax = x.abs().max().item()
    assert p.tensor_scale == amax / (6.0 * 448.0)
    ts = 1.0 + 2.0**-40
    assert block_pack(x, NVFP4, tensor_scale=ts).tensor_scale == ts
    assert block_pack(x.float(), NVFP4, tensor_scale=ts).tensor_scale == 1.0


@pytest.mark.parametrize("name", list(WIDE))
def test_wide_formats_are_refused_in_binary32(name):
    """A format only binary64 holds is built, runs on float64 tensors and
    with ``carrier=torch.float64``, and is refused in binary32, naming
    binary64."""
    fmt = WIDE[name]
    x = _inputs((4, 64), "cpu", seed=1) * WIDE_SPREAD.get(name, 1.0)
    for call in (
        lambda: block_pack(x, fmt),
        lambda: block_quantize(x, fmt),
        lambda: block_quantize(x.half(), fmt),
    ):
        with pytest.raises(ValueError, match="binary64 holds it"):
            call()
    p = block_pack(x, fmt, carrier=F64)
    assert p.carrier is F64 and p.dtype is torch.float32
    assert block_unpack(p).dtype is torch.float32
    with pytest.raises(ValueError, match="narrower than binary64"):
        block_unpack(p, carrier=torch.float32)


def test_carrier_is_never_narrower():
    x = _inputs64((4, 64), "cpu")
    with pytest.raises(ValueError, match="narrower"):
        block_pack(x, MXFP8_E4M3, carrier=torch.float32)
    p = block_pack(x, MXFP8_E4M3)
    with pytest.raises(ValueError, match="narrower than binary64"):
        block_unpack(p, carrier=torch.float32)
    pa = block_pack(_inputs((4, 64), "cpu"), MXFP8_E4M3)
    with pytest.raises(ValueError, match="narrower than binary64"):
        block_matmul(pa, p.mT, carrier=torch.float32)
    with pytest.raises(ValueError, match="carrier=torch.float64"):
        block_quantize_(_inputs((4, 64), "cpu"), MXFP8_E4M3, carrier=F64)


@pytest.mark.parametrize("device", devices)
def test_in_place_equals_out_of_place(device):
    x = _inputs64((8, 64), device, seed=4)
    want = block_quantize(x, NVFP4)
    y = x.clone()
    assert block_quantize_(y, NVFP4) is y
    assert torch.equal(y, want)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("name", ["mxfp8_e4m3", "nvfp4", "nvfp4_fos_mae"])
def test_sr_is_seeded_and_keyed_on_the_element(device, name):
    """In binary64 an element's draw is two Philox words: one seed one
    result, another seed another, and a row's codes do not depend on the
    rows after it."""
    fmt = FORMATS[name]
    sr = dataclasses.replace(fmt, elem=dataclasses.replace(fmt.elem, prng_bits=30))
    x = _inputs64((9, 64), device, seed=8)

    def run(t, seed):
        torch.manual_seed(seed)
        ts = 1.0 if fmt.has_tensor_scale else None
        return block_pack(t, sr, rounding=RoundMode.SR, tensor_scale=ts)

    first = run(x, 1)
    assert torch.equal(first.data, run(x, 1).data)
    assert not torch.equal(first.data, run(x, 2).data)
    assert torch.equal(first.data[:4], run(x[:4], 1).data)


def test_sr_is_independent_of_the_thread_count():
    fmt = dataclasses.replace(NVFP4, elem=dataclasses.replace(E2M1, prng_bits=30))
    x = _inputs64((70, 128), "cpu", seed=5)
    before = torch.get_num_threads()
    try:
        out = []
        for threads in (1, 4):
            torch.set_num_threads(threads)
            torch.manual_seed(3)
            out.append(block_quantize(x, fmt, rounding=RoundMode.SR, tensor_scale=1.0))
    finally:
        torch.set_num_threads(before)
    assert torch.equal(out[0], out[1])


# --- the block GEMM ------------------------------------------------------------------


def _gemm_reference(pa, pb, mode, fused, acc, algorithm=AccumulateAlgorithm.NAIVE, **kw):
    """``test_accumulators``' reference over the operands unpacked in
    binary64: each product in the carrier (or one fused multiply-add), each
    sum rounded to ``acc``."""
    a, b = block_unpack(pa, dtype=F64).cpu(), block_unpack(pb, dtype=F64).cpu()
    r = _Reference(a, b, None if fused else BinaryK(8, 4), acc, kw.get("outer"), mode)
    if not fused:
        r.mul = lambda v: v  # the block GEMM's multiply is the carrier's own
    return r.run(algorithm, kw.get("block_size"))


GEMM_PAIRS = {
    "mxfp8": (MXFP8_E4M3, MXFP8_E4M3),
    "mxfp4_nvfp4": (MXFP4_E2M1, NVFP4),
    "sfp8_sfp4": (FORMATS["sfp8"], FORMATS["sfp4"]),
    "tiled_fos": (
        dataclasses.replace(NVFP4, block_rows=16, scale_rounding=ScaleRounding.FOUR_OVER_SIX_MSE),
        FORMATS["e2m1_e5m2_scale"],
    ),
    **{name: (WIDE[name], WIDE[name]) for name in WIDE_GEMM},
}
GEMM_ACCS = {"b8p4": BinaryK(8, 4), "b32p27": BinaryK(32, 27), "none": None}


def _gemm_ops(fa, fb, device, seed, spread=1.0):
    a = _inputs64((7, 70), device, seed=seed, scale=spread)
    b = _inputs64((70, 9), device, seed=seed + 1, scale=spread)
    return block_pack(a, fa), block_pack(b, fb, 0)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", DETERMINISTIC)
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("acc", list(GEMM_ACCS))
@pytest.mark.parametrize("pair", list(GEMM_PAIRS))
def test_gemm_equals_the_reference(device, mode, fused, acc, pair):
    fa, fb = GEMM_PAIRS[pair]
    spread = WIDE_SPREAD.get(pair, 1.0)
    pa, pb = _gemm_ops(fa, fb, device, seed=len(pair), spread=spread)
    got = block_matmul(pa, pb, acc=GEMM_ACCS[acc], fused=fused, rounding=mode)
    assert got.dtype is F64
    assert torch.equal(got.cpu(), _gemm_reference(pa, pb, mode, fused, GEMM_ACCS[acc]))


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RZ])
@pytest.mark.parametrize(
    "alg,block_size,outer,fused",
    [
        (AccumulateAlgorithm.KAHAN, None, None, False),
        (AccumulateAlgorithm.KAHAN, None, None, True),
        (AccumulateAlgorithm.BLOCK, 8, BinaryK(16, 11), False),
        (AccumulateAlgorithm.BLOCK, 32, None, True),
        (AccumulateAlgorithm.TREE, 16, BinaryK(16, 11), False),
    ],
)
def test_gemm_accumulate_algorithms_equal_the_reference(
    device, mode, alg, block_size, outer, fused
):
    pa, pb = _gemm_ops(MXFP8_E4M3, NVFP4, device, seed=6)
    acc = BinaryK(12, 7)
    extra = dict(accumulate_algorithm=alg, block_size=block_size, outer=outer)
    got = block_matmul(pa, pb, acc=acc, fused=fused, rounding=mode, **extra)
    want = _gemm_reference(pa, pb, mode, fused, acc, alg, block_size=block_size, outer=outer)
    assert torch.equal(got.cpu(), want)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("fused", [False, True])
def test_gemm_widened_carrier(device, fused):
    """Operands packed in binary32, multiplied with ``carrier=torch.float64``:
    the binary64 GEMM of the same codes, decoded in binary64."""
    g = torch.Generator().manual_seed(9)
    pa = block_pack(torch.randn(7, 64, generator=g).to(device), NVFP4)
    pb = block_pack(torch.randn(64, 9, generator=g).to(device), MXFP8_E4M3, 0)
    assert pa.carrier is torch.float32
    acc = BinaryK(16, 11)
    got = block_matmul(pa, pb, acc=acc, fused=fused, carrier=F64)
    assert got.dtype is F64
    assert torch.equal(got.cpu(), _gemm_reference(pa, pb, RoundMode.RNE, fused, acc))
    # a binary64 operand takes the other one with it
    pc = block_pack(torch.randn(64, 9, generator=g).double().to(device), MXFP8_E4M3, 0)
    assert block_matmul(pa, pc, acc=acc).dtype is F64


@pytest.mark.parametrize("device", devices)
def test_gemm_tensor_scale_epilogue(device):
    """The epilogue multiplies a binary64 result by the float64 product of the
    tensor scales, once."""
    pa, pb = _gemm_ops(NVFP4, NVFP4, device, seed=11)

    def unit(p):
        return BlockPacked(p.data, p.scales, p.fmt, p.shape, p.axis, 1.0, p.dtype, p.carrier)

    got = block_matmul(pa, pb, acc=BinaryK(16, 11), tensor_scale_epilogue=True)
    loop = block_matmul(unit(pa), unit(pb), acc=BinaryK(16, 11))
    assert torch.equal(got, loop * (pa.tensor_scale * pb.tensor_scale))


@pytest.mark.parametrize("device", devices)
def test_gemm_sr_is_seeded(device):
    acc = BinaryK(16, 11, prng_bits=30)
    pa, pb = _gemm_ops(MXFP8_E4M3, MXFP8_E4M3, device, seed=12)

    def run(seed):
        torch.manual_seed(seed)
        return block_matmul(pa, pb, acc=acc, rounding=RoundMode.SR)

    assert torch.equal(run(1), run(1))
    assert not torch.equal(run(1), run(2))


# --- layers -------------------------------------------------------------------------


@pytest.mark.parametrize("device", devices)
def test_a_float64_layer_runs_in_binary64(device):
    """``block_gemm_formats`` on a float64 ``QLinear``: every pass packs and
    multiplies in binary64, and the forward is the binary64 block GEMM."""
    torch.manual_seed(0)
    layer = QLinear(64, 16, bias=False, formats=block_gemm_formats(NVFP4, MXFP8_E4M3)).to(
        device=device, dtype=F64
    )
    x = _inputs64((8, 64), device, seed=3).requires_grad_()
    y = layer(x)
    assert y.dtype is F64
    w = block_pack(layer.weight.detach(), MXFP8_E4M3)
    want = block_matmul(block_pack(x.detach(), NVFP4), w.mT)
    assert torch.equal(y.detach(), want)
    y.sum().backward()
    assert x.grad is not None and x.grad.dtype is F64
    assert layer.weight.grad is not None and layer.weight.grad.dtype is F64


@pytest.mark.parametrize("device", devices)
def test_a_block_mac_with_the_binary64_carrier(device):
    """``BlockMac(carrier=torch.float64)`` on float32 operands packs them in
    binary64, multiplies there, and narrows the product once."""
    g = torch.Generator().manual_seed(4)
    a = torch.randn(6, 64, generator=g).to(device)
    b = torch.randn(64, 5, generator=g).to(device)
    mac = BlockMac(NVFP4, MXFP8_E4M3, acc=BinaryK(16, 11), carrier=F64)
    assert isinstance(mac.a, BlockQuant) and isinstance(mac.b, BlockQuant)
    assert mac.a.carrier is F64 and mac.b.carrier is F64
    got = qmatmul(a, b, mac)
    pa = block_pack(a.double(), NVFP4)
    pb = block_pack(b.double(), MXFP8_E4M3, 0)
    want = _narrowed(block_matmul(pa, pb, acc=BinaryK(16, 11)), torch.float32)
    assert got.dtype is torch.float32 and torch.equal(got, want)
