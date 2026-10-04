"""The elementwise block ops: block_pack, block_unpack, block_quantize(_).

Tier 1 holds the kernels to ``tests/block_reference.py``, an independent
spelling of the formats written from the OCP MX text and the value sets: every
(code, scale) pair of every preset decodes to the reference, and packed bytes,
scales and decoded values equal the reference's for random inputs in every
deterministic mode, for 1D blocks and 2D tiles. Around that: the OCP tables,
the bit layouts, transpose invariance of square tiles, the non-finite rule,
the fused and in-place quantizers against pack + unpack, and, for stochastic
rounding, the properties the draws must have (reproducible, independent of
the thread count, unbiased).
"""

import dataclasses

import numpy as np
import pytest
import torch

from mptorch import (
    E2M1,
    E4M3,
    E8M0,
    MXFP4_E2M1,
    MXFP6_E2M3,
    MXFP6_E3M2,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
    BinaryK,
    BlockFormat,
    RoundMode,
    ScaleRounding,
    SubnormalsMode,
    SuperFP,
)
from mptorch.quant import (
    BlockPacked,
    BlockQuant,
    Quant,
    Quantizer,
    block_pack,
    block_quantize,
    block_quantize_,
    block_unpack,
    superfp_quantize,
)
from mptorch.quant.block import _block_format_ints

from . import block_reference as ref
from .markers import float64_devices, requires_mps

PRESETS = {
    "mxfp8_e4m3": MXFP8_E4M3,
    "mxfp8_e5m2": MXFP8_E5M2,
    "mxfp6_e2m3": MXFP6_E2M3,
    "mxfp6_e3m2": MXFP6_E3M2,
    "mxfp4_e2m1": MXFP4_E2M1,
    "nvfp4": NVFP4,
}
# Block formats over SuperFP: a 4-bit element (0, 1/8 .. 1/2 supernormal, 1 ..
# 6), an 8-bit one shaped like E4M3 but reaching 2^-104 (with E4M3's NaN code
# and largest value), one with no mantissa bits, an unsigned one, and superfp
# scales with and without a mantissa (the second unsigned), under binaryK and
# superfp elements.
SFP_SCALE = SuperFP(3, 4, 2, 7)
SUPERFP = {
    "sfp4": BlockFormat(SuperFP(1, 2, 1, 1), E8M0, 32),
    "sfp8": BlockFormat(SuperFP(3, 4, 2, 7), E8M0, 32, elem_max=448.0, nan_code=0x7F),
    "sfp4_m0": BlockFormat(SuperFP(0, 3, 2, 3), E8M0, 16),
    "sfp5_unsigned": BlockFormat(SuperFP(2, 3, 2, 3, is_signed=False), E8M0, 8),
    "e2m1_sfp_scale": BlockFormat(E2M1, SFP_SCALE, 16),
    "sfp6_sfp_scale": BlockFormat(SuperFP(2, 3, 2, 3), SFP_SCALE, 16),
    "e4m3_sfp_m0_scale": BlockFormat(
        E4M3, SuperFP(0, 7, 4, 63, is_signed=False), 32, elem_max=448.0, nan_code=0x7F
    ),
}
# Each scale rule on each kind of scale besides its default: a power of two
# (under E2M1, E4M3 and E2M3, whose top steps absorb different overshoots), E4M3,
# and superfp scales with and without a mantissa.
_R = ScaleRounding
RULES = {
    f"{name}_{rule.name.lower()}": dataclasses.replace(base, scale_rounding=rule)
    for name, base, rules in (
        ("mxfp4_e2m1", MXFP4_E2M1, (_R.NEAREST, _R.UP, _R.SELECTIVE)),
        ("mxfp8_e4m3", MXFP8_E4M3, (_R.UP, _R.SELECTIVE)),
        ("mxfp6_e2m3", MXFP6_E2M3, (_R.NEAREST, _R.SELECTIVE)),
        ("nvfp4", NVFP4, (_R.UP, _R.SELECTIVE)),
        ("e2m1_sfp_scale", SUPERFP["e2m1_sfp_scale"], (_R.NEAREST, _R.UP)),
        ("sfp6_sfp_scale", SUPERFP["sfp6_sfp_scale"], (_R.NEAREST, _R.UP)),
        ("e4m3_sfp_m0_scale", SUPERFP["e4m3_sfp_m0_scale"], (_R.NEAREST,)),
    )
    for rule in rules
}
FORMATS = PRESETS | SUPERFP | RULES
DETERMINISTIC = [m for m in RoundMode if m is not RoundMode.SR]

# The kernels are CPU and CUDA; MPS raises (phase H) and is tested for that.
devices = float64_devices


def _inputs(shape, device, seed=0, spread=4.0):
    """Values over several binades, with zeros, repeats and exact grid points,
    so that every rounding mode's cases arise."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * torch.exp2(
        torch.randint(-3, 4, shape, generator=g).float()
    )
    x = x * spread
    flat = x.view(-1)
    flat[::7] = 0.0
    flat[3::11] = torch.round(flat[3::11])
    return x.to(device)


def _as3(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() == 3 else t.unsqueeze(0)


# --- Tier 1: the decode, against the reference ---------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("name", list(FORMATS))
def test_every_code_and_scale_decodes_to_the_reference(device, name):
    fmt = FORMATS[name]
    bits, bs = fmt.elem_bits, fmt.block_size
    n_codes = 1 << bits
    blocks = -(-n_codes // bs)
    codes = np.zeros((blocks * bs,), dtype=np.int64)
    codes[:n_codes] = np.arange(n_codes)
    assert fmt.scale is not None
    # every magnitude code of the scale (a stored scale's sign bit is clear)
    s_bits = ref.Layout(fmt.scale).e + ref.Layout(fmt.scale).m
    s_codes = list(range(1 << s_bits))
    ts = 0.25 if fmt.has_tensor_scale else 1.0
    rows = np.tile(codes, (len(s_codes), 1))
    data = torch.from_numpy(ref.pack_bytes(rows[None], fmt)[0]).to(device)
    scales = torch.tensor(s_codes, dtype=torch.uint8).view(-1, 1).expand(-1, blocks).contiguous()
    p = BlockPacked(data, scales.to(device), fmt, (len(s_codes), n_codes), 1, ts, torch.float32)
    got = block_unpack(p).cpu().numpy()
    elem = ref.decode_elem(fmt, np.arange(n_codes))
    for i, sc in enumerate(s_codes):
        with np.errstate(over="ignore"):  # 448 * 2^127 is past binary32, as in the kernel
            want = (elem * ref.scale_value(fmt, sc, ts)).astype(np.float32)
        want = np.where(want == 0, np.float32(0), want)
        np.testing.assert_array_equal(got[i], want)
        assert not np.any(np.signbit(got[i]) & (got[i] == 0)), "a decode returned -0.0"


def test_ocp_tables():
    """The element values OCP MX v1.0 lists, decoded by the kernel."""
    e2m1 = block_unpack(
        BlockPacked(
            torch.tensor([[0x10, 0x32, 0x54, 0x76]], dtype=torch.uint8),
            torch.tensor([[127]], dtype=torch.uint8),
            dataclasses.replace(MXFP4_E2M1, block_size=8),
            (1, 8),
            1,
            1.0,
            torch.float32,
        )
    )
    assert e2m1.tolist() == [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]]

    def one(fmt, code, dtype=torch.float32):
        bs = fmt.block_size
        row = [code] + [0] * (bs - 1)
        data = torch.from_numpy(ref.pack_bytes(np.array([[row]]), fmt)[0])
        p = BlockPacked(data, torch.tensor([[127]], dtype=torch.uint8), fmt, (1, bs), 1, 1.0, dtype)
        return block_unpack(p)[0, 0].item()

    assert one(MXFP8_E4M3, 0x7E) == 448.0
    assert np.isnan(one(MXFP8_E4M3, 0x7F)) and np.isnan(one(MXFP8_E4M3, 0xFF))
    assert one(MXFP8_E4M3, 0x01) == 2.0**-9
    assert one(MXFP8_E5M2, 0x7B) == 57344.0
    assert one(MXFP8_E5M2, 0x7C) == float("inf") and one(MXFP8_E5M2, 0xFC) == float("-inf")
    assert all(np.isnan(one(MXFP8_E5M2, c)) for c in (0x7D, 0x7E, 0x7F))
    assert one(MXFP6_E2M3, 0x1F) == 7.5 and one(MXFP6_E3M2, 0x1F) == 28.0
    assert one(MXFP8_E4M3, 0x80) == 0.0 and not np.signbit(one(MXFP8_E4M3, 0x80))


# --- Tier 1: packing, against the reference ------------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", DETERMINISTIC)
@pytest.mark.parametrize("name", list(FORMATS))
def test_pack_equals_the_reference(device, mode, name):
    fmt = FORMATS[name]
    x = _inputs((6, 70), device, seed=sum(map(ord, name)))
    p = block_pack(x, fmt, rounding=mode)
    codes, scales, values = ref.reference(x, fmt, -1, mode)
    np.testing.assert_array_equal(_as3(p.data).cpu().numpy(), ref.pack_bytes(codes, fmt))
    np.testing.assert_array_equal(_as3(p.scales).cpu().numpy(), scales)
    assert torch.equal(block_unpack(p).cpu(), values)
    assert torch.equal(block_quantize(x, fmt, rounding=mode).cpu(), values)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RD, RoundMode.RO])
@pytest.mark.parametrize(
    "fmt",
    [
        BlockFormat(BinaryK(5, 2, bias=3), E8M0, 8),  # a 5-bit stream
        BlockFormat(BinaryK(11, 6, bias=7), E8M0, 8),  # codes spanning three bytes
        BlockFormat(BinaryK(8, 4, bias=7), None, 16, nan_code=0x7F, elem_max=448.0),
        BlockFormat(E2M1, E8M0, 128),
        BlockFormat(E4M3, E8M0, 2, elem_max=448.0),
        BlockFormat(BinaryK(6, 3, is_signed=False, bias=4), E8M0, 4),
        BlockFormat(BinaryK(6, 3, bias=3, subnormals=SubnormalsMode.NORMALS), E8M0, 16),
        BlockFormat(BinaryK(6, 3, bias=3, subnormals=SubnormalsMode.EXTENDED_NORMALS), E8M0, 16),
    ],
    ids=["5bit", "11bit", "noscale", "bs128", "bs2", "unsigned", "normals", "extended"],
)
def test_layouts_and_element_formats(device, mode, fmt):
    x = _inputs((5, 45), device, seed=7)
    p = block_pack(x, fmt, rounding=mode)
    codes, scales, values = ref.reference(x, fmt, -1, mode)
    np.testing.assert_array_equal(_as3(p.data).cpu().numpy(), ref.pack_bytes(codes, fmt))
    np.testing.assert_array_equal(_as3(p.scales).cpu().numpy(), scales)
    assert torch.equal(block_unpack(p).cpu(), values)


def test_nibble_order():
    """A 4-bit code's element 0 is the low nibble (torch.float4_e2m1fn_x2's order)."""
    x = torch.tensor([[0.5, 6.0] + [0.0] * 30])
    p = block_pack(x, MXFP4_E2M1)
    assert p.data[0, 0].item() == 0x71  # 0.5 = 0b0001 low, 6.0 = 0b0111 high


# --- 2D tiles ----------------------------------------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.RU, RoundMode.RZ])
@pytest.mark.parametrize(
    "fmt,shape",
    [
        (dataclasses.replace(NVFP4, block_rows=16), (40, 50)),  # square, partial tiles
        (dataclasses.replace(MXFP8_E4M3, block_rows=2), (7, 70)),
        (dataclasses.replace(MXFP8_E4M3, block_rows=32), (3, 70, 33)),  # 3D: tiles per batch
        (dataclasses.replace(MXFP6_E3M2, block_size=128, block_rows=128), (130, 129)),
        (dataclasses.replace(MXFP4_E2M1, block_size=8, block_rows=128), (129, 17)),
    ],
    ids=["nvfp4_16x16", "mxfp8_2x32", "mxfp8_32x32_3d", "mxfp6_128x128", "mxfp4_128x8"],
)
def test_tiles_equal_the_reference(device, mode, fmt, shape):
    x = _inputs(shape, device, seed=3)
    p = block_pack(x, fmt, rounding=mode)
    codes, scales, values = ref.reference(x, fmt, -1, mode)
    np.testing.assert_array_equal(_as3(p.data).cpu().numpy(), ref.pack_bytes(codes, fmt))
    np.testing.assert_array_equal(_as3(p.scales).cpu().numpy(), scales)
    assert torch.equal(block_unpack(p).cpu(), values)
    assert torch.equal(block_quantize(x, fmt, rounding=mode).cpu(), values)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", DETERMINISTIC)
@pytest.mark.parametrize("name", ["mxfp8_e4m3", "nvfp4", "mxfp4_e2m1", "sfp8", "sfp6_sfp_scale"])
def test_square_tiles_are_transpose_invariant(device, mode, name):
    """A square tile quantizes a matrix the same along either axis; a
    non-square one does not (the control)."""
    base = FORMATS[name]
    sq = dataclasses.replace(base, block_rows=base.block_size)
    w = _inputs((70, 45), device, seed=11)
    assert torch.equal(
        block_quantize(w, sq, 0, rounding=mode), block_quantize(w, sq, 1, rounding=mode)
    )
    assert not torch.equal(
        block_quantize(w, base, 0, rounding=mode), block_quantize(w, base, 1, rounding=mode)
    )


@pytest.mark.parametrize("device", devices)
def test_whole_tile_nan(device):
    """A NaN in a format with no NaN element code makes its whole tile NaN."""
    fmt = dataclasses.replace(NVFP4, block_rows=16)
    x = torch.randn(32, 32, device=device)
    x[3, 5] = float("nan")
    q = block_quantize(x, fmt, tensor_scale=0.01)
    assert torch.isnan(q[:16, :16]).all()
    assert not torch.isnan(q[16:, :]).any() and not torch.isnan(q[:, 16:]).any()


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.SR])
def test_in_place_equals_out_of_place_on_the_largest_tile(device, mode):
    fmt = dataclasses.replace(MXFP8_E4M3, block_size=128, block_rows=128)
    x = _inputs((256, 256), device, seed=5)
    torch.manual_seed(1)
    want = block_quantize(x, fmt, rounding=mode)
    torch.manual_seed(1)
    y = x.clone()
    assert block_quantize_(y, fmt, rounding=mode) is y
    assert torch.equal(y, want)


# --- the non-finite rule and edge cases -----------------------------------------


@pytest.mark.parametrize("device", devices)
def test_non_finite_rule(device):
    inf, nan = float("inf"), float("nan")
    x = torch.tensor([[inf, -inf, 1.0, 2.0] * 8, [nan, 1.0, 2.0, 3.0] * 8], device=device)
    # E4M3 has a NaN code: the NaN keeps it. The infinities saturate to
    # elem_max, under the largest scale (an infinite amax's), 2^127, and
    # 448 * 2^127 is past binary32: they decode to infinities again.
    q = block_quantize(x, MXFP8_E4M3).cpu()
    p = block_pack(x, MXFP8_E4M3)
    assert (p.data[0, :2].cpu().tolist(), p.scales[0, 0].item()) == ([0x7E, 0xFE], 254)
    assert q[0, 0].item() == float("inf") and q[0, 1].item() == float("-inf")
    assert torch.equal(q[0], ref.reference(x[:1], MXFP8_E4M3, -1, RoundMode.RNE)[2][0])
    nans = torch.isnan(x[1]).cpu()
    assert torch.equal(torch.isnan(q[1]), nans)
    # E2M1 has none: the block's scale becomes NaN
    q = block_quantize(x, MXFP4_E2M1).cpu()
    assert torch.isnan(q[1]).all() and not torch.isnan(q[0]).any()
    # neither (no scale, no NaN code): elem_max
    fmt = BlockFormat(E2M1, None, 32)
    q = block_quantize(x, fmt).cpu()
    assert q[1, 0].item() == 6.0 and q[0, 0].item() == 6.0 and q[0, 1].item() == -6.0


@pytest.mark.parametrize("device", devices)
def test_nvfp4_tensor_scale(device):
    x = _inputs((4, 64), device, seed=2)
    # default: the tensor's amax takes the largest scale
    p = block_pack(x, NVFP4)
    assert p.tensor_scale == ref.tensor_scale_of(x, NVFP4)
    _, _, values = ref.reference(x, NVFP4, -1, RoundMode.RNE)
    assert torch.equal(p.unpack().cpu(), values)
    # a static scale, and a scale so large the blocks underflow to the smallest
    for ts in (0.125, 1e6):
        p = block_pack(x, NVFP4, tensor_scale=ts)
        _, scales, values = ref.reference(x, NVFP4, -1, RoundMode.RNE, tensor_scale=ts)
        np.testing.assert_array_equal(_as3(p.scales).cpu().numpy(), scales)
        assert torch.equal(p.unpack().cpu(), values)
    # a zero block
    z = torch.zeros(2, 32, device=device)
    assert torch.equal(block_quantize(z, NVFP4), z)
    with pytest.raises(ValueError, match="no per-tensor scale"):
        block_pack(x, MXFP8_E4M3, tensor_scale=1.0)


@pytest.mark.parametrize("device", devices)
def test_zero_blocks_and_tiny_blocks(device):
    x = torch.zeros(3, 64, device=device)
    x[1] = 2.0**-140  # a block of binary32 subnormals
    x[2, 0] = 1e30
    for fmt in (MXFP8_E4M3, MXFP4_E2M1):
        codes, scales, values = ref.reference(x, fmt, -1, RoundMode.RNE)
        p = block_pack(x, fmt)
        np.testing.assert_array_equal(_as3(p.scales).cpu().numpy(), scales)
        assert torch.equal(p.unpack().cpu(), values)
    assert _as3(block_pack(x, MXFP8_E4M3).scales)[0, 0, 0].item() == 0


# --- choosing the scale: ScaleRounding --------------------------------------------


def _blocks_at(amax: torch.Tensor, block_size: int) -> torch.Tensor:
    """One block per ``amax``: that value first, then smaller ones of both signs."""
    fill = torch.linspace(-0.95, 0.9, block_size - 1)
    return torch.cat([amax[:, None], amax[:, None] * fill], dim=1)


def _around(v: torch.Tensor) -> torch.Tensor:
    """``v`` and its float32 neighbours."""
    v = v.float()
    return torch.cat([v, torch.nextafter(v, v * 2), torch.nextafter(v, v * 0)])


def _boundary_amax(fmt: BlockFormat) -> torch.Tensor:
    """Largest magnitudes at every place a scale rule decides: where a
    power-of-two ratio amax / elem_max is 0.75, 1 or 1.5 times a power of two,
    or a cast ratio is a scale value or midway between two; and where the
    block's largest element at a scale is at elem_max or at the fit threshold
    T (SELECTIVE's bound) -- each with its float32 neighbours."""
    em, fit = fmt.elem_max, ref.fit_threshold(fmt)
    assert em is not None
    if not fmt.has_tensor_scale:
        steps = torch.exp2(torch.arange(-4.0, 5.0))
        marks = [0.75 * em, em, 1.5 * em, fit, fit / 2, 2.0**fmt.emax, 2.0 ** (fmt.emax + 1)]
        return _around(torch.cat([steps * m for m in marks]))
    assert fmt.scale is not None
    grid = ref._grid(ref.Layout(fmt.scale), fmt.scale_max, set())[0][1:]
    g = torch.from_numpy(grid)
    mids = (g[:-1] + g[1:]) / 2
    return _around(torch.cat([em * g, em * mids, fit * g]))


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize(
    "name",
    [
        "mxfp4_e2m1",
        *(f"mxfp4_e2m1_{r}" for r in ("nearest", "up", "selective")),
        "mxfp8_e4m3",
        "mxfp8_e4m3_up",
        "mxfp8_e4m3_selective",
        "mxfp6_e2m3_selective",
        "nvfp4",
        "nvfp4_up",
        "nvfp4_selective",
        "e2m1_sfp_scale",
        "e2m1_sfp_scale_nearest",
        "e2m1_sfp_scale_up",
        "sfp6_sfp_scale",
        "e4m3_sfp_m0_scale",
    ],
)
def test_scale_rules_at_their_boundaries(device, name):
    """Every rule decides the way the reference's statement of it does at
    each of its boundaries, ties and their float32 neighbours included."""
    fmt = FORMATS[name]
    x = _blocks_at(_boundary_amax(fmt), fmt.block_size).to(device)
    ts = 1.0 if fmt.has_tensor_scale else None
    p = block_pack(x, fmt, tensor_scale=ts)
    codes, scales, values = ref.reference(x, fmt, -1, RoundMode.RNE, tensor_scale=ts)
    np.testing.assert_array_equal(_as3(p.scales).cpu().numpy(), scales)
    np.testing.assert_array_equal(_as3(p.data).cpu().numpy(), ref.pack_bytes(codes, fmt))
    assert torch.equal(block_unpack(p).cpu(), values)


# Where a block's largest element lands, in E2M1's units, under each rule from a
# power-of-two scale: E8M0's, or a superfp scale's below its normal binades.
WINDOWS = {
    ScaleRounding.OCP: (4.0, True, 8.0, False),
    ScaleRounding.NEAREST: (4.5, True, 9.0, True),
    ScaleRounding.UP: (3.0, False, 6.0, True),
    ScaleRounding.SELECTIVE: (3.5, False, 7.0, True),
}


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("rule", list(ScaleRounding))
@pytest.mark.parametrize("scale", ["e8m0", "superfp"])
def test_scale_rule_windows(device, rule, scale):
    """The rules' documented windows for amax / scale, and that each is
    reached at both ends: OCP [4, 8), NEAREST [4.5, 9], UP (3, 6] and
    SELECTIVE (3.5, 7]."""
    if scale == "superfp" and rule is ScaleRounding.OCP:
        pytest.skip("OCP is a power-of-two scale's rule")
    base = MXFP4_E2M1 if scale == "e8m0" else SUPERFP["e2m1_sfp_scale"]
    fmt = dataclasses.replace(base, scale_rounding=rule)
    amax = torch.linspace(1.0, 16.0, 6001)
    ts = 1.0 if fmt.has_tensor_scale else None
    p = block_pack(_blocks_at(amax, fmt.block_size).to(device), fmt, tensor_scale=ts)
    S = torch.tensor([ref.scale_value(fmt, int(c), 1.0) for c in p.scales.flatten().cpu()])
    at = (amax / S).double()
    lo, lo_in, hi, hi_in = WINDOWS[rule]
    assert bool(((at >= lo) if lo_in else (at > lo)).all()), float(at.min())
    assert bool(((at <= hi) if hi_in else (at < hi)).all()), float(at.max())
    assert float(at.min()) < lo + 0.01 and float(at.max()) > hi - 0.01


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize(
    "name",
    [
        "mxfp4_e2m1_selective",
        "mxfp8_e4m3_selective",
        "mxfp6_e2m3_selective",
        "nvfp4_selective",
        "e2m1_sfp_scale",
        "sfp6_sfp_scale",
        "mxfp4_e2m1_up",
        "mxfp8_e4m3_up",
        "nvfp4_up",
        "e2m1_sfp_scale_up",
    ],
)
def test_selective_and_up_bound_the_clamp(device, name):
    """SELECTIVE never lets the clamp at elem_max move a block's largest
    element further than rounding it to nearest would (half the element's top
    step at its scale); UP never lets that element pass elem_max at all (to a
    float32 rounding of the quotient, for a cast scale). Over ratios inside
    the scale's range: past its largest value no rule can help."""
    fmt = FORMATS[name]
    assert fmt.elem_max is not None and fmt.scale is not None
    if fmt.has_tensor_scale:
        smin = ref._grid(ref.Layout(fmt.scale), fmt.scale_max, set())[0][1]
        lo, hi = float(np.log2(smin)) + 1, float(np.log2(fmt.scale_max)) - 1
    else:
        lo, hi = -30.0, 30.0
    g = torch.Generator().manual_seed(3)
    amax = fmt.elem_max * torch.exp2(torch.rand(4000, generator=g) * (hi - lo) + lo)
    ts = 1.0 if fmt.has_tensor_scale else None
    x = _blocks_at(amax, fmt.block_size).to(device)
    p = block_pack(x, fmt, tensor_scale=ts)
    q = block_unpack(p).cpu()[:, 0].double()
    S = torch.tensor(
        [ref.scale_value(fmt, int(c), 1.0) for c in p.scales.flatten().cpu()], dtype=torch.float64
    )
    a = amax.double()
    if fmt.scale_rule is ScaleRounding.SELECTIVE:
        assert bool(((a - q).abs() <= (ref.fit_threshold(fmt) - fmt.elem_max) * S).all())
    else:
        assert bool((a / S <= fmt.elem_max * (1 + 2.0**-22)).all())


def test_nearest_from_a_power_of_two_is_not_idempotent():
    """Why test_idempotent leaves NEAREST on a power-of-two scale out, on one
    block: 4.8 has the scale 1 (4.8 / 6 = 0.8 rounds to 1) and rounds to 4;
    4 has the scale 1/2 (4 / 6 = 0.67 rounds to 0.5), is 8 at that scale and
    saturates to 6, so it comes back 3. SELECTIVE sees that 8 is past 7 and
    keeps the scale 1."""
    x = torch.tensor([[4.8] + [0.0] * 31])
    for rule, second in ((ScaleRounding.NEAREST, 3.0), (ScaleRounding.SELECTIVE, 4.0)):
        fmt = dataclasses.replace(MXFP4_E2M1, scale_rounding=rule)
        q = block_quantize(x, fmt)
        assert q[0, 0].item() == 4.0
        assert block_quantize(q, fmt)[0, 0].item() == second


def _idempotent(fmt: BlockFormat) -> bool:
    """Whether quantizing again is expected to change nothing, on these inputs.
    A power of two scale under OCP, UP or SELECTIVE, whose largest element
    comes back at a scale those rules choose again; NVFP4, whose blocks'
    ratios stay in E4M3's normals, where E2M1's top step absorbs the scale's
    rounding; and a superfp scale under SELECTIVE, whose normal binades
    E2M1's and sfp6's top steps absorb in the same way and whose powers of two
    it treats as a power-of-two scale. Not NEAREST from a power of two (its
    largest element can round below elem_max, to a ratio that rounds a binade
    down, under which it saturates), nor UP from a cast scale (an element
    rounded down makes a ratio that rounds up to a smaller scale)."""
    pow2 = not fmt.has_tensor_scale
    rule = fmt.scale_rule
    if rule is ScaleRounding.NEAREST:
        return not pow2 and not isinstance(fmt.scale, SuperFP)
    if rule is ScaleRounding.UP:
        return pow2
    return True


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("name", [n for n, f in FORMATS.items() if _idempotent(f)])
def test_idempotent(device, name):
    """Quantizing a quantized tensor changes nothing, where ``_idempotent``
    says it should."""
    fmt = FORMATS[name]
    x = _inputs((8, 96), device)
    q = block_quantize(x, fmt, tensor_scale=0.05 if fmt.has_tensor_scale else None)
    q2 = block_quantize(q, fmt, tensor_scale=0.05 if fmt.has_tensor_scale else None)
    assert torch.equal(q, q2)


@pytest.mark.parametrize("device", devices)
def test_empty_and_odd_shapes(device):
    for shape in ((0, 32), (4, 0), (0,), (2, 0, 32)):
        x = torch.zeros(shape, device=device)
        assert block_quantize(x, MXFP8_E4M3).shape == x.shape
        assert block_unpack(block_pack(x, MXFP8_E4M3)).shape == x.shape
    x = _inputs((33,), device)
    assert torch.equal(block_unpack(block_pack(x, MXFP8_E4M3)), block_quantize(x, MXFP8_E4M3))


# --- axes, ranks, dtypes, views ---------------------------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((6, 70), 0),
        ((6, 70), 1),
        ((3, 5, 40), 1),
        ((3, 5, 40), 0),
        ((2, 3, 4, 40), 2),
        ((2, 3, 4, 40), -1),
    ],
)
def test_axes_and_ranks(device, shape, axis):
    x = _inputs(shape, device, seed=9)
    _, _, values = ref.reference(x, MXFP8_E4M3, axis, RoundMode.RNE)
    p = block_pack(x, MXFP8_E4M3, axis)
    assert p.shape == tuple(shape) and p.axis == axis % len(shape)
    assert torch.equal(p.unpack().cpu(), values)
    assert torch.equal(block_quantize(x, MXFP8_E4M3, axis).cpu(), values)
    y = x.clone()
    block_quantize_(y, MXFP8_E4M3, axis)
    assert torch.equal(y.cpu(), values)


@pytest.mark.parametrize("device", devices)
def test_a_transposed_view_packs_as_its_copy(device):
    w = _inputs((64, 96), device)
    a, b = block_pack(w.T, MXFP8_E4M3), block_pack(w.T.contiguous(), MXFP8_E4M3)
    assert torch.equal(a.data, b.data) and torch.equal(a.scales, b.scales)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name", ["mxfp8_e4m3", "mxfp4_e2m1"])
def test_narrow_dtypes(device, dtype, name):
    fmt = PRESETS[name]
    x = _inputs((4, 64), device).to(dtype)
    q = block_quantize(x, fmt)
    assert q.dtype == dtype
    _, _, values = ref.reference(x, fmt, -1, RoundMode.RNE)
    assert torch.equal(q.cpu(), values.to(dtype))
    assert torch.equal(block_unpack(block_pack(x, fmt)), q)


# --- the fused and in-place quantizers, in every mode -------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", list(RoundMode))
@pytest.mark.parametrize("name", ["mxfp8_e4m3", "nvfp4", "sfp8", "sfp6_sfp_scale", "sfp4_m0"])
def test_quantize_equals_pack_then_unpack(device, mode, name):
    base = FORMATS[name]
    prng = min(8, 23 - ref.Layout(base.elem).m)
    fmt = dataclasses.replace(base, elem=dataclasses.replace(base.elem, prng_bits=prng))
    x = _inputs((5, 80), device)
    torch.manual_seed(3)
    packed = block_unpack(block_pack(x, fmt, rounding=mode))
    torch.manual_seed(3)
    fused = block_quantize(x, fmt, rounding=mode)
    torch.manual_seed(3)
    y = x.clone()
    block_quantize_(y, fmt, rounding=mode)
    assert torch.equal(packed, fused) and torch.equal(fused, y)


# --- Tier 2: stochastic rounding ------------------------------------------------------


@pytest.mark.parametrize("device", devices)
def test_sr_reproducible_and_unbiased(device):
    fmt = dataclasses.replace(MXFP8_E4M3, elem=dataclasses.replace(E4M3, prng_bits=12))
    x = _inputs((64, 64), device)
    torch.manual_seed(5)
    a = block_quantize(x, fmt, rounding=RoundMode.SR)
    torch.manual_seed(5)
    assert torch.equal(a, block_quantize(x, fmt, rounding=RoundMode.SR))
    # unbiased: the mean of many draws approaches the scaled input
    v = torch.full((1, 32), 1.0 + 1 / 32, device=device)  # between two E4M3 values
    v[0, 0] = 1.5  # fixes the block's scale
    mean = torch.stack([block_quantize(v, fmt, rounding=RoundMode.SR) for _ in range(2000)]).mean(0)
    assert abs(mean[0, 1].item() - (1.0 + 1 / 32)) < 4e-3


def test_sr_is_thread_count_independent():
    fmt = dataclasses.replace(MXFP8_E4M3, elem=dataclasses.replace(E4M3, prng_bits=10))
    x = _inputs((128, 256), "cpu")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(9)
        one = block_quantize(x, fmt, rounding=RoundMode.SR)
        torch.set_num_threads(max(threads, 4))
        torch.manual_seed(9)
        many = block_quantize(x, fmt, rounding=RoundMode.SR)
    finally:
        torch.set_num_threads(threads)
    assert torch.equal(one, many)


# --- the value tier ----------------------------------------------------------------------


def test_block_quant_and_quant():
    x = _inputs((4, 64), "cpu")
    assert torch.equal(BlockQuant(MXFP8_E4M3)(x), block_quantize(x, MXFP8_E4M3))
    assert torch.equal(BlockQuant(MXFP8_E4M3, axis=0)(x), block_quantize(x, MXFP8_E4M3, 0))
    assert torch.equal(Quant(MXFP8_E4M3)(x), block_quantize(x, MXFP8_E4M3))
    q = BlockQuant(MXFP8_E4M3, axis=1)
    w = torch.randn(8, 64)
    first = q.pack(w)
    assert q.pack(w) is first
    w.add_(1.0)  # an in-place write bumps the version counter
    assert q.pack(w) is not first
    # the straight-through estimator
    x = torch.randn(4, 64, requires_grad=True)
    Quantizer(BlockQuant(MXFP8_E4M3), BlockQuant(MXFP4_E2M1))(x).sum().backward()
    assert x.grad is not None


def test_descriptor():
    """The descriptor's field order is BlockFmtField's (csrc/common/block_decode.h):
    the element's bits, family, sign, widths, bias, subnormals and normal
    binades, its NaN and infinity codes and random bits, then the scale's kind,
    family, sign, widths, bias, subnormals and normal binades and its rule,
    then the block."""
    assert _block_format_ints(MXFP8_E4M3) == (
        *(8, 0, 1, 4, 3, 7, 0, 0, 0x7F, -1, 0),
        *(1, 0, 0, 8, 0, 127, 2, 0, 0),
        *(32, 1),
    )
    assert _block_format_ints(NVFP4) == (
        *(4, 0, 1, 2, 1, 1, 0, 0, -1, -1, 0),
        *(2, 0, 1, 4, 3, 7, 0, 0, 1),
        *(16, 1),
    )
    assert _block_format_ints(SUPERFP["sfp6_sfp_scale"]) == (
        *(6, 1, 1, 3, 2, 3, 0, 2, -1, -1, 0),
        *(2, 1, 1, 4, 3, 7, 0, 2, 3),
        *(16, 1),
    )
    assert _block_format_ints(RULES["mxfp4_e2m1_up"])[19] == 2


def test_scale_rounding_mirrors_the_header():
    """mptorch.ScaleRounding and csrc/common/block_decode.h's ScaleRounding
    name the same rules with the same values."""
    import pathlib
    import re

    header = pathlib.Path(__file__).parents[1] / "mptorch/csrc/common/block_decode.h"
    body = re.search(r"enum class ScaleRounding : int\s*\{([^}]*)\}", header.read_text())
    assert body is not None
    pairs = dict(re.findall(r"(\w+) = (\d+)", body.group(1)))
    assert {k: int(v) for k, v in pairs.items()} == {r.name: r.value for r in ScaleRounding}


def test_superfp_codes():
    """A superfp element's codes, as SuperFP lays them out: zero, then the
    supernormal powers of two from code 1, then the normal binades."""
    fmt = SUPERFP["sfp4"]
    data = torch.from_numpy(ref.pack_bytes(np.array([[list(range(8)) * 4]]), fmt)[0])
    p = BlockPacked(
        data, torch.tensor([[127]], dtype=torch.uint8), fmt, (1, 32), 1, 1.0, torch.float32
    )
    assert block_unpack(p)[0, :8].tolist() == [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0]
    x = torch.tensor([[0.2, 0.18, 0.75, 1.5, 3.0, 5.0, -0.1, 0.05] * 4])
    # 0.75, 1.5 and 3 are ties between powers of two, which go to the power
    # with an even exponent (1, 1, 4), as superfp_quantize rounds them; 5 is a
    # tie between 4 and 6 in the normal binade, which goes to the even code, 4;
    # 0.1 is above half the smallest value, 0.05 below it
    q = block_quantize(x, fmt)[0, :8].tolist()
    assert q == [0.25, 0.125, 1.0, 1.0, 4.0, 4.0, -0.125, 0.0]
    assert q[:6] == superfp_quantize(x[0, :6], 1, 2, 1, 1).tolist()


# --- what is refused -------------------------------------------------------------------------


def test_format_validation():
    with pytest.raises(ValueError, match="power of two"):
        BlockFormat(E4M3, E8M0, 24)
    with pytest.raises(ValueError, match="whole bytes"):
        BlockFormat(BinaryK(5, 2, bias=3), E8M0, 4)
    with pytest.raises(ValueError, match="block_rows"):
        BlockFormat(E4M3, None, 32, block_rows=2)
    with pytest.raises(ValueError, match="16384"):
        BlockFormat(E4M3, E8M0, 128, block_rows=256)
    with pytest.raises(ValueError, match="elem_max"):
        BlockFormat(E4M3, E8M0, 32, elem_max=447.0)
    with pytest.raises(ValueError, match="nan_code"):
        BlockFormat(E4M3, E8M0, 32, nan_code=0x7E)  # the code of 448, a value
    with pytest.raises(ValueError, match="binary32"):
        BlockFormat(BinaryK(16, 3), E8M0, 32)
    with pytest.raises(ValueError, match="unsigned"):
        BlockFormat(E4M3, BinaryK(8, 1, bias=127), 32)
    with pytest.raises(TypeError, match="phase K"):
        BlockFormat(1, E8M0, 32)  # type: ignore
    with pytest.raises(ValueError, match="binary32 holds entirely"):
        BlockFormat(SuperFP(3, 5, 1, 7), E8M0, 32)  # supernormals down to 2^-240
    with pytest.raises(ValueError, match="one byte"):
        BlockFormat(E4M3, SuperFP(4, 4, 2, 7), 32)  # a 9-bit scale
    with pytest.raises(ValueError, match="nan_code"):
        BlockFormat(SuperFP(3, 4, 2, 7), E8M0, 32, nan_code=0x7F)  # 0x7F is 480, a value
    with pytest.raises(ValueError, match="OCP MX's rule for a power-of-two scale"):
        BlockFormat(E2M1, E4M3, 16, scale_rounding=ScaleRounding.OCP)
    with pytest.raises(ValueError, match="needs a scale"):
        BlockFormat(E4M3, None, 32, scale_rounding=ScaleRounding.NEAREST)
    with pytest.raises(TypeError, match="ScaleRounding"):
        BlockFormat(E2M1, E8M0, 32, scale_rounding="up")  # type: ignore
    # the pack multiplies by the reciprocal of a power-of-two scale, so both
    # must be binary32 values: 2^-127 .. 2^127
    low = BinaryK(8, 1, bias=128, is_signed=False, subnormals=SubnormalsMode.EXTENDED_NORMALS)
    with pytest.raises(ValueError, match=r"2\^-127 \.\. 2\^127"):
        BlockFormat(E2M1, low, 32)


@pytest.mark.parametrize("device", devices)
def test_binary32_only(device):
    x = torch.randn(4, 32, device=device)
    with pytest.raises(ValueError, match="phase G"):
        block_pack(x.double(), MXFP8_E4M3)
    with pytest.raises(ValueError, match="phase G"):
        block_quantize(x, MXFP8_E4M3, carrier=torch.float64)


def test_requires_grad_raises():
    x = torch.randn(4, 32, requires_grad=True)
    with pytest.raises(RuntimeError, match="not differentiable"):
        torch.ops.mptorch.block_quant.default(
            x, list(_block_format_ints(MXFP8_E4M3)), 448.0, 2.0**127, 1.0, 0
        )


def test_raw_op_validates_its_descriptor():
    x = torch.randn(4, 32)
    bad = list(_block_format_ints(MXFP8_E4M3))
    bad[20] = 24  # block_size
    with pytest.raises(RuntimeError, match="block_size"):
        torch.ops.mptorch.block_pack.default(x, bad, 448.0, 2.0**127, 1.0, 0)
    with pytest.raises(RuntimeError, match="elem_max"):
        torch.ops.mptorch.block_pack.default(
            x, list(_block_format_ints(MXFP8_E4M3)), 449.0, 2.0**127, 1.0, 0
        )
    sfp = list(_block_format_ints(SUPERFP["sfp8"]))
    sfp[7] = 16  # normal_binades: every binade normal, no supernormal code left
    with pytest.raises(RuntimeError, match="superfp layout"):
        torch.ops.mptorch.block_pack.default(x, sfp, 448.0, 2.0**127, 1.0, 0)
    for rule in (4, -1):
        bad = list(_block_format_ints(MXFP8_E4M3))
        bad[19] = rule
        with pytest.raises(RuntimeError, match="scale_rounding"):
            torch.ops.mptorch.block_pack.default(x, bad, 448.0, 2.0**127, 1.0, 0)
    nv = list(_block_format_ints(NVFP4))
    nv[19] = 0  # OCP on a cast scale
    with pytest.raises(RuntimeError, match="scale_rounding"):
        torch.ops.mptorch.block_pack.default(x, nv, 6.0, 448.0, 1.0, 0)


@requires_mps
def test_mps_raises():
    x = torch.randn(4, 32, device="mps")
    with pytest.raises(RuntimeError, match="phase H"):
        block_quantize(x, MXFP8_E4M3)


# --- a second OCP oracle: gfloat ------------------------------------------------------


@pytest.mark.parametrize("mode", DETERMINISTIC)
@pytest.mark.parametrize(
    "name", ["mxfp8_e4m3", "mxfp8_e5m2", "mxfp6_e2m3", "mxfp6_e3m2", "mxfp4_e2m1"]
)
def test_mx_presets_match_gfloat(mode, name):
    """The MX presets against gfloat's quantize_block (the pinned commit in
    requirements-test.txt), an OCP implementation of its own, in every
    deterministic mode. The first row puts values between elem_max and
    2^(emax + 1) in a block (risk R4): both saturate them to elem_max."""
    gfloat = pytest.importorskip("gfloat")
    from gfloat import formats as gf
    from gfloat.types import RoundMode as GR

    infos = {
        "mxfp8_e4m3": gf.format_info_mxfp8_e4m3,
        "mxfp8_e5m2": gf.format_info_mxfp8_e5m2,
        "mxfp6_e2m3": gf.format_info_mxfp6_e2m3,
        "mxfp6_e3m2": gf.format_info_mxfp6_e3m2,
        "mxfp4_e2m1": gf.format_info_mxfp4_e2m1,
    }
    modes = {
        RoundMode.RNE: GR.TiesToEven,
        RoundMode.RNA: GR.TiesToAway,
        RoundMode.RZ: GR.TowardZero,
        RoundMode.RU: GR.TowardPositive,
        RoundMode.RD: GR.TowardNegative,
        RoundMode.RO: getattr(GR, "ToOdd", None),
    }
    if modes[mode] is None:
        pytest.skip("this gfloat has no ToOdd")
    fmt, fi = PRESETS[name], infos[name]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 32, generator=g) * torch.exp2(
        torch.randint(-6, 6, (64, 1), generator=g).float()
    )
    x[0] = torch.linspace(-(2.0 ** (fmt.emax + 1)) * 0.999, 2.0 ** (fmt.emax + 1) * 0.999, 32)
    ours = block_quantize(x, fmt, rounding=mode).double().numpy()
    theirs = np.stack(
        [
            gfloat.quantize_block(fi, x[i].double().numpy(), gfloat.compute_scale_amax, modes[mode])
            for i in range(x.shape[0])
        ]
    )
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize(
    "name", ["mxfp8_e4m3", "mxfp8_e5m2", "mxfp6_e2m3", "mxfp6_e3m2", "mxfp4_e2m1"]
)
def test_mx_up_rule_matches_gfloat(name):
    """ScaleRounding.UP on the MX presets against gfloat's quantize_block given
    the rule NVIDIA's MXFP8 recipe states (arXiv 2506.08027): the scale is
    amax / elem_max rounded up to a power of two, clamped to E8M0's codes."""
    gfloat = pytest.importorskip("gfloat")
    from gfloat import formats as gf

    infos = {
        "mxfp8_e4m3": gf.format_info_mxfp8_e4m3,
        "mxfp8_e5m2": gf.format_info_mxfp8_e5m2,
        "mxfp6_e2m3": gf.format_info_mxfp6_e2m3,
        "mxfp6_e3m2": gf.format_info_mxfp6_e3m2,
        "mxfp4_e2m1": gf.format_info_mxfp4_e2m1,
    }
    fmt = dataclasses.replace(PRESETS[name], scale_rounding=ScaleRounding.UP)
    assert fmt.elem_max is not None
    em = fmt.elem_max

    def up(emax, vals):
        amax = float(np.max(np.abs(vals)))
        if amax == 0.0:
            return 2.0**-127
        return 2.0 ** float(np.clip(np.ceil(np.log2(amax / em)), -127, 127))

    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 32, generator=g) * torch.exp2(
        torch.randint(-6, 6, (64, 1), generator=g).float()
    )
    x[0] = torch.linspace(-(2.0 ** (fmt.emax + 1)) * 0.999, 2.0 ** (fmt.emax + 1) * 0.999, 32)
    ours = block_quantize(x, fmt).double().numpy()
    theirs = np.stack(
        [gfloat.quantize_block(infos[name], x[i].double().numpy(), up) for i in range(64)]
    )
    np.testing.assert_array_equal(ours, theirs)
