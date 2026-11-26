"""
IEEE P3109's block operations (section VIII of arXiv:2606.04028), and mptorch's two reductions.

- **block_project** (the op): onto the scales ``block_pack`` chose, it is
  ``block_pack`` word for word in every format, every rounding mode (SR
  under the same seed), both carriers and every layout; onto other scale
  codes, it is an independent reference built from P3109's cases and
  ``block_reference.py``'s value-set rounding. Then the cases one by one: a
  NaN scale, a zero scale, a NaN the format cannot store, a
  format without a scale, and the checks of the scale codes.
- **block_map, block_exp, block_scaled_add**: the decode, the function in
  the carrier, and the projection, against the ops spelled out by hand.
- **convert_to_block_max_abs_finite**: the scale codes against a ``Fraction``
  reference of "the largest finite magnitude, rounded into the scale format".
- **block_reduce_add, block_dot**: against sums and dot products written out
  step by step.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from mptorch import (
    E2M1,
    E4M3,
    E8M0,
    MXFP4_E2M1,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
    BinaryK,
    BlockFormat,
    RoundMode,
    SuperFP,
)
from mptorch.number.block import _minifloat, _minifloat_value
from mptorch.quant import (
    FusedMac,
    Quant,
    block_dot,
    block_exp,
    block_map,
    block_pack,
    block_project,
    block_reduce_add,
    block_scaled_add,
    convert_to_block_max_abs_finite,
)

from . import block_reference as ref
from .markers import accelerator_devices, available_devices, float64_devices
from .test_block_pack import (
    DETERMINISTIC,
    FORMATS,
    FOUR_OVER_SIX_TILES,
    PRESETS,
    WIDE_BFP,
    _as3,
    _inputs,
)

# Formats without a scale: a NaN is then stored as the largest value, as
# block_pack stores it.
UNSCALED = {
    "e4m3_unscaled": BlockFormat(E4M3, None, 32, elem_max=448.0, nan_code=0x7F),
    "e2m1_unscaled": BlockFormat(E2M1, None, 16),
}
ALL = FORMATS | FOUR_OVER_SIX_TILES | WIDE_BFP | UNSCALED
devices = available_devices


def _pow2_scale(fmt: BlockFormat) -> bool:
    return isinstance(fmt.scale, BinaryK) and fmt.scale.P == 1


def _ts(p) -> float | None:
    """The tensor scale to project onto ``p``'s scales with: ``p``'s, for a
    format that has one."""
    return p.tensor_scale if p.fmt.has_tensor_scale else None


def _with_infinities(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    flat[5::29] = float("inf")
    flat[17::31] = -float("inf")
    return x


def _random_scales(fmt: BlockFormat, p, seed=0) -> torch.Tensor:
    """Scale codes shaped as ``p.scales``: a power of two within 2^8 of 1 or
    the NaN code, or for a cast scale any magnitude code, zero and NaN
    included."""
    g = torch.Generator().manual_seed(seed)
    shape = p.scales.shape
    if fmt.scale is None:
        return p.scales.clone()
    mf = _minifloat(fmt.scale)
    if isinstance(fmt.scale, BinaryK) and fmt.scale.P == 1:
        bias, K = mf.bias, fmt.scale.K
        lo, hi = max(bias - 8, 0), min(bias + 8, (1 << K) - 2)
        codes = torch.randint(lo, hi + 1, shape, generator=g)
        nan = (1 << K) - 1
    else:
        nan = (1 << (mf.exp_bits + mf.man_bits)) - 1
        codes = torch.randint(0, nan, shape, generator=g)
        codes.view(-1)[::13] = 0  # a zero scale
    codes.view(-1)[7::17] = nan
    return codes.to(torch.uint8).to(p.scales.device)


def _reference(x: torch.Tensor, fmt: BlockFormat, scales: torch.Tensor, mode, ts, F=np.float32):
    """BlockProject's codes, as ``block_pack`` lays them out, from P3109's
    cases and the reference's value-set rounding: S is the scale code's value
    (``block_reference.scale_value``), Z = NaN, 0 or X / S, and Z rounded by
    ``block_reference.round_elements``."""
    xs = _as3(x.detach().cpu().double()).numpy()
    B, R, C = xs.shape
    bs, br = fmt.block_size, fmt.block_rows
    nan_x = np.isnan(xs)
    if fmt.scale is None:
        S = np.ones_like(xs)
    else:
        sc = _as3(scales.cpu()).numpy().astype(np.int64)
        sv = np.vectorize(lambda c: ref.scale_value(fmt, int(c), ts, F), otypes=[np.float64])(sc)
        S = np.repeat(np.repeat(sv, br, axis=1)[:, :R], bs, axis=2)[:, :, :C]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if fmt.scale is None or _pow2_scale(fmt):
            scaled = xs / S
        else:
            scaled = (xs.astype(F) / S.astype(F)).astype(np.float64)
    live = ~nan_x & ~np.isnan(S) & (S != 0)
    codes = ref.round_elements(fmt, np.where(live, scaled, 0.0), mode)
    codes = np.where(S == 0, 0, codes)
    nan_code = 0 if fmt.nan_code is None else fmt.nan_code
    codes = np.where(np.isnan(S), nan_code, codes)
    if fmt.nan_code is not None:
        codes = np.where(nan_x, fmt.nan_code, codes)
    elif fmt.scale is None:
        top = ref.round_elements(fmt, np.array([fmt.elem_max]), RoundMode.RZ)[0]
        codes = np.where(nan_x, top, codes)
    return torch.from_numpy(ref.pack_bytes(codes, fmt))


# --- block_project onto block_pack's scales -------------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("name", list(ALL))
def test_projecting_onto_the_packs_scales_is_the_pack(device, name):
    fmt = ALL[name]
    x = _with_infinities(_inputs((2, 37, 70), device))
    for rounding in RoundMode:
        for axis, t in ((-1, x), (-2, x), (-1, x.transpose(-1, -2))):
            torch.manual_seed(7)
            p = block_pack(t, fmt, axis, rounding=rounding)
            torch.manual_seed(7)
            q = block_project(t, fmt, p.scales, axis, rounding=rounding, tensor_scale=_ts(p))
            assert torch.equal(q.data, p.data), (rounding, axis)
            assert q.scales is p.scales and q.tensor_scale == p.tensor_scale


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("name", ["mxfp8_e4m3", "mxfp4_e2m1", "nvfp4", "bfp8", "sfp8"])
def test_projecting_in_both_carriers_is_the_pack(device, name):
    fmt = ALL[name]
    x = _inputs((3, 20, 64), device)
    for rounding in (RoundMode.RNE, RoundMode.RD, RoundMode.SR, RoundMode.SR_C):
        for t, carrier in ((x.double(), None), (x.bfloat16(), None), (x, torch.float64)):
            torch.manual_seed(3)
            p = block_pack(t, fmt, rounding=rounding, carrier=carrier)
            torch.manual_seed(3)
            q = block_project(t, fmt, p.scales, rounding=rounding, carrier=carrier)
            assert torch.equal(q.data, p.data), (rounding, t.dtype, carrier)
            assert q.carrier is p.carrier


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("name", [n for n in ALL if ALL[n].nan_code is not None])
def test_a_nan_is_stored_as_the_packs_nan(device, name):
    fmt = ALL[name]
    x = _inputs((24, 64), device)
    x.view(-1)[3::37] = float("nan")
    p = block_pack(x, fmt, rounding=RoundMode.RNE)
    q = block_project(x, fmt, p.scales, tensor_scale=_ts(p))
    assert torch.equal(q.data, p.data)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("name", ["mxfp4_e2m1", "nvfp4", "mxfp6_e2m3", "bfp4"])
def test_a_block_the_pack_made_nan_is_all_zero_codes(device, name):
    """block_pack gives a block holding a NaN its elements have no code for
    the scale format's NaN and keeps the other codes; block_project writes
    code 0 for every element of it, since every Z is NaN. Both decode to NaN,
    and every other block is the pack's."""
    fmt = ALL[name]
    x = _inputs((8, 64), device)
    x[2, 5] = float("nan")
    p = block_pack(x, fmt)
    q = block_project(x, fmt, p.scales, tensor_scale=_ts(p))
    bb = fmt.block_size * fmt.elem_bits // 8
    nan_bytes = torch.zeros_like(p.data, dtype=torch.bool)
    nan_bytes[2, :bb] = True
    assert torch.equal(q.data[~nan_bytes], p.data[~nan_bytes])
    assert (q.data[2, :bb] == 0).all()
    assert q.unpack()[2, : fmt.block_size].isnan().all()


# --- block_project onto other scales: P3109's cases ------------------------------


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("name", list(FORMATS | UNSCALED))
def test_projecting_onto_any_scales_is_the_reference(device, name):
    fmt = ALL[name]
    x = _with_infinities(_inputs((2, 19, 70), device))
    if fmt.nan_code is not None:
        x.view(-1)[11::23] = float("nan")
    p = block_pack(x, fmt)
    scales = _random_scales(fmt, p)
    ts = 0.375 if fmt.has_tensor_scale else None
    for rounding in DETERMINISTIC:
        q = block_project(x, fmt, scales, rounding=rounding, tensor_scale=ts)
        want = _reference(x, fmt, scales, rounding, 1.0 if ts is None else ts)
        assert torch.equal(_as3(q.data).cpu(), want), rounding


@pytest.mark.parametrize("device", devices)
def test_a_zero_scale_stores_zeros_and_keeps_a_nan(device):
    """A cast scale's zero code makes every Z zero, infinities included, and a
    NaN stays NaN, in its element's code."""
    fmt = BlockFormat(E4M3, E4M3, 16, elem_max=448.0, nan_code=0x7F)
    x = torch.tensor([[1.0, -float("inf"), float("inf"), float("nan")] * 4], device=device)
    q = block_project(x, fmt, torch.zeros(1, 1, dtype=torch.uint8, device=device))
    codes = q.data[0].tolist()
    assert codes == [0, 0, 0, 0x7F] * 4
    assert q.unpack()[0, :4].tolist()[:3] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("fmt", [MXFP4_E2M1, NVFP4], ids=["mxfp4_e2m1", "nvfp4"])
def test_a_nan_the_format_cannot_store_raises(device, fmt):
    """A NaN element under a scale that is not NaN, in a format
    whose elements have no NaN code, raises; giving that block the scale's NaN
    code stores it, as code 0 throughout."""
    x = _inputs((4, 64), device)
    scales = block_pack(x, fmt).scales.clone()
    x[1, 3] = float("nan")
    with pytest.raises(RuntimeError, match="an element is NaN and its block's scale is not"):
        block_project(x, fmt, scales)
    scales[1, 0] = 0xFF if fmt is MXFP4_E2M1 else 0x7F  # the scale format's NaN
    q = block_project(x, fmt, scales)
    assert q.unpack()[1, : fmt.block_size].isnan().all()
    assert (q.data[1, : fmt.block_size // 2] == 0).all()


@pytest.mark.parametrize("device", devices)
def test_a_nan_under_a_zero_scale_is_still_nan(device):
    """P3109's first case comes before its second: under a zero scale a NaN
    element is NaN, which E2M1 cannot store."""
    x = _inputs((4, 64), device)
    x[1, 3] = float("nan")
    scales = block_pack(torch.nan_to_num(x), NVFP4).scales.clone()
    scales[1, 0] = 0
    with pytest.raises(RuntimeError, match="an element is NaN"):
        block_project(x, NVFP4, scales)


@pytest.mark.parametrize("device", devices)
def test_a_format_without_a_scale_stores_a_nan_as_its_largest_value(device):
    fmt = UNSCALED["e2m1_unscaled"]
    x = torch.tensor([[float("nan"), 1.0] * 8], device=device)
    q = block_project(x, fmt, torch.zeros(1, 0, dtype=torch.uint8, device=device))
    assert q.unpack()[0, :2].tolist() == [6.0, 1.0]


@pytest.mark.parametrize("device", devices)
def test_an_empty_tensor_gives_empty_codes(device):
    x = torch.zeros(0, 64, device=device)
    q = block_project(x, MXFP8_E4M3, torch.zeros(0, 2, dtype=torch.uint8, device=device))
    assert q.data.shape == (0, 64) and q.data.dtype == torch.uint8


@pytest.mark.parametrize("device", devices)
def test_the_scale_codes_are_checked(device):
    x = torch.randn(4, 64, device=device)
    good = torch.zeros(4, 2, dtype=torch.uint8, device=device)
    with pytest.raises(TypeError, match="uint8 tensor"):
        block_project(x, MXFP8_E4M3, [[0, 0]])  # ty: ignore[invalid-argument-type]
    with pytest.raises(RuntimeError, match="the scale codes are uint8"):
        block_project(x, MXFP8_E4M3, good.to(torch.int32))
    with pytest.raises(RuntimeError, match=r"takes scale codes of sizes \[4, 2\]"):
        block_project(x, MXFP8_E4M3, good[:, :1])
    with pytest.raises(RuntimeError, match="the scale codes of a 2D tensor are 2D"):
        block_project(x, MXFP8_E4M3, good.unsqueeze(0))
    if device != "cpu":
        with pytest.raises(RuntimeError, match="the scale codes are on cpu"):
            block_project(x, MXFP8_E4M3, good.cpu())


@pytest.mark.parametrize("device", accelerator_devices)
@pytest.mark.parametrize("name", list(PRESETS))
def test_an_accelerator_gives_the_cpus_codes(device, name):
    """Every deterministic mode on CUDA and MPS, and SR on MPS, which draws
    from the CPU generator as the CPU kernel does (CUDA draws from its own)."""
    fmt = ALL[name]
    x = _with_infinities(_inputs((3, 21, 96), "cpu"))
    p = block_pack(x, fmt)
    scales = _random_scales(fmt, p)
    modes = list(RoundMode) if device == "mps" else DETERMINISTIC
    for rounding in modes:
        torch.manual_seed(5)
        want = block_project(x, fmt, scales, rounding=rounding)
        torch.manual_seed(5)
        got = block_project(x.to(device), fmt, scales.to(device), rounding=rounding)
        assert torch.equal(got.data.cpu(), want.data), rounding


# --- block_map, block_exp, block_scaled_add ---------------------------------------


@pytest.mark.parametrize("device", devices)
def test_block_map_decodes_computes_and_projects(device):
    a = block_pack(_inputs((16, 64), device, seed=1), NVFP4)
    b = block_pack(_inputs((16, 64), device, seed=2), MXFP8_E4M3)
    y = torch.mul(a.unpack(torch.float32), b.unpack(torch.float32))
    onto = block_map(torch.mul, a, b, scales=a.scales)
    assert torch.equal(onto.data, block_project(y, NVFP4, a.scales, tensor_scale=_ts(a)).data)
    chosen = block_map(torch.mul, a, b)
    assert torch.equal(chosen.data, block_pack(y, NVFP4).data)
    assert torch.equal(chosen.scales, block_pack(y, NVFP4).scales)
    other = block_map(torch.mul, a, b, fmt=MXFP8_E5M2)
    assert other.fmt is MXFP8_E5M2 and torch.equal(other.data, block_pack(y, MXFP8_E5M2).data)


@pytest.mark.parametrize("device", devices)
def test_block_map_keeps_the_axis_and_the_dtype(device):
    x = _inputs((64, 8), device).bfloat16()
    p = block_pack(x, MXFP8_E4M3, axis=0)
    q = block_map(torch.neg, p)
    assert q.axis == 0 and q.shape == (64, 8) and q.dtype is torch.bfloat16
    assert torch.equal(q.unpack(), -p.unpack())


@pytest.mark.parametrize("device", float64_devices)
def test_block_map_computes_in_binary64_for_a_binary64_operand(device):
    x = _inputs((8, 64), device)
    a = block_pack(x, MXFP8_E4M3, carrier=torch.float64)
    b = block_pack(x, MXFP8_E4M3)
    q = block_map(torch.add, a, b)
    assert q.carrier is torch.float64
    y = a.unpack(torch.float64) + b.unpack(torch.float64)
    assert torch.equal(q.data, block_pack(y, MXFP8_E4M3).data)


def test_block_map_refuses_what_it_cannot_pair_or_store():
    p = block_pack(torch.randn(4, 64), MXFP8_E4M3)
    with pytest.raises(TypeError, match="at least one BlockPacked"):
        block_map(torch.neg)
    with pytest.raises(TypeError, match="BlockPacked operands"):
        block_map(torch.add, p, torch.randn(4, 64))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="one shape and one packed axis"):
        block_map(torch.add, p, block_pack(torch.randn(4, 64), MXFP8_E4M3, axis=0))
    with pytest.raises(TypeError, match="float32 tensor"):
        block_map(lambda v: v.double(), p)
    with pytest.raises(ValueError, match="operands' shape"):
        block_map(lambda v: v[:2], p)


@pytest.mark.parametrize("device", devices)
def test_block_exp_is_the_exponential_projected(device):
    p = block_pack(_inputs((16, 64), device, spread=0.5), MXFP8_E4M3)
    y = torch.exp(p.unpack(torch.float32))
    for rounding in (RoundMode.RNE, RoundMode.RU, RoundMode.SR):
        torch.manual_seed(2)
        chosen = block_exp(p, rounding=rounding)
        torch.manual_seed(2)
        want = block_pack(y, MXFP8_E4M3, rounding=rounding)
        assert torch.equal(chosen.data, want.data) and torch.equal(chosen.scales, want.scales)
        # P3109: the variant that chooses its scales gives what BlockExp gives on them
        torch.manual_seed(2)
        onto = block_exp(p, chosen.scales, rounding=rounding)
        assert torch.equal(onto.data, chosen.data)
        onto_p = block_exp(p, p.scales)
        assert torch.equal(onto_p.data, block_project(y, MXFP8_E4M3, p.scales).data)


@pytest.mark.parametrize("device", devices)
def test_block_scaled_add_is_the_sum_projected(device):
    a = block_pack(_inputs((8, 64), device, seed=3), MXFP4_E2M1)
    b = block_pack(_inputs((8, 64), device, seed=4), NVFP4)
    y = a.unpack(torch.float32) + b.unpack(torch.float32)
    assert torch.equal(block_scaled_add(a, b).data, block_pack(y, MXFP4_E2M1).data)
    q = block_scaled_add(a, b, a.scales, fmt=MXFP4_E2M1, rounding=RoundMode.RD)
    assert torch.equal(q.data, block_project(y, MXFP4_E2M1, a.scales, rounding=RoundMode.RD).data)


# --- convert_to_block_max_abs_finite ---------------------------------------------


def _nearest_pow2_code(m: Fraction, bias: int, top: int) -> int:
    """The code of the power of two nearest ``m``, a tie to the even
    exponent, clamped to [0, top]; 0 for m = 0."""
    if m == 0:
        return 0
    k = math.floor(math.log2(m))
    while Fraction(2) ** k > m:
        k -= 1
    while Fraction(2) ** (k + 1) <= m:
        k += 1
    lo, hi = Fraction(2) ** k, Fraction(2) ** (k + 1)
    if m - lo > hi - m or (m - lo == hi - m and (k + 1) % 2 == 0):
        k += 1
    return min(max(k + bias, 0), top)


def _nearest_scale_code(r: Fraction, scale: "BinaryK | SuperFP") -> int:
    """The magnitude code of the scale value nearest ``r``, a tie to the even
    code, at most the code below the NaN."""
    mf = _minifloat(scale)
    nan_mag = (1 << (mf.bits - int(mf.is_signed))) - 1
    values = [_minifloat_value(mf, c) for c in range(nan_mag)]
    best = min(range(nan_mag), key=lambda c: (abs(values[c] - r), c % 2))
    return best


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize(
    "fmt",
    [MXFP8_E4M3, MXFP4_E2M1, NVFP4, BlockFormat(E2M1, SuperFP(3, 4, 2, 7), 16)],
    ids=["mxfp8_e4m3", "mxfp4_e2m1", "nvfp4", "e2m1_sfp_scale"],
)
def test_each_scale_is_the_blocks_largest_finite_magnitude_rounded(device, fmt):
    x = _with_infinities(_inputs((6, 70), device))
    x[0, :32] = 0.0  # a block of zeros
    x[1, 0] = 3.0 * 2.0**-5  # a tie between two powers of two
    x[1, 1:32] = 0.0
    if fmt.nan_code is not None:
        x[2, 4] = float("nan")
    q = convert_to_block_max_abs_finite(x, fmt)
    xs = x.cpu().double()
    a = torch.where(torch.isfinite(xs), xs.abs(), torch.zeros(()))
    bs = fmt.block_size
    want = []
    for r in range(x.shape[0]):
        row = []
        for j in range(-(-x.shape[1] // bs)):
            m = Fraction(a[r, j * bs : (j + 1) * bs].max().item())
            if _pow2_scale(fmt):
                assert E8M0.bias is not None
                row.append(_nearest_pow2_code(m, E8M0.bias, 254))
            else:
                assert fmt.scale is not None
                row.append(_nearest_scale_code(m / Fraction(q.tensor_scale), fmt.scale))
        want.append(row)
    assert q.scales.cpu().tolist() == want
    assert torch.equal(q.data, block_project(x, fmt, q.scales, tensor_scale=_ts(q)).data)
    if fmt.has_tensor_scale:
        assert q.tensor_scale == np.float32(a.max().item() / fmt.scale_max)


@pytest.mark.parametrize("device", devices)
def test_a_block_whose_largest_is_a_scale_stores_it_as_one(device):
    x = torch.tensor([[0.25, -1.0, 2.0, 4.0] * 8, [-8.0, 0.5, 1.0, 0.125] * 8], device=device)
    q = convert_to_block_max_abs_finite(x, MXFP8_E4M3)
    one, minus_one = 0x38, 0xB8  # E4M3's 1 and -1
    assert q.data[0, 3].item() == one and q.data[1, 0].item() == minus_one
    assert torch.equal(q.unpack(), x)


def test_convert_refuses_a_format_without_a_scale():
    with pytest.raises(ValueError, match="needs a format with a scale"):
        convert_to_block_max_abs_finite(torch.randn(4, 32), UNSCALED["e2m1_unscaled"])
    with pytest.raises(TypeError, match="takes a BlockFormat"):
        convert_to_block_max_abs_finite(torch.randn(4, 32), E4M3)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="no per-tensor scale"):
        convert_to_block_max_abs_finite(torch.randn(4, 32), MXFP8_E4M3, tensor_scale=2.0)


# --- block_reduce_add, block_dot ---------------------------------------------------


def _steps(terms, fmt):
    """A sum written out: each step ``s + t`` rounded once in binary32 and then
    to ``fmt`` (a FusedMac's step), from s = 0, in order."""
    q = Quant(fmt)
    s = torch.zeros((), dtype=torch.float32)
    for t in terms:
        exact = s.double() + t
        s = q(exact.float())
    return s.item()


@pytest.mark.parametrize("device", devices)
def test_block_reduce_add_sums_each_block_in_order(device):
    x = _inputs((3, 70), device, spread=0.25)  # a last block of 6
    p = block_pack(x, MXFP8_E4M3)
    v = p.unpack(torch.float32).cpu().double()
    exact = block_reduce_add(p).cpu()
    assert exact.shape == (3, 3) and exact.dtype is torch.float32
    acc = BinaryK(8, 4)
    stepped = block_reduce_add(p, mac=FusedMac(acc)).cpu()
    for r in range(3):
        for j in range(3):
            block = v[r, j * 32 : (j + 1) * 32]
            assert abs(exact[r, j].item() - block.sum().item()) <= 1e-6 * block.abs().sum().item()
            assert stepped[r, j].item() == _steps(block.tolist(), acc)
    along0 = block_reduce_add(block_pack(x.t().contiguous(), MXFP8_E4M3, axis=0), mac=acc)
    assert torch.equal(along0.cpu(), stepped.t())


@pytest.mark.parametrize("device", devices)
def test_block_dot_multiplies_each_pair_of_blocks(device):
    a = block_pack(_inputs((2, 3, 40), device, seed=5, spread=0.1), MXFP8_E4M3)  # a last block of 8
    b = block_pack(_inputs((2, 3, 40), device, seed=6, spread=0.1), MXFP4_E2M1)
    va, vb = a.unpack(torch.float32).cpu().double(), b.unpack(torch.float32).cpu().double()
    exact = block_dot(a, b).cpu()
    assert exact.shape == (2, 3, 2)
    acc = BinaryK(8, 4)
    stepped = block_dot(a, b, FusedMac(acc)).cpu()
    for i in range(2):
        for r in range(3):
            for j in range(2):
                pa, pb = va[i, r, j * 32 : (j + 1) * 32], vb[i, r, j * 32 : (j + 1) * 32]
                bound = 1e-6 * (pa * pb).abs().sum().item()
                assert abs(exact[i, r, j].item() - (pa * pb).sum().item()) <= bound
                assert stepped[i, r, j].item() == _steps((pa * pb).tolist(), acc)


def test_block_dot_refuses_blocks_that_do_not_pair():
    a = block_pack(torch.randn(4, 64), MXFP8_E4M3)
    with pytest.raises(ValueError, match="one block size"):
        block_dot(a, block_pack(torch.randn(4, 64), NVFP4))
    with pytest.raises(ValueError, match="one shape and one packed axis"):
        block_dot(a, block_pack(torch.randn(4, 32), MXFP8_E4M3))
