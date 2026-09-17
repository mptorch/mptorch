"""The superfp mixed-format GEMM must reproduce the single-format one exactly.

``superfp_matmul_mixed`` reaches its arithmetic by a different route than
``superfp_matmul``: the policies are built into a ``FormatPalette`` and the
slot is resolved per output element, against one policy held for the whole
call. Both routes must read the one precomputed ``SuperfpParams`` that
``make_superfp_params`` builds; a second spelling of its constants on the
mixed path, rebuilt per tile instead of carried, is exactly where the two
could drift apart, and this sweep is what would show it. A one-entry palette
therefore has to agree with the single-format op word for word, and the
formats below straddle ``make_superfp_params``' fast-RNE gate in both
directions so the float fast path and the integer path are both exercised on
both routes.
"""

import pytest
import torch

from mptorch.number import RoundMode, SaturationMode
from mptorch.quant import superfp_matmul, superfp_matmul_mixed
from tests.markers import available_devices

# (man_bits, exp_bits, normal_binades, bias), with what each one is here for.
SUPERFP_FORMATS = [
    (3, 4, 1, 7),  # e4m3-shaped, one normal binade: fast path on
    (3, 4, 8, 7),  # every binade normal: supernormal region empty
    (2, 5, 1, 15),  # e5m2-shaped
    (1, 4, 2, 7),  # man_bits at the gate's lower edge
    (23, 8, 1, 127),  # man_bits at the gate's upper edge
    (8, 5, 1, 15),  # supernormal_cutoff far below -127: the no_underflow arm,
    # where the smallest supernormal is 0 and the flush threshold is the
    # largest float32 below 2^-127 rather than a power of two
    (0, 4, 1, 7),  # man_bits == 0: fast path off, integer path only
]

SATURATION_MODES = [
    SaturationMode.OVF_INF,
    SaturationMode.SAT_FINITE,
    SaturationMode.SAT_PROPAGATE,  # gates the fast path off
]

ROUND_MODES = [
    RoundMode.RNE,  # the mode the fast path exists for
    RoundMode.RNA,
    RoundMode.RU,
    RoundMode.RD,
    RoundMode.RZ,
    RoundMode.RO,
]


def assert_bit_identical(mixed, single, ctx=""):
    """Compare the raw words, not the values: ``nan == nan`` is false and
    ``-0.0 == 0.0`` is true, and this is a claim about bits."""
    assert mixed.dtype == single.dtype
    assert torch.equal(mixed.view(torch.int32), single.view(torch.int32)), ctx


def _operands(device, M=24, K=40, N=18, scale=1.0):
    """Seeded float32 operands; ``scale`` steers the products into a range region."""
    torch.manual_seed(1234)
    a = torch.randn(M, K, device=device) * scale
    b = torch.randn(K, N, device=device) * scale
    return a, b


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fmt", SUPERFP_FORMATS)
@pytest.mark.parametrize("saturation_mode", SATURATION_MODES)
def test_mixed_matches_single_format_across_formats(device, fmt, saturation_mode):
    """A one-entry palette gives the single-format op's words for every format
    and saturation mode, on both sides of the fast-RNE gate."""
    man_bits, exp_bits, normal_binades, bias = fmt
    a, b = _operands(device)
    prec_idx = torch.zeros(a.shape[0], b.shape[1], dtype=torch.int32, device=device)

    out_mixed = superfp_matmul_mixed(
        a,
        b,
        prec_idx,
        mul_man_bits=[man_bits],
        mul_exp_bits=[exp_bits],
        mul_normal_binades=normal_binades,
        mul_bias=bias,
        saturation_mode=saturation_mode,
    )
    out_single = superfp_matmul(
        a,
        b,
        mul_man_bits=man_bits,
        mul_exp_bits=exp_bits,
        mul_normal_binades=normal_binades,
        mul_bias=bias,
        saturation_mode=saturation_mode,
    )
    assert_bit_identical(out_mixed, out_single)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("round_mode", ROUND_MODES)
def test_mixed_matches_single_format_across_round_modes(device, round_mode):
    """Only RNE and SR read the fast-path constants, but the other modes share
    the params struct and would notice a field that moved."""
    a, b = _operands(device)
    prec_idx = torch.zeros(a.shape[0], b.shape[1], dtype=torch.int32, device=device)
    kw: dict = dict(mul_normal_binades=1, mul_bias=7, rounding_mode=round_mode)

    out_mixed = superfp_matmul_mixed(a, b, prec_idx, mul_man_bits=[3], mul_exp_bits=[4], **kw)
    out_single = superfp_matmul(a, b, mul_man_bits=3, mul_exp_bits=4, **kw)
    assert_bit_identical(out_mixed, out_single)


# e4m3-shaped with one normal binade and bias 7: largest finite 448, and the
# supernormal region runs down to 2^-111. The scales below put the products in
# the normal region, over the top of it, and under the bottom of it.
RANGE_EDGE_SCALES = [1.0, 12.0, 1e-17]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("scale", RANGE_EDGE_SCALES)
def test_mixed_matches_single_format_at_the_range_edges(device, scale):
    """Saturation and flush-to-zero are where fast_max_finite / fast_ovf and
    fast_super_min / fast_super_half decide the answer, so the operands are
    pushed out of the format's range in both directions."""
    a, b = _operands(device, scale=scale)
    prec_idx = torch.zeros(a.shape[0], b.shape[1], dtype=torch.int32, device=device)
    for saturation_mode in SATURATION_MODES:
        out_mixed = superfp_matmul_mixed(
            a,
            b,
            prec_idx,
            mul_man_bits=[3],
            mul_exp_bits=[4],
            mul_normal_binades=1,
            mul_bias=7,
            saturation_mode=saturation_mode,
        )
        out_single = superfp_matmul(
            a,
            b,
            mul_man_bits=3,
            mul_exp_bits=4,
            mul_normal_binades=1,
            mul_bias=7,
            saturation_mode=saturation_mode,
        )
        # a scale that pushed every element to NaN would compare equal while
        # testing nothing
        assert torch.isfinite(out_single).any(), (scale, saturation_mode)
        assert_bit_identical(out_mixed, out_single, (scale, saturation_mode))


@pytest.mark.parametrize("device", available_devices)
def test_palette_selects_the_same_format_per_row(device):
    """A genuinely mixed palette, checked one slot at a time: row r of the
    mixed output must equal row r of the single-format GEMM in the format that
    row's index selects. The single-format GEMMs are full size so the
    per-element geometry (and with it any SR stream) lines up."""
    a, b = _operands(device)
    M, N = a.shape[0], b.shape[1]
    formats = [(3, 4, 1, 7), (2, 5, 1, 15), (8, 5, 1, 15)]
    row_idx = torch.arange(M, device=device, dtype=torch.int32).remainder(len(formats))

    out_mixed = superfp_matmul_mixed(
        a,
        b,
        row_idx.reshape(M, 1),
        mul_man_bits=[f[0] for f in formats],
        mul_exp_bits=[f[1] for f in formats],
        mul_normal_binades=[f[2] for f in formats],
        mul_bias=[f[3] for f in formats],
    )
    assert out_mixed.shape == (M, N)
    for slot, (man_bits, exp_bits, normal_binades, bias) in enumerate(formats):
        rows = row_idx == slot
        out_single = superfp_matmul(
            a,
            b,
            mul_man_bits=man_bits,
            mul_exp_bits=exp_bits,
            mul_normal_binades=normal_binades,
            mul_bias=bias,
        )
        assert_bit_identical(out_mixed[rows], out_single[rows], slot)


@pytest.mark.parametrize("device", available_devices)
def test_mixed_matches_single_format_under_stochastic_rounding(device):
    """SR reads fast_max_finite / fast_ovf on its normal arm. The Philox stream
    is keyed by the output element's linear index and the launch's seed,
    neither of which the palette touches, so re-seeding gives the two paths
    the same draws and the comparison stays exact."""
    a, b = _operands(device)
    prec_idx = torch.zeros(a.shape[0], b.shape[1], dtype=torch.int32, device=device)
    kw: dict = dict(
        mul_normal_binades=1,
        mul_bias=7,
        rounding_mode=RoundMode.SR,
        mul_prng_bits=12,
        acc_prng_bits=12,
    )

    torch.manual_seed(4321)
    out_mixed = superfp_matmul_mixed(a, b, prec_idx, mul_man_bits=[3], mul_exp_bits=[4], **kw)
    torch.manual_seed(4321)
    out_single = superfp_matmul(a, b, mul_man_bits=3, mul_exp_bits=4, **kw)
    assert_bit_identical(out_mixed, out_single)
