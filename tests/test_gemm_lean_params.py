"""Gate for the lean superfp parameter struct -- finding G10 in
dev/gemm_perf_audit.md.

The mixed superfp GEMM's ``SplitMac`` carries two full ``SuperfpParams`` per
palette slot, and copying one into registers per output element cost it 100
registers and 2 resident blocks per SM against its single-format twin's 5.
``SuperfpParamsLean`` (cast_superfp.h) drops the seven precomputed fast-path
floats and rebuilds them where they are read, which buys back a block.

Correctness rests on one claim: the derived constants are bit-identical to the
stored ones. That is not observable from Python directly, so these tests use
the two GEMM paths as the two spellings -- ``superfp_matmul_mixed`` runs the
lean policy and ``superfp_matmul`` the stored one, over the same arithmetic, so
any disagreement between them is a disagreement between the params structs.
The format sweep is chosen to straddle the fast-RNE gate in both directions.
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
    (2, 5, 1, 15),
    (1, 4, 2, 7),  # man_bits at the gate's lower edge
    (23, 8, 1, 127),  # man_bits at the gate's upper edge
    (8, 5, 1, 15),  # supernormal_cutoff far below -127: the no_underflow arm,
    # where the stored form writes zeros and the derived
    # form has to reach them through the -127 clamp
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


def assert_bit_identical(lean, stored, ctx=""):
    """Compare the raw words, not the values: `nan == nan` is false and
    `-0.0 == 0.0` is true, and this is a claim about bits."""
    assert lean.dtype == stored.dtype
    assert torch.equal(lean.view(torch.int32), stored.view(torch.int32)), ctx


def _operands(device, M=24, K=40, N=18, scale=1.0):
    torch.manual_seed(1234)
    a = torch.randn(M, K, device=device) * scale
    b = torch.randn(K, N, device=device) * scale
    return a, b


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fmt", SUPERFP_FORMATS)
@pytest.mark.parametrize("saturation_mode", SATURATION_MODES)
def test_lean_params_match_stored_across_formats(device, fmt, saturation_mode):
    man_bits, exp_bits, normal_binades, bias = fmt
    a, b = _operands(device)
    prec_idx = torch.zeros(a.shape[0], b.shape[1], dtype=torch.int32, device=device)

    out_lean = superfp_matmul_mixed(
        a,
        b,
        prec_idx,
        mul_man_bits=[man_bits],
        mul_exp_bits=[exp_bits],
        mul_normal_binades=normal_binades,
        mul_bias=bias,
        saturation_mode=saturation_mode,
    )
    out_stored = superfp_matmul(
        a,
        b,
        mul_man_bits=man_bits,
        mul_exp_bits=exp_bits,
        mul_normal_binades=normal_binades,
        mul_bias=bias,
        saturation_mode=saturation_mode,
    )
    assert_bit_identical(out_lean, out_stored)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("round_mode", ROUND_MODES)
def test_lean_params_match_stored_across_round_modes(device, round_mode):
    # Only RNE and SR read the fast-path constants, but the other modes share
    # the params struct and would notice a field that moved.
    a, b = _operands(device)
    prec_idx = torch.zeros(a.shape[0], b.shape[1], dtype=torch.int32, device=device)
    kw: dict = dict(mul_normal_binades=1, mul_bias=7, rounding_mode=round_mode)

    out_lean = superfp_matmul_mixed(a, b, prec_idx, mul_man_bits=[3], mul_exp_bits=[4], **kw)
    out_stored = superfp_matmul(a, b, mul_man_bits=3, mul_exp_bits=4, **kw)
    assert_bit_identical(out_lean, out_stored)


# e4m3-shaped with one normal binade and bias 7: largest finite 448, and the
# supernormal region runs down to 2^-111. The scales below put the products in
# the normal region, over the top of it, and under the bottom of it.
RANGE_EDGE_SCALES = [1.0, 12.0, 1e-17]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("scale", RANGE_EDGE_SCALES)
def test_lean_params_match_stored_at_the_range_edges(device, scale):
    # Saturation and flush-to-zero are where fast_max_finite / fast_ovf and
    # fast_super_min / fast_super_half actually decide the answer, so push the
    # operands out of the format's range in both directions.
    a, b = _operands(device, scale=scale)
    prec_idx = torch.zeros(a.shape[0], b.shape[1], dtype=torch.int32, device=device)
    for saturation_mode in SATURATION_MODES:
        out_lean = superfp_matmul_mixed(
            a,
            b,
            prec_idx,
            mul_man_bits=[3],
            mul_exp_bits=[4],
            mul_normal_binades=1,
            mul_bias=7,
            saturation_mode=saturation_mode,
        )
        out_stored = superfp_matmul(
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
        assert torch.isfinite(out_stored).any(), (scale, saturation_mode)
        assert_bit_identical(out_lean, out_stored, (scale, saturation_mode))


@pytest.mark.parametrize("device", available_devices)
def test_lean_palette_selects_the_same_format_per_row(device):
    # A genuinely mixed palette, checked against the stored path one slot at a
    # time: row r of the mixed output must equal row r of the single-format
    # GEMM for the format that row's index selects. Full-size single-format
    # GEMMs, so the per-element geometry (and therefore any SR stream) lines up.
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
        out_stored = superfp_matmul(
            a,
            b,
            mul_man_bits=man_bits,
            mul_exp_bits=exp_bits,
            mul_normal_binades=normal_binades,
            mul_bias=bias,
        )
        assert_bit_identical(out_mixed[rows], out_stored[rows], slot)


@pytest.mark.parametrize("device", available_devices)
def test_lean_params_match_stored_under_stochastic_rounding(device):
    # SR reads fast_max_finite/fast_ovf on its normal arm. The Philox stream is
    # keyed by the output element's linear index and the launch's seed, neither
    # of which the palette touches, so re-seeding gives the two paths the same
    # draws and the comparison stays exact.
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
    out_lean = superfp_matmul_mixed(a, b, prec_idx, mul_man_bits=[3], mul_exp_bits=[4], **kw)
    torch.manual_seed(4321)
    out_stored = superfp_matmul(a, b, mul_man_bits=3, mul_exp_bits=4, **kw)
    assert_bit_identical(out_lean, out_stored)
