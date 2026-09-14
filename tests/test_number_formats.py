"""The format vocabulary: `BinaryK`/`SuperFP`, `Quant`, `SplitMac`/`FusedMac`,
`Palette`, and the differentiable entry points built on them (X1).

The load-bearing test here is the first one. Every format object resolves to
the same ``_GemmSpec`` the equivalent flat-wrapper call builds, over the
parameter grid the existing tests use -- that is the whole equivalence claim
between the two API tiers, and if it holds they cannot drift apart.
"""

import itertools

import pytest
import torch

from mptorch import (
    BinaryK,
    RoundMode,
    SaturationMode,
    SubnormalsMode,
    SuperFP,
)
from mptorch.quant import (
    FusedMac,
    Palette,
    QMatmul,
    QMatmulFormats,
    Quant,
    Quantizer,
    SplitMac,
    binaryK_matmul,
    binaryK_quantize,
    matmul_formats,
    qbmm,
    qmatmul,
    qmm,
    superfp_quantize,
)
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops import (
    _binaryK_fma_mixed_spec,
    _binaryK_fma_spec,
    _binaryK_mixed_spec,
    _binaryK_spec,
    _superfp_fma_mixed_spec,
    _superfp_fma_spec,
    _superfp_mixed_spec,
    _superfp_spec,
)
from tests.markers import available_devices

# ------------------------------------------------------------------------------------
# The equivalence claim: a mac resolves to the flat wrapper's spec.


def _binaryK_grid():
    """The (K, P) widths and the modes the existing tests use, crossed."""
    for (K, P), signed, rm, sat, sub in itertools.product(
        [(8, 4), (16, 11), (6, 3)],
        [True, False],
        [RoundMode.RNE, RoundMode.SR, RoundMode.RZ],
        [SaturationMode.OVF_INF, SaturationMode.SAT_FINITE],
        [SubnormalsMode.SUBNORMALS, SubnormalsMode.NORMALS],
    ):
        yield K, P, signed, rm, sat, sub


@pytest.mark.parametrize("K,P,is_signed,rm,sat,sub", list(_binaryK_grid()))
def test_binaryK_split_mac_resolves_to_flat_spec(K, P, is_signed, rm, sat, sub):
    fmt = BinaryK(K, P, is_signed=is_signed, saturation=sat, subnormals=sub)
    assert spec_for_mac(SplitMac(fmt, fmt, rounding=rm)) == _binaryK_spec(
        mul_K=K,
        mul_P=P,
        mul_is_signed=is_signed,
        acc_K=K,
        acc_P=P,
        acc_is_signed=is_signed,
        rounding_mode=rm,
        saturation_mode=sat,
        subnormals_mode=sub,
    )


def test_binaryK_split_mac_without_accumulate_format():
    """`acc=None` *is* the full-precision-sum case, not a boolean beside it."""
    assert spec_for_mac(SplitMac(BinaryK(8, 4))) == _binaryK_spec(
        mul_K=8, mul_P=4, accumulate_quant=False
    )


def test_binaryK_split_mac_asymmetric_slots():
    """The two halves of a mac carry their own saturation and subnormals."""
    mul = BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE)
    acc = BinaryK(12, 6, saturation=SaturationMode.OVF_INF, subnormals=SubnormalsMode.NORMALS)
    assert spec_for_mac(SplitMac(mul, acc)) == _binaryK_spec(
        mul_K=8,
        mul_P=4,
        acc_K=12,
        acc_P=6,
        saturation_mode=SaturationMode.SAT_FINITE,
        subnormals_mode=SubnormalsMode.SUBNORMALS,
        acc_saturation_mode=SaturationMode.OVF_INF,
        acc_subnormals_mode=SubnormalsMode.NORMALS,
    )


@pytest.mark.parametrize("rm", [RoundMode.RNE, RoundMode.SR])
@pytest.mark.parametrize("binades", [1, 8])
def test_superfp_split_mac_resolves_to_flat_spec(rm, binades):
    fmt = SuperFP(3, 4, binades, 7)
    assert spec_for_mac(SplitMac(fmt, fmt, rounding=rm)) == _superfp_spec(
        mul_man_bits=3,
        mul_exp_bits=4,
        mul_normal_binades=binades,
        mul_bias=7,
        acc_man_bits=3,
        acc_exp_bits=4,
        acc_normal_binades=binades,
        acc_bias=7,
        rounding_mode=rm,
    )


@pytest.mark.parametrize("rm", [RoundMode.RNE, RoundMode.SR])
def test_fused_macs_resolve_to_flat_specs(rm):
    assert spec_for_mac(FusedMac(BinaryK(8, 4), rounding=rm)) == _binaryK_fma_spec(
        fma_K=8, fma_P=4, rounding_mode=rm
    )
    assert spec_for_mac(FusedMac(SuperFP(3, 4, 8, 7), rounding=rm)) == _superfp_fma_spec(
        fma_man_bits=3, fma_exp_bits=4, fma_normal_binades=8, fma_bias=7, rounding_mode=rm
    )


def test_fused_mac_without_a_format_is_the_unrounded_step():
    spec = spec_for_mac(FusedMac(None))
    assert spec.op is torch.ops.mptorch.custom_matmul_binaryK_fma.default
    assert spec.args[0] is False  # fma_quant


def test_palette_macs_resolve_to_flat_specs():
    pal = [BinaryK(8, 4), BinaryK(6, 3)]
    assert spec_for_mac(SplitMac(pal, pal)) == _binaryK_mixed_spec(
        mul_K=[8, 6], mul_P=[4, 3], acc_K=[8, 6], acc_P=[4, 3]
    )
    assert spec_for_mac(FusedMac(pal)) == _binaryK_fma_mixed_spec(fma_K=[8, 6], fma_P=[4, 3])
    spal = [SuperFP(3, 4, 8, 7), SuperFP(2, 4, 8, 7)]
    assert spec_for_mac(SplitMac(spal, spal)) == _superfp_mixed_spec(
        mul_man_bits=[3, 2],
        mul_exp_bits=[4, 4],
        mul_normal_binades=[8, 8],
        mul_bias=[7, 7],
        acc_man_bits=[3, 2],
        acc_exp_bits=[4, 4],
        acc_normal_binades=[8, 8],
        acc_bias=[7, 7],
    )
    assert spec_for_mac(FusedMac(spal)) == _superfp_fma_mixed_spec(
        fma_man_bits=[3, 2], fma_exp_bits=[4, 4], fma_normal_binades=[8, 8], fma_bias=[7, 7]
    )


def test_palette_on_one_slot_broadcasts_the_other():
    """A single format opposite a palette is repeated to its length.

    Either way round: the mixed schemas read the palette size off the first
    pair of lists, so a scalar multiply against a palette accumulate would
    leave the op with no length to read.
    """
    assert spec_for_mac(SplitMac([BinaryK(8, 4), BinaryK(6, 3)], BinaryK(8, 4))) == (
        _binaryK_mixed_spec(mul_K=[8, 6], mul_P=[4, 3], acc_K=8, acc_P=4)
    )
    assert spec_for_mac(SplitMac(BinaryK(8, 4), [BinaryK(8, 4), BinaryK(6, 3)])) == (
        _binaryK_mixed_spec(mul_K=[8, 8], mul_P=[4, 4], acc_K=[8, 6], acc_P=[4, 3])
    )


def test_spec_resolution_is_memoized():
    """A mac built twice resolves once -- the memo P4's numbers require."""
    assert spec_for_mac(SplitMac(BinaryK(8, 4), BinaryK(8, 4))) is spec_for_mac(
        SplitMac(BinaryK(8, 4), BinaryK(8, 4))
    )


@pytest.mark.parametrize("carrier", [None, torch.float32, torch.float64])
def test_carrier_resolves_to_the_flat_spec(carrier):
    """The carrier is part of the spec, and both tiers put it in the same place."""
    fmt, wide = BinaryK(16, 8), BinaryK(40, 30, bias=512)
    assert spec_for_mac(SplitMac(fmt, wide, carrier=carrier)) == _binaryK_spec(
        mul_K=16, mul_P=8, acc_K=40, acc_P=30, acc_bias=512, carrier=carrier
    )
    assert spec_for_mac(FusedMac(wide, carrier=carrier)) == _binaryK_fma_spec(
        fma_K=40, fma_P=30, fma_bias=512, carrier=carrier
    )
    assert spec_for_mac(FusedMac([fmt, wide], carrier=carrier)) == _binaryK_fma_mixed_spec(
        fma_K=[16, 40], fma_P=[8, 30], fma_bias=[128, 512], carrier=carrier
    )
    sfp = SuperFP(3, 4, 1, 22)
    assert spec_for_mac(SplitMac(sfp, sfp, carrier=carrier)) == _superfp_spec(
        mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=1, mul_bias=22, carrier=carrier
    )
    assert spec_for_mac(SplitMac(fmt, wide, carrier=carrier)).carrier == carrier


def test_a_spec_holds_what_each_carrier_says_about_its_formats():
    """binary32's findings first, binary64's second, deduplicated, errors first."""
    spec = spec_for_mac(SplitMac(BinaryK(16, 8), BinaryK(40, 30, bias=512)))
    b32, b64 = spec.findings
    assert [error is not None for error, _ in b32] == [True, False]
    assert "30 bits of precision" in str(b32[0][0]) and "2^-134" in str(b32[1][1])
    assert b64 == ()
    assert spec_for_mac(SplitMac(BinaryK(8, 4), BinaryK(8, 4))).findings == ((), ())


def test_the_spec_memo_tells_carriers_apart():
    fmt = BinaryK(8, 4)
    assert spec_for_mac(SplitMac(fmt, fmt)) != spec_for_mac(
        SplitMac(fmt, fmt, carrier=torch.float64)
    )


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (
            lambda: SplitMac(BinaryK(8, 4), carrier=torch.float16),
            ValueError,
            "torch.float32 .binary32.",
        ),
        (
            lambda: FusedMac(BinaryK(8, 4), carrier=torch.int32),
            ValueError,
            "torch.float32 .binary32.",
        ),
        (
            lambda: Quant(BinaryK(8, 4), carrier="binary64"),  # ty: ignore[invalid-argument-type]
            TypeError,
            "must be a torch.dtype",
        ),
    ],
)
def test_a_carrier_must_name_one(build, error, match):
    with pytest.raises(error, match=match):
        build()


# ------------------------------------------------------------------------------------
# The format objects themselves.


def test_binaryK_bias_defaults_to_the_middle_of_the_exponent_range():
    assert BinaryK(8, 4).bias == 2 ** (8 - 4 - 1)
    assert BinaryK(8, 4, is_signed=False).bias == 2 ** (8 - 4)
    assert BinaryK(8, 4, bias=3).bias == 3


def test_superfp_bias_has_no_default_rule():
    """The cast has no derivation for one -- tests/test_superfp_quantize.py
    runs exp_bits=3, bias=7, which no rule produces."""
    with pytest.raises(TypeError):
        SuperFP(3, 4, 8)  # ty: ignore[missing-argument]


@pytest.mark.parametrize(
    "call",
    [
        lambda: BinaryK(4, 8),
        lambda: BinaryK(8, 0),
        lambda: BinaryK(8, 4, prng_bits=-1),
        lambda: SuperFP(3, 0, 8, 7),
        lambda: SuperFP(3, 4, 0, 7),
    ],
)
def test_format_objects_validate_at_construction(call):
    with pytest.raises(ValueError):
        call()


def test_formats_are_frozen_and_hashable():
    fmt = BinaryK(8, 4)
    assert hash(fmt) == hash(BinaryK(8, 4)) and fmt == BinaryK(8, 4)
    with pytest.raises(AttributeError):
        fmt.K = 9  # ty: ignore[invalid-assignment]


# ------------------------------------------------------------------------------------
# Palette validation.


def test_palette_rejects_disagreeing_shared_fields_and_names_the_entry():
    with pytest.raises(ValueError, match=r"is_signed.*entry 1"):
        Palette([BinaryK(8, 4), BinaryK(6, 3, is_signed=False)])
    with pytest.raises(ValueError, match=r"saturation.*entry 2"):
        Palette(
            [
                BinaryK(8, 4),
                BinaryK(6, 3),
                BinaryK(6, 3, saturation=SaturationMode.SAT_FINITE),
            ]
        )


def test_palette_rejects_mixed_families():
    with pytest.raises(TypeError, match="same type"):
        Palette([BinaryK(8, 4), SuperFP(3, 4, 8, 7)])


def test_palette_rejects_empty_and_oversized():
    with pytest.raises(ValueError, match="at least one"):
        Palette([])
    with pytest.raises(ValueError, match="at most 8"):
        Palette([BinaryK(8, 4)] * 9)


def test_split_mac_rejects_mixed_families_naming_r1():
    with pytest.raises(TypeError, match="R-1"):
        SplitMac(BinaryK(8, 4), SuperFP(3, 4, 8, 7))


def test_split_mac_rejects_unequal_palettes():
    with pytest.raises(ValueError, match="same length"):
        SplitMac([BinaryK(8, 4), BinaryK(6, 3)], [BinaryK(8, 4)] * 3)


# ------------------------------------------------------------------------------------
# Quant: the elementwise half of the vocabulary.


@pytest.mark.parametrize("device", available_devices)
def test_quant_equals_the_flat_quantizer(device):
    x = torch.randn(64, device=device)
    assert torch.equal(
        Quant(BinaryK(8, 4))(x), binaryK_quantize(x, K=8, P=4, rounding_mode=RoundMode.RNE)
    )
    assert torch.equal(
        Quant(SuperFP(3, 4, 8, 7), RoundMode.RZ)(x),
        superfp_quantize(
            x,
            man_bits=3,
            exp_bits=4,
            normal_binades=8,
            bias=7,
            rounding_mode=RoundMode.RZ,
        ),
    )


@pytest.mark.parametrize("device", available_devices)
def test_a_bias_of_zero_is_a_bias(device):
    """``bias=0`` is a format, not a request for the default: the wrappers used
    to test ``not bias``, and quantized ``BinaryK(14, 4, bias=0)`` -- the top of
    binary64's range -- at P3109's bias of 512."""
    x = torch.tensor([0.0352, 0.3, 3.0], dtype=torch.float64, device=device)
    raw = torch.ops.mptorch.binaryK_quant.default(x, 14, 4, 0, 0, True, 0, 2, 0)
    assert torch.equal(binaryK_quantize(x, 14, 4, bias=0), raw)
    assert torch.equal(Quant(BinaryK(14, 4, bias=0))(x), raw)
    assert not torch.equal(binaryK_quantize(x, 14, 4), raw)
    assert _binaryK_spec(mul_K=14, mul_P=4, mul_bias=0, acc_bias=0).args[2] == 0
    assert _binaryK_spec(mul_K=14, mul_P=4, mul_bias=0, acc_bias=0).args[7] == 0
    assert _binaryK_fma_spec(fma_K=14, fma_P=4, fma_bias=0).args[3] == 0


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
@pytest.mark.parametrize("carrier", [None, torch.float64])
def test_quant_carrier_is_the_flat_quantizers(device, dtype, carrier):
    x = (torch.randn(64, device=device) * 4).to(dtype)
    assert torch.equal(
        Quant(BinaryK(8, 4), carrier=carrier)(x), binaryK_quantize(x, 8, 4, carrier=carrier)
    )


@pytest.mark.parametrize("device", available_devices)
def test_raw_ops_raise_on_requires_grad_operands(device):
    a = torch.randn(4, 5, device=device, requires_grad=True)
    b = torch.randn(5, 3, device=device)
    with pytest.raises(RuntimeError, match="qmatmul"):
        binaryK_matmul(a, b, mul_K=8, mul_P=4)
    with pytest.raises(RuntimeError, match="Quantizer"):
        binaryK_quantize(a, K=8, P=4)


@pytest.mark.parametrize("device", available_devices)
def test_raw_ops_still_run_without_grad(device):
    """Detached, or under no_grad, nothing changes -- the check is about a
    gradient that would vanish, not about the operand."""
    a = torch.randn(4, 5, device=device, requires_grad=True)
    b = torch.randn(5, 3, device=device)
    expected = binaryK_matmul(a.detach(), b, mul_K=8, mul_P=4)
    with torch.no_grad():
        assert torch.equal(binaryK_matmul(a, b, mul_K=8, mul_P=4), expected)


# ------------------------------------------------------------------------------------
# qmm / qbmm / qmatmul.


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "formats",
    [
        None,
        BinaryK(8, 4),
        SplitMac(BinaryK(8, 4), BinaryK(8, 4)),
        FusedMac(BinaryK(8, 4)),
        SuperFP(3, 4, 8, 7),
    ],
)
def test_qmatmul_accepts_every_formats_spelling(device, formats):
    a = torch.randn(2, 6, 8, device=device)
    b = torch.randn(2, 8, 4, device=device)
    out = qmatmul(a, b, formats)
    assert out.shape == (2, 6, 4)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_number_is_shorthand_for_a_symmetric_split_mac(device):
    a = torch.randn(6, 8, device=device)
    b = torch.randn(8, 4, device=device)
    assert torch.equal(
        qmatmul(a, b, BinaryK(8, 4)),
        qmatmul(a, b, SplitMac(BinaryK(8, 4), BinaryK(8, 4))),
    )


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_equals_the_flat_wrapper(device):
    a = torch.randn(6, 8, device=device)
    b = torch.randn(8, 4, device=device)
    assert torch.equal(
        qmatmul(a, b, BinaryK(8, 4)),
        binaryK_matmul(a, b, mul_K=8, mul_P=4, acc_K=8, acc_P=4),
    )


@pytest.mark.parametrize("device", available_devices)
def test_qmm_and_qbmm_check_rank(device):
    a3 = torch.randn(2, 6, 8, device=device)
    b3 = torch.randn(2, 8, 4, device=device)
    with pytest.raises(ValueError, match="2D"):
        qmm(a3, b3)
    with pytest.raises(ValueError, match="3D"):
        qbmm(a3[0], b3[0])
    with pytest.raises(ValueError, match="equal batch"):
        qbmm(a3, b3[:1])
    assert qbmm(a3, b3).shape == (2, 6, 4)
    assert qmm(a3[0], b3[0]).shape == (6, 4)


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_palette_needs_a_map(device):
    a = torch.randn(6, 8, device=device)
    b = torch.randn(8, 4, device=device)
    mac = SplitMac([BinaryK(8, 4), BinaryK(6, 3)], [BinaryK(8, 4), BinaryK(6, 3)])
    with pytest.raises(ValueError, match="prec_idx"):
        qmatmul(a, b, mac)
    idx = torch.randint(0, 2, (6, 4), device=device)
    assert qmatmul(a, b, mac, prec_idx=idx).shape == (6, 4)
    with pytest.raises(ValueError, match="no meaning"):
        qmatmul(a, b, BinaryK(8, 4), prec_idx=idx)


@pytest.mark.parametrize("device", available_devices)
def test_palette_gradients_need_their_own_maps(device):
    """One map cannot serve three passes: their outputs have three shapes."""
    a = torch.randn(6, 8, device=device, requires_grad=True)
    b = torch.randn(8, 4, device=device, requires_grad=True)
    mac = SplitMac([BinaryK(8, 4), BinaryK(6, 3)], [BinaryK(8, 4), BinaryK(6, 3)])
    fwd_idx = torch.randint(0, 2, (6, 4), device=device)

    out = qmatmul(a, b, mac, prec_idx=fwd_idx)
    with pytest.raises(ValueError, match="agrad_prec_idx"):
        out.sum().backward()

    formats = matmul_formats(
        mac,
        prec_idx=fwd_idx,
        agrad_prec_idx=torch.randint(0, 2, (6, 8), device=device),
        bgrad_prec_idx=torch.randint(0, 2, (8, 4), device=device),
    )
    a2 = a.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    qmatmul(a2, b2, formats).sum().backward()
    assert a2.grad is not None and a2.grad.shape == a2.shape
    assert b2.grad is not None and b2.grad.shape == b2.shape


# ------------------------------------------------------------------------------------
# The differentiable layer.


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((6, 8), (8, 4)),
        ((2, 6, 8), (2, 8, 4)),
        ((2, 6, 8), (8, 4)),
        ((8,), (8, 4)),
        ((6, 8), (8,)),
        ((8,), (8,)),
        ((3, 1, 6, 8), (2, 8, 4)),
    ],
)
def test_qmatmul_backward_matches_autograd_through_torch_matmul(device, a_shape, b_shape):
    """With no formats, both the values and the gradients are torch's own."""
    a = torch.randn(*a_shape, device=device, requires_grad=True)
    b = torch.randn(*b_shape, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    out = qmatmul(a, b)
    ref = a_ref @ b_ref
    assert torch.equal(out, ref)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert a.grad is not None and b.grad is not None
    assert a_ref.grad is not None and b_ref.grad is not None
    assert a.grad.shape == a.shape and b.grad.shape == b.shape
    assert torch.allclose(a.grad, a_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(b.grad, b_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_backward_runs_in_the_named_arithmetic(device):
    """Each gradient goes through the GEMM its hook names, so a coarse format
    moves the gradients and not just the output."""
    a = torch.randn(4, 8, 6, device=device, requires_grad=True)
    b = torch.randn(4, 6, 5, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    out = qmatmul(a, b, BinaryK(8, 4))
    ref = a_ref @ b_ref
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert a.grad is not None and a_ref.grad is not None
    assert not torch.equal(a.grad, a_ref.grad)
    assert (
        torch.nn.functional.cosine_similarity(a.grad.flatten(), a_ref.grad.flatten(), dim=0) > 0.9
    )


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_only_computes_the_gradients_asked_for(device):
    a = torch.randn(4, 6, device=device, requires_grad=True)
    b = torch.randn(6, 5, device=device)
    qmatmul(a, b, BinaryK(8, 4)).sum().backward()
    assert a.grad is not None and b.grad is None


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_formats_quantizers_are_applied(device):
    a = torch.randn(4, 6, device=device)
    b = torch.randn(6, 5, device=device)
    fmt = BinaryK(8, 4)
    formats = QMatmulFormats(a_quant=Quant(fmt), b_quant=Quant(fmt))
    assert torch.equal(qmatmul(a, b, formats), Quant(fmt)(a) @ Quant(fmt)(b))


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_module_matches_the_function(device):
    a = torch.randn(2, 4, 6, device=device)
    b = torch.randn(2, 6, 5, device=device)
    mac = SplitMac(BinaryK(8, 4), BinaryK(8, 4))
    assert torch.equal(QMatmul(mac).to(device)(a, b), qmatmul(a, b, mac))


def test_qmatmul_module_registers_stateful_quantizers():
    """A quantizer that is a module lands in the parent's state_dict."""

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            formats = matmul_formats(SplitMac(BinaryK(8, 4), BinaryK(8, 4)))
            formats.a_quant = Quantizer(BinaryK(8, 4))
            self.mm = QMatmul(formats)

    block = Block()
    assert any("mm.formats" in k for k in block.state_dict()) or isinstance(
        block.mm.formats.a_quant, torch.nn.Module
    )
    assert isinstance(block.mm.formats, torch.nn.Module)


# ------------------------------------------------------------------------------------
# Quantizer: the straight-through estimator.


@pytest.mark.parametrize("device", available_devices)
def test_quantizer_forward_quantizes_and_backward_passes_through(device):
    x = torch.randn(32, device=device, requires_grad=True)
    q = Quantizer(BinaryK(8, 4))
    out = q(x)
    assert torch.equal(out, Quant(BinaryK(8, 4))(x.detach()))
    g = torch.randn_like(out)
    out.backward(g)
    assert x.grad is not None and torch.equal(x.grad, g)


@pytest.mark.parametrize("device", available_devices)
def test_quantizer_backward_format_is_its_own(device):
    x = torch.randn(32, device=device, requires_grad=True)
    q = Quantizer(BinaryK(8, 4), BinaryK(6, 3))
    g = torch.randn(32, device=device)
    q(x).backward(g)
    assert x.grad is not None and torch.equal(x.grad, Quant(BinaryK(6, 3))(g))


@pytest.mark.parametrize("device", available_devices)
def test_quantizer_with_no_formats_is_the_identity(device):
    x = torch.randn(32, device=device, requires_grad=True)
    out = Quantizer()(x)
    assert torch.equal(out, x)
    g = torch.randn_like(out)
    out.backward(g)
    assert x.grad is not None and torch.equal(x.grad, g)
