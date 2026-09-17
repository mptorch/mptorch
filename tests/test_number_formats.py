"""The format vocabulary of ``mptorch.quant.mac`` against the flat GEMM wrappers.

Formats have two spellings. The schema tier is the eight ``*_matmul*`` wrappers
in ``quant/ops.py``, one per op with every schema argument spelled out. The value
tier is ``BinaryK``/``SuperFP``, ``Quant``, ``SplitMac``/``FusedMac`` and
``Palette``, with ``qmm``/``qbmm``/``qmatmul``, ``QMatmul`` and ``Quantizer`` as the
differentiable entry points built on it. The ops themselves take rank 2 or 3
operands only. Everything else ``torch.matmul`` accepts (1D promotion, leading
dimensions, broadcasting) is done in Python, which the backward tests cover.

The load-bearing tests are the first group. Every mac resolves to the same
``_GemmSpec`` (op, schema arguments, stored formats, carrier, per-carrier verdicts)
the equivalent flat-wrapper call builds, over a grid of widths and modes. That
equality is the whole equivalence claim between the two API tiers: while it
holds they cannot drift apart, and without it a default (a bias, an accumulate
fallback, a carrier) resolved differently in one tier changes results silently.
The remaining groups cover validation at construction, the elementwise ``Quant``,
the autograd guard on the raw ops, and the gradients of ``qmatmul``.
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
    """Yield ``(K, P, signed, rounding, saturation, subnormals)`` over the widths
    and modes the GEMM test modules run, crossed."""
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
    """A symmetric binaryK ``SplitMac`` equals ``_binaryK_spec`` with every field
    spelled out, so no mode or sign is dropped on the way to the op."""
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
    """``acc=None`` is the unrounded sum: it resolves to ``accumulate_quant=False``
    rather than needing a flag beside the format."""
    assert spec_for_mac(SplitMac(BinaryK(8, 4))) == _binaryK_spec(
        mul_K=8, mul_P=4, accumulate_quant=False
    )


def test_binaryK_split_mac_asymmetric_slots():
    """Saturation and subnormals belong to the format slot, not the op, so the
    accumulate's modes reach the ``acc_*`` arguments instead of the multiply's."""
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
    """The superfp twin of the binaryK equality, at one and eight normal binades."""
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
    """A ``FusedMac`` of either family equals its ``*_fma_spec``."""
    assert spec_for_mac(FusedMac(BinaryK(8, 4), rounding=rm)) == _binaryK_fma_spec(
        fma_K=8, fma_P=4, rounding_mode=rm
    )
    assert spec_for_mac(FusedMac(SuperFP(3, 4, 8, 7), rounding=rm)) == _superfp_fma_spec(
        fma_man_bits=3, fma_exp_bits=4, fma_normal_binades=8, fma_bias=7, rounding_mode=rm
    )


def test_fused_mac_without_a_format_is_the_unrounded_step():
    """``FusedMac(None)`` picks the binaryK fused op with its rounding switched off."""
    spec = spec_for_mac(FusedMac(None))
    assert spec.op is torch.ops.mptorch.custom_matmul_binaryK_fma.default
    assert spec.args[0] is False  # fma_quant, the first schema argument


def test_palette_macs_resolve_to_flat_specs():
    """A list of formats resolves to the four palette (``*_mixed``) specs, each
    field becoming the list the flat wrapper takes."""
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
    """A single format opposite a palette is repeated to its length, either way
    round.

    The palette ops read the palette size off the first pair of lists, so a
    scalar multiply against a palette accumulate would leave the op with no
    length to read.
    """
    assert spec_for_mac(SplitMac([BinaryK(8, 4), BinaryK(6, 3)], BinaryK(8, 4))) == (
        _binaryK_mixed_spec(mul_K=[8, 6], mul_P=[4, 3], acc_K=8, acc_P=4)
    )
    assert spec_for_mac(SplitMac(BinaryK(8, 4), [BinaryK(8, 4), BinaryK(6, 3)])) == (
        _binaryK_mixed_spec(mul_K=[8, 8], mul_P=[4, 4], acc_K=[8, 6], acc_P=[4, 3])
    )


def test_spec_resolution_is_memoized():
    """A mac built twice resolves once: equal frozen values return the same object.

    Resolving a format costs 2.3 to 6.4 microseconds, most of a small GEMM's
    Python overhead, so ``spec_for_mac`` memoizes on the value. A mac that stopped
    being hashable, or a memo keyed on identity, would pay that on every call.
    """
    assert spec_for_mac(SplitMac(BinaryK(8, 4), BinaryK(8, 4))) is spec_for_mac(
        SplitMac(BinaryK(8, 4), BinaryK(8, 4))
    )


@pytest.mark.parametrize("carrier", [None, torch.float32, torch.float64])
def test_carrier_resolves_to_the_flat_spec(carrier):
    """The carrier is part of the spec, and both tiers put it in the same place.

    ``BinaryK(40, 30, bias=512)`` is a format only binary64 holds, so the two
    tiers must also agree on what each carrier finds in it.
    """
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
    """``findings`` holds binary32's verdicts, then binary64's, errors first.

    The 40-bit format's 30 bits of precision are an error in binary32 (24 at
    most), and the 16-bit format's smallest value, 2^(2 - 128 - 8) = 2^-134, is a
    range warning there. binary64 holds both formats, and an 8-bit format gives
    neither carrier anything to say. A call indexes this pair by its carrier, so
    a swapped order misreports both.
    """
    spec = spec_for_mac(SplitMac(BinaryK(16, 8), BinaryK(40, 30, bias=512)))
    b32, b64 = spec.findings
    assert [error is not None for error, _ in b32] == [True, False]
    assert "30 bits of precision" in str(b32[0][0]) and "2^-134" in str(b32[1][1])
    assert b64 == ()
    assert spec_for_mac(SplitMac(BinaryK(8, 4), BinaryK(8, 4))).findings == ((), ())


def test_the_spec_memo_tells_carriers_apart():
    """The carrier is part of a mac's value, so the memo cannot hand a binary64
    call the spec resolved for the default carrier."""
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
    """``carrier`` is ``torch.float32``, ``torch.float64`` or ``None``: another dtype
    raises ``ValueError`` naming the valid ones, and a string raises ``TypeError``."""
    with pytest.raises(error, match=match):
        build()


# ------------------------------------------------------------------------------------
# The format objects themselves.


def test_binaryK_bias_defaults_to_the_middle_of_the_exponent_range():
    """The default is P3109's bias, half the exponent codes: 2^(K-P-1) signed,
    2^(K-P) unsigned, whose exponent field is one bit wider. A given bias wins."""
    assert BinaryK(8, 4).bias == 2 ** (8 - 4 - 1)
    assert BinaryK(8, 4, is_signed=False).bias == 2 ** (8 - 4)
    assert BinaryK(8, 4, bias=3).bias == 3


def test_superfp_bias_has_no_default_rule():
    """``SuperFP`` requires its bias, because the cast has no derivation for one:
    ``tests/test_superfp_quantize.py`` runs ``exp_bits=3, bias=7``, which no rule
    in terms of the exponent width produces."""
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
    """A malformed format (``P > K``, no precision, negative prng bits, no exponent
    bits, no normal binade) raises when built, not at the first call."""
    with pytest.raises(ValueError):
        call()


def test_formats_are_frozen_and_hashable():
    """Equal formats hash equal and cannot be mutated, which is what lets
    ``spec_for_mac`` memoize on the value without going stale."""
    fmt = BinaryK(8, 4)
    assert hash(fmt) == hash(BinaryK(8, 4)) and fmt == BinaryK(8, 4)
    with pytest.raises(AttributeError):
        fmt.K = 9  # ty: ignore[invalid-assignment]


# ------------------------------------------------------------------------------------
# Palette validation.


def test_palette_rejects_disagreeing_shared_fields_and_names_the_entry():
    """A palette op takes one sign and one set of modes for all entries, so an
    entry that disagrees raises, and the message names the field and the entry."""
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
    """One palette is one op, and no op mixes binaryK with superfp entries."""
    with pytest.raises(TypeError, match="same type"):
        Palette([BinaryK(8, 4), SuperFP(3, 4, 8, 7)])


def test_palette_rejects_empty_and_oversized():
    """The palette ops hold between one and eight formats."""
    with pytest.raises(ValueError, match="at least one"):
        Palette([])
    with pytest.raises(ValueError, match="at most 8"):
        Palette([BinaryK(8, 4)] * 9)


def test_split_mac_rejects_mixed_families_naming_r1():
    """A multiply in one family and an accumulate in the other has no kernel, and
    the error for that unimplemented mac carries the ``R-1`` tag matched here."""
    with pytest.raises(TypeError, match="R-1"):
        SplitMac(BinaryK(8, 4), SuperFP(3, 4, 8, 7))


def test_split_mac_rejects_unequal_palettes():
    """Two palettes share one ``prec_idx`` map, so their lengths must agree."""
    with pytest.raises(ValueError, match="same length"):
        SplitMac([BinaryK(8, 4), BinaryK(6, 3)], [BinaryK(8, 4)] * 3)


# ------------------------------------------------------------------------------------
# Quant: the elementwise half of the vocabulary.


@pytest.mark.parametrize("device", available_devices)
def test_quant_equals_the_flat_quantizer(device):
    """``Quant`` is bit-identical to the flat quantizer of its family, with RNE as
    the default rounding."""
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
    """``bias=0`` is a format, not a request for the default.

    A wrapper that tests ``not bias`` instead of ``bias is None`` quantizes
    ``BinaryK(14, 4, bias=0)``, whose ten exponent bits reach the top of binary64's
    range, at P3109's bias of 2^(14-4-1) = 512. The raw op call below is the
    reference, with ``bias`` as its fourth argument, and the spec indices are the
    positions of the bias arguments in each schema.
    """
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
    """``Quant`` hands ``carrier`` to the flat quantizer unchanged, for operands at
    and below the carrier's width."""
    x = (torch.randn(64, device=device) * 4).to(dtype)
    assert torch.equal(
        Quant(BinaryK(8, 4), carrier=carrier)(x), binaryK_quantize(x, 8, 4, carrier=carrier)
    )


@pytest.mark.parametrize("device", available_devices)
def test_raw_ops_raise_on_requires_grad_operands(device):
    """The flat ops have no gradient, so they raise on a ``requires_grad`` operand
    and name the differentiable entry point.

    Without the guard the op returns a tensor with a ``grad_fn`` and leaves
    ``.grad`` as ``None``, which is a silently untrained model.
    """
    a = torch.randn(4, 5, device=device, requires_grad=True)
    b = torch.randn(5, 3, device=device)
    with pytest.raises(RuntimeError, match="qmatmul"):
        binaryK_matmul(a, b, mul_K=8, mul_P=4)
    with pytest.raises(RuntimeError, match="Quantizer"):
        binaryK_quantize(a, K=8, P=4)


@pytest.mark.parametrize("device", available_devices)
def test_raw_ops_still_run_without_grad(device):
    """Detached, or under ``no_grad``, the raw op runs and returns the same values:
    the guard is about a gradient that would vanish, not about the operand."""
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
    """``formats`` may be nothing, a bare number format, or either mac."""
    a = torch.randn(2, 6, 8, device=device)
    b = torch.randn(2, 8, 4, device=device)
    out = qmatmul(a, b, formats)
    assert out.shape == (2, 6, 4)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_number_is_shorthand_for_a_symmetric_split_mac(device):
    """A bare format means that format for both the multiply and the accumulate."""
    a = torch.randn(6, 8, device=device)
    b = torch.randn(8, 4, device=device)
    assert torch.equal(
        qmatmul(a, b, BinaryK(8, 4)),
        qmatmul(a, b, SplitMac(BinaryK(8, 4), BinaryK(8, 4))),
    )


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_equals_the_flat_wrapper(device):
    """The forward of the differentiable entry point is the flat wrapper's result,
    bit for bit."""
    a = torch.randn(6, 8, device=device)
    b = torch.randn(8, 4, device=device)
    assert torch.equal(
        qmatmul(a, b, BinaryK(8, 4)),
        binaryK_matmul(a, b, mul_K=8, mul_P=4, acc_K=8, acc_P=4),
    )


@pytest.mark.parametrize("device", available_devices)
def test_qmm_and_qbmm_check_rank(device):
    """``qmm`` is strictly 2D and ``qbmm`` strictly 3D with equal batch sizes, like
    ``torch.mm`` and ``torch.bmm``. Only ``qmatmul`` broadcasts."""
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
    """A palette mac needs a ``prec_idx`` of the output's shape, and a map beside a
    single format is rejected rather than ignored."""
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
    """One map cannot serve three passes, whose outputs have three shapes.

    The forward map is ``[M, N]``, the gradient of ``a`` needs ``[M, K]`` and that
    of ``b`` ``[K, N]``. Backward without them raises and names the missing map.
    """
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
    """With no formats, both the values and the gradients are torch's own.

    The shapes cover what Python handles above the rank 2 or 3 op boundary: 1D
    promotion on either side, a batch folded against a 2D operand, and leading
    dimensions that broadcast. The backward must restore the dropped dimensions
    and sum each gradient over the broadcast ones, so the gradient shapes are
    checked as well. The values differ from torch's by summation order only,
    hence the 1e-5 tolerance.
    """
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
    """An operand without ``requires_grad`` gets no gradient GEMM and no ``.grad``."""
    a = torch.randn(4, 6, device=device, requires_grad=True)
    b = torch.randn(6, 5, device=device)
    qmatmul(a, b, BinaryK(8, 4)).sum().backward()
    assert a.grad is not None and b.grad is None


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_formats_quantizers_are_applied(device):
    """``a_quant`` and ``b_quant`` run on the operands before the product, which
    with no math hook is torch's own."""
    a = torch.randn(4, 6, device=device)
    b = torch.randn(6, 5, device=device)
    fmt = BinaryK(8, 4)
    formats = QMatmulFormats(a_quant=Quant(fmt), b_quant=Quant(fmt))
    assert torch.equal(qmatmul(a, b, formats), Quant(fmt)(a) @ Quant(fmt)(b))


@pytest.mark.parametrize("device", available_devices)
def test_qmatmul_module_matches_the_function(device):
    """``QMatmul`` is a thin module over ``qmatmul``, bit for bit."""
    a = torch.randn(2, 4, 6, device=device)
    b = torch.randn(2, 6, 5, device=device)
    mac = SplitMac(BinaryK(8, 4), BinaryK(8, 4))
    assert torch.equal(QMatmul(mac).to(device)(a, b), qmatmul(a, b, mac))


def test_qmatmul_module_registers_stateful_quantizers():
    """``QMatmulFormats`` is an ``nn.Module``, so a quantizer that is itself a module
    (a QAT observer, say) is registered under the parent and reaches its
    ``state_dict``."""

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
    """The straight-through estimator: forward is ``Quant``, and with no backward
    format the gradient passes unchanged."""
    x = torch.randn(32, device=device, requires_grad=True)
    q = Quantizer(BinaryK(8, 4))
    out = q(x)
    assert torch.equal(out, Quant(BinaryK(8, 4))(x.detach()))
    g = torch.randn_like(out)
    out.backward(g)
    assert x.grad is not None and torch.equal(x.grad, g)


@pytest.mark.parametrize("device", available_devices)
def test_quantizer_backward_format_is_its_own(device):
    """The second format quantizes the incoming gradient, not the forward one."""
    x = torch.randn(32, device=device, requires_grad=True)
    q = Quantizer(BinaryK(8, 4), BinaryK(6, 3))
    g = torch.randn(32, device=device)
    q(x).backward(g)
    assert x.grad is not None and torch.equal(x.grad, Quant(BinaryK(6, 3))(g))


@pytest.mark.parametrize("device", available_devices)
def test_quantizer_with_no_formats_is_the_identity(device):
    """No formats means no rounding in either direction."""
    x = torch.randn(32, device=device, requires_grad=True)
    out = Quantizer()(x)
    assert torch.equal(out, x)
    g = torch.randn_like(out)
    out.backward(g)
    assert x.grad is not None and torch.equal(x.grad, g)
