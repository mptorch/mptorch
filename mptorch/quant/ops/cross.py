"""The GEMMs whose multiply and accumulate formats are of different families.

A binaryK multiply with a superfp accumulate, or the reverse
(`csrc/common/gemm_cross.h`), runs in two NAIVE ops and their `*_accumulated`
counterparts, laid out like the single-family split ops, so that a NAIVE
cross-family call costs what a single-family one does. Their arguments are
the single-family split op's, with the accumulate format given in its own
family and no `accumulate_quant`: a sum left in the carrier has only one
family, and is that family's op with `accumulate_quant=False`. So the
accumulate format does not fall back to the multiply format's fields. Its
sign and saturation do default to the multiply format's, and a binaryK
accumulate format under a superfp multiply, which has no subnormals mode to
pass on, defaults to `SUBNORMALS`.
"""

from collections.abc import Callable
from typing import Any

import torch

from mptorch import AccumulateAlgorithm, RoundMode, SaturationMode, SubnormalsMode
from mptorch.number.binaryk import _binaryK_findings, check_binaryK_storage
from mptorch.number.superfp import _superfp_findings, check_superfp_storage

from ._accumulate import _accumulated, _binaryK_accumulation, _superfp_accumulation
from ._checks import _checked_carrier, _Format, _format_findings, _Stored
from ._odd import _RNE, _checked_carrier_rounding_call, _checked_naive, _odd, _odd_accumulated
from ._operands import _gemm_nd
from ._spec import _binaryK_bias, _GemmSpec


def _cross_spec(
    op: Callable[..., torch.Tensor],
    args: tuple[Any, ...],
    rounded: tuple[_Format, _Format],
    stored: tuple[_Stored, ...],
    carrier: torch.dtype | None,
) -> _GemmSpec:
    """Build a cross-family op's spec, collecting the findings on its two formats."""
    return _GemmSpec(
        op,
        args,
        stored,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def _binaryK_superfp_spec(
    *,
    mul_K: int,
    mul_P: int,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    acc_man_bits: int,
    acc_exp_bits: int,
    acc_normal_binades: int,
    acc_bias: int,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_superfp_matmul`'s NAIVE formats into a ``_GemmSpec``."""
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if mul_bias is None:
        mul_bias = _binaryK_bias(mul_K, mul_P, mul_is_signed)
    mul = (mul_K, mul_P, mul_bias, mul_is_signed, saturation_mode, subnormals_mode)
    acc = (
        acc_man_bits,
        acc_exp_bits,
        acc_normal_binades,
        acc_bias,
        acc_is_signed,
        acc_saturation_mode,
    )
    return _cross_spec(
        torch.ops.mptorch.custom_matmul_binaryK_superfp.default,
        (
            mul_K,
            mul_P,
            mul_bias,
            mul_is_signed,
            acc_man_bits,
            acc_exp_bits,
            acc_normal_binades,
            acc_bias,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            acc_saturation_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        ((_binaryK_findings, (*mul, mul_prng_bits)), (_superfp_findings, (*acc, acc_prng_bits))),
        ((check_superfp_storage, acc),),
        carrier,
    )


def _superfp_binaryK_spec(
    *,
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    acc_K: int,
    acc_P: int,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_binaryK_matmul`'s NAIVE formats into a ``_GemmSpec``."""
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if acc_bias is None:
        acc_bias = _binaryK_bias(acc_K, acc_P, acc_is_signed)
    mul = (mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed, saturation_mode)
    acc = (acc_K, acc_P, acc_bias, acc_is_signed, acc_saturation_mode, acc_subnormals_mode)
    return _cross_spec(
        torch.ops.mptorch.custom_matmul_superfp_binaryK.default,
        (
            mul_man_bits,
            mul_exp_bits,
            mul_normal_binades,
            mul_bias,
            mul_is_signed,
            acc_K,
            acc_P,
            acc_bias,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            acc_saturation_mode.value,
            acc_subnormals_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        ((_superfp_findings, (*mul, mul_prng_bits)), (_binaryK_findings, (*acc, acc_prng_bits))),
        ((check_binaryK_storage, acc),),
        carrier,
    )


def _binaryK_superfp_accumulated_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    block_size: int | None = None,
    outer_man_bits: int | None = None,
    outer_exp_bits: int | None = None,
    outer_normal_binades: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_binaryK_superfp_spec` for any accumulate algorithm; the outer
    format is a superfp, the accumulate format's family."""
    spec = _binaryK_superfp_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    acc_is_signed, acc_saturation = spec.args[8], spec.args[13]
    accumulation = _superfp_accumulation(
        accumulate_algorithm,
        block_size,
        outer_man_bits,
        outer_exp_bits,
        outer_normal_binades,
        outer_bias,
        acc_is_signed if outer_is_signed is None else outer_is_signed,
        SaturationMode(acc_saturation) if outer_saturation_mode is None else outer_saturation_mode,
        outer_prng_bits,
        has_product=True,
    )
    if accumulation is None:
        return spec
    return _accumulated(
        spec,
        torch.ops.mptorch.custom_matmul_binaryK_superfp_accumulated.default,
        accumulation,
    )


def _superfp_binaryK_accumulated_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    block_size: int | None = None,
    outer_K: int | None = None,
    outer_P: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    outer_subnormals_mode: SubnormalsMode | None = None,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_superfp_binaryK_spec` for any accumulate algorithm; the outer
    format is a binaryK, the accumulate format's family."""
    spec = _superfp_binaryK_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    acc_is_signed, acc_saturation, acc_subnormals = spec.args[8], spec.args[12], spec.args[13]
    accumulation = _binaryK_accumulation(
        accumulate_algorithm,
        block_size,
        outer_K,
        outer_P,
        outer_bias,
        acc_is_signed if outer_is_signed is None else outer_is_signed,
        SaturationMode(acc_saturation) if outer_saturation_mode is None else outer_saturation_mode,
        SubnormalsMode(acc_subnormals) if outer_subnormals_mode is None else outer_subnormals_mode,
        outer_prng_bits,
        has_product=True,
    )
    if accumulation is None:
        return spec
    return _accumulated(
        spec,
        torch.ops.mptorch.custom_matmul_superfp_binaryK_accumulated.default,
        accumulation,
    )


def binaryK_superfp_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_K: int,
    mul_P: int,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    acc_man_bits: int,
    acc_exp_bits: int,
    acc_normal_binades: int,
    acc_bias: int,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    block_size: int | None = None,
    outer_man_bits: int | None = None,
    outer_exp_bits: int | None = None,
    outer_normal_binades: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM whose products are binaryK and whose sums are superfp.

    A split multiply-accumulate across the two families: each step computes
    ``s = round_acc(s + round_mul(a * b))`` in the carrier, as in
    :func:`binaryK_matmul`, with the product rounded to the binaryK format
    ``mul_K``/``mul_P`` and the running sum to the superfp format given by the
    four ``acc_*`` fields. The accumulate format is required: a sum left in
    the carrier is :func:`binaryK_matmul` with ``accumulate_quant=False``.
    :func:`binaryK_matmul` documents what the two have in common
    (``torch.matmul``'s operand rules, the transpose flags, the carrier, the
    storage check, resolving the formats on every call), and
    :func:`superfp_quantize` describes the accumulate format. The result
    holds the accumulate format's values or, under ``BLOCK`` or ``TREE`` with
    an outer format, the outer format's, which belongs to the accumulate
    format's family.

    :class:`mptorch.quant.SplitMac` with a :class:`mptorch.BinaryK` multiply
    and a :class:`mptorch.SuperFP` accumulate resolves to this op.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_K (int): width in bits of the binaryK multiply format.
        mul_P (int): precision of the multiply format.
        mul_bias (int, optional): exponent bias of the multiply format.
            Default: ``None``, P3109's ``2**(K - P - 1)`` if signed and
            ``2**(K - P)`` if unsigned.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            multiply rounding. Default: ``0``
        acc_man_bits (int): stored mantissa bits of the superfp accumulate
            format.
        acc_exp_bits (int): exponent bits of the accumulate format.
        acc_normal_binades (int): the number of top binades of the
            accumulate format that have a mantissa.
        acc_bias (int): exponent bias of the accumulate format.
        acc_is_signed (bool, optional): whether the accumulate format has a
            sign bit. Default: ``None``, ``mul_is_signed``.
        acc_prng_bits (int): random bits stochastic rounding draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :func:`binaryK_matmul`. Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of both formats. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply format's overflow
            behavior. Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the multiply format has below
            its smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            format's overflow behavior. Default: ``None``, ``saturation_mode``.
        block_size (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_man_bits (int, optional): mantissa bits of the superfp outer
            format, as for :func:`superfp_matmul`. The four outer fields are
            given together, or none of them. Default: ``None``
        outer_exp_bits (int, optional): exponent bits of the outer format.
            Default: ``None``
        outer_normal_binades (int, optional): normal binades of the outer
            format. Default: ``None``
        outer_bias (int, optional): exponent bias of the outer format.
            Default: ``None``
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, the accumulate format's.
        outer_prng_bits (int): random bits stochastic rounding draws for the
            outer rounding. Default: ``0``
        outer_saturation_mode (SaturationMode, optional): the outer format's
            overflow behavior. Default: ``None``, the accumulate format's.
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`.
            ``torch.float64`` widens narrower operands to binary64 and
            narrows the result back to their dtype with a single rounding.
            ``torch.float32`` refuses float64 operands. Default: ``None``,
            the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`, applied
            to both formats: ``RoundMode.RO`` runs the round-to-odd
            cross-family op (its ``*_accumulated_odd`` counterpart under
            ``KAHAN``, ``BLOCK`` and ``TREE``), and the multiply, accumulate
            and outer formats are each checked against the limits.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold a format, and in the other
            cases listed for :func:`binaryK_matmul`.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if a format's range exceeds the carrier's, or the
            accumulate format has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> binaryK_superfp_matmul(
        ...     a, a, mul_K=8, mul_P=4,
        ...     acc_man_bits=2, acc_exp_bits=4, acc_normal_binades=8, acc_bias=7,
        ... )
        tensor([[ 7., 10.],
                [16., 24.]])

    Every product is exact in Binary8p4. The sums 15 and 22 fall halfway
    between values of the superfp accumulate format, whose two mantissa bits
    space them 2 apart in [8, 16) and 4 apart in [16, 32), and round to the
    even mantissa: 16 and 24.
    """
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_man_bits is None
        and outer_exp_bits is None
        and outer_normal_binades is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _binaryK_superfp_spec(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            acc_saturation_mode=acc_saturation_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _binaryK_superfp_accumulated_spec(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            acc_saturation_mode=acc_saturation_mode,
            carrier=carrier,
            block_size=block_size,
            outer_man_bits=outer_man_bits,
            outer_exp_bits=outer_exp_bits,
            outer_normal_binades=outer_normal_binades,
            outer_bias=outer_bias,
            outer_is_signed=outer_is_signed,
            outer_prng_bits=outer_prng_bits,
            outer_saturation_mode=outer_saturation_mode,
        )
    else:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_man_bits is not None
            or outer_exp_bits is not None
            or outer_normal_binades is not None
            or outer_bias is not None
        ):
            spec = _binaryK_superfp_accumulated_odd_spec(
                mul_K=mul_K,
                mul_P=mul_P,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                acc_man_bits=acc_man_bits,
                acc_exp_bits=acc_exp_bits,
                acc_normal_binades=acc_normal_binades,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
                acc_saturation_mode=acc_saturation_mode,
                carrier=carrier,
                block_size=block_size,
                outer_man_bits=outer_man_bits,
                outer_exp_bits=outer_exp_bits,
                outer_normal_binades=outer_normal_binades,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
            )
        else:
            spec = _binaryK_superfp_odd_spec(
                mul_K=mul_K,
                mul_P=mul_P,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                acc_man_bits=acc_man_bits,
                acc_exp_bits=acc_exp_bits,
                acc_normal_binades=acc_normal_binades,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
                acc_saturation_mode=acc_saturation_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


def superfp_binaryK_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    acc_K: int,
    acc_P: int,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    block_size: int | None = None,
    outer_K: int | None = None,
    outer_P: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    outer_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM whose products are superfp and whose sums are binaryK.

    :func:`binaryK_superfp_matmul` with the families swapped: each product is
    rounded to the superfp format given by the four ``mul_*`` fields, and the
    running sum to the binaryK format ``acc_K``/``acc_P``, which is required
    (a sum left in the carrier is :func:`superfp_matmul` with
    ``accumulate_quant=False``). Under ``BLOCK`` or ``TREE`` the outer format
    is a binaryK, as for :func:`binaryK_matmul`.
    :class:`mptorch.quant.SplitMac` with a :class:`mptorch.SuperFP` multiply
    and a :class:`mptorch.BinaryK` accumulate resolves to this op.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_man_bits (int): stored mantissa bits of the superfp multiply
            format.
        mul_exp_bits (int): exponent bits of the multiply format.
        mul_normal_binades (int): the number of top binades of the multiply
            format that have a mantissa.
        mul_bias (int): exponent bias of the multiply format.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            multiply rounding. Default: ``0``
        acc_K (int): width in bits of the binaryK accumulate format.
        acc_P (int): precision of the accumulate format.
        acc_bias (int, optional): exponent bias of the accumulate format.
            Default: ``None``, P3109's for ``acc_K``/``acc_P``.
        acc_is_signed (bool, optional): whether the accumulate format has a
            sign bit. Default: ``None``, ``mul_is_signed``.
        acc_prng_bits (int): random bits stochastic rounding draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :func:`binaryK_matmul`. Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of both formats. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply format's overflow
            behavior. Default: ``SaturationMode.SAT_NONE``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            format's overflow behavior. Default: ``None``, ``saturation_mode``.
        acc_subnormals_mode (SubnormalsMode): what the accumulate format has
            below its smallest normal number. It does not default to the
            multiply format's, since a superfp has no subnormals mode.
            Default: ``SubnormalsMode.SUBNORMALS``
        block_size (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_K (int, optional): width of the binaryK outer format, as
            :func:`binaryK_matmul`'s. Default: ``None``
        outer_P (int, optional): precision of the outer format; given
            together with ``outer_K``. Default: ``None``
        outer_bias (int, optional): exponent bias of the outer format.
            Default: ``None``, P3109's for ``outer_K``/``outer_P``.
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, the accumulate format's.
        outer_prng_bits (int): random bits stochastic rounding draws for the
            outer rounding. Default: ``0``
        outer_saturation_mode (SaturationMode, optional): the outer format's
            overflow behavior. Default: ``None``, the accumulate format's.
        outer_subnormals_mode (SubnormalsMode, optional): what the outer
            format has below its smallest normal number. Default: ``None``,
            the accumulate format's.
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`.
            ``torch.float64`` widens narrower operands to binary64 and
            narrows the result back to their dtype with a single rounding.
            ``torch.float32`` refuses float64 operands. Default: ``None``,
            the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`, applied
            to both formats: ``RoundMode.RO`` runs the round-to-odd
            cross-family op (its ``*_accumulated_odd`` counterpart under
            ``KAHAN``, ``BLOCK`` and ``TREE``), and the multiply, accumulate
            and outer formats are each checked against the limits.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold a format, and in the other
            cases listed for :func:`binaryK_matmul`.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if a format's range exceeds the carrier's, or the
            accumulate format has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> superfp_binaryK_matmul(
        ...     a, a, mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=8, mul_bias=7,
        ...     acc_K=8, acc_P=3,
        ... )
        tensor([[ 7., 10.],
                [16., 24.]])
    """
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_K is None
        and outer_P is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _superfp_binaryK_spec(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _superfp_binaryK_accumulated_spec(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
            carrier=carrier,
            block_size=block_size,
            outer_K=outer_K,
            outer_P=outer_P,
            outer_bias=outer_bias,
            outer_is_signed=outer_is_signed,
            outer_prng_bits=outer_prng_bits,
            outer_saturation_mode=outer_saturation_mode,
            outer_subnormals_mode=outer_subnormals_mode,
        )
    else:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_K is not None
            or outer_P is not None
            or outer_bias is not None
        ):
            spec = _superfp_binaryK_accumulated_odd_spec(
                mul_man_bits=mul_man_bits,
                mul_exp_bits=mul_exp_bits,
                mul_normal_binades=mul_normal_binades,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                acc_K=acc_K,
                acc_P=acc_P,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                acc_saturation_mode=acc_saturation_mode,
                acc_subnormals_mode=acc_subnormals_mode,
                carrier=carrier,
                block_size=block_size,
                outer_K=outer_K,
                outer_P=outer_P,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
                outer_subnormals_mode=outer_subnormals_mode,
            )
        else:
            spec = _superfp_binaryK_odd_spec(
                mul_man_bits=mul_man_bits,
                mul_exp_bits=mul_exp_bits,
                mul_normal_binades=mul_normal_binades,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                acc_K=acc_K,
                acc_P=acc_P,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                acc_saturation_mode=acc_saturation_mode,
                acc_subnormals_mode=acc_subnormals_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


# --- the carrier step rounded to odd: the *_odd ops (see `_odd`) -------------


def _binaryK_superfp_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_binaryK_superfp_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    spec = _binaryK_superfp_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # mul_P [1], acc_man_bits [4], round_mode [10], prng [14], [15]
    slots = [("multiply", a[1], a[14]), ("accumulate", a[4] + 1, a[15])]
    op = torch.ops.mptorch.custom_matmul_binaryK_superfp_odd.default
    return _odd(spec, op, slots, RoundMode(a[10]))


def _superfp_binaryK_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_superfp_binaryK_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    spec = _superfp_binaryK_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # mul_man_bits [0], acc_P [6], round_mode [10], prng [14], [15]
    slots = [("multiply", a[0] + 1, a[14]), ("accumulate", a[6], a[15])]
    op = torch.ops.mptorch.custom_matmul_superfp_binaryK_odd.default
    return _odd(spec, op, slots, RoundMode(a[10]))


def _binaryK_superfp_accumulated_odd_spec(**keywords: Any) -> _GemmSpec:
    """:func:`_binaryK_superfp_accumulated_spec` with the carrier step rounded to odd."""
    spec = _binaryK_superfp_accumulated_spec(**keywords)
    a = spec.args  # as _binaryK_superfp_odd_spec's, then the tail from [16]
    slots = [("multiply", a[1], a[14]), ("accumulate", a[4] + 1, a[15])]
    op = torch.ops.mptorch.custom_matmul_binaryK_superfp_accumulated_odd.default
    return _odd_accumulated(spec, op, slots, 16, True, a[10])


def _superfp_binaryK_accumulated_odd_spec(**keywords: Any) -> _GemmSpec:
    """:func:`_superfp_binaryK_accumulated_spec` with the carrier step rounded to odd."""
    spec = _superfp_binaryK_accumulated_spec(**keywords)
    a = spec.args  # as _superfp_binaryK_odd_spec's, then the tail from [16]
    slots = [("multiply", a[0] + 1, a[14]), ("accumulate", a[6], a[15])]
    op = torch.ops.mptorch.custom_matmul_superfp_binaryK_accumulated_odd.default
    return _odd_accumulated(spec, op, slots, 16, False, a[10])
