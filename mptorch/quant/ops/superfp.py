"""superfp's flat functions: the elementwise quantizers and one wrapper per GEMM op.

The superfp counterpart of `binaryk`, laid out the same way: the quantizers,
the GEMM wrappers, each after its spec builder, and then the builders of the
`*_accumulated` and round-to-odd variants.
"""

from collections.abc import Sequence
from typing import Any

import torch

from mptorch import AccumulateAlgorithm, RoundMode, SaturationMode
from mptorch.number.superfp import (
    _superfp_findings,
    _superfp_storage_findings,
    check_superfp_carrier,
    check_superfp_storage,
)

from ._accumulate import _accumulated, _superfp_accumulation
from ._checks import (
    _BELOW_BINARY64,
    _CARRIER,
    _NARROW_STORAGE,
    _call_carrier,
    _checked_carrier,
    _Format,
    _format_findings,
    _narrowed,
    _palette_formats,
    _quantizer_operand,
    _refuse_widening_in_place,
    _Stored,
)
from ._odd import (
    _RNE,
    _checked_carrier_rounding_call,
    _checked_fma_quant,
    _checked_naive,
    _odd,
    _odd_accumulated,
    _palette_slots,
)
from ._operands import _gemm_mixed_nd, _gemm_nd
from ._spec import (
    _GemmSpec,
    _palette_list,
    _palette_list_or,
    _palette_pair,
    _quant_spec,
    _QuantSpec,
)

# --- elementwise quantizers --------------------------------------------------


def superfp_quantize(
    x: torch.Tensor,
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    prng_bits: int = 0,
    is_signed: bool = True,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a superfp (supernormal) format.

    The format has ``man_bits`` stored mantissa bits and ``exp_bits`` exponent
    bits, but only the top ``normal_binades`` binades have a mantissa. The
    codes of the lower binades encode zero and successive powers of two below
    the normal region, the *supernormals*. A value below the smallest
    supernormal rounds to zero or to it, as the rounding mode decides. There
    are no subnormals, and so no ``subnormals_mode``. ``bias`` is required,
    because there is no standard default for it. See
    :class:`mptorch.SuperFP`.

    Dtypes, the carrier, the storage check, ``prng_bits`` and
    differentiability are as for :func:`binaryK_quantize`.

    Args:
        x (Tensor): the tensor to round; float32, float64, float16 or
            bfloat16.
        man_bits (int): stored mantissa bits of a normal binade.
        exp_bits (int): exponent bits.
        normal_binades (int): how many of the top binades have a mantissa;
            the ``2**man_bits`` codes of each lower binade encode supernormal
            powers of two instead.
        bias (int): the exponent bias.
        prng_bits (int): the number of random bits the stochastic rounding
            modes draw below the format's last mantissa bit; ignored by the
            other modes. Default: ``0``
        is_signed (bool): whether the format has a sign bit. Default: ``True``
        rounding_mode (RoundMode): how a value that falls between two of the
            format's values is rounded. Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): what happens to a value beyond the
            largest finite value. Default: ``SaturationMode.SAT_NONE``
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`.
            Default: ``None``, ``x``'s own carrier.

    Returns:
        Tensor: a new tensor of ``x``'s shape and dtype holding the rounded
        values.

    Raises:
        ValueError: if the carrier cannot hold the format (more precision than
            the carrier has, ``man_bits + prng_bits`` above its mantissa bits,
            an exponent field wider than 30 bits, a bias that puts
            ``1 - bias`` or ``2**exp_bits - 1 - bias`` outside -2^30 to 2^30,
            a ``normal_binades`` outside ``1 .. 2**exp_bits - 1``, no finite
            normal value in its normal range), if the storage dtype can store
            none of the format's nonzero values, or if ``carrier`` is a dtype
            that names no carrier or is narrower than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode.

    Warns:
        FormatRangeWarning: if the format's range exceeds the carrier's, or a
            result at an edge of the format's range is not a value of a
            narrower storage dtype.

    Example::

        >>> x = torch.tensor([1.1, 3.3, 0.05])
        >>> superfp_quantize(x, man_bits=3, exp_bits=4, normal_binades=8, bias=7)
        tensor([1.0000, 3.2500, 0.0625])

    With ``bias=7`` the eight normal binades start at 2, so 3.3 is rounded on
    a grid of step 0.25. 1.1 and 0.05 fall in the supernormal region and are
    rounded to a power of two.
    """
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_superfp_carrier(
        man_bits, exp_bits, normal_binades, bias, saturation_mode, prng_bits, carrier=_CARRIER[wide]
    )
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_superfp_storage(
            man_bits,
            exp_bits,
            normal_binades,
            bias,
            saturation_mode,
            storage=dtype,
            elementwise=True,
        )

    out = torch.ops.mptorch.superfp_quant.default(
        _quantizer_operand(x, widen),
        man_bits,
        exp_bits,
        normal_binades,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
    )
    return _narrowed(out, dtype) if widen else out


def superfp_quantize_(
    x: torch.Tensor,
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    prng_bits: int = 0,
    is_signed: bool = True,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a superfp format, in place.

    This is the in-place version of :func:`superfp_quantize`: the same
    arguments, checks and result, written over ``x``, with no output
    allocated. Its purpose and the inputs it refuses are the same as for
    :func:`binaryK_quantize_`: a tensor that is not contiguous, a CUDA view
    that does not start on a 16-byte boundary, and a ``carrier`` wider than
    ``x``.

    Args:
        x (Tensor): the tensor to round and overwrite; float32, float64,
            float16 or bfloat16, contiguous. The format and mode arguments
            between it and ``carrier`` are :func:`superfp_quantize`'s.
        carrier (torch.dtype, optional): ``None`` or the carrier ``x`` already
            has. Default: ``None``

    Returns:
        Tensor: ``x`` itself, holding the rounded values.

    Raises:
        ValueError: as for :func:`superfp_quantize`, and if ``carrier`` is
            wider than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode, is not
            contiguous, or is a CUDA view off a 16-byte boundary.

    Warns:
        FormatRangeWarning: as for :func:`superfp_quantize`.

    Example::

        >>> x = torch.tensor([1.1, 3.3, 0.05])
        >>> superfp_quantize_(x, man_bits=3, exp_bits=4, normal_binades=8, bias=7)
        tensor([1.0000, 3.2500, 0.0625])
        >>> x
        tensor([1.0000, 3.2500, 0.0625])
    """
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_superfp_carrier(
        man_bits, exp_bits, normal_binades, bias, saturation_mode, prng_bits, carrier=_CARRIER[wide]
    )
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_superfp_storage(
            man_bits,
            exp_bits,
            normal_binades,
            bias,
            saturation_mode,
            storage=dtype,
            elementwise=True,
        )
    if widen:
        _refuse_widening_in_place("superfp_quantize", dtype)

    torch.ops.mptorch.superfp_quant_.default(
        x,
        man_bits,
        exp_bits,
        normal_binades,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
    )
    return x


def _superfp_quant_spec(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    prng_bits: int,
    is_signed: bool,
    rounding_mode: RoundMode,
    saturation_mode: SaturationMode,
    *,
    carrier: torch.dtype | None,
    in_place: bool,
) -> _QuantSpec:
    """Resolve a :func:`superfp_quantize` call (or a :func:`superfp_quantize_`
    one) once, for :class:`mptorch.quant.Quant`."""
    ops = (torch.ops.mptorch.superfp_quant.default, torch.ops.mptorch.superfp_quant_.default)
    args = (
        man_bits,
        exp_bits,
        normal_binades,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
    )
    fields = (man_bits, exp_bits, normal_binades, bias, saturation_mode)
    return _quant_spec(
        ops,
        args,
        "superfp_quantize",
        in_place,
        carrier,
        lambda c: _superfp_findings(*fields, prng_bits, c),
        lambda dtype, elementwise: _superfp_storage_findings(*fields, dtype, elementwise),
    )


# --- single-format GEMMs -----------------------------------------------------


def _superfp_spec(
    *,
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_man_bits: int | None = None,
    acc_exp_bits: int | None = None,
    acc_normal_binades: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul`'s formats into a ``_GemmSpec``.

    That function documents the arguments. This one fills in the ``acc_*``
    fallbacks to the multiply format and the zeroed accumulate format of
    ``accumulate_quant=False``, and raises binary64's errors.
    """
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if accumulate_quant:
        acc_man_bits = mul_man_bits if acc_man_bits is None else acc_man_bits
        acc_exp_bits = mul_exp_bits if acc_exp_bits is None else acc_exp_bits
        acc_normal_binades = (
            mul_normal_binades if acc_normal_binades is None else acc_normal_binades
        )
        acc_bias = mul_bias if acc_bias is None else acc_bias
    else:
        acc_man_bits = acc_man_bits or 0
        acc_exp_bits = acc_exp_bits or 0
        acc_normal_binades = acc_normal_binades or 0
        acc_bias = acc_bias or 0

    mul = (mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, saturation_mode)
    rounded: list[_Format] = [(_superfp_findings, (*mul, mul_prng_bits))]
    stored: tuple[_Stored, ...] = ()
    if accumulate_quant:
        acc = (acc_man_bits, acc_exp_bits, acc_normal_binades, acc_bias, acc_saturation_mode)
        rounded.append((_superfp_findings, (*acc, acc_prng_bits)))
        stored = ((check_superfp_storage, acc),)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_superfp.default,
        (
            mul_man_bits,
            mul_exp_bits,
            mul_normal_binades,
            mul_bias,
            mul_is_signed,
            accumulate_quant,
            acc_man_bits,
            acc_exp_bits,
            acc_normal_binades,
            acc_bias,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            acc_saturation_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        stored,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def superfp_matmul(
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
    accumulate_quant: bool = True,
    acc_man_bits: int | None = None,
    acc_exp_bits: int | None = None,
    acc_normal_binades: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
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
    """Quantized GEMM with a split multiply-accumulate in superfp formats.

    The superfp counterpart of :func:`binaryK_matmul`. Every partial product
    is rounded to the superfp format given by ``mul_man_bits``,
    ``mul_exp_bits``, ``mul_normal_binades`` and ``mul_bias``. When
    ``accumulate_quant`` is true, the running sum is rounded to the ``acc_*``
    format after every step. :func:`binaryK_matmul` documents what the two
    have in common: ``torch.matmul``'s operand rules, the transpose flags, the
    carrier, the storage check, and resolving the formats on every call.
    :func:`superfp_quantize` describes the format.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_man_bits (int): stored mantissa bits of the multiply format.
        mul_exp_bits (int): exponent bits of the multiply format.
        mul_normal_binades (int): the number of top binades of the multiply
            format that have a mantissa.
        mul_bias (int): exponent bias of the multiply format.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            multiply rounding. Default: ``0``
        accumulate_quant (bool): whether the running sum is rounded to the
            accumulate format after every step. Default: ``True``
        acc_man_bits (int, optional): the accumulate format's mantissa bits.
            Default: ``None``, ``mul_man_bits``.
        acc_exp_bits (int, optional): the accumulate format's exponent bits.
            Default: ``None``, ``mul_exp_bits``.
        acc_normal_binades (int, optional): the accumulate format's normal
            binades. Default: ``None``, ``mul_normal_binades``.
        acc_bias (int, optional): the accumulate format's exponent bias.
            Default: ``None``, ``mul_bias``.
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
        block_size (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_man_bits (int, optional): mantissa bits of the outer format, the
            format that ``BLOCK`` and ``TREE`` round the total to after each
            block is added (:func:`binaryK_matmul`'s ``outer_K``). The four
            outer fields (``outer_man_bits``, ``outer_exp_bits``,
            ``outer_normal_binades``, ``outer_bias``) are given together, or
            none of them, which leaves the total unrounded, in the carrier.
            Default: ``None``
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
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold a format (as for
            :func:`superfp_quantize`), and in the other cases listed for
            :func:`binaryK_matmul`.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if a format's range exceeds the carrier's, or the
            format the result holds has values a narrower storage dtype cannot
            represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> superfp_matmul(
        ...     a, a, mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=8, mul_bias=7
        ... )
        tensor([[ 7., 10.],
                [15., 22.]])
    """
    # A NAIVE call sets none of the accumulation keywords, and resolves
    # through the NAIVE builder with the same keywords as before the other
    # algorithms existed; this test is all it pays for them.
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_man_bits is None
        and outer_exp_bits is None
        and outer_normal_binades is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _superfp_spec(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            acc_saturation_mode=acc_saturation_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _superfp_accumulated_spec(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
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
            spec = _superfp_accumulated_odd_spec(
                mul_man_bits=mul_man_bits,
                mul_exp_bits=mul_exp_bits,
                mul_normal_binades=mul_normal_binades,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                accumulate_quant=accumulate_quant,
                acc_man_bits=acc_man_bits,
                acc_exp_bits=acc_exp_bits,
                acc_normal_binades=acc_normal_binades,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
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
            spec = _superfp_odd_spec(
                mul_man_bits=mul_man_bits,
                mul_exp_bits=mul_exp_bits,
                mul_normal_binades=mul_normal_binades,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                accumulate_quant=accumulate_quant,
                acc_man_bits=acc_man_bits,
                acc_exp_bits=acc_exp_bits,
                acc_normal_binades=acc_normal_binades,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                acc_saturation_mode=acc_saturation_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


def _superfp_fma_spec(
    *,
    fma_man_bits: int,
    fma_exp_bits: int,
    fma_normal_binades: int,
    fma_bias: int,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_fma`'s format into a ``_GemmSpec``.

    That function documents the arguments. With ``fma_quant=False`` no format
    rounds anything and no format's values are stored, so the carriers have
    nothing to check.
    """
    rounded: list[_Format] = []
    stored: tuple[_Stored, ...] = ()
    if fma_quant:
        fma = (fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias, saturation_mode)
        rounded.append((_superfp_findings, (*fma, fma_prng_bits)))
        stored = ((check_superfp_storage, fma),)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_superfp_fma.default,
        (
            fma_quant,
            fma_man_bits,
            fma_exp_bits,
            fma_normal_binades,
            fma_bias,
            fma_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            fma_prng_bits,
        ),
        stored,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def superfp_matmul_fma(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_man_bits: int,
    fma_exp_bits: int,
    fma_normal_binades: int,
    fma_bias: int,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
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
    """Quantized GEMM with a fused multiply-add in a superfp format.

    The superfp counterpart of :func:`binaryK_matmul_fma`. Each step of a dot
    product is one fused multiply-add, rounded once to the format given by
    ``fma_man_bits``, ``fma_exp_bits``, ``fma_normal_binades`` and
    ``fma_bias`` (see :func:`superfp_quantize`). With ``fma_quant=False`` the
    step is not rounded by any format. ``a`` and ``b`` follow
    ``torch.matmul``'s operand rules. The carrier, the storage check and
    resolving the format on every call are as for :func:`binaryK_matmul`.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        fma_man_bits (int): stored mantissa bits of the fused step's format.
        fma_exp_bits (int): its exponent bits.
        fma_normal_binades (int): the number of its top binades that have a
            mantissa.
        fma_bias (int): its exponent bias.
        fma_is_signed (bool): whether the format has a sign bit. Default:
            ``True``
        fma_quant (bool): whether the fused step is rounded to the format at
            all. Default: ``True``
        fma_prng_bits (int): random bits stochastic rounding draws for the fused
            rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :func:`binaryK_matmul`, with the fused step in place of the split
            one, except ``TREE``, which raises: a fused multiply-add has no
            product term to sum pairwise. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the fused step's rounding mode. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the format's overflow behavior.
            Default: ``SaturationMode.SAT_NONE``
        block_size (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_man_bits (int, optional): mantissa bits of the outer format, the
            format that ``BLOCK`` and ``TREE`` round the total to after each
            block is added (:func:`binaryK_matmul`'s ``outer_K``). The four
            outer fields (``outer_man_bits``, ``outer_exp_bits``,
            ``outer_normal_binades``, ``outer_bias``) are given together, or
            none of them, which leaves the total unrounded, in the carrier.
            Default: ``None``
        outer_exp_bits (int, optional): exponent bits of the outer format.
            Default: ``None``
        outer_normal_binades (int, optional): normal binades of the outer
            format. Default: ``None``
        outer_bias (int, optional): exponent bias of the outer format.
            Default: ``None``
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, ``fma_is_signed``.
        outer_prng_bits (int): random bits stochastic rounding draws for the
            outer rounding. Default: ``0``
        outer_saturation_mode (SaturationMode, optional): the outer format's
            overflow behavior. Default: ``None``, ``saturation_mode``.
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`;
            ``RoundMode.RO`` needs ``fma_quant=True``. Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: in the cases listed for :func:`binaryK_matmul`, and for
            ``accumulate_algorithm=TREE`` or for ``carrier_rounding=RO`` with
            ``fma_quant=False``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if the format's range exceeds the carrier's, or the
            format has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> superfp_matmul_fma(
        ...     a, a, fma_man_bits=2, fma_exp_bits=4, fma_normal_binades=8, fma_bias=7
        ... )
        tensor([[ 7., 10.],
                [16., 24.]])
    """
    # A NAIVE call sets none of the accumulation keywords, and resolves
    # through the NAIVE builder with the same keywords as before the other
    # algorithms existed; this test is all it pays for them.
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_man_bits is None
        and outer_exp_bits is None
        and outer_normal_binades is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _superfp_fma_spec(
            fma_man_bits=fma_man_bits,
            fma_exp_bits=fma_exp_bits,
            fma_normal_binades=fma_normal_binades,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _superfp_fma_accumulated_spec(
            fma_man_bits=fma_man_bits,
            fma_exp_bits=fma_exp_bits,
            fma_normal_binades=fma_normal_binades,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
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
            spec = _superfp_fma_accumulated_odd_spec(
                fma_man_bits=fma_man_bits,
                fma_exp_bits=fma_exp_bits,
                fma_normal_binades=fma_normal_binades,
                fma_bias=fma_bias,
                fma_is_signed=fma_is_signed,
                fma_quant=fma_quant,
                fma_prng_bits=fma_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
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
            spec = _superfp_fma_odd_spec(
                fma_man_bits=fma_man_bits,
                fma_exp_bits=fma_exp_bits,
                fma_normal_binades=fma_normal_binades,
                fma_bias=fma_bias,
                fma_is_signed=fma_is_signed,
                fma_quant=fma_quant,
                fma_prng_bits=fma_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


# --- palette GEMMs, whose format is chosen per output element ----------------


def _superfp_mixed_spec(
    *,
    mul_man_bits: Sequence[int],
    mul_exp_bits: Sequence[int],
    mul_normal_binades: int | Sequence[int],
    mul_bias: int | Sequence[int],
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_man_bits: int | Sequence[int] | None = None,
    acc_exp_bits: int | Sequence[int] | None = None,
    acc_normal_binades: int | Sequence[int] | None = None,
    acc_bias: int | Sequence[int] | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_mixed`'s palette into a ``_GemmSpec``.

    That function documents the arguments. This one turns every palette
    argument into a list of the palette's length, with each entry's defaults
    filled in, and raises binary64's errors.
    """
    mul_mb_l, mul_eb_l, n = _palette_pair(
        mul_man_bits, mul_exp_bits, "mul_exp_bits", "superfp_matmul_mixed"
    )
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode

    mul_nb_l = _palette_list(mul_normal_binades, n, "mul_normal_binades")
    mul_bias_l = _palette_list(mul_bias, n, "mul_bias")

    if accumulate_quant:
        acc_mb_l = _palette_list_or(acc_man_bits, mul_mb_l, n, "acc_man_bits")
        acc_eb_l = _palette_list_or(acc_exp_bits, mul_eb_l, n, "acc_exp_bits")
        acc_nb_l = _palette_list_or(acc_normal_binades, mul_nb_l, n, "acc_normal_binades")
        acc_bias_l = _palette_list_or(acc_bias, mul_bias_l, n, "acc_bias")
    else:
        acc_mb_l = [0] * n
        acc_eb_l = [0] * n
        acc_nb_l = [0] * n
        acc_bias_l = [0] * n

    rounded = _palette_formats(
        _superfp_findings,
        zip(mul_mb_l, mul_eb_l, mul_nb_l, mul_bias_l, strict=True),
        (saturation_mode, mul_prng_bits),
    )
    stored: tuple[_Stored, ...] = ()
    if accumulate_quant:
        acc_widths = tuple(zip(acc_mb_l, acc_eb_l, acc_nb_l, acc_bias_l, strict=True))
        rounded += _palette_formats(
            _superfp_findings, acc_widths, (acc_saturation_mode, acc_prng_bits)
        )
        stored = _palette_formats(check_superfp_storage, acc_widths, (acc_saturation_mode,))

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_superfp_mixed.default,
        (
            mul_mb_l,
            mul_eb_l,
            mul_nb_l,
            mul_bias_l,
            mul_is_signed,
            accumulate_quant,
            acc_mb_l,
            acc_eb_l,
            acc_nb_l,
            acc_bias_l,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            acc_saturation_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        stored,
        mixed=True,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def superfp_matmul_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_man_bits: Sequence[int],
    mul_exp_bits: Sequence[int],
    mul_normal_binades: int | Sequence[int],
    mul_bias: int | Sequence[int],
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_man_bits: int | Sequence[int] | None = None,
    acc_exp_bits: int | Sequence[int] | None = None,
    acc_normal_binades: int | Sequence[int] | None = None,
    acc_bias: int | Sequence[int] | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM whose superfp format varies per output element.

    The superfp counterpart of :func:`binaryK_matmul_mixed`, which documents
    the palette, ``prec_idx`` and the carrier. ``mul_man_bits`` and
    ``mul_exp_bits`` are per-entry sequences, and their common length is the
    palette size ``n``. ``mul_normal_binades``, ``mul_bias`` and every
    ``acc_*`` may be a scalar or an ``n``-length sequence. An ``acc_*`` may
    also be ``None``, which falls back to the corresponding ``mul_*`` entry.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        prec_idx (Tensor): the palette index of each output element, in
            ``[0, n)``.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_man_bits (Sequence[int]): stored mantissa bits of each entry's
            multiply format.
        mul_exp_bits (Sequence[int]): exponent bits of each entry's multiply
            format.
        mul_normal_binades (int or Sequence[int]): the number of top binades
            of each entry's multiply format that have a mantissa.
        mul_bias (int or Sequence[int]): exponent bias of each entry's
            multiply format.
        mul_is_signed (bool): whether the multiply formats have a sign bit;
            shared by the palette. Default: ``True``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            multiply rounding; shared by the palette. Default: ``0``
        accumulate_quant (bool): whether the running sum is rounded to the
            accumulate format after every step. Default: ``True``
        acc_man_bits (int or Sequence[int], optional): the accumulate
            formats' mantissa bits. Default: ``None``, the ``mul_man_bits``
            entry.
        acc_exp_bits (int or Sequence[int], optional): the accumulate
            formats' exponent bits. Default: ``None``, the ``mul_exp_bits``
            entry.
        acc_normal_binades (int or Sequence[int], optional): the accumulate
            formats' normal binades. Default: ``None``, the
            ``mul_normal_binades`` entry.
        acc_bias (int or Sequence[int], optional): the accumulate formats'
            exponent bias. Default: ``None``, the ``mul_bias`` entry.
        acc_is_signed (bool, optional): whether the accumulate formats have a
            sign bit. Default: ``None``, ``mul_is_signed``.
        acc_prng_bits (int): random bits stochastic rounding draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed; the palette ops implement ``NAIVE`` only.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of every format.
            Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply formats' overflow
            behavior. Default: ``SaturationMode.SAT_NONE``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            formats' overflow behavior. Default: ``None``, ``saturation_mode``.
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`, applied
            to every palette entry: ``RoundMode.RO`` runs the palette's
            round-to-odd op, and each entry's ``P`` (``P + prng_bits`` under
            the stochastic modes) is checked against the limits.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if the palette is empty or a palette argument's length is
            not ``n``, if a carrier cannot hold an entry, if an entry the
            result holds has more precision than a narrower storage dtype, if
            ``carrier`` names no carrier or is narrower than the operands, if
            ``carrier_rounding`` is neither ``RNE`` nor ``RO`` or an entry
            exceeds the round-to-odd limits, or if the operands' leading
            dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``n`` is above 8, ``prec_idx`` has the wrong shape or
            an entry outside ``[0, n)``, ``accumulate_algorithm`` is not
            ``NAIVE`` (a ``ValueError`` under ``carrier_rounding=RO``), the
            inner dimensions do not match, the operands differ in dtype or
            device, or an operand requires grad under grad mode.

    Warns:
        FormatRangeWarning: if an entry's range exceeds the carrier's, or an
            accumulate entry has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> prec_idx = torch.tensor([[0, 0], [0, 1]])
        >>> superfp_matmul_mixed(
        ...     a, a, prec_idx, mul_man_bits=[3, 2], mul_exp_bits=[4, 4],
        ...     mul_normal_binades=8, mul_bias=7,
        ... )
        tensor([[ 7., 10.],
                [15., 24.]])
    """
    build = _superfp_mixed_spec
    if carrier_rounding is not _RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        build = _superfp_mixed_odd_spec
    return _gemm_mixed_nd(
        build(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            acc_saturation_mode=acc_saturation_mode,
            carrier=carrier,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


def _superfp_fma_mixed_spec(
    *,
    fma_man_bits: Sequence[int],
    fma_exp_bits: Sequence[int],
    fma_normal_binades: int | Sequence[int],
    fma_bias: int | Sequence[int],
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_fma_mixed`'s palette into a ``_GemmSpec``.

    That function documents the arguments. ``fma_quant=False`` is passed on,
    for the op to reject.
    """
    fma_mb_l, fma_eb_l, n = _palette_pair(
        fma_man_bits, fma_exp_bits, "fma_exp_bits", "superfp_matmul_fma_mixed"
    )
    fma_nb_l = _palette_list(fma_normal_binades, n, "fma_normal_binades")
    fma_bias_l = _palette_list(fma_bias, n, "fma_bias")
    rounded: tuple[_Format, ...] = ()
    stored: tuple[_Stored, ...] = ()
    if fma_quant:
        widths = tuple(zip(fma_mb_l, fma_eb_l, fma_nb_l, fma_bias_l, strict=True))
        rounded = _palette_formats(_superfp_findings, widths, (saturation_mode, fma_prng_bits))
        stored = _palette_formats(check_superfp_storage, widths, (saturation_mode,))

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_superfp_fma_mixed.default,
        (
            fma_quant,
            fma_mb_l,
            fma_eb_l,
            fma_nb_l,
            fma_bias_l,
            fma_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            fma_prng_bits,
        ),
        stored,
        mixed=True,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def superfp_matmul_fma_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_man_bits: Sequence[int],
    fma_exp_bits: Sequence[int],
    fma_normal_binades: int | Sequence[int],
    fma_bias: int | Sequence[int],
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized fused-multiply-add GEMM whose superfp format varies per output element.

    The superfp counterpart of :func:`binaryK_matmul_fma_mixed`, which
    documents the palette, ``prec_idx`` and the carrier. ``fma_quant=False``
    is rejected here too, because an unrounded fused step has no format for
    ``prec_idx`` to choose. ``fma_man_bits`` and ``fma_exp_bits`` are
    per-entry sequences, and their common length is the palette size ``n``.
    ``fma_normal_binades`` and ``fma_bias`` may each be a scalar or an
    ``n``-length sequence.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        prec_idx (Tensor): the palette index of each output element, in
            ``[0, n)``.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        fma_man_bits (Sequence[int]): stored mantissa bits of each entry's
            format.
        fma_exp_bits (Sequence[int]): exponent bits of each entry's format.
        fma_normal_binades (int or Sequence[int]): the number of top binades
            of each entry's format that have a mantissa.
        fma_bias (int or Sequence[int]): exponent bias of each entry's
            format.
        fma_is_signed (bool): whether the formats have a sign bit; shared by
            the palette. Default: ``True``
        fma_quant (bool): must be ``True``. Default: ``True``
        fma_prng_bits (int): random bits stochastic rounding draws for the fused
            rounding; shared by the palette. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the fused steps are
            summed; the palette ops implement ``NAIVE`` only. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of every entry.
            Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): the formats' overflow behavior.
            Default: ``SaturationMode.SAT_NONE``
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`, applied
            to every palette entry: ``RoundMode.RO`` runs the palette's
            round-to-odd op, and each entry's ``P`` (``P + prng_bits`` under
            the stochastic modes) is checked against the limits.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if the palette is empty or a palette argument's length is
            not ``n``, if a carrier cannot hold an entry, if an entry the
            result holds has more precision than a narrower storage dtype, if
            ``carrier`` names no carrier or is narrower than the operands, if
            ``carrier_rounding`` is neither ``RNE`` nor ``RO`` or an entry
            exceeds the round-to-odd limits, or if the operands' leading
            dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``fma_quant`` is false or ``accumulate_algorithm``
            is not ``NAIVE`` (each a ``ValueError`` under
            ``carrier_rounding=RO``), ``n`` is above 8, ``prec_idx`` has the
            wrong shape or an entry outside ``[0, n)``, the inner dimensions do
            not match, the operands differ in dtype or device, or an operand
            requires grad under grad mode.

    Warns:
        FormatRangeWarning: if an entry's range exceeds the carrier's, or an
            entry has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> prec_idx = torch.tensor([[0, 0], [0, 1]])
        >>> superfp_matmul_fma_mixed(
        ...     a, a, prec_idx, fma_man_bits=[3, 2], fma_exp_bits=[4, 4],
        ...     fma_normal_binades=8, fma_bias=7,
        ... )
        tensor([[ 7., 10.],
                [15., 24.]])
    """
    build = _superfp_fma_mixed_spec
    if carrier_rounding is not _RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        build = _superfp_fma_mixed_odd_spec
    return _gemm_mixed_nd(
        build(
            fma_man_bits=fma_man_bits,
            fma_exp_bits=fma_exp_bits,
            fma_normal_binades=fma_normal_binades,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            carrier=carrier,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


# --- KAHAN, BLOCK and TREE: the *_accumulated ops (see `_accumulate`) --------


def _superfp_accumulated_spec(
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
    """:func:`_superfp_spec` for any accumulate algorithm; ``formats`` are its keywords."""
    spec = _superfp_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    acc_is_signed, acc_saturation = spec.args[10], spec.args[14]
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
        spec, torch.ops.mptorch.custom_matmul_superfp_accumulated.default, accumulation
    )


def _superfp_fma_accumulated_spec(
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
    """:func:`_superfp_fma_spec` for any accumulate algorithm; ``formats`` are its keywords."""
    spec = _superfp_fma_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    fma_is_signed, fma_saturation = spec.args[5], spec.args[8]
    accumulation = _superfp_accumulation(
        accumulate_algorithm,
        block_size,
        outer_man_bits,
        outer_exp_bits,
        outer_normal_binades,
        outer_bias,
        fma_is_signed if outer_is_signed is None else outer_is_signed,
        SaturationMode(fma_saturation) if outer_saturation_mode is None else outer_saturation_mode,
        outer_prng_bits,
        has_product=False,
    )
    if accumulation is None:
        return spec
    return _accumulated(
        spec, torch.ops.mptorch.custom_matmul_superfp_fma_accumulated.default, accumulation
    )


# --- the carrier step rounded to odd: the *_odd ops (see `_odd`) -------------


def _superfp_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_superfp_spec` with the carrier step rounded to odd; ``formats`` are its keywords."""
    _checked_naive(accumulate_algorithm)
    spec = _superfp_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    # mul_man_bits [0], accumulate_quant [5], acc_man_bits [6], round_mode [12], prng [15], [16]
    a = spec.args
    slots = [("multiply", a[0] + 1, a[15])] + ([("accumulate", a[6] + 1, a[16])] if a[5] else [])
    return _odd(spec, torch.ops.mptorch.custom_matmul_superfp_odd.default, slots, RoundMode(a[12]))


def _superfp_fma_odd_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    fma_quant: bool = True,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_superfp_fma_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    _checked_fma_quant(fma_quant)
    spec = _superfp_fma_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # fma_man_bits [1], round_mode [7], fma_prng_bits [9]
    return _odd(
        spec,
        torch.ops.mptorch.custom_matmul_superfp_fma_odd.default,
        [("fused", a[1] + 1, a[9])],
        RoundMode(a[7]),
    )


def _superfp_mixed_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_superfp_mixed_spec` with the carrier step rounded to odd, every entry checked."""
    _checked_naive(accumulate_algorithm)
    spec = _superfp_mixed_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    # mul_man_bits [0], accumulate_quant [5], acc_man_bits [6], round_mode [12], prng [15], [16]
    a = spec.args
    slots = _palette_slots("multiply", [m + 1 for m in a[0]], a[15])
    if a[5]:
        slots += _palette_slots("accumulate", [m + 1 for m in a[6]], a[16])
    op = torch.ops.mptorch.custom_matmul_superfp_mixed_odd.default
    return _odd(spec, op, slots, RoundMode(a[12]))


def _superfp_fma_mixed_odd_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    fma_quant: bool = True,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_superfp_fma_mixed_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    _checked_fma_quant(fma_quant)
    spec = _superfp_fma_mixed_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # fma_man_bits [1], round_mode [7], fma_prng_bits [9]
    op = torch.ops.mptorch.custom_matmul_superfp_fma_mixed_odd.default
    return _odd(spec, op, _palette_slots("fused", [m + 1 for m in a[1]], a[9]), RoundMode(a[7]))


def _superfp_accumulated_odd_spec(**keywords: Any) -> _GemmSpec:
    """:func:`_superfp_accumulated_spec` with the carrier step rounded to odd."""
    spec = _superfp_accumulated_spec(**keywords)
    a = spec.args  # as _superfp_odd_spec's, then the tail from [17]
    slots = [("multiply", a[0] + 1, a[15])] + ([("accumulate", a[6] + 1, a[16])] if a[5] else [])
    op = torch.ops.mptorch.custom_matmul_superfp_accumulated_odd.default
    return _odd_accumulated(spec, op, slots, 17, True, a[12])


def _superfp_fma_accumulated_odd_spec(*, fma_quant: bool = True, **keywords: Any) -> _GemmSpec:
    """:func:`_superfp_fma_accumulated_spec` with the carrier step rounded to odd."""
    _checked_fma_quant(fma_quant)
    spec = _superfp_fma_accumulated_spec(fma_quant=fma_quant, **keywords)
    a = spec.args  # as _superfp_fma_odd_spec's, then the tail from [10]
    op = torch.ops.mptorch.custom_matmul_superfp_fma_accumulated_odd.default
    return _odd_accumulated(spec, op, [("fused", a[1] + 1, a[9])], 10, True, a[7])
