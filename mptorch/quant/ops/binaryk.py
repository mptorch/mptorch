"""binaryK's flat functions: the elementwise quantizers and one wrapper per GEMM op.

Each GEMM wrapper resolves its formats with a spec builder
(`_binaryK_spec` and its variants), which `mptorch.quant.mac` and
`mptorch.quant.gemm` call too, so that a format object, a layer factory and a
flat call resolve to the same `_GemmSpec`. The builders of the
`*_accumulated` and round-to-odd variants follow, at the end.
"""

from collections.abc import Sequence
from typing import Any

import torch

from mptorch import AccumulateAlgorithm, RoundMode, SaturationMode, SubnormalsMode
from mptorch.number.binaryk import (
    _binaryK_findings,
    _binaryK_storage_findings,
    check_binaryK_carrier,
    check_binaryK_storage,
)

from ._accumulate import _accumulated, _binaryK_accumulation
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
    _binaryK_bias,
    _binaryK_palette_bias,
    _GemmSpec,
    _palette_list_or,
    _palette_pair,
    _quant_spec,
    _QuantSpec,
)

# --- elementwise quantizers --------------------------------------------------


def binaryK_quantize(
    x: torch.Tensor,
    K: int,
    P: int,
    bias: int | None = None,
    prng_bits: int = 0,
    is_signed: bool = True,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a binaryK floating-point format.

    binaryK is the parameterized family of IEEE P3109, the draft Standard for
    Arithmetic Formats for Machine Learning (see :class:`mptorch.BinaryK`). A
    format has ``K`` bits in total. ``P`` of them are precision (``P - 1``
    stored mantissa bits plus the implicit one), and ``K - P`` are exponent
    bits (``K - P + 1`` when ``is_signed`` is false, because there is no sign
    bit). Pass ``bias`` explicitly for a format outside P3109. The layouts of
    the OCP 8-bit formats are ``K=8, P=4, bias=7`` (E4M3) and
    ``K=8, P=3, bias=15`` (E5M2).

    The rounding happens in a *carrier*, the binary floating-point type the
    arithmetic runs in. It is binary64 for a float64 ``x``. For float32,
    float16 and bfloat16 it is binary32, applied to their float32 value. Every
    call checks the format against what its carrier can hold: at most 24 bits
    of precision and seven exponent bits in binary32, and 53 and ten in
    binary64 (see :doc:`/concepts`). The random bits of stochastic rounding are
    drawn in the carrier too, so ``P - 1 + prng_bits`` is limited to its 23 or
    52 mantissa bits.

    When ``x`` is narrower than its carrier (float16 or bfloat16, or float32
    under binary64), the result is rounded a second time when it is stored in
    ``x``'s dtype, the *storage dtype*, so the format is also checked against
    that dtype. The inputs are already values of the storage dtype, so only a
    result at an edge of the format's range can fail to be one, and that case
    warns.

    NaN inputs pass through. So do infinities, except under
    ``SaturationMode.SAT_FINITE`` and ``SAT_FINITE_EXTENDED`` (P3109's
    ``SatFinite`` in its two domains), which clamp them to the largest finite
    value.

    The function is not differentiable: rounding to a coarse grid has a zero
    derivative almost everywhere. On a tensor that requires grad, under grad
    mode, it raises and points to :class:`mptorch.quant.Quantizer`, the
    straight-through estimator. :class:`mptorch.quant.Quant` makes the same
    call from a format object.

    Args:
        x (Tensor): the tensor to round; float32, float64, float16 or
            bfloat16.
        K (int): the format's width in bits.
        P (int): the format's precision: ``P - 1`` stored mantissa bits plus
            the implicit leading bit.
        bias (int, optional): the exponent bias. Default: ``None``, which is
            P3109's ``2**(K - P - 1)`` if signed and ``2**(K - P)`` if
            unsigned, one more than the IEEE 754 bias for the same exponent
            width.
        prng_bits (int): the number of random bits the stochastic rounding
            modes (``RoundMode.SR``, ``SR_B``, ``SR_C``) draw below the
            format's last mantissa bit; ignored by the other modes.
            Default: ``0``
        is_signed (bool): whether the format has a sign bit. Default: ``True``
        rounding_mode (RoundMode): how a value that falls between two of the
            format's values is rounded. Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): what happens to a value beyond the
            largest finite value. Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the format has below its
            smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        carrier (torch.dtype, optional): ``torch.float64`` rounds any ``x`` in
            binary64: a narrower ``x`` is widened first, and the result is
            narrowed back to its dtype with a single rounding.
            ``torch.float32`` names binary32, which a float64 ``x`` refuses,
            because the carrier is never narrower than the tensor. Default:
            ``None``, ``x``'s own carrier.

    Returns:
        Tensor: a new tensor of ``x``'s shape and dtype holding the rounded
        values.

    Raises:
        ValueError: if the carrier cannot hold the format (more precision than
            the carrier has, ``P - 1 + prng_bits`` above its mantissa bits, an
            exponent field wider than 30 bits, a bias that puts ``1 - bias``
            or ``2**exp_bits - 1 - bias`` outside -2^30 to 2^30, no finite
            normal value in its normal range), if the storage dtype can store
            none of the format's nonzero values, or if ``carrier`` is a dtype
            that names no carrier or is narrower than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode.

    Warns:
        FormatRangeWarning: if the format's range exceeds the carrier's, or a
            result at an edge of the format's range is not a value of a
            narrower storage dtype. The format still quantizes correctly over
            the part that fits.

    Example::

        >>> x = torch.tensor([1.0, 1.1, 0.5])
        >>> binaryK_quantize(x, K=8, P=4)
        tensor([1.0000, 1.1250, 0.5000])
        >>> binaryK_quantize(x.half(), K=8, P=4, carrier=torch.float64)
        tensor([1.0000, 1.1250, 0.5000], dtype=torch.float16)
    """
    if bias is None:
        bias = _binaryK_bias(K, P, is_signed)
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_binaryK_carrier(
        K, P, bias, is_signed, saturation_mode, subnormals_mode, prng_bits, carrier=_CARRIER[wide]
    )
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_binaryK_storage(
            K,
            P,
            bias,
            is_signed,
            saturation_mode,
            subnormals_mode,
            storage=dtype,
            elementwise=True,
        )

    out = torch.ops.mptorch.binaryK_quant.default(
        _quantizer_operand(x, widen),
        K,
        P,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
        subnormals_mode.value,
    )
    return _narrowed(out, dtype) if widen else out


def binaryK_quantize_(
    x: torch.Tensor,
    K: int,
    P: int,
    bias: int | None = None,
    prng_bits: int = 0,
    is_signed: bool = True,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a binaryK format, in place.

    This is the in-place version of :func:`binaryK_quantize`. It takes the
    same arguments, makes the same per-call format checks, and gives the same
    result element for element, written over ``x``. That includes stochastic
    rounding with the same seed, because each element's random bits are
    selected by its index. No output is allocated, so peak memory is one
    tensor instead of two. The time is the same, because the kernel reads and
    writes the same bytes either way. Use it when the caller owns ``x``: a
    weight quantized once at load time, or an activation that nothing else
    will read.

    :func:`binaryK_quantize` handles some inputs by copying them. This
    function refuses those inputs, because the kernel would write the copy
    and leave ``x`` unchanged:

    * a tensor that is not contiguous;
    * on CUDA, a view that does not start on a 16-byte boundary (``x[1:]``),
      which the kernel's vector loads cannot address;
    * ``carrier=torch.float64`` on a float32, float16 or bfloat16 tensor,
      which rounds a float64 copy and narrows the result. A float64 tensor
      does round in place in binary64, which is its own carrier.

    Args:
        x (Tensor): the tensor to round and overwrite; float32, float64,
            float16 or bfloat16, contiguous. The format and mode arguments
            between it and ``carrier`` are :func:`binaryK_quantize`'s.
        carrier (torch.dtype, optional): ``None`` or the carrier ``x`` already
            has (``torch.float32`` for float32, float16 and bfloat16,
            ``torch.float64`` for float64). Default: ``None``

    Returns:
        Tensor: ``x`` itself, holding the rounded values.

    Raises:
        ValueError: as for :func:`binaryK_quantize`, and if ``carrier`` is
            wider than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode, is not
            contiguous, or is a CUDA view off a 16-byte boundary.

    Warns:
        FormatRangeWarning: as for :func:`binaryK_quantize`.

    Example::

        >>> x = torch.tensor([1.0, 1.1, 0.5])
        >>> binaryK_quantize_(x, K=8, P=4) is x
        True
        >>> x
        tensor([1.0000, 1.1250, 0.5000])
    """
    if bias is None:
        bias = _binaryK_bias(K, P, is_signed)
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_binaryK_carrier(
        K, P, bias, is_signed, saturation_mode, subnormals_mode, prng_bits, carrier=_CARRIER[wide]
    )
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_binaryK_storage(
            K,
            P,
            bias,
            is_signed,
            saturation_mode,
            subnormals_mode,
            storage=dtype,
            elementwise=True,
        )
    if widen:
        _refuse_widening_in_place("binaryK_quantize", dtype)

    return torch.ops.mptorch.binaryK_quant_.default(
        x,
        K,
        P,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
        subnormals_mode.value,
    )


def _binaryK_quant_spec(
    K: int,
    P: int,
    bias: int,
    prng_bits: int,
    is_signed: bool,
    rounding_mode: RoundMode,
    saturation_mode: SaturationMode,
    subnormals_mode: SubnormalsMode,
    *,
    carrier: torch.dtype | None,
    in_place: bool,
) -> _QuantSpec:
    """Resolve a :func:`binaryK_quantize` call (or a :func:`binaryK_quantize_`
    one) once, for :class:`mptorch.quant.Quant`. ``bias`` is the resolved bias."""
    op = torch.ops.mptorch.binaryK_quant_ if in_place else torch.ops.mptorch.binaryK_quant
    args = (
        K,
        P,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
        subnormals_mode.value,
    )
    fields = (K, P, bias, is_signed, saturation_mode, subnormals_mode)
    return _quant_spec(
        op.default,
        args,
        "binaryK_quantize",
        in_place,
        carrier,
        lambda c: _binaryK_findings(*fields, prng_bits, c),
        lambda dtype: _binaryK_storage_findings(*fields, dtype, True),
    )


# --- single-format GEMMs -----------------------------------------------------


def _binaryK_spec(
    *,
    mul_K: int,
    mul_P: int,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_K: int | None = None,
    acc_P: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul`'s formats into a ``_GemmSpec``.

    That function documents the arguments. This one fills in the ``acc_*``
    fallbacks to the multiply format, the default biases, and the zeroed
    accumulate format of ``accumulate_quant=False``, and raises binary64's
    errors.
    """
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if acc_subnormals_mode is None:
        acc_subnormals_mode = subnormals_mode
    if accumulate_quant:
        acc_K = mul_K if acc_K is None else acc_K
        acc_P = mul_P if acc_P is None else acc_P
    else:
        acc_K = acc_K or 0
        acc_P = acc_P or 0

    if mul_bias is None:
        mul_bias = _binaryK_bias(mul_K, mul_P, mul_is_signed)
    if accumulate_quant and acc_bias is None:
        acc_bias = _binaryK_bias(acc_K, acc_P, acc_is_signed)
    elif acc_bias is None:
        acc_bias = 0

    mul = (mul_K, mul_P, mul_bias, mul_is_signed, saturation_mode, subnormals_mode)
    rounded: list[_Format] = [(_binaryK_findings, (*mul, mul_prng_bits))]
    # The result holds values of the accumulate format (of no format with
    # accumulate_quant=False); the products are never stored.
    stored: tuple[_Stored, ...] = ()
    if accumulate_quant:
        acc = (acc_K, acc_P, acc_bias, acc_is_signed, acc_saturation_mode, acc_subnormals_mode)
        rounded.append((_binaryK_findings, (*acc, acc_prng_bits)))
        stored = ((check_binaryK_storage, acc),)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_binaryK.default,
        (
            mul_K,
            mul_P,
            mul_bias,
            mul_is_signed,
            accumulate_quant,
            acc_K,
            acc_P,
            acc_bias,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            acc_saturation_mode.value,
            acc_subnormals_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        stored,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def binaryK_matmul(
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
    accumulate_quant: bool = True,
    acc_K: int | None = None,
    acc_P: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
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
    """Quantized GEMM with a split multiply-accumulate in binaryK formats.

    Computes ``op(a) @ op(b)``, where ``op(x) = x.T`` when the corresponding
    ``trans_*`` flag is set, and ``op(x) = x`` otherwise. This is not the
    same as quantizing ``a`` and ``b`` and then calling ``torch.matmul``. It
    quantizes the *arithmetic* of each dot product. Every partial product is
    rounded to the binaryK format given by ``mul_K``/``mul_P``. When
    ``accumulate_quant`` is true, the running sum is rounded to the format
    given by ``acc_K``/``acc_P`` after every accumulation step. With
    ``accumulate_quant=False`` the running sum stays at the carrier's
    precision and only the multiply is quantized.

    Everything the formats do not round (a product or a sum before its
    rounding) is computed in a *carrier*, the binary floating-point type the
    kernel computes in, and so are the roundings. The carrier is binary64 for
    float64 operands and binary32 for float32, float16 and bfloat16, unless
    ``carrier`` names one. Every call checks both formats against that
    carrier. The result holds the accumulate format's values, so that format
    is also checked against the *storage dtype*, the operands' dtype, when it
    is narrower than the carrier. Unlike for :func:`binaryK_quantize`, every
    value of the accumulate format must then be a value of the storage dtype,
    since a sum can take any of them.

    ``a`` and ``b`` follow ``torch.matmul``'s operand rules: 1D operands are
    promoted (and their dimension is dropped from the result), leading
    dimensions broadcast against each other, and ``trans_a``/``trans_b``
    apply to the last two dimensions. Three cases need no copy:

    * An operand shared across the batch is read at stride 0 instead of being
      expanded.
    * An operand that is the transpose of a contiguous tensor flips the
      kernel's transpose flag instead of being copied.
    * ``[..., M, K] @ [K, N]`` folds its batch into ``M`` as a view.

    These cover the two shapes that a quantized attention block uses. Under
    stochastic rounding each output element's random stream is selected by
    its index into the ``[batch, M, N]`` result, so the folded and the
    batched forms give bit-identical results.

    This function resolves the format on every call. A caller that keeps one
    format across many calls should use the factories of
    `mptorch.quant.gemm`, which resolve it once per layer.
    :func:`mptorch.quant.qmatmul` is the differentiable entry point.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_K (int): width in bits of the multiply format.
        mul_P (int): precision of the multiply format.
        mul_bias (int, optional): exponent bias of the multiply format.
            Default: ``None``, P3109's ``2**(K - P - 1)`` if signed and
            ``2**(K - P)`` if unsigned.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits the stochastic rounding modes draw
            for the multiply rounding, as :func:`binaryK_quantize`'s
            ``prng_bits``. Default: ``0``
        accumulate_quant (bool): whether the running sum is rounded to the
            accumulate format after every step. Default: ``True``
        acc_K (int, optional): width of the accumulate format. Default:
            ``None``, ``mul_K``.
        acc_P (int, optional): precision of the accumulate format. Default:
            ``None``, ``mul_P``.
        acc_bias (int, optional): exponent bias of the accumulate format.
            Default: ``None``, P3109's for ``acc_K``/``acc_P``.
        acc_is_signed (bool, optional): whether the accumulate format has a
            sign bit. Default: ``None``, ``mul_is_signed``.
        acc_prng_bits (int): random bits the stochastic rounding modes draw
            for the accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed. ``NAIVE`` is the step above, in order. ``KAHAN`` carries a
            compensation term next to the sum, rounded to the accumulate
            format. ``BLOCK`` runs the step above over blocks of
            ``block_size`` products and adds each block's sum to a total,
            which the outer format rounds. ``TREE`` sums each block's
            products pairwise, rounding every pair's sum to the accumulate
            format, and adds the block's sum to the total in the same way.
            :class:`mptorch.AccumulateAlgorithm` states each precisely.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of both formats. One
            mode applies to the whole op, because the kernel is compiled for
            it. Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply format's overflow
            behavior. Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the multiply format has below
            its smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            format's overflow behavior. Default: ``None``, ``saturation_mode``.
        acc_subnormals_mode (SubnormalsMode, optional): what the accumulate
            format has below its smallest normal number. Default: ``None``,
            ``subnormals_mode``.
        block_size (int, optional): products per block. ``BLOCK`` requires
            one, which must divide 16 or be a multiple of 16 (the kernels
            work in slabs of 16 steps). ``TREE`` takes a power of two in
            [2, 256] and defaults to 16. ``NAIVE`` and ``KAHAN`` take none.
            Default: ``None``
        outer_K (int, optional): width of the outer format, the format that
            ``BLOCK`` and ``TREE`` round the total to after each block is
            added. ``None`` (together with ``outer_P``) leaves the total
            unrounded, in the carrier. When given, it is the format whose
            values the result holds, and so the one checked against a
            narrower storage dtype. Default: ``None``
        outer_P (int, optional): precision of the outer format; given
            together with ``outer_K``. Default: ``None``
        outer_bias (int, optional): exponent bias of the outer format.
            Default: ``None``, P3109's for ``outer_K``/``outer_P``.
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, the accumulate format's.
        outer_prng_bits (int): random bits the stochastic rounding modes draw
            for the outer rounding. Default: ``0``
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
        carrier_rounding (RoundMode): how the carrier rounds each product
            and sum before the formats round it. ``RoundMode.RNE`` uses the
            carrier's own round-to-nearest multiply and add, after which a
            directed rounding mode can return the wrong neighbour (the value
            is rounded twice). ``RoundMode.RO`` rounds them to odd instead,
            so that each format rounds the exact product and the exact sum.
            That holds for formats with ``P`` at most 22 under ``RNE`` and
            ``RNA`` and 23 under the directed modes and ``RO``, and with
            ``P + prng_bits`` at most 23 under ``SR`` and 22 under ``SR_B``
            and ``SR_C``, in binary32 (51, 52, 52 and 51 in binary64); a call
            outside those limits raises. ``RO`` runs the
            ``custom_matmul_*_odd`` ops, on CPU, CUDA and MPS tensors
            (binary32 only on MPS). Under ``KAHAN``, ``BLOCK`` and ``TREE``
            it runs the ``*_accumulated_odd`` ops, which also check the outer
            format against the limits and need a build with
            ``MPTORCH_ODD_ACCUMULATED=1``. With ``accumulate_quant=False``
            only the product is rounded to odd. See
            :doc:`/kernels/arithmetic`, "What the carrier rounds".
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold a format (as for
            :func:`binaryK_quantize`), if the format the result holds has
            more precision than a narrower storage dtype, if ``carrier`` names
            no carrier or is narrower than the operands, if ``block_size`` or
            the outer format does not suit ``accumulate_algorithm``, if
            ``carrier_rounding`` is neither ``RNE`` nor ``RO`` or a format
            exceeds the round-to-odd limits, or if the operands' leading
            dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode (use :func:`mptorch.quant.qmatmul`).

    Warns:
        FormatRangeWarning: if a format's range exceeds the carrier's, or the
            format the result holds (the accumulate or outer format) has
            values that a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> binaryK_matmul(a, a, mul_K=8, mul_P=4)
        tensor([[ 7., 10.],
                [15., 22.]])
        >>> binaryK_matmul(a, a, mul_K=8, mul_P=3)
        tensor([[ 7., 10.],
                [16., 24.]])

    With three stored mantissa bits (``P=4``) every product and sum is exact.
    With two (``P=3``) the sums 15 and 22 are rounded to 16 and 24.
    """
    # A NAIVE call sets none of the accumulation keywords, and resolves
    # through the NAIVE builder with the same keywords as before the other
    # algorithms existed; this test is all it pays for them.
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_K is None
        and outer_P is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _binaryK_spec(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _binaryK_accumulated_spec(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
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
            spec = _binaryK_accumulated_odd_spec(
                mul_K=mul_K,
                mul_P=mul_P,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                accumulate_quant=accumulate_quant,
                acc_K=acc_K,
                acc_P=acc_P,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
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
            spec = _binaryK_odd_spec(
                mul_K=mul_K,
                mul_P=mul_P,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                accumulate_quant=accumulate_quant,
                acc_K=acc_K,
                acc_P=acc_P,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
                acc_saturation_mode=acc_saturation_mode,
                acc_subnormals_mode=acc_subnormals_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


def _binaryK_fma_spec(
    *,
    fma_K: int,
    fma_P: int,
    fma_bias: int | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_fma`'s format into a ``_GemmSpec``.

    That function documents the arguments. With ``fma_quant=False`` no format
    rounds anything and no format's values are stored, so the carriers have
    nothing to check.
    """
    if fma_bias is None:
        fma_bias = _binaryK_bias(fma_K, fma_P, fma_is_signed)
    rounded: list[_Format] = []
    stored: tuple[_Stored, ...] = ()
    if fma_quant:
        fma = (fma_K, fma_P, fma_bias, fma_is_signed, saturation_mode, subnormals_mode)
        rounded.append((_binaryK_findings, (*fma, fma_prng_bits)))
        stored = ((check_binaryK_storage, fma),)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_binaryK_fma.default,
        (
            fma_quant,
            fma_K,
            fma_P,
            fma_bias,
            fma_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            fma_prng_bits,
        ),
        stored,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def binaryK_matmul_fma(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_K: int,
    fma_P: int,
    fma_bias: int | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
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
    """Quantized GEMM with a fused multiply-add in a binaryK format.

    The fused counterpart of :func:`binaryK_matmul`. Each step of a dot
    product is one fused multiply-add, as in hardware: ``s + a * b``, with
    ``s`` the running sum, rounded once, with no separate rounding of the
    product. A hardware FMA unit rounds only its output, so there is a single
    ``fma_K``/``fma_P`` format instead of a ``mul_*``/``acc_*`` pair. With
    ``fma_quant=False`` no format rounds the step, and the GEMM runs entirely
    in the carrier.

    ``a`` and ``b`` follow ``torch.matmul``'s operand rules. The carrier, the
    storage check and resolving the format on every call are as for
    :func:`binaryK_matmul`.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        fma_K (int): width in bits of the fused step's format.
        fma_P (int): precision of the fused step's format.
        fma_bias (int, optional): its exponent bias. Default: ``None``,
            P3109's ``2**(K - P - 1)`` if signed and ``2**(K - P)`` if
            unsigned.
        fma_is_signed (bool): whether the format has a sign bit. Default:
            ``True``
        fma_quant (bool): whether the fused step is rounded to the format at
            all. Default: ``True``
        fma_prng_bits (int): random bits stochastic rounding draws for the fused
            rounding, as :func:`binaryK_quantize`'s ``prng_bits``. Default:
            ``0``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :func:`binaryK_matmul`, with the fused step in place of the split
            one, except ``TREE``, which raises: a fused multiply-add has no
            product term to sum pairwise. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the fused step's rounding mode. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the format's overflow behavior.
            Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the format has below its
            smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        block_size (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_K (int, optional): as for :func:`binaryK_matmul`. Default: ``None``
        outer_P (int, optional): as for :func:`binaryK_matmul`. Default: ``None``
        outer_bias (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, ``fma_is_signed``.
        outer_prng_bits (int): as for :func:`binaryK_matmul`. Default: ``0``
        outer_saturation_mode (SaturationMode, optional): the outer format's
            overflow behavior. Default: ``None``, ``saturation_mode``.
        outer_subnormals_mode (SubnormalsMode, optional): what the outer format
            has below its smallest normal number. Default: ``None``,
            ``subnormals_mode``.
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
        >>> binaryK_matmul_fma(a, a, fma_K=8, fma_P=3)
        tensor([[ 7., 10.],
                [16., 24.]])
    """
    # A NAIVE call sets none of the accumulation keywords, and resolves
    # through the NAIVE builder with the same keywords as before the other
    # algorithms existed; this test is all it pays for them.
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_K is None
        and outer_P is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _binaryK_fma_spec(
            fma_K=fma_K,
            fma_P=fma_P,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _binaryK_fma_accumulated_spec(
            fma_K=fma_K,
            fma_P=fma_P,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
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
            spec = _binaryK_fma_accumulated_odd_spec(
                fma_K=fma_K,
                fma_P=fma_P,
                fma_bias=fma_bias,
                fma_is_signed=fma_is_signed,
                fma_quant=fma_quant,
                fma_prng_bits=fma_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
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
            spec = _binaryK_fma_odd_spec(
                fma_K=fma_K,
                fma_P=fma_P,
                fma_bias=fma_bias,
                fma_is_signed=fma_is_signed,
                fma_quant=fma_quant,
                fma_prng_bits=fma_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


# --- palette GEMMs, whose format is chosen per output element ----------------


def _binaryK_mixed_spec(
    *,
    mul_K: Sequence[int],
    mul_P: Sequence[int],
    mul_bias: int | Sequence[int] | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_K: int | Sequence[int] | None = None,
    acc_P: int | Sequence[int] | None = None,
    acc_bias: int | Sequence[int] | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_mixed`'s palette into a ``_GemmSpec``.

    That function documents the arguments. This one turns every palette
    argument into a list of the palette's length, with each entry's defaults
    filled in, and raises binary64's errors.
    """
    mul_K_l, mul_P_l, n = _palette_pair(mul_K, mul_P, "mul_P", "binaryK_matmul_mixed")
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if acc_subnormals_mode is None:
        acc_subnormals_mode = subnormals_mode

    mul_bias_l = _binaryK_palette_bias(mul_bias, mul_K_l, mul_P_l, mul_is_signed, "mul_bias")
    if accumulate_quant:
        acc_K_l = _palette_list_or(acc_K, mul_K_l, n, "acc_K")
        acc_P_l = _palette_list_or(acc_P, mul_P_l, n, "acc_P")
        acc_bias_l = _binaryK_palette_bias(acc_bias, acc_K_l, acc_P_l, acc_is_signed, "acc_bias")
    else:
        acc_K_l = [0] * n
        acc_P_l = [0] * n
        acc_bias_l = [0] * n

    rounded = _palette_formats(
        _binaryK_findings,
        zip(mul_K_l, mul_P_l, mul_bias_l, strict=True),
        (mul_is_signed, saturation_mode, subnormals_mode, mul_prng_bits),
    )
    stored: tuple[_Stored, ...] = ()
    if accumulate_quant:
        acc_widths = tuple(zip(acc_K_l, acc_P_l, acc_bias_l, strict=True))
        acc_shared = (acc_is_signed, acc_saturation_mode, acc_subnormals_mode)
        rounded += _palette_formats(_binaryK_findings, acc_widths, (*acc_shared, acc_prng_bits))
        stored = _palette_formats(check_binaryK_storage, acc_widths, acc_shared)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_binaryK_mixed.default,
        (
            mul_K_l,
            mul_P_l,
            mul_bias_l,
            mul_is_signed,
            accumulate_quant,
            acc_K_l,
            acc_P_l,
            acc_bias_l,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            acc_saturation_mode.value,
            acc_subnormals_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        stored,
        mixed=True,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def binaryK_matmul_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_K: Sequence[int],
    mul_P: Sequence[int],
    mul_bias: int | Sequence[int] | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_K: int | Sequence[int] | None = None,
    acc_P: int | Sequence[int] | None = None,
    acc_bias: int | Sequence[int] | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM whose binaryK format varies per output element.

    The palette counterpart of :func:`binaryK_matmul`. The arithmetic is the
    same split multiply-accumulate, a rounded multiply followed by a rounded
    accumulate, but the binaryK formats of each output element's dot product
    are chosen by ``prec_idx`` from a *palette* of up to 8 entries, each a
    pair of multiply and accumulate formats.

    ``mul_K`` and ``mul_P`` are per-entry sequences, and their common length
    is the palette size ``n``. Each of the other palette arguments may be a
    scalar (used for every entry), an ``n``-length sequence, or ``None``.
    ``None`` gives each entry the default of :func:`binaryK_matmul`: an
    ``acc_*`` argument falls back to the entry's ``mul_*`` value, and a bias to
    P3109's. Every format is checked against the call's carrier, and every
    accumulate format against a narrower storage dtype.

    ``prec_idx`` selects a palette entry per output element. Its shape must
    be the GEMM's output shape ``[M, N]`` (dense), ``[M, 1]`` (per row) or
    ``[1, N]`` (per column). It may have a leading batch dimension of ``B``
    (one map per batch element) or 1. The batch is never folded into ``M``
    here, because the map is indexed by output element.

    Any integer dtype and layout is accepted. A map that is already
    ``int32``, contiguous and on ``a``'s device is used as it is; any other
    map is converted on every call. On CUDA the bounds check copies the map's
    smallest and largest entries back to the host, which stalls the device,
    so its result is cached on the map tensor's identity and version (the
    count of its in-place writes). Keep one map and reuse it across calls: a
    map built anew for every call is checked on every call.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        prec_idx (Tensor): the palette index of each output element, in
            ``[0, n)``.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_K (Sequence[int]): width in bits of each entry's multiply format.
        mul_P (Sequence[int]): precision of each entry's multiply format.
        mul_bias (int or Sequence[int], optional): exponent bias of each
            entry's multiply format. Default: ``None``, P3109's per entry.
        mul_is_signed (bool): whether the multiply formats have a sign bit;
            shared by the palette. Default: ``True``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            multiply rounding; shared by the palette. Default: ``0``
        accumulate_quant (bool): whether the running sum is rounded to the
            accumulate format after every step. Default: ``True``
        acc_K (int or Sequence[int], optional): width of each entry's
            accumulate format. Default: ``None``, the ``mul_K`` entry.
        acc_P (int or Sequence[int], optional): precision of each entry's
            accumulate format. Default: ``None``, the ``mul_P`` entry.
        acc_bias (int or Sequence[int], optional): exponent bias of each
            entry's accumulate format. Default: ``None``, P3109's per entry.
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
        subnormals_mode (SubnormalsMode): what the multiply formats have
            below their smallest normal number.
            Default: ``SubnormalsMode.SUBNORMALS``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            formats' overflow behavior. Default: ``None``, ``saturation_mode``.
        acc_subnormals_mode (SubnormalsMode, optional): what the accumulate
            formats have below their smallest normal number. Default: ``None``,
            ``subnormals_mode``.
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
        >>> binaryK_matmul_mixed(a, a, prec_idx, mul_K=[8, 8], mul_P=[4, 3])
        tensor([[ 7., 10.],
                [15., 24.]])

    Three output elements use the ``P=4`` entry and are exact. The last uses
    the ``P=3`` entry, whose two mantissa bits round the sum 22 to 24.
    """
    build = _binaryK_mixed_spec
    if carrier_rounding is not _RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        build = _binaryK_mixed_odd_spec
    return _gemm_mixed_nd(
        build(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
            carrier=carrier,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


def _binaryK_fma_mixed_spec(
    *,
    fma_K: Sequence[int],
    fma_P: Sequence[int],
    fma_bias: int | Sequence[int] | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_fma_mixed`'s palette into a ``_GemmSpec``.

    That function documents the arguments. ``fma_quant=False`` is passed on,
    for the op to reject.
    """
    fma_K_l, fma_P_l, _ = _palette_pair(fma_K, fma_P, "fma_P", "binaryK_matmul_fma_mixed")
    fma_bias_l = _binaryK_palette_bias(fma_bias, fma_K_l, fma_P_l, fma_is_signed, "fma_bias")
    rounded: tuple[_Format, ...] = ()
    stored: tuple[_Stored, ...] = ()
    if fma_quant:
        widths = tuple(zip(fma_K_l, fma_P_l, fma_bias_l, strict=True))
        shared = (fma_is_signed, saturation_mode, subnormals_mode)
        rounded = _palette_formats(_binaryK_findings, widths, (*shared, fma_prng_bits))
        stored = _palette_formats(check_binaryK_storage, widths, shared)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_binaryK_fma_mixed.default,
        (
            fma_quant,
            fma_K_l,
            fma_P_l,
            fma_bias_l,
            fma_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            fma_prng_bits,
        ),
        stored,
        mixed=True,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def binaryK_matmul_fma_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_K: Sequence[int],
    fma_P: Sequence[int],
    fma_bias: int | Sequence[int] | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized fused-multiply-add GEMM whose binaryK format varies per output element.

    The palette counterpart of :func:`binaryK_matmul_fma`. Each step of a dot
    product is one fused multiply-add, rounded once, and the binaryK format of
    that rounding is chosen for each output element by ``prec_idx``, from a
    palette of up to 8 formats. ``prec_idx`` is as for
    :func:`binaryK_matmul_mixed`.

    ``fma_K`` and ``fma_P`` are per-entry sequences, and their common length
    is the palette size ``n``. ``fma_bias`` may be a scalar, an ``n``-length
    sequence, or ``None`` for P3109's bias per entry.

    ``fma_quant=False`` is rejected. An unrounded fused step has no format,
    so a palette would leave ``prec_idx`` nothing to choose. Use
    :func:`binaryK_matmul_fma` for the unquantized fused step.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        prec_idx (Tensor): the palette index of each output element, in
            ``[0, n)``.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        fma_K (Sequence[int]): width in bits of each entry's format.
        fma_P (Sequence[int]): precision of each entry's format.
        fma_bias (int or Sequence[int], optional): exponent bias of each
            entry's format. Default: ``None``, P3109's per entry.
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
        subnormals_mode (SubnormalsMode): what the formats have below their
            smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
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
        >>> binaryK_matmul_fma_mixed(a, a, prec_idx, fma_K=[8, 8], fma_P=[4, 3])
        tensor([[ 7., 10.],
                [15., 24.]])
    """
    build = _binaryK_fma_mixed_spec
    if carrier_rounding is not _RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        build = _binaryK_fma_mixed_odd_spec
    return _gemm_mixed_nd(
        build(
            fma_K=fma_K,
            fma_P=fma_P,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            carrier=carrier,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


# --- KAHAN, BLOCK and TREE: the *_accumulated ops (see `_accumulate`) --------


def _binaryK_accumulated_spec(
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
    """:func:`_binaryK_spec` for any accumulate algorithm; ``formats`` are its keywords."""
    spec = _binaryK_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    acc_is_signed, acc_saturation, acc_subnormals = spec.args[8], spec.args[13], spec.args[14]
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
        spec, torch.ops.mptorch.custom_matmul_binaryK_accumulated.default, accumulation
    )


def _binaryK_fma_accumulated_spec(
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
    """:func:`_binaryK_fma_spec` for any accumulate algorithm; ``formats`` are its keywords."""
    spec = _binaryK_fma_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    fma_is_signed, fma_saturation, fma_subnormals = spec.args[4], spec.args[7], spec.args[8]
    accumulation = _binaryK_accumulation(
        accumulate_algorithm,
        block_size,
        outer_K,
        outer_P,
        outer_bias,
        fma_is_signed if outer_is_signed is None else outer_is_signed,
        SaturationMode(fma_saturation) if outer_saturation_mode is None else outer_saturation_mode,
        SubnormalsMode(fma_subnormals) if outer_subnormals_mode is None else outer_subnormals_mode,
        outer_prng_bits,
        has_product=False,
    )
    if accumulation is None:
        return spec
    return _accumulated(
        spec, torch.ops.mptorch.custom_matmul_binaryK_fma_accumulated.default, accumulation
    )


# --- the carrier step rounded to odd: the *_odd ops (see `_odd`) -------------


def _binaryK_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_binaryK_spec` with the carrier step rounded to odd; ``formats`` are its keywords."""
    _checked_naive(accumulate_algorithm)
    spec = _binaryK_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # mul_P [1], accumulate_quant [4], acc_P [6], round_mode [10], prng [15], [16]
    slots = [("multiply", a[1], a[15])] + ([("accumulate", a[6], a[16])] if a[4] else [])
    return _odd(spec, torch.ops.mptorch.custom_matmul_binaryK_odd.default, slots, RoundMode(a[10]))


def _binaryK_fma_odd_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    fma_quant: bool = True,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_binaryK_fma_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    _checked_fma_quant(fma_quant)
    spec = _binaryK_fma_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # fma_P [2], round_mode [6], fma_prng_bits [9]
    return _odd(
        spec,
        torch.ops.mptorch.custom_matmul_binaryK_fma_odd.default,
        [("fused", a[2], a[9])],
        RoundMode(a[6]),
    )


def _binaryK_mixed_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_binaryK_mixed_spec` with the carrier step rounded to odd, every entry checked."""
    _checked_naive(accumulate_algorithm)
    spec = _binaryK_mixed_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # mul_P [1], accumulate_quant [4], acc_P [6], round_mode [10], prng [15], [16]
    slots = _palette_slots("multiply", a[1], a[15])
    if a[4]:
        slots += _palette_slots("accumulate", a[6], a[16])
    op = torch.ops.mptorch.custom_matmul_binaryK_mixed_odd.default
    return _odd(spec, op, slots, RoundMode(a[10]))


def _binaryK_fma_mixed_odd_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    fma_quant: bool = True,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_binaryK_fma_mixed_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    _checked_fma_quant(fma_quant)
    spec = _binaryK_fma_mixed_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # fma_P [2], round_mode [6], fma_prng_bits [9]
    op = torch.ops.mptorch.custom_matmul_binaryK_fma_mixed_odd.default
    return _odd(spec, op, _palette_slots("fused", a[2], a[9]), RoundMode(a[6]))


def _binaryK_accumulated_odd_spec(**keywords: Any) -> _GemmSpec:
    """:func:`_binaryK_accumulated_spec` with the carrier step rounded to odd."""
    spec = _binaryK_accumulated_spec(**keywords)
    a = spec.args  # as _binaryK_odd_spec's, then the tail from [17]
    slots = [("multiply", a[1], a[15])] + ([("accumulate", a[6], a[16])] if a[4] else [])
    op = torch.ops.mptorch.custom_matmul_binaryK_accumulated_odd.default
    return _odd_accumulated(spec, op, slots, 17, False, a[10])


def _binaryK_fma_accumulated_odd_spec(*, fma_quant: bool = True, **keywords: Any) -> _GemmSpec:
    """:func:`_binaryK_fma_accumulated_spec` with the carrier step rounded to odd."""
    _checked_fma_quant(fma_quant)
    spec = _binaryK_fma_accumulated_spec(fma_quant=fma_quant, **keywords)
    a = spec.args  # as _binaryK_fma_odd_spec's, then the tail from [10]
    op = torch.ops.mptorch.custom_matmul_binaryK_fma_accumulated_odd.default
    return _odd_accumulated(spec, op, [("fused", a[2], a[9])], 10, False, a[6])
