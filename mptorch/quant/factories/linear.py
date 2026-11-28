"""Factories that make a ``QLinear``'s matrix products run in quantized GEMMs.

Each factory returns a :class:`~mptorch.quant.QAffineFormats` whose three
math hooks call a quantized GEMM op (``_hooks.py`` says how
``CustomArithLinear`` calls them):

* the eight factories of the simulated formats (:func:`binaryK_gemm_formats`,
  its ``_fma`` and superfp versions, the two cross-family ones and the two
  fixed-point ones). They differ only in which op their hooks call, so they
  share ``_gemm_formats`` and differ only in the arguments they resolve;
* :func:`block_gemm_formats`, which makes a ``QLinear`` multiply block-format
  operands.

Each factory resolves its formats once, when it is called, into a
``_GemmSpec``. A layer's formats are fixed for its lifetime, and redoing that
work (the ``acc_*`` fallbacks, the default exponent biases and the carrier
checks) on every forward and backward pass would cost 2.3 to 6.4 microseconds
per call (measured by ``dev/benchmarks/timing/python_call_overhead.py``),
more than the rest of the Python call path.

The factories set only the math hooks. The caller adds the elementwise
operand quantizers (``weight_quant``, ``input_quant`` and the other
``*_quant`` slots) to the returned container, so the rounding of the
operands and the arithmetic of the dot product can be chosen independently.
"""

from dataclasses import replace
from functools import partial

import torch
import torch.nn.functional as F

from mptorch.number import (
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    Overflow,
    RoundMode,
    SaturationMode,
    SubnormalsMode,
)

from ..mac import BlockQuant, _as_block_quant, _as_quant
from ..modules.format import QAffineFormats
from ..ops._odd import _checked_carrier_rounding_call
from ..ops._spec import _GemmSpec, _run_gemm
from ..ops.binaryk import (
    _binaryK_accumulated_odd_spec,
    _binaryK_accumulated_spec,
    _binaryK_fma_accumulated_odd_spec,
    _binaryK_fma_accumulated_spec,
    _binaryK_fma_odd_spec,
    _binaryK_odd_spec,
)
from ..ops.block import _block_gemm_spec
from ..ops.cross import (
    _binaryK_superfp_accumulated_odd_spec,
    _binaryK_superfp_accumulated_spec,
    _binaryK_superfp_odd_spec,
    _superfp_binaryK_accumulated_odd_spec,
    _superfp_binaryK_accumulated_spec,
    _superfp_binaryK_odd_spec,
)
from ..ops.fixedpoint import _fixedpoint_fma_spec, _fixedpoint_spec
from ..ops.superfp import (
    _superfp_accumulated_odd_spec,
    _superfp_accumulated_spec,
    _superfp_fma_accumulated_odd_spec,
    _superfp_fma_accumulated_spec,
    _superfp_fma_odd_spec,
    _superfp_odd_spec,
)
from ._hooks import (
    _block_linear_fwd,
    _block_linear_igrad,
    _block_linear_wgrad,
    _gemm_formats,
    _linear_fwd,
    _spec_matmul,
    _with_output,
)

__all__ = [
    "binaryK_gemm_formats",
    "superfp_gemm_formats",
    "binaryK_gemm_formats_fma",
    "superfp_gemm_formats_fma",
    "binaryK_superfp_gemm_formats",
    "superfp_binaryK_gemm_formats",
    "fixedpoint_gemm_formats",
    "fixedpoint_gemm_formats_fma",
    "block_gemm_formats",
]


def binaryK_gemm_formats(
    mul_K: int,
    mul_P: int,
    *,
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
) -> QAffineFormats:
    """Build a ``QAffineFormats`` whose matrix products run in a binaryK GEMM.

    The returned container sets ``fwd_math``, ``bwd_igrad_math`` and
    ``bwd_wgrad_math``, so a Linear layer's forward product and both of its
    gradient products run through :func:`~mptorch.quant.binaryK_matmul`,
    which takes these same arguments. Every product is rounded to the
    multiply format (the ``mul_*`` arguments). When ``accumulate_quant`` is
    set, every partial sum is rounded to the accumulate format (the ``acc_*``
    arguments).

    The formats are resolved once, here, so each pass costs only the op. Each
    pass checks the formats against the carrier of the tensors it runs on
    (binary64 for a float64 layer and binary32 otherwise, unless ``carrier``
    names one), and warns with :class:`mptorch.FormatRangeWarning` about any
    part of their range that the carrier cannot hold.

    Args:
        mul_K (int): total width in bits of the multiply format.
        mul_P (int): precision of the multiply format: its significand bits,
            including the implicit leading bit.
        mul_bias (int, optional): exponent bias of the multiply format.
            ``None`` takes IEEE P3109's, ``2**(K-P-1)`` if signed and
            ``2**(K-P)`` if unsigned. Default: ``None``
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): number of random bits drawn to round a product
            stochastically; read only under a stochastic rounding mode
            (``RoundMode.SR``, ``SR_B`` or ``SR_C``). Default: ``0``
        accumulate_quant (bool): whether the running sum is rounded to the
            accumulate format after every step. ``False`` leaves it in the
            carrier and ignores the ``acc_*`` widths. Default: ``True``
        acc_K (int, optional): total width of the accumulate format. ``None``
            takes ``mul_K``. Default: ``None``
        acc_P (int, optional): precision of the accumulate format. ``None``
            takes ``mul_P``. Default: ``None``
        acc_bias (int, optional): exponent bias of the accumulate format.
            ``None`` takes P3109's for ``acc_K`` and ``acc_P``. Default: ``None``
        acc_is_signed (bool, optional): whether the accumulate format has a
            sign bit. ``None`` takes ``mul_is_signed``. Default: ``None``
        acc_prng_bits (int): number of random bits drawn to round a partial
            sum stochastically. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed: ``NAIVE``, ``KAHAN``, ``BLOCK`` or ``TREE``, as for
            :func:`mptorch.quant.binaryK_matmul`.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of every format the GEMM
            rounds to. There is a single mode because the kernels take it as
            a compile-time template parameter. Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): what the multiply format does on
            overflow. Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the multiply format has below
            its smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        acc_saturation_mode (SaturationMode, optional): what the accumulate
            format does on overflow. ``None`` takes ``saturation_mode``.
            Default: ``None``
        acc_subnormals_mode (SubnormalsMode, optional): what the accumulate
            format has below its smallest normal number. ``None`` takes
            ``subnormals_mode``. Default: ``None``
        block_size (int, optional): the block size of ``BLOCK`` and ``TREE``,
            as for :func:`mptorch.quant.binaryK_matmul`. Default: ``None``
        outer_K (int, optional): total width of the outer format, which
            ``BLOCK`` and ``TREE`` round the sum of the blocks to. The outer
            format's other arguments are ``outer_P``, ``outer_bias``,
            ``outer_is_signed``, ``outer_prng_bits``,
            ``outer_saturation_mode`` and ``outer_subnormals_mode``, all as
            documented for :func:`mptorch.quant.binaryK_matmul`.
            Default: ``None``, which leaves the sum of the blocks unrounded.
        carrier (torch.dtype, optional): the float arithmetic the GEMM rounds
            in. ``None`` follows the layer's dtype (binary64 for float64,
            binary32 for the others). ``torch.float64`` also runs a float32 or
            half-precision layer in binary64, and narrows each result back to
            the layer's dtype. ``torch.float32`` on a float64 layer raises
            when the layer is called. Default: ``None``
        carrier_rounding (RoundMode): ``RoundMode.RO`` rounds each operation
            in the carrier to odd instead of to nearest, in every pass of the
            layer, as for :func:`mptorch.quant.binaryK_matmul`.
            Default: ``RoundMode.RNE``

    Returns:
        QAffineFormats: with the three ``*_math`` hooks set and every
        ``*_quant`` slot left ``None``.

    Raises:
        ValueError: for a format that no carrier can hold (more than 53 bits
            of precision, an exponent field wider than 30 bits, a bias that
            puts ``1 - bias`` or ``2**exp_bits - 1 - bias`` outside -2^30 to
            2^30, or too many stochastic-rounding bits to fit in the
            significand), a
            ``carrier`` other than ``torch.float32``, ``torch.float64`` or
            ``None``, or a ``carrier_rounding`` other than ``RNE`` or ``RO``.
        TypeError: for a ``carrier`` that is not a ``torch.dtype``.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QLinear, Quant, binaryK_gemm_formats
        >>> formats = binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=16, acc_P=11)
        >>> formats.input_quant = Quant(BinaryK(8, 4))
        >>> layer = QLinear(256, 64, formats=formats)
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """
    if carrier_rounding is not RoundMode.RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_K is not None
            or outer_P is not None
            or outer_bias is not None
        ):
            odd_spec = _binaryK_accumulated_odd_spec(
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
                block_size=block_size,
                outer_K=outer_K,
                outer_P=outer_P,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
                outer_subnormals_mode=outer_subnormals_mode,
                carrier=carrier,
            )
        else:
            odd_spec = _binaryK_odd_spec(
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
        return _gemm_formats(partial(_run_gemm, odd_spec))
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
        block_size=block_size,
        outer_K=outer_K,
        outer_P=outer_P,
        outer_bias=outer_bias,
        outer_is_signed=outer_is_signed,
        outer_prng_bits=outer_prng_bits,
        outer_saturation_mode=outer_saturation_mode,
        outer_subnormals_mode=outer_subnormals_mode,
        carrier=carrier,
    )
    return _gemm_formats(partial(_run_gemm, spec))


def binaryK_gemm_formats_fma(
    fma_K: int,
    fma_P: int,
    *,
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
) -> QAffineFormats:
    """The fused multiply-add version of :func:`binaryK_gemm_formats`.

    Each step of the dot product is one fused multiply-add,
    ``round_fma(s + a * b)``, rounded once to the fused format (the ``fma_*``
    arguments; :func:`~mptorch.quant.binaryK_matmul_fma`), in place of a
    rounded product followed by a separately rounded sum. Otherwise it works
    as :func:`binaryK_gemm_formats` does: the formats are resolved when it is
    called, and each pass checks them against its carrier and warns.

    Args:
        fma_K (int): total width in bits of the fused format.
        fma_P (int): precision of the fused format, including the implicit
            leading bit.
        fma_bias (int, optional): exponent bias; ``None`` takes IEEE P3109's.
            Default: ``None``
        fma_is_signed (bool): whether the fused format has a sign bit.
            Default: ``True``
        fma_quant (bool): whether each fused step is rounded to the fused
            format. ``False`` leaves the running sum in the carrier.
            Default: ``True``
        fma_prng_bits (int): number of random bits drawn to round a step
            stochastically; read only under a stochastic rounding mode
            (``RoundMode.SR``, ``SR_B`` or ``SR_C``). Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the steps are summed:
            ``NAIVE``, ``KAHAN`` or ``BLOCK``, as for
            :func:`mptorch.quant.binaryK_matmul`. ``TREE`` raises, since it
            adds products in pairs and a fused multiply-add has no separate
            product. Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of each fused step (and
            of the outer format). Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): what the fused format does on
            overflow. Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the fused format has below its
            smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        block_size (int, optional): the block size of ``BLOCK``, as for
            :func:`mptorch.quant.binaryK_matmul`. Default: ``None``
        outer_K (int, optional): total width of the outer format, which
            ``BLOCK`` rounds the sum of the blocks to. The outer format's
            other arguments are ``outer_P``, ``outer_bias``,
            ``outer_is_signed``, ``outer_prng_bits``,
            ``outer_saturation_mode`` and ``outer_subnormals_mode``, all as
            documented for :func:`mptorch.quant.binaryK_matmul`.
            Default: ``None``, which leaves the sum of the blocks unrounded.
        carrier (torch.dtype, optional): as for :func:`binaryK_gemm_formats`.
            Default: ``None``
        carrier_rounding (RoundMode): ``RoundMode.RO`` rounds each operation
            in the carrier to odd, in every pass of the layer, as for
            :func:`mptorch.quant.binaryK_matmul_fma`. Default: ``RoundMode.RNE``

    Returns:
        QAffineFormats: with the three ``*_math`` hooks set.

    Raises:
        ValueError: as for :func:`binaryK_gemm_formats`, and for ``TREE``.
        TypeError: as for :func:`binaryK_gemm_formats`.

    Example::

        >>> import torch
        >>> from mptorch.quant import QLinear, binaryK_gemm_formats_fma
        >>> layer = QLinear(256, 64, formats=binaryK_gemm_formats_fma(8, 4))
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """
    if carrier_rounding is not RoundMode.RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_K is not None
            or outer_P is not None
            or outer_bias is not None
        ):
            odd_spec = _binaryK_fma_accumulated_odd_spec(
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
                block_size=block_size,
                outer_K=outer_K,
                outer_P=outer_P,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
                outer_subnormals_mode=outer_subnormals_mode,
                carrier=carrier,
            )
        else:
            odd_spec = _binaryK_fma_odd_spec(
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
        return _gemm_formats(partial(_run_gemm, odd_spec))
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
        block_size=block_size,
        outer_K=outer_K,
        outer_P=outer_P,
        outer_bias=outer_bias,
        outer_is_signed=outer_is_signed,
        outer_prng_bits=outer_prng_bits,
        outer_saturation_mode=outer_saturation_mode,
        outer_subnormals_mode=outer_subnormals_mode,
        carrier=carrier,
    )
    return _gemm_formats(partial(_run_gemm, spec))


def superfp_gemm_formats(
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    *,
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
) -> QAffineFormats:
    """The superfp version of :func:`binaryK_gemm_formats`.

    The matrix products run through :func:`~mptorch.quant.superfp_matmul`.
    Products are rounded to the superfp multiply format (the ``mul_*``
    arguments). When ``accumulate_quant`` is set, partial sums are rounded to
    the superfp accumulate format (the ``acc_*`` arguments). A superfp format
    has a full mantissa only in the top ``normal_binades`` binades of its
    range, and uses the rest of its codes for *supernormals*: powers of two
    with no mantissa, which extend the range downward. It has no subnormals,
    so there is no ``subnormals_mode``. :func:`superfp_gemm_formats_fma` is
    the fused multiply-add version.

    Args:
        mul_man_bits (int): mantissa bits of the multiply format.
        mul_exp_bits (int): exponent bits of the multiply format.
        mul_normal_binades (int): number of binades with a full mantissa in
            the multiply format.
        mul_bias (int): exponent bias of the multiply format. It is required,
            since superfp has no standard default.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): number of random bits drawn to round a product
            stochastically. Default: ``0``
        accumulate_quant (bool): whether partial sums are rounded to the
            accumulate format. Default: ``True``
        acc_man_bits (int, optional): mantissa bits of the accumulate format.
            ``None`` takes ``mul_man_bits``. Default: ``None``
        acc_exp_bits (int, optional): exponent bits of the accumulate format.
            ``None`` takes ``mul_exp_bits``. Default: ``None``
        acc_normal_binades (int, optional): binades with a full mantissa in
            the accumulate format. ``None`` takes ``mul_normal_binades``.
            Default: ``None``
        acc_bias (int, optional): exponent bias of the accumulate format.
            ``None`` takes ``mul_bias``. Default: ``None``
        acc_is_signed (bool, optional): whether the accumulate format has a
            sign bit. ``None`` takes ``mul_is_signed``. Default: ``None``
        acc_prng_bits (int): number of random bits drawn to round a partial
            sum stochastically. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed: ``NAIVE``, ``KAHAN``, ``BLOCK`` or ``TREE``, as for
            :func:`mptorch.quant.binaryK_matmul`.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of every format the GEMM
            rounds to. Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): what the multiply format does on
            overflow. Default: ``SaturationMode.SAT_NONE``
        acc_saturation_mode (SaturationMode, optional): what the accumulate
            format does on overflow. ``None`` takes ``saturation_mode``.
            Default: ``None``
        block_size (int, optional): the block size of ``BLOCK`` and ``TREE``,
            as for :func:`mptorch.quant.binaryK_matmul`. Default: ``None``
        outer_man_bits (int, optional): mantissa bits of the outer format,
            which ``BLOCK`` and ``TREE`` round the sum of the blocks to. The
            outer format's other arguments are ``outer_exp_bits``,
            ``outer_normal_binades``, ``outer_bias``, ``outer_is_signed``,
            ``outer_prng_bits`` and ``outer_saturation_mode``, all as
            documented for :func:`mptorch.quant.superfp_matmul`.
            Default: ``None``, which leaves the sum of the blocks unrounded.
        carrier (torch.dtype, optional): as for :func:`binaryK_gemm_formats`.
            Default: ``None``
        carrier_rounding (RoundMode): ``RoundMode.RO`` rounds each operation
            in the carrier to odd, in every pass of the layer, as for
            :func:`mptorch.quant.superfp_matmul`. Default: ``RoundMode.RNE``

    Returns:
        QAffineFormats: with the three ``*_math`` hooks set.

    Raises:
        ValueError: as for :func:`binaryK_gemm_formats`, and for a
            ``normal_binades`` that leaves no codes for supernormals.
        TypeError: as for :func:`binaryK_gemm_formats`.

    Example::

        >>> import torch
        >>> from mptorch.quant import QLinear, superfp_gemm_formats
        >>> formats = superfp_gemm_formats(3, 4, 8, 7, accumulate_quant=False)
        >>> layer = QLinear(256, 64, formats=formats)
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """
    if carrier_rounding is not RoundMode.RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_man_bits is not None
            or outer_exp_bits is not None
            or outer_normal_binades is not None
            or outer_bias is not None
        ):
            odd_spec = _superfp_accumulated_odd_spec(
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
                block_size=block_size,
                outer_man_bits=outer_man_bits,
                outer_exp_bits=outer_exp_bits,
                outer_normal_binades=outer_normal_binades,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
                carrier=carrier,
            )
        else:
            odd_spec = _superfp_odd_spec(
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
        return _gemm_formats(partial(_run_gemm, odd_spec))
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
        block_size=block_size,
        outer_man_bits=outer_man_bits,
        outer_exp_bits=outer_exp_bits,
        outer_normal_binades=outer_normal_binades,
        outer_bias=outer_bias,
        outer_is_signed=outer_is_signed,
        outer_prng_bits=outer_prng_bits,
        outer_saturation_mode=outer_saturation_mode,
        carrier=carrier,
    )
    return _gemm_formats(partial(_run_gemm, spec))


def superfp_gemm_formats_fma(
    fma_man_bits: int,
    fma_exp_bits: int,
    fma_normal_binades: int,
    fma_bias: int,
    *,
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
) -> QAffineFormats:
    """The superfp version of :func:`binaryK_gemm_formats_fma`.

    Each step of the dot product is one fused multiply-add, rounded once to
    the superfp fused format (:func:`~mptorch.quant.superfp_matmul_fma`).
    The arguments are the multiply-format arguments of
    :func:`superfp_gemm_formats` with the prefix ``fma_``, plus ``fma_quant``,
    as in :func:`binaryK_gemm_formats_fma`.

    Args:
        fma_man_bits (int): mantissa bits of the fused format.
        fma_exp_bits (int): exponent bits of the fused format.
        fma_normal_binades (int): number of binades with a full mantissa in
            the fused format.
        fma_bias (int): exponent bias of the fused format (required).
        fma_is_signed (bool): whether the fused format has a sign bit.
            Default: ``True``
        fma_quant (bool): whether each fused step is rounded to the fused
            format. ``False`` leaves the running sum in the carrier.
            Default: ``True``
        fma_prng_bits (int): number of random bits drawn to round a step
            stochastically. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the steps are summed:
            ``NAIVE``, ``KAHAN`` or ``BLOCK``, as for
            :func:`mptorch.quant.binaryK_matmul`. ``TREE`` raises, since a
            fused multiply-add has no separate product to add in pairs.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of each fused step (and
            of the outer format). Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): what the fused format does on
            overflow. Default: ``SaturationMode.SAT_NONE``
        block_size (int, optional): the block size of ``BLOCK``, as for
            :func:`mptorch.quant.binaryK_matmul`. Default: ``None``
        outer_man_bits (int, optional): mantissa bits of the outer format,
            which ``BLOCK`` rounds the sum of the blocks to. The outer
            format's other arguments are ``outer_exp_bits``,
            ``outer_normal_binades``, ``outer_bias``, ``outer_is_signed``,
            ``outer_prng_bits`` and ``outer_saturation_mode``, all as
            documented for :func:`mptorch.quant.superfp_matmul`.
            Default: ``None``, which leaves the sum of the blocks unrounded.
        carrier (torch.dtype, optional): as for :func:`binaryK_gemm_formats`.
            Default: ``None``
        carrier_rounding (RoundMode): ``RoundMode.RO`` rounds each operation
            in the carrier to odd, in every pass of the layer, as for
            :func:`mptorch.quant.superfp_matmul_fma`. Default: ``RoundMode.RNE``

    Returns:
        QAffineFormats: with the three ``*_math`` hooks set.

    Raises:
        ValueError: as for :func:`superfp_gemm_formats`, and for ``TREE``.
        TypeError: as for :func:`binaryK_gemm_formats`.

    Example::

        >>> import torch
        >>> from mptorch.quant import QLinear, superfp_gemm_formats_fma
        >>> formats = superfp_gemm_formats_fma(3, 4, 8, 7)
        >>> layer = QLinear(256, 64, formats=formats)
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """
    if carrier_rounding is not RoundMode.RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_man_bits is not None
            or outer_exp_bits is not None
            or outer_normal_binades is not None
            or outer_bias is not None
        ):
            odd_spec = _superfp_fma_accumulated_odd_spec(
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
                block_size=block_size,
                outer_man_bits=outer_man_bits,
                outer_exp_bits=outer_exp_bits,
                outer_normal_binades=outer_normal_binades,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
                carrier=carrier,
            )
        else:
            odd_spec = _superfp_fma_odd_spec(
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
        return _gemm_formats(partial(_run_gemm, odd_spec))
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
        block_size=block_size,
        outer_man_bits=outer_man_bits,
        outer_exp_bits=outer_exp_bits,
        outer_normal_binades=outer_normal_binades,
        outer_bias=outer_bias,
        outer_is_signed=outer_is_signed,
        outer_prng_bits=outer_prng_bits,
        outer_saturation_mode=outer_saturation_mode,
        carrier=carrier,
    )
    return _gemm_formats(partial(_run_gemm, spec))


def binaryK_superfp_gemm_formats(
    mul_K: int,
    mul_P: int,
    acc_man_bits: int,
    acc_exp_bits: int,
    acc_normal_binades: int,
    acc_bias: int,
    *,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
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
) -> QAffineFormats:
    """A version of :func:`binaryK_gemm_formats` with binaryK products and superfp sums.

    The matrix products run through
    :func:`~mptorch.quant.binaryK_superfp_matmul`, which documents every
    argument. Products are rounded to the binaryK multiply format (the
    ``mul_*`` arguments) and partial sums to the superfp accumulate format
    (the ``acc_*`` arguments). The accumulate format is required: for an
    unrounded sum, use :func:`binaryK_gemm_formats` with
    ``accumulate_quant=False``. Under ``BLOCK`` or ``TREE`` the outer format
    is a superfp format too. ``carrier_rounding=RoundMode.RO`` rounds each
    operation in the carrier to odd, in every pass, as for
    :func:`~mptorch.quant.binaryK_superfp_matmul`.

    Returns:
        QAffineFormats: with the three ``*_math`` hooks set.

    Raises:
        ValueError: as for :func:`superfp_gemm_formats`.
        TypeError: as for :func:`binaryK_gemm_formats`.

    Example::

        >>> import torch
        >>> from mptorch.quant import QLinear, binaryK_superfp_gemm_formats
        >>> formats = binaryK_superfp_gemm_formats(8, 4, 3, 4, 8, 7)
        >>> layer = QLinear(256, 64, formats=formats)
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """
    if carrier_rounding is not RoundMode.RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_man_bits is not None
            or outer_exp_bits is not None
            or outer_normal_binades is not None
            or outer_bias is not None
        ):
            odd_spec = _binaryK_superfp_accumulated_odd_spec(
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
                block_size=block_size,
                outer_man_bits=outer_man_bits,
                outer_exp_bits=outer_exp_bits,
                outer_normal_binades=outer_normal_binades,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
                carrier=carrier,
            )
        else:
            odd_spec = _binaryK_superfp_odd_spec(
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
        return _gemm_formats(partial(_run_gemm, odd_spec))
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
        block_size=block_size,
        outer_man_bits=outer_man_bits,
        outer_exp_bits=outer_exp_bits,
        outer_normal_binades=outer_normal_binades,
        outer_bias=outer_bias,
        outer_is_signed=outer_is_signed,
        outer_prng_bits=outer_prng_bits,
        outer_saturation_mode=outer_saturation_mode,
        carrier=carrier,
    )
    return _gemm_formats(partial(_run_gemm, spec))


def superfp_binaryK_gemm_formats(
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    acc_K: int,
    acc_P: int,
    *,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
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
) -> QAffineFormats:
    """A version of :func:`superfp_gemm_formats` with superfp products and binaryK sums.

    The matrix products run through
    :func:`~mptorch.quant.superfp_binaryK_matmul`, which documents every
    argument. Products are rounded to the superfp multiply format (the
    ``mul_*`` arguments) and partial sums to the binaryK accumulate format
    (the ``acc_*`` arguments). The accumulate format is required: for an
    unrounded sum, use :func:`superfp_gemm_formats` with
    ``accumulate_quant=False``. Under ``BLOCK`` or ``TREE`` the outer format
    is a binaryK format too. ``carrier_rounding=RoundMode.RO`` rounds each
    operation in the carrier to odd, in every pass, as for
    :func:`~mptorch.quant.superfp_binaryK_matmul`.

    Returns:
        QAffineFormats: with the three ``*_math`` hooks set.

    Raises:
        ValueError: as for :func:`superfp_gemm_formats`.
        TypeError: as for :func:`binaryK_gemm_formats`.

    Example::

        >>> import torch
        >>> from mptorch.quant import QLinear, superfp_binaryK_gemm_formats
        >>> formats = superfp_binaryK_gemm_formats(3, 4, 8, 7, 16, 11)
        >>> layer = QLinear(256, 64, formats=formats)
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """
    if carrier_rounding is not RoundMode.RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_K is not None
            or outer_P is not None
            or outer_bias is not None
        ):
            odd_spec = _superfp_binaryK_accumulated_odd_spec(
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
                block_size=block_size,
                outer_K=outer_K,
                outer_P=outer_P,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
                outer_subnormals_mode=outer_subnormals_mode,
                carrier=carrier,
            )
        else:
            odd_spec = _superfp_binaryK_odd_spec(
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
        return _gemm_formats(partial(_run_gemm, odd_spec))
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
        block_size=block_size,
        outer_K=outer_K,
        outer_P=outer_P,
        outer_bias=outer_bias,
        outer_is_signed=outer_is_signed,
        outer_prng_bits=outer_prng_bits,
        outer_saturation_mode=outer_saturation_mode,
        outer_subnormals_mode=outer_subnormals_mode,
        carrier=carrier,
    )
    return _gemm_formats(partial(_run_gemm, spec))


def _fixedpoint_formats(spec: _GemmSpec, output_quant: object) -> QAffineFormats:
    """A Linear layer's three hooks over a fixed-point spec, with the forward's
    output rounding when one is given (which a wide result needs)."""
    grad = _spec_matmul(spec, _run_gemm, "gradient")
    formats = _gemm_formats(_spec_matmul(spec, _run_gemm, "forward"), grad, grad)
    if spec.wide:
        formats._wide_fwd = _linear_fwd(partial(_run_gemm, spec))
    if output_quant is not None:
        _with_output(formats, _as_quant(output_quant, "output_quant"), F.linear)
    return formats


def fixedpoint_gemm_formats(
    mul_wl: int,
    mul_fl: int,
    *,
    mul_is_signed: bool = True,
    mul_symmetric: bool = False,
    mul_prng_bits: int = 0,
    acc_wl: int,
    acc_fl: int,
    acc_is_signed: bool = True,
    acc_symmetric: bool = False,
    acc_prng_bits: int = 0,
    rounding_mode: RoundMode = RoundMode.RNE,
    overflow: Overflow = Overflow.SATURATE,
    carrier: torch.dtype | None = None,
    output_quant: object = None,
) -> QAffineFormats:
    """Layer formats whose three passes run :func:`mptorch.quant.fixedpoint_matmul`.

    Every pass rounds each product to the multiply format and each exact sum
    to the accumulate format. When the accumulate format has more magnitude
    bits than the layer's dtype holds (24 for float32), the product is
    computed in float64, and the forward pass rounds it once with
    ``output_quant``, after the bias, before storing it in the layer's dtype.
    Without ``output_quant`` such a forward raises, and so does a gradient
    pass, which is stored in the gradient's dtype: use float64 layers for a
    wide accumulator there.

    Args:
        mul_wl (int): the multiply format's word length.
        mul_fl (int): the multiply format's fractional length.
        mul_is_signed (bool): see :func:`mptorch.quant.fixedpoint_matmul`.
            Default: ``True``
        mul_symmetric (bool): see :func:`mptorch.quant.fixedpoint_matmul`.
            Default: ``False``
        mul_prng_bits (int): see :func:`mptorch.quant.fixedpoint_matmul`.
            Default: 0
        acc_wl (int): the accumulate format's word length.
        acc_fl (int): the accumulate format's fractional length.
        acc_is_signed (bool): see :func:`mptorch.quant.fixedpoint_matmul`.
            Default: ``True``
        acc_symmetric (bool): see :func:`mptorch.quant.fixedpoint_matmul`.
            Default: ``False``
        acc_prng_bits (int): see :func:`mptorch.quant.fixedpoint_matmul`.
            Default: 0
        rounding_mode (RoundMode): the mode of both roundings, in every pass.
            Default: ``RoundMode.RNE``
        overflow (Overflow): what the sum does past the accumulate format's
            range. Default: ``Overflow.SATURATE``
        carrier (torch.dtype, optional): see
            :func:`mptorch.quant.fixedpoint_matmul`. Default: ``None``
        output_quant (Quant or Number or callable, optional): the forward
            result's rounding, after the bias: a bare format means
            ``Quant(fmt)``. Default: ``None``

    Returns:
        QAffineFormats: with ``fwd_math``, ``bwd_igrad_math`` and
        ``bwd_wgrad_math`` set and every ``*_quant`` slot left ``None``.

    Raises:
        ValueError: in the cases :func:`mptorch.quant.fixedpoint_matmul`
            lists, when the formats are resolved; and, at the call, for a wide
            result without ``output_quant`` or in a gradient pass.

    Example::

        >>> import torch
        >>> from mptorch import FixedPoint
        >>> from mptorch.quant import QLinear, Quant, fixedpoint_gemm_formats
        >>> formats = fixedpoint_gemm_formats(16, 8, acc_wl=32, acc_fl=8,
        ...                                   output_quant=FixedPoint(16, 6))
        >>> formats.input_quant = formats.weight_quant = Quant(FixedPoint(8, 4))
        >>> layer = QLinear(16, 4, bias=False, formats=formats)
        >>> layer(torch.randn(2, 16)).dtype
        torch.float32
    """
    spec = _fixedpoint_spec(
        mul_wl=mul_wl,
        mul_fl=mul_fl,
        mul_is_signed=mul_is_signed,
        mul_symmetric=mul_symmetric,
        mul_prng_bits=mul_prng_bits,
        acc_wl=acc_wl,
        acc_fl=acc_fl,
        acc_is_signed=acc_is_signed,
        acc_symmetric=acc_symmetric,
        acc_prng_bits=acc_prng_bits,
        rounding_mode=rounding_mode,
        overflow=overflow,
        carrier=carrier,
    )
    return _fixedpoint_formats(spec, output_quant)


def fixedpoint_gemm_formats_fma(
    fma_wl: int,
    fma_fl: int,
    *,
    fma_is_signed: bool = True,
    fma_symmetric: bool = False,
    fma_prng_bits: int = 0,
    rounding_mode: RoundMode = RoundMode.RNE,
    overflow: Overflow = Overflow.SATURATE,
    carrier: torch.dtype | None = None,
    output_quant: object = None,
) -> QAffineFormats:
    """The fused counterpart of :func:`fixedpoint_gemm_formats`: every pass runs
    :func:`mptorch.quant.fixedpoint_matmul_fma`, one rounding per step.

    Args:
        fma_wl (int): the format's word length.
        fma_fl (int): the format's fractional length.
        fma_is_signed (bool): see :func:`mptorch.quant.fixedpoint_matmul_fma`.
            Default: ``True``
        fma_symmetric (bool): see :func:`mptorch.quant.fixedpoint_matmul_fma`.
            Default: ``False``
        fma_prng_bits (int): see :func:`mptorch.quant.fixedpoint_matmul_fma`.
            Default: 0
        rounding_mode (RoundMode): the step's rounding mode, in every pass.
            Default: ``RoundMode.RNE``
        overflow (Overflow): what the sum does past the format's range.
            Default: ``Overflow.SATURATE``
        carrier (torch.dtype, optional): see
            :func:`mptorch.quant.fixedpoint_matmul_fma`. Default: ``None``
        output_quant (Quant or Number or callable, optional): as for
            :func:`fixedpoint_gemm_formats`. Default: ``None``

    Returns:
        QAffineFormats: as for :func:`fixedpoint_gemm_formats`.

    Raises:
        ValueError: as for :func:`fixedpoint_gemm_formats`.
    """
    spec = _fixedpoint_fma_spec(
        fma_wl=fma_wl,
        fma_fl=fma_fl,
        fma_is_signed=fma_is_signed,
        fma_symmetric=fma_symmetric,
        fma_prng_bits=fma_prng_bits,
        rounding_mode=rounding_mode,
        overflow=overflow,
        carrier=carrier,
    )
    return _fixedpoint_formats(spec, output_quant)


def block_gemm_formats(
    input: BlockFormat | BlockQuant,
    weight: BlockFormat | BlockQuant,
    grad: BlockFormat | BlockQuant | None = None,
    *,
    acc: BinaryK | None = None,
    fused: bool = False,
    rounding: RoundMode = RoundMode.RNE,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    block_size: int | None = None,
    outer: BinaryK | None = None,
    carrier: torch.dtype | None = None,
    tensor_scale_epilogue: bool = False,
) -> QAffineFormats:
    """Build a ``QAffineFormats`` whose three Linear products are block GEMMs.

    Each hook packs the tensors it multiplies along the dimension it sums
    over, each in the format of its role:

    - The forward product ``x @ w^T`` packs the input along its features in
      ``input``'s format, and the weight along its input features in
      ``weight``'s.
    - The input gradient ``g @ w`` packs the output gradient along its
      features in ``grad``'s format (``input``'s by default), and the weight
      along its output features.
    - The weight gradient ``g^T @ x`` packs both along the batch dimension.

    The arithmetic (``acc``, ``fused``, ``rounding`` and the accumulate
    algorithm) is that of :func:`mptorch.quant.block_matmul`, and all three
    passes share it. Each pass computes its result in the carrier (float32
    for binary32, float64 for binary64) and stores it in the layer's dtype.

    A weight that arrives already packed, from
    ``weight_quant = BlockQuant(fmt, axis=1).pack``, is used as it is. The
    ``pack`` method caches its result until the weight changes, so the weight
    is packed once per optimizer step instead of once per hook, and only once
    in inference.

    With square tiles (``dataclasses.replace(fmt, block_rows=fmt.block_size)``,
    such as NVFP4 with 16 x 16 weight tiles) the input gradient reads that
    packing transposed, so the forward and backward passes see the same
    quantized weight. A 1D or non-square weight format quantizes differently
    along the other axis, so a weight packed in such a format raises in the
    input gradient. To train with one, leave ``weight_quant`` unset.

    For inference, :meth:`mptorch.quant.QLinear.freeze_packed_` replaces the
    layer's float weight with its codes in ``weight``'s format, and the layer
    then holds only the codes (:class:`mptorch.quant.QPackedLinear`).

    Args:
        input (BlockFormat or BlockQuant): the input's format (a
            ``BlockQuant`` also sets its rounding and tensor scale).
        weight (BlockFormat or BlockQuant): the weight's format.
        grad (BlockFormat or BlockQuant, optional): the output gradient's
            format. Default: ``None``, which takes ``input``'s.
        acc (BinaryK, optional): the accumulate format. Default: ``None``,
            which leaves the sum in the carrier.
        fused (bool): whether each step is a fused multiply-add rounded to
            ``acc``. Default: ``False``
        rounding (RoundMode): the rounding mode of the sum. Default:
            ``RoundMode.RNE``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed, as for :class:`mptorch.quant.SplitMac`.
            Default: ``AccumulateAlgorithm.NAIVE``
        block_size (int, optional): the block size of ``BLOCK`` and ``TREE``.
            Default: ``None``
        outer (BinaryK, optional): the outer format of ``BLOCK`` and
            ``TREE``. Default: ``None``
        carrier (torch.dtype, optional): ``torch.float64`` packs and
            multiplies in binary64 whatever the layer's dtype, as for
            :class:`mptorch.quant.BlockMac` (a ``BlockQuant`` that names its
            own carrier keeps it). Default: ``None``, which follows the
            layer's dtype: binary64 for a float64 layer, binary32 otherwise.
        tensor_scale_epilogue (bool): whether to apply the operands'
            per-tensor scales to each GEMM's result instead of while decoding
            the elements, as NVIDIA's NVFP4 GEMMs do
            (:func:`mptorch.quant.block_matmul`). Default: ``False``

    Returns:
        QAffineFormats: with the three math hooks set and every ``*_quant``
        slot left ``None``.

    Example::

        >>> import dataclasses, torch
        >>> from mptorch import BinaryK, NVFP4
        >>> from mptorch.quant import BlockQuant, QLinear, block_gemm_formats
        >>> w16 = dataclasses.replace(NVFP4, block_rows=16)
        >>> formats = block_gemm_formats(NVFP4, w16, acc=BinaryK(16, 11))
        >>> formats.weight_quant = BlockQuant(w16, axis=1).pack
        >>> layer = QLinear(64, 32, formats=formats)
        >>> layer(torch.randn(8, 64)).sum().backward()
    """

    def operand(slot: BlockFormat | BlockQuant, name: str) -> BlockQuant:
        q = _as_block_quant(slot, name)
        # the factory's carrier also applies to packing, unless the operand names its own
        if carrier is torch.float64 and q.carrier is None:
            q = replace(q, carrier=torch.float64)
        return q

    qi = operand(input, "input")
    qw = operand(weight, "weight")
    qg = qi if grad is None else operand(grad, "grad")
    spec = _block_gemm_spec(
        acc,
        fused,
        rounding,
        accumulate_algorithm,
        block_size,
        outer,
        carrier,
        tensor_scale_epilogue,
    )

    formats = QAffineFormats(
        fwd_math=_block_linear_fwd(spec, qi, qw),
        bwd_igrad_math=_block_linear_igrad(spec, qg, qw),
        bwd_wgrad_math=_block_linear_wgrad(spec, qg, qi),
    )
    # the weight's quantizer, which QLinear.freeze_packed_ uses to pack a frozen weight
    formats._block_weight = qw
    return formats
