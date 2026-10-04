"""
Factories that wire the custom-arithmetic GEMM ops
(:func:`mptorch.quant.ops.binaryK_matmul` and its seven siblings) into the
``fwd_math`` / ``bwd_igrad_math`` / ``bwd_wgrad_math`` hooks of a
:class:`~mptorch.quant.modules.format.QAffineFormats`, under the call contract
``CustomArithLinear`` (``mptorch/quant/modules/linear.py``) uses::

    fwd_math(q_input, q_weight, q_bias)        # q_input @ q_weight.T + q_bias
    bwd_igrad_math(q_igrad_output, q_weight)   # q_igrad_output @ q_weight
    bwd_wgrad_math(q_wgrad_output, q_input)    # q_wgrad_output.T @ q_input

``q_input``, ``q_igrad_output`` and ``q_wgrad_output`` may carry any number of
leading (batch) dimensions, as ``nn.Linear`` allows. Each hook flattens them
into one row dimension before the 2D GEMM and reshapes the result back, the
same reshape ``CustomArithLinear._default_bwd_wgrad`` makes.

The four ``*_gemm_formats`` factories differ only in which op the three hooks
call, so they share :func:`_gemm_formats` and differ only in the argument list
they resolve. Each resolves its format once, at construction, into a
``_GemmSpec``: a layer's format is fixed for its lifetime, and re-deriving the
``acc_*`` fallbacks, the default exponent biases and the carrier checks on
every forward and backward pass would cost 2.3 to 6.4 microseconds per call
(measured by ``dev/benchmarks/python_call_overhead.py``), more than the rest
of the Python call path.

These factories set only the ``*_math`` hooks. Elementwise operand
quantization (``weight_quant``, ``input_quant`` and the other ``*_quant``
slots) is layered onto the returned ``QAffineFormats`` by the caller, so that
operand rounding and dot-product arithmetic stay independently composable.
"""

from collections.abc import Callable
from functools import partial

import torch

from mptorch.number import (
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    RoundMode,
    SaturationMode,
    SubnormalsMode,
)

from .block import BlockPacked, _block_gemm_spec, _run_block_gemm, _stored_as
from .mac import BlockMac, BlockQuant, Mac, _as_block_quant, spec_for_block_mac, spec_for_mac
from .modules.format import QAffineFormats, QMatmulFormats
from .ops import (
    _binaryK_accumulated_spec,
    _binaryK_fma_accumulated_spec,
    _gemm_mixed_nd,
    _gemm_nd,
    _run_gemm,
    _superfp_accumulated_spec,
    _superfp_fma_accumulated_spec,
)

__all__ = [
    "binaryK_gemm_formats",
    "superfp_gemm_formats",
    "binaryK_gemm_formats_fma",
    "superfp_gemm_formats_fma",
    "matmul_formats",
    "block_matmul_formats",
    "block_gemm_formats",
]

# What a resolved GEMM still takes per call: ``op(a) @ op(b)`` for 2D operands
# ``a`` and ``b``, where each boolean says whether ``op`` transposes that
# operand (a flag the kernel reads, so no transposed copy is made).
_Matmul2D = Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor]


def _flatten_leading(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Flatten all but the last dimension of ``x`` into one leading dimension.

    Returns the 2D view and the original shape, so the caller can reshape a
    result back to the leading dimensions ``x`` came with.
    """
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape


def _gemm_formats(matmul: _Matmul2D) -> QAffineFormats:
    """Build the three Linear math hooks over one 2D ``matmul``.

    Which operand each hook transposes, how the leading dimensions are folded
    into one row dimension and put back, and how the bias is added are the
    Linear call contract rather than the format's, so they are written here
    once and each factory below passes only its resolved ``matmul``.
    """

    def fwd(
        q_input: torch.Tensor, q_weight: torch.Tensor, q_bias: torch.Tensor | None
    ) -> torch.Tensor:
        x_flat, x_shape = _flatten_leading(q_input)
        out = matmul(x_flat, q_weight, False, True)
        out = out.reshape(*x_shape[:-1], q_weight.shape[0])
        # add_ rather than +: `out` is a view of a tensor the op allocated
        # inside this call and nothing else references it, so folding the
        # bias in place is bit-identical to `out + q_bias` and saves one
        # output-sized allocation per forward. The dtype guard keeps a bias
        # in a wider dtype promoting the way `out + q_bias` does, where
        # `add_` would raise.
        if q_bias is None:
            return out
        return out.add_(q_bias) if q_bias.dtype == out.dtype else out + q_bias

    def bwd_igrad(q_igrad_output: torch.Tensor, q_weight: torch.Tensor) -> torch.Tensor:
        g_flat, g_shape = _flatten_leading(q_igrad_output)
        out = matmul(g_flat, q_weight, False, False)
        return out.reshape(*g_shape[:-1], q_weight.shape[1])

    def bwd_wgrad(q_wgrad_output: torch.Tensor, q_input: torch.Tensor) -> torch.Tensor:
        g_flat, _ = _flatten_leading(q_wgrad_output)
        i_flat, _ = _flatten_leading(q_input)
        return matmul(g_flat, i_flat, True, False)

    return QAffineFormats(fwd_math=fwd, bwd_igrad_math=bwd_igrad, bwd_wgrad_math=bwd_wgrad)


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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
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
) -> QAffineFormats:
    """Build a ``QAffineFormats`` whose dot products run in a binaryK GEMM.

    The returned formats set ``fwd_math``, ``bwd_igrad_math`` and
    ``bwd_wgrad_math`` so that a Linear layer's forward and both of its
    gradient GEMMs run through :func:`mptorch.quant.ops.binaryK_matmul`, whose
    arguments these are: every product is rounded to the ``mul_*`` format and,
    when ``accumulate_quant`` is set, every partial sum to the ``acc_*``
    format. The format is resolved once here, so each pass pays only the op.
    Every pass holds the formats to the carrier of the tensors it runs on,
    binary64 for a float64 layer and binary32 otherwise, unless ``carrier``
    names one, and warns with :class:`mptorch.FormatRangeWarning` about a
    range the carrier cannot hold.

    Args:
        mul_K (int): total width in bits of the multiply format.
        mul_P (int): precision (significand bits, the implicit one included)
            of the multiply format.
        mul_bias (int, optional): exponent bias of the multiply format.
            ``None`` takes IEEE P3109's, ``2**(K-P-1)`` signed and
            ``2**(K-P)`` unsigned. Default: ``None``
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits drawn per stochastic rounding of a
            product; only read when ``rounding_mode`` is ``RoundMode.SR``.
            Default: ``0``
        accumulate_quant (bool): whether the running sum is rounded to the
            ``acc_*`` format after every step. ``False`` keeps it in the
            carrier's precision and ignores the ``acc_*`` widths.
            Default: ``True``
        acc_K (int, optional): width of the accumulate format. ``None`` takes
            ``mul_K``. Default: ``None``
        acc_P (int, optional): precision of the accumulate format. ``None``
            takes ``mul_P``. Default: ``None``
        acc_bias (int, optional): exponent bias of the accumulate format.
            ``None`` takes P3109's for ``acc_K``/``acc_P``. Default: ``None``
        acc_is_signed (bool, optional): sign bit of the accumulate format.
            ``None`` takes ``mul_is_signed``. Default: ``None``
        acc_prng_bits (int): random bits per stochastic rounding of a partial
            sum. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed, as for :func:`mptorch.quant.binaryK_matmul`: ``NAIVE``,
            ``KAHAN``, ``BLOCK`` or ``TREE`` (the last not for a fused mac),
            the three past ``NAIVE`` in binary32 only so far.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): rounding for both formats; one mode for
            both because the kernels take it as a template parameter.
            Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): overflow behaviour of the multiply
            format. Default: ``SaturationMode.OVF_INF``
        subnormals_mode (SubnormalsMode): what the multiply format keeps
            below its smallest normal. Default: ``SubnormalsMode.SUBNORMALS``
        acc_saturation_mode (SaturationMode, optional): overflow behaviour of
            the accumulate format. ``None`` takes ``saturation_mode``.
            Default: ``None``
        acc_subnormals_mode (SubnormalsMode, optional): subnormal handling of
            the accumulate format. ``None`` takes ``subnormals_mode``.
            Default: ``None``
        block_size (int, optional): ``BLOCK``'s and ``TREE``'s block size, as
            for :func:`mptorch.quant.binaryK_matmul`. Default: ``None``
        outer_K (int, optional): width of the outer format ``BLOCK`` and
            ``TREE`` round their total to, as for
            :func:`mptorch.quant.binaryK_matmul`, which documents it and the
            rest of its spelling: ``outer_P``, ``outer_bias``,
            ``outer_is_signed``, ``outer_prng_bits``,
            ``outer_saturation_mode`` and ``outer_subnormals_mode``.
            Default: ``None``, an unrounded total.
        carrier (torch.dtype, optional): the float arithmetic the GEMM rounds
            in. ``None`` takes the layer's (binary64 for float64, binary32 for
            the rest); ``torch.float64`` runs a float32 or half-precision
            layer in binary64 too and narrows each result back to the layer's
            dtype; ``torch.float32`` on a float64 layer raises at the call.
            Default: ``None``

    Returns:
        QAffineFormats: with the three ``*_math`` hooks set and every
        ``*_quant`` slot left ``None``.

    Raises:
        ValueError: for a format no carrier can hold (precision above 53
            bits, an exponent field wider than ten bits, stochastic bits with
            no significand left) or a ``carrier`` that names no carrier.

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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
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
) -> QAffineFormats:
    """Fused-multiply-add analog of :func:`binaryK_gemm_formats`.

    Each dot-product step is one fused multiply-add ``round(a * b + s)``,
    rounded once to the ``fma_*`` format
    (:func:`mptorch.quant.ops.binaryK_matmul_fma`), rather than a rounded
    multiply followed by a separately rounded add. The general contract
    (resolution at construction, the carrier, the warnings) is
    :func:`binaryK_gemm_formats`'s.

    Args:
        fma_K (int): total width in bits of the fused format.
        fma_P (int): precision of the fused format, the implicit bit included.
        fma_bias (int, optional): exponent bias; ``None`` takes IEEE P3109's.
            Default: ``None``
        fma_is_signed (bool): whether the format has a sign bit.
            Default: ``True``
        fma_quant (bool): whether each fused step is rounded at all.
            ``False`` runs unrounded fused steps in the carrier.
            Default: ``True``
        fma_prng_bits (int): random bits per stochastic rounding; only read
            when ``rounding_mode`` is ``RoundMode.SR``. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed, as for :func:`mptorch.quant.binaryK_matmul`: ``NAIVE``,
            ``KAHAN``, ``BLOCK`` or ``TREE`` (the last not for a fused mac),
            the three past ``NAIVE`` in binary32 only so far.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): rounding of each fused step.
            Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): overflow behaviour.
            Default: ``SaturationMode.OVF_INF``
        subnormals_mode (SubnormalsMode): subnormal handling.
            Default: ``SubnormalsMode.SUBNORMALS``
        block_size (int, optional): ``BLOCK``'s and ``TREE``'s block size, as
            for :func:`mptorch.quant.binaryK_matmul`. Default: ``None``
        outer_K (int, optional): width of the outer format ``BLOCK`` and
            ``TREE`` round their total to, as for
            :func:`mptorch.quant.binaryK_matmul`, which documents it and the
            rest of its spelling: ``outer_P``, ``outer_bias``,
            ``outer_is_signed``, ``outer_prng_bits``,
            ``outer_saturation_mode`` and ``outer_subnormals_mode``.
            Default: ``None``, an unrounded total.
        carrier (torch.dtype, optional): as for :func:`binaryK_gemm_formats`.
            Default: ``None``

    Returns:
        QAffineFormats: with the three ``*_math`` hooks set.

    Raises:
        ValueError: as for :func:`binaryK_gemm_formats`.

    Example::

        >>> import torch
        >>> from mptorch.quant import QLinear, binaryK_gemm_formats_fma
        >>> layer = QLinear(256, 64, formats=binaryK_gemm_formats_fma(8, 4))
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
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
) -> QAffineFormats:
    """superfp analog of :func:`binaryK_gemm_formats`.

    The dot products run through :func:`mptorch.quant.ops.superfp_matmul`:
    products rounded to the ``mul_*`` superfp format and, when
    ``accumulate_quant`` is set, partial sums to the ``acc_*`` one. A superfp
    format has ``normal_binades`` binades of normal values around 1.0 and
    spends the rest of its exponent range on supernormal values, powers of two
    with no mantissa; it has no subnormals, so there is no ``subnormals_mode``.
    See :func:`superfp_gemm_formats_fma` for the fused-multiply-add analog.

    Args:
        mul_man_bits (int): mantissa bits of the multiply format.
        mul_exp_bits (int): exponent bits of the multiply format.
        mul_normal_binades (int): binades with a full mantissa in the multiply
            format.
        mul_bias (int): exponent bias of the multiply format (required; superfp
            has no conventional default).
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits per stochastic rounding of a product.
            Default: ``0``
        accumulate_quant (bool): whether partial sums are rounded to the
            ``acc_*`` format. Default: ``True``
        acc_man_bits (int, optional): ``None`` takes ``mul_man_bits``.
            Default: ``None``
        acc_exp_bits (int, optional): ``None`` takes ``mul_exp_bits``.
            Default: ``None``
        acc_normal_binades (int, optional): ``None`` takes
            ``mul_normal_binades``. Default: ``None``
        acc_bias (int, optional): ``None`` takes ``mul_bias``. Default: ``None``
        acc_is_signed (bool, optional): ``None`` takes ``mul_is_signed``.
            Default: ``None``
        acc_prng_bits (int): random bits per stochastic rounding of a partial
            sum. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed, as for :func:`mptorch.quant.binaryK_matmul`: ``NAIVE``,
            ``KAHAN``, ``BLOCK`` or ``TREE`` (the last not for a fused mac),
            the three past ``NAIVE`` in binary32 only so far.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): rounding for both formats.
            Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): overflow behaviour of the multiply
            format. Default: ``SaturationMode.OVF_INF``
        acc_saturation_mode (SaturationMode, optional): ``None`` takes
            ``saturation_mode``. Default: ``None``
        block_size (int, optional): ``BLOCK``'s and ``TREE``'s block size, as
            for :func:`mptorch.quant.binaryK_matmul`. Default: ``None``
        outer_man_bits (int, optional): mantissa bits of the outer format
            ``BLOCK`` and ``TREE`` round their total to, as for
            :func:`mptorch.quant.superfp_matmul`, which documents it and the
            rest of its spelling: ``outer_exp_bits``,
            ``outer_normal_binades``, ``outer_bias``, ``outer_is_signed``,
            ``outer_prng_bits`` and ``outer_saturation_mode``.
            Default: ``None``, an unrounded total.
        carrier (torch.dtype, optional): as for :func:`binaryK_gemm_formats`.
            Default: ``None``

    Returns:
        QAffineFormats: with the three ``*_math`` hooks set.

    Raises:
        ValueError: for a format no carrier can hold, a ``normal_binades``
            that leaves no supernormal codes, or a ``carrier`` that names no
            carrier.

    Example::

        >>> import torch
        >>> from mptorch.quant import QLinear, superfp_gemm_formats
        >>> formats = superfp_gemm_formats(3, 4, 8, 7, accumulate_quant=False)
        >>> layer = QLinear(256, 64, formats=formats)
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    block_size: int | None = None,
    outer_man_bits: int | None = None,
    outer_exp_bits: int | None = None,
    outer_normal_binades: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> QAffineFormats:
    """superfp analog of :func:`binaryK_gemm_formats_fma`.

    Each dot-product step is one fused multiply-add rounded once to the
    ``fma_*`` superfp format (:func:`mptorch.quant.ops.superfp_matmul_fma`).
    The arguments are :func:`superfp_gemm_formats`'s multiply-format arguments
    under the ``fma_`` prefix, plus ``fma_quant``, which is
    :func:`binaryK_gemm_formats_fma`'s.

    Args:
        fma_man_bits (int): mantissa bits of the fused format.
        fma_exp_bits (int): exponent bits of the fused format.
        fma_normal_binades (int): binades with a full mantissa.
        fma_bias (int): exponent bias (required).
        fma_is_signed (bool): whether the format has a sign bit.
            Default: ``True``
        fma_quant (bool): whether each fused step is rounded at all.
            Default: ``True``
        fma_prng_bits (int): random bits per stochastic rounding.
            Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed, as for :func:`mptorch.quant.binaryK_matmul`: ``NAIVE``,
            ``KAHAN``, ``BLOCK`` or ``TREE`` (the last not for a fused mac),
            the three past ``NAIVE`` in binary32 only so far.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): rounding of each fused step.
            Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): overflow behaviour.
            Default: ``SaturationMode.OVF_INF``
        block_size (int, optional): ``BLOCK``'s and ``TREE``'s block size, as
            for :func:`mptorch.quant.binaryK_matmul`. Default: ``None``
        outer_man_bits (int, optional): mantissa bits of the outer format
            ``BLOCK`` and ``TREE`` round their total to, as for
            :func:`mptorch.quant.superfp_matmul`, which documents it and the
            rest of its spelling: ``outer_exp_bits``,
            ``outer_normal_binades``, ``outer_bias``, ``outer_is_signed``,
            ``outer_prng_bits`` and ``outer_saturation_mode``.
            Default: ``None``, an unrounded total.
        carrier (torch.dtype, optional): as for :func:`binaryK_gemm_formats`.
            Default: ``None``

    Returns:
        QAffineFormats: with the three ``*_math`` hooks set.

    Raises:
        ValueError: as for :func:`superfp_gemm_formats`.

    Example::

        >>> import torch
        >>> from mptorch.quant import QLinear, superfp_gemm_formats_fma
        >>> formats = superfp_gemm_formats_fma(3, 4, 8, 7)
        >>> layer = QLinear(256, 64, formats=formats)
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """
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


# What a matmul's three hooks call: ``op(a) @ op(b)`` under torch.matmul's
# operand rules (batch broadcasting and 1D promotion), where the Linear hooks
# above flatten to 2D themselves. The flags transpose ``a`` and ``b``.
_MatmulNd = Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor]


def _matmul_formats(
    fwd_matmul: _MatmulNd, agrad_matmul: _MatmulNd, bgrad_matmul: _MatmulNd
) -> QMatmulFormats:
    """Build the three matmul math hooks, one resolved GEMM per pass.

    The transpose table is the whole difference from :func:`_gemm_formats`:
    the forward is ``a @ b``, ``a``'s gradient is ``grad @ b^T`` and ``b``'s
    is ``a^T @ grad``. Nothing is flattened and no bias is added, since a
    matmul has no leading dimension it may fold (broadcasting over the batch
    is part of its contract) and no bias. It takes three matmuls rather than
    one because a palette's ``prec_idx`` map is per pass; every other format
    passes the same matmul three times.
    """

    def fwd(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        return fwd_matmul(q_a, q_b, False, False)

    def bwd_agrad(q_grad: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        return agrad_matmul(q_grad, q_b, False, True)

    def bwd_bgrad(q_grad: torch.Tensor, q_a: torch.Tensor) -> torch.Tensor:
        return bgrad_matmul(q_a, q_grad, True, False)

    return QMatmulFormats(fwd_math=fwd, bwd_agrad_math=bwd_agrad, bwd_bgrad_math=bwd_bgrad)


def _mixed_matmul(spec, prec_idx: torch.Tensor | None, pass_name: str, argument: str):
    """One pass's matmul over a palette op, or a hook that names the missing map.

    A ``prec_idx`` is indexed by *output* element, and the three passes of a
    matmul have three different output shapes: ``[M, N]``, ``[M, K]`` and
    ``[K, N]``. One map cannot serve all three, so a pass whose map was not
    given gets a hook that raises naming the ``matmul_formats`` argument to
    pass, rather than a shape error from the op or a silent fallback to
    unquantized arithmetic.
    """
    if prec_idx is None:

        def missing(*_args, **_kwargs) -> torch.Tensor:
            raise ValueError(
                f"this format is a palette, so its {pass_name} needs its own prec_idx map "
                f"(shaped like that pass's output): pass {argument}= to matmul_formats"
            )

        return missing

    def matmul(a: torch.Tensor, b: torch.Tensor, ta: bool, tb: bool) -> torch.Tensor:
        return _gemm_mixed_nd(spec, a, b, prec_idx, ta, tb)

    return matmul


def matmul_formats(
    mac: Mac,
    *,
    prec_idx: torch.Tensor | None = None,
    agrad_prec_idx: torch.Tensor | None = None,
    bgrad_prec_idx: torch.Tensor | None = None,
) -> QMatmulFormats:
    """Build a ``QMatmulFormats`` whose three passes run in ``mac``'s arithmetic.

    The forward ``a @ b``, ``a``'s gradient ``grad @ b^T`` and ``b``'s gradient
    ``a^T @ grad`` each run through the GEMM op ``mac`` resolves to. A palette
    in any of ``mac``'s slots selects the per-output-element op, which needs a
    ``prec_idx`` map per pass; the maps are bound into the hooks here rather
    than threaded through every call, and a gradient pass whose map is missing
    raises when it is reached rather than falling back to unquantized
    arithmetic. Like the four ``*_gemm_formats`` factories, this sets only the
    math hooks; elementwise operand quantization (``a_quant``, ``b_quant``,
    ``agrad_quant``, ``bgrad_quant``) is layered on by the caller.

    Args:
        mac (SplitMac or FusedMac): the dot-product arithmetic, resolved once
            through :func:`mptorch.quant.mac.spec_for_mac`.
        prec_idx (Tensor, optional): integer map selecting a palette entry per
            element of the forward's output, shaped like ``a @ b`` (``[M, N]``,
            ``[M, 1]`` or ``[1, N]``, optionally batched). Required by, and
            only by, a palette ``mac``. Default: ``None``
        agrad_prec_idx (Tensor, optional): the map for ``a``'s gradient,
            shaped like ``grad @ b^T``. Default: ``None``
        bgrad_prec_idx (Tensor, optional): the map for ``b``'s gradient,
            shaped like ``a^T @ grad``. Default: ``None``

    Returns:
        QMatmulFormats: with ``fwd_math``, ``bwd_agrad_math`` and
        ``bwd_bgrad_math`` set and every ``*_quant`` slot left ``None``.

    Raises:
        ValueError: if ``mac`` holds a palette and ``prec_idx`` is ``None``, or
            if a map is given for a ``mac`` with no palette to select from.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import Palette, QMatmul, SplitMac, matmul_formats
        >>> palette = Palette([BinaryK(8, 4), BinaryK(8, 5)])
        >>> prec_idx = torch.zeros(4, 3, dtype=torch.int64)
        >>> formats = matmul_formats(SplitMac(palette, palette), prec_idx=prec_idx)
        >>> QMatmul(formats)(torch.randn(4, 8), torch.randn(8, 3)).shape
        torch.Size([4, 3])
    """
    spec = spec_for_mac(mac)
    maps = (prec_idx, agrad_prec_idx, bgrad_prec_idx)
    if spec.mixed and prec_idx is None:
        raise ValueError(
            "a palette format selects its entry per output element, so it needs a "
            "prec_idx map; pass one, or give a single format per slot"
        )
    if not spec.mixed and any(m is not None for m in maps):
        raise ValueError("prec_idx has no meaning without a palette format to select from")
    if spec.mixed:
        return _matmul_formats(
            _mixed_matmul(spec, prec_idx, "forward", "prec_idx"),
            _mixed_matmul(spec, agrad_prec_idx, "gradient of a", "agrad_prec_idx"),
            _mixed_matmul(spec, bgrad_prec_idx, "gradient of b", "bgrad_prec_idx"),
        )
    matmul = partial(_gemm_nd, spec)
    return _matmul_formats(matmul, matmul, matmul)


# --- block formats --------------------------------------------------------------
#
# The block GEMM (`mptorch.quant.block_matmul`) multiplies packed operands,
# each read along its K: packed along K, or along its other dimension and read
# transposed. So each hook packs its operands along the dimension the pass
# reduces over, as MX training recipes do, except an operand that arrives
# packed: a weight packed once by `BlockQuant.pack` (a `weight_quant` slot), or
# an operand of a square-tiled format, which quantizes to the same values
# along either axis and so serves the backward pass read transposed. A 1D or
# non-square format quantizes differently along each axis, so its backward
# re-packs the float operand, and a weight that arrives packed in one cannot
# be used for the input gradient.


def _square_tile_hint(fmt: BlockFormat) -> str:
    """How to spell the square-tiled variant of a format, for messages."""
    return f"dataclasses.replace(fmt, block_rows={fmt.block_size})"


def _packed_operand(q: BlockQuant, x: "torch.Tensor | BlockPacked", axis: int) -> BlockPacked:
    """``x`` packed along ``axis`` by ``q``, unless it already is a packing."""
    return x if isinstance(x, BlockPacked) else q.pack_fresh(x, axis)


def _reused(x: "torch.Tensor | BlockPacked", q: BlockQuant, where: str) -> BlockPacked | None:
    """A packing the forward pass made, for a backward pass to read along its
    other axis, or ``None`` when the float operand was saved (and is packed
    anew). Only a square tile quantizes the same along both axes, so a packing
    in a 1D or non-square format, or in a format other than the one this pass
    rounds to, raises."""
    if not isinstance(x, BlockPacked):
        return None
    if x.fmt != q.fmt:
        raise ValueError(
            f"{where} rounds this operand to {q.fmt}, and the forward pass packed it in "
            f"{x.fmt}: leave the *_quant slot unset so it is packed per pass"
        )
    if not x.fmt.is_square:
        raise ValueError(
            f"{where} reads the operand the forward pass packed along its other axis, and "
            f"{x.fmt}'s tiles are not square, so its values along that axis are not the ones "
            f"this pass rounds to: leave the *_quant slot unset so it is packed per pass, or use "
            f"a square tile, {_square_tile_hint(x.fmt)}, which quantizes the same along either axis"
        )
    return x


def block_matmul_formats(
    mac: BlockMac, *, agrad_mac: BlockMac | None = None, bgrad_mac: BlockMac | None = None
) -> QMatmulFormats:
    """Build a ``QMatmulFormats`` whose three passes are block GEMMs.

    The forward ``a @ b`` packs ``a`` along its last dimension and ``b`` along
    its second-to-last (their K) in ``mac``'s two formats; ``a``'s gradient
    ``grad @ b^T`` packs ``grad`` along its last dimension and ``b`` along its
    last; ``b``'s gradient ``a^T @ grad`` packs ``a`` along its second-to-last
    and ``grad`` along its second-to-last. A gradient pass takes the formats
    of the slots it fills from ``agrad_mac`` / ``bgrad_mac``, which default to
    ``mac``. An operand whose format has square tiles is packed once, by the
    ``a_quant`` / ``b_quant`` slot this sets for it, and the gradient passes
    read that packing through :attr:`mptorch.quant.BlockPacked.mT`. Each
    pass's result is float32, stored in the operands' dtype.

    Args:
        mac (BlockMac): the forward's operand formats and arithmetic.
        agrad_mac (BlockMac, optional): ``a``'s gradient's. Default: ``None``
        bgrad_mac (BlockMac, optional): ``b``'s gradient's. Default: ``None``

    Returns:
        QMatmulFormats: the three math hooks, and ``a_quant`` / ``b_quant``
        for square-tiled operands.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK, MXFP8_E4M3
        >>> from mptorch.quant import BlockMac, QMatmul, block_matmul_formats
        >>> layer = QMatmul(block_matmul_formats(BlockMac(MXFP8_E4M3, acc=BinaryK(16, 11))))
        >>> layer(torch.randn(4, 64), torch.randn(64, 8)).shape
        torch.Size([4, 8])
    """
    agrad_mac = mac if agrad_mac is None else agrad_mac
    bgrad_mac = mac if bgrad_mac is None else bgrad_mac
    fwd_spec, agrad_spec, bgrad_spec = (spec_for_block_mac(m) for m in (mac, agrad_mac, bgrad_mac))
    qa, qb = cast_quant(mac.a), cast_quant(mac.b)
    ag_a, ag_b = cast_quant(agrad_mac.a), cast_quant(agrad_mac.b)
    bg_a, bg_b = cast_quant(bgrad_mac.a), cast_quant(bgrad_mac.b)

    def fwd(q_a, q_b):
        dtype = q_a.dtype
        out = _run_block_gemm(fwd_spec, _packed_operand(qa, q_a, -1), _packed_operand(qb, q_b, -2))
        return _stored_as(fwd_spec, out, dtype)

    def bwd_agrad(q_grad, q_b):
        b = _reused(q_b, ag_b, "a's gradient")
        b_t = b.mT if b is not None else ag_b.pack_fresh(q_b, -1).mT
        out = _run_block_gemm(agrad_spec, ag_a.pack_fresh(q_grad, -1), b_t)
        return _stored_as(agrad_spec, out, q_grad.dtype)

    def bwd_bgrad(q_grad, q_a):
        a = _reused(q_a, bg_a, "b's gradient")
        a_t = a.mT if a is not None else bg_a.pack_fresh(q_a, -2).mT
        out = _run_block_gemm(bgrad_spec, a_t, bg_b.pack_fresh(q_grad, -2))
        return _stored_as(bgrad_spec, out, q_grad.dtype)

    formats = QMatmulFormats(fwd_math=fwd, bwd_agrad_math=bwd_agrad, bwd_bgrad_math=bwd_bgrad)
    if qa.fmt.is_square:
        formats.a_quant = partial(qa.pack_fresh, axis=-1)
    if qb.fmt.is_square:
        formats.b_quant = partial(qb.pack_fresh, axis=-2)
    return formats


def cast_quant(slot: "BlockFormat | BlockQuant | None") -> BlockQuant:
    """A ``BlockMac`` slot as the ``BlockQuant`` it was normalized to."""
    assert slot is not None  # BlockMac.__post_init__ fills b in from a
    return _as_block_quant(slot, "a BlockMac slot")


def block_gemm_formats(
    input: "BlockFormat | BlockQuant",
    weight: "BlockFormat | BlockQuant",
    grad: "BlockFormat | BlockQuant | None" = None,
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
    """Build a ``QAffineFormats`` whose three Linear GEMMs are block GEMMs.

    Each hook packs what it multiplies along the dimension it reduces over, in
    the format of the tensor's role: the forward ``x @ w^T`` packs the input
    along its features in ``input``'s format and the weight along its input
    features in ``weight``'s; the input gradient ``g @ w`` packs the output
    gradient along its features in ``grad``'s format (``input``'s by default)
    and the weight along its output features; the weight gradient ``g^T @ x``
    packs both along the batch. The arithmetic (``acc``, ``fused``,
    ``rounding``, the accumulate algorithm) is :func:`mptorch.quant.block_matmul`'s,
    shared by the three. Results are float32, stored in the input's dtype.

    A weight that arrives packed, from ``weight_quant = BlockQuant(fmt,
    axis=1).pack``, is used as it is: that packs it once per optimizer step
    (the memo sees the step's in-place write) instead of in each hook, and in
    inference once for good. With square tiles
    (``dataclasses.replace(fmt, block_rows=fmt.block_size)``, NVFP4's 16 x 16
    weights) the input gradient reads that packing transposed, so the forward
    and backward passes see the same quantized weight; a 1D or non-square
    weight format quantizes differently along the other axis, so a weight
    packed in one raises in the input gradient, and training with one leaves
    ``weight_quant`` unset.

    Args:
        input (BlockFormat or BlockQuant): the input's format (a
            ``BlockQuant`` names its rounding and tensor scale).
        weight (BlockFormat or BlockQuant): the weight's.
        grad (BlockFormat or BlockQuant, optional): the output gradient's.
            Default: ``None``, ``input``'s.
        acc (BinaryK, optional): the accumulate format. Default: ``None``
        fused (bool): a fused multiply-add per step. Default: ``False``
        rounding (RoundMode): the sum's rounding mode. Default:
            ``RoundMode.RNE``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :class:`mptorch.quant.SplitMac`. Default: ``AccumulateAlgorithm.NAIVE``
        block_size (int, optional): BLOCK's and TREE's. Default: ``None``
        outer (BinaryK, optional): BLOCK's and TREE's outer format.
            Default: ``None``
        carrier (torch.dtype, optional): ``None`` or ``torch.float32``.
            Default: ``None``
        tensor_scale_epilogue (bool): apply the operands' per-tensor scales
            to each GEMM's result rather than in the decode, as NVIDIA's NVFP4
            GEMMs do (:func:`mptorch.quant.block_matmul`). Default: ``False``

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
    qi = _as_block_quant(input, "input")
    qw = _as_block_quant(weight, "weight")
    qg = qi if grad is None else _as_block_quant(grad, "grad")
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

    def fwd(q_input, q_weight, q_bias):
        x_flat, x_shape = _flatten_leading(q_input)
        w = q_weight if isinstance(q_weight, BlockPacked) else qw.pack_fresh(q_weight, 1)
        out = _stored_as(
            spec, _run_block_gemm(spec, qi.pack_fresh(x_flat, -1), w.mT), q_input.dtype
        )
        out = out.reshape(*x_shape[:-1], w.shape[0])
        if q_bias is None:
            return out
        return out.add_(q_bias) if q_bias.dtype == out.dtype else out + q_bias

    def bwd_igrad(q_igrad_output, q_weight):
        g_flat, g_shape = _flatten_leading(q_igrad_output)
        w = _reused(q_weight, qw, "the input gradient")
        if w is None:
            w = qw.pack_fresh(q_weight, 0)
        out = _run_block_gemm(spec, qg.pack_fresh(g_flat, -1), w)
        return _stored_as(spec, out, q_igrad_output.dtype).reshape(*g_shape[:-1], w.shape[1])

    def bwd_wgrad(q_wgrad_output, q_input):
        g_flat, _ = _flatten_leading(q_wgrad_output)
        i_flat, _ = _flatten_leading(q_input)
        out = _run_block_gemm(spec, qg.pack_fresh(g_flat, 0).mT, qi.pack_fresh(i_flat, 0))
        return _stored_as(spec, out, q_wgrad_output.dtype)

    return QAffineFormats(fwd_math=fwd, bwd_igrad_math=bwd_igrad, bwd_wgrad_math=bwd_wgrad)
