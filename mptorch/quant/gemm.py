"""Factories that make a layer's matrix products run in quantized GEMMs.

Each factory returns a format container whose math hooks call a quantized
GEMM op. For a ``QLinear`` the container is a
:class:`~mptorch.quant.modules.format.QAffineFormats`, and
``CustomArithLinear`` (``mptorch/quant/modules/linear.py``) calls its three
hooks like this::

    fwd_math(q_input, q_weight, q_bias)        # q_input @ q_weight.T + q_bias
    bwd_igrad_math(q_igrad_output, q_weight)   # q_igrad_output @ q_weight
    bwd_wgrad_math(q_wgrad_output, q_input)    # q_wgrad_output.T @ q_input

``q_input``, ``q_igrad_output`` and ``q_wgrad_output`` may have any number of
leading (batch) dimensions, as ``nn.Linear`` allows. Each hook flattens them
into one row dimension before the 2D GEMM, the same reshape
``CustomArithLinear._default_bwd_wgrad`` makes, and the forward and input
gradient hooks restore them on the result.

The module holds three kinds of factory:

* the six binaryK and superfp factories (:func:`binaryK_gemm_formats`, its
  ``_fma`` and superfp versions, and the two cross-family ones), for a
  ``QLinear``. They differ only in which op their hooks call, so they share
  :func:`_gemm_formats` and differ only in the arguments they resolve;
* :func:`matmul_formats` and :func:`block_matmul_formats`, which build a
  ``QMatmulFormats`` for a ``QMatmul``;
* :func:`block_gemm_formats`, which makes a ``QLinear`` multiply block-format
  operands.

Each factory resolves its formats once, when it is called, into a
``_GemmSpec``. A layer's formats are fixed for its lifetime, and redoing that
work (the ``acc_*`` fallbacks, the default exponent biases and the carrier
checks) on every forward and backward pass would cost 2.3 to 6.4 microseconds
per call (measured by ``dev/benchmarks/timing/python_call_overhead.py``),
more than the rest of the Python call path.

The factories set only the math hooks, except where a block format needs an
operand packed once (:func:`block_matmul_formats`). The caller adds the
elementwise operand quantizers (``weight_quant``, ``input_quant`` and the
other ``*_quant`` slots) to the returned container, so the rounding of the
operands and the arithmetic of the dot product can be chosen independently.
"""

from collections.abc import Callable
from dataclasses import replace
from functools import partial

import torch
import torch.nn.functional as F

from mptorch.number import (
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    RoundMode,
    SaturationMode,
    SubnormalsMode,
)

from .block import BlockPacked, _block_gemm_spec, _run_block_gemm, _stored_as
from .mac import (
    BlockMac,
    BlockQuant,
    Mac,
    Quant,
    _as_block_quant,
    spec_for_block_mac,
    spec_for_mac,
)
from .modules.format import QAffineFormats, QMatmulFormats
from .ops import (
    _binaryK_accumulated_odd_spec,
    _binaryK_accumulated_spec,
    _binaryK_fma_accumulated_odd_spec,
    _binaryK_fma_accumulated_spec,
    _binaryK_fma_odd_spec,
    _binaryK_odd_spec,
    _binaryK_superfp_accumulated_odd_spec,
    _binaryK_superfp_accumulated_spec,
    _binaryK_superfp_odd_spec,
    _checked_carrier_rounding_call,
    _gemm_mixed_nd,
    _gemm_nd,
    _run_gemm,
    _superfp_accumulated_odd_spec,
    _superfp_accumulated_spec,
    _superfp_binaryK_accumulated_odd_spec,
    _superfp_binaryK_accumulated_spec,
    _superfp_binaryK_odd_spec,
    _superfp_fma_accumulated_odd_spec,
    _superfp_fma_accumulated_spec,
    _superfp_fma_odd_spec,
    _superfp_odd_spec,
)

__all__ = [
    "binaryK_gemm_formats",
    "superfp_gemm_formats",
    "binaryK_gemm_formats_fma",
    "superfp_gemm_formats_fma",
    "binaryK_superfp_gemm_formats",
    "superfp_binaryK_gemm_formats",
    "matmul_formats",
    "block_matmul_formats",
    "block_gemm_formats",
]

# A resolved GEMM, called once per pass as `matmul(a, b, ta, tb)`: it computes
# `op(a) @ op(b)` for 2D operands, where `ta` and `tb` say whether `op`
# transposes `a` and `b` (a flag the kernel reads, so no transposed copy is made).
_Matmul2D = Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor]


def _flatten_leading(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Flatten all but the last dimension of ``x`` into one leading dimension.

    Return the 2D tensor and the original shape, so the caller can give a
    result the leading dimensions ``x`` had.
    """
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape


def _gemm_formats(
    matmul: _Matmul2D, igrad: _Matmul2D | None = None, wgrad: _Matmul2D | None = None
) -> QAffineFormats:
    """Build the three math hooks of a Linear layer, each calling a resolved 2D GEMM.

    Which operand each hook transposes, how the leading dimensions are
    flattened into one row dimension and restored, and how the bias is added
    belong to the Linear layer, not to the formats, so they are written here
    once and each factory passes only its resolved ``matmul``. That GEMM runs
    every pass, unless ``igrad`` (for the input gradient) or ``wgrad`` (for
    the weight gradient) gives a different one.
    """
    igrad_matmul = matmul if igrad is None else igrad
    wgrad_matmul = matmul if wgrad is None else wgrad

    def fwd(
        q_input: torch.Tensor, q_weight: torch.Tensor, q_bias: torch.Tensor | None
    ) -> torch.Tensor:
        x_flat, x_shape = _flatten_leading(q_input)
        out = matmul(x_flat, q_weight, False, True)
        out = out.reshape(*x_shape[:-1], q_weight.shape[0])
        # add_ rather than +: `out` is a view of a tensor the op allocated
        # inside this call, and nothing else refers to it, so adding the bias
        # in place gives the same bits as `out + q_bias` and saves one
        # output-sized allocation per forward. A bias of a different dtype
        # takes `out + q_bias`, which promotes the result's dtype where `add_`
        # would raise.
        if q_bias is None:
            return out
        return out.add_(q_bias) if q_bias.dtype == out.dtype else out + q_bias

    def bwd_igrad(q_igrad_output: torch.Tensor, q_weight: torch.Tensor) -> torch.Tensor:
        g_flat, g_shape = _flatten_leading(q_igrad_output)
        out = igrad_matmul(g_flat, q_weight, False, False)
        return out.reshape(*g_shape[:-1], q_weight.shape[1])

    def bwd_wgrad(q_wgrad_output: torch.Tensor, q_input: torch.Tensor) -> torch.Tensor:
        g_flat, _ = _flatten_leading(q_wgrad_output)
        i_flat, _ = _flatten_leading(q_input)
        return wgrad_matmul(g_flat, i_flat, True, False)

    return QAffineFormats(fwd_math=fwd, bwd_igrad_math=bwd_igrad, bwd_wgrad_math=bwd_wgrad)


def _plain_conv(
    q_input: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor | None,
    *,
    stride,
    padding,
    dilation,
    groups: int,
    nd: int,
) -> torch.Tensor:
    """The plain convolution ``CustomArithConvNd`` runs when it has no
    ``fwd_math`` hook, with the arguments of a convolution hook."""
    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[nd]
    return conv(q_input, q_weight, q_bias, stride, padding, dilation, groups)


def _with_output[FormatsT: (QAffineFormats, QMatmulFormats)](
    formats: FormatsT, out: Callable, plain: Callable
) -> FormatsT:
    """Make ``formats`` round each forward result with ``out``, after the bias is added.

    The forward runs ``formats.fwd_math``, or ``plain`` when there is none
    (``F.linear``, :func:`_plain_conv` or ``torch.matmul``, the call the
    layer's autograd Function would make), and then ``out`` rounds its result.
    The rounding happens inside the Function's forward, which the backward
    pass does not differentiate, so its gradient passes straight through. The
    result is a new tensor the hook allocated (true of every hook in this
    module), so a ``Quant`` rounds it in place. The exceptions are a
    ``Quant`` with ``carrier=torch.float64`` and a result that is not
    contiguous (a channels_last convolution's), which the in-place quantizers
    refuse; those round into a new tensor. ``formats`` is modified in place
    and returned.
    """
    run = formats.fwd_math or plain
    in_place = (
        replace(out, inplace=True)
        if type(out) is Quant and not out.inplace and out.carrier is not torch.float64
        else None
    )
    if in_place is None:

        def fwd(*args, **kwargs):
            return out(run(*args, **kwargs))

    else:

        def fwd(*args, **kwargs):
            result = run(*args, **kwargs)
            return in_place(result) if result.is_contiguous() else out(result)

    formats.fwd_math = fwd
    return formats


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
    gradient products run through :func:`mptorch.quant.ops.binaryK_matmul`,
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
            of precision, an exponent field wider than 30 bits, or too many
            stochastic-rounding bits to fit in the significand), a
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
    arguments; :func:`mptorch.quant.ops.binaryK_matmul_fma`), in place of a
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

    The matrix products run through :func:`mptorch.quant.ops.superfp_matmul`.
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
    the superfp fused format (:func:`mptorch.quant.ops.superfp_matmul_fma`).
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


# A resolved GEMM for a matmul's hooks, called as `matmul(a, b, ta, tb)`: like
# `_Matmul2D`, but under torch.matmul's operand rules (batch broadcasting and
# 1D promotion) rather than on 2D operands, which the Linear hooks above make
# by flattening. `ta` and `tb` say whether `a` and `b` are transposed.
_MatmulNd = Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor]


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
    :func:`mptorch.quant.ops.binaryK_superfp_matmul`, which documents every
    argument. Products are rounded to the binaryK multiply format (the
    ``mul_*`` arguments) and partial sums to the superfp accumulate format
    (the ``acc_*`` arguments). The accumulate format is required: for an
    unrounded sum, use :func:`binaryK_gemm_formats` with
    ``accumulate_quant=False``. Under ``BLOCK`` or ``TREE`` the outer format
    is a superfp format too. ``carrier_rounding=RoundMode.RO`` rounds each
    operation in the carrier to odd, in every pass, as for
    :func:`mptorch.quant.ops.binaryK_superfp_matmul`.

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
    :func:`mptorch.quant.ops.superfp_binaryK_matmul`, which documents every
    argument. Products are rounded to the superfp multiply format (the
    ``mul_*`` arguments) and partial sums to the binaryK accumulate format
    (the ``acc_*`` arguments). The accumulate format is required: for an
    unrounded sum, use :func:`superfp_gemm_formats` with
    ``accumulate_quant=False``. Under ``BLOCK`` or ``TREE`` the outer format
    is a binaryK format too. ``carrier_rounding=RoundMode.RO`` rounds each
    operation in the carrier to odd, in every pass, as for
    :func:`mptorch.quant.ops.superfp_binaryK_matmul`.

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


def _matmul_formats(
    fwd_matmul: _MatmulNd, agrad_matmul: _MatmulNd, bgrad_matmul: _MatmulNd
) -> QMatmulFormats:
    """Build the three math hooks of a matmul, each calling a resolved GEMM.

    The difference from :func:`_gemm_formats` is which operands each pass
    transposes: the forward is ``a @ b``, ``a``'s gradient is ``grad @ b^T``
    and ``b``'s is ``a^T @ grad``. Nothing is flattened and no bias is added,
    since a matmul broadcasts over its batch dimensions instead of merging
    them, and has no bias. It takes one GEMM per pass because a palette needs
    a different ``prec_idx`` map in each pass; every other format passes the
    same GEMM three times.
    """

    def fwd(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        return fwd_matmul(q_a, q_b, False, False)

    def bwd_agrad(q_grad: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        return agrad_matmul(q_grad, q_b, False, True)

    def bwd_bgrad(q_grad: torch.Tensor, q_a: torch.Tensor) -> torch.Tensor:
        return bgrad_matmul(q_a, q_grad, True, False)

    return QMatmulFormats(fwd_math=fwd, bwd_agrad_math=bwd_agrad, bwd_bgrad_math=bwd_bgrad)


def _mixed_matmul(spec, prec_idx: torch.Tensor | None, pass_name: str, argument: str):
    """Return one pass's GEMM over a palette op, or, without a map, a hook that raises.

    A ``prec_idx`` map has one entry per *output* element, and the three
    passes of a matmul have outputs of three different shapes: ``[M, N]``,
    ``[M, K]`` and ``[K, N]``. One map cannot serve all three, so a pass whose
    map was not given gets a hook that raises an error naming the
    ``matmul_formats`` argument to pass, instead of failing with a shape error
    from the op or silently falling back to unquantized arithmetic.
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

    The forward product ``a @ b``, ``a``'s gradient ``grad @ b^T`` and ``b``'s
    gradient ``a^T @ grad`` each run through the GEMM op that ``mac`` resolves
    to.

    A ``Palette`` in any of ``mac``'s slots selects a format per output
    element, which needs one ``prec_idx`` map per pass. The maps are bound
    into the hooks here, so they need not be passed on every call. A gradient
    pass whose map is missing raises when it runs; it does not fall back to
    unquantized arithmetic.

    Like the six binaryK and superfp ``*_gemm_formats`` factories, this sets
    only the math hooks. The caller adds the elementwise operand quantizers
    (``a_quant``, ``b_quant``, ``agrad_quant``, ``bgrad_quant``) separately.

    Args:
        mac (SplitMac or FusedMac): the dot-product arithmetic, resolved once
            through :func:`mptorch.quant.mac.spec_for_mac`.
        prec_idx (Tensor, optional): integer map selecting a palette entry for
            each element of the forward pass's output, shaped like ``a @ b``
            (``[M, N]``, ``[M, 1]`` or ``[1, N]``, optionally batched). A
            palette ``mac`` requires it, and any other ``mac`` rejects it.
            Default: ``None``
        agrad_prec_idx (Tensor, optional): the map for ``a``'s gradient,
            shaped like ``grad @ b^T``. Default: ``None``
        bgrad_prec_idx (Tensor, optional): the map for ``b``'s gradient,
            shaped like ``a^T @ grad``. Default: ``None``

    Returns:
        QMatmulFormats: with ``fwd_math``, ``bwd_agrad_math`` and
        ``bwd_bgrad_math`` set and every ``*_quant`` slot left ``None``.

    Raises:
        ValueError: if ``mac`` holds a palette and ``prec_idx`` is ``None``,
            or if any map is given for a ``mac`` with no palette to select
            from.

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
# and reads each along its reduction dimension K: an operand is either packed
# along K, or packed along its other dimension and read transposed. So each
# hook packs its operands along the dimension its pass sums over, as MX
# training recipes do, except an operand that arrives already packed. That is
# a weight packed once by `BlockQuant.pack` (in a `weight_quant` slot), or an
# operand in a square-tiled format, which quantizes to the same values along
# either axis and so can serve the backward pass, read transposed. A 1D or
# non-square format quantizes differently along each axis, so its backward
# pass packs the float operand again, and a weight that arrives packed in such
# a format cannot be used for the input gradient.


def _square_tile_hint(fmt: BlockFormat) -> str:
    """The expression that builds the square-tiled variant of a format, for messages."""
    return f"dataclasses.replace(fmt, block_rows={fmt.block_size})"


def _packed_operand(q: BlockQuant, x: "torch.Tensor | BlockPacked", axis: int) -> BlockPacked:
    """Return ``x`` packed along ``axis`` by ``q``, or ``x`` itself if it is already packed."""
    return x if isinstance(x, BlockPacked) else q.pack_fresh(x, axis)


def _reused(x: "torch.Tensor | BlockPacked", q: BlockQuant, where: str) -> BlockPacked | None:
    """Return the packing the forward pass made, for a backward pass to read
    along its other axis, or ``None`` if the float operand was saved instead
    (the caller then packs it again).

    Only a square tile quantizes to the same values along both axes, so a
    packing in a 1D or non-square format raises, and so does a packing in a
    format other than the one this pass rounds to."""
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
            f"a square tile, {_square_tile_hint(x.fmt)}, which quantizes the same along either "
            "axis (a layer frozen by freeze_packed_ has no float weight to pack again: freeze it "
            "in a square-tile format to train below it)"
        )
    return x


def block_matmul_formats(
    mac: BlockMac, *, agrad_mac: BlockMac | None = None, bgrad_mac: BlockMac | None = None
) -> QMatmulFormats:
    """Build a ``QMatmulFormats`` whose three passes are block GEMMs.

    Each pass packs its operands along the dimension it sums over:

    - The forward product ``a @ b`` packs ``a`` along its last dimension and
      ``b`` along its second-to-last (the dimension K they share), in
      ``mac``'s two formats.
    - ``a``'s gradient ``grad @ b^T`` packs both ``grad`` and ``b`` along
      their last dimension.
    - ``b``'s gradient ``a^T @ grad`` packs both ``a`` and ``grad`` along
      their second-to-last dimension.

    The gradient passes take their operands' formats from ``agrad_mac`` and
    ``bgrad_mac``, which default to ``mac``. An operand whose format has
    square tiles is packed only once, by the ``a_quant`` or ``b_quant`` slot
    that this function sets for it, and the gradient passes read that packing
    transposed, through :attr:`mptorch.quant.BlockPacked.mT`. Each pass
    computes its result in the carrier (float32 for binary32, float64 for
    binary64) and stores it in the operands' dtype.

    Args:
        mac (BlockMac): the forward pass's operand formats and arithmetic.
        agrad_mac (BlockMac, optional): the same for ``a``'s gradient.
            Default: ``None``, which takes ``mac``.
        bgrad_mac (BlockMac, optional): the same for ``b``'s gradient.
            Default: ``None``, which takes ``mac``.

    Returns:
        QMatmulFormats: with the three math hooks set, and ``a_quant`` or
        ``b_quant`` for each operand in a square-tiled format.

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
    """Return a ``BlockMac`` slot as the ``BlockQuant`` that ``BlockMac`` converted it to."""
    assert slot is not None  # BlockMac.__post_init__ sets an unset b to a
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

    def operand(slot: "BlockFormat | BlockQuant", name: str) -> BlockQuant:
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

    formats = QAffineFormats(fwd_math=fwd, bwd_igrad_math=bwd_igrad, bwd_wgrad_math=bwd_wgrad)
    # the weight's quantizer, which QLinear.freeze_packed_ uses to pack a frozen weight
    formats._block_weight = qw
    return formats
