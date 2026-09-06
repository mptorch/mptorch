"""
Factory helpers that wire the custom-arithmetic GEMM ops
(:func:`mptorch.quant.ops.binaryK_matmul` / :func:`superfp_matmul`) into a
:class:`~mptorch.quant.modules.format.QAffineFormats`'s ``fwd_math`` /
``bwd_igrad_math`` / ``bwd_wgrad_math`` hooks, matching the call contract
``CustomArithLinear`` (``mptorch/quant/modules/linear.py``) actually uses::

    fwd_math(q_input, q_weight, q_bias)        # q_input @ q_weight.T + q_bias
    bwd_igrad_math(q_igrad_output, q_weight)   # q_igrad_output @ q_weight
    bwd_wgrad_math(q_wgrad_output, q_input)    # q_wgrad_output.T @ q_input

``q_input``/``q_igrad_output``/``q_wgrad_output`` may carry arbitrary
leading (batch) dims, as ``nn.Linear`` allows; they're flattened to 2D
before each GEMM call and reshaped back afterwards, mirroring
``CustomArithLinear._default_bwd_wgrad``'s own reshape.

The four factories differ in exactly one thing -- which op the three hooks
call -- so they share :func:`_gemm_formats` and differ only in the argument
list they resolve. Each resolves its format once, at construction: a layer's
format is fixed for its lifetime, so the ``acc_*`` fallbacks, the default
exponent biases and the stochastic-rounding bound have no business being
re-derived on every forward and backward pass (dev/gemm_roadmap.md, finding
P4).

These factories only set the ``*_math`` hooks -- layer on
``weight_quant``/``input_quant``/etc. yourself on the returned
``QAffineFormats`` if elementwise operand quantization is also wanted; the
two concerns (operand quantization vs. dot-product arithmetic) are
deliberately kept composable rather than bundled.
"""

from collections.abc import Callable
from functools import partial

import torch

from mptorch.number import AccumulateAlgorithm, RoundMode, SaturationMode, SubnormalsMode

from .modules.format import QAffineFormats
from .ops import (
    _binaryK_fma_spec,
    _binaryK_spec,
    _run_gemm,
    _superfp_fma_spec,
    _superfp_spec,
)

__all__ = [
    "binaryK_gemm_formats",
    "superfp_gemm_formats",
    "binaryK_gemm_formats_fma",
    "superfp_gemm_formats_fma",
]

# ``op(a) @ op(b)`` for 2D operands, with ``op(x) = x.T`` where the flag is
# set: what a resolved GEMM still takes per call.
_Matmul2D = Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor]


def _flatten_leading(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Flatten all but the last dim of x into a single leading dim."""
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape


def _gemm_formats(matmul: _Matmul2D) -> QAffineFormats:
    """The three Linear math hooks over one 2D matmul.

    Which operand each hook transposes, how the leading dims are folded away
    and put back, and how the bias is added are the Linear contract, not the
    format's -- so they are written here once and every factory below passes
    only its ``matmul``.
    """

    def fwd(
        q_input: torch.Tensor, q_weight: torch.Tensor, q_bias: torch.Tensor | None
    ) -> torch.Tensor:
        x_flat, x_shape = _flatten_leading(q_input)
        out = matmul(x_flat, q_weight, False, True)
        out = out.reshape(*x_shape[:-1], q_weight.shape[0])
        # add_ rather than +: `out` is a view of a tensor the op allocated
        # inside this call and has no other referent, so folding the bias in
        # place is bit-identical and saves an output-sized allocation per
        # forward. Guarded on dtype so a bias in a wider dtype still promotes
        # the way `out + q_bias` did instead of raising.
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
) -> QAffineFormats:
    """
    Build a ``QAffineFormats`` whose ``fwd_math``/``bwd_igrad_math``/
    ``bwd_wgrad_math`` run a Linear layer's dot products through a
    quantized binaryK GEMM core instead of a plain matmul.

    ``mul_prng_bits``/``acc_prng_bits`` only matter when ``rounding_mode``
    is ``RoundMode.SR`` -- see :func:`mptorch.quant.ops.binaryK_matmul`,
    whose arguments these are.
    """
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
) -> QAffineFormats:
    """
    Fused-multiply-add analog of :func:`binaryK_gemm_formats` -- see its
    docstring for the general contract. Each dot-product step is a single
    hardware-style fused multiply-add rounded once (via
    :func:`mptorch.quant.ops.binaryK_matmul_fma`), rather than a quantized
    multiply followed by a separately quantized add.

    ``fma_prng_bits`` only matters when ``rounding_mode`` is
    ``RoundMode.SR``.
    """
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
) -> QAffineFormats:
    """superfp analog of :func:`binaryK_gemm_formats` -- see its docstring.

    See :func:`superfp_gemm_formats_fma` for the fused-multiply-add analog.
    """
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
) -> QAffineFormats:
    """superfp analog of :func:`binaryK_gemm_formats_fma` -- see its docstring."""
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
    )
    return _gemm_formats(partial(_run_gemm, spec))
