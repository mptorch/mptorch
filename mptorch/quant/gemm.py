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

These factories only set the ``*_math`` hooks -- layer on
``weight_quant``/``input_quant``/etc. yourself on the returned
``QAffineFormats`` if elementwise operand quantization is also wanted; the
two concerns (operand quantization vs. dot-product arithmetic) are
deliberately kept composable rather than bundled.
"""

import torch

from mptorch.number import AccumulateAlgorithm, RoundMode, SaturationMode, SubnormalsMode

from .modules.format import QAffineFormats
from .ops import binaryK_matmul, binaryK_matmul_fma, superfp_matmul, superfp_matmul_fma

__all__ = [
    "binaryK_gemm_formats",
    "superfp_gemm_formats",
    "binaryK_gemm_formats_fma",
    "superfp_gemm_formats_fma",
]


def _flatten_leading(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Flatten all but the last dim of x into a single leading dim."""
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape


def binaryK_gemm_formats(
    mul_K: int,
    mul_P: int,
    *,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    accumulate_quant: bool = True,
    acc_K: int | None = None,
    acc_P: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
) -> QAffineFormats:
    """
    Build a ``QAffineFormats`` whose ``fwd_math``/``bwd_igrad_math``/
    ``bwd_wgrad_math`` run a Linear layer's dot products through a
    quantized binaryK GEMM core instead of a plain matmul.
    """

    def _matmul(a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool) -> torch.Tensor:
        return binaryK_matmul(
            a,
            b,
            trans_a=trans_a,
            trans_b=trans_b,
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            accumulate_quant=accumulate_quant,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
        )

    def fwd(
        q_input: torch.Tensor, q_weight: torch.Tensor, q_bias: torch.Tensor | None
    ) -> torch.Tensor:
        x_flat, x_shape = _flatten_leading(q_input)
        out = _matmul(x_flat, q_weight, trans_a=False, trans_b=True)
        out = out.reshape(*x_shape[:-1], q_weight.shape[0])
        return out + q_bias if q_bias is not None else out

    def bwd_igrad(q_igrad_output: torch.Tensor, q_weight: torch.Tensor) -> torch.Tensor:
        g_flat, g_shape = _flatten_leading(q_igrad_output)
        out = _matmul(g_flat, q_weight, trans_a=False, trans_b=False)
        return out.reshape(*g_shape[:-1], q_weight.shape[1])

    def bwd_wgrad(q_wgrad_output: torch.Tensor, q_input: torch.Tensor) -> torch.Tensor:
        g_flat, _ = _flatten_leading(q_wgrad_output)
        i_flat, _ = _flatten_leading(q_input)
        return _matmul(g_flat, i_flat, trans_a=True, trans_b=False)

    return QAffineFormats(fwd_math=fwd, bwd_igrad_math=bwd_igrad, bwd_wgrad_math=bwd_wgrad)


def binaryK_gemm_formats_fma(
    fma_K: int,
    fma_P: int,
    *,
    fma_bias: int | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
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
    """

    def _matmul(a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool) -> torch.Tensor:
        return binaryK_matmul_fma(
            a,
            b,
            trans_a=trans_a,
            trans_b=trans_b,
            fma_K=fma_K,
            fma_P=fma_P,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
        )

    def fwd(
        q_input: torch.Tensor, q_weight: torch.Tensor, q_bias: torch.Tensor | None
    ) -> torch.Tensor:
        x_flat, x_shape = _flatten_leading(q_input)
        out = _matmul(x_flat, q_weight, trans_a=False, trans_b=True)
        out = out.reshape(*x_shape[:-1], q_weight.shape[0])
        return out + q_bias if q_bias is not None else out

    def bwd_igrad(q_igrad_output: torch.Tensor, q_weight: torch.Tensor) -> torch.Tensor:
        g_flat, g_shape = _flatten_leading(q_igrad_output)
        out = _matmul(g_flat, q_weight, trans_a=False, trans_b=False)
        return out.reshape(*g_shape[:-1], q_weight.shape[1])

    def bwd_wgrad(q_wgrad_output: torch.Tensor, q_input: torch.Tensor) -> torch.Tensor:
        g_flat, _ = _flatten_leading(q_wgrad_output)
        i_flat, _ = _flatten_leading(q_input)
        return _matmul(g_flat, i_flat, trans_a=True, trans_b=False)

    return QAffineFormats(fwd_math=fwd, bwd_igrad_math=bwd_igrad, bwd_wgrad_math=bwd_wgrad)


def superfp_gemm_formats(
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    *,
    mul_is_signed: bool = True,
    accumulate_quant: bool = True,
    acc_man_bits: int | None = None,
    acc_exp_bits: int | None = None,
    acc_normal_binades: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> QAffineFormats:
    """superfp analog of :func:`binaryK_gemm_formats` -- see its docstring.

    See :func:`superfp_gemm_formats_fma` for the fused-multiply-add analog.
    """

    def _matmul(a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool) -> torch.Tensor:
        return superfp_matmul(
            a,
            b,
            trans_a=trans_a,
            trans_b=trans_b,
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            accumulate_quant=accumulate_quant,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
        )

    def fwd(
        q_input: torch.Tensor, q_weight: torch.Tensor, q_bias: torch.Tensor | None
    ) -> torch.Tensor:
        x_flat, x_shape = _flatten_leading(q_input)
        out = _matmul(x_flat, q_weight, trans_a=False, trans_b=True)
        out = out.reshape(*x_shape[:-1], q_weight.shape[0])
        return out + q_bias if q_bias is not None else out

    def bwd_igrad(q_igrad_output: torch.Tensor, q_weight: torch.Tensor) -> torch.Tensor:
        g_flat, g_shape = _flatten_leading(q_igrad_output)
        out = _matmul(g_flat, q_weight, trans_a=False, trans_b=False)
        return out.reshape(*g_shape[:-1], q_weight.shape[1])

    def bwd_wgrad(q_wgrad_output: torch.Tensor, q_input: torch.Tensor) -> torch.Tensor:
        g_flat, _ = _flatten_leading(q_wgrad_output)
        i_flat, _ = _flatten_leading(q_input)
        return _matmul(g_flat, i_flat, trans_a=True, trans_b=False)

    return QAffineFormats(fwd_math=fwd, bwd_igrad_math=bwd_igrad, bwd_wgrad_math=bwd_wgrad)


def superfp_gemm_formats_fma(
    fma_man_bits: int,
    fma_exp_bits: int,
    fma_normal_binades: int,
    fma_bias: int,
    *,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> QAffineFormats:
    """superfp analog of :func:`binaryK_gemm_formats_fma` -- see its docstring."""

    def _matmul(a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool) -> torch.Tensor:
        return superfp_matmul_fma(
            a,
            b,
            trans_a=trans_a,
            trans_b=trans_b,
            fma_man_bits=fma_man_bits,
            fma_exp_bits=fma_exp_bits,
            fma_normal_binades=fma_normal_binades,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
        )

    def fwd(
        q_input: torch.Tensor, q_weight: torch.Tensor, q_bias: torch.Tensor | None
    ) -> torch.Tensor:
        x_flat, x_shape = _flatten_leading(q_input)
        out = _matmul(x_flat, q_weight, trans_a=False, trans_b=True)
        out = out.reshape(*x_shape[:-1], q_weight.shape[0])
        return out + q_bias if q_bias is not None else out

    def bwd_igrad(q_igrad_output: torch.Tensor, q_weight: torch.Tensor) -> torch.Tensor:
        g_flat, g_shape = _flatten_leading(q_igrad_output)
        out = _matmul(g_flat, q_weight, trans_a=False, trans_b=False)
        return out.reshape(*g_shape[:-1], q_weight.shape[1])

    def bwd_wgrad(q_wgrad_output: torch.Tensor, q_input: torch.Tensor) -> torch.Tensor:
        g_flat, _ = _flatten_leading(q_wgrad_output)
        i_flat, _ = _flatten_leading(q_input)
        return _matmul(g_flat, i_flat, trans_a=True, trans_b=False)

    return QAffineFormats(fwd_math=fwd, bwd_igrad_math=bwd_igrad, bwd_wgrad_math=bwd_wgrad)
