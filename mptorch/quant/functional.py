from typing import Callable, Literal
import torch
import math
from .quant_function import *
from ..number import FloatType
from .quant_format import (
    QAffineFormats,
    QGELUFormats,
    QSoftmaxFormats,
    make_quant_function,
)

__all__ = [
    "qlinear",
    "qmatmul",
    "qmm",
    "qadd",
    "qmul",
    "qsqrt",
    "qdiv",
    "qpow",
    "qsum",
    "qmean",
    "qlayernorm",
    "qsoftmax",
    "qgelu",
]


# Utility functions to perform tensor scaling operations in low precison
# (8-bit and below) DNN training
# See: https://arxiv.org/pdf/2309.17224
# NOTE: using this routine seems quite sensitive to how the margin term is picked,
# notably the empirical value suggested in the aformentioned reference leads to
# divergence (i.e. grad_input computation in the linear layer backward pass is
# unstable and produces NaNs right from the start); more investigation is needed
# and this might need to be revisited in the case of supernormals
def compute_bias(x: torch.Tensor, cast_to: FloatType, margin: int = 11) -> torch.Tensor:
    with torch.no_grad():
        (amax, _) = torch.max(torch.abs(x), dim=-1, keepdim=True)
        (amax, _) = torch.max(amax, dim=-2, keepdim=True)
    return torch.floor(torch.log2(cast_to.normal_max / amax)) - margin


def scale(x: torch.Tensor, x_scale: torch.Tensor | int) -> torch.Tensor:
    if isinstance(x_scale, torch.Tensor):
        return x * torch.pow(2.0 * torch.ones_like(x), x_scale)
    else:
        return x


def unscale(x: torch.Tensor, x_scale: torch.Tensor | int) -> torch.Tensor:
    if isinstance(x_scale, torch.Tensor):
        return x * torch.pow(2.0 * torch.ones_like(x), -x_scale)
    else:
        return x


# Take inspiration for defining the custom derivation formulas from the PyTorch repository
# See: https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml


class qlinear_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, formats):
        ctx.formats = formats
        # NOTE: (optimization) precompute the LP weights and weight scale
        if (
            (formats.weight_scaled_format is not None)
            and (formats.input_scaled_format is not None)
            and formats.use_scaling
        ):
            ctx.weight_scale = compute_bias(weight, formats.weight_scaled_format)
            ctx.input_scale = compute_bias(input, formats.input_scaled_format)
        else:
            ctx.input_scale, ctx.weight_scale = 0, 0
        qinput = formats.input_quant(scale(input, ctx.input_scale))
        qweight = formats.weight_quant(scale(weight, ctx.weight_scale))
        # NOTE: investigate if the bias term needs to be scaled as well
        if bias is not None:
            qbias = formats.bias_quant(bias)
        else:
            qbias = None

        if formats.fwd_use_default_prec:
            with torch.no_grad():
                output = torch.nn.functional.linear(qinput, qweight)
        else:
            output = mp_bmm(qinput, qweight.t(), formats, use_forward=True)

        output = unscale(output, ctx.input_scale + ctx.weight_scale)

        if qbias is not None:
            output += qbias.view(
                -1, qbias.shape[-1]
            )  # broadcasting should be done automatically

        ctx.save_for_backward(qinput, qweight, qbias)
        qoutput = formats.output_quant(output)

        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qweight, qbias = ctx.saved_tensors
        qgrad_input, qgrad_weight, qgrad_bias = None, None, None

        if ctx.formats.grad_scaled_format is not None and ctx.formats.use_scaling:
            grad_scale = compute_bias(grad_output, ctx.formats.grad_scaled_format)
        else:
            grad_scale = 0
        qgrad_output = ctx.formats.grad_quant(scale(grad_output, grad_scale))

        if ctx.needs_input_grad[0]:
            if ctx.formats.bwd_use_default_prec:
                qgrad_input = torch.matmul(qgrad_output, qweight)
            else:
                qgrad_input = mp_bmm(
                    qgrad_output,
                    qweight,
                    formats=ctx.formats,
                    use_forward=False,
                )

            qgrad_input = unscale(qgrad_input, grad_scale + ctx.weight_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_input = ctx.formats.grad_quant(qgrad_input)

        if ctx.needs_input_grad[1]:
            if ctx.formats.bwd_use_default_prec:
                qgrad_weight = torch.matmul(qgrad_output.transpose(-2, -1), qinput)
            else:
                qgrad_weight = mp_bmm(
                    qgrad_output.transpose(-2, -1),
                    qinput,
                    formats=ctx.formats,
                    use_forward=False,
                )
            qgrad_weight = unscale(qgrad_weight, grad_scale + ctx.input_scale)

            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_weight = ctx.formats.grad_quant(qgrad_weight)

        if qbias is not None and ctx.needs_input_grad[2]:
            if ctx.formats.bwd_use_default_prec:
                qgrad_bias = qgrad_output.sum(
                    dim=tuple([i for i in range(len(qgrad_output.shape) - 1)])
                ).reshape(qbias.shape)
            else:
                with torch.no_grad():
                    qgrad_bias = qsum(
                        qgrad_output,
                        dim=tuple([i for i in range(len(qgrad_output.shape) - 1)]),
                        quant=make_quant_function(
                            ctx.formats.bwd_add,
                            ctx.formats.bwd_rnd,
                            ctx.formats.rbits_add,
                        ),
                    ).reshape(qbias.shape)
            qgrad_bias = unscale(qgrad_bias, grad_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_bias = ctx.formats.grad_quant(qgrad_bias)

        return qgrad_input, qgrad_weight, qgrad_bias, None, None


def qlinear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    formats: QAffineFormats = QAffineFormats(),
) -> torch.Tensor:
    r"""Applies a linear transformation to the incoming data: :math:`y=xW^T + b`

    The :attr:`formats` parameter allows one to specify if I/O signals should be
    quantized during inference & training (needed for instance in QAT and PTQ methods),
    but also the precision(s) to be used in internal GEMM computations (addition and
    multiplication, fused or not). This allows simulating the effect of custom precision
    during GEMM calls in the forward and backward pass and is helpful in studying the
    effect of low precision compute during inference and training (not just data
    quantization).

    This is the functional version of :class:`~mptorch.quant.modules.QLinear`.

    Args:
        input: the input :math:`x` to the linear layer of the form :math:`(*, H_\text{in})`,
            where :math:`*` means any number of dimensions including none and
            :math:`H_{in} = \text{in_features}`
        weight: the weight tensor :math:`W` of shape :math:`(\text{out_features}, \text{in_features})`
        bias: optional bias term of shape :math:`(\text{out_features})`
        formats: the configuration object for how quantization (if any!) should be handled
            on the matrix inputs and how the MAC and summation operations should be performed
            (e.g. using compensated algorithms or not)

    Returns:
        the result of the affine operation :math:`xW^T + b`

    Example:
        .. code-block:: python

            # TODO

    """
    return qlinear_kernel.apply(input, weight, bias, formats)


class qmm_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other, formats):
        ctx.formats = formats
        if (
            (formats.weight_scaled_format is not None)
            and (formats.input_scaled_format is not None)
            and formats.use_scaling
        ):
            ctx.weight_scale = compute_bias(other, formats.weight_scaled_format)
            ctx.input_scale = compute_bias(input, formats.input_scaled_format)
        else:
            ctx.input_scale, ctx.weight_scale = 0, 0

        qinput = formats.input_quant(scale(input, ctx.input_scale))
        qother = formats.weight_quant(scale(other, ctx.weight_scale))

        if formats.fwd_use_default_prec:
            with torch.no_grad():
                output = torch.mm(qinput, qother)
        else:
            output = mp_mm(
                qinput,
                qother,
                formats=formats,
                use_forward=True,
            )

        output = unscale(output, ctx.input_scale + ctx.weight_scale)

        ctx.save_for_backward(qinput, qother)
        qoutput = formats.output_quant(output)
        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qother = ctx.saved_tensors
        qgrad_input, qgrad_other = None, None

        if ctx.formats.grad_scaled_format is not None and ctx.formats.use_scaling:
            grad_scale = compute_bias(grad_output, ctx.formats.grad_scaled_format)
        else:
            grad_scale = 0
        qgrad_output = ctx.formats.grad_quant(scale(grad_output, grad_scale))

        if ctx.needs_input_grad[0]:
            if ctx.formats.bwd_use_default_prec:
                grad_input = torch.mm(qgrad_output, qother.transpose(-2, -1))
            else:
                grad_input = mp_mm(
                    qgrad_output,
                    qother.transpose(-2, -1),
                    formats=ctx.formats,
                    use_forward=False,
                )
            qgrad_input = unscale(grad_input, grad_scale + ctx.weight_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_input = ctx.formats.grad_quant(grad_input)

        if ctx.needs_input_grad[1]:
            if ctx.formats.bwd_use_default_prec:
                grad_other = torch.mm(qinput.transpose(-2, -1), qgrad_output)
            else:
                grad_other = mp_mm(
                    qinput.transpose(-2, -1),
                    qgrad_output,
                    formats=ctx.formats,
                    use_forward=False,
                )
            qgrad_other = unscale(grad_other, grad_scale + ctx.input_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_other = ctx.formats.grad_quant(grad_other)

        return qgrad_input, qgrad_other, None


def qmm(
    input: torch.Tensor, mat2: torch.Tensor, formats: QAffineFormats = QAffineFormats()
) -> torch.Tensor:
    r"""
    Simulates a mixed-precision computation pipeline for matrix multiplication of the
    matrices :attr:`input` and :attr:`mat2`.

    If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
    :math:`(m \times p)` tensor, the output tensor will be a :math:`(n \times p)`
    tensor.

    .. note:: This function does not broadcast. For broadcasting quantized matrix
              products, see :func:`mptorch.quant.functional.qmatmul`.

    Args:
        input: the first matrix to be multiplied
        mat2: the second matrix to be multiplied
        formats: the configuration object for how quantization (if any!) should be handled on the matrix inputs
            and how the MAC and summation operations should be performed (e.g. using compensated algorithms or not)

    Returns:
        the result of the matrix multiplication between :attr:`input` and :attr:`mat2`

    Example:
        .. code-block:: python

            # TODO
    """
    return qmm_kernel.apply(input, mat2, formats)


class qmatmul_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other, formats):
        ctx.formats = formats
        # NOTE: need to see how we do things for each matrix in
        # the batched version
        if (
            (formats.weight_scaled_format is not None)
            and (formats.input_scaled_format is not None)
            and formats.use_scaling
        ):
            ctx.weight_scale = compute_bias(other, formats.weight_scaled_format)
            ctx.input_scale = compute_bias(input, formats.input_scaled_format)
        else:
            ctx.input_scale, ctx.weight_scale = 0, 0
        qinput = formats.input_quant(scale(input, ctx.input_scale))
        qother = formats.weight_quant(scale(other, ctx.weight_scale))

        if formats.fwd_use_default_prec:
            with torch.no_grad():
                output = torch.matmul(qinput, qother)
        else:
            output = mp_bmm(
                qinput,
                qother,
                formats=formats,
                use_forward=True,
            )
        output = unscale(output, ctx.input_scale + ctx.weight_scale)

        ctx.save_for_backward(qinput, qother)
        qoutput = formats.output_quant(output)
        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qother = ctx.saved_tensors
        qgrad_input, qgrad_other = None, None

        if ctx.formats.grad_scaled_format is not None and ctx.formats.use_scaling:
            grad_scale = compute_bias(grad_output, ctx.formats.grad_scaled_format)
        else:
            grad_scale = 0
        qgrad_output = ctx.formats.grad_quant(scale(grad_output, grad_scale))

        if ctx.needs_input_grad[0]:
            if ctx.formats.bwd_use_default_prec:
                grad_input = torch.matmul(qgrad_output, qother.transpose(-2, -1))
                grad_input = torch.matmul(qgrad_output, qother.transpose(-2, -1))
            else:
                grad_input = mp_bmm(
                    qgrad_output,
                    qother.transpose(-2, -1),
                    formats=ctx.formats,
                    use_forward=False,
                )
            qgrad_input = unscale(grad_input, grad_scale + ctx.weight_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_input = ctx.formats.grad_quant(qgrad_input)

        if ctx.needs_input_grad[1]:
            if ctx.formats.bwd_use_default_prec:
                grad_other = torch.matmul(qinput.transpose(-2, -1), qgrad_output)
                grad_other = torch.matmul(qinput.transpose(-2, -1), qgrad_output)
            else:
                grad_other = mp_bmm(
                    qinput.transpose(-2, -1),
                    qgrad_output,
                    formats=ctx.formats,
                    use_forward=False,
                )
            qgrad_other = unscale(grad_other, grad_scale + ctx.input_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_other = ctx.formats.grad_quant(qgrad_other)

        return qgrad_input, qgrad_other, None


def qmatmul(
    input: torch.Tensor, other: torch.Tensor, formats: QAffineFormats = QAffineFormats()
) -> torch.Tensor:
    r"""
    Simulates a mixed-precision computation pipeline for (batched) matrix multiplication of the
    tensors :attr:`input` and :attr:`other`.

    The behavior depends on the dimensionality of the tensors as follows:

    - If both tensors are 1-dimensional, the dot product (scalar) is returned.
    - If both arguments are 2-dimensional, the matrix-matrix product is returned.
    - If the first argument is 1-dimensional and the second argument is 2-dimensional,
      a 1 is prepended to its dimension for the purpose of the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.
    - If the first argument is 2-dimensional and the second argument is 1-dimensional,
      the matrix-vector product is returned.
    - If both arguments are at least 1-dimensional and at least one argument is
      N-dimensional (where 2 < N < 5), then a batched matrix multiply is returned.  If the first
      argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
      batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
      1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
      The non-matrix (i.e. batch) dimensions are broadcasted (and thus
      must be broadcastable in the PyTorch sense).  For example, if :attr:`input` is a
      :math:`(j \times 1 \times n \times n)` tensor and :attr:`other` is a :math:`(k \times n \times n)`
      tensor, :attr:`out` will be a :math:`(j \times k \times n \times n)` tensor.

      Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs
      are broadcastable, and not the matrix dimensions. For example, if :attr:`input` is a
      :math:`(j \times 1 \times n \times m)` tensor and :attr:`other` is a :math:`(k \times m \times p)`
      tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
      matrix dimensions) are different. :attr:`out` will be a :math:`(j \times k \times n \times p)` tensor.

    Args:
        input: the first tensor to be multiplied
        other: the second tensor to be multiplied
        formats: the configuration object for how quantization (if any!) should be handled on the tensor inputs
            and how the MAC and summation operations should be performed (e.g. using compensated algorithms or not)

    Returns:
        the result of the (batched) matrix multiplication between :attr:`input` and :attr:`other`

    Example:
        .. code-block:: python

            # TODO
    """
    return qmatmul_kernel.apply(input, other, formats)


class qadd_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, fwd_quant, bwd_quant):
        ctx.save_for_backward(x, y)
        ctx.bwd_quant = bwd_quant
        z = x + y
        z = fwd_quant(z.contiguous())
        return z

    @staticmethod
    def backward(ctx, grad_z):
        x, y = ctx.saved_tensors
        grad_x = grad_z * torch.ones_like(x, device=x.device)
        grad_y = grad_z * torch.ones_like(y, device=y.device)

        grad_x = ctx.bwd_quant(grad_x.contiguous())
        grad_y = ctx.bwd_quant(grad_y.contiguous())
        return grad_x, grad_y, None, None


def qadd(
    x: torch.Tensor,
    y: torch.Tensor,
    fwd_quant: Callable[[torch.Tensor], torch.Tensor],
    bwd_quant: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""
    Adds :attr:`x` to :attr:`y`. Uses :attr:`fwd_quant` to quantize the result of the addition
    (e.g. can simulate the execution of the addition in low-precision, assuming the inputs are
    already in low precision). The :attr:`bwd_quant` function is used to quantize the gradients
    from the operator during the backward pass.

    For the forward computation:

        .. math::

            \text{out} = \mathcal{Q}_\text{fwd}(\text{x} + \text{y})

    For the backward computation:

        .. math::

            \text{grad_x} = \mathcal{Q}_\text{bwd}(\text{grad_z} * \text{ones_like}(\text{x}))

            \text{grad_y} = \mathcal{Q}_\text{bwd}(\text{grad_z} * \text{ones_like}(\text{y}))

    Args:
        x: the input tensor
        y: the other tensor to add to :attr:`x`
        fwd_quant: the quantization function to apply on the forward addition
        bwd_quant: the quantization function to apply on the gradient computations in the backward pass

    Returns:
        the quantized result of the addition operation between `x` and `y`

    Example:
        .. code-block:: python

            # TODO
    """
    # NOTE: see if it makes sense to implement the `alpha` version
    # NOTE: maybe worthwhile to have a version where `y` can be a number as well (in that case, no gradient for it)
    return qadd_kernel.apply(x, y, fwd_quant, bwd_quant)


class qmul_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, fwd_quant, bwd_quant):
        ctx.save_for_backward(x, y)
        ctx.bwd_quant = bwd_quant
        z = x * y
        z = fwd_quant(z.contiguous())
        return z

    @staticmethod
    def backward(ctx, grad_z):
        x, y = ctx.saved_tensors
        grad_x = grad_z * y
        grad_y = grad_z * x
        grad_x = ctx.bwd_quant(grad_x.contiguous())
        grad_y = ctx.bwd_quant(grad_y.contiguous())
        return grad_x, grad_y, None, None


def qmul(
    x: torch.Tensor,
    y: torch.Tensor,
    fwd_quant: Callable[[torch.Tensor], torch.Tensor],
    bwd_quant: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""
    Multiplies :attr:`x` by :attr:`y`. Uses :attr:`fwd_quant` to quantize the result of the multiplication
    (e.g. can simulate the execution of the multiplication in low-precision, assuming the inputs are
    already in low precision). The :attr:`bwd_quant` function is used to quantize the gradients
    from the operator during the backward pass.

    For the forward computation:

        .. math::

            \text{out} = \mathcal{Q}_\text{fwd}(\text{x} * \text{y})

    For the backward computation:

        .. math::

            \text{grad_x} = \mathcal{Q}_\text{bwd}(\text{grad_z} * \text{y})

            \text{grad_y} = \mathcal{Q}_\text{bwd}(\text{grad_z} * \text{x})

    Args:
        x: the input tensor
        y: the other tensor to add to :attr:`x`
        fwd_quant: the quantization function to apply on the forward multiplication
        bwd_quant: the quantization function to apply on the gradient computations in the backward pass

    Returns:
        the quantized result of the multiplication operation between `x` and `y`

    Example:
        .. code-block:: python

            # TODO
    """
    return qmul_kernel.apply(x, y, fwd_quant, bwd_quant)


# see the following link for a discussion regarding numerical stability of
# backward propagation for division operations in PyTorch and for the basis
# of this implementation: https://github.com/pytorch/pytorch/issues/43414
class qdiv_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, fwd_quant, bwd_quant):
        z = fwd_quant(x / y)
        ctx.save_for_backward(x, y, z)
        ctx.bwd_quant = bwd_quant
        return z

    @staticmethod
    def backward(ctx, grad_z):
        x, y, z = ctx.saved_tensors
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.bwd_quant(grad_z / y)
        if ctx.needs_input_grad[1]:
            grad_y = ctx.bwd_quant(-grad_z * z)
            grad_y = ctx.bwd_quant(grad_y / y)
        return grad_x, grad_y, None, None


def qdiv(
    x: torch.Tensor,
    y: torch.Tensor,
    fwd_quant: Callable[[torch.Tensor], torch.Tensor],
    bwd_quant: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""
    Divides :attr:`x` by :attr:`y`. Uses :attr:`fwd_quant` to quantize the result of the division
    (e.g. can simulate the execution of the division in low-precision, assuming the inputs are
    already in low precision). The :attr:`bwd_quant` function is used to quantize the gradients
    from the operator during the backward pass.

    For the forward computation:

        .. math::

            \text{out}_i = \mathcal{Q}_\text{fwd}\left(\frac{\text{x}_i}{\text{y}_i}\right)

    For the backward computation:

        .. math::

            \text{grad_x} = \mathcal{Q}_\text{bwd}\left(\frac{\text{grad_z}}{\text{y}}\right)

            \text{grad_y} = \mathcal{Q}_\text{bwd}\left(\frac{\mathcal{Q}_\text{bwd}\left(-\text{grad_z} * \text{x}\right)}{\text{y}}\right)

    Args:
        x: the input tensor
        y: the other tensor to add to :attr:`x`
        fwd_quant: the quantization function to apply on the forward division
        bwd_quant: the quantization function to apply on the gradient computations in the backward pass

    Returns:
        the quantized result of the division operation between `x` and `y`

    Example:
        .. code-block:: python

            # TODO
    """
    return qdiv_kernel.apply(x, y, fwd_quant, bwd_quant)


class qpow_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fwd_quant, bwd_quant, n=2):
        ctx.n = n
        ctx.bwd_quant = bwd_quant
        y = fwd_quant(x**n)
        gy = bwd_quant(x ** (n - 1))
        ctx.save_for_backward(gy)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (gy,) = ctx.saved_tensors
        grad_x = ctx.bwd_quant(gy * ctx.n)
        grad_x = ctx.bwd_quant(grad_y * grad_x)
        return grad_x, None, None, None


def qpow(
    x: torch.Tensor,
    fwd_quant: Callable[[torch.Tensor], torch.Tensor],
    bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    n: torch.Tensor | float = 2.0,
) -> torch.Tensor:
    r"""
    Takes the power of each element in :attr:`x` with :attr:`n` and
    returns a tensor with the result. :attr:`n` can be either a single
    ``float`` number or a `torch.Tensor` with the same number of
    elements as :attr:`x`.

    When :attr:`n` is a scalar value, the forward operation applied is:

    .. math::

        \text{out}_i = \mathcal{Q}_\text{fwd}\left(\text{x}_i^\text{n}\right)

    When :attr:`n` is a tensor, the operation applied is:

    .. math::

        \text{out}_i = \mathcal{Q}_\text{fwd}\left(\text{x}_i^{\text{n}_i}\right)

    When :attr:`n` is a tensor, the shapes of :attr:`x` and
    :attr:`n` must be broadcastable.

    The backward operation is (applied element-wise):

    .. math::

        \text{out} = \mathcal{Q}_\text{bwd}\left(\text{grad_out} *\mathcal{Q}_\text{bwd}\left(\mathcal{Q}_\text{bwd}\left(\text{x}^{\text{n}-1}\right) * \text{n}\right)\right)

    Args:
        x: the input tensor
        fwd_quant: the quantization function to apply on the forward division
        bwd_quant: the quantization function to apply on the gradient computations in the backward pass
        n: the exponent value

    Returns:
        the quantized result of the power operation between `x` and `n`

    Example:
        .. code-block:: python

            # TODO
    """
    return qpow_kernel.apply(x, fwd_quant, bwd_quant, n)


class qsqrt_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fwd_quant, bwd_quant):
        ctx.bwd_quant = bwd_quant
        x = torch.sqrt(x)
        y = fwd_quant(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (y,) = ctx.saved_tensors
        grad_x = ctx.bwd_quant(y * 2)
        grad_x = ctx.bwd_quant(grad_y / grad_x)
        return grad_x, None, None


def qsqrt(
    x: torch.Tensor,
    fwd_quant: Callable[[torch.Tensor], torch.Tensor],
    bwd_quant: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""
    Returns a new tensor with the square-root of the elements of :attr:`x`.

    .. math::
        \text{out}_{i} = \sqrt{\text{x}_{i}}

    Args:
        x: the input tensor
        fwd_quant: the quantization function to apply on the forward square root operation
        bwd_quant: the quantization function to apply on the gradient computations in the backward pass

    Returns:
        the quantized result of the square root operation on `x`

    Example:
        .. code-block:: python

            # TODO
    """
    return qsqrt_kernel.apply(x, fwd_quant, bwd_quant)


# NOTE: look into CUDA-accelerated version of this routine
# and also of sums across multiple dimensions
def qsum_1d(x, dim, quant):
    shape = list(x.shape)
    shape[dim] = 1
    vs = torch.zeros(shape, device=x.device)
    vs = vs.transpose(0, dim).reshape(1, -1).transpose(0, 1)
    vx = torch.transpose(x, 0, dim).reshape(x.shape[dim], -1).transpose(0, 1)
    shape[dim] = x.shape[0]
    shape[0] = 1
    for k in range(x.shape[dim]):
        vs = quant(vs + vx[:, k : k + 1])
    return vs.transpose(0, 1).reshape(shape).transpose(0, dim)


class qsum_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quant, dim=0, keepdim=False):
        ctx.save_for_backward(x)

        if dim is None:
            dim = range(x.dim())
        elif isinstance(dim, tuple) == False:
            dim = (dim,)

        ctx.dim = dim
        sums = x
        for d in dim:
            sums = qsum_1d(sums, d, quant)
        if keepdim is False:
            for _ in dim:
                idx = 0
                while sums.shape[idx] != 1:
                    idx += 1
                sums = torch.squeeze(sums, dim=idx)
        ctx.shape = sums.shape
        return sums

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        for d in ctx.dim:
            grad_output = torch.unsqueeze(grad_output, d)
        grad_x = grad_output * torch.ones_like(x, device=x.device)
        return grad_x, None, None, None


def qsum(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    quant: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    keepdim: bool = False,
) -> torch.Tensor:
    r""""""
    return qsum_kernel.apply(x, quant, dim, keepdim)


# NOTE: similar to sum, look into a CUDA-accelerated version of this routine
class qmean_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fwd_quant, bwd_quant, dim=3, keepdim=False):
        ctx.bwd_quant = bwd_quant
        ctx.save_for_backward(x)

        if dim is None:
            dim = range(x.dim())
        elif isinstance(dim, tuple) == False:
            dim = (dim,)

        ctx.dim = dim
        sums = x
        numel = 1
        for d in dim:
            sums = qsum(sums, d, fwd_quant, keepdim)
            numel *= x.shape[d]
        ctx.numel = numel
        sums = fwd_quant(sums / numel)
        if keepdim is False:
            for _ in dim:
                idx = 0
                while idx < len(sums.shape) and sums.shape[idx] != 1:
                    idx += 1
                if idx < len(sums.shape):
                    sums = torch.squeeze(sums, dim=idx)
        ctx.shape = sums.shape
        return sums

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        for d in ctx.dim:
            grad_output = torch.unsqueeze(grad_output, d)
        grad_x = ctx.bwd_quant(
            grad_output * torch.ones_like(x, device=x.device) / ctx.numel
        )
        return grad_x, None, None, None, None


def qmean(
    x: torch.Tensor,
    fwd_quant: Callable[[torch.Tensor], torch.Tensor],
    bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    dim: int | tuple[int, ...] | None = 3,
    keepdim: bool = False,
) -> torch.Tensor:
    return qmean_kernel.apply(x, fwd_quant, bwd_quant, dim, keepdim)


class qlayernorm_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, formats):
        ctx.formats = formats
        ctx.eps = eps
        qinput = formats.input_quant(x)
        output = torch.zeros_like(x)

        if weight is not None:
            qweight = formats.weight_quant(weight)
        else:
            qweight = torch.ones(normalized_shape, device=x.device)

        if bias is not None:
            qbias = formats.bias_quant(bias)
        else:
            qbias = torch.zeros(normalized_shape, device=x.device)

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        assert normalized_shape == x.shape[-len(normalized_shape) :]
        dims = [-(i + 1) for i in range(len(normalized_shape))]
        ctx.dims = dims

        if ctx.formats.fwd_use_default_prec:
            B, T, C = x.size()

            mean = x.sum(dims, keepdim=True) / C

            xshift = x - mean
            variance = (xshift**2).sum(dims, keepdim=True) / C

            rstd = (variance + eps) ** -0.5
            norm = xshift * rstd
            output = norm * qweight + qbias
        else:
            output, mean, rstd = mp_layernorm_forward(
                qinput, qweight, qbias, eps, dims, formats
            )

        qoutput = formats.output_quant(output)

        ctx.save_for_backward(qinput, qweight, qbias, mean, rstd)

        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        (
            qinput,
            qweight,
            qbias,
            mean,
            rstd,
        ) = ctx.saved_tensors
        formats = ctx.formats
        dims = ctx.dims

        qgrad_output = formats.grad_quant(grad_output)

        qgrad_input = None
        qgrad_weight = None
        qgrad_bias = None

        if ctx.formats.bwd_use_default_prec:
            norm = (qinput - mean) * rstd

            grad_bias = grad_output.sum((0, 1))
            grad_weight = (grad_output * norm).sum((0, 1))

            grad_norm = grad_output * grad_weight
            grad_input = (
                grad_norm
                - grad_norm.mean(-1, keepdim=True)
                - norm * (grad_norm * norm).mean(-1, keepdim=True)
            )
            grad_input *= rstd
        else:
            grad_input, grad_weight, grad_bias = mp_layernorm_backward(
                qinput, qgrad_output, qweight, qbias, mean, rstd, dims, formats
            )

        if ctx.needs_input_grad[0]:
            qgrad_input = formats.grad_quant(grad_input)
        if qweight is not None and ctx.needs_input_grad[2]:
            qgrad_weight = formats.grad_quant(grad_weight)
        if qbias is not None and ctx.needs_input_grad[3]:
            qgrad_bias = formats.grad_quant(grad_bias)

        return qgrad_input, None, qgrad_weight, qgrad_bias, None, None


def qlayernorm(x: torch.Tensor, normalized_shape, weight, bias, eps, formats):
    return qlayernorm_kernel.apply(x, normalized_shape, weight, bias, eps, formats)


class qsoftmax_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, dim, formats):
        ctx.formats = formats
        ctx.dim = dim

        qinput = formats.input_quant(a)

        if formats.fwd_use_default_prec:
            with torch.no_grad():
                output = torch.nn.functional.softmax(qinput, dim)
        else:
            output = mp_softmax_forward(qinput, dim, formats)

        qoutput = formats.output_quant(output)
        ctx.save_for_backward(qoutput)
        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        (qoutput,) = ctx.saved_tensors
        formats = ctx.formats
        dim = ctx.dim

        qgrad_output = formats.grad_quant(grad_output)

        if formats.bwd_use_default_prec:
            weighted_grad_sum = torch.sum(qoutput * qgrad_output, dim=dim, keepdim=True)
            grad_input = qoutput * (qgrad_output - weighted_grad_sum)
        else:
            grad_input = mp_softmax_backward(qoutput, qgrad_output, dim, formats)

        qgrad_input = formats.grad_quant(grad_input)

        return qgrad_input, None, None


def qsoftmax(
    x: torch.Tensor, dim: int | None, formats: QSoftmaxFormats
) -> torch.Tensor:
    return qsoftmax_kernel.apply(x, dim, formats)


class qgelu_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, formats, approximate):
        ctx.formats = formats
        ctx.approximate = approximate
        qinput = formats.input_quant(input)

        with torch.no_grad():
            if approximate == "tanh":
                PI = torch.tensor(math.pi, device=qinput.device)
                intermediate_output = torch.tanh(
                    torch.sqrt(2 / PI) * (qinput + 0.044715 * qinput**3)
                )
            else:
                SQRT2 = torch.tensor(1.41421356237, device=qinput.device)
                intermediate_output = torch.erf(qinput / SQRT2)

            quantized_intermediate_output = formats.inter_quant(intermediate_output)
            output = 0.5 * qinput * (1 + quantized_intermediate_output)

        qoutput = formats.output_quant(output)
        ctx.save_for_backward(qinput, quantized_intermediate_output)
        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        qinput, quantized_intermediate_output = ctx.saved_tensors
        qgrad_output = ctx.formats.grad_quant(grad_output)

        PI = torch.tensor(math.pi, device=grad_output.device)

        if ctx.approximate == "tanh":
            tanh_term = torch.sqrt(2 / PI) * (qinput + 0.044715 * qinput**3)
            dtanh_term = 1 - quantized_intermediate_output**2
            grad_input = qgrad_output * (
                0.5 * (1 + quantized_intermediate_output)
                + 0.5 * qinput * dtanh_term * (tanh_term + 0.134145 * qinput**3)
            )
        else:
            cdf = 0.5 * (1 + quantized_intermediate_output)
            pdf = torch.exp(-0.5 * qinput**2) / torch.sqrt(2.0 * PI)
            grad_input = qgrad_output * (cdf + qinput * pdf)

        qgrad_input = ctx.formats.grad_quant(grad_input)

        return qgrad_input, None, None


def qgelu(
    input: torch.Tensor,
    formats: QGELUFormats,
    approximate: Literal["tanh", "none"] = "none",
) -> torch.Tensor:
    return qgelu_kernel.apply(input, formats, approximate)
