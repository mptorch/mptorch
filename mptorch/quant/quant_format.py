from typing import Optional

import torch
from ..number import *
from typing import Union, Optional, Tuple, Callable
from .quant_function import (
    float_quantize,
    fixed_point_quantize,
    superfp_quantize,
    block_quantize,
    binary8_quantize,
)

__all__ = ["QAffineFormats", "QSoftmaxFormats", "QLayerNormFormats", "QGELUFormats"]

id_quant = lambda x: x


def make_quant_function(num: Number, rounding: str, prng_bits: int = 0) -> Callable:
    if isinstance(num, FloatingPoint):
        return lambda x: float_quantize(
            x, num.exp, num.man, rounding, num.subnormals, num.saturate, prng_bits
        )
    elif isinstance(num, FixedPoint):
        return lambda x: fixed_point_quantize(
            x, num.wl, num.fl, num.clamp, num.symmetric, rounding
        )
    elif isinstance(num, SuperNormalFloat):
        return lambda x: superfp_quantize(
            x, num.exp, num.man, num.binades, rounding, num.saturate
        )
    elif isinstance(num, BlockFloatingPoint):
        return lambda x: block_quantize(x, num.wl, num.dim, rounding)
    elif isinstance(num, Binary8):
        return lambda x: binary8_quantize(
            x,
            num.P,
            rounding,
            num.overflow_policy,
            num.signed,
            num.subnormals,
            prng_bits,
        )
    raise NotImplementedError


class QAffineFormats:
    r"""
    Configuration class for number formats to use during compute (forward
    and/or backward pass) of affine layers (e.g. linear and convolutional).
    One can optionally specify quantizer objects for the signals in the
    layer (I/O activations, weights/bias terms and weight/error gradients)
    to facilitate quantization-aware-training (QAT) and post-training
    quantization (PTQ) workloads. Format parameters can also be specified
    for tensor scaling operations, in a similar way to what is described
    in: https://arxiv.org/pdf/2309.17224

    Args:
        fwd_mac: compute configuration (add and multiply) for forward MAC operations
        bwd_mac: compute configuration (add and multiply) for backward MAC operations
        fwd_rnd: rounding mode for FWD computations
        bwd_rnd: rounding mode for BWD computations
        weight_quant: quantization function or format and rounding on the weight signal inputs
        bias_quant: quantization function or format and rounding on the bias signal inputs
        input_quant: quantization function or format and rounding on the output signal from the layer
        grad_quant: quantization function or format and rounding on the gradient signals in the BWD pass
        use_scaling: whether to use weight, input and grad scaling during forward/backward pass
        weight_scaled_format: number format to be used during weight tensor scaling (optional, matches weight_quant if format specified)
        input_scaled_format: number format to be used during input tensor scaling (optional, matches input_quant if format specified)
        grad_scaled_format: number format to be used during output tensor scaling (optional, matches grad_quant if format specified)
        rbits: number of bits used for random number generation when rounding is stochastic (for add and multiply)

    """

    def __init__(
        self,
        fwd_mac: Number | tuple[Number] | tuple[Number, Number] | None = None,
        bwd_mac: Number | tuple[Number] | tuple[Number, Number] | None = None,
        fwd_rnd: str | None = None,
        bwd_rnd: str | None = None,
        weight_quant: (
            Callable[[torch.Tensor], torch.Tensor] | tuple[Number, str]
        ) = id_quant,
        bias_quant: (
            Callable[[torch.Tensor], torch.Tensor] | tuple[Number, str]
        ) = id_quant,
        input_quant: (
            Callable[[torch.Tensor], torch.Tensor] | tuple[Number, str]
        ) = id_quant,
        output_quant: (
            Callable[[torch.Tensor], torch.Tensor] | tuple[Number, str]
        ) = id_quant,
        grad_quant: (
            Callable[[torch.Tensor], torch.Tensor] | Tuple[Number, str]
        ) = id_quant,
        compensated: bool | None = None,
        use_scaling: bool = False,
        weight_scaled_format: FloatType | None = None,
        input_scaled_format: FloatType | None = None,
        grad_scaled_format: FloatType | None = None,
        rbits: int | tuple[int] | tuple[int, int] = 0,
        s=None,
    ) -> None:
        if fwd_mac is not None:
            if not isinstance(fwd_mac, tuple):
                fwd_mac = (fwd_mac,)
            if len(fwd_mac) > 1:
                (self.fwd_add, self.fwd_mul) = fwd_mac
                self.fwd_fma = False
            elif len(fwd_mac) == 1:
                self.fwd_add = self.fwd_mul = fwd_mac[0]
                self.fwd_fma = True
            else:
                self.fwd_add = self.fwd_mul = None
                self.fwd_fma = False
            self.fwd_use_default_prec = False
        else:
            self.fwd_use_default_prec = True

        if bwd_mac is not None:
            if not isinstance(bwd_mac, tuple):
                bwd_mac = (bwd_mac,)
            if len(bwd_mac) > 1:
                (self.bwd_add, self.bwd_mul) = bwd_mac
                self.bwd_fma = False
            elif len(bwd_mac) == 1:
                self.bwd_add = self.bwd_mul = bwd_mac[0]
                self.bwd_fma = True
            else:
                self.bwd_add = self.bwd_mul = None
                self.bwd_fma = False
            self.bwd_use_default_prec = False
        else:
            self.bwd_use_default_prec = True

        self.fwd_rnd = fwd_rnd
        self.bwd_rnd = bwd_rnd
        if not isinstance(rbits, tuple):
            rbits = (rbits,)
            if len(rbits) > 1:
                (self.rbits_add, self.rbits_mul) = rbits
            else:
                self.rbits_add = self.rbits_mul = rbits[0]
        self.compensated = compensated

        self.weight_scaled_format = weight_scaled_format
        self.input_scaled_format = input_scaled_format
        self.grad_scaled_format = grad_scaled_format

        if isinstance(weight_quant, tuple):
            num, rnd = weight_quant
            self.weight_quant = make_quant_function(num, rnd, self.rbits_add)
            if self.weight_scaled_format is None and isinstance(num, FloatType):
                self.weight_scaled_format = num
        else:
            self.weight_quant = weight_quant

        if isinstance(bias_quant, tuple):
            num, rnd = bias_quant
            self.bias_quant = make_quant_function(num, rnd, self.rbits_add)
        else:
            self.bias_quant = bias_quant

        if isinstance(input_quant, tuple):
            num, rnd = input_quant
            self.input_quant = make_quant_function(num, rnd, self.rbits_add)
            if self.input_scaled_format is None and isinstance(num, FloatType):
                self.input_scaled_format = num
        else:
            self.input_quant = input_quant

        if isinstance(output_quant, tuple):
            num, rnd = output_quant
            self.output_quant = make_quant_function(num, rnd, self.rbits_add)
        else:
            self.output_quant = output_quant

        if isinstance(grad_quant, tuple):
            num, rnd = grad_quant
            self.grad_quant = make_quant_function(num, rnd, self.rbits_add)
            if self.grad_scaled_format is None and isinstance(num, FloatType):
                self.grad_scaled_format = num
        else:
            self.grad_quant = grad_quant

        self.use_scaling = use_scaling
        self.s = s

    def __repr__(self) -> str:
        out = []
        if self.fwd_use_default_prec:
            out.append("default_fwd")
        elif self.fwd_fma:
            out.append(f"fwd_fma={self.fwd_add}")
            out.append(f"fwd_rnd={self.fwd_rnd}")
        else:
            out.append(f"fwd_mul={self.fwd_mul}")
            out.append(f"fwd_add={self.fwd_add}")
            out.append(f"fwd_rnd={self.fwd_rnd}")
        if self.bwd_use_default_prec:
            out.append("default_bwd")
        elif self.bwd_fma:
            out.append(f"bwd_fma={self.bwd_add}")
            out.append(f"bwd_rnd={self.bwd_rnd}")
        else:
            out.append(f"bwd_mul={self.bwd_mul}")
            out.append(f"bwd_add={self.bwd_add}")
            out.append(f"bwd_rnd={self.bwd_rnd}")
        if self.weight_scaled_format is not None:
            out.append(f"weight_scaled_format={self.weight_scaled_format}")
        if self.input_scaled_format is not None:
            out.append(f"input_scaled_format={self.input_scaled_format}")
        if self.grad_scaled_format is not None:
            out.append(f"grad_scaled_format={self.grad_scaled_format}")
        out.append(f"rbits_add={self.rbits_add}")
        out.append(f"rbits_mul={self.rbits_mul}")
        sep = ", "
        return f"QAffineFormats ({sep.join(out)})"

    def __str__(self) -> str:
        return self.__repr__()


class QLayerNormFormats:
    r"""
    Configuration class for number formats to use during compute (forward
    and/or backward pass) of layer normalization.
    One can optionally specify quantizer objects for the signals in the
    layer (I/O activations, weights/bias terms and weight/error gradients)
    to facilitate quantization-aware-training (QAT) and post-training
    quantization (PTQ) workloads.

    Args:
        fwd_acc: compute configuration for forward add operations
        fwd_mul: compute configuration for forward multiply operations
        fwd_div: compute configuration for forward divide operations
        fwd_sqrt: compute configuration for forward square root operations
        bwd_acc: compute configuration for backward add operations
        bwd_mul: compute configuration for backward multiply operations
        bwd_div: compute configuration for backward divide operations,
        fwd_rnd: rounding mode for forward computations
        bwd_rnd: rounding mode for backward computations
        input_quant: quantization function on the input signal
        output_quant: quantization function on the output signal
        grad_quant: quantization function on the gradients
        weight_quant: quantization function on the weights when applied to an input
        bias_quant: quantization function on the bias when applied to an input
    """

    def __init__(
        self,
        fwd_acc: Number | None = None,
        fwd_mul: Number | None = None,
        fwd_div: Number | None = None,
        fwd_sqrt: Number | None = None,
        bwd_acc: Number | None = None,
        bwd_mul: Number | None = None,
        bwd_div: Number | None = None,
        fwd_rnd: str | None = "nearest",
        bwd_rnd: str | None = "nearest",
        input_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
        output_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
        grad_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
        weight_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
        bias_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
    ) -> None:
        if (
            fwd_acc is not None
            and fwd_mul is not None
            and fwd_div is not None
            and fwd_sqrt is not None
        ):
            self.fwd_acc = fwd_acc
            self.fwd_mul = fwd_mul
            self.fwd_div = fwd_div
            self.fwd_sqrt = fwd_sqrt
            self.fwd_use_default_prec = False
        else:
            self.fwd_use_default_prec = True

        if bwd_acc is not None and bwd_mul is not None and bwd_div is not None:
            self.bwd_acc = bwd_acc
            self.bwd_mul = bwd_mul
            self.bwd_div = bwd_div
            self.bwd_use_default_prec = False
        else:
            self.bwd_use_default_prec = True

        self.fwd_rnd = fwd_rnd
        self.bwd_rnd = bwd_rnd
        self.input_quant = input_quant
        self.output_quant = output_quant
        self.grad_quant = grad_quant
        self.weight_quant = weight_quant
        self.bias_quant = bias_quant

    def __repr__(self) -> str:
        out = []
        if self.fwd_use_default_prec:
            out.append("default_fwd")
        else:
            out.append(f"fwd_acc={self.fwd_acc}")
            out.append(f"fwd_mul={self.fwd_mul}")
            out.append(f"fwd_div={self.fwd_div}")
            out.append(f"fwd_sqrt={self.fwd_sqrt}")
            out.append(f"fwd_rnd={self.fwd_rnd}")

        if self.bwd_use_default_prec:
            out.append("default_bwd")
        else:
            out.append(f"bwd_acc={self.bwd_acc}")
            out.append(f"bwd_mul={self.bwd_mul}")
            out.append(f"bwd_div={self.bwd_div}")
            out.append(f"bwd_rnd={self.bwd_rnd}")

        sep = " , "
        return f"QLayerNormFormats ({sep.join(out)})"

    def __str__(self) -> str:
        return self.__repr__()


class QSoftmaxFormats:
    r"""
    Configuration class for number formats to use during compute (forward
    and/or backward pass) of softmax layers.
    One can optionally specify quantizer objects for the signals in the
    layer (I/O activations and weight/error gradients)
    to facilitate quantization-aware-training (QAT) and post-training
    quantization (PTQ) workloads.

    Two implementations of the forward softmax are provided: regular and
    LogSumExp-based (LSE).

    Regular softmax is used when `fwd_off`, `fwd_exp` and `fwd_acc` are
    set, and is implemented as follows:

    .. math::
        \textrm{softmax}(x)_i = \frac{\exp(x_i - \max x)}{
            \sum_j \exp(x_j - \max x)}

    The LogSumExp implementation is used when `fwd_off` and `fwd_exp` are
    set, and is implemented as follows:

    .. math::
        \textrm{softmax}(x)_i = \ln(\textrm{LSE}(x_1, ..., x_n))

    where :math:`\textrm{LSE}(x_1, ..., x_n)` is computed iteratively
    with the relation:

    .. math::
        \textrm{LSE}(x_1, ..., x_{j+1})
            = \ln(\exp \textrm{LSE}(x_1, ..., x_{j}) + \exp x_{j+1})

    with the internal part of the log being computed at full precision.

    Args:
        fwd_off: compute configuration for forward subtraction
        fwd_exp: compute configuration for forward exponential operations
        fwd_acc: compute configuration for forward add operations
        fwd_lse: compute configuration for forward LSE iteration
        bwd_add: compute configuration for backward add operations
        bwd_mul: compute configuration for backward multiply operations
        fwd_rnd: rounding mode for forward computations
        bwd_rnd: rounding mode for backward computations
        input_quant: quantization function on the input signal
        output_quant: quantization function on the output signal
        grad_quant: quantization function on the gradients
    """

    def __init__(
        self,
        fwd_off: Number | None = None,
        fwd_exp: Number | None = None,
        fwd_acc: Number | None = None,
        fwd_lse: Number | None = None,
        bwd_add: Number | None = None,
        bwd_mul: Number | None = None,
        fwd_rnd: str | None = "nearest",
        bwd_rnd: str | None = "nearest",
        input_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
        output_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
        grad_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
    ) -> None:
        if fwd_off is not None:
            self.fwd_off = fwd_off
            if fwd_acc is not None and fwd_exp is not None:
                self.fwd_acc = fwd_acc
                self.fwd_exp = fwd_exp
                self.use_lse = False
            elif fwd_lse is not None:
                self.fwd_lse = fwd_lse
                self.use_lse = True
            else:
                raise ValueError(
                    "Incomplete softmax format, " "missing fwd_acc/fwd_exp or fwd_lse."
                )
            self.fwd_use_default_prec = False
        else:
            self.fwd_use_default_prec = True

        if bwd_add is not None and bwd_mul is not None:
            self.bwd_add = bwd_add
            self.bwd_mul = bwd_mul
            self.bwd_use_default_prec = False
        else:
            self.bwd_use_default_prec = True

        self.fwd_rnd = fwd_rnd
        self.bwd_rnd = bwd_rnd
        self.input_quant = input_quant
        self.output_quant = output_quant
        self.grad_quant = grad_quant

    def __repr__(self) -> str:
        out = []
        if self.fwd_use_default_prec:
            out.append("default_fwd")
        elif self.use_lse:
            out.append(f"fwd_off={self.fwd_off}")
            out.append(f"fwd_lse={self.fwd_lse}")
            out.append(f"fwd_rnd={self.fwd_rnd}")
        else:
            out.append(f"fwd_off={self.fwd_off}")
            out.append(f"fwd_exp={self.fwd_exp}")
            out.append(f"fwd_acc={self.fwd_acc}")
            out.append(f"fwd_rnd={self.fwd_rnd}")
        if self.bwd_use_default_prec:
            out.append("default_bwd")
        else:
            out.append(f"bwd_add={self.bwd_add}")
            out.append(f"bwd_mul={self.bwd_mul}")
            out.append(f"bwd_rnd={self.bwd_rnd}")
        sep = ", "
        return f"QSoftmaxFormats ({sep.join(out)})"

    def __str__(self) -> str:
        return self.__repr__()


class QGELUFormats:
    r"""
    Configuration class for number formats to use during compute (forward
    and/or backward pass) of GELU activation.

    Args:
        input_quant: quantization function on the input signal
        inter_quant: quantization function on intermediate
            computation, depends on wether *tanh* approximation is used
        output_quant: quantization function on the output signal
        grad_quant: quantization function on the gradients
    """

    def __init__(
        self,
        input_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
        inter_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
        output_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
        grad_quant: Callable[[torch.Tensor], torch.Tensor] = id_quant,
    ):
        self.input_quant = input_quant
        self.inter_quant = inter_quant
        self.output_quant = output_quant
        self.grad_quant = grad_quant
