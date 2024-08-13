from typing import Optional
from ..number import *
from typing import Union, Optional, Tuple, Callable
from .quant_function import *

__all__ = ["QAffineFormats", "QSoftmaxFormats", "QGELUFormats"]

id_quant = lambda x: x

def make_quant_function(num: Number, rounding: str, prng_bits: int = 0):
    if isinstance(num, FloatingPoint):
        return lambda x: float_quantize(
            x, num.exp, num.man, rounding, num.subnormals, num.saturate
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
        return lambda x: block_quantize(
            x, num.wl, num.dim, rounding
        )
    elif isinstance(num, Binary8):
        return lambda x: binary8_quantize(
            x, num.P, rounding, num.overflow_policy, num.signed, num.subnormals, prng_bits
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
        - :attr: `fwd_mac` (Number or (Number, Number)) : compute configuration (add and multiply) for forward MAC operations
        - :attr: `bwd_mac` (Number or (Number, Number)) : compute configuration (add and multiply) for backward MAC operations
        - :attr: `fwd_rnd` (str) : rounding mode for FWD computations
        - :attr: `bwd_rnd` (str) : rounding mode for BWD computations
        - :attr: `weight_quant` (function or (Number, str)) : quantization function or format and rounding on the weight signal inputs
        - :attr: `bias_quant` (function or (Number, str)) : quantization function or format and rounding on the bias signal inputs
        - :attr: `input_quant` (function or (Number, str)) : quantization function or format and rounding on the output signal from the layer
        - :attr: `grad_quant` (function or (Number, str)) : quantization function or format and rounding on the gradient signals in the BWD pass
        - :attr: `use_scaling` (bool) : whether to use weight, input and grad scaling during forward/backward pass
        - :attr: `weight_scaled_format` (FloatType) : number format to be used during weight tensor scaling (optional, matches weight_quant if format specified)
        - :attr: `input_scaled_format` (FloatType) : number format to be used during input tensor scaling (optional, matches input_quant if format specified)
        - :attr: `grad_scaled_format` (FloatType) : number format to be used during output tensor scaling (optional, matches grad_quant if format specified)
        - :attr: `prng_bits` (int) : number of bits used for random number generation when rounding is stochastic
 
    """
    def __init__(
        self,
        fwd_mac: Optional[Union[Number, Tuple[Number, Number]]] = None,
        bwd_mac: Optional[Union[Number, Tuple[Number, Number]]] = None,
        fwd_rnd: Optional[str] = "nearest",
        bwd_rnd: Optional[str] = "nearest",
        weight_quant: Union[Callable, Tuple[Number, str]] = id_quant,
        bias_quant: Union[Callable, Tuple[Number, str]] = id_quant,
        input_quant: Union[Callable, Tuple[Number, str]] = id_quant,
        output_quant: Union[Callable, Tuple[Number, str]] = id_quant,
        grad_quant: Union[Callable, Tuple[Number, str]] = id_quant,
        use_scaling: bool = False,
        weight_scaled_format: Optional[FloatType] = None,
        input_scaled_format: Optional[FloatType] = None,
        grad_scaled_format: Optional[FloatType] = None,
        prng_bits: int = 0
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
        self.prng_bits = prng_bits

        self.weight_scaled_format = weight_scaled_format
        self.input_scaled_format = input_scaled_format
        self.grad_scaled_format = grad_scaled_format

        if isinstance(weight_quant, tuple):
            num, rnd = weight_quant
            self.weight_quant = make_quant_function(num, rnd, prng_bits)
            if self.weight_scaled_format is None and isinstance(num, FloatType):
                self.weight_scaled_format = num
        else:
            self.weight_quant = weight_quant
        
        if isinstance(bias_quant, tuple):
            num, rnd = bias_quant
            self.bias_quant = make_quant_function(num, rnd, prng_bits)
        else:
            self.bias_quant = bias_quant
        
        if isinstance(input_quant, tuple):
            num, rnd = input_quant
            self.input_quant = make_quant_function(num, rnd, prng_bits)
            if self.input_scaled_format is None and isinstance(num, FloatType):
                self.input_scaled_format = num
        else:
            self.input_quant = input_quant

        if isinstance(output_quant, tuple):
            num, rnd = output_quant
            self.output_quant = make_quant_function(num, rnd, prng_bits)
        else:
            self.output_quant = output_quant
        
        if isinstance(grad_quant, tuple):
            num, rnd = grad_quant
            self.grad_quant = make_quant_function(num, rnd, prng_bits)
            if self.grad_scaled_format is None and isinstance(num, FloatType):
                self.grad_scaled_format = num
        else:
            self.grad_quant = grad_quant
        
        self.use_scaling = use_scaling
    
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
        sep = ", "
        return f"QAffineFormats ({sep.join(out)})"
    
    def __str__(self) -> str:
        return self.__repr__()

class QSoftmaxFormats:
    def __init__(
        self,
        fwd_off: Optional[Number] = None,
        fwd_exp: Optional[Number] = None,
        fwd_acc: Optional[Number] = None,
        fwd_lse: Optional[Number] = None,

        bwd_add: Optional[Number] = None,
        bwd_mul: Optional[Number] = None,
        
        fwd_rnd: Optional[str] = "nearest",
        bwd_rnd: Optional[str] = "nearest",
        
        input_quant=id_quant,
        output_quant=id_quant,
        grad_quant=id_quant,
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
                raise ValueError("Incomplete softmax format, "
                                 "missing fwd_acc/fwd_exp or fwd_lse.")
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
    def __init__(
        self,
        input_quant=id_quant,
        inter_quant=id_quant,
        output_quant=id_quant,
        grad_quant=id_quant,
    ):
        self.input_quant = input_quant 
        self.inter_quant = inter_quant
        self.output_quant = output_quant
        self.grad_quant = grad_quant