from typing import Optional
from ..number import Number, FloatingPoint
from typing import Union, Optional, Tuple, Callable
from .quant_function import float_quantize

__all__ = ["QAffineFormats"]

id_quant = lambda x: x

def make_quant_function(num: Number, rounding: str):
    if isinstance(num, FloatingPoint):
        return lambda x: float_quantize(
            x, num.exp, num.man, rounding, num.subnormals, num.saturate
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
        - :attr: `weight_scaled_format` (FloatingPoint) : number format to be used during weight tensor scaling (optional, matches weight_quant if format specified)
        - :attr: `input_scaled_format` (FloatingPoint) : number format to be used during input tensor scaling (optional, matches input_quant if format specified)
        - :attr: `grad_scaled_format` (FloatingPoint) : number format to be used during output tensor scaling (optional, matches grad_quant if format specified)
 
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
        weight_scaled_format: Optional[FloatingPoint] = None,
        input_scaled_format: Optional[FloatingPoint] = None,
        grad_scaled_format: Optional[FloatingPoint] = None,
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

        self.weight_scaled_format = weight_scaled_format
        self.input_scaled_format = input_scaled_format
        self.grad_scaled_format = grad_scaled_format

        if isinstance(weight_quant, tuple):
            num, rnd = weight_quant
            self.weight_quant = make_quant_function(num, rnd)
            if self.weight_scaled_format is None and isinstance(num, FloatingPoint):
                self.weight_scaled_format = num
        else:
            self.weight_quant = weight_quant
        
        if isinstance(bias_quant, tuple):
            num, rnd = bias_quant
            self.bias_quant = make_quant_function(num, rnd)
        else:
            self.bias_quant = bias_quant
        
        if isinstance(input_quant, tuple):
            num, rnd = input_quant
            self.input_quant = make_quant_function(num, rnd)
            if self.input_scaled_format is None and isinstance(num, FloatingPoint):
                self.input_scaled_format = num
        else:
            self.input_quant = input_quant

        if isinstance(output_quant, tuple):
            num, rnd = output_quant
            self.output_quant = make_quant_function(num, rnd)
        else:
            self.output_quant = output_quant
        
        if isinstance(grad_quant, tuple):
            num, rnd = grad_quant
            self.grad_quant = make_quant_function(num, rnd)
            if self.grad_scaled_format is None and isinstance(num, FloatingPoint):
                self.grad_scaled_format = num
        else:
            self.grad_quant = grad_quant
    
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
