from typing import Optional
from ..number import Number, FloatingPoint
from typing import Union, Optional, Tuple

__all__ = ["QAffineFormats"]

id_quant = lambda x: x


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
        - :attr: `fwd_mac` (Number or (Number, Number)) : compute configuration (add and multiply) for FWD MAC operations
        - :attr: `bwd_mac` (Number or (Number, Number)) : compute configuration (add and multiply) for BWD MAC operations
        - :attr: `fwd_rnd` (str) : rounding mode for FWD computations
        - :attr: `bwd_rnd` (str) : rounding mode for BWD computations
        - :attr: `weight_quant` (function) : differentiable quantization function on the weight signal inputs
        - :attr: `bias_quant` (function) : differentiable quantization function on the bias signal inputs
        - :attr: `input_quant` (function) : differentiable quantization function on the output signal from the layer
        - :attr: `grad_quant` (function) : differentiable quantization function on the gradient signals in the BWD pass
        - :attr: `weight_scaled_format` (FloatingPoint) : number format to be used during weight tensor scaling (optional) 
        - :attr: `input_scaled_format` (FloatingPoint) : number format to be used during input tensor scaling (optional)  
        - :attr: `grad_scaled_format` (FloatingPoint) : number format to be used during output tensor scaling (optional)  
 
    """
    def __init__(
        self,
        fwd_mac: Optional[Union[Number, Tuple[Number, Number]]] = None,
        bwd_mac: Optional[Union[Number, Tuple[Number, Number]]] = None,
        fwd_rnd: Optional[str] = "nearest",
        bwd_rnd: Optional[str] = "nearest",
        weight_quant=id_quant,
        bias_quant=id_quant,
        input_quant=id_quant,
        output_quant=id_quant,
        grad_quant=id_quant,
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
                (self.bwd_add, self.bwd_mul) = fwd_mac
                self.bwd_fma = False
            elif len(fwd_mac) == 1:
                self.bwd_add = self.bwd_mul = fwd_mac[0]
                self.bwd_fma = True
            else:
                self.bwd_add = self.bwd_mul = None
                self.bwd_fma = False
            self.bwd_use_default_prec = False
        else:
            self.bwd_use_default_prec = True

        self.fwd_rnd = fwd_rnd
        self.bwd_rnd = bwd_rnd
        self.weight_quant = weight_quant
        self.bias_quant = bias_quant
        self.input_quant = input_quant
        self.output_quant = output_quant
        self.grad_quant = grad_quant
        self.weight_scaled_format = weight_scaled_format
        self.input_scaled_format = input_scaled_format
        self.grad_scaled_format = grad_scaled_format
    
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
        sep = ", "
        return f"QAffineFormats ({sep.join(out)})"
    
    def __str__(self) -> str:
        return self.__repr__()
