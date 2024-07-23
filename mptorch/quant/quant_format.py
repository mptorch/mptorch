from typing import Optional
from ..number import Number
from typing import Union, Optional, Tuple

__all__ = ["QAffineFormats", "QLayerNormFormats"]

id_quant = lambda x: x


class QAffineFormats:
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
        self.weight_quant = weight_quant
        self.bias_quant = bias_quant
        self.input_quant = input_quant
        self.output_quant = output_quant
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
        sep = ", "
        return f"QAffineFormats ({sep.join(out)})"
    
    def __str__(self) -> str:
        return self.__repr__()

class QLayerNormFormats:
    def __init__(
        self,
        fwd_acc: Optional[Number] = None,
        fwd_mul: Optional[Number] = None,
        fwd_div: Optional[Number] = None,
        fwd_sqrt: Optional[Number] = None,

        bwd_acc: Optional[Number] = None,
        bwd_mul: Optional[Number] = None,
        bwd_div: Optional[Number] = None,
        
        fwd_rnd: Optional[str] = "nearest",
        bwd_rnd: Optional[str] = "nearest",

        input_quant = id_quant,
        output_quant = id_quant,
        grad_quant = id_quant,
        weight_quant = id_quant,
        bias_quant = id_quant,
    )-> None:
        if fwd_acc is not None and fwd_mul is not None and fwd_div is not None and fwd_sqrt is not None:
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