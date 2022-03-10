import torch
from mptorch import Number, FixedPoint, FloatingPoint, BlockFloatingPoint
from .quant_function import *

__all__ = ["QAffineFormats"]

id_quant = lambda x: x


class QAffineFormats:
    def __init__(
        self,
        fwd_add,
        fwd_mul,
        fwd_rnd,
        bwd_add,
        bwd_mul,
        bwd_rnd,
        param_quant=id_quant,
        input_quant=id_quant,
        grad_quant=id_quant,
        fwd_fma=False,
        bwd_fma=False,
    ):
        self.fwd_add = fwd_add
        self.fwd_mul = fwd_mul
        self.bwd_add = bwd_add
        self.bwd_mul = bwd_mul
        self.fwd_rnd = fwd_rnd
        self.bwd_rnd = bwd_rnd
        self.fwd_fma = fwd_fma
        self.bwd_fma = bwd_fma
        self.param_quant = param_quant
        self.input_quant = input_quant
        self.grad_quant = grad_quant
