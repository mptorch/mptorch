from typing import Optional
from ..number import Number
from typing import Union, Optional, Tuple

class QGeLUFormats:
    def __init__(self, input_quant, output_quant, grad_quant):
        self.input_quant = input_quant
        self.output_quant = output_quant
        self.grad_quant = grad_quant

