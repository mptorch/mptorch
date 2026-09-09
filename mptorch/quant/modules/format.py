from collections.abc import Callable

from torch import nn


class QAffineFormats(nn.Module):
    """
    Configuration module that holds quantization functions and custom math
    operations for affine layers (e.g. Linear, Conv2d).

    By inheriting from nn.Module, if any of the assigned quantizers are
    themselves nn.Modules (e.g. for QAT/PTQ state like running max/min),
    PyTorch will automatically register them and include their state in
    the model's state_dict.
    """

    def __init__(
        self,
        weight_quant: Callable | None = None,
        input_quant: Callable | None = None,
        bias_quant: Callable | None = None,
        wgrad_quant: Callable | None = None,
        igrad_quant: Callable | None = None,
        bgrad_quant: Callable | None = None,
        fwd_math: Callable | None = None,
        bwd_igrad_math: Callable | None = None,
        bwd_wgrad_math: Callable | None = None,
    ):
        super().__init__()

        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.bias_quant = bias_quant
        self.wgrad_quant = wgrad_quant
        self.igrad_quant = igrad_quant
        self.bgrad_quant = bgrad_quant

        self.fwd_math = fwd_math
        self.bwd_igrad_math = bwd_igrad_math
        self.bwd_wgrad_math = bwd_wgrad_math


class QMatmulFormats(nn.Module):
    """
    Configuration module for a quantized matmul (:class:`QMatmul`,
    :func:`mptorch.quant.qmatmul`), holding quantization functions and custom
    math operations for a symmetric two-operand op.

    The slots mirror ``QAffineFormats``'s, renamed for an op whose operands
    have equal standing: ``a``/``b`` rather than input/weight, and one
    gradient quantizer per operand rather than one shared by both -- the
    question v1 left open at four call sites and this version already answered
    for Linear (``wgrad_quant``/``igrad_quant``).

    Every slot is optional and ``None`` means "plain PyTorch": a
    ``QMatmulFormats()`` with nothing set computes ``torch.matmul`` and its
    ordinary gradients. There is deliberately no ``output_quant`` -- rounding
    the result of the op is an elementwise step, and an elementwise step is a
    quantizer the caller applies, not a slot on every op's formats.

    Inheriting from ``nn.Module`` is what registers a stateful quantizer (a
    QAT observer) in the parent model's ``state_dict``.
    """

    def __init__(
        self,
        a_quant: Callable | None = None,
        b_quant: Callable | None = None,
        agrad_quant: Callable | None = None,
        bgrad_quant: Callable | None = None,
        fwd_math: Callable | None = None,
        bwd_agrad_math: Callable | None = None,
        bwd_bgrad_math: Callable | None = None,
    ):
        super().__init__()

        self.a_quant = a_quant
        self.b_quant = b_quant
        self.agrad_quant = agrad_quant
        self.bgrad_quant = bgrad_quant

        self.fwd_math = fwd_math
        self.bwd_agrad_math = bwd_agrad_math
        self.bwd_bgrad_math = bwd_bgrad_math
