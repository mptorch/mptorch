"""The two format containers a quantized layer is configured with.

Both are ``nn.Module`` subclasses holding optional callables, and both mean
the same thing by ``None``: that step is plain PyTorch. A quantizer slot is a
``Tensor -> Tensor`` callable applied to one operand or gradient before the
math runs, and a math slot replaces one of the layer's GEMMs (or convolutions)
with custom arithmetic. Which slots exist, and what each math hook is called
with, is decided by the autograd Function that reads the container
(``CustomArithLinear``, ``CustomArithConvNd``, ``CustomArithMatmul``).
"""

from collections.abc import Callable

from torch import nn


class QAffineFormats(nn.Module):
    """Quantizers and custom math for an affine layer (Linear, Conv1d/2d/3d).

    Every slot is optional, and ``None`` means plain PyTorch behaviour for
    that step. Quantization is therefore opt-in per tensor and per pass: with
    a ``QAffineFormats()`` that has nothing set, ``QLinear`` behaves exactly
    as ``nn.Linear``.

    The forward pass quantizes the input, the weight and the bias through
    their ``*_quant`` slots, then runs ``fwd_math`` (or the plain op). The
    backward pass quantizes the incoming gradient *separately* for the
    input-gradient path (``igrad_quant``) and the weight-gradient path
    (``wgrad_quant``), because the two GEMMs may need different formats. It
    then runs ``bwd_igrad_math`` and ``bwd_wgrad_math`` (or PyTorch's
    defaults).

    The class inherits from ``nn.Module``. A quantizer that is itself an
    ``nn.Module`` (a QAT observer with running statistics) is therefore
    registered with the parent model and appears in its ``state_dict``.

    Args:
        weight_quant (Callable, optional): applied to the weight in the
            forward. Default: ``None``
        input_quant (Callable, optional): applied to the input in the
            forward. Default: ``None``
        bias_quant (Callable, optional): applied to the bias in the forward.
            Default: ``None``
        wgrad_quant (Callable, optional): applied to the output gradient
            before the weight-gradient GEMM. Default: ``None``
        igrad_quant (Callable, optional): applied to the output gradient
            before the input-gradient GEMM. Default: ``None``
        bgrad_quant (Callable, optional): applied to the output gradient
            before the bias-gradient sum. Default: ``None``
        fwd_math (Callable, optional): the forward's arithmetic. For Linear,
            ``fwd_math(q_input, q_weight, q_bias)``; for Conv, the same plus
            ``stride``, ``padding``, ``dilation``, ``groups`` and ``nd`` as
            keywords. Default: ``None``
        bwd_igrad_math (Callable, optional): the input gradient's arithmetic,
            ``bwd_igrad_math(q_grad_output, q_weight)`` for Linear (Conv adds
            ``input_size`` and the convolution keywords). Default: ``None``
        bwd_wgrad_math (Callable, optional): the weight gradient's
            arithmetic, ``bwd_wgrad_math(q_grad_output, q_input)`` for Linear
            (Conv adds ``weight_size`` and the convolution keywords).
            Default: ``None``

    Example::

        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QAffineFormats, QLinear, Quant, binaryK_gemm_formats
        >>> formats = binaryK_gemm_formats(mul_K=8, mul_P=4, accumulate_quant=False)
        >>> formats.input_quant = Quant(BinaryK(8, 4))
        >>> formats.weight_quant = Quant(BinaryK(8, 4))
        >>> layer = QLinear(256, 64, formats=formats)
        >>> plain = QLinear(256, 64, formats=QAffineFormats())   # as nn.Linear
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
    """Quantizers and custom math for a quantized matmul.

    :class:`QMatmul` and :func:`mptorch.quant.qmatmul` read this container.
    Its slots mirror those of :class:`QAffineFormats`, renamed for an op whose
    two operands play the same role: ``a``/``b`` instead of input/weight, and
    one gradient quantizer per operand, because the two gradient GEMMs may
    need different formats. Every slot is optional, and ``None`` means plain
    PyTorch. A ``QMatmulFormats()`` with nothing set computes
    ``torch.matmul`` and its ordinary gradients.

    There is no ``output_quant``, on purpose. Rounding the result of the op
    is an elementwise step, so the caller applies a quantizer for it; it is
    not a slot on every op's formats.

    The class inherits from ``nn.Module``, which registers a stateful
    quantizer (a QAT observer) in the parent model's ``state_dict``.

    Args:
        a_quant (Callable, optional): applied to ``a`` in the forward.
            Default: ``None``
        b_quant (Callable, optional): applied to ``b`` in the forward.
            Default: ``None``
        agrad_quant (Callable, optional): applied to the output gradient
            before ``a``'s gradient GEMM. Default: ``None``
        bgrad_quant (Callable, optional): applied to the output gradient
            before ``b``'s gradient GEMM. Default: ``None``
        fwd_math (Callable, optional): ``fwd_math(q_a, q_b) -> a @ b``.
            Default: ``None``
        bwd_agrad_math (Callable, optional):
            ``bwd_agrad_math(q_grad, q_b) -> grad @ b^T``. Default: ``None``
        bwd_bgrad_math (Callable, optional):
            ``bwd_bgrad_math(q_grad, q_a) -> a^T @ grad``. Default: ``None``

    Example::

        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QMatmul, Quant, SplitMac, matmul_formats
        >>> formats = matmul_formats(SplitMac(BinaryK(8, 4), BinaryK(16, 11)))
        >>> formats.a_quant = Quant(BinaryK(8, 4))
        >>> formats.b_quant = Quant(BinaryK(8, 4))
        >>> attn = QMatmul(formats)
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
