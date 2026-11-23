"""The two format containers that configure a quantized layer.

Both are ``nn.Module`` subclasses holding optional callables, and in both a
``None`` slot means that step is plain PyTorch. A quantizer slot holds a
``Tensor -> Tensor`` callable applied to one operand or gradient before the
math runs. A math slot holds a callable that replaces one of the layer's
GEMMs (or convolutions) with custom arithmetic. The autograd Function that
reads the container (``CustomArithLinear``, ``CustomArithConvNd`` or
``CustomArithMatmul``) decides what arguments each math slot is called with.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from ..mac import BlockQuant
    from ..recipe import GemmFormats


class QAffineFormats(nn.Module):
    """Quantizers and custom math for an affine layer (Linear, Conv1d/2d/3d).

    Every slot is optional, and ``None`` means plain PyTorch for that step.
    Quantization is therefore opt-in for each tensor and each pass: with a
    ``QAffineFormats()`` that has nothing set, ``QLinear`` behaves exactly as
    ``nn.Linear``.

    The forward pass quantizes the input, the weight and the bias through
    their ``*_quant`` slots, then runs ``fwd_math`` (or the plain op). The
    backward pass quantizes the incoming gradient *separately* for the
    input-gradient GEMM (``igrad_quant``) and the weight-gradient GEMM
    (``wgrad_quant``), because the two may need different formats. It then
    runs ``bwd_igrad_math`` and ``bwd_wgrad_math`` (or PyTorch's defaults).

    The class is an ``nn.Module``, so a quantizer that is itself an
    ``nn.Module`` (a QAT observer with running statistics) is registered
    with the parent model and appears in its ``state_dict``.

    Backward needs the quantized input (for the weight gradient) and the
    quantized weight (for the input gradient) again. For each, the layer can
    keep a quantized copy, a tensor the size of the operand, or keep the raw
    operand and quantize it again in backward. Recomputing saves the copy's
    memory wherever the raw operand stays alive anyway: the weight always
    does, and an input does when the op that produced it keeps its output
    (ReLU does). The two ``recompute_*`` flags choose, and ``None`` gives the
    layer's default: recompute both for ``QConv1d/2d/3d``, and only the weight
    for ``QLinear``. The gradients are bit-identical either way, since under
    stochastic rounding the second call sees the random generators in the
    state the first one did. :doc:`/layers` ("What a layer keeps for
    backward") says when each choice saves memory.

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
        fwd_math (Callable, optional): the forward pass's arithmetic. For
            Linear, ``fwd_math(q_input, q_weight, q_bias)``; for Conv, the
            same plus ``stride``, ``padding``, ``dilation``, ``groups`` and
            ``nd`` as keywords. Default: ``None``
        bwd_igrad_math (Callable, optional): the input gradient's arithmetic,
            ``bwd_igrad_math(q_grad_output, q_weight)`` for Linear (Conv adds
            ``input_size`` and the convolution keywords). Default: ``None``
        bwd_wgrad_math (Callable, optional): the weight gradient's
            arithmetic, ``bwd_wgrad_math(q_grad_output, q_input)`` for Linear
            (Conv adds ``weight_size`` and the convolution keywords).
            Default: ``None``
        recompute_input (bool, optional): keep the raw input for backward
            and run ``input_quant`` on it again there, instead of keeping its
            result. ``None`` gives the layer's default: ``True`` for a
            convolution, ``False`` for ``QLinear``. Under ``None`` only a
            quantizer known to be stateless is run twice (a
            :class:`mptorch.quant.Quant` not in place, a
            :class:`mptorch.quant.ScaledQuant`, a
            :class:`mptorch.quant.BlockQuant` called as a fake quantizer, or a
            :class:`mptorch.quant.Quantizer` over any of them), and any other
            callable keeps its result saved, since one with state (a QAT
            observer) would update it twice. ``True`` runs any callable again,
            which is the caller's promise that it depends only on its input
            and the random generators. A quantizer whose result is not a
            tensor (a packed block format) always keeps it saved.
            Default: ``None``
        recompute_weight (bool, optional): the same for the weight and
            ``weight_quant``. ``None`` means ``True`` for every layer, since
            the raw weight stays alive anyway as the layer's parameter.
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

    #: The :class:`mptorch.quant.BlockQuant` of the weight, set by
    #: :func:`mptorch.quant.block_gemm_formats` so that
    #: :meth:`mptorch.quant.QLinear.freeze_packed_` knows which format to
    #: pack the weight in. It is a class attribute, so that other formats
    #: carry nothing.
    _block_weight: "BlockQuant | None" = None

    #: The :class:`mptorch.quant.GemmFormats` that built this container, which
    #: :func:`mptorch.quant.profile` reports a layer's formats by, or ``None``
    #: for a container built otherwise. A class attribute, as above.
    _source: "GemmFormats | None" = None

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
        recompute_input: bool | None = None,
        recompute_weight: bool | None = None,
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

        self.recompute_input = recompute_input
        self.recompute_weight = recompute_weight


class QMatmulFormats(nn.Module):
    """Quantizers and custom math for a quantized matmul.

    :class:`QMatmul` and :func:`mptorch.quant.qmatmul` read this container.
    Its slots mirror those of :class:`QAffineFormats`, renamed for an op whose
    two operands play the same role: ``a`` and ``b`` instead of input and
    weight, and one gradient quantizer per operand, because the two gradient
    GEMMs may need different formats. Every slot is optional, and ``None``
    means plain PyTorch. A ``QMatmulFormats()`` with nothing set computes
    ``torch.matmul`` and its ordinary gradients.

    There is deliberately no ``output_quant``. Rounding the op's result is an
    elementwise step, so the caller applies a quantizer to it; it is not a
    slot in every op's formats.

    The class is an ``nn.Module``, so a stateful quantizer (a QAT observer)
    is registered in the parent model's ``state_dict``.

    The ``recompute_*`` flags work as in :class:`QAffineFormats`: keep ``a``
    (or ``b``) raw for backward and quantize it again there, instead of
    keeping a quantized copy. Both default to keeping the copy, since a
    matmul's operands are activations, and whether anything else keeps one
    alive depends on the model.

    Args:
        a_quant (Callable, optional): applied to ``a`` in the forward.
            Default: ``None``
        b_quant (Callable, optional): applied to ``b`` in the forward.
            Default: ``None``
        agrad_quant (Callable, optional): applied to the output gradient
            before ``a``'s gradient GEMM. Default: ``None``
        bgrad_quant (Callable, optional): applied to the output gradient
            before ``b``'s gradient GEMM. Default: ``None``
        fwd_math (Callable, optional): the forward pass's arithmetic,
            ``fwd_math(q_a, q_b)``, computing ``a @ b``. Default: ``None``
        bwd_agrad_math (Callable, optional): ``a``'s gradient,
            ``bwd_agrad_math(q_grad, q_b)``, computing ``grad @ b^T``.
            Default: ``None``
        bwd_bgrad_math (Callable, optional): ``b``'s gradient,
            ``bwd_bgrad_math(q_grad, q_a)``, computing ``a^T @ grad``.
            Default: ``None``
        recompute_a (bool, optional): keep ``a`` raw for backward and run
            ``a_quant`` on it again there. ``None`` keeps the copy. Set it
            where something else keeps ``a`` alive, as attention's softmax
            keeps the probabilities that multiply ``V``. ``True`` runs any
            callable again, as for :class:`QAffineFormats`. Default: ``None``
        recompute_b (bool, optional): the same for ``b`` and ``b_quant``.
            Default: ``None``

    Example::

        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QMatmul, Quant, SplitMac, matmul_formats
        >>> formats = matmul_formats(SplitMac(BinaryK(8, 4), BinaryK(16, 11)))
        >>> formats.a_quant = Quant(BinaryK(8, 4))
        >>> formats.b_quant = Quant(BinaryK(8, 4))
        >>> attn = QMatmul(formats)
    """

    #: The :class:`mptorch.quant.GemmFormats` that built this container, as
    #: for :class:`QAffineFormats`.
    _source: "GemmFormats | None" = None

    def __init__(
        self,
        a_quant: Callable | None = None,
        b_quant: Callable | None = None,
        agrad_quant: Callable | None = None,
        bgrad_quant: Callable | None = None,
        fwd_math: Callable | None = None,
        bwd_agrad_math: Callable | None = None,
        bwd_bgrad_math: Callable | None = None,
        recompute_a: bool | None = None,
        recompute_b: bool | None = None,
    ):
        super().__init__()

        self.a_quant = a_quant
        self.b_quant = b_quant
        self.agrad_quant = agrad_quant
        self.bgrad_quant = bgrad_quant

        self.fwd_math = fwd_math
        self.bwd_agrad_math = bwd_agrad_math
        self.bwd_bgrad_math = bwd_bgrad_math

        self.recompute_a = recompute_a
        self.recompute_b = recompute_b
