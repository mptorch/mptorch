"""Ready-made recipes: one format everywhere, FP8 training, and MX block formats.

Each function returns a :class:`~mptorch.quant.Recipe`, which ``override``
then adjusts for the parts of a model that need something else, such as a
classifier head left in float32. None of them sets the ``matmul`` kind for
the whole model: that kind is every product of two activations, which in a
transformer means its attention's scores and weighted sums, and whatever
else the model's own code multiplies. A recipe that means to round some of
them names them, by module or by site, or gives an attention its formats with
:meth:`~mptorch.quant.Recipe.attention_parts`.

The formats a model is converted to are named in :mod:`mptorch` itself
(``E4M3``, ``MXFP8_E4M3`` and the others, from ``mptorch.number.presets``);
these are recipes built from them.
"""

from collections.abc import Sequence
from dataclasses import replace

from mptorch.number import (
    E4M3,
    E5M2,
    MXFP8_E4M3,
    MXFP8_E5M2,
    BinaryK,
    BlockFormat,
    Number,
    RoundMode,
    SaturationMode,
)

from .functional.activation import ActivationFormats
from .functional.attention import AttentionFormats
from .functional.norm import NormFormats, SoftmaxFormats
from .functional.pointwise import PointwiseFormats
from .functional.reduce import ReductionFormats
from .instrument.recipe import _KINDS, GemmFormats, Recipe
from .mac import BlockMac, FusedMac, Quant, SplitMac
from .rotation import Hadamard
from .scaled import ScaledQuant

__all__ = ["uniform", "fp8", "mx"]

# The largest values of OCP's E4M3 and E5M2, keyed by their binaryK layouts
# (K, P, bias). OCP reserves codes at the top that the binaryK layouts count as
# finite, so its largest values are below the layouts' own max_finite.
_OCP_LARGEST = {(8, 4, 7): 448.0, (8, 3, 15): 57344.0}


def uniform(
    fmt: Number,
    *,
    acc: Number | None = None,
    rounding: RoundMode = RoundMode.RNE,
    kinds: Sequence[str] = ("linear", "conv"),
) -> Recipe:
    """One format for every operand, gradient and product of the given kinds.

    Every GEMM pass multiplies in ``SplitMac(fmt, acc, rounding=rounding)``,
    and every operand and output gradient is rounded by ``Quant(fmt,
    rounding)``; the bias and the output are not rounded. The other kinds
    round every slot with that ``Quant``: an activation's gate, result and
    gradient (``ActivationFormats.uniform``), a pointwise op's or an
    embedding's result and gradients (``PointwiseFormats.uniform``), and
    each step of a softmax, a norm or a mean (``SoftmaxFormats.uniform``,
    ``NormFormats.uniform``, ``ReductionFormats.uniform``), whose sums run in
    the GEMMs' ``SplitMac``, and an attention's two products and softmax
    (``AttentionFormats.uniform``).

    Args:
        fmt (Number): the format.
        acc (Number, optional): the accumulate format. Default: ``None``,
            which leaves the running sums in the carrier.
        rounding (RoundMode): the rounding mode of every rounding.
            Default: ``RoundMode.RNE``
        kinds (Sequence[str]): the kinds the recipe sets: any of ``linear``,
            ``conv``, ``matmul``, ``activation``, ``elementwise``,
            ``embedding``, ``softmax``, ``norm``, ``reduction`` and
            ``attention``.
            Default: ``("linear", "conv")``

    Example::

        >>> from mptorch import BinaryK
        >>> from mptorch.quant import presets
        >>> recipe = presets.uniform(BinaryK(8, 4), acc=BinaryK(16, 11))
    """
    unknown = [k for k in kinds if k not in _KINDS]
    if unknown:
        raise ValueError(f"unknown kinds {unknown}: the kinds are {', '.join(_KINDS)}")
    quant = Quant(fmt, rounding)
    mac = SplitMac(fmt, acc, rounding=rounding)
    gf = GemmFormats.uniform(mac, quant)
    values = {
        "activation": ActivationFormats.uniform(quant),
        "elementwise": PointwiseFormats.uniform(quant),
        "embedding": PointwiseFormats.uniform(quant),
        "softmax": SoftmaxFormats.uniform(mac, quant),
        "norm": NormFormats.uniform(mac, quant),
        "reduction": ReductionFormats.uniform(mac, quant),
        "attention": AttentionFormats.uniform(mac, quant),
    }
    return Recipe(**{kind: values.get(kind, gf) for kind in kinds})


def _scaled_to_largest(fmt: Number) -> ScaledQuant:
    """A scaled quantizer for one of fp8's formats: OCP's layouts saturate and
    are scaled to OCP's largest values, any other format to its own."""
    if isinstance(fmt, BinaryK) and fmt.is_signed:
        largest = _OCP_LARGEST.get((fmt.K, fmt.P, fmt.bias))
        if largest is not None:
            # `domain=None` lets SAT_FINITE choose the finite domain, where
            # `replace` would otherwise keep the format's own
            saturating = replace(fmt, saturation=SaturationMode.SAT_FINITE, domain=None)
            return ScaledQuant(saturating, target=largest)
    return ScaledQuant(fmt)


def fp8(fwd: Number = E4M3, bwd: Number = E5M2, acc: Number | None = None) -> Recipe:
    """FP8 training: E4M3 operands, E5M2 gradients, each scaled per tensor.

    Every operand the forward pass reads is rounded to ``fwd`` and every
    output gradient to ``bwd``, through a :class:`~mptorch.quant.ScaledQuant`
    of the whole tensor, as Transformer Engine's current scaling does. For
    OCP's two layouts (``mptorch.E4M3`` and ``mptorch.E5M2``) the
    format saturates (``SaturationMode.SAT_FINITE``) and the scale targets
    OCP's largest values, 448 and 57344; any other format is scaled to its
    own ``max_finite`` and keeps its saturation mode. The product of two
    8-bit values (at most 4 significant bits each) is exact in the carrier, so
    every pass multiplies in ``FusedMac(acc)``: each step's one rounding is
    the sum's, to ``acc``, which is what an FP8 tensor core does. A
    ``SplitMac`` with a multiply format wide enough to hold every product
    gives the same results, more slowly. The bias and the output are not
    rounded. The recipe sets ``linear`` and ``conv``.

    Args:
        fwd (Number): the forward operands' format. Default: ``E4M3``
        bwd (Number): the output gradient's format. Default: ``E5M2``
        acc (Number, optional): the accumulate format. Default: ``None``,
            which leaves the running sums in the carrier, as an FP8 tensor
            core accumulates in float32.

    Example::

        >>> from mptorch.quant import presets
        >>> recipe = presets.fp8().override("head", linear=None)
    """
    gf = GemmFormats.uniform(
        FusedMac(acc),
        _scaled_to_largest(fwd),
        grad_quant=_scaled_to_largest(bwd),
    )
    return Recipe(linear=gf, conv=gf)


def mx(
    fwd: BlockFormat = MXFP8_E4M3,
    bwd: BlockFormat = MXFP8_E5M2,
    acc: BinaryK | None = None,
    *,
    rounding: RoundMode = RoundMode.RNE,
    rotation: Hadamard | None = None,
) -> Recipe:
    """MX training: every Linear product a block GEMM over OCP MX formats.

    The forward pass multiplies the input and the weight packed in ``fwd``,
    and the two gradient passes multiply the output gradient, packed in
    ``bwd``, by the weight and by the input, packed in ``fwd``: each operand
    is packed along the dimension its pass sums over, as
    :func:`~mptorch.quant.block_gemm_formats` describes. The recipe sets
    ``linear`` only, since a convolution has no block-format arithmetic.

    ``rotation`` rotates the input and the weight along their input features
    before they are packed, and the gradients back
    (:class:`~mptorch.quant.GemmFormats`). On GPT-2 small, over WikiText-2's
    first 32,768 tokens, ``Hadamard(128)`` took the perplexity from 27.83 to
    27.01 under MXFP8 E4M3 and from 123.2 to 39.2 under MXFP4 E2M1, against
    26.76 in float32 (``kernels/block.rst``). It does not help NVFP4.

    Args:
        fwd (BlockFormat): the format of the input and the weight.
            Default: ``MXFP8_E4M3``
        bwd (BlockFormat): the format of the output gradient.
            Default: ``MXFP8_E5M2``
        acc (BinaryK, optional): the accumulate format. Default: ``None``,
            which leaves the running sums in the carrier.
        rounding (RoundMode): the rounding mode of the sum.
            Default: ``RoundMode.RNE``
        rotation (Hadamard, optional): the rotation of the forward product's
            operands. Default: ``None``

    Example::

        >>> from mptorch.quant import Hadamard, presets
        >>> recipe = presets.mx(rotation=Hadamard(128))
    """
    return Recipe(
        linear=GemmFormats(
            fwd_mac=BlockMac(fwd, fwd, acc, rounding=rounding),
            igrad_mac=BlockMac(bwd, fwd, acc, rounding=rounding),
            wgrad_mac=BlockMac(bwd, fwd, acc, rounding=rounding),
            rotation=rotation,
        )
    )
