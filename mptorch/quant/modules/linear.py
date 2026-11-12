"""The quantized Linear layer: ``CustomArithLinear`` and the ``QLinear`` module.

``QLinear`` is a thin ``nn.Linear`` subclass whose ``forward`` delegates to a
``torch.autograd.Function``. The Function is where quantization happens, once
per operand and once per gradient path, so that the layer's ``formats`` can
round the input, the weight and the bias before the forward GEMM, round the
incoming gradient differently for the input-gradient and weight-gradient GEMMs,
and replace each of the three GEMMs with custom arithmetic. Any slot left
``None`` is plain PyTorch, so a ``QLinear`` with empty formats is exactly an
``nn.Linear``.

``QPackedLinear`` is a ``QLinear`` over block formats whose weight is stored
only as its block codes: ``freeze_packed_`` switches a layer's class
to it in place, and ``unfreeze_packed_`` back.
"""

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from mptorch.number import _block_label

from ..block import BlockPacked, _block_format_ints, block_unpack
from ..mac import BlockQuant
from ..ops import _CARRIER, _call_carrier
from ._saved import load, quantize_operand, restore, save
from .format import QAffineFormats


class CustomArithLinear(torch.autograd.Function):
    """Autograd Function for a quantized linear layer.

    ``forward(input, weight, bias, formats)`` quantizes the three operands
    through ``formats``' ``*_quant`` slots and computes the output with
    ``formats.fwd_math`` or ``F.linear``. ``backward`` quantizes the output
    gradient separately for the input-gradient path (``igrad_quant``) and the
    weight-gradient path (``wgrad_quant``), then runs ``bwd_igrad_math`` and
    ``bwd_wgrad_math`` or their plain-matmul defaults, and sums the
    ``bgrad_quant``-quantized gradient over the leading dimensions for the
    bias. ``formats`` gets no gradient.
    """

    @staticmethod
    def _default_bwd_igrad(grad_output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """The input gradient without a custom hook: ``grad_output @ weight``,
        with ``grad_output``'s leading dimensions kept."""
        return grad_output.matmul(weight)

    @staticmethod
    def _default_bwd_wgrad(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """The weight gradient without a custom hook: ``grad_output^T @ input``
        over every leading dimension flattened into the row dimension."""
        gO_flat = grad_output.reshape(-1, grad_output.shape[-1])
        i_flat = input.reshape(-1, input.shape[-1])
        return gO_flat.t().matmul(i_flat)

    #: What a ``None`` recompute flag means here, as (input, weight): the
    #: weight is quantized again in backward rather than kept, since the raw
    #: weight is alive anyway as the layer's parameter; the input is kept,
    #: since a Linear's input usually comes from an op that keeps its own
    #: input rather than its output (LayerNorm, GELU, SiLU), so the raw input
    #: would be alive only for this layer (``_saved``: a raw operand is kept
    #: only where it is alive anyway).
    _RECOMPUTE_DEFAULT = (False, True)

    @staticmethod
    def forward(ctx, input, weight, bias, formats):
        ctx.formats = formats
        recompute_input, recompute_weight = CustomArithLinear._RECOMPUTE_DEFAULT

        # Save only what backward will read. q_bias is never read there, and
        # each of the other two feeds exactly one gradient path: q_input the
        # weight gradient, q_weight the input gradient. So a layer whose
        # weight is frozen does not pin a quantized copy of its entire
        # activation for the length of the backward pass, on top of the
        # activation autograd already holds. ctx.needs_input_grad is
        # populated before forward runs and is the same tuple backward sees.
        # What is saved of an operand that is read is its quantized value or
        # the raw operand, quantized again in backward (_saved).
        q_input, saved_input, ctx.replay_input = quantize_operand(
            input,
            formats.input_quant,
            getattr(formats, "recompute_input", None),
            recompute_input,
            ctx.needs_input_grad[1],
        )
        q_weight, saved_weight, ctx.replay_weight = quantize_operand(
            weight,
            formats.weight_quant,
            getattr(formats, "recompute_weight", None),
            recompute_weight,
            ctx.needs_input_grad[0],
        )
        q_bias = formats.bias_quant(bias) if formats.bias_quant and bias is not None else bias

        if formats.fwd_math is not None:
            output = formats.fwd_math(q_input, q_weight, q_bias)
        else:
            output = F.linear(q_input, q_weight, q_bias)

        save(ctx, saved_input, saved_weight)
        ctx.needs_bias_grad = bias is not None

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        formats = ctx.formats

        saved_input, saved_weight = load(ctx)

        grad_input = grad_weight = grad_bias = None

        # Quantize grad_output only for the paths that will consume it, so a
        # frozen layer, or the first layer of a network, does not pay a full
        # grad-output-sized quantize kernel and allocation for a tensor it
        # then discards. Both quantizations run ahead of both math calls, in
        # this order, so a stochastic quantizer draws from the global RNG in
        # the same sequence however many gradients are requested; the *_math
        # hooks may consume randomness of their own afterwards.
        q_igrad_output = (
            (formats.igrad_quant(grad_output) if formats.igrad_quant else grad_output)
            if ctx.needs_input_grad[0]
            else None
        )
        q_wgrad_output = (
            (formats.wgrad_quant(grad_output) if formats.wgrad_quant else grad_output)
            if ctx.needs_input_grad[1]
            else None
        )

        if q_igrad_output is not None:
            bwd_igrad_math = (
                formats.bwd_igrad_math
                if formats.bwd_igrad_math
                else CustomArithLinear._default_bwd_igrad
            )
            # A recomputed operand is quantized again only here, so its copy
            # lives for this one GEMM.
            grad_input = bwd_igrad_math(q_igrad_output, restore(saved_weight, ctx.replay_weight))

        if q_wgrad_output is not None:
            bwd_wgrad_math = (
                formats.bwd_wgrad_math
                if formats.bwd_wgrad_math
                else CustomArithLinear._default_bwd_wgrad
            )
            grad_weight = bwd_wgrad_math(q_wgrad_output, restore(saved_input, ctx.replay_input))

        if ctx.needs_bias_grad and ctx.needs_input_grad[2]:
            q_bgrad_output = (
                formats.bgrad_quant(grad_output) if formats.bgrad_quant else grad_output
            )
            # The bias gradient is the output gradient summed over every
            # leading (batch) dimension.
            grad_bias = q_bgrad_output.reshape(-1, q_bgrad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, grad_bias, None


class QLinear(nn.Linear):
    """``torch.nn.Linear`` with per-tensor quantization and custom arithmetic.

    It holds the same parameters as ``nn.Linear`` and computes the same
    ``input @ weight.T + bias``, but through :class:`CustomArithLinear`. The
    input, the weight, the bias and each gradient can therefore be rounded by
    the quantizers in ``formats``, and each GEMM can run in custom arithmetic
    (see :func:`mptorch.quant.binaryK_gemm_formats`). With ``formats`` left
    ``None``, or with a ``QAffineFormats`` whose slots are all ``None``, the
    layer is numerically identical to ``nn.Linear``. ``formats`` is a
    submodule, so a stateful quantizer in it appears in the layer's
    ``state_dict``. For backward the layer keeps a quantized copy of its
    input and its raw weight, which it quantizes again there, unless
    ``formats``' ``recompute_input`` / ``recompute_weight`` say otherwise.
    A layer over :func:`mptorch.quant.block_gemm_formats` can drop its float
    weight for its block codes with :meth:`freeze_packed_`.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool): whether the layer learns an additive bias.
            Default: ``True``
        device: as for ``nn.Linear``. Default: ``None``
        dtype: as for ``nn.Linear``. Default: ``None``
        formats (QAffineFormats, optional): the quantizers and math hooks.
            ``None`` builds an empty ``QAffineFormats()``. Default: ``None``

    Shape:
        - Input: ``(*, in_features)``, any number of leading dimensions.
        - Output: ``(*, out_features)``.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QLinear, Quant, binaryK_gemm_formats
        >>> formats = binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=16, acc_P=11)
        >>> formats.input_quant = Quant(BinaryK(8, 4))
        >>> formats.weight_quant = Quant(BinaryK(8, 4))
        >>> layer = QLinear(256, 64, formats=formats)
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        formats: QAffineFormats | None = None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CustomArithLinear.apply(input, self.weight, self.bias, self.formats)

    def freeze_packed_(self, *, pack: bool = True) -> "QPackedLinear":
        """Replace the float weight by its block codes, in place.

        The layer becomes a :class:`QPackedLinear`: its ``weight`` is
        ``None``, and two ``uint8`` buffers, ``weight_codes`` and
        ``weight_scales``, hold the weight packed along its input features.
        The codes are what the layer's forward multiplies, so the frozen layer
        returns what it returned before:

        - with ``formats.weight_quant`` unset, the weight packed in the
          weight format of :func:`mptorch.quant.block_gemm_formats`;
        - with ``weight_quant = BlockQuant(fmt, axis=1).pack``, that
          packing: the memoized one, if the weight has not changed since the
          last forward (its stochastic draws included), and the memo is then
          cleared, since it would keep the codes alive;
        - with another quantizer, the weight format's packing of its result.

        Nothing is kept of the float weight, so its memory is freed once
        nothing else refers to it: freeze before building an optimizer or
        any copy of the parameters. A square-tiled weight still serves the
        input gradient, so layers below a frozen one can train; there is no
        weight gradient, and the bias stays a float parameter.

        Args:
            pack (bool): pack the weight. ``False`` allocates zeroed buffers
                of the right shapes instead, on the weight's device, for
                :meth:`~torch.nn.Module.load_state_dict` of a frozen
                checkpoint to fill; on the meta device it allocates
                nothing. Default: ``True``

        Returns:
            QPackedLinear: this layer.

        Raises:
            ValueError: for formats that do not come from
                :func:`mptorch.quant.block_gemm_formats`, for ``pack=True``
                on a weight on the meta device, which has no values, and for
                a ``weight_quant`` result the block hooks cannot multiply.
            TypeError: for a layer that is frozen already (quantizing twice
                is not always the identity), or of a subclass of ``QLinear``,
                whose class the switch would drop.

        Example::

            >>> import torch
            >>> from mptorch import BinaryK, MXFP8_E4M3
            >>> from mptorch.quant import QLinear, block_gemm_formats
            >>> layer = QLinear(64, 32, formats=block_gemm_formats(MXFP8_E4M3, MXFP8_E4M3))
            >>> x = torch.randn(4, 64)
            >>> with torch.no_grad():
            ...     before = layer(x)
            ...     _ = layer.freeze_packed_()
            ...     torch.equal(layer(x), before), layer.weight
            (True, None)
            >>> sorted(layer.state_dict())
            ['_extra_state', 'bias', 'weight_codes', 'weight_scales']
        """
        if type(self) is not QLinear:
            if isinstance(self, QPackedLinear):
                raise TypeError(
                    "this layer is frozen already: freezing it again would quantize its codes "
                    "a second time, which is not always the identity"
                )
            raise TypeError(
                f"freeze_packed_ switches a QLinear's class to QPackedLinear, and this layer is "
                f"a {type(self).__name__}, whose own class the switch would drop"
            )
        packed = _frozen_codes(self, pack)
        self.register_parameter("weight", None)
        self.register_buffer("weight_codes", packed.data)
        self.register_buffer("weight_scales", packed.scales)
        self._packing = (
            packed.fmt,
            packed.shape,
            packed.tensor_scale,
            packed.dtype,
            packed.carrier,
        )
        _switch_class(self, QPackedLinear)
        return cast(QPackedLinear, self)


def _switch_class(layer: Any, cls: type) -> None:
    """Change ``layer``'s class in place, as ``torch.nn.utils.parametrize``
    does to the modules it wraps: the object, its parameters, buffers,
    hooks and submodules stay what they are."""
    layer.__class__ = cls


#: The version of a frozen layer's checkpoint layout, in its extra state.
#: A layout a checkpoint file exists in is a commitment, which is why the
#: extra state has carried this version since the first layout.
_PACKED_LAYOUT = 1

#: The dtypes a packing records, by the names its extra state uses.
_DTYPES = {
    str(d).removeprefix("torch."): d
    for d in (torch.float32, torch.float64, torch.float16, torch.bfloat16)
}


def _frozen_codes(layer: QLinear, pack: bool) -> BlockPacked:
    """What ``freeze_packed_`` stores: the packing the layer's forward
    multiplies, along the weight's input features."""
    formats = layer.formats
    qw = getattr(formats, "_block_weight", None)
    if not isinstance(qw, BlockQuant):
        raise ValueError(
            "freeze_packed_ stores the weight in the block format the layer multiplies it in, "
            "and this layer's formats do not come from mptorch.quant.block_gemm_formats, whose "
            "hooks are the ones that take a packed weight"
        )
    weight_quant = formats.weight_quant
    # a BlockQuant's own (memoized) .pack: the recipe that packs the weight once
    owner = getattr(weight_quant, "__self__", None)
    packer = owner if getattr(weight_quant, "__func__", None) is BlockQuant.pack else None
    weight = layer.weight
    if not pack:
        if isinstance(packer, BlockQuant):
            packer._memo[0] = None
        return _zeroed_codes(weight, packer if isinstance(packer, BlockQuant) else qw)
    if weight.is_meta:
        raise ValueError(
            "this layer's weight is on the meta device and has no values to pack: freeze it "
            "with pack=False, then load a frozen checkpoint (load_state_dict(..., assign=True))"
        )
    # the codes carry no gradient, and the packing ops refuse a weight that
    # requires one under grad mode
    with torch.no_grad():
        q_weight = weight_quant(weight) if weight_quant else weight
        if isinstance(q_weight, torch.Tensor):
            return qw.pack_fresh(q_weight, 1)
    if isinstance(packer, BlockQuant):
        packer._memo[0] = None
    if isinstance(q_weight, BlockPacked):
        if q_weight.shape != tuple(weight.shape) or q_weight.axis != 1:
            raise ValueError(
                f"freeze_packed_ stores the weight packed along its input features (axis 1), and "
                f"this layer's weight_quant packed it as {q_weight}"
            )
        return q_weight
    raise ValueError(
        f"freeze_packed_ stores what the layer's forward multiplies, and its weight_quant "
        f"returns a {type(q_weight).__name__}, which the block hooks do not take"
    )


def _zeroed_codes(weight: torch.Tensor, q: BlockQuant) -> BlockPacked:
    """A packing of ``weight``'s shape in ``q``'s format, all codes zero, for
    a checkpoint to fill. The shapes come from ``block_pack``'s Meta kernel,
    so nothing is packed."""
    fmt = q.fmt
    wide, _ = _call_carrier(q.carrier, weight.dtype)
    carrier = _CARRIER[wide]
    meta = torch.empty(weight.shape, dtype=carrier, device="meta")
    data, scales = torch.ops.mptorch.block_pack.default(
        meta, _block_format_ints(fmt), fmt.elem_max, fmt.scale_max, 1.0, q.rounding.value
    )
    codes = torch.zeros(data.shape, dtype=torch.uint8, device=weight.device)
    scale_codes = torch.zeros(scales.shape, dtype=torch.uint8, device=weight.device)
    return BlockPacked(codes, scale_codes, fmt, tuple(weight.shape), 1, 1.0, weight.dtype, carrier)


class QPackedLinear(QLinear):
    """A :class:`QLinear` whose weight is stored only as its block codes.

    :meth:`QLinear.freeze_packed_` and :func:`mptorch.quant.freeze_packed_`
    make one, in place, from a ``QLinear`` over
    :func:`mptorch.quant.block_gemm_formats`; the class is not constructed
    directly. It is still a ``QLinear`` (``isinstance`` holds) and returns
    what the layer returned before the freeze, from a fraction of the
    memory: the weight's codes and scales are ``uint8`` buffers,
    ``weight_codes`` and ``weight_scales``, which follow ``.to()`` and stay
    ``uint8`` under ``.half()``, and ``weight`` is ``None``. The packing's
    format, shape, tensor scale, dtype and carrier are in the
    ``state_dict``'s ``_extra_state``, as plain values.

    The format itself is code, as every hook of the layer's formats is. To
    load a frozen checkpoint, build the model as it was built, freeze it with
    ``pack=False``, which allocates zeroed codes and packs nothing, and load
    the ``state_dict``. A model built on the meta device and frozen that way
    allocates its codes only when the state is loaded with ``assign=True``,
    so the float model never exists. A checkpoint frozen in another format
    or shape raises, naming both.

    :attr:`packed_weight` is the codes as a :class:`mptorch.quant.BlockPacked`,
    and :meth:`unfreeze_packed_` turns the layer back into a ``QLinear``.
    """

    weight_codes: torch.Tensor
    weight_scales: torch.Tensor
    _packing: tuple[Any, tuple[int, ...], float, torch.dtype, torch.dtype]

    def __init__(self, *args: Any, **kwargs: Any):
        raise TypeError(
            "a QPackedLinear is made from a QLinear over block_gemm_formats, by "
            "layer.freeze_packed_() or mptorch.quant.freeze_packed_(model)"
        )

    @property
    def packed_weight(self) -> BlockPacked:
        """The weight as a :class:`mptorch.quant.BlockPacked` over the two
        buffers, built per call, so that it follows the buffers through
        ``.to()``."""
        fmt, shape, tensor_scale, dtype, carrier = self._packing
        return BlockPacked(
            self.weight_codes, self.weight_scales, fmt, shape, 1, tensor_scale, dtype, carrier
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CustomArithLinear.apply(input, self.packed_weight, self.bias, self.formats)

    def reset_parameters(self) -> None:
        raise RuntimeError(
            "a frozen layer has no float weight to initialize: unfreeze_packed_() it first"
        )

    def extra_repr(self) -> str:
        fmt = self._packing[0]
        nbytes = self.weight_codes.numel() + self.weight_scales.numel()
        return f"{super().extra_repr()}, weight={_block_label(fmt)}, packed_bytes={nbytes}"

    def get_extra_state(self) -> dict[str, Any]:
        fmt, shape, tensor_scale, dtype, carrier = self._packing
        return {
            "layout": _PACKED_LAYOUT,
            "format": list(_block_format_ints(fmt)),
            "elem_max": fmt.elem_max,
            "scale_max": fmt.scale_max,
            "shape": list(shape),
            "axis": 1,
            "tensor_scale": tensor_scale,
            "dtype": str(dtype).removeprefix("torch."),
            "carrier": str(carrier).removeprefix("torch."),
        }

    def set_extra_state(self, state: Any) -> None:
        fmt, shape, _, _, _ = self._packing
        layout = state.get("layout") if isinstance(state, dict) else None
        if layout != _PACKED_LAYOUT:
            raise ValueError(
                f"this checkpoint's frozen weight has layout version {layout!r}, and this "
                f"mptorch reads version {_PACKED_LAYOUT}"
            )
        own = self.get_extra_state()
        for key in ("format", "elem_max", "scale_max", "shape", "axis"):
            if state.get(key) != own[key]:
                raise ValueError(
                    f"this checkpoint's frozen weight has {key} {state.get(key)!r}, and this "
                    f"layer's ({_block_label(fmt)}, shape {shape}) has {own[key]!r}: build the "
                    "layer with the formats the checkpoint was frozen with"
                )
        dtype, carrier = _DTYPES.get(state.get("dtype")), _DTYPES.get(state.get("carrier"))
        if dtype is None or carrier not in _CARRIER:
            raise ValueError(
                f"this checkpoint's frozen weight names dtype {state.get('dtype')!r} and carrier "
                f"{state.get('carrier')!r}, which are not a float dtype and a carrier"
            )
        self._packing = (fmt, shape, float(state["tensor_scale"]), dtype, carrier)

    def unfreeze_packed_(self) -> QLinear:
        """Turn the layer back into a ``QLinear``, in place, whose float
        weight is the decoded codes (:func:`mptorch.quant.block_unpack`): the
        quantized weight, not the one the layer was frozen from, in the dtype
        it was packed from. For fine-tuning from a frozen checkpoint, or
        exporting to tools that read a plain ``nn.Linear``."""
        weight = block_unpack(self.packed_weight).contiguous()
        del self.weight_codes
        del self.weight_scales
        del self._packing
        _switch_class(self, QLinear)
        self.weight = nn.Parameter(weight)
        return self


def freeze_packed_(model: nn.Module, *, pack: bool = True) -> int:
    """:meth:`QLinear.freeze_packed_` on every ``QLinear`` of ``model`` whose
    formats come from :func:`mptorch.quant.block_gemm_formats`.

    The layers are frozen one at a time, so the freeze holds at most one
    layer's float weight and its codes at once beyond the model. A model
    that does not fit on its device as float can be frozen on the CPU and
    then moved. Other modules are left as they are.

    Args:
        model (nn.Module): the model, or one layer.
        pack (bool): as for :meth:`QLinear.freeze_packed_`: ``False``
            allocates zeroed codes for a frozen checkpoint to fill.
            Default: ``True``

    Returns:
        int: how many layers were frozen.

    Example::

        >>> import torch
        >>> from torch import nn
        >>> from mptorch import BinaryK, MXFP8_E4M3
        >>> from mptorch.quant import QLinear, block_gemm_formats, freeze_packed_
        >>> def mlp():
        ...     return nn.Sequential(
        ...         QLinear(64, 64, formats=block_gemm_formats(MXFP8_E4M3, MXFP8_E4M3)),
        ...         nn.ReLU(),
        ...         QLinear(64, 8, formats=block_gemm_formats(MXFP8_E4M3, MXFP8_E4M3)),
        ...     )
        >>> model = mlp()
        >>> freeze_packed_(model)
        2
        >>> skeleton = mlp()  # load: build, freeze without packing, then load
        >>> freeze_packed_(skeleton, pack=False)
        2
        >>> skeleton.load_state_dict(model.state_dict())
        <All keys matched successfully>
    """
    layers = [
        m
        for m in model.modules()
        if type(m) is QLinear and isinstance(getattr(m.formats, "_block_weight", None), BlockQuant)
    ]
    for m in layers:
        m.freeze_packed_(pack=pack)
    return len(layers)
