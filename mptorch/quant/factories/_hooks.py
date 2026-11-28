"""The hooks of a layer's format container: one builder per pass and per kind of GEMM.

A ``QLinear``'s container is a :class:`~mptorch.quant.QAffineFormats`, and
``CustomArithLinear`` (``mptorch/quant/modules/linear.py``) calls its three
hooks like this::

    fwd_math(q_input, q_weight, q_bias)        # q_input @ q_weight.T + q_bias
    bwd_igrad_math(q_igrad_output, q_weight)   # q_igrad_output @ q_weight
    bwd_wgrad_math(q_wgrad_output, q_input)    # q_wgrad_output.T @ q_input

``q_input``, ``q_igrad_output`` and ``q_wgrad_output`` may have any number of
leading (batch) dimensions, as ``nn.Linear`` allows. Each hook flattens them
into one row dimension before the 2D GEMM, the same reshape
``CustomArithLinear._default_bwd_wgrad`` makes, and the forward and input
gradient hooks restore them on the result. A ``QMatmul``'s container is a
:class:`~mptorch.quant.QMatmulFormats`, whose hooks follow ``torch.matmul``.

Each hook comes from a builder of its own, one per pass and per kind of
GEMM (``_linear_fwd``, ``_block_linear_igrad``, ``_matmul_bgrad`` and the
others), over a GEMM resolved once into a ``_GemmSpec``. The factories of
``factories/linear.py`` and ``factories/matmul.py`` compose them, and so does
:class:`mptorch.quant.GemmFormats`, which may run each pass in different
arithmetic or leave a pass to PyTorch.
"""

from collections.abc import Callable
from dataclasses import replace
from functools import partial

import torch

from mptorch.number import (
    BlockFormat,
)

from ..mac import BlockMac, BlockQuant, Quant, cast_quant, spec_for_block_mac
from ..modules.format import QAffineFormats, QMatmulFormats
from ..ops import BlockPacked
from ..ops._operands import _gemm_mixed_nd
from ..ops._spec import _GemmSpec, _round_result, _wide_refusal
from ..ops.block import _BlockGemmSpec, _run_block_gemm, _stored_as

# A resolved GEMM, called once per pass as `matmul(a, b, ta, tb)`: it computes
# `op(a) @ op(b)` for 2D operands, where `ta` and `tb` say whether `op`
# transposes `a` and `b` (a flag the kernel reads, so no transposed copy is made).
_Matmul2D = Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor]


def _unviewed(view: torch.Tensor) -> torch.Tensor:
    """``view``, a view of a result the hook allocated, as a tensor autograd
    does not take for a view.

    A layer's autograd Function returns what its forward hook returns, and
    autograd refuses an in-place op on a view that a custom Function created
    (it cannot rebase the view's history onto the Function), so an in-place
    activation after the layer (``nn.ReLU(inplace=True)``) would raise.
    ``detach()`` shares the storage without the view relation; nothing else
    refers to the result, and the Function's own backward computes the
    gradient. This is what ``torch.matmul`` does internally, with
    ``_unsafe_view``, for the same reason.
    """
    return view.detach()


def _flatten_leading(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Flatten all but the last dimension of ``x`` into one leading dimension.

    Return the 2D tensor and the original shape, so the caller can give a
    result the leading dimensions ``x`` had.
    """
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape


def _spec_matmul(spec: _GemmSpec, runner: Callable, where: str) -> _Matmul2D:
    """``runner`` (``_run_gemm`` or ``_gemm_nd``) bound to ``spec``, as a pass's
    hook calls it.

    A spec whose result can be wide (the fixed-point GEMM's, with an
    accumulate format wider than the operands' dtype) computes that result in
    float64. A forward pass with an output rounding rounds it once and stores
    it in the operands' dtype (:func:`_with_output`, through the container's
    ``_wide_fwd``); any other pass refuses it here, naming ``where``, rather
    than let a float64 result reach the layer's output or autograd, which
    would round it to the gradient's dtype without saying so. Every other
    spec gets the plain partial.
    """
    call = partial(runner, spec)
    if not spec.wide:
        return call

    def refusing(a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool) -> torch.Tensor:
        if a.dtype in spec.wide:
            raise _wide_refusal(spec, a.dtype, where, a.device)
        return call(a, b, trans_a, trans_b)

    return refusing


def _gemm_formats(
    matmul: _Matmul2D, igrad: _Matmul2D | None = None, wgrad: _Matmul2D | None = None
) -> QAffineFormats:
    """Build the three math hooks of a Linear layer, each calling a resolved 2D GEMM.

    Which operand each hook transposes, how the leading dimensions are
    flattened into one row dimension and restored, and how the bias is added
    belong to the Linear layer, not to the formats, so they are written once,
    in the three hook builders below, and each factory passes only its
    resolved ``matmul``. That GEMM runs every pass, unless ``igrad`` (for the
    input gradient) or ``wgrad`` (for the weight gradient) gives a different
    one.
    """
    return QAffineFormats(
        fwd_math=_linear_fwd(matmul),
        bwd_igrad_math=_linear_igrad(matmul if igrad is None else igrad),
        bwd_wgrad_math=_linear_wgrad(matmul if wgrad is None else wgrad),
    )


def _linear_fwd(matmul: _Matmul2D) -> Callable:
    """A Linear layer's forward hook over a resolved 2D GEMM: ``x @ w^T + b``."""

    def fwd(
        q_input: torch.Tensor, q_weight: torch.Tensor, q_bias: torch.Tensor | None
    ) -> torch.Tensor:
        return _linear_forward(matmul, q_input, q_weight, q_bias)

    return fwd


def _linear_forward(
    matmul: _Matmul2D, q_input: torch.Tensor, q_weight: torch.Tensor, q_bias: torch.Tensor | None
) -> torch.Tensor:
    """``x @ w^T + b`` over ``matmul``, the leading dimensions flattened into
    rows and restored: the body of every Linear forward hook."""
    x_flat, x_shape = _flatten_leading(q_input)
    out = matmul(x_flat, q_weight, False, True)
    if q_input.dim() != 2:
        out = _unviewed(out.reshape(*x_shape[:-1], q_weight.shape[0]))
    # add_ rather than +: `out` is a view of a tensor the op allocated
    # inside this call, and nothing else refers to it, so adding the bias
    # in place gives the same bits as `out + q_bias` and saves one
    # output-sized allocation per forward. A bias of a different dtype
    # takes `out + q_bias`, which promotes the result's dtype where `add_`
    # would raise.
    if q_bias is None:
        return out
    return out.add_(q_bias) if q_bias.dtype == out.dtype else out + q_bias


def _linear_igrad(matmul: _Matmul2D) -> Callable:
    """A Linear layer's input-gradient hook over a resolved 2D GEMM: ``g @ w``."""

    def bwd_igrad(q_igrad_output: torch.Tensor, q_weight: torch.Tensor) -> torch.Tensor:
        g_flat, g_shape = _flatten_leading(q_igrad_output)
        out = matmul(g_flat, q_weight, False, False)
        return out.reshape(*g_shape[:-1], q_weight.shape[1])

    return bwd_igrad


def _linear_wgrad(matmul: _Matmul2D) -> Callable:
    """A Linear layer's weight-gradient hook over a resolved 2D GEMM: ``g^T @ x``."""

    def bwd_wgrad(q_wgrad_output: torch.Tensor, q_input: torch.Tensor) -> torch.Tensor:
        g_flat, _ = _flatten_leading(q_wgrad_output)
        i_flat, _ = _flatten_leading(q_input)
        return matmul(g_flat, i_flat, True, False)

    return bwd_wgrad


def _with_output[FormatsT: (QAffineFormats, QMatmulFormats)](
    formats: FormatsT, out: Callable, plain: Callable
) -> FormatsT:
    """Make ``formats`` round each forward result with ``out``, after the bias is added.

    The forward runs ``formats.fwd_math``, or ``plain`` when there is none
    (``F.linear``, ``conv._plain_conv`` or ``torch.matmul``, the call the
    layer's autograd Function would make), and then ``out`` rounds its result.
    The rounding happens inside the Function's forward, which the backward
    pass does not differentiate, so its gradient passes straight through. A
    container with a wide fixed-point forward (``_wide_fwd``, set by the
    builders over a spec whose result can be float64) runs that hook instead,
    and rounds a float64 result once into the input's dtype
    (``ops._spec._round_result``). The result is a new tensor the hook
    allocated (true of every hook in this module), so a ``Quant`` rounds it
    in place. The exceptions are a
    ``Quant`` with ``carrier=torch.float64`` and a result that is not
    contiguous (a channels_last convolution's), which the in-place quantizers
    refuse; those round into a new tensor. ``formats`` is modified in place
    and returned.
    """
    wide = formats._wide_fwd
    run = (formats.fwd_math or plain) if wide is None else wide
    rounded = _output_rounding(out)
    if wide is None:

        def fwd(*args, **kwargs):
            return rounded(run(*args, **kwargs))

    else:
        # A fixed-point GEMM whose result can be wide (`_spec_matmul`): such a
        # result is float64, the bias added in float64 too, and the output
        # rounding rounds it once, then it is stored in the input's dtype.

        def fwd(*args, **kwargs):
            result = run(*args, **kwargs)
            dtype = args[0].dtype
            return rounded(result) if result.dtype is dtype else _round_result(out, result, dtype)

    formats.fwd_math = fwd
    return formats


def _output_rounding(out: Callable) -> Callable[[torch.Tensor], torch.Tensor]:
    """``out``, as it rounds a forward result the hook has just allocated: in
    place for a ``Quant`` that can (not one with ``carrier=torch.float64``,
    nor on a result that is not contiguous, which the in-place quantizers
    refuse), into a new tensor otherwise."""
    if type(out) is not Quant or out.inplace or out.carrier is torch.float64:
        return out
    in_place = replace(out, inplace=True)

    def rounded(result: torch.Tensor) -> torch.Tensor:
        return in_place(result) if result.is_contiguous() else out(result)

    return rounded


# A resolved GEMM for a matmul's hooks, called as `matmul(a, b, ta, tb)`: like
# `_Matmul2D`, but under torch.matmul's operand rules (batch broadcasting and
# 1D promotion) rather than on 2D operands, which the Linear hooks above make
# by flattening. `ta` and `tb` say whether `a` and `b` are transposed.
_MatmulNd = Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor]


def _matmul_formats(
    fwd_matmul: _MatmulNd, agrad_matmul: _MatmulNd, bgrad_matmul: _MatmulNd
) -> QMatmulFormats:
    """Build the three math hooks of a matmul, each calling a resolved GEMM.

    The difference from :func:`_gemm_formats` is which operands each pass
    transposes: the forward is ``a @ b``, ``a``'s gradient is ``grad @ b^T``
    and ``b``'s is ``a^T @ grad``. Nothing is flattened and no bias is added,
    since a matmul broadcasts over its batch dimensions instead of merging
    them, and has no bias. It takes one GEMM per pass because a palette needs
    a different ``prec_idx`` map in each pass; every other format passes the
    same GEMM three times.
    """

    return QMatmulFormats(
        fwd_math=_matmul_fwd(fwd_matmul),
        bwd_agrad_math=_matmul_agrad(agrad_matmul),
        bwd_bgrad_math=_matmul_bgrad(bgrad_matmul),
    )


def _matmul_fwd(matmul: _MatmulNd) -> Callable:
    """A matmul's forward hook over a resolved GEMM: ``a @ b``."""

    def fwd(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        return matmul(q_a, q_b, False, False)

    return fwd


def _matmul_agrad(matmul: _MatmulNd) -> Callable:
    """``a``'s gradient hook over a resolved GEMM: ``grad @ b^T``."""

    def bwd_agrad(q_grad: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        return matmul(q_grad, q_b, False, True)

    return bwd_agrad


def _matmul_bgrad(matmul: _MatmulNd) -> Callable:
    """``b``'s gradient hook over a resolved GEMM: ``a^T @ grad``."""

    def bwd_bgrad(q_grad: torch.Tensor, q_a: torch.Tensor) -> torch.Tensor:
        return matmul(q_a, q_grad, True, False)

    return bwd_bgrad


def _mixed_matmul(spec, prec_idx: torch.Tensor | None, pass_name: str, argument: str):
    """Return one pass's GEMM over a palette op, or, without a map, a hook that raises.

    A ``prec_idx`` map has one entry per *output* element, and the three
    passes of a matmul have outputs of three different shapes: ``[M, N]``,
    ``[M, K]`` and ``[K, N]``. One map cannot serve all three, so a pass whose
    map was not given gets a hook that raises an error naming the
    ``matmul_formats`` argument to pass, instead of failing with a shape error
    from the op or silently falling back to unquantized arithmetic.
    """
    if prec_idx is None:

        def missing(*_args, **_kwargs) -> torch.Tensor:
            raise ValueError(
                f"this format is a palette, so its {pass_name} needs its own prec_idx map "
                f"(shaped like that pass's output): pass {argument}= to matmul_formats"
            )

        return missing

    def matmul(a: torch.Tensor, b: torch.Tensor, ta: bool, tb: bool) -> torch.Tensor:
        return _gemm_mixed_nd(spec, a, b, prec_idx, ta, tb)

    return matmul


# --- block formats --------------------------------------------------------------
#
# The block GEMM (`mptorch.quant.block_matmul`) multiplies packed operands,
# and reads each along its reduction dimension K: an operand is either packed
# along K, or packed along its other dimension and read transposed. So each
# hook packs its operands along the dimension its pass sums over, as MX
# training recipes do, except an operand that arrives already packed. That is
# a weight packed once by `BlockQuant.pack` (in a `weight_quant` slot), or an
# operand in a square-tiled format, which quantizes to the same values along
# either axis and so can serve the backward pass, read transposed. A 1D or
# non-square format quantizes differently along each axis, so its backward
# pass packs the float operand again, and a weight that arrives packed in such
# a format cannot be used for the input gradient.


def _square_tile_hint(fmt: BlockFormat) -> str:
    """The expression that builds the square-tiled variant of a format, for messages."""
    return f"dataclasses.replace(fmt, block_rows={fmt.block_size})"


def _packed_operand(q: BlockQuant, x: torch.Tensor | BlockPacked, axis: int) -> BlockPacked:
    """Return ``x`` packed along ``axis`` by ``q``, or ``x`` itself if it is already packed."""
    return x if isinstance(x, BlockPacked) else q.pack_fresh(x, axis)


def _reused(x: torch.Tensor | BlockPacked, q: BlockQuant, where: str) -> BlockPacked | None:
    """Return the packing the forward pass made, for a backward pass to read
    along its other axis, or ``None`` if the float operand was saved instead
    (the caller then packs it again).

    Only a square tile quantizes to the same values along both axes, so a
    packing in a 1D or non-square format raises, and so does a packing in a
    format other than the one this pass rounds to."""
    if not isinstance(x, BlockPacked):
        return None
    if x.fmt != q.fmt:
        raise ValueError(
            f"{where} rounds this operand to {q.fmt}, and the forward pass packed it in "
            f"{x.fmt}: leave the *_quant slot unset so it is packed per pass"
        )
    if not x.fmt.is_square:
        raise ValueError(
            f"{where} reads the operand the forward pass packed along its other axis, and "
            f"{x.fmt}'s tiles are not square, so its values along that axis are not the ones "
            f"this pass rounds to: leave the *_quant slot unset so it is packed per pass, or use "
            f"a square tile, {_square_tile_hint(x.fmt)}, which quantizes the same along either "
            "axis (a layer frozen by freeze_packed_ has no float weight to pack again: freeze it "
            "in a square-tile format to train below it)"
        )
    return x


def _block_matmul_fwd(mac: BlockMac) -> Callable:
    """A matmul's forward hook over ``mac``'s block GEMM: ``a`` packed along
    its last dimension and ``b`` along its second-to-last."""
    spec = spec_for_block_mac(mac)
    qa, qb = cast_quant(mac.a), cast_quant(mac.b)

    def fwd(q_a, q_b):
        dtype = q_a.dtype
        out = _run_block_gemm(spec, _packed_operand(qa, q_a, -1), _packed_operand(qb, q_b, -2))
        return _stored_as(spec, out, dtype)

    return fwd


def _block_matmul_agrad(mac: BlockMac) -> Callable:
    """``a``'s gradient hook, ``grad @ b^T``, over ``mac``'s block GEMM: both
    packed along their last dimension, ``b`` read from the forward's packing
    when its tiles are square."""
    spec = spec_for_block_mac(mac)
    ag_a, ag_b = cast_quant(mac.a), cast_quant(mac.b)

    def bwd_agrad(q_grad, q_b):
        b = _reused(q_b, ag_b, "a's gradient")
        b_t = b.mT if b is not None else ag_b.pack_fresh(q_b, -1).mT
        out = _run_block_gemm(spec, ag_a.pack_fresh(q_grad, -1), b_t)
        return _stored_as(spec, out, q_grad.dtype)

    return bwd_agrad


def _block_matmul_bgrad(mac: BlockMac) -> Callable:
    """``b``'s gradient hook, ``a^T @ grad``, over ``mac``'s block GEMM: both
    packed along their second-to-last dimension, ``a`` read from the
    forward's packing when its tiles are square."""
    spec = spec_for_block_mac(mac)
    bg_a, bg_b = cast_quant(mac.a), cast_quant(mac.b)

    def bwd_bgrad(q_grad, q_a):
        a = _reused(q_a, bg_a, "b's gradient")
        a_t = a.mT if a is not None else bg_a.pack_fresh(q_a, -2).mT
        out = _run_block_gemm(spec, a_t, bg_b.pack_fresh(q_grad, -2))
        return _stored_as(spec, out, q_grad.dtype)

    return bwd_bgrad


def _block_linear_fwd(spec: _BlockGemmSpec, qi: BlockQuant, qw: BlockQuant) -> Callable:
    """A Linear layer's forward hook over a block GEMM: the input packed along
    its features by ``qi``, the weight along its input features by ``qw``
    (or taken as it is when it arrives packed)."""

    def fwd(q_input, q_weight, q_bias):
        x_flat, x_shape = _flatten_leading(q_input)
        w = q_weight if isinstance(q_weight, BlockPacked) else qw.pack_fresh(q_weight, 1)
        out = _stored_as(
            spec, _run_block_gemm(spec, qi.pack_fresh(x_flat, -1), w.mT), q_input.dtype
        )
        if q_input.dim() != 2:
            out = _unviewed(out.reshape(*x_shape[:-1], w.shape[0]))
        if q_bias is None:
            return out
        return out.add_(q_bias) if q_bias.dtype == out.dtype else out + q_bias

    return fwd


def _block_linear_igrad(spec: _BlockGemmSpec, qg: BlockQuant, qw: BlockQuant) -> Callable:
    """A Linear layer's input-gradient hook over a block GEMM: the output
    gradient packed along its features by ``qg``, the weight along its output
    features by ``qw`` (or read from the forward's packing, square tiles only)."""

    def bwd_igrad(q_igrad_output, q_weight):
        g_flat, g_shape = _flatten_leading(q_igrad_output)
        w = _reused(q_weight, qw, "the input gradient")
        if w is None:
            w = qw.pack_fresh(q_weight, 0)
        out = _run_block_gemm(spec, qg.pack_fresh(g_flat, -1), w)
        return _stored_as(spec, out, q_igrad_output.dtype).reshape(*g_shape[:-1], w.shape[1])

    return bwd_igrad


def _block_linear_wgrad(spec: _BlockGemmSpec, qg: BlockQuant, qi: BlockQuant) -> Callable:
    """A Linear layer's weight-gradient hook over a block GEMM: the output
    gradient (by ``qg``) and the input (by ``qi``) both packed along the
    batch dimension."""

    def bwd_wgrad(q_wgrad_output, q_input):
        g_flat, _ = _flatten_leading(q_wgrad_output)
        i_flat, _ = _flatten_leading(q_input)
        out = _run_block_gemm(spec, qg.pack_fresh(g_flat, 0).mT, qi.pack_fresh(i_flat, 0))
        return _stored_as(spec, out, q_wgrad_output.dtype)

    return bwd_wgrad
