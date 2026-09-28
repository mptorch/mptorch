"""The convolutions in custom arithmetic: :func:`conv_formats`.

A convolution's three passes, the forward and the two gradients, are GEMMs
whose operands are gathers of the convolution's tensors. The eight
``custom_conv_*`` ops run each pass in the arithmetic of the GEMM op it is
named after, with the gather in the kernel's tile load
(``mptorch/csrc/common/gemm_gather.h``), so no ``unfold`` buffer is ever
built: the forward of a 3x3 convolution would otherwise materialize nine times
its input. Every format, accumulate algorithm and rounding mode of the GEMM
applies, and each pass is bit for bit the GEMM over the operands ``unfold``
would have built, zeros included, in the same order; ``tests/test_qconv_gemm.py``
holds all three to that reference.

``conv_formats(mac)`` resolves a mac once, as ``matmul_formats`` does, and
returns a ``QAffineFormats`` whose three math hooks are the three passes, in
the keyword-signature contract ``CustomArithConvNd`` calls them with, so
``QConv1d``/``QConv2d``/``QConv3d`` run them unchanged.
"""

from functools import lru_cache
from typing import Any

import torch

from mptorch.number import Number

from .mac import FusedMac, Mac, SplitMac, spec_for_mac
from .modules.format import QAffineFormats
from .ops import _NARROW_STORAGE, _check_stored, _GemmSpec, _report_findings

__all__ = ["conv_formats"]

# The three passes, as the op's `conv_pass` argument.
_FWD, _IGRAD, _WGRAD = 0, 1, 2

# What a NAIVE single-format spec's arguments need after them to be a conv
# op's: the conv ops take the *_accumulated schema, whose tail is a block size
# and an outer format, none of which NAIVE has. The two families' tails differ
# in their outer format's fields, not in their length.
_NAIVE_TAIL = {
    "binaryK": (0, False, 0, 0, 0, True, 0, 0, 0),
    "superfp": (0, False, 0, 0, 0, 0, True, 0, 0),
}

_BINARY32_ONLY = (
    "the conv ops have binary32 kernels only so far, so they take float32, float16 or "
    "bfloat16 operands and no carrier=torch.float64 (dev/continuation_plan.md, phase G)"
)


@lru_cache(maxsize=1)
def _conv_ops() -> dict[Any, tuple[Any, tuple[Any, ...]]]:
    """Each GEMM op's conv op, and the arguments a spec of it lacks.

    Keyed by the GEMM op a spec names, so a spec from ``spec_for_mac`` maps to
    its conv op without being built twice: a NAIVE single-format spec runs on
    the conv op of its ``*_accumulated`` twin with a NAIVE tail, the other two
    kinds on theirs as they are.
    """
    ops = torch.ops.mptorch
    table: dict[Any, tuple[Any, tuple[Any, ...]]] = {}
    for family in ("binaryK", "superfp"):
        for mac in ("", "_fma"):
            conv = getattr(ops, f"custom_conv_{family}{mac}").default
            table[getattr(ops, f"custom_matmul_{family}{mac}").default] = (
                conv,
                _NAIVE_TAIL[family],
            )
            table[getattr(ops, f"custom_matmul_{family}{mac}_accumulated").default] = (conv, ())
            table[getattr(ops, f"custom_matmul_{family}{mac}_mixed").default] = (
                getattr(ops, f"custom_conv_{family}{mac}_mixed").default,
                (),
            )
    return table


def _ntuple(value: Any, nd: int, name: str) -> tuple[int, ...]:
    """``value`` as ``nd`` integers: an int is repeated, a sequence checked."""
    if isinstance(value, int):
        return (value,) * nd
    out = tuple(int(v) for v in value)
    if len(out) != nd:
        raise ValueError(f"{name} must have {nd} entries for a {nd}d convolution, got {out}")
    return out


def _leading_padding(
    padding: Any, kernel: tuple[int, ...], stride: tuple[int, ...], dilation: tuple[int, ...]
) -> tuple[int, ...]:
    """The padding before each spatial dimension, the only side the op is told.

    ``"same"`` pads ``d*(k - 1)`` in all, and where that is odd puts the extra
    position after, as ``F.conv*d`` does; the op reads zero past the input's
    end without being told how far, so the asymmetry costs no copy.
    """
    nd = len(kernel)
    if padding == "valid":
        return (0,) * nd
    if padding == "same":
        if any(s != 1 for s in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")
        return tuple(d * (k - 1) // 2 for k, d in zip(kernel, dilation, strict=True))
    return _ntuple(padding, nd, "padding")


def _output_extent(
    padding: Any,
    lead: tuple[int, ...],
    size: tuple[int, ...],
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
) -> tuple[int, ...]:
    """The forward's spatial output extent, as ``F.conv*d`` computes it."""
    if padding == "same":
        return size
    out = tuple(
        (n + 2 * p - d * (k - 1) - 1) // s + 1
        for n, p, k, s, d in zip(size, lead, kernel, stride, dilation, strict=True)
    )
    if any(o < 1 for o in out):
        raise ValueError(
            f"the kernel {kernel} (dilation {dilation}) does not fit the padded input {size}"
        )
    return out


class _ConvOp:
    """One resolved conv op: what a pass calls, bound once per layer.

    ``args`` are the spec's arguments with whatever the conv op takes after
    them, ``prec_idx`` the pass's map for a palette op. The per-call part is
    the GEMM's: the carrier's findings and the storage check for the operand
    dtype, and a refusal of float64, which has no conv kernels yet.
    """

    __slots__ = ("op", "args", "spec", "prec_idx")

    def __init__(self, spec: _GemmSpec, prec_idx: torch.Tensor | None):
        self.op, tail = _conv_ops()[spec.op]
        self.args = (*spec.args, *tail)
        self.spec = spec
        self.prec_idx = prec_idx

    def __call__(
        self,
        conv_pass: int,
        a: torch.Tensor,
        b: torch.Tensor,
        out_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        groups: int,
    ) -> torch.Tensor:
        dtype = a.dtype
        if dtype is torch.float64:
            raise ValueError(_BINARY32_ONLY)
        spec = self.spec
        found = spec.findings[0]
        if found:
            _report_findings(found)
        if spec.stored and dtype in _NARROW_STORAGE:
            _check_stored(spec.stored, dtype)
        geometry = (conv_pass, out_size, stride, padding, dilation, groups)
        if self.prec_idx is None:
            return self.op(a, b, *geometry, *self.args)
        return self.op(a, b, self.prec_idx, *geometry, *self.args)


def _missing_map(pass_name: str, argument: str):
    """A pass of a palette format whose map was not given: raises when reached."""

    def missing(*_args, **_kwargs) -> torch.Tensor:
        raise ValueError(
            f"this format is a palette, so its {pass_name} needs its own prec_idx map "
            f"(shaped like that pass's result): pass {argument}= to conv_formats"
        )

    return missing


def _batched(x: torch.Tensor, nd: int) -> tuple[torch.Tensor, bool]:
    """``x`` with a batch dimension, and whether one was added."""
    if x.dim() == nd + 1:
        return x.unsqueeze(0), True
    return x, False


def _fwd_hook(run: _ConvOp):
    def fwd(
        q_input: torch.Tensor,
        q_weight: torch.Tensor,
        q_bias: torch.Tensor | None,
        *,
        stride,
        padding,
        dilation,
        groups: int,
        nd: int,
    ) -> torch.Tensor:
        x, unbatched = _batched(q_input, nd)
        kernel = tuple(q_weight.shape[2:])
        s = _ntuple(stride, nd, "stride")
        d = _ntuple(dilation, nd, "dilation")
        lead = _leading_padding(padding, kernel, s, d)
        out_size = _output_extent(padding, lead, tuple(x.shape[2:]), kernel, s, d)
        out = run(_FWD, q_weight, x, out_size, s, lead, d, groups)
        if q_bias is not None:
            # In place, as the Linear hooks add theirs: `out` is the op's own
            # fresh tensor, so this is `out + bias` bit for bit without a
            # second output-sized allocation; a wider bias promotes instead.
            view = q_bias.view(-1, *([1] * nd))
            out = out.add_(view) if q_bias.dtype == out.dtype else out + view
        return out.squeeze(0) if unbatched else out

    return fwd


def _igrad_hook(run: _ConvOp):
    def bwd_igrad(
        q_grad: torch.Tensor,
        q_weight: torch.Tensor,
        *,
        input_size,
        stride,
        padding,
        dilation,
        groups: int,
        nd: int,
    ) -> torch.Tensor:
        g, unbatched = _batched(q_grad, nd)
        kernel = tuple(q_weight.shape[2:])
        s = _ntuple(stride, nd, "stride")
        d = _ntuple(dilation, nd, "dilation")
        lead = _leading_padding(padding, kernel, s, d)
        out = run(_IGRAD, q_weight, g, tuple(input_size[-nd:]), s, lead, d, groups)
        return out.squeeze(0) if unbatched else out

    return bwd_igrad


def _wgrad_hook(run: _ConvOp):
    def bwd_wgrad(
        q_grad: torch.Tensor,
        q_input: torch.Tensor,
        *,
        weight_size,
        stride,
        padding,
        dilation,
        groups: int,
        nd: int,
    ) -> torch.Tensor:
        g, _ = _batched(q_grad, nd)
        x, _ = _batched(q_input, nd)
        kernel = tuple(weight_size[2:])
        s = _ntuple(stride, nd, "stride")
        d = _ntuple(dilation, nd, "dilation")
        lead = _leading_padding(padding, kernel, s, d)
        return run(_WGRAD, g, x, kernel, s, lead, d, groups)

    return bwd_wgrad


def conv_formats(
    mac: Mac | Number,
    *,
    prec_idx: torch.Tensor | None = None,
    igrad_prec_idx: torch.Tensor | None = None,
    wgrad_prec_idx: torch.Tensor | None = None,
) -> QAffineFormats:
    """Build a ``QAffineFormats`` whose three convolution passes run in ``mac``'s arithmetic.

    The forward convolution, the input gradient and the weight gradient of a
    ``QConv1d``, ``QConv2d`` or ``QConv3d`` each run as one GEMM in the
    arithmetic ``mac`` names, with the operands gathered from the
    convolution's tensors inside the kernel instead of unfolded, so none of
    the three allocates more than its result. Each pass is bit-identical to
    the GEMM of the same mac over ``F.unfold``'s operands. Any stride,
    padding (``"same"`` and ``"valid"`` included), dilation and ``groups``
    works; groups share one launch, and a depthwise convolution (one channel
    per group) runs at a sixteenth of the kernel's tile width, which is
    correct but not fast. The input gradient sums over every kernel tap,
    including the ``1 - 1/stride**nd`` share of them that a stride leaves
    zero, because that is the sum the reference computes, and a zero term is
    not free for every accumulate algorithm. Like :func:`matmul_formats`,
    this sets only the math hooks; operand quantization (``weight_quant``,
    ``input_quant`` and the other ``*_quant`` slots) is layered on by the
    caller.

    The kernels compute in binary32: the operands may be float32, float16 or
    bfloat16, and a float64 model, or a mac with ``carrier=torch.float64``,
    raises (the binary64 kernels are ``dev/continuation_plan.md``'s phase
    G). MPS tensors raise too (phase H).

    Args:
        mac (SplitMac or FusedMac or Number): the dot-product arithmetic,
            resolved once through :func:`mptorch.quant.mac.spec_for_mac`. A
            bare ``Number`` is ``SplitMac(fmt, fmt)``. Every
            ``accumulate_algorithm`` applies; a palette selects the
            per-output-element ops, which need a map per pass.
        prec_idx (Tensor, optional): the forward's palette map, indexed like
            the output with its spatial dimensions flattened: ``[Cout, L]``,
            ``[Cout, 1]`` (one format per output channel) or ``[1, L]``,
            optionally with a leading batch dimension. Required by, and only
            by, a palette ``mac``. Default: ``None``
        igrad_prec_idx (Tensor, optional): the input gradient's map, shaped
            the same way over ``[Cin, H*W...]``. Default: ``None``
        wgrad_prec_idx (Tensor, optional): the weight gradient's map, over the
            weight as ``[Cout, Cin/groups * prod(kernel)]``. Default: ``None``

    Returns:
        QAffineFormats: with ``fwd_math``, ``bwd_igrad_math`` and
        ``bwd_wgrad_math`` set and every ``*_quant`` slot left ``None``.

    Raises:
        ValueError: for a palette ``mac`` without ``prec_idx``, a map given
            to a ``mac`` without a palette, or ``carrier=torch.float64``.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QConv2d, SplitMac, conv_formats
        >>> fmt = BinaryK(8, 4)
        >>> conv = QConv2d(3, 8, 3, padding=1, formats=conv_formats(SplitMac(fmt, fmt)))
        >>> conv(torch.randn(2, 3, 16, 16)).shape
        torch.Size([2, 8, 16, 16])
    """
    if isinstance(mac, Number):
        mac = SplitMac(mac, mac)
    if not isinstance(mac, SplitMac | FusedMac):
        raise TypeError(f"mac must be a SplitMac, a FusedMac or a Number, got {type(mac).__name__}")
    spec = spec_for_mac(mac)
    if spec.carrier is torch.float64:
        raise ValueError(_BINARY32_ONLY)
    maps = (prec_idx, igrad_prec_idx, wgrad_prec_idx)
    if spec.mixed and prec_idx is None:
        raise ValueError(
            "a palette format selects its entry per output element, so it needs a "
            "prec_idx map; pass one, or give a single format per slot"
        )
    if not spec.mixed and any(m is not None for m in maps):
        raise ValueError("prec_idx has no meaning without a palette format to select from")
    if not spec.mixed:
        run = _ConvOp(spec, None)
        return QAffineFormats(
            fwd_math=_fwd_hook(run),
            bwd_igrad_math=_igrad_hook(run),
            bwd_wgrad_math=_wgrad_hook(run),
        )
    return QAffineFormats(
        fwd_math=_fwd_hook(_ConvOp(spec, prec_idx)),
        bwd_igrad_math=(
            _igrad_hook(_ConvOp(spec, igrad_prec_idx))
            if igrad_prec_idx is not None
            else _missing_map("input gradient", "igrad_prec_idx")
        ),
        bwd_wgrad_math=(
            _wgrad_hook(_ConvOp(spec, wgrad_prec_idx))
            if wgrad_prec_idx is not None
            else _missing_map("weight gradient", "wgrad_prec_idx")
        ),
    )
