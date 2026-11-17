"""Convolutions in quantized arithmetic: :func:`conv_formats`.

Each of a convolution's three passes (the forward pass and the two
gradients) is a GEMM whose operands are gathered from the convolution's
tensors. The ``custom_conv_*`` ops (twenty of them: single-format, palette
and cross-family, each with an odd twin for ``carrier_rounding=RoundMode.RO``)
run each pass in the arithmetic of the GEMM op they are named after, and
gather the operands while loading the kernel's tiles
(``mptorch/csrc/common/gemm_gather.h``). No ``unfold`` buffer is ever built,
which matters: the one for the forward pass of a 3x3 convolution would be
nine times the size of its input. Every simulated format, accumulate
algorithm and rounding mode that the GEMM supports works here too.

The forward pass and the weight gradient are bit for bit the GEMM over the
operands ``unfold`` would have built, zeros included, summed in the same
order. The input gradient is one GEMM per residue class of the stride, over
only the kernel taps that reach that class's positions, which leaves out the
zeros a transposed convolution inserts. ``tests/test_qconv_gemm.py`` checks
all three passes against references built that way, and
``docs/source/kernels/convolutions.rst`` states the sums.

``conv_formats(mac)`` resolves a mac once, as ``matmul_formats`` does, and
returns a ``QAffineFormats`` whose three math hooks are the three passes,
with the keyword signatures ``CustomArithConvNd`` calls them with, so
``QConv1d``, ``QConv2d`` and ``QConv3d`` run them unchanged.
"""

from functools import lru_cache
from typing import Any

import torch

from mptorch.number import Number

from .mac import FusedMac, Mac, SplitMac, spec_for_mac
from .modules.format import QAffineFormats
from .ops import (
    _NARROW_STORAGE,
    _check_stored,
    _GemmSpec,
    _named_carrier_operands,
    _narrowed,
    _report_findings,
)

__all__ = ["conv_formats"]

# The three passes, as values of the conv op's `conv_pass` argument.
_FWD, _IGRAD, _WGRAD = 0, 1, 2

# The arguments a NAIVE single-format spec lacks for a conv op. The conv ops
# take the *_accumulated schema, which ends with a block size, an outer_quant
# flag and the outer format's fields; NAIVE has none of them, so these are
# placeholders with outer_quant=False. Both families' tails have nine entries,
# but the outer format's fields differ (outer_is_signed is the True).
_NAIVE_TAIL = {
    "binaryK": (0, False, 0, 0, 0, True, 0, 0, 0),
    "superfp": (0, False, 0, 0, 0, 0, True, 0, 0),
}


@lru_cache(maxsize=1)
def _conv_ops() -> dict[Any, tuple[Any, tuple[Any, ...]]]:
    """Map each GEMM op to its conv op and to the arguments a spec of it lacks there.

    The table is keyed by the GEMM op a ``_GemmSpec`` names, so a spec from
    ``spec_for_mac`` finds its conv op without being resolved again. A NAIVE
    single-format spec runs on the conv op of its ``*_accumulated`` twin, with
    the NAIVE tail (``_NAIVE_TAIL``) appended; an accumulated or palette spec
    runs on its conv op with its arguments unchanged.
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
            # The odd ops (carrier_rounding=RoundMode.RO), the same way: the
            # single-format odd conv op takes the accumulated schema, so a NAIVE
            # odd spec gets the NAIVE tail and an accumulated one runs as it is;
            # the palette odd conv op is NAIVE only and takes its GEMM twin's
            # arguments.
            odd_conv = getattr(ops, f"custom_conv_{family}{mac}_odd").default
            table[getattr(ops, f"custom_matmul_{family}{mac}_odd").default] = (
                odd_conv,
                _NAIVE_TAIL[family],
            )
            table[getattr(ops, f"custom_matmul_{family}{mac}_accumulated_odd").default] = (
                odd_conv,
                (),
            )
            table[getattr(ops, f"custom_matmul_{family}{mac}_mixed_odd").default] = (
                getattr(ops, f"custom_conv_{family}{mac}_mixed_odd").default,
                (),
            )
    # The cross-family ops, named multiply family first. Their outer format, and
    # so the NAIVE tail, is of the accumulate format's family.
    for mul, acc in (("binaryK", "superfp"), ("superfp", "binaryK")):
        conv = getattr(ops, f"custom_conv_{mul}_{acc}").default
        table[getattr(ops, f"custom_matmul_{mul}_{acc}").default] = (conv, _NAIVE_TAIL[acc])
        table[getattr(ops, f"custom_matmul_{mul}_{acc}_accumulated").default] = (conv, ())
        odd_conv = getattr(ops, f"custom_conv_{mul}_{acc}_odd").default
        table[getattr(ops, f"custom_matmul_{mul}_{acc}_odd").default] = (odd_conv, _NAIVE_TAIL[acc])
        table[getattr(ops, f"custom_matmul_{mul}_{acc}_accumulated_odd").default] = (odd_conv, ())
    return table


def _ntuple(value: Any, nd: int, name: str) -> tuple[int, ...]:
    """Return ``value`` as ``nd`` integers: an int is repeated, a sequence's length checked."""
    if isinstance(value, int):
        return (value,) * nd
    out = tuple(int(v) for v in value)
    if len(out) != nd:
        raise ValueError(f"{name} must have {nd} entries for a {nd}d convolution, got {out}")
    return out


def _leading_padding(
    padding: Any, kernel: tuple[int, ...], stride: tuple[int, ...], dilation: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the padding before each spatial dimension; the op is told only this side.

    ``"same"`` pads ``d*(k - 1)`` positions in all and, where that number is
    odd, puts the extra one after the input, as ``F.conv*d`` does. The op
    reads zeros past the end of the input without being told how many, so
    the uneven split needs no padded copy.
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
    """Return the forward pass's spatial output size, as ``F.conv*d`` computes it."""
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
    """One resolved conv op, which a pass calls; it is built once per layer.

    ``args`` are the spec's arguments followed by whatever more the conv op
    takes, and ``prec_idx`` is the pass's map for a palette op. What happens
    on each call is what happens on each call of a GEMM (``ops._run_gemm``):
    the carrier's findings and the storage check for the operands' dtype are
    reported, and, for a spec that names ``carrier=torch.float64``, narrower
    operands are widened and the result is narrowed back to their dtype.
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
        spec = self.spec
        narrow = None
        if spec.carrier is not None:
            a, b, narrow = _named_carrier_operands(spec, a, b)
        else:
            dtype = a.dtype
            found = spec.findings[dtype is torch.float64]
            if found:
                _report_findings(found)
            if spec.stored and dtype in _NARROW_STORAGE:
                _check_stored(spec.stored, dtype)
        geometry = (conv_pass, out_size, stride, padding, dilation, groups)
        if self.prec_idx is None:
            out = self.op(a, b, *geometry, *self.args)
        else:
            out = self.op(a, b, self.prec_idx, *geometry, *self.args)
        return out if narrow is None else _narrowed(out, narrow)


def _missing_map(pass_name: str, argument: str):
    """A hook for a palette pass whose map was not given; it raises when the pass runs."""

    def missing(*_args, **_kwargs) -> torch.Tensor:
        raise ValueError(
            f"this format is a palette, so its {pass_name} needs its own prec_idx map "
            f"(shaped like that pass's result): pass {argument}= to conv_formats"
        )

    return missing


def _batched(x: torch.Tensor, nd: int) -> tuple[torch.Tensor, bool]:
    """Return ``x`` with a batch dimension, and whether one had to be added."""
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
            # Add the bias in place, as the Linear hooks do: `out` is a fresh
            # tensor the op just made, so this equals `out + bias` bit for bit
            # without allocating a second output. A bias of a wider dtype is
            # added out of place instead, so that the result is promoted.
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


def _conv_formats(fwd: _GemmSpec, igrad: _GemmSpec, wgrad: _GemmSpec) -> QAffineFormats:
    """Build a ``QAffineFormats`` with the three convolution hooks, one resolved
    conv op per pass: the forward pass in ``fwd``'s arithmetic, the input
    gradient in ``igrad``'s and the weight gradient in ``wgrad``'s. The specs
    must be single-format, because a palette's passes need their maps, which
    only :func:`conv_formats` takes. A spec given for more than one pass is
    bound once."""
    runs = {spec: _ConvOp(spec, None) for spec in (fwd, igrad, wgrad)}
    return QAffineFormats(
        fwd_math=_fwd_hook(runs[fwd]),
        bwd_igrad_math=_igrad_hook(runs[igrad]),
        bwd_wgrad_math=_wgrad_hook(runs[wgrad]),
    )


def conv_formats(
    mac: Mac | Number,
    *,
    prec_idx: torch.Tensor | None = None,
    igrad_prec_idx: torch.Tensor | None = None,
    wgrad_prec_idx: torch.Tensor | None = None,
) -> QAffineFormats:
    """Build a ``QAffineFormats`` whose three convolution passes run in ``mac``'s arithmetic.

    The forward pass, the input gradient and the weight gradient of a
    ``QConv1d``, ``QConv2d`` or ``QConv3d`` each run in ``mac``'s arithmetic:
    every product and every addition of every dot product is rounded as a
    GEMM with that mac rounds it. The operands are gathered from the
    convolution's tensors inside the kernel instead of being unfolded, so no
    pass allocates more than its result.

    The forward pass and the weight gradient are each one GEMM, bit-identical
    to the GEMM with the same mac over ``F.unfold``'s operands.

    The input gradient is one GEMM per residue class of the stride, each over
    only the kernel taps that reach that class's positions. A strided
    convolution's input gradient therefore costs what its forward pass costs,
    where a transposed convolution would cost ``stride**nd`` times more,
    because it also sums the zeros it inserts. Under ``NAIVE`` summation and
    a deterministic rounding mode the two give bit-identical results, because
    adding a zero then changes nothing. :doc:`/kernels/convolutions` has the
    equations and says what differs in the other cases.

    Any stride, padding, dilation and ``groups`` work. Padding may be
    ``"valid"``, or ``"same"`` for a convolution without stride. All groups
    run in one kernel launch. A depthwise convolution (one channel per group)
    is a GEMM with one row (``M = 1``), so it fills only one row of the
    kernel's 16 x 16 tile, which is correct but not fast.

    Like :func:`matmul_formats`, this sets only the math hooks. The caller
    sets the operand quantizers (``weight_quant``, ``input_quant`` and the
    other ``*_quant`` slots) separately.

    The carrier is chosen as for a GEMM: float64 tensors round in binary64,
    and float32, float16 and bfloat16 tensors in binary32, unless the mac
    names ``carrier=torch.float64``. In that case narrower operands are
    widened, each pass runs the binary64 kernels, and its result is narrowed
    back to the operands' dtype, bit-identical to the pass on the widened
    operands. On MPS, which has no float64, binary32 is the only carrier.

    Args:
        mac (SplitMac or FusedMac or Number): the dot-product arithmetic,
            resolved once through :func:`mptorch.quant.mac.spec_for_mac`. A
            bare ``Number`` means ``SplitMac(fmt, fmt)``. Every
            ``accumulate_algorithm`` applies, and a ``SplitMac`` may round
            its products and its sums to formats of two different families,
            as for a GEMM. A palette selects the ops with a format per output
            element, which need a map for each pass. A mac with
            ``carrier_rounding=RoundMode.RO`` runs the odd conv ops in every
            pass, under every algorithm its GEMM takes (``KAHAN``, ``BLOCK``
            and ``TREE`` need a build with ``MPTORCH_ODD_ACCUMULATED=1``).
        prec_idx (Tensor, optional): the forward pass's palette map, indexed
            like the output with its spatial dimensions flattened:
            ``[Cout, OUT]`` with ``OUT`` the number of output positions,
            ``[Cout, 1]`` (one format per output channel) or ``[1, OUT]``,
            optionally with a leading batch dimension. A palette ``mac``
            requires it, and any other ``mac`` refuses it. Default: ``None``
        igrad_prec_idx (Tensor, optional): the input gradient's map, shaped
            the same way over ``[Cin, IN]``, with ``IN`` the number of input
            positions. Each residue-class GEMM of a strided input gradient
            reads the map at its own input positions. Default: ``None``
        wgrad_prec_idx (Tensor, optional): the weight gradient's map, over the
            weight as ``[Cout, Cin/groups * prod(kernel)]``, or ``[Cout, 1]``.
            Default: ``None``

    Returns:
        QAffineFormats: with ``fwd_math``, ``bwd_igrad_math`` and
        ``bwd_wgrad_math`` set and every ``*_quant`` slot left ``None``.

    Raises:
        ValueError: for a palette ``mac`` without ``prec_idx``, or a map given
            for a ``mac`` without a palette. A palette ``mac`` without
            ``igrad_prec_idx`` or ``wgrad_prec_idx`` raises when that
            gradient is computed.
        TypeError: if ``mac`` is not a ``SplitMac``, a ``FusedMac`` or a
            ``Number``.

    Example:

        Binary8p4 products summed in Binary12p7, with the operands also
        rounded to Binary8p4, for a strided layer. The backward pass computes
        both gradients in the same arithmetic::

            >>> import torch
            >>> from mptorch import AccumulateAlgorithm, BinaryK, RoundMode
            >>> from mptorch.quant import FusedMac, QConv1d, QConv2d, Quant, SplitMac
            >>> from mptorch.quant import conv_formats
            >>> binary8p4 = BinaryK(8, 4)
            >>> formats = conv_formats(SplitMac(binary8p4, BinaryK(12, 7)))
            >>> formats.input_quant = formats.weight_quant = Quant(binary8p4)
            >>> conv = QConv2d(3, 8, 3, stride=2, padding=1, formats=formats)
            >>> x = torch.randn(2, 3, 16, 16, requires_grad=True)
            >>> conv(x).sum().backward()
            >>> x.grad.shape, conv.weight.grad.shape
            (torch.Size([2, 3, 16, 16]), torch.Size([8, 3, 3, 3]))

        Any mac: a fused multiply-add with Kahan summation under stochastic
        rounding, on a grouped, dilated 1D convolution::

            >>> fma = FusedMac(BinaryK(12, 7, prng_bits=8), rounding=RoundMode.SR,
            ...                accumulate_algorithm=AccumulateAlgorithm.KAHAN)
            >>> conv1d = QConv1d(4, 8, 3, dilation=2, groups=2, formats=conv_formats(fma))
            >>> conv1d(torch.randn(1, 4, 20)).shape
            torch.Size([1, 8, 16])

        A palette, here with one format per output channel of the forward
        pass (a ``[Cout, 1]`` map), and a map for each gradient (here all
        zeros, so every element uses entry 0)::

            >>> pal = SplitMac([binary8p4, BinaryK(8, 5)], BinaryK(12, 7))
            >>> per_channel = conv_formats(
            ...     pal,
            ...     prec_idx=torch.tensor([[0], [1]] * 4),
            ...     igrad_prec_idx=torch.zeros(3, 1, dtype=torch.int64),
            ...     wgrad_prec_idx=torch.zeros(8, 1, dtype=torch.int64),
            ... )
            >>> QConv2d(3, 8, 3, padding=1, formats=per_channel)(torch.randn(1, 3, 6, 6)).shape
            torch.Size([1, 8, 6, 6])
    """
    if isinstance(mac, Number):
        mac = SplitMac(mac, mac)
    if not isinstance(mac, SplitMac | FusedMac):
        raise TypeError(f"mac must be a SplitMac, a FusedMac or a Number, got {type(mac).__name__}")
    spec = spec_for_mac(mac)
    maps = (prec_idx, igrad_prec_idx, wgrad_prec_idx)
    if spec.mixed and prec_idx is None:
        raise ValueError(
            "a palette format selects its entry per output element, so it needs a "
            "prec_idx map; pass one, or give a single format per slot"
        )
    if not spec.mixed and any(m is not None for m in maps):
        raise ValueError("prec_idx has no meaning without a palette format to select from")
    if not spec.mixed:
        return _conv_formats(spec, spec, spec)
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
