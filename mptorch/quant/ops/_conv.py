"""The conv ops: which op runs each pass of a convolution, and the call itself.

Every GEMM op except the block GEMM has a conv op (``custom_conv_*``) that
runs one pass of a convolution (the forward pass or one of the two gradients)
in the GEMM op's arithmetic, gathering the operands from the convolution's
tensors as it loads the kernel's tiles (``csrc/common/gemm_gather.h``). A
NAIVE single-format op and its ``*_accumulated`` twin share one conv op,
which takes the accumulated schema. A convolution layer
resolves its mac to a `_GemmSpec` exactly as a GEMM does
(`mptorch.quant.mac.spec_for_mac`). `_conv_ops` maps that spec's GEMM op to
its conv op and to the arguments the conv op takes beyond the GEMM op's, and
`_ConvOp` holds the result, built once per layer, and makes the call. The
conv ops have no flat function of their own: `mptorch.quant.conv_formats`
builds the layer hooks that call them.
"""

from functools import lru_cache
from typing import Any

import torch

from ._checks import _NARROW_STORAGE, _check_stored, _narrowed, _report_findings
from ._spec import _GemmSpec, _named_carrier_operands

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


class _ConvOp:
    """One resolved conv op, which a pass calls; it is built once per layer.

    ``args`` are the spec's arguments followed by whatever more the conv op
    takes, and ``prec_idx`` is the pass's map for a palette op. What happens
    on each call is what happens on each call of a GEMM (``_spec._run_gemm``):
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
