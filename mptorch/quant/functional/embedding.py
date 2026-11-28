"""An embedding whose rows round to a simulated format: ``qembedding``.

A gather, rounded as a :class:`~mptorch.quant.PointwiseFormats` says:
``formats.quant`` rounds the rows it gathers, in the carrier, and
``formats.grad_quant`` the weight's gradient, after the sum over repeated
indices, as :mod:`mptorch.quant.functional.pointwise`'s ops round theirs.
"""

import torch
import torch.nn.functional as F

from .._opaque import _opaque_backward
from ..ops._spec import _round_result
from .pointwise import _RO, PointwiseFormats, _carrier_dtype

__all__ = ["qembedding"]


# --- the embedding ---------------------------------------------------------------


def _padding(padding_idx: int | None, num_weights: int) -> int:
    """``padding_idx`` as ``F.embedding`` hands it to the op: from the end
    when negative, and -1 for none."""
    if padding_idx is None:
        return -1
    if padding_idx < 0:
        return num_weights + padding_idx
    return padding_idx


class _QEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, formats, padding_idx, max_norm, norm_type, by_freq, sparse):
        with torch._C.DisableTorchFunction():
            rows = F.embedding(input, weight, padding_idx, max_norm, norm_type, by_freq, sparse)
            ctx.save_for_backward(input)
            n = weight.shape[0]
            ctx.embedding = (formats, n, _padding(padding_idx, n), by_freq, sparse)
            q = formats.quant
            if q is None:
                return rows
            return _round_result(
                q, rows.to(_carrier_dtype(formats.carrier, rows.dtype)), rows.dtype
            )

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            g = grads[0]
            (input,) = ctx.saved_tensors
            formats, n, padding_idx, by_freq, sparse = ctx.embedding
            q = formats.grad_quant
            if q is None:
                gw = torch.ops.aten.embedding_backward(g, input, n, padding_idx, by_freq, sparse)
            else:
                gc = g.to(_carrier_dtype(formats.carrier, g.dtype))
                gw = torch.ops.aten.embedding_backward(gc, input, n, padding_idx, by_freq, sparse)
                gw = _round_result(q, gw, g.dtype)
            return None, gw, None, None, None, None, None, None


def qembedding(
    input: torch.Tensor,
    weight: torch.Tensor,
    formats: PointwiseFormats | None = None,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """``F.embedding``, its rows rounded by ``formats.quant`` and its weight's
    gradient by ``formats.grad_quant``.

    The gather is exact, so the rows are the format's rounding of the
    weight's. The weight's gradient is torch's (``embedding_backward``),
    summed over repeated indices in the carrier, then rounded once. The other
    arguments are ``F.embedding``'s.

    Raises:
        ValueError: for ``carrier_rounding=RoundMode.RO`` (a gather has no
            carrier step to round), or a sparse gradient with
            ``grad_quant`` set.
    """
    if formats is not None and not isinstance(formats, PointwiseFormats):
        raise TypeError(f"qembedding's formats is a PointwiseFormats or None, got {formats!r}")
    if formats is None or (formats.quant is None and formats.grad_quant is None):
        with torch._C.DisableTorchFunction():
            return F.embedding(
                input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
            )
    if formats.carrier_rounding is _RO:
        raise ValueError(
            "carrier_rounding=RoundMode.RO: an embedding gathers its rows, which leaves no "
            "carrier step to round to odd -- use RoundMode.RNE"
        )
    if sparse and formats.grad_quant is not None:
        raise ValueError(
            "qembedding: a sparse gradient cannot be rounded by grad_quant -- pass sparse=False, "
            "or leave grad_quant unset"
        )
    return _QEmbedding.apply(
        input, weight, formats, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    )
