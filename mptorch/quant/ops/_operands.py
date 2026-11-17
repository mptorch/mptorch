"""torch.matmul's operand rules, mapped onto the rank-2 and rank-3 ops.

The ops take operands of rank 2 or rank 3 only: at most one batch dimension,
which each operand either has or shares across the batch, read at batch
stride 0 (`common/gemm_host.h`). Everything else that `torch.matmul` accepts
(a 1D operand, any number of leading dimensions, broadcasting between them)
is turned into that form here, with views wherever `torch.matmul` itself
would use one, so that the two copy data in the same cases.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import NamedTuple

import torch

from ._spec import _GemmSpec, _run_gemm, _run_gemm_mixed


class _MatmulLayout(NamedTuple):
    """The operands and flags the op receives, and the shape to return its result in.

    ``out_shape`` already omits the dimension added to a 1D operand, so the
    caller only has to reshape the op's ``[B, M, N]`` (or ``[M, N]``) result
    to it.
    """

    a: torch.Tensor
    b: torch.Tensor
    trans_a: bool
    trans_b: bool
    out_shape: tuple[int, ...]


def _collapse(x: torch.Tensor) -> torch.Tensor:
    """Merge every leading dimension of an operand of rank 3 or more into one,
    without a copy.

    ``reshape`` alone would copy a transposed view, which is the layout
    :func:`_packed` passes on without a copy, so a transposed view is reshaped
    in its untransposed form and then transposed back.
    """
    # The batch size is computed rather than left to `reshape(-1, r, c)`, which
    # is ambiguous for an empty tensor (a zero batch or a zero M).
    batch = 1
    for d in x.shape[:-2]:
        batch *= d
    if x.is_contiguous() or not x.transpose(-2, -1).is_contiguous():
        return x.reshape(batch, *x.shape[-2:])
    xt = x.transpose(-2, -1)
    return xt.reshape(batch, *xt.shape[-2:]).transpose(-2, -1)


def _operand_3d(x: torch.Tensor, batch: tuple[int, ...]) -> torch.Tensor:
    """Return one operand as the rank-2 or rank-3 tensor the op takes.

    An operand with no leading dimensions, or only leading dimensions of size
    1, is returned as a matrix, which the op reads at batch stride 0: shared
    across the batch, never expanded. Leading dimensions that already equal
    the broadcast batch are merged into one, as a view. Anything else is a
    partial broadcast (``[4, 1, M, K] @ [1, 3, K, N]``), and expanding it
    copies exactly where ``torch.matmul`` itself copies.
    """
    lead = x.shape[:-2]
    if all(d == 1 for d in lead):
        while x.dim() > 2:  # squeeze(0) returns a view whatever the layout
            x = x.squeeze(0)
        return x
    if tuple(lead) != tuple(batch):
        x = x.expand(*batch, *x.shape[-2:])
    return _collapse(x)


def _broadcast_batch(a_lead: Sequence[int], b_lead: Sequence[int]) -> tuple[int, ...]:
    """Return the leading dimensions that two operands broadcast to.

    This is ``torch.broadcast_shapes`` written out by hand, because that
    function costs 8.7 us, most of a whole resolved GEMM call, for two tuples
    that are usually equal and always short. The rule is the same and this
    takes about 0.3 us, but a mismatch raises a ``ValueError`` of its own
    where torch raises ``RuntimeError``.
    """
    if a_lead == b_lead:
        return tuple(a_lead)
    n = max(len(a_lead), len(b_lead))
    out = []
    for i in range(-n, 0):
        x = a_lead[i] if -i <= len(a_lead) else 1
        y = b_lead[i] if -i <= len(b_lead) else 1
        if x != y and x != 1 and y != 1:
            raise ValueError(
                f"matmul operands' leading dimensions do not broadcast: {tuple(a_lead)} "
                f"against {tuple(b_lead)}"
            )
        out.append(x if y == 1 else y)
    return tuple(out)


def _matmul_operands(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    fold: bool = True,
) -> _MatmulLayout:
    """Apply ``torch.matmul``'s operand rules, putting the operands in the
    rank-2 or rank-3 form the op takes.

    Return the operands as the op takes them, the transpose flags to read them
    with, and the shape to return the result in. ``trans_a`` and ``trans_b``
    apply to the last two dimensions, as they would to a ``bmm`` of
    transposed views; they are ignored for a 1D operand, which
    ``torch.matmul`` cannot transpose.

    ``fold=False`` keeps a batch as a batch instead of folding it into ``M``.
    Only a palette op needs that: its ``prec_idx`` is indexed by output
    element, so a map shaped for the unfolded ``[M, N]`` output does not fit
    the folded ``[B*M, N]`` one.
    """
    a_1d, b_1d = a.dim() == 1, b.dim() == 1
    if a_1d:
        a, trans_a = a.unsqueeze(0), False
    if b_1d:
        b, trans_b = b.unsqueeze(-1), False
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("matmul operands must have at least one dimension")

    batch = _broadcast_batch(a.shape[:-2], b.shape[:-2])
    M = a.shape[-1] if trans_a else a.shape[-2]
    N = b.shape[-2] if trans_b else b.shape[-1]
    out_shape = (*batch, *((() if a_1d else (M,)) + (() if b_1d else (N,))))

    a3 = _operand_3d(a, batch)
    b3 = _operand_3d(b, batch)

    # `[..., M, K] @ [K, N]`, with `b` shared across the batch and `a`
    # contiguous and not transposed: folding the batch into M is a view, and
    # gives one large GEMM instead of B small ones. The result is bit-identical
    # to the batched one, not merely equivalent, because stochastic rounding
    # selects each element's random stream by its index into the dense
    # [batch, M, N] output: `(b*M + row)*N + col` batched and `row'*N + col`
    # folded, with `row' = b*M + row`, are the same index. So the fold is safe
    # in every rounding mode. This is the QLinear case, which `gemm.py` folds
    # the same way.
    if fold and a3.dim() == 3 and b3.dim() == 2 and not trans_a and a3.is_contiguous():
        a3 = a3.reshape(-1, a3.shape[-1])

    return _MatmulLayout(a3, b3, trans_a, trans_b, out_shape)


def _gemm_nd(
    spec: _GemmSpec, a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> torch.Tensor:
    """:func:`_run_gemm` with ``torch.matmul``'s operand rules.

    The flat wrappers call this; the layer hooks in `mptorch.quant.gemm` call
    :func:`_run_gemm` directly, on operands they have already flattened.
    """
    return _matmul_nd(partial(_run_gemm, spec), a, b, trans_a, trans_b)


def _gemm_mixed_nd(
    spec: _GemmSpec,
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    """:func:`_gemm_nd` for the palette ops.

    The map's leading dimensions are merged the same way as the operands', so
    a map with one entry per batch element still lines up with the output of
    a call of rank 4 or more. The batch is never folded into ``M``, because
    the map is indexed by output element, and a map of shape ``[M, N]``
    shared across the batch does not fit a folded ``[B*M, N]`` output.
    """
    if prec_idx.dim() > 3:
        prec_idx = prec_idx.reshape(-1, *prec_idx.shape[-2:])

    def run(a3, b3, ta, tb):
        return _run_gemm_mixed(spec, a3, b3, prec_idx, ta, tb)

    return _matmul_nd(run, a, b, trans_a, trans_b, fold=False)


def _matmul_nd(
    run: Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    fold: bool = True,
) -> torch.Tensor:
    """Call ``run``, a resolved rank-2 or rank-3 GEMM, with ``torch.matmul``'s
    operand rules.

    Two cases go straight to the op, and they are the ones every layer uses:
    two matrices, and two batches of the same size. Both are already in the
    form the op takes, and the general path (:func:`_matmul_operands`, mostly
    its shape broadcasting) would spend about 12 us of Python returning the
    operands unchanged, several times a resolved call's whole overhead.
    """
    if a.dim() == 2 and b.dim() == 2:
        return run(a, b, trans_a, trans_b)
    if a.dim() == 3 and b.dim() == 3 and a.shape[0] == b.shape[0]:
        return run(a, b, trans_a, trans_b)
    layout = _matmul_operands(a, b, trans_a, trans_b, fold)
    out = run(layout.a, layout.b, layout.trans_a, layout.trans_b)
    return out if out.shape == layout.out_shape else out.reshape(layout.out_shape)
