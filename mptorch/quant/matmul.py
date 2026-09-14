"""The differentiable matmul entry points: ``qmm``, ``qbmm`` and ``qmatmul``.

These are the functions to reach for. The eight ``*_matmul*`` wrappers in
``mptorch.quant.ops`` stay as the schema-faithful tier -- one function per op,
every schema argument spelled out, no autograd -- and these sit above them,
taking a format object instead of twenty integers and carrying a gradient.

``formats`` accepts, in increasing order of specificity:

* ``None`` -- plain ``torch.matmul``, which is what an A/B against
  unquantized arithmetic wants;
* a ``Number`` (``BinaryK(8, 4)``) -- shorthand for
  ``SplitMac(mul=fmt, acc=fmt)``, the common case where one format is used
  for both halves of the dot product;
* a ``SplitMac`` / ``FusedMac`` -- the dot-product arithmetic, with a palette
  in any slot selecting the spatially-varying op and requiring ``prec_idx``,
  and a ``carrier`` choosing the float arithmetic it is computed in (binary64
  for any operands with ``torch.float64``, never narrower than theirs);
* a ``QMatmulFormats`` -- per-pass arithmetic *and* operand quantizers, i.e.
  everything the layer can vary.

The first three are resolved (and memoized) into the fourth, so there is one
implementation and one thing to test.
"""

from functools import lru_cache
from typing import Any

import torch

from mptorch.number import Number

from .gemm import matmul_formats
from .mac import FusedMac, Mac, SplitMac
from .modules.format import QMatmulFormats
from .modules.matmul import CustomArithMatmul

__all__ = ["qmm", "qbmm", "qmatmul", "as_matmul_formats"]


@lru_cache(maxsize=128)
def _formats_for(spelling: "Number | Mac", prec_idx: Any = None) -> QMatmulFormats:
    """One ``QMatmulFormats`` per (format spelling, prec_idx), built once.

    Memoized for the same reason ``spec_for_mac`` is: an ad-hoc
    ``qmatmul(a, b, BinaryK(8, 4))`` in a loop must not pay P4's 2.3-6.4 us of
    re-resolution per call. The bare-``Number`` shorthand is expanded *inside*
    the cache rather than outside it, so that spelling costs one lookup on the
    format rather than a fresh ``SplitMac`` (and the two ``Palette`` objects it wraps) per
    call. ``prec_idx`` participates in the key by identity (it is a tensor, so
    it is not hashable by value), which is exactly right: a map rebuilt per
    call defeats this cache *and* the C++ bounds-check memo (finding G5b),
    and a map held and reused hits both.
    """
    if isinstance(spelling, Number):
        spelling = SplitMac(spelling, spelling)
    return matmul_formats(spelling, prec_idx=prec_idx)


def as_matmul_formats(formats: Any, prec_idx: torch.Tensor | None = None) -> QMatmulFormats:
    """Whatever a caller passed as ``formats``, as a ``QMatmulFormats``."""
    if formats is None:
        if prec_idx is not None:
            raise ValueError("prec_idx has no meaning without a format to select")
        return QMatmulFormats()
    if isinstance(formats, QMatmulFormats):
        if prec_idx is not None:
            raise ValueError(
                "a QMatmulFormats already carries its math hooks: give prec_idx to the "
                "factory that built it (mptorch.quant.matmul_formats) rather than to the call"
            )
        return formats
    if not isinstance(formats, Number | SplitMac | FusedMac):
        raise TypeError(
            "formats must be None, a Number, a SplitMac/FusedMac or a QMatmulFormats, "
            f"got {type(formats).__name__}"
        )
    return _formats_for(formats, prec_idx)


def qmatmul(
    a: torch.Tensor,
    b: torch.Tensor,
    formats: Any = None,
    *,
    prec_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """``torch.matmul``'s contract, computed in the arithmetic ``formats`` names.

    Accepts everything ``torch.matmul`` accepts -- 1D promotion on either
    operand, leading dimensions of any rank broadcast against each other --
    and is differentiable in both operands, with the gradient of each computed
    in the arithmetic its own hook names. float64 operands are computed in
    binary64, and the rest in binary32 unless a mac's ``carrier`` asks for
    binary64; either way each pass holds the formats to its carrier and warns
    (:class:`mptorch.FormatRangeWarning`) about what it cannot hold.

    Two shapes cost no copy: an operand that is the transpose of a contiguous
    tensor (``q @ k.mT``) sets the kernel's flag instead of being
    materialized, and ``[..., M, K] @ [K, N]`` folds its batch into ``M`` for
    one large GEMM rather than a batch of small ones.

    ``prec_idx`` is required by, and only by, a palette format -- an integer
    map of the forward's output shape (``[M, N]``, ``[M, 1]``, ``[1, N]``,
    optionally batched) selecting a palette entry per output element. It is
    the *forward's* map; a palette that also has to differentiate needs one
    per pass, which is what :func:`mptorch.quant.matmul_formats`'
    ``agrad_prec_idx``/``bgrad_prec_idx`` are for.
    """
    return CustomArithMatmul.apply(a, b, as_matmul_formats(formats, prec_idx))


def qmm(
    a: torch.Tensor,
    b: torch.Tensor,
    formats: Any = None,
    *,
    prec_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`qmatmul` restricted to two matrices, as ``torch.mm`` is."""
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"qmm expects 2D tensors, got {a.dim()}D and {b.dim()}D")
    return qmatmul(a, b, formats, prec_idx=prec_idx)


def qbmm(
    a: torch.Tensor,
    b: torch.Tensor,
    formats: Any = None,
    *,
    prec_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`qmatmul` restricted to two batches of matrices, as ``torch.bmm`` is.

    No broadcasting: both batch dimensions must be equal, which is the check
    ``torch.bmm`` makes and ``qmatmul`` does not.
    """
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError(f"qbmm expects 3D tensors, got {a.dim()}D and {b.dim()}D")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"qbmm expects equal batch dimensions, got {a.shape[0]} and {b.shape[0]}")
    return qmatmul(a, b, formats, prec_idx=prec_idx)
