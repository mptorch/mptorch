"""The differentiable matrix products: ``qmm``, ``qbmm`` and ``qmatmul``.

These are the functions to use in a model. The ten ``*_matmul*`` functions in
``mptorch.quant.ops`` are the low-level tier: one function per kernel op,
every op argument spelled out, and no gradient. The functions here sit above
them: they take a format object instead of a long list of integer arguments,
and they are differentiable.

``formats`` accepts five kinds of value, from the simplest to the most
complete:

* ``None``: plain ``torch.matmul``, useful as an unquantized baseline;
* a ``Number``: for a ``BinaryK`` or ``SuperFP`` such as ``BinaryK(8, 4)``,
  shorthand for ``SplitMac(fmt, fmt)``, the common case where one format
  rounds both the products and the running sum; for a ``BlockFormat``,
  shorthand for ``BlockMac(fmt)``, which packs both operands in it and leaves
  the sum unrounded in the carrier;
* a ``BlockMac``: the block formats of two packed operands and the
  arithmetic of their product (:func:`mptorch.quant.block_matmul_formats`);
* a ``SplitMac`` or ``FusedMac``: the dot-product arithmetic. A ``Palette``
  in any of its slots selects a format per output element and requires
  ``prec_idx``, and its ``carrier`` chooses the float arithmetic the product
  is computed in (binary64 for any operands with ``carrier=torch.float64``;
  never narrower than the operands);
* a ``QMatmulFormats``: the arithmetic of each pass *and* the operand
  quantizers, that is, everything a layer can vary.

Every kind but the last is turned into a ``QMatmulFormats`` (through a cache,
for a ``Number`` or a mac), so there is one implementation and one thing to
test.
"""

from functools import lru_cache
from typing import Any

import torch

from mptorch.number import BlockFormat, Number

from ..factories.matmul import block_matmul_formats, matmul_formats
from ..mac import BlockMac, FusedMac, Mac, SplitMac
from ..modules.format import QMatmulFormats
from ..modules.matmul import CustomArithMatmul

__all__ = ["qmm", "qbmm", "qmatmul", "as_matmul_formats"]


@lru_cache(maxsize=128)
def _formats_for(spelling: Number | Mac | BlockMac, prec_idx: Any = None) -> QMatmulFormats:
    """Return the ``QMatmulFormats`` for a format object and ``prec_idx``, built once.

    Cached for the same reason ``spec_for_mac`` is: a call such as
    ``qmatmul(a, b, BinaryK(8, 4))`` in a loop must not pay 2.3 to 6.4
    microseconds to resolve the format again on every call. A bare ``Number``
    is expanded *inside* the cache, so that it costs one lookup instead of
    building a new ``SplitMac`` (and the two ``Palette`` objects it holds) on
    every call. ``prec_idx`` is part of the key by identity, since a tensor
    cannot be hashed by value, and that is the right key: a map rebuilt on
    every call misses both this cache and the C++ cache of maps whose bounds
    were already checked (``csrc/common/gemm_host.h``), while a map kept and
    reused hits both.
    """
    if isinstance(spelling, BlockFormat):
        spelling = BlockMac(spelling)
    elif isinstance(spelling, Number):
        spelling = SplitMac(spelling, spelling)
    if isinstance(spelling, BlockMac):
        if prec_idx is not None:
            raise ValueError("prec_idx selects a palette entry, and a block format has none")
        return block_matmul_formats(spelling)
    assert isinstance(spelling, SplitMac | FusedMac)
    return matmul_formats(spelling, prec_idx=prec_idx)


def as_matmul_formats(formats: Any, prec_idx: torch.Tensor | None = None) -> QMatmulFormats:
    """Convert any accepted ``formats`` value to a ``QMatmulFormats``.

    This is the only place that tells the five kinds of value in the module
    docstring apart, so :func:`~mptorch.quant.qmatmul` and :class:`mptorch.quant.QMatmul`
    accept exactly the same values.

    Args:
        formats (None, Number, SplitMac, FusedMac, BlockMac or QMatmulFormats):
            the arithmetic, as any of the accepted kinds.
        prec_idx (Tensor, optional): the forward pass's palette map, for a
            ``SplitMac`` or ``FusedMac`` that holds a palette. Default: ``None``

    Returns:
        QMatmulFormats: ``formats`` itself if it already is one, an empty
        ``QMatmulFormats()`` for ``None``, and otherwise the cached one built
        from the format or mac.

    Raises:
        ValueError: for a ``prec_idx`` given with ``None`` or a block format
            (which have no palette to select from), or with a
            ``QMatmulFormats`` (whose maps were already bound by
            :func:`mptorch.quant.matmul_formats`).
        TypeError: for a ``formats`` value of any other type.
    """
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
    if not isinstance(formats, Number | SplitMac | FusedMac | BlockMac):
        raise TypeError(
            "formats must be None, a Number, a SplitMac/FusedMac/BlockMac or a QMatmulFormats, "
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
    """``torch.matmul``, computed in the arithmetic that ``formats`` describes.

    It accepts every operand shape ``torch.matmul`` accepts: a 1D operand on
    either side is promoted to a matrix, and leading (batch) dimensions of any
    rank broadcast against each other. It is differentiable in both operands,
    and each gradient is computed in the arithmetic of its own hook. Float64
    operands are computed in binary64, and all others in binary32, unless a
    mac's ``carrier`` asks for binary64. Either way, each pass checks the
    formats against its carrier and warns (:class:`mptorch.FormatRangeWarning`)
    about any part of their range that the carrier cannot hold.

    Two operand shapes need no copy:

    * An operand that is the transpose of a contiguous tensor (``q @ k.mT``)
      sets the kernel's transpose flag instead of being copied.
    * ``[..., M, K] @ [K, N]`` merges its batch dimensions into ``M``, giving
      one large GEMM instead of a batch of small ones.

    Both give exactly the same bits as the batched computation, stochastic
    rounding included, because each output element draws its random bits from
    a stream selected by its index in the output.

    Args:
        a (Tensor): the left operand, 1D or higher.
        b (Tensor): the right operand, 1D or higher, of the same dtype.
        formats (None, Number, SplitMac, FusedMac, BlockMac or QMatmulFormats):
            the arithmetic. The module docstring says what each kind of value
            means. Default: ``None`` (plain ``torch.matmul``)
        prec_idx (Tensor, optional): an integer map shaped like the forward
            pass's output (``[M, N]``, ``[M, 1]`` or ``[1, N]``, optionally
            batched) that selects a palette entry for each output element. A
            palette format requires it, and every other format rejects it. It
            is the *forward pass's* map only: the gradient passes have
            outputs of other shapes and need maps of their own, which the
            ``agrad_prec_idx`` and ``bgrad_prec_idx`` arguments of
            :func:`mptorch.quant.matmul_formats` provide. Default: ``None``

    Returns:
        Tensor: ``a @ b`` in the operands' dtype, with ``torch.matmul``'s
        output shape.

    Raises:
        ValueError: for a palette without ``prec_idx``, a ``prec_idx``
            without a palette, or a ``carrier`` narrower than the operands.
        TypeError: for a ``formats`` value of any other type.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import SplitMac, qmatmul
        >>> a, b = torch.randn(5, 2, 4, 8), torch.randn(8, 3)
        >>> qmatmul(a, b, BinaryK(8, 4)).shape
        torch.Size([5, 2, 4, 3])
        >>> qmatmul(a, b, SplitMac(BinaryK(8, 4), acc=None)).shape
        torch.Size([5, 2, 4, 3])
    """
    return CustomArithMatmul.apply(a, b, as_matmul_formats(formats, prec_idx))


def qmm(
    a: torch.Tensor,
    b: torch.Tensor,
    formats: Any = None,
    *,
    prec_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`qmatmul` restricted to two matrices, as ``torch.mm`` is.

    Args:
        a (Tensor): a ``[M, K]`` matrix.
        b (Tensor): a ``[K, N]`` matrix of the same dtype.
        formats: as for :func:`qmatmul`. Default: ``None``
        prec_idx (Tensor, optional): as for :func:`qmatmul`. Default: ``None``

    Returns:
        Tensor: the ``[M, N]`` product.

    Raises:
        ValueError: if either operand is not 2D, plus :func:`qmatmul`'s.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import qmm
        >>> qmm(torch.randn(4, 8), torch.randn(8, 3), BinaryK(8, 4)).shape
        torch.Size([4, 3])
    """
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

    Unlike :func:`qmatmul`, it does not broadcast: the two batch sizes must be
    equal, as ``torch.bmm`` requires.

    Args:
        a (Tensor): a ``[B, M, K]`` batch of matrices.
        b (Tensor): a ``[B, K, N]`` batch of the same dtype.
        formats: as for :func:`qmatmul`. Default: ``None``
        prec_idx (Tensor, optional): as for :func:`qmatmul`. Default: ``None``

    Returns:
        Tensor: the ``[B, M, N]`` batch of products.

    Raises:
        ValueError: if either operand is not 3D or the batch sizes differ,
            plus :func:`qmatmul`'s.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import FusedMac, qbmm
        >>> a, b = torch.randn(2, 4, 8), torch.randn(2, 8, 3)
        >>> qbmm(a, b, FusedMac(BinaryK(8, 4))).shape
        torch.Size([2, 4, 3])
    """
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError(f"qbmm expects 3D tensors, got {a.dim()}D and {b.dim()}D")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"qbmm expects equal batch dimensions, got {a.shape[0]} and {b.shape[0]}")
    return qmatmul(a, b, formats, prec_idx=prec_idx)
