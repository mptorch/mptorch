"""Factories that make a ``QMatmul``'s products run in quantized GEMMs.

:func:`matmul_formats` and :func:`block_matmul_formats` each return a
:class:`~mptorch.quant.QMatmulFormats` whose hooks (``_hooks.py``) follow
``torch.matmul``'s operand rules. Like ``linear.py``'s factories they
resolve their formats once and set only the math hooks, leaving the operand
quantizers to the caller, except where a block format needs an operand
packed once (:func:`block_matmul_formats`).
"""

from functools import partial

import torch

from ..mac import BlockMac, Mac, cast_quant, spec_for_mac
from ..modules.format import QMatmulFormats
from ..ops._operands import _gemm_nd
from ._hooks import (
    _block_matmul_agrad,
    _block_matmul_bgrad,
    _block_matmul_fwd,
    _matmul_formats,
    _matmul_fwd,
    _mixed_matmul,
    _spec_matmul,
)

__all__ = [
    "matmul_formats",
    "block_matmul_formats",
]


def matmul_formats(
    mac: Mac,
    *,
    prec_idx: torch.Tensor | None = None,
    agrad_prec_idx: torch.Tensor | None = None,
    bgrad_prec_idx: torch.Tensor | None = None,
) -> QMatmulFormats:
    """Build a ``QMatmulFormats`` whose three passes run in ``mac``'s arithmetic.

    The forward product ``a @ b``, ``a``'s gradient ``grad @ b^T`` and ``b``'s
    gradient ``a^T @ grad`` each run through the GEMM op that ``mac`` resolves
    to.

    A ``Palette`` in any of ``mac``'s slots selects a format per output
    element, which needs one ``prec_idx`` map per pass. The maps are bound
    into the hooks here, so they need not be passed on every call. A gradient
    pass whose map is missing raises when it runs; it does not fall back to
    unquantized arithmetic.

    Like the six binaryK and superfp ``*_gemm_formats`` factories, this sets
    only the math hooks. The caller adds the elementwise operand quantizers
    (``a_quant``, ``b_quant``, ``agrad_quant``, ``bgrad_quant``) separately.

    Args:
        mac (SplitMac or FusedMac): the dot-product arithmetic, resolved once
            through ``mptorch.quant.mac.spec_for_mac``.
        prec_idx (Tensor, optional): integer map selecting a palette entry for
            each element of the forward pass's output, shaped like ``a @ b``
            (``[M, N]``, ``[M, 1]`` or ``[1, N]``, optionally batched). A
            palette ``mac`` requires it, and any other ``mac`` rejects it.
            Default: ``None``
        agrad_prec_idx (Tensor, optional): the map for ``a``'s gradient,
            shaped like ``grad @ b^T``. Default: ``None``
        bgrad_prec_idx (Tensor, optional): the map for ``b``'s gradient,
            shaped like ``a^T @ grad``. Default: ``None``

    Returns:
        QMatmulFormats: with ``fwd_math``, ``bwd_agrad_math`` and
        ``bwd_bgrad_math`` set and every ``*_quant`` slot left ``None``.

    Raises:
        ValueError: if ``mac`` holds a palette and ``prec_idx`` is ``None``,
            or if any map is given for a ``mac`` with no palette to select
            from.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import Palette, QMatmul, SplitMac, matmul_formats
        >>> palette = Palette([BinaryK(8, 4), BinaryK(8, 5)])
        >>> prec_idx = torch.zeros(4, 3, dtype=torch.int64)
        >>> formats = matmul_formats(SplitMac(palette, palette), prec_idx=prec_idx)
        >>> QMatmul(formats)(torch.randn(4, 8), torch.randn(8, 3)).shape
        torch.Size([4, 3])
    """
    spec = spec_for_mac(mac)
    maps = (prec_idx, agrad_prec_idx, bgrad_prec_idx)
    if spec.mixed and prec_idx is None:
        raise ValueError(
            "a palette format selects its entry per output element, so it needs a "
            "prec_idx map; pass one, or give a single format per slot"
        )
    if not spec.mixed and any(m is not None for m in maps):
        raise ValueError("prec_idx has no meaning without a palette format to select from")
    if spec.mixed:
        return _matmul_formats(
            _mixed_matmul(spec, prec_idx, "forward", "prec_idx"),
            _mixed_matmul(spec, agrad_prec_idx, "gradient of a", "agrad_prec_idx"),
            _mixed_matmul(spec, bgrad_prec_idx, "gradient of b", "bgrad_prec_idx"),
        )
    grad = _spec_matmul(spec, _gemm_nd, "gradient")
    formats = _matmul_formats(_spec_matmul(spec, _gemm_nd, "forward"), grad, grad)
    if spec.wide:
        formats._wide_fwd = _matmul_fwd(partial(_gemm_nd, spec))
    return formats


def block_matmul_formats(
    mac: BlockMac, *, agrad_mac: BlockMac | None = None, bgrad_mac: BlockMac | None = None
) -> QMatmulFormats:
    """Build a ``QMatmulFormats`` whose three passes are block GEMMs.

    Each pass packs its operands along the dimension it sums over:

    - The forward product ``a @ b`` packs ``a`` along its last dimension and
      ``b`` along its second-to-last (the dimension K they share), in
      ``mac``'s two formats.
    - ``a``'s gradient ``grad @ b^T`` packs both ``grad`` and ``b`` along
      their last dimension.
    - ``b``'s gradient ``a^T @ grad`` packs both ``a`` and ``grad`` along
      their second-to-last dimension.

    The gradient passes take their operands' formats from ``agrad_mac`` and
    ``bgrad_mac``, which default to ``mac``. An operand whose format has
    square tiles is packed only once, by the ``a_quant`` or ``b_quant`` slot
    that this function sets for it, and the gradient passes read that packing
    transposed, through :attr:`mptorch.quant.BlockPacked.mT`. Each pass
    computes its result in the carrier (float32 for binary32, float64 for
    binary64) and stores it in the operands' dtype.

    Args:
        mac (BlockMac): the forward pass's operand formats and arithmetic.
        agrad_mac (BlockMac, optional): the same for ``a``'s gradient.
            Default: ``None``, which takes ``mac``.
        bgrad_mac (BlockMac, optional): the same for ``b``'s gradient.
            Default: ``None``, which takes ``mac``.

    Returns:
        QMatmulFormats: with the three math hooks set, and ``a_quant`` or
        ``b_quant`` for each operand in a square-tiled format.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK, MXFP8_E4M3
        >>> from mptorch.quant import BlockMac, QMatmul, block_matmul_formats
        >>> layer = QMatmul(block_matmul_formats(BlockMac(MXFP8_E4M3, acc=BinaryK(16, 11))))
        >>> layer(torch.randn(4, 64), torch.randn(64, 8)).shape
        torch.Size([4, 8])
    """
    agrad_mac = mac if agrad_mac is None else agrad_mac
    bgrad_mac = mac if bgrad_mac is None else bgrad_mac
    qa, qb = cast_quant(mac.a), cast_quant(mac.b)
    formats = QMatmulFormats(
        fwd_math=_block_matmul_fwd(mac),
        bwd_agrad_math=_block_matmul_agrad(agrad_mac),
        bwd_bgrad_math=_block_matmul_bgrad(bgrad_mac),
    )
    if qa.fmt.is_square:
        formats.a_quant = partial(qa.pack_fresh, axis=-1)
    if qb.fmt.is_square:
        formats.b_quant = partial(qb.pack_fresh, axis=-2)
    return formats
