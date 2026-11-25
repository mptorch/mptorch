"""Walsh-Hadamard rotations along a product's reduction axis.

A block format chooses one scale per block, so one large element costs the
rest of its block most of their precision. Rotating both operands of a
product along K before they are quantized spreads that element across its
block, and the product is unchanged in exact arithmetic:
``(x R)(W R)^T = x W^T`` for an orthonormal ``R``. On GPT-2 small this
removes most of the perplexity MXFP4 and MXFP8 lose, and adds to NVFP4's
(``docs/source/kernels/block.rst``, "Rotating before block quantization").

``R`` is block-diagonal: ``D`` times the orthonormal Walsh-Hadamard
transform of order ``r`` repeated along K, where ``D`` is a diagonal of
+1 and -1 (``signed=True``) or the identity. :class:`Hadamard` is the value
and :func:`hadamard_rotate` the transform. A :class:`~mptorch.quant.GemmFormats`
with a ``rotation`` rotates its forward product's operands and rotates each
gradient back: the layers' autograd Functions pass a gradient straight
through their operand quantizers, so the gradient hooks see the rotated
basis, and their results are rotated back along K (``_RotatedGrad``).

The transform is a butterfly of additions in the carrier, not a matmul by
the Hadamard matrix. Additions give the same words on every backend and
under every ``torch.backends`` setting, where a float32 matmul on CUDA runs
in TF32 under ``torch.set_float32_matmul_precision("high")``; it costs
O(K log r); and it traces under Dynamo. The signs are an integer hash of the
seed and the channel's index, computed on the operand's device: no generator
(the streams stochastic rounding draws from are untouched), no host copy,
and the same signs on every device.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .ops._checks import _call_carrier, _checked_carrier, _narrowed

_SEED_LIMIT = 2**31


@dataclass(frozen=True)
class Hadamard:
    """An orthonormal Walsh-Hadamard rotation along a product's reduction axis.

    ``R`` is block-diagonal, with blocks of ``size`` channels along K, each
    the Walsh-Hadamard transform of order ``size`` scaled by
    ``1/sqrt(size)``. With ``signed`` it is first multiplied by a diagonal of
    +1 and -1 over the K channels, drawn from ``seed`` (QuaRot's randomized
    Hadamard), which breaks up outliers that the transform alone can line
    up with. A ``K`` that ``size`` does not divide is rotated in blocks of
    the largest power of two that divides both, and is left as it is when
    that is 1.

    Give it to :class:`~mptorch.quant.GemmFormats` as ``rotation``, which
    rotates both operands of the forward product along K before they are
    quantized and rotates the gradients back, or call
    :func:`hadamard_rotate` with it.

    Args:
        size (int): the order of each block, a power of two of at least 2.
        signed (bool): multiply by random signs first. Default: ``True``
        seed (int): the signs' seed, in [0, 2^31). Default: ``0``
        carrier (torch.dtype, optional): the dtype the rotation computes in,
            as for :class:`~mptorch.quant.Quant`. Default: ``None``, the
            operand's own: binary32 for float16, bfloat16 and float32, and
            binary64 for float64.

    Raises:
        TypeError: for a ``size``, ``signed`` or ``seed`` of the wrong type,
            or a ``carrier`` that is not a ``torch.dtype``.
        ValueError: for a ``size`` that is not a power of two of at least 2,
            a ``seed`` outside [0, 2^31), or a ``carrier`` dtype that is not
            a carrier.

    Example::

        >>> from mptorch import MXFP4_E2M1
        >>> from mptorch.quant import BlockMac, GemmFormats, Hadamard
        >>> gf = GemmFormats(fwd_mac=BlockMac(MXFP4_E2M1), rotation=Hadamard(128))
    """

    size: int
    signed: bool = True
    seed: int = 0
    carrier: torch.dtype | None = None

    def __post_init__(self) -> None:
        if type(self.size) is not int:
            raise TypeError(f"Hadamard's size is an int, got {self.size!r}")
        if self.size < 2 or self.size & (self.size - 1):
            raise ValueError(
                f"Hadamard's size is the order of a Walsh-Hadamard transform, a power of two of "
                f"at least 2, got {self.size}"
            )
        if type(self.signed) is not bool:
            raise TypeError(f"Hadamard's signed is True or False, got {self.signed!r}")
        if type(self.seed) is not int:
            raise TypeError(f"Hadamard's seed is an int, got {self.seed!r}")
        if not 0 <= self.seed < _SEED_LIMIT:
            raise ValueError(f"Hadamard's seed is in [0, 2**31), got {self.seed}")
        _checked_carrier(self.carrier)

    def block(self, k: int) -> int:
        """The order of the blocks a ``k``-channel axis is rotated in: the
        largest power of two dividing both ``k`` and ``size``, which is 1 (no
        rotation) for an odd ``k``."""
        return math.gcd(self.size, k)

    def label(self) -> str:
        """``H128`` or ``DH128``: the transform, with ``D`` for the signs."""
        return f"{'DH' if self.signed else 'H'}{self.size}"


def _signs(k: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``k`` signs, +1 or -1, one per channel: the low bit of a 32-bit
    integer hash of the channel's index and the seed, computed on ``device``.

    Every intermediate stays below 2^59, inside int64, so the hash is exact
    on every backend: a 32-bit value times the 27-bit multiplier, masked
    back to 32 bits after each product."""
    mask = 0xFFFFFFFF
    x = torch.arange(k, device=device, dtype=torch.int64) ^ ((seed * 0x9E3779B9) & mask)
    x = (((x >> 16) ^ x) * 0x45D9F3B) & mask
    x = (((x >> 16) ^ x) * 0x45D9F3B) & mask
    x = (x >> 16) ^ x
    return (1 - 2 * (x & 1)).to(dtype)


def _butterfly(x: torch.Tensor, r: int) -> torch.Tensor:
    """``x``'s last axis times the Walsh-Hadamard matrix of order ``r``
    (unscaled, Sylvester's order), block by block: log2(r) stages of
    ``(a + b, a - b)``, each pair ``h`` channels apart."""
    shape = x.shape
    blocks = x.reshape(-1, shape[-1] // r, r)
    rows, n = blocks.shape[0], blocks.shape[1]
    h = 1
    while h < r:
        pairs = blocks.view(rows, n, r // (2 * h), 2, h)
        a, b = pairs[..., 0, :], pairs[..., 1, :]
        blocks = torch.stack((a + b, a - b), dim=-2).view(rows, n, r)
        h *= 2
    return blocks.reshape(shape)


def _store(y: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """A result computed in its carrier, stored in ``dtype`` with one rounding."""
    if y.dtype is dtype:
        return y
    if y.dtype is torch.float64:
        return _narrowed(y, dtype)
    return y.to(dtype)


def hadamard_rotate(
    x: torch.Tensor,
    rotation: Hadamard,
    dim: int = -1,
    *,
    inverse: bool = False,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Rotate ``x`` along ``dim`` by ``rotation``'s ``R``, or by its inverse.

    The forward rotation is ``x D`` followed by the scaled Walsh-Hadamard
    transform of each block of ``rotation.block(k)`` channels, ``k`` being
    ``x.shape[dim]``; the inverse is the transform followed by ``D`` (the
    transform is symmetric and orthonormal, so ``R^-1 = R^T``). It computes
    in the carrier, one rounding per butterfly stage and one for the scale
    by ``1/sqrt(r)``, and stores the result in ``x``'s dtype. An axis that
    no power of two above 1 divides is returned as it is.

    Args:
        x (torch.Tensor): a floating-point tensor.
        rotation (Hadamard): the rotation.
        dim (int): the axis to rotate along. Default: ``-1``
        inverse (bool): apply ``R^T`` instead of ``R``. Default: ``False``
        carrier (torch.dtype, optional): the dtype to compute in, never
            narrower than ``x``'s. Default: ``None``, ``rotation.carrier``,
            and if that is ``None`` too, the operand's own (binary32 for
            float16, bfloat16 and float32).

    Returns:
        torch.Tensor: the rotated tensor, of ``x``'s dtype and shape.

    Raises:
        TypeError: for a ``carrier`` that is not a ``torch.dtype``.
        ValueError: for a ``carrier`` narrower than ``x``, or a dtype that is
            not a carrier.

    Example::

        >>> import torch
        >>> from mptorch.quant import Hadamard, hadamard_rotate
        >>> x = torch.randn(4, 64)
        >>> back = hadamard_rotate(hadamard_rotate(x, Hadamard(32)), Hadamard(32), inverse=True)
        >>> bool(torch.allclose(back, x, atol=1e-6))
        True
    """
    k = x.shape[dim]
    r = rotation.block(k)
    if r == 1:
        return x
    wide, _ = _call_carrier(
        _checked_carrier(carrier if carrier is not None else rotation.carrier), x.dtype
    )
    c = torch.float64 if wide else torch.float32
    moved = dim not in (-1, x.dim() - 1)
    t = (x.movedim(dim, -1) if moved else x).to(c)
    signs = _signs(k, rotation.seed, x.device, c) if rotation.signed else None
    if signs is not None and not inverse:
        t = t * signs
    t = _butterfly(t, r) * torch.tensor(1 / math.sqrt(r), dtype=c, device=x.device)
    if signs is not None and inverse:
        t = t * signs
    y = _store(t, x.dtype)
    return y.movedim(-1, dim) if moved else y


class _Rotation(torch.autograd.Function):
    """``hadamard_rotate(x, rotation, dim)`` with the inverse rotation as its
    backward: the gradient rotated back exactly as ``_RotatedGrad`` rotates
    it, rather than autograd's reversal of the butterfly's stages. Flash
    attention rotates ``q`` and ``k`` whole with it, outside its tile loop,
    so that their gradients are summed over the tiles in the rotated basis
    and rotated back once."""

    @staticmethod
    def forward(ctx, x, rotation, dim):
        with torch._C.DisableTorchFunction():
            ctx.rotation, ctx.dim = rotation, dim
            return hadamard_rotate(x, rotation, dim)

    @staticmethod
    def backward(ctx, *grads):
        from .modules._opaque import _opaque_backward  # imported here: modules imports this

        with _opaque_backward():
            return hadamard_rotate(grads[0], ctx.rotation, ctx.dim, inverse=True), None, None


class _RotatedQuant:
    """An operand slot that rotates along K, then runs the slot's quantizer
    (or nothing, for an empty slot).

    K is the last axis of a Linear's input and weight and of a matmul's
    ``a`` (``dim=-1``), and axis -2 of a matmul's ``b`` (``dim=-2``), whose
    rotation is ``R^T b``; a 1D ``b`` is rotated along its only axis. The
    attributes are read by ``_saved._known``, which counts the slot
    stateless when its quantizer is, and by ``QPackedLinear``."""

    def __init__(self, rotation: Hadamard, quant: Callable | None, dim: int) -> None:
        self.rotation, self.quant, self.dim = rotation, quant, dim

    def __call__(self, x: Any) -> Any:
        rotated = hadamard_rotate(x, self.rotation, self.dim if x.dim() > 1 else -1)
        return self.quant(rotated) if self.quant is not None else rotated

    def __repr__(self) -> str:
        return f"rotate {self.rotation.label()}, then {self.quant!r}"


class _RotatedGrad:
    """A gradient hook whose result is rotated back along K.

    The Functions pass gradients straight through the operand quantizers,
    so a hook over rotated operands returns its operand's gradient in the
    rotated basis: ``dX R`` for ``dX``. The hook's result is ``[..., K]``
    for a Linear's input and weight and a matmul's ``a`` (``dim=-1``), and
    ``[..., K, N]`` for a matmul's ``b`` (``dim=-2``): the Function promotes
    a 1D operand before it calls the hook."""

    def __init__(self, rotation: Hadamard, hook: Callable, dim: int) -> None:
        self.rotation, self.hook, self.dim = rotation, hook, dim

    def __call__(self, grad: torch.Tensor, operand: Any) -> torch.Tensor:
        return hadamard_rotate(self.hook(grad, operand), self.rotation, self.dim, inverse=True)
