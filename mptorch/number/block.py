"""Block formats, and the bit layout of the codes their kernels encode and decode.

A block format stores a tensor as narrow codes in blocks, and each block
shares one scale. Examples are OCP's MX formats (Microscaling Formats v1.0:
an E8M0 power-of-two scale per 32 elements) and NVIDIA's NVFP4 (an E4M3 scale
per 16 elements, times one float32 scale per tensor). Elements and scales are
stored as *minifloat codes*: a sign bit (if any), then `exp_bits` exponent
bits and `man_bits` mantissa bits with a bias, where exponent field 0 holds
subnormals (or, under `SubnormalsMode.EXTENDED_NORMALS`, one more binade of
normals). A `BinaryK` describes exactly this layout, so one type covers every
element and scale of these formats, and `csrc/common/block_decode.h` has a
single decoder and encoder for all of them. A `SuperFP` code uses the same
fields, with its supernormal powers of two in the lowest codes, and a
`FixedPoint` element is stored as the binaryK layout that its integer code
amounts to (`_minifloat`).

An element is rounded by the cast of its own format under `SAT_FINITE` and
then clamped to `elem_max`. This is where the OCP formats differ from
P3109's: E4M3's largest value under `SAT_FINITE` is 480 but OCP's is 448,
because OCP uses the code above 448 for NaN; for E5M2 they are 114688 and
57344, because OCP's E5M2 has infinities. Clamping after rounding gives the
right result in every rounding mode: `elem_max` is a value of the element
format, so an input above it rounds to `elem_max` or higher, and the clamp
brings it back to `elem_max`.

A scale with no mantissa (`P = 1`) is read the way E8M0 is: code `c` stands
for the power of two `2**(c - bias)`, except the all-ones code, which is
NaN. Such a scale is computed from the block's largest magnitude with integer
arithmetic, not by a cast. A scale with a mantissa (a *cast scale*) is
rounded to its format under `SAT_FINITE`, by the rule `ScaleRounding`
selects, and also uses its all-ones magnitude code for NaN.

The block kernels round in the call's carrier, binary32 or binary64, and
build decoded values directly from the carrier's bits. So the element format
and a cast scale's format must fit the carrier entirely, and a power-of-two
scale's values must lie within the carrier's range. When it is built, a
`BlockFormat` raises only for what binary64 cannot do, just as `BinaryK`
raises only for what no carrier can do. It stores binary32's verdict, which
each call that rounds in binary32 raises (`check_block_carrier`).
"""

from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from typing import NamedTuple

import torch

from ._carrier import _BINARY32, _BINARY64, _BINARY64_HINT, _CARRIERS, _top_of_range
from ._storage import _STORAGE
from .base import Number, _report_per_call
from .binaryk import BinaryK, _binaryK_findings, _binaryK_label
from .fixedpoint import FixedPoint, _fixedpoint_findings, _fixedpoint_label
from .modes import SaturationMode, SubnormalsMode
from .superfp import SuperFP, _superfp_findings, _superfp_label


class ScaleRounding(Enum):
    """How a :class:`BlockFormat` chooses each block's scale from the block's
    largest magnitude ``amax`` (and, under the ``FOUR_OVER_SIX`` rules, from
    its elements).

    Mirrors ``ScaleRounding`` in ``csrc/common/block_decode.h``, by name and
    value. Two quantities are used below:

    * ``r = amax / elem_max`` is the scale that would put ``amax`` exactly on
      ``elem_max``. For a cast scale it is also divided by the tensor scale.
    * ``T`` is ``elem_max`` plus half the element format's spacing at the top
      (7 for E2M1, whose two largest values, 4 and 6, are 2 apart). It is the
      largest scaled element that round-to-nearest would still bring down to
      ``elem_max``.

    Each rule's result is clamped to the range of the scale format, and a
    cast scale that comes out as zero is raised to the smallest positive
    scale. No rule depends on the rounding mode used for the elements. For
    example, with E2M1 elements and a power-of-two scale, a block's largest
    element, once divided by the scale, lands in ``[4, 8)`` under
    :attr:`OCP`, ``[4.5, 9]`` under :attr:`NEAREST`, ``(3, 6]`` under
    :attr:`UP` and ``(3.5, 7]`` under :attr:`SELECTIVE`; anything above 6 then
    saturates to 6.

    The three ``FOUR_OVER_SIX`` rules (Cook et al., `Four Over Six
    <https://arxiv.org/abs/2512.02010>`__) look at the block's elements, not
    only at ``amax``. A block has two candidate scales:

    * :attr:`NEAREST`'s scale, under which the block's largest element lands
      near ``elem_max`` (6 for E2M1);
    * the scale nearest to ``r * elem_max / 2**emax``, under which it lands
      near ``2**emax`` (4 for E2M1).

    The block is quantized with both, in the call's rounding mode, and keeps
    the second scale only if its error is strictly smaller. The three rules
    differ in how they measure the error. They need a cast scale, because
    the two candidates differ by a factor of ``elem_max / 2**emax`` (1.5 for
    E2M1), which two power-of-two scales cannot. With ``block_rows > 1`` the
    whole tile is treated as one block: one pair of candidates, one error
    over all of its elements, and one choice. Under these rules the default
    tensor scale gives the tensor's largest block the scale ``top_scale``
    instead of ``scale_max`` (256 instead of 448 for E4M3), so that this
    block also has both candidates. Errors are accumulated as integers, in
    units of ``2**(floor(log2(amax)) - 22)``, so a block's error is exact and
    does not depend on the order in which its elements are added. The choice
    is therefore the same on every backend, and a square tile makes the same
    choice whichever of its two axes the tensor is packed along.

    Example::

        >>> import dataclasses
        >>> from mptorch import MXFP4_E2M1, ScaleRounding
        >>> MXFP4_E2M1.scale_rule
        <ScaleRounding.OCP: 0>
        >>> dataclasses.replace(MXFP4_E2M1, scale_rounding=ScaleRounding.UP).scale_rule
        <ScaleRounding.UP: 2>
    """

    #: OCP MX's rule, for power-of-two scales only:
    #: ``2**(floor(log2(amax)) - emax)``. A block's largest element can exceed
    #: ``elem_max`` by a factor of up to ``2**(emax + 1) / elem_max`` (4/3 for
    #: E2M1), and then saturates. The default for a power-of-two scale (E8M0).
    OCP = 0
    #: ``r`` rounded to nearest (ties to even) in the scale format. The scale
    #: can be up to half a scale step below ``r``, so the largest element can
    #: land above ``elem_max``. Among E4M3's normal values that is a factor of
    #: at most 1.0625, which E2M1's top step absorbs: 6 x 1.0625 = 6.375 still
    #: rounds to 6 under round-to-nearest. Where the scale's values are powers
    #: of two it is a factor of up to 1.5, and the largest element can
    #: saturate, losing up to a third of its value. NVFP4's rule, and the
    #: default for a binaryK scale with a mantissa (E4M3).
    NEAREST = 1
    #: The smallest scale that is at least ``r``. No element exceeds
    #: ``elem_max``, but the scale can be up to twice ``r`` (with a
    #: power-of-two scale), which makes the grid up to twice as coarse for
    #: the rest of the block. The rule that NVIDIA's MXFP8 pretraining recipe
    #: uses for E8M0 (arXiv 2506.08027).
    UP = 2
    #: :attr:`NEAREST`'s scale, or the next scale up if under it the block's
    #: largest element would land above ``T``: the only case in which the
    #: clamp at ``elem_max`` moves the element further than rounding to
    #: nearest would. With a power-of-two scale this is the smallest scale
    #: with ``amax / scale <= T``. The default for a :class:`SuperFP` scale,
    #: whose values below its normal binades are powers of two.
    SELECTIVE = 3
    #: Four Over Six, comparing the sums of squared errors; the paper finds it
    #: best for post-training quantization.
    FOUR_OVER_SIX_MSE = 4
    #: Four Over Six, comparing the sums of absolute errors; the paper finds
    #: it best for pre-training.
    FOUR_OVER_SIX_MAE = 5
    #: Four Over Six, comparing the largest absolute errors.
    FOUR_OVER_SIX_ABS_MAX = 6

    @property
    def is_four_over_six(self) -> bool:
        """Whether this is one of the three ``FOUR_OVER_SIX`` rules."""
        return self.value >= ScaleRounding.FOUR_OVER_SIX_MSE.value


@dataclass(frozen=True)
class BlockFormat(Number):
    """A block format: narrow element codes that share one scale per block.

    Covers OCP's Microscaling (MX) formats and NVIDIA's NVFP4, and the same
    construction with :class:`SuperFP` elements or scales, or with
    :class:`FixedPoint` elements (block floating point, see
    :func:`~mptorch.bfp`). A tensor is split into blocks of
    ``block_size`` consecutive elements along one axis, the *packed* axis,
    which the functions that take a block format select with ``axis``. Each
    block stores one ``scale`` code and ``block_size`` element codes, and an
    element's value is the value of its code in the ``elem`` format times its
    block's scale. :func:`mptorch.quant.block_quantize` rounds a tensor to the
    format, :func:`mptorch.quant.block_pack` stores it as ``uint8`` codes, and
    :func:`mptorch.quant.block_matmul` multiplies two packed operands.

    The type of ``scale`` decides whether the tensor also has a per-tensor
    scale; there is no separate option for it:

    - A *power-of-two* scale (``scale.P == 1``: E8M0, the MX presets) covers
      ``2**-127`` to ``2**127``, so the block scale is enough. The format has
      no tensor scale, and passing ``tensor_scale=`` raises.
    - A *cast* scale is a binaryK with a mantissa (NVFP4's E4M3) or a
      superfp. It can match a block's largest magnitude more closely, but
      covers a much shorter range (E4M3: ``2**-9`` to 448). The format
      therefore also has one scale per tensor, a carrier value (float32 in
      binary32, float64 in binary64) that brings the tensor into that range.
      A block's effective scale is its scale code's value times the tensor
      scale, computed as one product in the carrier. The functions that
      quantize or pack take the tensor scale as ``tensor_scale`` (its default
      is given below); ``tensor_scale=1.0`` leaves the block scales as they
      are.
    - ``scale=None`` stores the element codes unscaled.

    A tensor scale without block scales is not available.

    With ``block_rows > 1``, one scale is shared by a *tile* of
    ``block_rows`` consecutive rows of ``block_size`` elements each. The rows
    are taken along the axis just before the packed one, never across a
    batch dimension. A square tile (``block_rows == block_size``) quantizes a
    matrix to the same values whichever of its two axes is packed, in every
    deterministic rounding mode. One packed weight can therefore serve both a
    layer's forward pass and its input gradient, which reads the weight
    transposed. ``dataclasses.replace(NVFP4, block_rows=16)`` is NVFP4 with
    16 x 16 weight tiles.

    **Choosing the scale.** Each block's scale is chosen from the block's
    largest magnitude ``amax`` by the ``scale_rounding`` rule
    (:class:`ScaleRounding`). The defaults are:

    - A power-of-two scale (``scale.P == 1``, E8M0) is
      ``2**(floor(log2(amax)) - emax)``, clamped to the scale format's range,
      where ``emax = floor(log2(elem_max))``. This is OCP MX's rule.
    - A ``BinaryK`` scale with a mantissa (E4M3) is
      ``amax / (elem_max * tensor_scale)`` rounded to nearest (ties to even)
      by the scale format's cast. This is NVFP4's rule.
    - A ``SuperFP`` scale rounds that ratio the same way, except where the
      block's largest element would then saturate by more than rounding to
      nearest would move it. There it takes the next scale up.

    A cast scale is clamped to the scale format's largest value, and raised
    to its smallest positive value if it would be zero. The tensor scale
    defaults to ``max(|x|) / (elem_max * top_scale)`` over the whole tensor
    ``x``.

    **Rounding the elements.** Each element is divided by its scale and
    rounded, in the call's rounding mode, by ``elem``'s cast under
    ``SaturationMode.SAT_FINITE`` (a ``FixedPoint`` element by the cast of
    the binaryK layout its code amounts to). It is then clamped to
    ``elem_max``. ``elem``'s own ``saturation`` is ignored; its
    ``subnormals`` and ``prng_bits`` are used.

    **Special values.** An infinite input saturates to ``elem_max``. A NaN
    input takes ``nan_code`` when the element format has one. Otherwise the
    block's scale becomes the scale format's NaN, and the whole block (every
    element of a 2D tile) decodes to NaN. With neither (``scale=None`` and no
    ``nan_code``) the NaN becomes ``elem_max``. No value decodes to ``-0.0``.

    **Superfp codes.** A superfp code follows its format's layout: the normal
    binades' codes as in a binaryK, and below them, from code 1, the
    supernormal powers of two (see :class:`SuperFP`). It reserves no code for
    NaN or infinity unless ``nan_code`` or ``inf_code`` says so, as for a
    binaryK element. A superfp scale uses its all-ones code for NaN, like
    every cast scale.

    **Carriers.** The block ops round in binary32, or in binary64 for a
    float64 tensor or under ``carrier=torch.float64``, and the format must
    fit the call's carrier: the element format and a cast scale's format
    entirely, and a power-of-two scale's values within ``2**-127 .. 2**127``
    in binary32 and ``2**-1023 .. 2**1023`` in binary64. Building a format
    checks binary64's limits, and each call that rounds in binary32 checks
    binary32's. An element with eight to ten exponent bits, or an E8M0-style
    scale whose bias puts it outside binary32's range, can therefore be used
    only with float64 tensors or ``carrier=torch.float64``.

    Args:
        elem (BinaryK, SuperFP or FixedPoint): the element format: at most 16
            bits, and one that binary64 holds entirely (and binary32 too, for
            a call that rounds in binary32). A ``FixedPoint`` element must be
            signed and symmetric.
        scale (BinaryK or SuperFP, optional): the scale format, at most one
            byte. It is E8M0 (``BinaryK(8, 1, bias=127, is_signed=False,
            subnormals=SubnormalsMode.EXTENDED_NORMALS)``) or another unsigned
            binaryK with ``P = 1``, a binaryK with a mantissa such as E4M3, or
            a superfp. The last two add a per-tensor scale. ``None`` stores
            the element codes unscaled.
        block_size (int): elements per block along the packed axis: a power
            of two in [2, 128], with ``elem_bits * block_size`` a multiple of
            8, so that a block fills whole bytes.
        block_rows (int): rows per scale tile: a power of two in [1, 128],
            with ``block_rows * block_size <= 16384``; it must be 1 without a
            scale. Default: ``1``.
        elem_max (float, optional): the largest magnitude an element may
            take; it must be a value of ``elem``. Default: ``None``, meaning
            ``elem``'s largest finite value under ``SAT_FINITE``.
        nan_code (int, optional): the magnitude code (sign bit clear) that
            the element format uses for NaN; it must be above ``elem_max``'s
            code. Default: ``None``.
        inf_code (int, optional): the magnitude code it uses for infinity,
            also above ``elem_max``'s code. It is decoded but never produced.
            Default: ``None``.
        scale_rounding (ScaleRounding, optional): how each block's scale is
            chosen. ``OCP`` needs a power-of-two scale, and a
            ``FOUR_OVER_SIX`` rule needs a cast scale. Default: ``None``,
            meaning the scale's own default rule: ``OCP`` for a power-of-two
            scale, ``NEAREST`` for a binaryK with a mantissa, ``SELECTIVE``
            for a superfp. The default is resolved against the scale each
            time the format is built, so
            ``dataclasses.replace(NVFP4, scale=SuperFP(...))`` gets the
            superfp's rule.

    These attributes are derived when the format is built and are read-only:

    - ``scale_rule``: the :class:`ScaleRounding` in effect (``None`` without
      a scale). Formats compare by it rather than by ``scale_rounding``, so
      leaving ``scale_rounding`` at its default and passing that default
      explicitly give equal formats.
    - ``elem_bits``: the bits per element code, including the sign bit
      (``elem.K`` for a binaryK, ``elem.wl`` for a fixed point).
    - ``emax``: ``floor(log2(elem_max))``.
    - ``scale_max``: the largest scale, the value of the code below the
      scale's NaN (448 for E4M3, ``2**127`` for E8M0, 0 without a scale).
    - ``top_scale``: the scale that the default tensor scale gives the
      tensor's largest block. It is ``scale_max``, except under a
      ``FOUR_OVER_SIX`` rule, where it is the largest scale ``v`` for which
      ``v * elem_max / 2**emax`` is also a scale (256 for E4M3, since
      256 x 1.5 = 384 is an E4M3 value).
    - ``has_tensor_scale``: whether the format has a per-tensor scale, which
      is the case for a cast scale (any scale but a power-of-two one).
    - ``bytes_per_block``: the bytes of element codes in one row of a block.
    - ``is_square``: ``block_rows == block_size``.

    Raises:
        ValueError: for any of the following: sizes outside the ranges above;
            an element over 16 bits or a scale over 8; an element format or
            cast scale format that binary64 does not hold entirely; a
            ``FixedPoint`` element that is not signed and symmetric, or that
            is given a ``nan_code`` or ``inf_code``; an ``elem_max`` that is
            not a value of ``elem``; a ``nan_code`` or ``inf_code`` at or
            below ``elem_max``'s code, or the two equal; a power-of-two scale
            that is signed or reaches beyond ``2**-1023`` or ``2**1023`` (the
            pack multiplies by its reciprocal); a cast scale with no positive
            value below its NaN; ``scale_rounding=OCP`` on a cast scale; a
            ``FOUR_OVER_SIX`` rule on a power-of-two scale; or any
            ``scale_rounding`` without a scale.
        TypeError: if ``elem`` is not a ``BinaryK``, ``SuperFP`` or
            ``FixedPoint``, ``scale`` is not a ``BinaryK``, ``SuperFP`` or
            ``None``, or ``scale_rounding`` is not a ``ScaleRounding`` or
            ``None``.

    Example::

        >>> from mptorch import MXFP8_E4M3, NVFP4
        >>> MXFP8_E4M3.elem_max, MXFP8_E4M3.emax, MXFP8_E4M3.bytes_per_block
        (448.0, 8, 32)
        >>> NVFP4.scale_max, NVFP4.has_tensor_scale
        (448.0, True)
        >>> import dataclasses
        >>> dataclasses.replace(NVFP4, block_rows=16).is_square
        True
        >>> from mptorch import E2M1, E8M0, BlockFormat, SuperFP
        >>> sfp4 = BlockFormat(SuperFP(1, 2, 1, 1), E8M0, 32)  # 0 .. 6, down to 1/8
        >>> sfp4.elem_bits, sfp4.elem_max, sfp4.emax
        (4, 6.0, 2)
        >>> BlockFormat(E2M1, SuperFP(3, 4, 2, 7), 16).scale_rule
        <ScaleRounding.SELECTIVE: 3>
    """

    elem: "BinaryK | SuperFP | FixedPoint"
    scale: "BinaryK | SuperFP | None"
    block_size: int
    _: KW_ONLY
    block_rows: int = 1
    elem_max: float | None = None
    nan_code: int | None = None
    inf_code: int | None = None
    scale_rounding: ScaleRounding | None = field(default=None, compare=False)
    scale_rule: ScaleRounding | None = field(init=False, repr=False)
    elem_bits: int = field(init=False, repr=False, compare=False)
    emax: int = field(init=False, repr=False, compare=False)
    scale_max: float = field(init=False, repr=False, compare=False)
    top_scale: float = field(init=False, repr=False, compare=False)
    has_tensor_scale: bool = field(init=False, repr=False, compare=False)
    bytes_per_block: int = field(init=False, repr=False, compare=False)
    is_square: bool = field(init=False, repr=False, compare=False)
    # Why binary32 cannot carry the format (its element, its cast scale, or a
    # power-of-two scale's range), or None. A call that rounds in binary32
    # raises it (`check_block_carrier`); binary64's verdict was already
    # enforced when the format was built.
    _binary32_error: str | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        elem, scale = self.elem, self.scale
        if not isinstance(elem, BinaryK | SuperFP | FixedPoint):
            raise TypeError(
                f"a BlockFormat's element must be a BinaryK, a SuperFP or a FixedPoint, got "
                f"{type(elem).__name__}"
            )
        if isinstance(elem, FixedPoint):
            if not (elem.is_signed and elem.symmetric):
                raise ValueError(
                    "a FixedPoint block element must be signed and symmetric, got "
                    f"{_part_label(elem)}: block codes are sign-magnitude, which cannot encode "
                    "two's complement's most negative value, and an unsigned element's top "
                    "code would be its layout's NaN"
                )
            if self.nan_code is not None or self.inf_code is not None:
                raise ValueError(
                    "a FixedPoint block element has a value at every code, so it takes no "
                    "nan_code or inf_code"
                )
        mf = _minifloat(elem)
        bits = mf.bits
        if bits > 16:
            raise ValueError(f"a BlockFormat's element codes can have at most 16 bits, got {bits}")
        bs, br = self.block_size, self.block_rows
        if not 2 <= bs <= 128 or bs & (bs - 1):
            raise ValueError(f"block_size must be a power of two in [2, 128], got {bs}")
        if (bits * bs) % 8:
            raise ValueError(
                f"a block of {bs} {bits}-bit codes is {bits * bs} bits, but a block must fill "
                "whole bytes: elem_bits * block_size must be a multiple of 8"
            )
        if not 1 <= br <= 128 or br & (br - 1) or br * bs > 16384:
            raise ValueError(
                f"block_rows must be a power of two in [1, 128] with block_rows * block_size "
                f"<= 16384, got block_rows={br} and block_size={bs}"
            )
        error = _block_part_error("element", elem, torch.float64)
        if error is not None:
            raise ValueError(error)
        binary32_error = _block_part_error("element", elem, torch.float32)

        # under SAT_FINITE an unsigned binaryK still reserves its top code for
        # P3109's NaN; a signed one keeps its NaN at -0's code, and superfp has none
        reserved = 0 if elem.is_signed or isinstance(elem, SuperFP) else 1
        _, top_exp, top_code = _top_of_range(mf.exp_bits, mf.man_bits, mf.bias, reserved, _BINARY64)
        top = Fraction((1 << mf.man_bits) + top_code) * Fraction(2) ** (top_exp - mf.man_bits)
        if self.elem_max is None:
            elem_max = top
        else:
            elem_max = Fraction(self.elem_max)
            if not 0 < elem_max <= top or _minifloat_code(mf, elem_max) is None:
                raise ValueError(
                    "elem_max must be a positive value of the element format "
                    f"{_part_label(elem)}, at most its largest finite value {float(top)}, got "
                    f"{self.elem_max}"
                )
        max_code = _minifloat_code(mf, elem_max)
        assert max_code is not None
        magnitudes = 1 << (bits - int(elem.is_signed))
        for name, code in (("nan_code", self.nan_code), ("inf_code", self.inf_code)):
            if code is not None and not max_code < code < magnitudes:
                raise ValueError(
                    f"{name} must be a magnitude code above elem_max's ({max_code:#x}) and "
                    f"below {magnitudes:#x}, so that the encoder never produces it for a value; "
                    f"got {code}"
                )
        if self.nan_code is not None and self.nan_code == self.inf_code:
            raise ValueError("nan_code and inf_code must differ")

        rounding = self.scale_rounding
        if rounding is not None and not isinstance(rounding, ScaleRounding):
            raise TypeError(
                f"scale_rounding must be a ScaleRounding or None, got {type(rounding).__name__}"
            )
        if scale is None:
            if br != 1:
                raise ValueError("block_rows > 1 shares one scale across rows, so it needs a scale")
            if rounding is not None:
                raise ValueError("scale_rounding chooses each block's scale, so it needs a scale")
            scale_max, tensor_scaled = 0.0, False
        else:
            if not isinstance(scale, BinaryK | SuperFP):
                raise TypeError(
                    "a BlockFormat's scale must be a BinaryK, a SuperFP or None, got "
                    f"{type(scale).__name__}"
                )
            smf = _minifloat(scale)
            if smf.bits > 8:
                raise ValueError(f"a BlockFormat's scale must fit in one byte, got {smf.bits} bits")
            if isinstance(scale, BinaryK) and scale.P == 1:
                if scale.is_signed:
                    raise ValueError(
                        "a power-of-two scale (P=1, like E8M0) must be unsigned: its codes "
                        "are exponents"
                    )
                lo, hi = -_bias(scale), (1 << scale.K) - 2 - _bias(scale)
                for carrier in (_BINARY64, _BINARY32):
                    if lo < -carrier.top_exp or hi > carrier.top_exp:
                        error = (
                            f"a power-of-two scale's values are 2^{lo} .. 2^{hi}, but they must "
                            f"lie within 2^-{carrier.top_exp} .. 2^{carrier.top_exp}, so that "
                            f"{carrier.name} can also represent their reciprocals, which the pack "
                            "multiplies by"
                        )
                        if carrier is _BINARY64:
                            raise ValueError(error)
                        binary32_error = binary32_error or f"{error}. {_BINARY64_HINT}"
                scale_max, tensor_scaled = 2.0**hi, False
                if rounding is not None and rounding.is_four_over_six:
                    raise ValueError(
                        f"ScaleRounding.{rounding.name} chooses between two scales a factor of "
                        "elem_max / 2**emax apart (1.5 for E2M1), which two powers of two cannot "
                        "be: it needs a cast scale, such as NVFP4's E4M3"
                    )
            else:
                error = _block_part_error("scale", scale, torch.float64)
                if error is not None:
                    raise ValueError(error)
                binary32_error = binary32_error or _block_part_error("scale", scale, torch.float32)
                nan_mag = (1 << (smf.bits - int(scale.is_signed))) - 1
                scale_max, tensor_scaled = float(_minifloat_value(smf, nan_mag - 1)), True
                if scale_max <= 0:
                    raise ValueError(
                        f"the scale format {_part_label(scale)} has no positive value below its NaN"
                    )
                if rounding is ScaleRounding.OCP:
                    raise ValueError(
                        "ScaleRounding.OCP is OCP MX's rule for a power-of-two scale, but "
                        f"{_part_label(scale)} is a cast scale: use NEAREST, UP, SELECTIVE or a "
                        "FOUR_OVER_SIX rule"
                    )
            if rounding is None:
                if not tensor_scaled:
                    rounding = ScaleRounding.OCP
                elif isinstance(scale, SuperFP):
                    rounding = ScaleRounding.SELECTIVE
                else:
                    rounding = ScaleRounding.NEAREST

        # top_scale: the scale the default tensor scale gives the tensor's
        # largest block. Under Four Over Six that block also needs its second
        # candidate, a factor of elem_max / 2^emax larger, so top_scale is the
        # largest scale whose product with that factor is also a scale (256
        # for E4M3: 256 * 1.5 = 384 is an E4M3 value, and 288 * 1.5 = 432 is
        # not).
        top_scale = scale_max
        if scale is not None and rounding is not None and rounding.is_four_over_six:
            smf = _minifloat(scale)
            factor = elem_max / Fraction(2) ** _floor_log2(elem_max)
            for mag in range((1 << (smf.bits - int(scale.is_signed))) - 2, 0, -1):
                value = _minifloat_value(smf, mag)
                if value * factor <= Fraction(scale_max) and (
                    _minifloat_code(smf, value * factor) is not None
                ):
                    top_scale = float(value)
                    break

        derived = {
            "scale_rule": rounding,
            "elem_max": float(elem_max),
            "elem_bits": bits,
            "emax": _floor_log2(elem_max),
            "scale_max": scale_max,
            "top_scale": top_scale,
            "has_tensor_scale": tensor_scaled,
            "bytes_per_block": bits * bs // 8,
            "is_square": br == bs,
            "_binary32_error": binary32_error,
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)


def check_block_carrier(fmt: BlockFormat, *, carrier: torch.dtype) -> None:
    """Check a block format against the carrier a call rounds in.

    Building ``fmt`` already raised for anything binary64 cannot carry (the
    element, the scale, or the scale's range). What remains for each call is
    binary32's verdict, which the format stores.

    Raises:
        ValueError: for ``carrier=torch.float32`` when binary32 does not
            entirely hold the element format, the cast scale's format, or a
            power-of-two scale's range. binary64 does hold them, so a float64
            tensor, or ``carrier=torch.float64``, can use the format.
    """
    if carrier is torch.float32 and fmt._binary32_error is not None:
        raise ValueError(
            f"{_block_label(fmt)} cannot be used in binary32, the carrier this call rounds "
            f"in: {fmt._binary32_error}"
        )


def check_block_storage(
    fmt: BlockFormat, *, storage: torch.dtype, elementwise: bool = True
) -> None:
    """Check whether the storage dtype of a result can hold a block format's decoded values.

    ``storage``, the storage dtype, is ``torch.float16`` or ``torch.bfloat16``
    (or ``torch.float32`` for a result decoded in binary64). Writing a decoded
    value that the storage dtype does not hold rounds it a second time.
    ``elementwise=True`` is the rule for :func:`mptorch.quant.block_quantize`,
    and for :func:`mptorch.quant.block_unpack`, of a tensor whose inputs are
    already values of the storage dtype; ``elementwise=False`` checks every
    value the block format can decode to. See :func:`check_binaryK_storage`
    for the two rules.

    Warns:
        FormatRangeWarning: where a decoded value may not be a value of the storage dtype.
    """
    _report_per_call(*_block_storage_findings(fmt, storage, elementwise))


# --- the bit layout of the codes ---------------------------------------------


class _Minifloat(NamedTuple):
    """The bit layout of a `BinaryK`, `SuperFP` or `FixedPoint` code, as the
    block kernels encode and decode it."""

    is_signed: bool
    exp_bits: int
    man_bits: int
    bias: int
    extended: bool  # EXTENDED_NORMALS: exponent field 0 is a binade of normals
    # superfp only (0 otherwise): magnitude code 0 is the zero, and codes 1 to
    # super_codes - 1 are the supernormals, code c standing for the power of
    # two 2**(super_cutoff + c - 1)
    super_codes: int = 0
    super_cutoff: int = 0

    @property
    def bits(self) -> int:
        """Bits per code, including the sign bit."""
        return int(self.is_signed) + self.exp_bits + self.man_bits


def _bias(f: BinaryK) -> int:
    """The bias of a built ``BinaryK``, which ``__post_init__`` has filled in."""
    assert f.bias is not None
    return f.bias


def _minifloat(f: "BinaryK | SuperFP | FixedPoint") -> _Minifloat:
    """Return the code layout of ``f``.

    For a ``BinaryK`` it is P3109's field widths in sign-magnitude form; for
    a ``SuperFP``, its fields plus its supernormal codes (``cast_superfp.h``'s
    ``superfp_region_cutoffs``); for a ``FixedPoint``, the binaryK layout that
    its sign-magnitude integer code amounts to.

    A ``FixedPoint`` with ``M`` magnitude bits is, bit for bit, the binaryK
    layout with one exponent bit, ``M - 1`` mantissa bits and bias
    ``2 - M + fl``. Exponent field 0 (the subnormals) holds ``m * 2**-fl``
    and field 1 holds ``(2**(M-1) + m) * 2**-fl``, so code ``c`` has the
    value ``c * 2**-fl``: the same step from zero to the largest value, as in
    the fixed-point format. The block kernels decode, round and encode it as
    that binaryK, so no kernel needs to know about fixed point."""
    if isinstance(f, FixedPoint):
        m = f.mag_bits
        return _Minifloat(f.is_signed, 1, m - 1, 2 - m + f.fl, False)
    if isinstance(f, SuperFP):
        normal_cutoff = (1 << f.exp_bits) - f.normal_binades - f.bias
        super_codes = ((1 << f.exp_bits) - f.normal_binades) << f.man_bits
        return _Minifloat(
            f.is_signed,
            f.exp_bits,
            f.man_bits,
            f.bias,
            False,
            super_codes,
            normal_cutoff - super_codes + 1,
        )
    exp_bits = f.K - f.P if f.is_signed else f.K - f.P + 1
    return _Minifloat(
        f.is_signed, exp_bits, f.P - 1, _bias(f), f.subnormals is SubnormalsMode.EXTENDED_NORMALS
    )


def _minifloat_value(mf: _Minifloat, mag: int) -> Fraction:
    """The magnitude that a code (with its sign bit cleared) encodes, ignoring NaN and infinity."""
    if mag < mf.super_codes:
        return Fraction(0) if mag == 0 else Fraction(2) ** (mf.super_cutoff + mag - 1)
    e, m = mag >> mf.man_bits, mag & ((1 << mf.man_bits) - 1)
    if e == 0 and not mf.extended:
        return Fraction(m) * Fraction(2) ** (1 - mf.bias - mf.man_bits)
    if e == 0 and m == 0:
        return Fraction(0)  # EXTENDED_NORMALS: field 0's zero-mantissa code is the zero
    return Fraction((1 << mf.man_bits) + m) * Fraction(2) ** (e - mf.bias - mf.man_bits)


def _floor_log2(value: Fraction) -> int:
    """``floor(log2(value))`` of a positive rational, exactly."""
    num, den = value.numerator, value.denominator
    e = num.bit_length() - den.bit_length()
    if (num << -e if e < 0 else num) < (den if e < 0 else den << e):
        e -= 1
    return e


def _minifloat_code(mf: _Minifloat, value: Fraction) -> int | None:
    """The magnitude code of a positive ``value``, or ``None`` if the layout has
    no code for it."""
    e = _floor_log2(value)
    if mf.super_codes and e - mf.super_cutoff + 1 < mf.super_codes:
        # below a superfp's normal binades only its powers of two are values
        code = e - mf.super_cutoff + 1
        return code if code >= 1 and value == Fraction(2) ** e else None
    lowest = -mf.bias if mf.extended else 1 - mf.bias
    if e >= lowest:
        field_ = e + mf.bias
        m = (value / Fraction(2) ** e - 1) * (1 << mf.man_bits)
        if m.denominator != 1 or field_ >= (1 << mf.exp_bits) or (field_ == 0 and m == 0):
            return None
        return (field_ << mf.man_bits) | int(m)
    if mf.extended:
        return None
    m = value / Fraction(2) ** (1 - mf.bias - mf.man_bits)
    return int(m) if m.denominator == 1 else None


def _significant_bits(value: Fraction) -> int:
    """The number of significant bits of a dyadic ``value`` (a rational whose
    denominator is a power of two)."""
    num = value.numerator
    return (num >> ((num & -num).bit_length() - 1)).bit_length()


# --- how the checks reach their findings -------------------------------------


def _block_part_error(
    role: str, f: "BinaryK | SuperFP | FixedPoint", carrier: torch.dtype
) -> str | None:
    """Return why a block format cannot use ``f`` as its element or cast scale
    in ``carrier``, or ``None``.

    The block kernels build decoded values directly from the carrier's bits,
    so the format they round with (``f`` under ``SAT_FINITE``) must fit the
    carrier entirely: the carrier check must find neither an error nor a
    range warning. A ``FixedPoint`` element is checked by its own rule,
    `_fixedpoint_findings`, which amounts to the binaryK rule for its layout,
    except that the stochastic-rounding bits count against the magnitude
    bits.
    """
    if isinstance(f, FixedPoint):
        error, warning = _fixedpoint_findings(
            f.wl, f.fl, f.is_signed, f.symmetric, f.prng_bits, carrier
        )
    elif isinstance(f, SuperFP):
        error, warning = _superfp_findings(
            f.man_bits,
            f.exp_bits,
            f.normal_binades,
            f.bias,
            SaturationMode.SAT_FINITE,
            f.prng_bits,
            carrier,
        )
    else:
        error, warning = _binaryK_findings(
            f.K,
            f.P,
            f.bias,
            f.is_signed,
            SaturationMode.SAT_FINITE,
            f.subnormals,
            f.prng_bits,
            carrier,
        )
    found = error or warning
    if found is None:
        return None
    name = _CARRIERS[carrier].name
    return f"a BlockFormat's {role} must be a format that fits entirely in {name}, but {found}"


def _part_label(f: "BinaryK | SuperFP | FixedPoint | None") -> str:
    """A short label for an element or scale format, for messages: its preset
    name (``E4M3``) if it is one, and otherwise its constructor call."""
    from . import presets  # imported here, since `presets` imports this module

    for name in ("E8M0", "E4M3", "E5M2", "E2M1", "E2M3", "E3M2"):
        if f == getattr(presets, name):
            return name
    if isinstance(f, FixedPoint):
        return _fixedpoint_label(f.wl, f.fl, f.is_signed, f.symmetric)
    if isinstance(f, SuperFP):
        return _superfp_label(f.man_bits, f.exp_bits, f.normal_binades, f.bias)
    if isinstance(f, BinaryK):
        return _binaryK_label(f.K, f.P, _bias(f), f.is_signed, f.subnormals)
    return repr(f)


def _block_label(fmt: BlockFormat) -> str:
    """The format as a short label in constructor form, for messages."""
    rows = f", block_rows={fmt.block_rows}" if fmt.block_rows != 1 else ""
    return f"BlockFormat({_part_label(fmt.elem)}, {_part_label(fmt.scale)}, {fmt.block_size}{rows})"


@lru_cache(maxsize=256)
def _block_storage_findings(
    fmt: BlockFormat, storage: torch.dtype, elementwise: bool
) -> tuple[str | None, str | None]:
    """Return the findings on storing a block format's decoded values in
    ``storage``, as ``(error, warning)``; this only ever warns.

    A decoded value is an element value times its block's scale (times the
    tensor scale, if there is one).

    * With a cast scale, a decoded value is a product rounded in the carrier,
      which in general the storage dtype does not hold, so this always warns.
    * Otherwise ``elem_max`` must have no more significant bits than the
      storage dtype, since a block that saturates stores ``elem_max`` times
      its scale.
    * Under the elementwise rule (a quantizer, or an unpack, of a tensor of
      the storage dtype) with a power-of-two scale, nothing else can go
      wrong. The scale follows the block's largest input, so a decoded value
      is either the input itself, where the scaled element grid is at least
      as fine as the storage dtype's, or a point of a coarser binary grid,
      which the storage dtype also holds.
    * Otherwise (no scale, or the value-set rule, ``elementwise=False``) the
      values the scale's range can produce must not exceed the storage
      dtype's range, and under the value-set rule they must not be more
      finely spaced than the storage dtype's smallest subnormal.
    """
    st = _STORAGE[storage]
    label = _block_label(fmt)
    if fmt.has_tensor_scale:
        return None, (
            f"{label}'s decoded values are products (element x block scale x tensor scale) "
            f"rounded in the carrier, which {st.name} cannot represent in general: storing a "
            f"result in {st.name} rounds them a second time"
        )
    assert fmt.elem_max is not None  # resolved by __post_init__
    top = Fraction(fmt.elem_max)
    if _significant_bits(top) > st.man_bits + 1:
        return None, (
            f"{label}'s largest element value, {fmt.elem_max}, has {_significant_bits(top)} "
            f"significant bits, and {st.name} holds {st.man_bits + 1}: storing a block that "
            "saturates rounds that value a second time"
        )
    if fmt.scale is None:
        lo, hi = 0, 0
    elif elementwise:
        return None, None
    else:
        assert isinstance(fmt.scale, BinaryK)  # a cast scale has a tensor scale: handled above
        lo, hi = -_bias(fmt.scale), (1 << fmt.scale.K) - 2 - _bias(fmt.scale)
    mf = _minifloat(fmt.elem)
    top_exp = fmt.emax + hi
    if top_exp > st.top_exp:
        return None, (
            f"{label} reaches 2^{top_exp}, above {st.name}'s largest finite value: storing a "
            f"result in {st.name} turns the values above it into infinity"
        )
    if mf.super_codes:
        step_exp = mf.super_cutoff + lo  # the smallest supernormal, also the smallest gap
    else:
        step_exp = (-mf.bias if mf.extended else 1 - mf.bias) - mf.man_bits + lo
    if not elementwise and step_exp < st.min_exp:
        return None, (
            f"{label} has values 2^{step_exp} apart at the bottom of its range, finer than "
            f"{st.name}'s smallest subnormal, 2^{st.min_exp}: storing a result in {st.name} "
            "rounds them a second time"
        )
    return None, None
