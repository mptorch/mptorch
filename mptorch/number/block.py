"""Block formats, and the code layout their kernels encode and decode.

A block format stores a tensor as narrow codes that share one scale per
block: OCP's MX formats (Microscaling Formats v1.0: an E8M0 power-of-two
scale per 32 elements) and NVIDIA's NVFP4 (an E4M3 scale per 16 elements,
times a float32 scale per tensor). The element and the scale are each a
*minifloat code*: a sign-magnitude layout of `exp_bits` exponent and
`man_bits` mantissa bits over a bias, with exponent field 0 holding
subnormals (or, under `SubnormalsMode.EXTENDED_NORMALS`, one more binade of
normals). A `BinaryK` names that layout, so one value type covers every
element and scale the formats use, and `csrc/common/block_decode.h` has one
decode and one encode for all of them.

The element is rounded by the `BinaryK` cast of its own format under
`SAT_FINITE` and then clamped to `elem_max`, which is where the OCP formats
part from P3109's: E4M3's `SAT_FINITE` top is 480 and OCP's 448, since OCP
spends the code above on NaN; E5M2's are 114688 and 57344, below its
infinity. The clamp is exact in every rounding mode: every value between
`elem_max` and the format's own top rounds to a code the clamp folds onto
`elem_max`. A scale with no mantissa (`P = 1`) is read as E8M0 is, a power
of two `2**(code - bias)` for every code but the all-ones one, its NaN, and
is set from the block's largest magnitude by integer arithmetic, not by a
cast; a scale with a mantissa is cast (to nearest even, under `SAT_FINITE`)
and spends its all-ones magnitude code on NaN in the same way.

The block kernels round in binary32 or binary64, the call's carrier, and
assemble decoded values on its word, so a block format's element, and a
scale that is cast, must be formats that carrier holds entirely, and a
power-of-two scale's codes must lie in its range. `BlockFormat` raises when
it is built only for what binary64 cannot do, as `BinaryK` raises only for
what no carrier can, and keeps binary32's verdict, which a call that rounds
in binary32 raises (`check_block_carrier`).
"""

from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from typing import NamedTuple

import torch

from ._carrier import _BINARY32, _BINARY64, _CARRIERS, _top_of_range
from ._storage import _STORAGE
from .base import Number, _report_per_call
from .binaryk import BinaryK, _binaryK_findings
from .fixedpoint import FixedPoint, _fixedpoint_findings
from .modes import SaturationMode, SubnormalsMode
from .superfp import SuperFP, _superfp_findings


class ScaleRounding(Enum):
    """How a :class:`BlockFormat` chooses a block's scale from the block's
    largest magnitude ``amax``.

    Mirrors ``ScaleRounding`` in ``csrc/common/block_decode.h``, by name and
    value. Two quantities are used below:

    * ``r = amax / elem_max`` is the scale that would put ``amax`` exactly on
      ``elem_max``. For a cast scale it is also divided by the tensor scale.
    * ``T`` is ``elem_max`` plus half the element format's step above it (7
      for E2M1, whose largest values are 4 and 6). It is the largest scaled
      element that rounding to nearest would still take to ``elem_max``.

    Every rule's result is clamped to the scale's codes, and a zero cast
    scale is raised to the smallest positive one. No rule depends on the
    elements' rounding mode. With E2M1 elements and a power-of-two scale, a
    block's largest element lands, in E2M1's units, in ``[4, 8)`` under
    :attr:`OCP`, ``[4.5, 9]`` under :attr:`NEAREST`, ``(3, 6]`` under
    :attr:`UP` and ``(3.5, 7]`` under :attr:`SELECTIVE`. Anything above 6
    saturates to 6.

    The three ``FOUR_OVER_SIX`` rules (Cook et al., `Four Over Six
    <https://arxiv.org/abs/2512.02010>`__) read the block's elements, not only
    ``amax``. A block has two candidate scales:

    * :attr:`NEAREST`'s scale, under which the block's largest element lands
      on ``elem_max`` (6 for E2M1);
    * the scale nearest to ``r * elem_max / 2**emax``, under which it lands
      on ``elem_max``'s power of two (4).

    The block is quantized under both, in the call's rounding mode, and takes
    the second scale if its error is strictly the smaller. The three rules
    differ in the error they compare. They need a cast scale, because a power
    of two cannot hold the factor of 1.5 between the candidates. With
    ``block_rows > 1`` the block is the whole tile: it has one pair of
    candidates, the error is taken over all of its elements, and it makes
    one choice. Under these rules the default tensor scale gives the
    tensor's largest block the scale ``top_scale`` instead of ``scale_max``
    (256 instead of 448 for E4M3), so that this block also has both
    candidates. Errors are counted in fixed point, in units of
    ``2**(floor(log2(amax)) - 22)``. A block's error is therefore exact, and
    the choice is the same on every backend, in whatever order the elements
    are added up. A square tile therefore also makes the same choice
    whichever of its two axes the tensor is packed along.

    Example::

        >>> import dataclasses
        >>> from mptorch import MXFP4_E2M1, ScaleRounding
        >>> MXFP4_E2M1.scale_rule
        <ScaleRounding.OCP: 0>
        >>> dataclasses.replace(MXFP4_E2M1, scale_rounding=ScaleRounding.UP).scale_rule
        <ScaleRounding.UP: 2>
    """

    #: OCP MX's rule, for a power-of-two scale only: ``2**(floor(log2(amax)) -
    #: emax)``. A block's largest element can pass ``elem_max`` by up to
    #: ``2**(emax + 1) / elem_max`` (4/3 for E2M1) and saturate. The default
    #: for a power-of-two scale (E8M0).
    OCP = 0
    #: ``r`` rounded to nearest even in the scale format. The scale can be up
    #: to half a scale step below ``r``. In E4M3's normals that is a factor
    #: of 1.0625, which E2M1's top step absorbs. Where the scale's values are
    #: powers of two it is a factor of 1.5, which saturates the largest
    #: element by up to a third. NVFP4's rule, and the default for a binaryK
    #: scale with a mantissa (E4M3).
    NEAREST = 1
    #: The smallest scale that is at least ``r``. No element passes
    #: ``elem_max``, but the grid can be up to twice as coarse for the rest
    #: of the block. The rule that NVIDIA's MXFP8 pretraining recipe uses for
    #: E8M0 (arXiv 2506.08027).
    UP = 2
    #: :attr:`NEAREST`, or the next scale up if the block's largest element
    #: would otherwise land above ``T``. That is the only case where the
    #: clamp at ``elem_max`` moves the element further than rounding to
    #: nearest does. With a power-of-two scale this is the smallest scale
    #: with ``amax / scale <= T``. The default for a :class:`SuperFP` scale,
    #: whose values below its normal binades are powers of two.
    SELECTIVE = 3
    #: Four Over Six by the sum of squared errors, which the paper finds best
    #: for post-training quantization.
    FOUR_OVER_SIX_MSE = 4
    #: Four Over Six by the sum of absolute errors, which the paper finds
    #: best for pre-training.
    FOUR_OVER_SIX_MAE = 5
    #: Four Over Six by the largest absolute error.
    FOUR_OVER_SIX_ABS_MAX = 6

    @property
    def is_four_over_six(self) -> bool:
        """Whether this is one of the three ``FOUR_OVER_SIX`` rules."""
        return self.value >= ScaleRounding.FOUR_OVER_SIX_MSE.value


@dataclass(frozen=True)
class BlockFormat(Number):
    """A block format: narrow element codes that share one scale per block.

    Covers OCP's Microscaling (MX) formats and NVIDIA's NVFP4, and the same
    construction over :class:`SuperFP` elements and scales. A tensor is split
    into blocks of ``block_size`` consecutive elements along one axis, the
    *packed* axis. The functions that take a block format name that axis with
    ``axis``. Each block stores one ``scale`` code and ``block_size`` element
    codes, and an element decodes to its ``elem`` value times its block's
    scale. :func:`mptorch.quant.block_quantize` rounds a tensor to the
    format, :func:`mptorch.quant.block_pack` stores it in ``uint8`` codes, and
    :func:`mptorch.quant.block_matmul` multiplies two packed operands.

    The kind of ``scale`` decides whether the tensor has a second scale. The
    tensor scale is not an option of its own:

    - A *power-of-two* scale (``scale.P == 1``: E8M0, the MX presets) reaches
      from ``2**-127`` to ``2**127``, so the block scale is enough. The
      format has no tensor scale, and ``tensor_scale=`` raises.
    - A *cast* scale is a binaryK with a mantissa (NVFP4's E4M3) or a
      superfp. It follows a block's largest magnitude more closely, but over
      a much shorter range (E4M3: ``2**-9`` to 448). The format therefore
      also has one scale per tensor, a value of the carrier (float32 in
      binary32, float64 in binary64), which brings the tensor into that
      range. A block's scale is then its scale code's value times the tensor
      scale, one product in the carrier. The functions that quantize or pack
      take it as ``tensor_scale`` (its default is given below).
      ``tensor_scale=1.0`` leaves the block scales alone.
    - ``scale=None`` stores the element codes unscaled.

    A tensor scale without block scales is not available.

    With ``block_rows > 1`` a scale is shared by a *tile* of ``block_rows``
    consecutive rows of ``block_size`` elements. The rows run along the axis
    before the packed one, never across a batch dimension. A square tile
    (``block_rows == block_size``) quantizes a matrix to the same values
    whichever of its two axes is packed, in every deterministic rounding
    mode. One packed weight can therefore serve a layer's forward pass and
    its input gradient, which reads it transposed.
    ``dataclasses.replace(NVFP4, block_rows=16)`` is NVFP4 with 16 x 16
    weight tiles.

    **Choosing the scale.** The block scale is chosen from the block's
    largest magnitude ``amax`` by ``scale_rounding``
    (:class:`ScaleRounding`). The defaults are:

    - A power-of-two scale (``scale.P == 1``, E8M0) is
      ``2**(floor(log2(amax)) - emax)``, clamped to its codes, where
      ``emax = floor(log2(elem_max))``. This is OCP MX's rule.
    - A ``BinaryK`` scale with a mantissa (E4M3) is
      ``amax / (elem_max * tensor_scale)`` rounded to nearest even by the
      scale format's cast. This is NVFP4's rule.
    - A ``SuperFP`` scale rounds that ratio the same way, except where the
      block's largest element would then saturate further than rounding to
      nearest allows. There it takes the next scale up.

    A cast scale is clamped to its largest value and raised to its smallest
    positive one. The tensor scale defaults to
    ``max(|x|) / (elem_max * top_scale)`` over the whole tensor ``x``.

    **Rounding the elements.** Each element is divided by its scale and
    rounded, in the call's rounding mode, by ``elem``'s cast (binaryK or
    superfp) under ``SaturationMode.SAT_FINITE``. It is then clamped to
    ``elem_max``. ``elem``'s own ``saturation`` is not read; its
    ``subnormals`` and ``prng_bits`` are.

    **Special values.** An infinite input saturates to ``elem_max``. A NaN
    input takes ``nan_code`` when the element format has one. Otherwise the
    block's scale becomes the scale format's NaN, and the whole block (every
    element of a 2D tile) decodes to NaN. With neither (``scale=None`` and no
    ``nan_code``) the NaN becomes ``elem_max``. No value decodes to ``-0.0``.

    **Superfp codes.** A superfp code follows its format's layout: the normal
    binades' codes as in a binaryK, and below them, from code 1, the
    supernormal powers of two (see :class:`SuperFP`). It spends no code on
    NaN or infinity unless ``nan_code`` or ``inf_code`` says so, as for a
    binaryK element. A superfp scale spends its all-ones code on NaN, like
    every cast scale.

    **Carriers.** The block ops round in binary32, or in binary64 for a
    float64 tensor or ``carrier=torch.float64``, and a format has to be one
    the call's carrier holds: its element and a cast scale entirely, and a
    power-of-two scale's codes within ``2**-127 .. 2**127`` in binary32 and
    ``2**-1023 .. 2**1023`` in binary64. Building a format checks binary64's
    bounds, and a call that rounds in binary32 checks binary32's, so an
    element with nine or ten exponent bits, or an E8M0-style scale biased
    past binary32's range, is a format of float64 tensors.

    Args:
        elem (BinaryK or SuperFP): the element format. At most 16 bits, and a
            format that binary64 holds entirely (binary32 too, for a call that
            rounds in it).
        scale (BinaryK or SuperFP, optional): the scale format, one byte. It
            is E8M0 (``BinaryK(8, 1, bias=127, is_signed=False,
            subnormals=SubnormalsMode.EXTENDED_NORMALS)``), a binaryK with a
            mantissa such as E4M3, or a superfp. The last two also bring a
            per-tensor scale. ``None`` stores the element codes unscaled.
        block_size (int): elements per block along the packed axis. A power
            of two in [2, 128], with ``elem_bits * block_size`` a multiple of
            8.
        block_rows (int): rows per scale tile. A power of two in [1, 128],
            with ``block_rows * block_size <= 16384``; 1 without a scale.
            Default: ``1``
        elem_max (float, optional): the largest magnitude an element takes, a
            value of ``elem``. Default: ``None``, ``elem``'s largest finite
            value under ``SAT_FINITE``.
        nan_code (int, optional): the magnitude code (sign bit clear) that
            the element format spends on NaN, above ``elem_max``'s. Default:
            ``None``
        inf_code (int, optional): the magnitude code it spends on infinity.
            It is decoded but never produced. Default: ``None``
        scale_rounding (ScaleRounding, optional): how a block's scale is
            chosen. ``OCP`` needs a power-of-two scale. A ``FOUR_OVER_SIX``
            rule needs a cast scale. Default:
            ``None``, the scale's own rule: ``OCP`` for a power-of-two scale,
            ``NEAREST`` for a binaryK with a mantissa, ``SELECTIVE`` for a
            superfp. The default is resolved against the scale each time the
            format is built, so
            ``dataclasses.replace(NVFP4, scale=SuperFP(...))`` takes the
            superfp's rule.

    These attributes are derived when the format is built and are read-only:

    - ``scale_rule``: the :class:`ScaleRounding` in effect (``None`` without
      a scale). Formats compare by it, so leaving ``scale_rounding`` at its
      default and naming that default give equal formats.
    - ``elem_bits``: the bits per element code, sign included (``elem.K`` for
      a binaryK, ``elem.wl`` for a fixed point).
    - ``emax``: ``floor(log2(elem_max))``.
    - ``scale_max``: the largest scale, the value of the code below the
      scale's NaN (448 for E4M3, ``2**127`` for E8M0, 0 without a scale).
    - ``top_scale``: the scale that the default tensor scale gives the
      tensor's largest block. It is ``scale_max``, or under a
      ``FOUR_OVER_SIX`` rule the largest scale ``v`` for which
      ``v * elem_max / 2**emax`` is also a scale (256 for E4M3, since 384 is
      one).
    - ``has_tensor_scale``: whether the scale is cast (anything but
      E8M0-style), which brings the per-tensor scale.
    - ``bytes_per_block``: the bytes of element codes in one row of a block.
    - ``is_square``: ``block_rows == block_size``.

    Raises:
        ValueError: for any of the following: sizes outside the ranges above;
            an element over 16 bits or a scale over 8; an element or cast
            scale that binary64 does not hold entirely; an ``elem_max`` that
            is not a value of ``elem``; a ``nan_code`` or ``inf_code`` at or
            below ``elem_max``'s code; a power-of-two scale that is signed or
            reaches past ``2**-1023`` or ``2**1023`` (the pack multiplies by
            its reciprocal); ``scale_rounding=OCP`` on a cast scale; a
            ``FOUR_OVER_SIX`` rule on a power-of-two scale; or any
            ``scale_rounding`` without a scale.
        TypeError: if ``elem`` or ``scale`` is not a ``BinaryK`` or a
            ``SuperFP``.

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
    # power-of-two scale's range), or None: what a call that rounds in
    # binary32 raises (`check_block_carrier`). binary64's verdict raised when
    # the format was built.
    _binary32_error: str | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        elem, scale = self.elem, self.scale
        if not isinstance(elem, BinaryK | SuperFP | FixedPoint):
            raise TypeError(
                f"a BlockFormat's element is a BinaryK, a SuperFP or a FixedPoint, got "
                f"{type(elem).__name__}"
            )
        if isinstance(elem, FixedPoint):
            if not (elem.is_signed and elem.symmetric):
                raise ValueError(
                    f"a FixedPoint block element must be signed and symmetric, got {elem}: a "
                    "block code is sign-magnitude, which has no code for two's complement's "
                    "bottom, and an unsigned element's top code would be its layout's NaN"
                )
            if self.nan_code is not None or self.inf_code is not None:
                raise ValueError(
                    "a FixedPoint block element has a value at every code, so it has no "
                    "nan_code or inf_code"
                )
        mf = _minifloat(elem)
        bits = mf.bits
        if bits > 16:
            raise ValueError(f"a BlockFormat's element codes are at most 16 bits, got {bits}")
        bs, br = self.block_size, self.block_rows
        if not 2 <= bs <= 128 or bs & (bs - 1):
            raise ValueError(f"block_size must be a power of two in [2, 128], got {bs}")
        if (bits * bs) % 8:
            raise ValueError(
                f"a block of {bs} {bits}-bit codes is {bits * bs} bits; a block must fill "
                "whole bytes, so elem_bits * block_size must be a multiple of 8"
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

        # P3109's NaN at the top of an unsigned binaryK; superfp spends no code on one
        reserved = 0 if elem.is_signed or isinstance(elem, SuperFP) else 1
        _, top_exp, top_code = _top_of_range(mf.exp_bits, mf.man_bits, mf.bias, reserved, _BINARY64)
        top = Fraction((1 << mf.man_bits) + top_code) * Fraction(2) ** (top_exp - mf.man_bits)
        if self.elem_max is None:
            elem_max = top
        else:
            elem_max = Fraction(self.elem_max)
            if not 0 < elem_max <= top or _minifloat_code(mf, elem_max) is None:
                raise ValueError(
                    f"elem_max must be a positive value of the element format {elem} no larger "
                    f"than its largest finite value {float(top)}, got {self.elem_max}"
                )
        max_code = _minifloat_code(mf, elem_max)
        assert max_code is not None
        magnitudes = 1 << (bits - int(elem.is_signed))
        for name, code in (("nan_code", self.nan_code), ("inf_code", self.inf_code)):
            if code is not None and not max_code < code < magnitudes:
                raise ValueError(
                    f"{name} is a magnitude code above elem_max's ({max_code:#x}) and below "
                    f"{magnitudes:#x}, which the encoder never produces for a value; got {code}"
                )
        if self.nan_code is not None and self.nan_code == self.inf_code:
            raise ValueError("nan_code and inf_code must differ")

        rounding = self.scale_rounding
        if rounding is not None and not isinstance(rounding, ScaleRounding):
            raise TypeError(
                f"scale_rounding is a ScaleRounding or None, got {type(rounding).__name__}"
            )
        if scale is None:
            if br != 1:
                raise ValueError("block_rows is the rows a scale is shared by, and needs a scale")
            if rounding is not None:
                raise ValueError("scale_rounding chooses a block's scale, and needs a scale")
            scale_max, tensor_scaled = 0.0, False
        else:
            if not isinstance(scale, BinaryK | SuperFP):
                raise TypeError(
                    "a BlockFormat's scale is a BinaryK, a SuperFP or None, got "
                    f"{type(scale).__name__}"
                )
            smf = _minifloat(scale)
            if smf.bits > 8:
                raise ValueError(f"a BlockFormat's scale is one byte, got {smf.bits} bits")
            if isinstance(scale, BinaryK) and scale.P == 1:
                if scale.is_signed:
                    raise ValueError(
                        "a power-of-two scale (P=1, E8M0) is unsigned: its codes are exponents"
                    )
                lo, hi = -_bias(scale), (1 << scale.K) - 2 - _bias(scale)
                for carrier in (_BINARY64, _BINARY32):
                    if lo < -carrier.top_exp or hi > carrier.top_exp:
                        error = (
                            f"a power-of-two scale's codes are 2^{lo} .. 2^{hi}, and must lie in "
                            f"2^-{carrier.top_exp} .. 2^{carrier.top_exp}, where {carrier.name} "
                            "holds the reciprocal the pack multiplies by as well"
                        )
                        if carrier is _BINARY64:
                            raise ValueError(error)
                        binary32_error = binary32_error or error
                scale_max, tensor_scaled = 2.0**hi, False
                if rounding is not None and rounding.is_four_over_six:
                    raise ValueError(
                        f"ScaleRounding.{rounding.name} chooses between two scales a factor of "
                        "elem_max / 2**emax apart (1.5 for E2M1), which a power-of-two scale "
                        "cannot hold: it takes a cast scale, such as NVFP4's E4M3"
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
                        f"the scale format {scale} has no positive value below its NaN"
                    )
                if rounding is ScaleRounding.OCP:
                    raise ValueError(
                        "ScaleRounding.OCP is OCP MX's rule for a power-of-two scale, and "
                        f"{scale} is a cast scale: use NEAREST, UP or SELECTIVE"
                    )
            if rounding is None:
                if not tensor_scaled:
                    rounding = ScaleRounding.OCP
                elif isinstance(scale, SuperFP):
                    rounding = ScaleRounding.SELECTIVE
                else:
                    rounding = ScaleRounding.NEAREST

        # The scale the default tensor scale gives the tensor's largest block.
        # Under Four Over Six that block needs its second candidate too, a
        # factor of elem_max / 2^emax up, so it is the largest scale whose
        # multiple by that factor is still a scale (256 for E4M3: 384 is one,
        # and 288 * 1.5 = 432 is not).
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
    """Hold a block format against the carrier a call rounds in.

    binary64's verdict on the element, the scale and the scale's range was
    given when ``fmt`` was built, which raised for anything binary64 cannot
    carry. What is left per call is binary32's, kept on the format.

    Raises:
        ValueError: for ``carrier=torch.float32`` when binary32 does not hold
            the element or the cast scale entirely, or a power-of-two scale's
            range; binary64 does, so a float64 tensor or
            ``carrier=torch.float64`` takes the format.
    """
    if carrier is torch.float32 and fmt._binary32_error is not None:
        raise ValueError(
            f"{_block_label(fmt)} rounds in binary32 here, and {fmt._binary32_error}; "
            "binary64 holds it: pack a float64 tensor, or pass carrier=torch.float64"
        )


def check_block_storage(
    fmt: BlockFormat, *, storage: torch.dtype, elementwise: bool = True
) -> None:
    """Hold a block format's decoded values against the dtype a result is written in.

    ``storage`` is ``torch.float16`` or ``torch.bfloat16`` (or ``torch.float32``
    for a result decoded in binary64), whose result rounds a decoded value a
    second time where the dtype does not hold it.
    ``elementwise=True`` is the rule for :func:`mptorch.quant.block_quantize`
    and :func:`mptorch.quant.block_unpack` on a tensor of that dtype (its
    inputs are already the dtype's values); ``False`` holds every value the
    format can decode to. See :func:`check_binaryK_storage` for the two rules.

    Warns:
        FormatRangeWarning: where a decoded value can land off the dtype's grid.
    """
    _report_per_call(*_block_storage_findings(fmt, storage, elementwise))


# --- the code layout ---------------------------------------------------------


class _Minifloat(NamedTuple):
    """A `BinaryK`'s, a `SuperFP`'s or a `FixedPoint`'s code layout, as the
    block kernels encode and decode it."""

    is_signed: bool
    exp_bits: int
    man_bits: int
    bias: int
    extended: bool  # EXTENDED_NORMALS: exponent field 0 is a binade of normals
    # superfp: magnitude codes below `super_codes` are the supernormals, code c
    # the power of two 2**(super_cutoff + c - 1); 0 for a binaryK layout
    super_codes: int = 0
    super_cutoff: int = 0

    @property
    def bits(self) -> int:
        """Bits per code, the sign's included."""
        return int(self.is_signed) + self.exp_bits + self.man_bits


def _bias(f: BinaryK) -> int:
    """A built ``BinaryK``'s bias, which ``__post_init__`` has resolved."""
    assert f.bias is not None
    return f.bias


def _minifloat(f: "BinaryK | SuperFP | FixedPoint") -> _Minifloat:
    """The code layout of ``f``: P3109's field widths in sign-magnitude for a
    ``BinaryK``, for a ``SuperFP`` its fields and its supernormal codes
    (``cast_superfp.h``'s ``superfp_region_cutoffs``), and for a
    ``FixedPoint`` the binaryK layout its sign-magnitude integer code is.

    A ``FixedPoint`` with ``M`` magnitude bits is bit for bit the binaryK
    layout with one exponent bit, ``M - 1`` mantissa bits and bias ``2 - M +
    fl``: field 0, the subnormals, holds ``m * 2**-fl``, and field 1 holds
    ``(2**(M-1) + m) * 2**-fl``, so the code ``c`` is the value ``c *
    2**-fl``. The block kernels decode, round and encode it as that binaryK,
    so no kernel knows about fixed point: one step from zero to the top,
    which the element's ``elem_max`` keeps it to."""
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
    """The magnitude a code (sign bit removed) encodes, specials aside."""
    if mag < mf.super_codes:
        return Fraction(0) if mag == 0 else Fraction(2) ** (mf.super_cutoff + mag - 1)
    e, m = mag >> mf.man_bits, mag & ((1 << mf.man_bits) - 1)
    if e == 0 and not mf.extended:
        return Fraction(m) * Fraction(2) ** (1 - mf.bias - mf.man_bits)
    if e == 0 and m == 0:
        return Fraction(0)  # EXTENDED_NORMALS keeps its binade's foot for the zero
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
    """How many bits the significand of a dyadic ``value`` needs."""
    num = value.numerator
    return (num >> ((num & -num).bit_length() - 1)).bit_length()


# --- what the checks find ----------------------------------------------------


def _block_part_error(
    role: str, f: "BinaryK | SuperFP | FixedPoint", carrier: torch.dtype
) -> str | None:
    """Why a block format cannot use ``f`` as its element or cast scale in
    ``carrier``, or ``None``.

    The block kernels assemble decoded values on the carrier's word, so the
    format rounded with (``f`` under ``SAT_FINITE``) has to be one the
    carrier holds entirely: no error and no range warning. A ``FixedPoint``
    element is held to its own rule, `_fixedpoint_findings`, which is the
    binaryK rule of its layout with the stochastic-rounding bits counted
    against the magnitude bits.
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
    return f"a BlockFormat's {role} must be a format {name} holds entirely: {found}"


def _block_label(fmt: BlockFormat) -> str:
    """The format as a short constructor-like label, for messages."""
    rows = f", block_rows={fmt.block_rows}" if fmt.block_rows != 1 else ""
    return f"BlockFormat({fmt.elem}, {fmt.scale}, {fmt.block_size}{rows})"


@lru_cache(maxsize=256)
def _block_storage_findings(
    fmt: BlockFormat, storage: torch.dtype, elementwise: bool
) -> tuple[str | None, str | None]:
    """What storing a block format's decoded values in ``storage`` says.

    Returns ``(error, warning)``; only warnings arise. A decoded value is an
    element value times its block's scale (times the tensor scale). The
    elementwise rule, for a quantizer or an unpack of a tensor of that dtype:
    with a power-of-two scale the scale follows the block's largest input, so
    a decoded value is the input itself wherever the scaled element grid is
    at least as fine as the dtype's and a point of a coarser binary grid
    elsewhere, a value of the dtype either way, except where a block
    saturates onto ``elem_max`` times its scale, which needs ``elem_max``'s
    significand to fit the dtype's. With a mantissa-bearing scale a decoded
    value is a product in the carrier the dtype does not hold in general. The
    value-set rule (``elementwise=False``) asks the same of every value the
    scale's range can produce.
    """
    st = _STORAGE[storage]
    label = _block_label(fmt)
    if fmt.has_tensor_scale:
        return None, (
            f"{label}'s values are an element times its block's scale times the tensor scale, "
            f"products rounded in the carrier that {st.name} does not hold in general: a {st.name} "
            "result rounds them a second time when it is stored"
        )
    assert fmt.elem_max is not None  # resolved by __post_init__
    top = Fraction(fmt.elem_max)
    if _significant_bits(top) > st.man_bits + 1:
        return None, (
            f"{label}'s largest element value {fmt.elem_max} has {_significant_bits(top)} "
            f"significant bits and {st.name} holds {st.man_bits + 1}: a block that saturates "
            f"stores it rounded a second time"
        )
    if fmt.scale is None:
        lo, hi = 0, 0
    elif elementwise:
        return None, None
    else:
        assert isinstance(fmt.scale, BinaryK)  # a cast scale has a tensor scale, above
        lo, hi = -_bias(fmt.scale), (1 << fmt.scale.K) - 2 - _bias(fmt.scale)
    mf = _minifloat(fmt.elem)
    top_exp = fmt.emax + hi
    if top_exp > st.top_exp:
        return None, (
            f"{label} reaches 2^{top_exp}, above {st.name}'s largest finite value: a "
            f"{st.name} result stores those values as infinity"
        )
    if mf.super_codes:
        step_exp = mf.super_cutoff + lo  # the smallest supernormal, the finest spacing
    else:
        step_exp = (-mf.bias if mf.extended else 1 - mf.bias) - mf.man_bits + lo
    if not elementwise and step_exp < st.min_exp:
        return None, (
            f"{label} spaces its values 2^{step_exp} apart at the bottom, below {st.name}'s "
            f"2^{st.min_exp}: a {st.name} result rounds them a second time when it is stored"
        )
    return None, None
