"""The value types a quantized computation is spelled with: :class:`Quant` for
an elementwise step, :class:`SplitMac` and :class:`FusedMac` for a reduction,
and :class:`Palette` for a set of formats selected per output element.

Every step of a quantized layer is one of exactly two things, and only one of
them needs an arithmetic policy:

* an **elementwise** step (a weight, activation or gradient quantizer) computes
  in the carrier and rounds the result to a format. That is a format plus a
  rounding mode, and it yields a callable: :class:`Quant`.
* a **reduction** step is the only place the *internal* arithmetic is
  simulated, so it is the only place a multiply/accumulate policy means
  anything. That is the GEMM: :class:`SplitMac` and :class:`FusedMac`, named
  after the two policies the kernels implement (``common/gemm_policy.h``).

The vocabulary is therefore two value types that do not grow with the number
of ops, rather than one formats class per layer kind, each with its own slots
and its own bookkeeping of which slots are in use.

Two things follow from spelling a reduction as a value:

* ``accumulate_quant`` and ``fma_quant`` stop being booleans.
  ``SplitMac(mul=f, acc=None)`` *is* the full-precision-sum case, so the state
  where an accumulate format is present but ignored cannot be expressed.
* the palette stops being a separate op family. Any format slot takes a
  ``Number`` or a sequence of them, and a sequence anywhere selects the
  ``_mixed`` op and requires a ``prec_idx``; op choice is a table keyed on
  ``(family, mac kind, is palette)`` rather than eight function names.

Resolution to a ``_GemmSpec`` happens once and is memoized on the value: the
classes are frozen dataclasses, so they hash by value, and :func:`spec_for_mac`
caches on them. Re-deriving a format's defaults and carrier checks costs 2.3 to
6.4 microseconds per call, more than the rest of the Python call path, so an
object API that resolved per call would be slower than the flat wrappers it
sits over.
"""

from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import lru_cache
from typing import Any, cast

import torch

from mptorch.number import (
    AccumulateAlgorithm,
    BinaryK,
    Number,
    RoundMode,
    SuperFP,
)

from .ops import (
    _binaryK_accumulated_spec,
    _binaryK_fma_accumulated_spec,
    _binaryK_fma_mixed_spec,
    _binaryK_fma_spec,
    _binaryK_mixed_spec,
    _binaryK_spec,
    _checked_block_size,
    _checked_carrier,
    _GemmSpec,
    _superfp_accumulated_spec,
    _superfp_fma_accumulated_spec,
    _superfp_fma_mixed_spec,
    _superfp_fma_spec,
    _superfp_mixed_spec,
    _superfp_spec,
    binaryK_quantize,
    binaryK_quantize_,
    superfp_quantize,
    superfp_quantize_,
)

__all__ = ["Quant", "Palette", "SplitMac", "FusedMac"]


# The formats a GEMM kernel exists for. `Number` is the wider vocabulary, the
# base class a fixed-point or block format would join; this is the part of it
# the eight ops implement today, and a Palette validates membership against it.
GemmFormat = BinaryK | SuperFP

# The fields the mixed schemas take once for a whole palette rather than per
# entry: the kernels tabulate only the format *widths* (`BinaryKCommon` and
# `SuperfpCommon` in common/gemm_args.h), so every entry must agree on these.
_SHARED_FIELDS: dict[type, tuple[str, ...]] = {
    BinaryK: ("is_signed", "prng_bits", "saturation", "subnormals"),
    SuperFP: ("is_signed", "prng_bits", "saturation"),
}

MAX_PALETTE = 8  # must equal MAX_GEMM_FORMATS in common/gemm_policy.h


@dataclass(frozen=True, init=False)
class Palette:
    """Up to eight formats of one family, selected per output element.

    A palette is what a GEMM slot holds when the format varies across the
    output: the ``_mixed`` ops take the palette's widths as a table and a
    ``prec_idx`` map that picks one entry per output element. A plain
    sequence of formats works anywhere a ``Palette`` does and is converted to
    one; the class exists for the validation. Every entry must be of the same
    family, and entries that disagree on a field the schema shares across the
    palette (sign, stochastic bits, saturation, subnormals) are rejected here,
    naming the first that does, rather than silently taking entry 0's.

    Args:
        formats (Number or Sequence[Number] or Palette): one format, a
            sequence of up to ``MAX_PALETTE`` formats, or another palette
            (copied).

    Raises:
        ValueError: for an empty sequence, more than eight formats, or entries
            that disagree on a shared field.
        TypeError: for a format family no GEMM kernel takes, or a mix of
            families.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import Palette, SplitMac, qmm
        >>> palette = Palette([BinaryK(8, 4), BinaryK(8, 5), BinaryK(16, 11)])
        >>> len(palette)
        3
        >>> prec_idx = torch.randint(3, (4, 3))
        >>> qmm(torch.randn(4, 8), torch.randn(8, 3), SplitMac(palette, palette),
        ...     prec_idx=prec_idx).shape
        torch.Size([4, 3])
    """

    formats: tuple[GemmFormat, ...]

    def __init__(self, formats: "Number | Sequence[Number] | Palette"):
        if isinstance(formats, Palette):
            formats = formats.formats
        elif isinstance(formats, Number):
            formats = (formats,)
        object.__setattr__(self, "formats", tuple(formats))
        self._validate()

    def _validate(self) -> None:
        if not self.formats:
            raise ValueError("a Palette needs at least one format")
        if len(self.formats) > MAX_PALETTE:
            raise ValueError(
                f"a Palette holds at most {MAX_PALETTE} formats, got {len(self.formats)}"
            )
        first = self.formats[0]
        if type(first) not in _SHARED_FIELDS:
            raise TypeError(f"no GEMM kernel takes {type(first).__name__} formats")
        for i, fmt in enumerate(self.formats[1:], start=1):
            if type(fmt) is not type(first):
                raise TypeError(
                    f"every format in a Palette must be the same type: entry {i} is "
                    f"{type(fmt).__name__}, entry 0 is {type(first).__name__}"
                )
            for name in _SHARED_FIELDS[type(first)]:
                if getattr(fmt, name) != getattr(first, name):
                    raise ValueError(
                        f"a Palette shares {name} across its entries (the kernel tabulates "
                        f"only the format widths): entry {i} has {name}={getattr(fmt, name)!r}, "
                        f"entry 0 has {getattr(first, name)!r}"
                    )

    def __len__(self) -> int:
        return len(self.formats)

    def __iter__(self):
        return iter(self.formats)

    def __getitem__(self, i: int) -> GemmFormat:
        return self.formats[i]


def _as_palette(slot: "Number | Sequence[Number] | Palette | None") -> Palette | None:
    """Normalize a format slot to a ``Palette``; ``None`` means the step is not
    quantized and stays ``None``."""
    return None if slot is None else Palette(slot)


def _varies(slot: Palette | None) -> bool:
    """Whether this slot asks for a ``_mixed`` op, that is, holds more than one
    format."""
    return slot is not None and len(slot) > 1


# --- an elementwise quantizer ------------------------------------------------


@dataclass(frozen=True)
class Quant:
    """A format plus a rounding mode, as a callable ``Tensor -> Tensor``.

    Calling it rounds every element of a tensor to the format in the carrier
    and returns a tensor of the same dtype and shape. This is what every
    ``*_quant`` slot of a ``QAffineFormats`` / ``QMatmulFormats`` takes, and
    what a straight-through :class:`mptorch.quant.Quantizer` or a QAT observer
    wraps. The call is bound at construction, so a call pays only the op; the
    class is frozen so that a ``Quant`` can be a dictionary key and a
    memoization key like the mac objects.

    Args:
        fmt (Number): the format to round to, a :class:`mptorch.BinaryK` or
            :class:`mptorch.SuperFP`.
        rounding (RoundMode): the rounding mode. Default: ``RoundMode.RNE``
        carrier (torch.dtype, optional): the float arithmetic the cast rounds
            in, :func:`mptorch.quant.binaryK_quantize`'s argument of that
            name. ``None`` rounds in the tensor's own default carrier
            (binary32 for float16, bfloat16 and float32 tensors, binary64 for
            float64 ones); ``torch.float64`` rounds in binary64 whatever the
            tensor; ``torch.float32`` rounds in binary32 and refuses a float64
            tensor. Default: ``None``
        inplace (bool): round the tensor in place and return it, through
            :func:`mptorch.quant.binaryK_quantize_` /
            :func:`mptorch.quant.superfp_quantize_`, instead of allocating a
            result. For a tensor the caller owns (a weight quantized once at
            load); never for a ``*_quant`` slot of a layer's formats, whose
            input belongs to the graph of whatever produced it. The call then
            refuses what the in-place ops refuse: a tensor that is not
            contiguous, a CUDA view off a 16-byte boundary, a ``carrier``
            wider than the tensor, an MPS tensor. Default: ``False``

    Raises:
        TypeError: for a format with no elementwise quantizer, or a
            ``carrier`` that is not a ``torch.dtype``.
        ValueError: for a ``carrier`` dtype that names no carrier.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK, RoundMode, SuperFP
        >>> from mptorch.quant import QAffineFormats, Quant
        >>> q = Quant(BinaryK(8, 4))
        >>> q(torch.tensor([1.0, 0.3, 0.7]))
        tensor([1.0000, 0.3125, 0.6875])
        >>> formats = QAffineFormats()
        >>> formats.input_quant = Quant(SuperFP(3, 4, 8, 7), RoundMode.SR)
    """

    fmt: Number
    rounding: RoundMode = RoundMode.RNE
    _: KW_ONLY
    carrier: torch.dtype | None = None
    inplace: bool = False
    _call: Callable[[torch.Tensor], torch.Tensor] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        fmt, rm, carrier = self.fmt, self.rounding, _checked_carrier(self.carrier)
        if isinstance(fmt, BinaryK):
            binaryK = binaryK_quantize_ if self.inplace else binaryK_quantize

            def call(x: torch.Tensor) -> torch.Tensor:
                return binaryK(
                    x,
                    K=fmt.K,
                    P=fmt.P,
                    bias=fmt.bias,
                    prng_bits=fmt.prng_bits,
                    is_signed=fmt.is_signed,
                    rounding_mode=rm,
                    saturation_mode=fmt.saturation,
                    subnormals_mode=fmt.subnormals,
                    carrier=carrier,
                )

        elif isinstance(fmt, SuperFP):
            superfp = superfp_quantize_ if self.inplace else superfp_quantize

            def call(x: torch.Tensor) -> torch.Tensor:
                return superfp(
                    x,
                    man_bits=fmt.man_bits,
                    exp_bits=fmt.exp_bits,
                    normal_binades=fmt.normal_binades,
                    bias=fmt.bias,
                    prng_bits=fmt.prng_bits,
                    is_signed=fmt.is_signed,
                    rounding_mode=rm,
                    saturation_mode=fmt.saturation,
                    carrier=carrier,
                )

        else:
            raise TypeError(f"no elementwise quantizer for {type(fmt).__name__}")
        object.__setattr__(self, "_call", call)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._call(x)


# --- the two reduction policies ----------------------------------------------


def _checked_accumulation(
    mac: "SplitMac | FusedMac", slots: tuple[Palette | None, ...], has_product: bool
) -> None:
    """Validate a mac's ``accumulate_algorithm``, ``block_size`` and ``outer``.

    ``slots`` is the mac's normalized format slots, the last of them the one
    whose family ``outer`` must share (the accumulate format, else the
    multiply's; a fused mac's one format). The block-size rule is the flat
    wrappers' (:func:`mptorch.quant.ops._checked_block_size`), so the two
    tiers raise the same errors.
    """
    outer = mac.outer
    if outer is not None and not isinstance(outer, GemmFormat):
        raise TypeError(
            f"outer must be one BinaryK or SuperFP format or None, got {type(outer).__name__}: "
            "a palette of outer formats is not implemented"
        )
    _checked_block_size(mac.accumulate_algorithm, mac.block_size, outer is not None, has_product)
    if mac.accumulate_algorithm is AccumulateAlgorithm.NAIVE:
        return
    if any(_varies(slot) for slot in slots):
        raise ValueError(
            f"AccumulateAlgorithm.{mac.accumulate_algorithm.name} is implemented for a single "
            "format per slot: the palette ops accumulate with NAIVE only (a palette of "
            "accumulators is a follow-up of dev/continuation_plan.md's phase B)"
        )
    family = next((slot[0] for slot in reversed(slots) if slot is not None), None)
    if outer is not None and family is not None and type(outer) is not type(family):
        raise TypeError(
            f"outer must be of the mac's format family: got {type(outer).__name__} over "
            f"{type(family).__name__}. Mixing families is roadmap item R-1"
        )


@dataclass(frozen=True)
class SplitMac:
    """Quantize the product and the running sum separately: two roundings.

    Each step of a dot product computes ``s = round_acc(s + round_mul(a * b))``
    in the carrier, the product rounded to the ``mul`` format and the new
    partial sum to the ``acc`` format. ``mul`` and ``acc`` are each a format,
    a sequence of formats (a palette, which selects the ``_mixed`` op and
    requires a ``prec_idx`` at the call), or, for ``acc`` alone, ``None``,
    which leaves the running sum in the carrier's full precision. They must
    belong to the same family: no kernel implements a binaryK multiply with a
    superfp accumulate.

    ``rounding`` is one mode for both halves because the kernels take the
    rounding mode as a template parameter (a runtime switch inside the
    accumulate step cost 1.2 to 2.3 times the kernel time), and a per-format
    mode would square the number of kernel instantiations. Saturation,
    subnormals and the stochastic-rounding width *are* per format, and live
    on the format objects.

    Args:
        mul (Number or Sequence[Number] or Palette): the multiply format, or a
            palette of them.
        acc (Number or Sequence[Number] or Palette, optional): the accumulate
            format, or a palette of them; ``None`` leaves the running sum
            unrounded. Default: ``None``
        rounding (RoundMode): rounding mode of both halves.
            Default: ``RoundMode.RNE``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed. ``NAIVE`` is the step above, in order. ``KAHAN`` carries a
            compensation term in the ``acc`` format alongside the sum.
            ``BLOCK`` runs the step above over blocks of ``block_size``
            products and folds each block's sum into a total with a third
            rounding, ``outer``. ``TREE`` sums each block's products
            pairwise, rounding every pair's sum to ``acc``, and folds the
            block's root the same way. :class:`mptorch.AccumulateAlgorithm`
            states each precisely. The three are binary32-only so far:
            float64 operands and ``carrier=torch.float64`` raise, and so does
            a palette in any slot. Default: ``AccumulateAlgorithm.NAIVE``
        carrier (torch.dtype, optional): the float arithmetic both halves and
            everything between them are computed in,
            :func:`mptorch.quant.binaryK_matmul`'s argument of that name.
            ``None`` takes the operands' (binary64 for float64, binary32 for
            the rest); ``torch.float64`` widens narrower operands to binary64
            and narrows the result back to their dtype; ``torch.float32``
            refuses float64 operands. Every call holds the formats to that
            carrier. Default: ``None``
        block_size (int, optional): products per block. ``BLOCK`` requires
            one, which must divide 16 or be a multiple of 16; ``TREE`` takes a
            power of two in [2, 256] and defaults to 16; ``NAIVE`` and
            ``KAHAN`` take none. Default: ``None``
        outer (Number, optional): the format ``BLOCK`` and ``TREE`` round the
            total to after each block is folded in, of the mac's family.
            ``None`` leaves the total in the carrier's precision, as
            ``acc=None`` leaves the sum. When given, it is the format the
            result holds. Default: ``None``

    Raises:
        ValueError: if ``mul`` is ``None``, if both slots are palettes of
            different lengths, if ``carrier`` names no carrier, or if
            ``block_size`` or ``outer`` does not fit the algorithm.
        TypeError: if ``mul`` and ``acc`` are of different families, ``outer``
            is of another family or not a single format, or ``carrier`` is not
            a ``torch.dtype``.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import SplitMac, qmm
        >>> e4m3 = BinaryK(8, 4)
        >>> mac = SplitMac(e4m3, BinaryK(16, 11))   # E4M3 products, 11-bit sums
        >>> qmm(torch.randn(4, 8), torch.randn(8, 3), mac).shape
        torch.Size([4, 3])
        >>> exact_sum = SplitMac(e4m3, None)   # products rounded, sum exact
        >>> from mptorch import AccumulateAlgorithm
        >>> blocked = SplitMac(e4m3, e4m3, accumulate_algorithm=AccumulateAlgorithm.BLOCK,
        ...                    block_size=8, outer=BinaryK(16, 11))
        >>> qmm(torch.randn(4, 64), torch.randn(64, 3), blocked).shape
        torch.Size([4, 3])
    """

    mul: "Number | Sequence[Number] | Palette"
    acc: "Number | Sequence[Number] | Palette | None" = None
    _: KW_ONLY
    rounding: RoundMode = RoundMode.RNE
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE
    carrier: torch.dtype | None = None
    block_size: int | None = None
    outer: Number | None = None

    def __post_init__(self) -> None:
        _checked_carrier(self.carrier)
        mul, acc = _as_palette(self.mul), _as_palette(self.acc)
        if mul is None:
            raise ValueError("a SplitMac needs a multiply format")
        if acc is not None:
            if type(acc[0]) is not type(mul[0]):
                raise TypeError(
                    f"a SplitMac's multiply and accumulate must be the same format family: "
                    f"got {type(mul[0]).__name__} and {type(acc[0]).__name__}. Mixing families "
                    "is roadmap item R-1 -- no kernel implements it"
                )
            if _varies(mul) and _varies(acc) and len(mul) != len(acc):
                raise ValueError(
                    "a SplitMac's multiply and accumulate palettes must have the same length, "
                    f"got {len(mul)} and {len(acc)}"
                )
        _checked_accumulation(self, (mul, acc), has_product=True)
        object.__setattr__(self, "mul", mul)
        object.__setattr__(self, "acc", acc)


@dataclass(frozen=True)
class FusedMac:
    """One hardware-style fused multiply-add per K-step, rounded once.

    Each step computes ``s = round_fma(s + a * b)`` with the product exact,
    so there is one rounding per step where :class:`SplitMac` has two.
    ``fma`` is a format, a palette, or ``None`` for an unrounded fused step
    (the carrier's own FMA, in order). ``None`` carries no format for
    ``prec_idx`` to select, so it never selects a ``_mixed`` op.

    Args:
        fma (Number or Sequence[Number] or Palette, optional): the format each
            fused step is rounded to, a palette of them, or ``None``.
        rounding (RoundMode): rounding mode of the fused step.
            Default: ``RoundMode.RNE``
        accumulate_algorithm (AccumulateAlgorithm): as for :class:`SplitMac`,
            with the fused step in place of the split one, except ``TREE``,
            which raises: a fused multiply-add has no product term to sum
            pairwise. Default: ``AccumulateAlgorithm.NAIVE``
        carrier (torch.dtype, optional): as for :class:`SplitMac`.
            Default: ``None``
        block_size (int, optional): as for :class:`SplitMac`. Default: ``None``
        outer (Number, optional): as for :class:`SplitMac`; with ``fma=None``
            it may be of either family. Default: ``None``

    Raises:
        ValueError: if ``carrier`` names no carrier, a palette is invalid, or
            ``block_size`` or ``outer`` does not fit the algorithm.
        TypeError: if ``carrier`` is not a ``torch.dtype``, or ``outer`` is of
            another family or not a single format.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import FusedMac, qmm
        >>> qmm(torch.randn(4, 8), torch.randn(8, 3), FusedMac(BinaryK(8, 4))).shape
        torch.Size([4, 3])
        >>> unrounded = FusedMac(None)   # unrounded FMAs in the carrier
    """

    fma: "Number | Sequence[Number] | Palette | None"
    _: KW_ONLY
    rounding: RoundMode = RoundMode.RNE
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE
    carrier: torch.dtype | None = None
    block_size: int | None = None
    outer: Number | None = None

    def __post_init__(self) -> None:
        _checked_carrier(self.carrier)
        fma = _as_palette(self.fma)
        _checked_accumulation(self, (fma,), has_product=False)
        object.__setattr__(self, "fma", fma)


Mac = SplitMac | FusedMac


# --- resolution to the op ----------------------------------------------------
#
# One table keyed on (family, mac kind, does a slot vary) instead of eight
# function names. A slot of one format is spelled as scalars, which is both
# what a single-format op takes and what a `_mixed` op broadcasts across the
# other slot's palette, so a palette on one side of a SplitMac and a single
# format on the other needs no special case.


def _resolved(slot: object) -> Palette:
    """The ``Palette`` a mac's ``__post_init__`` put in one of its slots.

    The declared field types are what a *caller* may pass (a format, a
    sequence, a Palette); what is stored is always the normalized form, and
    this is where that invariant is stated for the reader and the type
    checker.
    """
    assert isinstance(slot, Palette), f"mac slot was not normalized: {slot!r}"
    return slot


def _slot_fields(pal: Palette, prefix: str, n: int) -> dict[str, Any]:
    """One format slot as the schema arguments that name it.

    ``n`` is the call's palette size, the longer of the two slots, since a
    ``SplitMac`` may vary one and not the other. A slot of one format opposite
    a palette is repeated to that length rather than left a scalar: the mixed
    schemas take the palette size from the *first* pair of lists, so a scalar
    there would leave the op with no length to read.
    """
    first = pal[0]
    # Every entry is `first`'s type; Palette._validate is what guarantees it.
    entries = list(pal.formats) if len(pal) > 1 else [first] * n
    widths: dict[str, list]
    if isinstance(first, BinaryK):
        bk = cast(list[BinaryK], entries)
        widths = {
            f"{prefix}_K": [f.K for f in bk],
            f"{prefix}_P": [f.P for f in bk],
            f"{prefix}_bias": [f.bias for f in bk],
        }
    else:
        sfp = cast(list[SuperFP], entries)
        widths = {
            f"{prefix}_man_bits": [f.man_bits for f in sfp],
            f"{prefix}_exp_bits": [f.exp_bits for f in sfp],
            f"{prefix}_normal_binades": [f.normal_binades for f in sfp],
            f"{prefix}_bias": [f.bias for f in sfp],
        }
    fields: dict[str, Any] = widths if n > 1 else {k: v[0] for k, v in widths.items()}
    fields[f"{prefix}_is_signed"] = first.is_signed
    fields[f"{prefix}_prng_bits"] = first.prng_bits
    return fields


def _outer_fields(outer: Number | None) -> dict[str, Any]:
    """A mac's outer format as the schema arguments that name it; none for ``None``."""
    if outer is None:
        return {}
    # `_checked_accumulation` has held it to the two families the ops take.
    assert isinstance(outer, GemmFormat), f"mac.outer was not validated: {outer!r}"
    fields: dict[str, Any] = {
        "outer_bias": outer.bias,
        "outer_is_signed": outer.is_signed,
        "outer_prng_bits": outer.prng_bits,
        "outer_saturation_mode": outer.saturation,
    }
    if isinstance(outer, BinaryK):
        return fields | {
            "outer_K": outer.K,
            "outer_P": outer.P,
            "outer_subnormals_mode": outer.subnormals,
        }
    return fields | {
        "outer_man_bits": outer.man_bits,
        "outer_exp_bits": outer.exp_bits,
        "outer_normal_binades": outer.normal_binades,
    }


@lru_cache(maxsize=256)
def spec_for_mac(mac: Mac) -> _GemmSpec:
    """The resolved GEMM this mac names, as a ``_GemmSpec``.

    It is the same ``_GemmSpec`` the flat ``ops.py`` wrapper for that op
    builds from the same numbers; ``tests/test_number_formats.py`` asserts
    the equality, which is what keeps the two API tiers from drifting apart.

    Memoized on the mac value, which is the reason the mac classes are frozen
    (and so hashable by value): an ad-hoc ``qmatmul(a, b, SplitMac(...))`` in
    a loop would otherwise re-derive every default on every call, at 2.3 to
    6.4 microseconds, more than an object API saves. A cache miss costs
    exactly what the flat wrapper costs.
    """
    fields: dict[str, Any] = {
        "accumulate_algorithm": mac.accumulate_algorithm,
        "rounding_mode": mac.rounding,
        "carrier": mac.carrier,
    }
    # A NAIVE mac resolves through the builders it always resolved through.
    # Any other algorithm is single-format (`_checked_accumulation`) and goes
    # through the builder of its family's `*_accumulated` op, which also takes
    # the block size and the outer format.
    naive = mac.accumulate_algorithm is AccumulateAlgorithm.NAIVE
    if not naive:
        fields |= {"block_size": mac.block_size} | _outer_fields(mac.outer)

    if isinstance(mac, FusedMac):
        if mac.fma is None:
            # Nothing is rounded per step, so no cast reads the widths and the
            # family is the outer format's, if there is one; without one it
            # does not matter, and binaryK's op is as good as superfp's.
            if naive:
                return _binaryK_fma_spec(fma_K=2, fma_P=1, fma_quant=False, **fields)
            if isinstance(mac.outer, SuperFP):
                return _superfp_fma_accumulated_spec(
                    fma_man_bits=1,
                    fma_exp_bits=2,
                    fma_normal_binades=1,
                    fma_bias=1,
                    fma_quant=False,
                    **fields,
                )
            return _binaryK_fma_accumulated_spec(fma_K=2, fma_P=1, fma_quant=False, **fields)
        fma = _resolved(mac.fma)
        fma_0 = fma[0]
        binaryK = isinstance(fma_0, BinaryK)
        fields |= _slot_fields(fma, "fma", len(fma))
        fields["saturation_mode"] = fma_0.saturation
        if isinstance(fma_0, BinaryK):
            fields["subnormals_mode"] = fma_0.subnormals
        if _varies(fma):
            build = _binaryK_fma_mixed_spec if binaryK else _superfp_fma_mixed_spec
            return build(**fields)
        if naive:
            build = _binaryK_fma_spec if binaryK else _superfp_fma_spec
        else:
            build = _binaryK_fma_accumulated_spec if binaryK else _superfp_fma_accumulated_spec
        return build(fma_quant=True, **fields)

    mul = _resolved(mac.mul)
    acc = None if mac.acc is None else _resolved(mac.acc)
    mul_0 = mul[0]
    # One palette size for the call: a SplitMac may vary its multiply, its
    # accumulate, or both, and the op takes a single tabulated length.
    n = max(len(mul), len(acc) if acc is not None else 1)
    fields |= _slot_fields(mul, "mul", n)
    fields["saturation_mode"] = mul_0.saturation
    if isinstance(mul_0, BinaryK):
        fields["subnormals_mode"] = mul_0.subnormals
    fields["accumulate_quant"] = acc is not None
    if acc is not None:
        acc_0 = acc[0]
        fields |= _slot_fields(acc, "acc", n)
        fields["acc_saturation_mode"] = acc_0.saturation
        if isinstance(acc_0, BinaryK):
            fields["acc_subnormals_mode"] = acc_0.subnormals
    binaryK = isinstance(mul_0, BinaryK)
    if n > 1:
        build = _binaryK_mixed_spec if binaryK else _superfp_mixed_spec
        return build(**fields)
    if naive:
        build = _binaryK_spec if binaryK else _superfp_spec
    else:
        build = _binaryK_accumulated_spec if binaryK else _superfp_accumulated_spec
    return build(**fields)
