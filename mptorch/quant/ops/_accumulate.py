"""The accumulate algorithms other than NAIVE: the arguments of the *_accumulated ops.

KAHAN, BLOCK and TREE (`csrc/common/gemm_accumulate.h`) run in four ops of
their own, the `*_accumulated` counterparts of the four single-format ops.
Each takes its counterpart's arguments (which already include the
algorithm), then nine more: a block size, a flag saying whether there is an
outer format, and the outer format's seven fields. The outer format is the
format that BLOCK and TREE round the running total in when they add each
block's sum to it. These are separate ops rather than extra arguments on the
four because every argument costs every call: appended with defaults, the
nine cost 1.0-1.4 us on each NAIVE call through `torch.ops` (measured on the
CPU at 1x1x1 when 0479b96 added the algorithms), because the dispatcher fills
in and unboxes every argument whether or not the caller passes it. A NAIVE
spec therefore has the same op and `args` as before the three algorithms
existed.

Each family module has the builders of its accumulated specs
(`binaryk._binaryK_accumulated_spec` and its counterparts, and the
cross-family ones in `cross`). Each takes its NAIVE builder's keywords plus
the accumulation's, and returns that builder's spec unchanged when the
algorithm is NAIVE. They are separate functions, and the NAIVE builders were
left as they were, because a flat wrapper resolves its formats on every
call: passing eight more keywords through the NAIVE builders, and testing
that they are all unset, cost 0.3-1.3 us on each NAIVE call (measured when
0479b96 added them).

The outer format's sign, saturation and subnormals default to those of the
mac's last format. They are read back from the resolved NAIVE spec, whose
`args` holds them at fixed positions of the op's arguments, rather than
worked out a second time.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from mptorch import AccumulateAlgorithm, SaturationMode, SubnormalsMode
from mptorch.number.binaryk import _binaryK_findings, check_binaryK_storage
from mptorch.number.superfp import _superfp_findings, check_superfp_storage

from ._checks import _errors_first, _Format, _format_findings, _Stored
from ._spec import _binaryK_bias, _GemmSpec


class _Accumulation(NamedTuple):
    """What an accumulate algorithm other than NAIVE adds to a single-format spec.

    ``tail`` is the arguments its op takes after the NAIVE op's. ``rounded``
    is the outer format, in the form the carrier findings take (empty without
    an outer format). ``stored`` is the storage check of the format whose
    values the result holds, or ``None`` when that is the mac's own last
    rounding. That is the case for KAHAN, whose result is a sum rounded by
    the accumulate format. For BLOCK and TREE the result is the running total,
    which the outer format rounds, or which nothing rounds when there is no
    outer format (``stored`` is then empty).
    """

    tail: tuple[Any, ...]
    rounded: tuple[_Format, ...]
    stored: tuple[_Stored, ...] | None


def _checked_block_size(
    algorithm: AccumulateAlgorithm, block_size: int | None, has_outer: bool, has_product: bool
) -> int:
    """Return the checked ``block_size`` argument for ``algorithm``'s op, or 0
    when the algorithm takes none.

    NAIVE and KAHAN take neither a block size nor an outer format. BLOCK needs
    a size that divides 16 or is a multiple of 16: the kernels work in slabs
    of 16 steps, so the end of a block is found with a mask per step or a
    modulo per slab. TREE needs a power of two in [2, 256] (16 when none is
    given), and a mac with a separate product, which a fused multiply-add
    does not have.
    """
    name = f"AccumulateAlgorithm.{algorithm.name}"
    if algorithm in (AccumulateAlgorithm.NAIVE, AccumulateAlgorithm.KAHAN):
        if block_size is not None or has_outer:
            raise ValueError(
                f"{name} takes no block_size and no outer format: those are BLOCK's and TREE's"
            )
        return 0
    if algorithm is AccumulateAlgorithm.BLOCK:
        if block_size is None:
            raise ValueError(f"{name} needs a block_size")
        if block_size < 1 or (16 % block_size != 0 and block_size % 16 != 0):
            raise ValueError(
                f"{name} needs a block_size that divides 16 or is a multiple of 16, "
                f"got {block_size}"
            )
        return block_size
    if not has_product:
        raise ValueError(
            f"{name} sums products pairwise, and a fused multiply-add has no product term: "
            "use a split multiply-accumulate (binaryK_matmul, superfp_matmul, SplitMac)"
        )
    if block_size is None:
        return 16
    if not 2 <= block_size <= 256 or block_size & (block_size - 1):
        raise ValueError(
            f"{name} needs a block_size that is a power of two in [2, 256], got {block_size}"
        )
    return block_size


def _binaryK_accumulation(
    algorithm: AccumulateAlgorithm,
    block_size: int | None,
    outer_K: int | None,
    outer_P: int | None,
    outer_bias: int | None,
    outer_is_signed: bool,
    outer_saturation_mode: SaturationMode,
    outer_subnormals_mode: SubnormalsMode,
    outer_prng_bits: int,
    has_product: bool,
) -> _Accumulation | None:
    """Return what a binaryK op's accumulate algorithm adds to its spec, or
    ``None`` for NAIVE.

    The outer format's sign, saturation and subnormals arrive already
    defaulted, by the caller, to those of the mac's last format, which only
    the caller knows.
    """
    has_outer = outer_K is not None or outer_P is not None
    block = _checked_block_size(algorithm, block_size, has_outer, has_product)
    if algorithm is AccumulateAlgorithm.NAIVE:
        return None
    keep = None if algorithm is AccumulateAlgorithm.KAHAN else ()
    if not has_outer:
        return _Accumulation((block, False, 0, 0, 0, True, 0, 0, 0), (), keep)
    if outer_K is None or outer_P is None:
        raise ValueError("outer_K and outer_P name the outer format together")
    if outer_bias is None:
        outer_bias = _binaryK_bias(outer_K, outer_P, outer_is_signed)
    outer = (
        outer_K,
        outer_P,
        outer_bias,
        outer_is_signed,
        outer_saturation_mode,
        outer_subnormals_mode,
    )
    return _Accumulation(
        (
            block,
            True,
            outer_K,
            outer_P,
            outer_bias,
            outer_is_signed,
            outer_saturation_mode.value,
            outer_subnormals_mode.value,
            outer_prng_bits,
        ),
        ((_binaryK_findings, (*outer, outer_prng_bits)),),
        ((check_binaryK_storage, outer),),
    )


def _superfp_accumulation(
    algorithm: AccumulateAlgorithm,
    block_size: int | None,
    outer_man_bits: int | None,
    outer_exp_bits: int | None,
    outer_normal_binades: int | None,
    outer_bias: int | None,
    outer_is_signed: bool,
    outer_saturation_mode: SaturationMode,
    outer_prng_bits: int,
    has_product: bool,
) -> _Accumulation | None:
    """The superfp counterpart of :func:`_binaryK_accumulation`.

    A superfp outer format is given by four numbers (``man_bits``,
    ``exp_bits``, ``normal_binades`` and ``bias``), and all four are
    required, since a superfp bias has no default.
    """
    widths = (outer_man_bits, outer_exp_bits, outer_normal_binades, outer_bias)
    has_outer = any(w is not None for w in widths)
    block = _checked_block_size(algorithm, block_size, has_outer, has_product)
    if algorithm is AccumulateAlgorithm.NAIVE:
        return None
    keep = None if algorithm is AccumulateAlgorithm.KAHAN else ()
    if not has_outer:
        return _Accumulation((block, False, 0, 0, 0, 0, True, 0, 0), (), keep)
    if (
        outer_man_bits is None
        or outer_exp_bits is None
        or outer_normal_binades is None
        or outer_bias is None
    ):
        raise ValueError(
            "outer_man_bits, outer_exp_bits, outer_normal_binades and outer_bias name the "
            "outer format together"
        )
    outer = (
        outer_man_bits,
        outer_exp_bits,
        outer_normal_binades,
        outer_bias,
        outer_is_signed,
        outer_saturation_mode,
    )
    return _Accumulation(
        (
            block,
            True,
            outer_man_bits,
            outer_exp_bits,
            outer_normal_binades,
            outer_bias,
            outer_is_signed,
            outer_saturation_mode.value,
            outer_prng_bits,
        ),
        ((_superfp_findings, (*outer, outer_prng_bits)),),
        ((check_superfp_storage, outer),),
    )


def _accumulated(
    spec: _GemmSpec, op: Callable[..., torch.Tensor], accumulation: _Accumulation
) -> _GemmSpec:
    """Turn a NAIVE single-format spec into the spec for its accumulate algorithm.

    ``spec`` is what the NAIVE builder returns for the same formats (its
    ``args`` already include the algorithm). ``op`` is the NAIVE op's
    ``*_accumulated`` counterpart, whose arguments are the NAIVE op's
    followed by ``accumulation.tail``. The outer format's findings are added
    to the spec's.
    """
    found32, found64 = spec.findings
    if accumulation.rounded:
        outer32, outer64 = _format_findings(accumulation.rounded)
        found32 = _errors_first([*found32, *outer32])
        found64 = _errors_first([*found64, *outer64])
    return spec._replace(
        op=op,
        args=spec.args + accumulation.tail,
        stored=spec.stored if accumulation.stored is None else accumulation.stored,
        findings=(found32, found64),
    )
