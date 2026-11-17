from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any, NamedTuple

import torch

from mptorch import (
    AccumulateAlgorithm,
    RoundMode,
    SaturationMode,
    SubnormalsMode,
)
from mptorch.number.base import _report_per_call
from mptorch.number.binaryk import _binaryK_findings, check_binaryK_carrier, check_binaryK_storage
from mptorch.number.fixedpoint import check_fixedpoint_carrier, check_fixedpoint_storage
from mptorch.number.superfp import _superfp_findings, check_superfp_carrier, check_superfp_storage

__all__ = [
    "binaryK_quantize",
    "binaryK_quantize_",
    "superfp_quantize",
    "superfp_quantize_",
    "fixedpoint_quantize",
    "fixedpoint_quantize_",
    "binaryK_matmul",
    "superfp_matmul",
    "binaryK_matmul_fma",
    "superfp_matmul_fma",
    "binaryK_matmul_mixed",
    "superfp_matmul_mixed",
    "binaryK_matmul_fma_mixed",
    "superfp_matmul_fma_mixed",
    "binaryK_superfp_matmul",
    "superfp_binaryK_matmul",
]

# --- format defaults, worked out in one place -------------------------------
#
# Every GEMM wrapper works out the same few things: a binaryK format's default
# exponent bias, the accumulate format's fallback to the multiply format, and
# which format's values the result holds. The palette (mixed) wrappers do the
# same for each palette entry. The helpers below hold that logic, so each flat
# wrapper is only a signature, a docstring, and a call that passes its
# arguments in the op's order.


def _binaryK_bias(K: int, P: int, is_signed: bool) -> int:
    """Return binaryK's default exponent bias, as IEEE P3109 defines it.

    It is ``2**(K - P - 1)`` for a signed format and ``2**(K - P)`` for an
    unsigned one. The exponent field is ``K - P`` bits wide (one bit wider
    without a sign bit), and this bias puts 1.0 at the middle code of the
    format; it is one more than the IEEE 754 bias for the same field width.
    """
    return 2 ** (K - P - 1) if is_signed else 2 ** (K - P)


# --- checking the simulated format against the carrier -------------------------
#
# Each call rounds in a carrier, the binary float the kernel computes in:
# binary64 for float64 operands, binary32 for float32, float16 and bfloat16,
# or the one `carrier=` names, which is never narrower than the operands.
# `mptorch.number`'s findings say whether a carrier can hold a simulated
# format: a range beyond the carrier's warns (the format still quantizes
# correctly over the part that fits), and precision beyond it raises. Only the
# call knows its operands' dtype, and so its carrier, so the format is checked
# against the carrier here, on each call; `BinaryK` and `SuperFP` raise only
# for what no carrier can do. The wrappers below take a format's parameters as
# plain integers and never build a format object, so they raise those errors
# themselves when they resolve a format:
# `binaryK_matmul(a, b, mul_K=60, mul_P=54)` raises what `BinaryK(60, 54)` raises.
#
# A GEMM's formats are fixed when its spec (`_GemmSpec`) is built, so both
# carriers' findings are looked up then (`mptorch.number` caches them per
# format), and a call only indexes a tuple. A quantizer resolves nothing
# before the call and looks its format up on each call, which hits the same
# cache.
#
# `carrier=torch.float64` on narrower operands makes the call the float64 call
# on their values: the operands are widened, the binary64 kernel runs, and the
# result is narrowed back to the operands' dtype with a single rounding
# (`_narrowed`). This happens here in Python, not through an op argument, so
# that it is bit-identical to widening the operands by hand and calling the
# op: the same binary64 kernel on the same values, drawing the same random
# words. Only the final narrowing differs from torch's own `.to()`, which can
# round twice (always for bfloat16, and for float16 on some platforms). A
# carrier narrower than the operands is refused, because rounding a float64
# tensor in binary32 would round every input once before the format does.

# One carrier's findings on one GEMM's formats: (error, warning) pairs, each
# with exactly one of the two set, errors first. Empty when the carrier holds
# every format.
_Findings = tuple[tuple[str | None, str | None], ...]

# One format an op rounds with or stores, as `(function, format arguments)`,
# called as `function(*arguments, ...)`. It is a tuple of plain values rather
# than a `partial`, so that two spellings of the same GEMM build equal specs;
# `tests/test_number_formats.py` checks that `mac.py`'s format objects and the
# flat wrappers do.
_Format = tuple[Callable[..., Any], tuple[Any, ...]]


def _checked_carrier(carrier: torch.dtype | None) -> torch.dtype | None:
    """Return the ``carrier`` argument after checking it.

    It is returned unchanged when it is ``None``, ``torch.float32`` or
    ``torch.float64``. Any other dtype raises ``ValueError``, and anything
    that is not a dtype raises ``TypeError`` (a string such as ``"float64"``
    is the likely mistake).
    """
    if carrier is None or carrier is torch.float32 or carrier is torch.float64:
        return carrier
    if isinstance(carrier, torch.dtype):
        raise ValueError(
            f"carrier must be torch.float32 (binary32), torch.float64 (binary64) or None, "
            f"got {carrier}: the kernels round in binary32 or binary64, and a narrower "
            f"tensor is stored in its own dtype afterwards"
        )
    raise TypeError(
        f"carrier must be a torch.dtype -- torch.float32, torch.float64 -- or None, got {carrier!r}"
    )


def _call_carrier(carrier: torch.dtype | None, dtype: torch.dtype) -> tuple[bool, bool]:
    """Work out the carrier of a call on ``dtype`` operands.

    Return ``(wide, widen)``: whether the call rounds in binary64, and whether
    the operands must first be widened to float64 to get there. With
    ``carrier=None``, float64 operands round in binary64 and all others in
    binary32, and nothing is widened. ``carrier=torch.float32`` on float64
    operands raises, since a carrier is never narrower than its operands.
    """
    if carrier is None:
        return dtype is torch.float64, False
    if carrier is torch.float64:
        return True, dtype is not torch.float64
    if carrier is torch.float32 and dtype is torch.float64:
        raise ValueError(
            "carrier=torch.float32 is narrower than float64 operands, whose inputs it "
            "would round once before the format does; a carrier is at least as wide as "
            "its operands -- leave carrier unset, or narrow the operands yourself"
        )
    _checked_carrier(carrier)
    return False, False


def _narrowed(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return a float64 result converted to ``dtype`` with a single rounding
    to nearest (ties to even).

    float64 to float32 is a single conversion, which ``.float()`` performs.
    To bfloat16, torch converts through float32, which rounds twice: a value
    just above a midpoint between two bfloat16 values first rounds onto the
    midpoint, and then, as a tie, to even. ``1 + 2**-8 + 2**-30`` comes back
    as bfloat16 1.0, where a single rounding gives ``1 + 2**-7``. On some
    platforms torch converts to float16 the same way, and 65519.999999 then
    comes back as float16 infinity instead of 65504; an arm64 CPU converts
    float64 to float16 in one step, correctly. The ``narrow_float64`` op
    rounds the float64 value to the dtype directly, with integer arithmetic on
    its bits (``csrc/common/narrow_binary64.h``), at about the cost of
    ``.to()`` on CUDA, and gives the single rounding on every platform.
    """
    if dtype is torch.float32:
        return x.float()
    return torch.ops.mptorch.narrow_float64.default(x, dtype)


# What a findings function returns for a format its carrier holds entirely,
# and a spec's findings when every format is like that.
_NOTHING = (None, None)
_NO_FINDINGS: tuple[_Findings, _Findings] = ((), ())


def _format_findings(formats: Iterable[_Format]) -> tuple[_Findings, _Findings]:
    """Collect binary32's and binary64's findings on every format a GEMM rounds with.

    Return ``(binary32's findings, binary64's findings)``, each without
    duplicates and with the errors first. They are collected once, when the
    spec is built, and each call reports those of the carrier it rounds in.
    binary64's errors are things no carrier can do, so they raise here: this
    is the plain-integer counterpart of what ``BinaryK`` and ``SuperFP`` raise
    when they are built.

    A flat wrapper builds its spec on every call, so the common case, a format
    both carriers hold, is kept cheap. binary64 is wider than binary32 at
    every edge, so a format is looked up against binary64 only when binary32
    finds something, and duplicates (a palette repeats a scalar across its
    entries, and a ``SplitMac`` often uses one format for both of its
    roundings) are removed only when there is something to remove.
    """
    b32: list[tuple[str | None, str | None]] = []
    b64: list[tuple[str | None, str | None]] = []
    for fn, args in formats:
        found = fn(*args, torch.float32)
        if found != _NOTHING:
            b32.append(found)
            found = fn(*args, torch.float64)
            if found != _NOTHING:
                b64.append(found)
    if not b32:
        return _NO_FINDINGS
    ordered32, ordered64 = _errors_first(b32), _errors_first(b64)
    if ordered64 and ordered64[0][0] is not None:
        raise ValueError(ordered64[0][0])
    return ordered32, ordered64


def _errors_first(found: list[tuple[str | None, str | None]]) -> _Findings:
    """Return ``found`` without duplicates, in first-seen order, errors before warnings."""
    unique = dict.fromkeys(found)
    return tuple([f for f in unique if f[0] is not None] + [f for f in unique if f[0] is None])


def _report_findings(found: _Findings) -> None:
    """Raise the first error in ``found``, if any; otherwise issue each warning.

    Each warning is attributed to the caller's line (``_report_per_call``).
    """
    for error, warning in found:
        _report_per_call(error, warning)


# --- checking the simulated format against the storage dtype -----------------
#
# When the operands are narrower than the carrier (float16 or bfloat16 under
# binary32; those and float32 under binary64), the result is converted back
# to the operands' dtype, the storage dtype, when it is written, which rounds
# it a second time. `mptorch.number`'s `check_*_storage` functions decide
# whether the storage dtype can hold the simulated format's results. Only the
# call knows the storage dtype and the carrier, so the check runs here, on
# each call, and only when the storage dtype is narrower than the carrier: a
# result stored in the carrier's own dtype keeps every value the carrier
# computed. The format checked is the one whose values the result holds: a
# GEMM's last rounding (its accumulate or fused format), whose sums can take
# every value of that format, or an elementwise quantizer's format, whose
# inputs are already values of the storage dtype, so that only the edges of
# the format's range matter (`elementwise=True`). A GEMM's multiply format is
# never stored, since its products stay in the carrier.
#
# The storage dtype limits only the precision of the values written into it,
# not `man_bits + prng_bits`: stochastic rounding draws its random bits in the
# carrier, so the carrier check holds that sum against the carrier's 23 or 52
# mantissa bits. Limiting it by the storage dtype's mantissa would describe a
# kernel that rounds in the storage dtype, which these kernels do not.

# The storage dtypes that round a result a second time: the first set for a
# binary32 result, the second for a binary64 result.
_NARROW_STORAGE = frozenset((torch.float16, torch.bfloat16))
_BELOW_BINARY64 = frozenset((torch.float32, torch.float16, torch.bfloat16))

# The carrier dtype that `mptorch.number`'s findings are keyed on, indexed by
# `_call_carrier`'s `wide` answer.
_CARRIER = (torch.float32, torch.float64)

# The storage check of one format whose values a result may hold, as
# `(check, format arguments)`, called as `check(*arguments, storage=dtype)`.
_Stored = _Format


def _palette_formats(
    fn: Callable[..., Any], widths: Iterable[tuple[Any, ...]], shared: tuple[Any, ...]
) -> tuple[_Format, ...]:
    """Return one ``_Format`` per distinct palette entry.

    An entry's arguments are its own widths followed by the fields that every
    entry shares. Equal entries are merged, so a scalar broadcast across the
    palette is one format and is checked once.
    """
    return tuple(dict.fromkeys((fn, (*w, *shared)) for w in widths))


def _check_stored(stored: tuple[_Stored, ...], storage: torch.dtype) -> None:
    """Check each format whose values the result holds against the storage dtype."""
    for check, fmt in stored:
        check(*fmt, storage=storage)


def _palette_list(val: int | Sequence[int], n: int, name: str) -> list[int]:
    """Return ``val`` as a list of length ``n``: a scalar repeated ``n`` times,
    or a sequence checked to have length ``n`` (``name`` names it in the error
    message)."""
    if isinstance(val, int):
        return [val] * n
    out = list(val)
    if len(out) != n:
        raise ValueError(f"{name} must have length {n} (the palette size), got {len(out)}")
    return out


def _palette_list_or(
    val: int | Sequence[int] | None, default: list[int], n: int, name: str
) -> list[int]:
    """:func:`_palette_list`, except that ``None`` returns ``default``."""
    return default if val is None else _palette_list(val, n, name)


def _palette_pair(
    first: Sequence[int], second: Sequence[int], second_name: str, fn: str
) -> tuple[list[int], list[int], int]:
    """Return the two per-entry sequences that define a palette, as lists, and
    the palette size.

    Every palette op takes one such pair, ``(K, P)`` or
    ``(man_bits, exp_bits)``. Their common length is the palette size, which
    every other palette argument is broadcast to or checked against. An empty
    palette raises ``ValueError`` naming ``fn``, the wrapper being called, and
    a length mismatch raises one naming ``second_name``.
    """
    first_l, second_l = list(first), list(second)
    n = len(first_l)
    if n < 1:
        raise ValueError(f"{fn} needs at least one palette format")
    if len(second_l) != n:
        raise ValueError(
            f"{second_name} must have length {n} (the palette size), got {len(second_l)}"
        )
    return first_l, second_l, n


def _binaryK_palette_bias(
    bias: int | Sequence[int] | None, K_l: list[int], P_l: list[int], is_signed: bool, name: str
) -> list[int]:
    """Return the per-entry exponent biases of a binaryK palette: each entry's
    default bias (:func:`_binaryK_bias`) when ``bias`` is ``None``, and
    otherwise ``bias`` repeated or length-checked as by :func:`_palette_list`."""
    if bias is None:
        return [_binaryK_bias(k, p, is_signed) for k, p in zip(K_l, P_l, strict=True)]
    return _palette_list(bias, len(K_l), name)


# --- a GEMM call with its formats resolved -----------------------------------


class _GemmSpec(NamedTuple):
    """One GEMM call's formats, resolved into exactly the arguments its op takes.

    ``args`` holds every op argument after the operands and the transpose
    flags, in the op's order. ``stored`` holds the storage checks of the
    formats whose values the result holds: the last rounding's format (one
    per distinct palette entry for a palette op), or nothing when the last
    step is not rounded. ``findings`` holds each carrier's findings on every
    format the op rounds with. Those two are all that a call still has to
    check once it knows the operands' dtype.

    Resolving is separate from calling so that a caller with fixed formats
    can resolve once and call many times: `mptorch.quant.gemm`'s factories
    build one spec per layer and bind it into the layer's math hooks, and
    `mptorch.quant.mac` caches one per frozen format value. Working out every
    default and rerunning every check on each call would cost 2.3-6.4 us per
    call, most of a small GEMM's Python overhead.
    """

    op: Callable[..., torch.Tensor]
    args: tuple[Any, ...]
    stored: tuple[_Stored, ...] = ()
    # Whether `op` is a palette op, which takes a `prec_idx` (the map from
    # each output element to its palette entry) between the operands and the
    # transpose flags. `mptorch.quant.mac` reads this to decide whether a call
    # needs a map, instead of working it out from the format again.
    mixed: bool = False
    # The carrier the caller named, as a dtype, or None to use the operands' own.
    carrier: torch.dtype | None = None
    # binary32's findings, then binary64's, as `_format_findings` returns them.
    findings: tuple[_Findings, _Findings] = ((), ())


# --- the accumulate algorithms other than NAIVE --------------------------------
#
# KAHAN, BLOCK and TREE (`csrc/common/gemm_accumulate.h`) run in four ops of
# their own, the `*_accumulated` counterparts of the four single-format ops.
# Each takes its counterpart's arguments (which already include the
# algorithm), then nine more: a block size, a flag saying whether there is an
# outer format, and the outer format's seven fields. The outer format is the
# format that BLOCK and TREE round the running total in when they add each
# block's sum to it. These are separate ops rather than extra arguments on the
# four because every argument costs every call: appended with defaults, the
# nine cost 1.0-1.4 us on each NAIVE call through `torch.ops` (measured on the
# CPU at 1x1x1 when 0479b96 added the algorithms), because the dispatcher
# fills in and unboxes every argument whether or not the caller passes it. A
# NAIVE spec therefore has the same op and `args` as before the three
# algorithms existed.


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


# The four builders of the accumulated specs. Each takes its NAIVE builder's
# keywords plus the accumulation's, and returns that builder's spec unchanged
# when the algorithm is NAIVE. They are separate functions, and the NAIVE
# builders were left as they were, because a flat wrapper resolves its formats
# on every call: passing eight more keywords through the NAIVE builders, and
# testing that they are all unset, cost 0.3-1.3 us on each NAIVE call
# (measured when 0479b96 added them).
#
# The outer format's sign, saturation and subnormals default to those of the
# mac's last format. They are read back from the resolved NAIVE spec, whose
# `args` holds them at fixed positions of the op's arguments, rather than
# worked out a second time.


def _binaryK_accumulated_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    block_size: int | None = None,
    outer_K: int | None = None,
    outer_P: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    outer_subnormals_mode: SubnormalsMode | None = None,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_binaryK_spec` for any accumulate algorithm; ``formats`` are its keywords."""
    spec = _binaryK_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    acc_is_signed, acc_saturation, acc_subnormals = spec.args[8], spec.args[13], spec.args[14]
    accumulation = _binaryK_accumulation(
        accumulate_algorithm,
        block_size,
        outer_K,
        outer_P,
        outer_bias,
        acc_is_signed if outer_is_signed is None else outer_is_signed,
        SaturationMode(acc_saturation) if outer_saturation_mode is None else outer_saturation_mode,
        SubnormalsMode(acc_subnormals) if outer_subnormals_mode is None else outer_subnormals_mode,
        outer_prng_bits,
        has_product=True,
    )
    if accumulation is None:
        return spec
    return _accumulated(
        spec, torch.ops.mptorch.custom_matmul_binaryK_accumulated.default, accumulation
    )


def _superfp_accumulated_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    block_size: int | None = None,
    outer_man_bits: int | None = None,
    outer_exp_bits: int | None = None,
    outer_normal_binades: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_superfp_spec` for any accumulate algorithm; ``formats`` are its keywords."""
    spec = _superfp_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    acc_is_signed, acc_saturation = spec.args[10], spec.args[14]
    accumulation = _superfp_accumulation(
        accumulate_algorithm,
        block_size,
        outer_man_bits,
        outer_exp_bits,
        outer_normal_binades,
        outer_bias,
        acc_is_signed if outer_is_signed is None else outer_is_signed,
        SaturationMode(acc_saturation) if outer_saturation_mode is None else outer_saturation_mode,
        outer_prng_bits,
        has_product=True,
    )
    if accumulation is None:
        return spec
    return _accumulated(
        spec, torch.ops.mptorch.custom_matmul_superfp_accumulated.default, accumulation
    )


def _binaryK_fma_accumulated_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    block_size: int | None = None,
    outer_K: int | None = None,
    outer_P: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    outer_subnormals_mode: SubnormalsMode | None = None,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_binaryK_fma_spec` for any accumulate algorithm; ``formats`` are its keywords."""
    spec = _binaryK_fma_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    fma_is_signed, fma_saturation, fma_subnormals = spec.args[4], spec.args[7], spec.args[8]
    accumulation = _binaryK_accumulation(
        accumulate_algorithm,
        block_size,
        outer_K,
        outer_P,
        outer_bias,
        fma_is_signed if outer_is_signed is None else outer_is_signed,
        SaturationMode(fma_saturation) if outer_saturation_mode is None else outer_saturation_mode,
        SubnormalsMode(fma_subnormals) if outer_subnormals_mode is None else outer_subnormals_mode,
        outer_prng_bits,
        has_product=False,
    )
    if accumulation is None:
        return spec
    return _accumulated(
        spec, torch.ops.mptorch.custom_matmul_binaryK_fma_accumulated.default, accumulation
    )


def _superfp_fma_accumulated_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    block_size: int | None = None,
    outer_man_bits: int | None = None,
    outer_exp_bits: int | None = None,
    outer_normal_binades: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_superfp_fma_spec` for any accumulate algorithm; ``formats`` are its keywords."""
    spec = _superfp_fma_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    fma_is_signed, fma_saturation = spec.args[5], spec.args[8]
    accumulation = _superfp_accumulation(
        accumulate_algorithm,
        block_size,
        outer_man_bits,
        outer_exp_bits,
        outer_normal_binades,
        outer_bias,
        fma_is_signed if outer_is_signed is None else outer_is_signed,
        SaturationMode(fma_saturation) if outer_saturation_mode is None else outer_saturation_mode,
        outer_prng_bits,
        has_product=False,
    )
    if accumulation is None:
        return spec
    return _accumulated(
        spec, torch.ops.mptorch.custom_matmul_superfp_fma_accumulated.default, accumulation
    )


# The carrier step rounded to odd (`csrc/common/carrier_odd.h`). With
# `carrier_rounding=RoundMode.RO`, each mac step rounds its carrier operation
# to odd instead of to nearest, so that the cast to the simulated format then
# rounds the exact value. Each `custom_matmul_*_odd` op takes the same
# arguments as its default counterpart, so its spec is the counterpart's with
# the op replaced, plus a check of what the guarantee needs of the formats:
# the cast of a result rounded to odd rounds the exact value only if every
# rounding boundary of the format is a carrier value with an even last bit.
# With p the carrier's precision (24 in binary32, 53 in binary64), a format of
# precision P needs
#
#   RNE, RNA           P <= p - 2
#   RU, RD, RZ, RO     P <= p - 1
#   SR (StochasticA)   P + prng_bits <= p - 1
#   SR_B, SR_C         P + prng_bits <= p - 2
#
# The nearest modes need one bit more because their boundaries are midpoints
# between format values, and SR_B and SR_C because each reads one bit of the
# value below the random bits (SR_C to find a midpoint there). A format
# outside these limits raises rather than warns, on each call and for each
# carrier, and the binary32 error names carrier=torch.float64 when binary64
# can hold the format: binary64's verdict raises when the spec is built, and
# binary32's when a call rounds in binary32, as for every other carrier
# finding.

_RNE = RoundMode.RNE
_CARRIER_PRECISION = ((24, "binary32"), (53, "binary64"))


def _odd_findings(
    slots: Iterable[tuple[str, int, int]], rounding_mode: RoundMode
) -> tuple[_Findings, _Findings]:
    """Check the formats in ``slots`` against each carrier's round-to-odd step.

    Return ``(binary32's findings, binary64's findings)``. ``slots`` holds
    ``(name, P, prng_bits)`` for every format that a cast rounds a
    round-to-odd carrier result to; ``name`` names the format in the message.
    """
    found32: list[tuple[str | None, str | None]] = []
    found64: list[tuple[str | None, str | None]] = []
    for name, P, prng_bits in slots:
        verdicts = []
        for p, carrier in _CARRIER_PRECISION:
            if rounding_mode.is_stochastic:
                bits, spelled = P + prng_bits, "P + prng_bits"
                limit = p - 1 if rounding_mode is RoundMode.SR else p - 2
            else:
                bits, spelled = P, "P"
                limit = p - 2 if rounding_mode in (RoundMode.RNE, RoundMode.RNA) else p - 1
            verdicts.append(
                None
                if bits <= limit
                else (
                    f"carrier_rounding=RoundMode.RO: the {name} format's {spelled} is {bits}, "
                    f"and a {carrier} step rounded to odd is rounded exactly under "
                    f"RoundMode.{rounding_mode.name} only for {spelled} <= {limit}"
                )
            )
        error32, error64 = verdicts
        if error64 is not None:
            found64.append((error64, None))
        if error32 is not None:
            if error64 is None:
                error32 += "; binary64 holds it: pass carrier=torch.float64"
            found32.append((error32, None))
    return tuple(found32), tuple(found64)


def _odd(
    spec: _GemmSpec, op: Callable, slots: Iterable[tuple[str, int, int]], rounding_mode: RoundMode
) -> _GemmSpec:
    """Return ``spec`` with its op replaced by the round-to-odd counterpart.

    The round-to-odd checks are added to its findings; binary64's errors raise
    here.
    """
    odd32, odd64 = _odd_findings(slots, rounding_mode)
    if odd64:
        raise ValueError(odd64[0][0])
    found32, found64 = spec.findings
    if odd32:
        found32 = _errors_first([*found32, *odd32])
    return spec._replace(op=op, findings=(found32, found64))


def _checked_naive(accumulate_algorithm: AccumulateAlgorithm) -> None:
    if accumulate_algorithm is not AccumulateAlgorithm.NAIVE:
        raise ValueError(
            "carrier_rounding=RoundMode.RO: this op implements AccumulateAlgorithm.NAIVE only; "
            "KAHAN, BLOCK and TREE are the single-format and cross-family ops' (their "
            "*_accumulated_odd twins), and the palette ops have none"
        )


def _checked_fma_quant(fma_quant: bool) -> None:
    if not fma_quant:
        raise ValueError(
            "carrier_rounding=RoundMode.RO needs fma_quant=True: a carrier step rounded to odd "
            "is only correct when a format rounds it again"
        )


def _binaryK_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_binaryK_spec` with the carrier step rounded to odd; ``formats`` are its keywords."""
    _checked_naive(accumulate_algorithm)
    spec = _binaryK_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # mul_P [1], accumulate_quant [4], acc_P [6], round_mode [10], prng [15], [16]
    slots = [("multiply", a[1], a[15])] + ([("accumulate", a[6], a[16])] if a[4] else [])
    return _odd(spec, torch.ops.mptorch.custom_matmul_binaryK_odd.default, slots, RoundMode(a[10]))


def _superfp_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_superfp_spec` with the carrier step rounded to odd; ``formats`` are its keywords."""
    _checked_naive(accumulate_algorithm)
    spec = _superfp_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    # mul_man_bits [0], accumulate_quant [5], acc_man_bits [6], round_mode [12], prng [15], [16]
    a = spec.args
    slots = [("multiply", a[0] + 1, a[15])] + ([("accumulate", a[6] + 1, a[16])] if a[5] else [])
    return _odd(spec, torch.ops.mptorch.custom_matmul_superfp_odd.default, slots, RoundMode(a[12]))


def _binaryK_fma_odd_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    fma_quant: bool = True,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_binaryK_fma_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    _checked_fma_quant(fma_quant)
    spec = _binaryK_fma_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # fma_P [2], round_mode [6], fma_prng_bits [9]
    return _odd(
        spec,
        torch.ops.mptorch.custom_matmul_binaryK_fma_odd.default,
        [("fused", a[2], a[9])],
        RoundMode(a[6]),
    )


def _superfp_fma_odd_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    fma_quant: bool = True,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_superfp_fma_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    _checked_fma_quant(fma_quant)
    spec = _superfp_fma_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # fma_man_bits [1], round_mode [7], fma_prng_bits [9]
    return _odd(
        spec,
        torch.ops.mptorch.custom_matmul_superfp_fma_odd.default,
        [("fused", a[1] + 1, a[9])],
        RoundMode(a[7]),
    )


def _palette_slots(name: str, Ps: Sequence[int], prng_bits: int) -> list[tuple[str, int, int]]:
    """Return ``_odd_findings``' slots for a palette: one per entry, named with its index."""
    return [(f"{name} (entry {i})", P, prng_bits) for i, P in enumerate(Ps)]


def _binaryK_mixed_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_binaryK_mixed_spec` with the carrier step rounded to odd, every entry checked."""
    _checked_naive(accumulate_algorithm)
    spec = _binaryK_mixed_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # mul_P [1], accumulate_quant [4], acc_P [6], round_mode [10], prng [15], [16]
    slots = _palette_slots("multiply", a[1], a[15])
    if a[4]:
        slots += _palette_slots("accumulate", a[6], a[16])
    op = torch.ops.mptorch.custom_matmul_binaryK_mixed_odd.default
    return _odd(spec, op, slots, RoundMode(a[10]))


def _superfp_mixed_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_superfp_mixed_spec` with the carrier step rounded to odd, every entry checked."""
    _checked_naive(accumulate_algorithm)
    spec = _superfp_mixed_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    # mul_man_bits [0], accumulate_quant [5], acc_man_bits [6], round_mode [12], prng [15], [16]
    a = spec.args
    slots = _palette_slots("multiply", [m + 1 for m in a[0]], a[15])
    if a[5]:
        slots += _palette_slots("accumulate", [m + 1 for m in a[6]], a[16])
    op = torch.ops.mptorch.custom_matmul_superfp_mixed_odd.default
    return _odd(spec, op, slots, RoundMode(a[12]))


def _binaryK_fma_mixed_odd_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    fma_quant: bool = True,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_binaryK_fma_mixed_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    _checked_fma_quant(fma_quant)
    spec = _binaryK_fma_mixed_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # fma_P [2], round_mode [6], fma_prng_bits [9]
    op = torch.ops.mptorch.custom_matmul_binaryK_fma_mixed_odd.default
    return _odd(spec, op, _palette_slots("fused", a[2], a[9]), RoundMode(a[6]))


def _superfp_fma_mixed_odd_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    fma_quant: bool = True,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_superfp_fma_mixed_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    _checked_fma_quant(fma_quant)
    spec = _superfp_fma_mixed_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # fma_man_bits [1], round_mode [7], fma_prng_bits [9]
    op = torch.ops.mptorch.custom_matmul_superfp_fma_mixed_odd.default
    return _odd(spec, op, _palette_slots("fused", [m + 1 for m in a[1]], a[9]), RoundMode(a[7]))


def _binaryK_superfp_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_binaryK_superfp_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    spec = _binaryK_superfp_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # mul_P [1], acc_man_bits [4], round_mode [10], prng [14], [15]
    slots = [("multiply", a[1], a[14]), ("accumulate", a[4] + 1, a[15])]
    op = torch.ops.mptorch.custom_matmul_binaryK_superfp_odd.default
    return _odd(spec, op, slots, RoundMode(a[10]))


def _superfp_binaryK_odd_spec(
    *, accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE, **formats: Any
) -> _GemmSpec:
    """:func:`_superfp_binaryK_spec` with the carrier step rounded to odd."""
    _checked_naive(accumulate_algorithm)
    spec = _superfp_binaryK_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    a = spec.args  # mul_man_bits [0], acc_P [6], round_mode [10], prng [14], [15]
    slots = [("multiply", a[0] + 1, a[14]), ("accumulate", a[6], a[15])]
    op = torch.ops.mptorch.custom_matmul_superfp_binaryK_odd.default
    return _odd(spec, op, slots, RoundMode(a[10]))


# The carrier step rounded to odd under KAHAN, BLOCK and TREE: the accumulated
# spec with its op replaced by the `*_accumulated_odd` one. When an outer
# format rounds the running total, it is checked against the round-to-odd
# limits along with the mac's formats, since its cast also rounds a sum that
# was rounded to odd. The tail of `args` is the accumulation's
# (`_binaryK_accumulation`, `_superfp_accumulation`): block_size, outer_quant,
# the outer format's fields, and its prng_bits last. A NAIVE call never comes
# here: `_checked_block_size` rejects one that names a block size or an outer
# format, and the callers send a plain NAIVE call to the NAIVE round-to-odd
# builders.


def _odd_accumulated(
    spec: _GemmSpec,
    op: Callable,
    slots: list[tuple[str, int, int]],
    naive_len: int,
    outer_superfp: bool,
    round_mode: int,
) -> _GemmSpec:
    """Return the accumulated ``spec`` with its op replaced by the round-to-odd ``op``.

    The outer format, when there is one, is checked along with ``slots``.
    ``naive_len`` is the length of the NAIVE op's arguments, where the
    accumulation's tail starts in ``spec.args``, and ``outer_superfp`` says
    whether the outer format is a superfp, whose precision is
    ``man_bits + 1``.
    """
    tail = spec.args[naive_len:]
    if tail[1]:  # outer_quant
        P = tail[2] + 1 if outer_superfp else tail[3]
        slots = [*slots, ("outer", P, tail[8])]
    return _odd(spec, op, slots, RoundMode(round_mode))


def _binaryK_accumulated_odd_spec(**keywords: Any) -> _GemmSpec:
    """:func:`_binaryK_accumulated_spec` with the carrier step rounded to odd."""
    spec = _binaryK_accumulated_spec(**keywords)
    a = spec.args  # as _binaryK_odd_spec's, then the tail from [17]
    slots = [("multiply", a[1], a[15])] + ([("accumulate", a[6], a[16])] if a[4] else [])
    op = torch.ops.mptorch.custom_matmul_binaryK_accumulated_odd.default
    return _odd_accumulated(spec, op, slots, 17, False, a[10])


def _superfp_accumulated_odd_spec(**keywords: Any) -> _GemmSpec:
    """:func:`_superfp_accumulated_spec` with the carrier step rounded to odd."""
    spec = _superfp_accumulated_spec(**keywords)
    a = spec.args  # as _superfp_odd_spec's, then the tail from [17]
    slots = [("multiply", a[0] + 1, a[15])] + ([("accumulate", a[6] + 1, a[16])] if a[5] else [])
    op = torch.ops.mptorch.custom_matmul_superfp_accumulated_odd.default
    return _odd_accumulated(spec, op, slots, 17, True, a[12])


def _binaryK_fma_accumulated_odd_spec(*, fma_quant: bool = True, **keywords: Any) -> _GemmSpec:
    """:func:`_binaryK_fma_accumulated_spec` with the carrier step rounded to odd."""
    _checked_fma_quant(fma_quant)
    spec = _binaryK_fma_accumulated_spec(fma_quant=fma_quant, **keywords)
    a = spec.args  # as _binaryK_fma_odd_spec's, then the tail from [10]
    op = torch.ops.mptorch.custom_matmul_binaryK_fma_accumulated_odd.default
    return _odd_accumulated(spec, op, [("fused", a[2], a[9])], 10, False, a[6])


def _superfp_fma_accumulated_odd_spec(*, fma_quant: bool = True, **keywords: Any) -> _GemmSpec:
    """:func:`_superfp_fma_accumulated_spec` with the carrier step rounded to odd."""
    _checked_fma_quant(fma_quant)
    spec = _superfp_fma_accumulated_spec(fma_quant=fma_quant, **keywords)
    a = spec.args  # as _superfp_fma_odd_spec's, then the tail from [10]
    op = torch.ops.mptorch.custom_matmul_superfp_fma_accumulated_odd.default
    return _odd_accumulated(spec, op, [("fused", a[1] + 1, a[9])], 10, True, a[7])


def _binaryK_superfp_accumulated_odd_spec(**keywords: Any) -> _GemmSpec:
    """:func:`_binaryK_superfp_accumulated_spec` with the carrier step rounded to odd."""
    spec = _binaryK_superfp_accumulated_spec(**keywords)
    a = spec.args  # as _binaryK_superfp_odd_spec's, then the tail from [16]
    slots = [("multiply", a[1], a[14]), ("accumulate", a[4] + 1, a[15])]
    op = torch.ops.mptorch.custom_matmul_binaryK_superfp_accumulated_odd.default
    return _odd_accumulated(spec, op, slots, 16, True, a[10])


def _superfp_binaryK_accumulated_odd_spec(**keywords: Any) -> _GemmSpec:
    """:func:`_superfp_binaryK_accumulated_spec` with the carrier step rounded to odd."""
    spec = _superfp_binaryK_accumulated_spec(**keywords)
    a = spec.args  # as _superfp_binaryK_odd_spec's, then the tail from [16]
    slots = [("multiply", a[0] + 1, a[14]), ("accumulate", a[6], a[15])]
    op = torch.ops.mptorch.custom_matmul_superfp_binaryK_accumulated_odd.default
    return _odd_accumulated(spec, op, slots, 16, False, a[10])


def _checked_carrier_rounding_call(carrier_rounding: RoundMode) -> None:
    """Check a flat wrapper's ``carrier_rounding`` when it is not RNE: it must be RO."""
    if carrier_rounding is not RoundMode.RO:
        raise ValueError(
            "carrier_rounding is RoundMode.RNE (the carrier's nearest step) or RoundMode.RO "
            f"(rounded to odd), got {carrier_rounding!r}"
        )


def _packed(x: torch.Tensor, trans: bool) -> tuple[torch.Tensor, bool]:
    """Return an operand the op can read without a copy, and the transpose flag
    to read it with.

    The op reads a contiguous tensor, with a flag saying whether to read it
    transposed. A tensor that is the transpose of a contiguous one
    (``k.mT`` in ``q @ k.mT``, the layout of a batched attention GEMM)
    therefore needs no copy: it is passed as that contiguous tensor with the
    opposite flag. Any other non-contiguous layout is copied. The results are
    the same either way: the kernel visits the same elements, and stochastic
    rounding selects its random stream by the output element, not by the
    operand's layout. A contiguous operand takes the first branch, so the
    common case costs one ``is_contiguous()``.
    """
    if x.is_contiguous():
        return x, trans
    xt = x.transpose(-2, -1)
    if xt.is_contiguous():
        return xt, not trans
    return x.contiguous(), trans


def _named_carrier_operands(
    spec: _GemmSpec, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype | None]:
    """Prepare the operands of a call whose spec names its carrier.

    Run the checks that depend on the carrier and the operands' dtype (the
    carrier's findings, then the storage check when that dtype is narrower
    than the carrier), and return the operands, widened to float64 when the
    carrier requires it, plus the dtype to narrow the result back to after
    widening, or ``None``.
    """
    dtype = a.dtype
    wide, widen = _call_carrier(spec.carrier, dtype)
    found = spec.findings[wide]
    if found:
        _report_findings(found)
    if spec.stored and dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        _check_stored(spec.stored, dtype)
    # If the operands' dtypes differ, the op refuses them; widening both here
    # would hide the mismatch.
    if widen and b.dtype is dtype:
        return a.double(), b.double(), dtype
    return a, b, None


def _run_gemm(
    spec: _GemmSpec, a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> torch.Tensor:
    """Call a resolved GEMM on rank-2 or rank-3 operands.

    What remains to do on each call is what the operands' dtype decides (the
    carrier's findings, the storage check, widening for a named carrier) and
    passing transposed views without a copy (:func:`_packed`).
    ``torch.matmul``'s operand rules (promoting a 1D operand, broadcasting
    leading dimensions, ranks above 3) are applied before this, by
    :func:`_matmul_operands`.
    """
    narrow = None
    if spec.carrier is not None:
        a, b, narrow = _named_carrier_operands(spec, a, b)
    else:
        # Every call without `carrier=` pays this, so it is written out here
        # and in `_run_gemm_mixed` instead of behind a function call (which
        # costs 0.25 us): the findings of the operands' carrier, indexed by a
        # bool (binary32's, then binary64's), and the storage check, of which a
        # float32 call pays only the set-membership test.
        dtype = a.dtype
        found = spec.findings[dtype is torch.float64]
        if found:
            _report_findings(found)
        if spec.stored and dtype in _NARROW_STORAGE:
            _check_stored(spec.stored, dtype)
    a, trans_a = _packed(a, trans_a)
    b, trans_b = _packed(b, trans_b)
    out = spec.op(a, b, trans_a, trans_b, *spec.args)
    return out if narrow is None else _narrowed(out, narrow)


def _run_gemm_mixed(
    spec: _GemmSpec,
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    """:func:`_run_gemm` for the palette ops, which take ``prec_idx`` as their
    third argument.

    ``prec_idx`` is passed on unchanged. Converting and packing it is done in
    C++, which, for a map on a GPU, caches its bounds check on the identity
    and version of the tensor it receives. Converting it here with ``.to()``
    would pass a new tensor on every call and make the check run every time.
    """
    narrow = None
    if spec.carrier is not None:
        a, b, narrow = _named_carrier_operands(spec, a, b)
    else:
        dtype = a.dtype
        found = spec.findings[dtype is torch.float64]
        if found:
            _report_findings(found)
        if spec.stored and dtype in _NARROW_STORAGE:
            _check_stored(spec.stored, dtype)
    a, trans_a = _packed(a, trans_a)
    b, trans_b = _packed(b, trans_b)
    out = spec.op(a, b, prec_idx, trans_a, trans_b, *spec.args)
    return out if narrow is None else _narrowed(out, narrow)


# --- torch.matmul's operand rules, mapped onto the rank-2 and rank-3 ops ------
#
# The ops take operands of rank 2 or rank 3 only: at most one batch dimension,
# which each operand either has or shares across the batch, read at batch
# stride 0 (`common/gemm_host.h`). Everything else that `torch.matmul` accepts
# (a 1D operand, any number of leading dimensions, broadcasting between them)
# is turned into that form here, with views wherever `torch.matmul` itself
# would use one, so that the two copy data in the same cases.


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


# --- elementwise quantizers --------------------------------------------------


def binaryK_quantize(
    x: torch.Tensor,
    K: int,
    P: int,
    bias: int | None = None,
    prng_bits: int = 0,
    is_signed: bool = True,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a binaryK floating-point format.

    binaryK is the parameterized family of IEEE P3109, the draft Standard for
    Arithmetic Formats for Machine Learning (see :class:`mptorch.BinaryK`). A
    format has ``K`` bits in total. ``P`` of them are precision (``P - 1``
    stored mantissa bits plus the implicit one), and ``K - P`` are exponent
    bits (``K - P + 1`` when ``is_signed`` is false, because there is no sign
    bit). Pass ``bias`` explicitly for a format outside P3109. The layouts of
    the OCP 8-bit formats are ``K=8, P=4, bias=7`` (E4M3) and
    ``K=8, P=3, bias=15`` (E5M2).

    The rounding happens in a *carrier*, the binary floating-point type the
    arithmetic runs in. It is binary64 for a float64 ``x``. For float32,
    float16 and bfloat16 it is binary32, applied to their float32 value. Every
    call checks the format against what its carrier can hold: at most 24 bits
    of precision and seven exponent bits in binary32, and 53 and ten in
    binary64 (see :doc:`/concepts`). The random bits of stochastic rounding are
    drawn in the carrier too, so ``P - 1 + prng_bits`` is limited to its 23 or
    52 mantissa bits.

    When ``x`` is narrower than its carrier (float16 or bfloat16, or float32
    under binary64), the result is rounded a second time when it is stored in
    ``x``'s dtype, the *storage dtype*, so the format is also checked against
    that dtype. The inputs are already values of the storage dtype, so only a
    result at an edge of the format's range can fail to be one, and that case
    warns.

    NaN inputs pass through. So do infinities, except under
    ``SaturationMode.SAT_FINITE`` and ``SAT_FINITE_EXTENDED`` (P3109's
    ``SatFinite`` in its two domains), which clamp them to the largest finite
    value.

    The function is not differentiable: rounding to a coarse grid has a zero
    derivative almost everywhere. On a tensor that requires grad, under grad
    mode, it raises and points to :class:`mptorch.quant.Quantizer`, the
    straight-through estimator. :class:`mptorch.quant.Quant` makes the same
    call from a format object.

    Args:
        x (Tensor): the tensor to round; float32, float64, float16 or
            bfloat16.
        K (int): the format's width in bits.
        P (int): the format's precision: ``P - 1`` stored mantissa bits plus
            the implicit leading bit.
        bias (int, optional): the exponent bias. Default: ``None``, which is
            P3109's ``2**(K - P - 1)`` if signed and ``2**(K - P)`` if
            unsigned, one more than the IEEE 754 bias for the same exponent
            width.
        prng_bits (int): the number of random bits the stochastic rounding
            modes (``RoundMode.SR``, ``SR_B``, ``SR_C``) draw below the
            format's last mantissa bit; ignored by the other modes.
            Default: ``0``
        is_signed (bool): whether the format has a sign bit. Default: ``True``
        rounding_mode (RoundMode): how a value that falls between two of the
            format's values is rounded. Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): what happens to a value beyond the
            largest finite value. Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the format has below its
            smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        carrier (torch.dtype, optional): ``torch.float64`` rounds any ``x`` in
            binary64: a narrower ``x`` is widened first, and the result is
            narrowed back to its dtype with a single rounding.
            ``torch.float32`` names binary32, which a float64 ``x`` refuses,
            because the carrier is never narrower than the tensor. Default:
            ``None``, ``x``'s own carrier.

    Returns:
        Tensor: a new tensor of ``x``'s shape and dtype holding the rounded
        values.

    Raises:
        ValueError: if the carrier cannot hold the format (more precision than
            the carrier has, ``P - 1 + prng_bits`` above its mantissa bits, an
            exponent field wider than 30 bits, no finite normal value in its
            normal range), if the storage dtype can store none of the format's
            nonzero values, or if ``carrier`` is a dtype that names no carrier
            or is narrower than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode.

    Warns:
        FormatRangeWarning: if the format's range exceeds the carrier's, or a
            result at an edge of the format's range is not a value of a
            narrower storage dtype. The format still quantizes correctly over
            the part that fits.

    Example::

        >>> x = torch.tensor([1.0, 1.1, 0.5])
        >>> binaryK_quantize(x, K=8, P=4)
        tensor([1.0000, 1.1250, 0.5000])
        >>> binaryK_quantize(x.half(), K=8, P=4, carrier=torch.float64)
        tensor([1.0000, 1.1250, 0.5000], dtype=torch.float16)
    """
    if bias is None:
        bias = _binaryK_bias(K, P, is_signed)
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_binaryK_carrier(
        K, P, bias, is_signed, saturation_mode, subnormals_mode, prng_bits, carrier=_CARRIER[wide]
    )
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_binaryK_storage(
            K,
            P,
            bias,
            is_signed,
            saturation_mode,
            subnormals_mode,
            storage=dtype,
            elementwise=True,
        )

    out = torch.ops.mptorch.binaryK_quant.default(
        _quantizer_operand(x, widen),
        K,
        P,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
        subnormals_mode.value,
    )
    return _narrowed(out, dtype) if widen else out


def _quantizer_operand(x: torch.Tensor, widen: bool) -> torch.Tensor:
    """Return ``x`` as the elementwise op reads it: contiguous, and widened to
    float64 when the call's carrier requires it, in a single copy."""
    if widen:
        return x.to(torch.float64, memory_format=torch.contiguous_format)
    return x.contiguous()


def _refuse_widening_in_place(op: str, dtype: torch.dtype) -> None:
    """Raise for an in-place quantizer whose carrier is wider than its tensor.

    ``carrier=torch.float64`` on a float32, float16 or bfloat16 tensor rounds
    a float64 copy of it and narrows the result back, which needs a buffer of
    another dtype and so cannot be done in place.
    """
    raise ValueError(
        f"{op}_ cannot round a {dtype} tensor in carrier=torch.float64 in place: that carrier "
        f"needs a float64 copy of the tensor and a narrowed result, which is what an in-place "
        f"op exists to avoid -- use {op}(x, ..., carrier=torch.float64) and keep its result, "
        f"or leave carrier unset to round in the tensor's own carrier"
    )


def binaryK_quantize_(
    x: torch.Tensor,
    K: int,
    P: int,
    bias: int | None = None,
    prng_bits: int = 0,
    is_signed: bool = True,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a binaryK format, in place.

    This is the in-place version of :func:`binaryK_quantize`. It takes the
    same arguments, makes the same per-call format checks, and gives the same
    result element for element, written over ``x``. That includes stochastic
    rounding with the same seed, because each element's random bits are
    selected by its index. No output is allocated, so peak memory is one
    tensor instead of two. The time is the same, because the kernel reads and
    writes the same bytes either way. Use it when the caller owns ``x``: a
    weight quantized once at load time, or an activation that nothing else
    will read.

    :func:`binaryK_quantize` handles some inputs by copying them. This
    function refuses those inputs, because the kernel would write the copy
    and leave ``x`` unchanged:

    * a tensor that is not contiguous;
    * on CUDA, a view that does not start on a 16-byte boundary (``x[1:]``),
      which the kernel's vector loads cannot address;
    * ``carrier=torch.float64`` on a float32, float16 or bfloat16 tensor,
      which rounds a float64 copy and narrows the result. A float64 tensor
      does round in place in binary64, which is its own carrier.

    Args:
        x (Tensor): the tensor to round and overwrite; float32, float64,
            float16 or bfloat16, contiguous. The format and mode arguments
            between it and ``carrier`` are :func:`binaryK_quantize`'s.
        carrier (torch.dtype, optional): ``None`` or the carrier ``x`` already
            has (``torch.float32`` for float32, float16 and bfloat16,
            ``torch.float64`` for float64). Default: ``None``

    Returns:
        Tensor: ``x`` itself, holding the rounded values.

    Raises:
        ValueError: as for :func:`binaryK_quantize`, and if ``carrier`` is
            wider than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode, is not
            contiguous, or is a CUDA view off a 16-byte boundary.

    Warns:
        FormatRangeWarning: as for :func:`binaryK_quantize`.

    Example::

        >>> x = torch.tensor([1.0, 1.1, 0.5])
        >>> binaryK_quantize_(x, K=8, P=4) is x
        True
        >>> x
        tensor([1.0000, 1.1250, 0.5000])
    """
    if bias is None:
        bias = _binaryK_bias(K, P, is_signed)
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_binaryK_carrier(
        K, P, bias, is_signed, saturation_mode, subnormals_mode, prng_bits, carrier=_CARRIER[wide]
    )
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_binaryK_storage(
            K,
            P,
            bias,
            is_signed,
            saturation_mode,
            subnormals_mode,
            storage=dtype,
            elementwise=True,
        )
    if widen:
        _refuse_widening_in_place("binaryK_quantize", dtype)

    return torch.ops.mptorch.binaryK_quant_.default(
        x,
        K,
        P,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
        subnormals_mode.value,
    )


def superfp_quantize(
    x: torch.Tensor,
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    prng_bits: int = 0,
    is_signed: bool = True,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a superfp (supernormal) format.

    The format has ``man_bits`` stored mantissa bits and ``exp_bits`` exponent
    bits, but only the top ``normal_binades`` binades have a mantissa. The
    codes of the lower binades encode zero and successive powers of two below
    the normal region, the *supernormals*. A value below the smallest
    supernormal rounds to zero or to it, as the rounding mode decides. There
    are no subnormals, and so no ``subnormals_mode``. ``bias`` is required,
    because there is no standard default for it. See
    :class:`mptorch.SuperFP`.

    Dtypes, the carrier, the storage check, ``prng_bits`` and
    differentiability are as for :func:`binaryK_quantize`.

    Args:
        x (Tensor): the tensor to round; float32, float64, float16 or
            bfloat16.
        man_bits (int): stored mantissa bits of a normal binade.
        exp_bits (int): exponent bits.
        normal_binades (int): how many of the top binades have a mantissa;
            the ``2**man_bits`` codes of each lower binade encode supernormal
            powers of two instead.
        bias (int): the exponent bias.
        prng_bits (int): the number of random bits the stochastic rounding
            modes draw below the format's last mantissa bit; ignored by the
            other modes. Default: ``0``
        is_signed (bool): whether the format has a sign bit. Default: ``True``
        rounding_mode (RoundMode): how a value that falls between two of the
            format's values is rounded. Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): what happens to a value beyond the
            largest finite value. Default: ``SaturationMode.SAT_NONE``
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`.
            Default: ``None``, ``x``'s own carrier.

    Returns:
        Tensor: a new tensor of ``x``'s shape and dtype holding the rounded
        values.

    Raises:
        ValueError: if the carrier cannot hold the format (more precision than
            the carrier has, ``man_bits + prng_bits`` above its mantissa bits,
            an exponent field wider than 30 bits, a ``normal_binades`` that
            leaves no supernormal codes, no finite normal value in its normal
            range), if the storage dtype can store none of the format's
            nonzero values, or if ``carrier`` is a dtype that names no carrier
            or is narrower than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode.

    Warns:
        FormatRangeWarning: if the format's range exceeds the carrier's, or a
            result at an edge of the format's range is not a value of a
            narrower storage dtype.

    Example::

        >>> x = torch.tensor([1.1, 3.3, 0.05])
        >>> superfp_quantize(x, man_bits=3, exp_bits=4, normal_binades=8, bias=7)
        tensor([1.0000, 3.2500, 0.0625])

    With ``bias=7`` the eight normal binades start at 2, so 3.3 is rounded on
    a grid of step 0.25. 1.1 and 0.05 fall in the supernormal region and are
    rounded to a power of two.
    """
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_superfp_carrier(
        man_bits, exp_bits, normal_binades, bias, saturation_mode, prng_bits, carrier=_CARRIER[wide]
    )
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_superfp_storage(
            man_bits,
            exp_bits,
            normal_binades,
            bias,
            saturation_mode,
            storage=dtype,
            elementwise=True,
        )

    out = torch.ops.mptorch.superfp_quant.default(
        _quantizer_operand(x, widen),
        man_bits,
        exp_bits,
        normal_binades,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
    )
    return _narrowed(out, dtype) if widen else out


def superfp_quantize_(
    x: torch.Tensor,
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    prng_bits: int = 0,
    is_signed: bool = True,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a superfp format, in place.

    This is the in-place version of :func:`superfp_quantize`: the same
    arguments, checks and result, written over ``x``, with no output
    allocated. Its purpose and the inputs it refuses are the same as for
    :func:`binaryK_quantize_`: a tensor that is not contiguous, a CUDA view
    that does not start on a 16-byte boundary, and a ``carrier`` wider than
    ``x``.

    Args:
        x (Tensor): the tensor to round and overwrite; float32, float64,
            float16 or bfloat16, contiguous. The format and mode arguments
            between it and ``carrier`` are :func:`superfp_quantize`'s.
        carrier (torch.dtype, optional): ``None`` or the carrier ``x`` already
            has. Default: ``None``

    Returns:
        Tensor: ``x`` itself, holding the rounded values.

    Raises:
        ValueError: as for :func:`superfp_quantize`, and if ``carrier`` is
            wider than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode, is not
            contiguous, or is a CUDA view off a 16-byte boundary.

    Warns:
        FormatRangeWarning: as for :func:`superfp_quantize`.

    Example::

        >>> x = torch.tensor([1.1, 3.3, 0.05])
        >>> superfp_quantize_(x, man_bits=3, exp_bits=4, normal_binades=8, bias=7)
        tensor([1.0000, 3.2500, 0.0625])
        >>> x
        tensor([1.0000, 3.2500, 0.0625])
    """
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_superfp_carrier(
        man_bits, exp_bits, normal_binades, bias, saturation_mode, prng_bits, carrier=_CARRIER[wide]
    )
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_superfp_storage(
            man_bits,
            exp_bits,
            normal_binades,
            bias,
            saturation_mode,
            storage=dtype,
            elementwise=True,
        )
    if widen:
        _refuse_widening_in_place("superfp_quantize", dtype)

    return torch.ops.mptorch.superfp_quant_.default(
        x,
        man_bits,
        exp_bits,
        normal_binades,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
    )


def fixedpoint_quantize(
    x: torch.Tensor,
    wl: int,
    fl: int,
    prng_bits: int = 0,
    is_signed: bool = True,
    symmetric: bool = False,
    rounding_mode: RoundMode = RoundMode.RNE,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a fixed-point format.

    The format's values are the integers that a ``wl``-bit word holds, times
    the step ``2**-fl``: ``[-2**(wl - 1 - fl), 2**(wl - 1 - fl) - 2**-fl]`` in
    two's complement, one step narrower at the bottom with ``symmetric``, and
    ``[0, 2**(wl - fl) - 2**-fl]`` unsigned (see :class:`mptorch.FixedPoint`).
    A value beyond the range, including an infinite one, saturates to the end
    of the range on its side, since a fixed-point word has no code for
    infinity; hence there is no ``saturation_mode``. NaN inputs pass through.

    Dtypes, the carrier, the storage check, ``prng_bits`` and
    differentiability are as for :func:`binaryK_quantize`.

    Args:
        x (Tensor): the tensor to round; float32, float64, float16 or
            bfloat16.
        wl (int): word length in bits, including the sign bit.
        fl (int): fractional length; the step between values is ``2**-fl``.
        prng_bits (int): the number of random bits the stochastic rounding
            modes draw below the step; ignored by the other modes.
            Default: ``0``
        is_signed (bool): whether the format has a sign bit. Default: ``True``
        symmetric (bool): drop the most negative code, so that the range is
            symmetric about zero; signed formats only. Default: ``False``
        rounding_mode (RoundMode): how a value that falls between two of the
            format's values is rounded. Default: ``RoundMode.RNE``
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`.
            Default: ``None``, ``x``'s own carrier.

    Returns:
        Tensor: a new tensor of ``x``'s shape and dtype holding the rounded
        values.

    Raises:
        ValueError: if the carrier cannot hold the format (more magnitude
            bits than its significand has, ``prng_bits`` included, a largest
            value below its smallest normal, or an end of the range beyond
            its largest finite value), for an invalid layout (``wl`` too
            short, a symmetric unsigned format, a negative ``prng_bits``), if
            the storage dtype can store none of the format's nonzero values,
            or if ``carrier`` is a dtype that names no carrier or is narrower
            than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode.

    Warns:
        FormatRangeWarning: if the format's range exceeds the carrier's, or a
            result at an edge of the format's range is not a value of a
            narrower storage dtype.

    Example::

        >>> x = torch.tensor([1.1, 3.3, -9.0, 0.03])
        >>> fixedpoint_quantize(x, wl=8, fl=4)
        tensor([ 1.1250,  3.3125, -8.0000,  0.0000])

    With ``fl=4`` the step is 1/16: 1.1 and 3.3 round to the nearest
    sixteenth, -9.0 saturates to the bottom of the range, -8, and 0.03 is
    nearer to zero than to the step.
    """
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_fixedpoint_carrier(wl, fl, is_signed, symmetric, prng_bits, carrier=_CARRIER[wide])
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_fixedpoint_storage(
            wl,
            fl,
            is_signed,
            symmetric,
            storage=dtype,
            elementwise=True,
        )

    out = torch.ops.mptorch.fixedpoint_quant.default(
        _quantizer_operand(x, widen),
        wl,
        fl,
        prng_bits,
        is_signed,
        symmetric,
        rounding_mode.value,
    )
    return _narrowed(out, dtype) if widen else out


def fixedpoint_quantize_(
    x: torch.Tensor,
    wl: int,
    fl: int,
    prng_bits: int = 0,
    is_signed: bool = True,
    symmetric: bool = False,
    rounding_mode: RoundMode = RoundMode.RNE,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a fixed-point format, in place.

    The in-place version of :func:`fixedpoint_quantize`: the same arguments,
    checks and result, written over ``x``, with no output allocated. Its
    purpose and the inputs it refuses (a tensor that is not contiguous, a CUDA
    view that does not start on a 16-byte boundary, a ``carrier`` wider than
    ``x``) are as for :func:`binaryK_quantize_`.

    Args:
        x (Tensor): the tensor to round and overwrite; float32, float64,
            float16 or bfloat16, contiguous. The format and mode arguments
            between it and ``carrier`` are :func:`fixedpoint_quantize`'s.
        carrier (torch.dtype, optional): ``None`` or the carrier ``x`` already
            has. Default: ``None``

    Returns:
        Tensor: ``x`` itself, holding the rounded values.

    Raises:
        ValueError: as for :func:`fixedpoint_quantize`, and if ``carrier`` is
            wider than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode, is not
            contiguous, or is a CUDA view off a 16-byte boundary.

    Warns:
        FormatRangeWarning: as for :func:`fixedpoint_quantize`.

    Example::

        >>> x = torch.tensor([1.1, 3.3, -9.0, 0.03])
        >>> fixedpoint_quantize_(x, wl=8, fl=4)
        tensor([ 1.1250,  3.3125, -8.0000,  0.0000])
        >>> x
        tensor([ 1.1250,  3.3125, -8.0000,  0.0000])
    """
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_fixedpoint_carrier(wl, fl, is_signed, symmetric, prng_bits, carrier=_CARRIER[wide])
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_fixedpoint_storage(
            wl,
            fl,
            is_signed,
            symmetric,
            storage=dtype,
            elementwise=True,
        )
    if widen:
        _refuse_widening_in_place("fixedpoint_quantize", dtype)

    return torch.ops.mptorch.fixedpoint_quant_.default(
        x,
        wl,
        fl,
        prng_bits,
        is_signed,
        symmetric,
        rounding_mode.value,
    )


# --- single-format GEMMs -----------------------------------------------------


def _binaryK_spec(
    *,
    mul_K: int,
    mul_P: int,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_K: int | None = None,
    acc_P: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul`'s formats into a ``_GemmSpec``.

    That function documents the arguments. This one fills in the ``acc_*``
    fallbacks to the multiply format, the default biases, and the zeroed
    accumulate format of ``accumulate_quant=False``, and raises binary64's
    errors.
    """
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if acc_subnormals_mode is None:
        acc_subnormals_mode = subnormals_mode
    if accumulate_quant:
        acc_K = mul_K if acc_K is None else acc_K
        acc_P = mul_P if acc_P is None else acc_P
    else:
        acc_K = acc_K or 0
        acc_P = acc_P or 0

    if mul_bias is None:
        mul_bias = _binaryK_bias(mul_K, mul_P, mul_is_signed)
    if accumulate_quant and acc_bias is None:
        acc_bias = _binaryK_bias(acc_K, acc_P, acc_is_signed)
    elif acc_bias is None:
        acc_bias = 0

    mul = (mul_K, mul_P, mul_bias, mul_is_signed, saturation_mode, subnormals_mode)
    rounded: list[_Format] = [(_binaryK_findings, (*mul, mul_prng_bits))]
    # The result holds values of the accumulate format (of no format with
    # accumulate_quant=False); the products are never stored.
    stored: tuple[_Stored, ...] = ()
    if accumulate_quant:
        acc = (acc_K, acc_P, acc_bias, acc_is_signed, acc_saturation_mode, acc_subnormals_mode)
        rounded.append((_binaryK_findings, (*acc, acc_prng_bits)))
        stored = ((check_binaryK_storage, acc),)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_binaryK.default,
        (
            mul_K,
            mul_P,
            mul_bias,
            mul_is_signed,
            accumulate_quant,
            acc_K,
            acc_P,
            acc_bias,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            acc_saturation_mode.value,
            acc_subnormals_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        stored,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def binaryK_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_K: int,
    mul_P: int,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_K: int | None = None,
    acc_P: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    block_size: int | None = None,
    outer_K: int | None = None,
    outer_P: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    outer_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM with a split multiply-accumulate in binaryK formats.

    Computes ``op(a) @ op(b)``, where ``op(x) = x.T`` when the corresponding
    ``trans_*`` flag is set, and ``op(x) = x`` otherwise. This is not the
    same as quantizing ``a`` and ``b`` and then calling ``torch.matmul``. It
    quantizes the *arithmetic* of each dot product. Every partial product is
    rounded to the binaryK format given by ``mul_K``/``mul_P``. When
    ``accumulate_quant`` is true, the running sum is rounded to the format
    given by ``acc_K``/``acc_P`` after every accumulation step. With
    ``accumulate_quant=False`` the running sum stays at the carrier's
    precision and only the multiply is quantized.

    Everything the formats do not round (a product or a sum before its
    rounding) is computed in a *carrier*, the binary floating-point type the
    kernel computes in, and so are the roundings. The carrier is binary64 for
    float64 operands and binary32 for float32, float16 and bfloat16, unless
    ``carrier`` names one. Every call checks both formats against that
    carrier. The result holds the accumulate format's values, so that format
    is also checked against the *storage dtype*, the operands' dtype, when it
    is narrower than the carrier. Unlike for :func:`binaryK_quantize`, every
    value of the accumulate format must then be a value of the storage dtype,
    since a sum can take any of them.

    ``a`` and ``b`` follow ``torch.matmul``'s operand rules: 1D operands are
    promoted (and their dimension is dropped from the result), leading
    dimensions broadcast against each other, and ``trans_a``/``trans_b``
    apply to the last two dimensions. Three cases need no copy:

    * An operand shared across the batch is read at stride 0 instead of being
      expanded.
    * An operand that is the transpose of a contiguous tensor flips the
      kernel's transpose flag instead of being copied.
    * ``[..., M, K] @ [K, N]`` folds its batch into ``M`` as a view.

    These cover the two shapes that a quantized attention block uses. Under
    stochastic rounding each output element's random stream is selected by
    its index into the ``[batch, M, N]`` result, so the folded and the
    batched forms give bit-identical results.

    This function resolves the format on every call. A caller that keeps one
    format across many calls should use the factories of
    `mptorch.quant.gemm`, which resolve it once per layer.
    :func:`mptorch.quant.qmatmul` is the differentiable entry point.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_K (int): width in bits of the multiply format.
        mul_P (int): precision of the multiply format.
        mul_bias (int, optional): exponent bias of the multiply format.
            Default: ``None``, P3109's ``2**(K - P - 1)`` if signed and
            ``2**(K - P)`` if unsigned.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits the stochastic rounding modes draw
            for the multiply rounding, as :func:`binaryK_quantize`'s
            ``prng_bits``. Default: ``0``
        accumulate_quant (bool): whether the running sum is rounded to the
            accumulate format after every step. Default: ``True``
        acc_K (int, optional): width of the accumulate format. Default:
            ``None``, ``mul_K``.
        acc_P (int, optional): precision of the accumulate format. Default:
            ``None``, ``mul_P``.
        acc_bias (int, optional): exponent bias of the accumulate format.
            Default: ``None``, P3109's for ``acc_K``/``acc_P``.
        acc_is_signed (bool, optional): whether the accumulate format has a
            sign bit. Default: ``None``, ``mul_is_signed``.
        acc_prng_bits (int): random bits the stochastic rounding modes draw
            for the accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed. ``NAIVE`` is the step above, in order. ``KAHAN`` carries a
            compensation term next to the sum, rounded to the accumulate
            format. ``BLOCK`` runs the step above over blocks of
            ``block_size`` products and adds each block's sum to a total,
            which the outer format rounds. ``TREE`` sums each block's
            products pairwise, rounding every pair's sum to the accumulate
            format, and adds the block's sum to the total in the same way.
            :class:`mptorch.AccumulateAlgorithm` states each precisely.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of both formats. One
            mode applies to the whole op, because the kernel is compiled for
            it. Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply format's overflow
            behavior. Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the multiply format has below
            its smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            format's overflow behavior. Default: ``None``, ``saturation_mode``.
        acc_subnormals_mode (SubnormalsMode, optional): what the accumulate
            format has below its smallest normal number. Default: ``None``,
            ``subnormals_mode``.
        block_size (int, optional): products per block. ``BLOCK`` requires
            one, which must divide 16 or be a multiple of 16 (the kernels
            work in slabs of 16 steps). ``TREE`` takes a power of two in
            [2, 256] and defaults to 16. ``NAIVE`` and ``KAHAN`` take none.
            Default: ``None``
        outer_K (int, optional): width of the outer format, the format that
            ``BLOCK`` and ``TREE`` round the total to after each block is
            added. ``None`` (together with ``outer_P``) leaves the total
            unrounded, in the carrier. When given, it is the format whose
            values the result holds, and so the one checked against a
            narrower storage dtype. Default: ``None``
        outer_P (int, optional): precision of the outer format; given
            together with ``outer_K``. Default: ``None``
        outer_bias (int, optional): exponent bias of the outer format.
            Default: ``None``, P3109's for ``outer_K``/``outer_P``.
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, the accumulate format's.
        outer_prng_bits (int): random bits the stochastic rounding modes draw
            for the outer rounding. Default: ``0``
        outer_saturation_mode (SaturationMode, optional): the outer format's
            overflow behavior. Default: ``None``, the accumulate format's.
        outer_subnormals_mode (SubnormalsMode, optional): what the outer
            format has below its smallest normal number. Default: ``None``,
            the accumulate format's.
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`.
            ``torch.float64`` widens narrower operands to binary64 and
            narrows the result back to their dtype with a single rounding.
            ``torch.float32`` refuses float64 operands. Default: ``None``,
            the operands' own carrier.
        carrier_rounding (RoundMode): how the carrier rounds each product
            and sum before the formats round it. ``RoundMode.RNE`` uses the
            carrier's own round-to-nearest multiply and add, after which a
            directed rounding mode can return the wrong neighbour (the value
            is rounded twice). ``RoundMode.RO`` rounds them to odd instead,
            so that each format rounds the exact product and the exact sum.
            That holds for formats with ``P`` at most 22 under ``RNE`` and
            ``RNA`` and 23 under the directed modes and ``RO``, and with
            ``P + prng_bits`` at most 23 under ``SR`` and 22 under ``SR_B``
            and ``SR_C``, in binary32 (51, 52, 52 and 51 in binary64); a call
            outside those limits raises. ``RO`` runs the
            ``custom_matmul_*_odd`` ops, on CPU, CUDA and MPS tensors
            (binary32 only on MPS). Under ``KAHAN``, ``BLOCK`` and ``TREE``
            it runs the ``*_accumulated_odd`` ops, which also check the outer
            format against the limits and need a build with
            ``MPTORCH_ODD_ACCUMULATED=1``. With ``accumulate_quant=False``
            only the product is rounded to odd. See
            :doc:`/kernels/arithmetic`, "What the carrier rounds".
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold a format (as for
            :func:`binaryK_quantize`), if the format the result holds has
            more precision than a narrower storage dtype, if ``carrier`` names
            no carrier or is narrower than the operands, if ``block_size`` or
            the outer format does not suit ``accumulate_algorithm``, if
            ``carrier_rounding`` is neither ``RNE`` nor ``RO`` or a format
            exceeds the round-to-odd limits, or if the operands' leading
            dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode (use :func:`mptorch.quant.qmatmul`).

    Warns:
        FormatRangeWarning: if a format's range exceeds the carrier's, or the
            format the result holds (the accumulate or outer format) has
            values that a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> binaryK_matmul(a, a, mul_K=8, mul_P=4)
        tensor([[ 7., 10.],
                [15., 22.]])
        >>> binaryK_matmul(a, a, mul_K=8, mul_P=3)
        tensor([[ 7., 10.],
                [16., 24.]])

    With three stored mantissa bits (``P=4``) every product and sum is exact.
    With two (``P=3``) the sums 15 and 22 are rounded to 16 and 24.
    """
    # A NAIVE call sets none of the accumulation keywords, and resolves
    # through the NAIVE builder with the same keywords as before the other
    # algorithms existed; this test is all it pays for them.
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_K is None
        and outer_P is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _binaryK_spec(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _binaryK_accumulated_spec(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
            carrier=carrier,
            block_size=block_size,
            outer_K=outer_K,
            outer_P=outer_P,
            outer_bias=outer_bias,
            outer_is_signed=outer_is_signed,
            outer_prng_bits=outer_prng_bits,
            outer_saturation_mode=outer_saturation_mode,
            outer_subnormals_mode=outer_subnormals_mode,
        )
    else:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_K is not None
            or outer_P is not None
            or outer_bias is not None
        ):
            spec = _binaryK_accumulated_odd_spec(
                mul_K=mul_K,
                mul_P=mul_P,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                accumulate_quant=accumulate_quant,
                acc_K=acc_K,
                acc_P=acc_P,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
                acc_saturation_mode=acc_saturation_mode,
                acc_subnormals_mode=acc_subnormals_mode,
                carrier=carrier,
                block_size=block_size,
                outer_K=outer_K,
                outer_P=outer_P,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
                outer_subnormals_mode=outer_subnormals_mode,
            )
        else:
            spec = _binaryK_odd_spec(
                mul_K=mul_K,
                mul_P=mul_P,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                accumulate_quant=accumulate_quant,
                acc_K=acc_K,
                acc_P=acc_P,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
                acc_saturation_mode=acc_saturation_mode,
                acc_subnormals_mode=acc_subnormals_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


def _superfp_spec(
    *,
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_man_bits: int | None = None,
    acc_exp_bits: int | None = None,
    acc_normal_binades: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul`'s formats into a ``_GemmSpec``.

    That function documents the arguments. This one fills in the ``acc_*``
    fallbacks to the multiply format and the zeroed accumulate format of
    ``accumulate_quant=False``, and raises binary64's errors.
    """
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if accumulate_quant:
        acc_man_bits = mul_man_bits if acc_man_bits is None else acc_man_bits
        acc_exp_bits = mul_exp_bits if acc_exp_bits is None else acc_exp_bits
        acc_normal_binades = (
            mul_normal_binades if acc_normal_binades is None else acc_normal_binades
        )
        acc_bias = mul_bias if acc_bias is None else acc_bias
    else:
        acc_man_bits = acc_man_bits or 0
        acc_exp_bits = acc_exp_bits or 0
        acc_normal_binades = acc_normal_binades or 0
        acc_bias = acc_bias or 0

    mul = (mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, saturation_mode)
    rounded: list[_Format] = [(_superfp_findings, (*mul, mul_prng_bits))]
    stored: tuple[_Stored, ...] = ()
    if accumulate_quant:
        acc = (acc_man_bits, acc_exp_bits, acc_normal_binades, acc_bias, acc_saturation_mode)
        rounded.append((_superfp_findings, (*acc, acc_prng_bits)))
        stored = ((check_superfp_storage, acc),)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_superfp.default,
        (
            mul_man_bits,
            mul_exp_bits,
            mul_normal_binades,
            mul_bias,
            mul_is_signed,
            accumulate_quant,
            acc_man_bits,
            acc_exp_bits,
            acc_normal_binades,
            acc_bias,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            acc_saturation_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        stored,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def superfp_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_man_bits: int | None = None,
    acc_exp_bits: int | None = None,
    acc_normal_binades: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    block_size: int | None = None,
    outer_man_bits: int | None = None,
    outer_exp_bits: int | None = None,
    outer_normal_binades: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM with a split multiply-accumulate in superfp formats.

    The superfp counterpart of :func:`binaryK_matmul`. Every partial product
    is rounded to the superfp format given by ``mul_man_bits``,
    ``mul_exp_bits``, ``mul_normal_binades`` and ``mul_bias``. When
    ``accumulate_quant`` is true, the running sum is rounded to the ``acc_*``
    format after every step. :func:`binaryK_matmul` documents what the two
    have in common: ``torch.matmul``'s operand rules, the transpose flags, the
    carrier, the storage check, and resolving the formats on every call.
    :func:`superfp_quantize` describes the format.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_man_bits (int): stored mantissa bits of the multiply format.
        mul_exp_bits (int): exponent bits of the multiply format.
        mul_normal_binades (int): the number of top binades of the multiply
            format that have a mantissa.
        mul_bias (int): exponent bias of the multiply format.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            multiply rounding. Default: ``0``
        accumulate_quant (bool): whether the running sum is rounded to the
            accumulate format after every step. Default: ``True``
        acc_man_bits (int, optional): the accumulate format's mantissa bits.
            Default: ``None``, ``mul_man_bits``.
        acc_exp_bits (int, optional): the accumulate format's exponent bits.
            Default: ``None``, ``mul_exp_bits``.
        acc_normal_binades (int, optional): the accumulate format's normal
            binades. Default: ``None``, ``mul_normal_binades``.
        acc_bias (int, optional): the accumulate format's exponent bias.
            Default: ``None``, ``mul_bias``.
        acc_is_signed (bool, optional): whether the accumulate format has a
            sign bit. Default: ``None``, ``mul_is_signed``.
        acc_prng_bits (int): random bits stochastic rounding draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :func:`binaryK_matmul`. Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of both formats. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply format's overflow
            behavior. Default: ``SaturationMode.SAT_NONE``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            format's overflow behavior. Default: ``None``, ``saturation_mode``.
        block_size (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_man_bits (int, optional): mantissa bits of the outer format, the
            format that ``BLOCK`` and ``TREE`` round the total to after each
            block is added (:func:`binaryK_matmul`'s ``outer_K``). The four
            outer fields (``outer_man_bits``, ``outer_exp_bits``,
            ``outer_normal_binades``, ``outer_bias``) are given together, or
            none of them, which leaves the total unrounded, in the carrier.
            Default: ``None``
        outer_exp_bits (int, optional): exponent bits of the outer format.
            Default: ``None``
        outer_normal_binades (int, optional): normal binades of the outer
            format. Default: ``None``
        outer_bias (int, optional): exponent bias of the outer format.
            Default: ``None``
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, the accumulate format's.
        outer_prng_bits (int): random bits stochastic rounding draws for the
            outer rounding. Default: ``0``
        outer_saturation_mode (SaturationMode, optional): the outer format's
            overflow behavior. Default: ``None``, the accumulate format's.
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold a format (as for
            :func:`superfp_quantize`), and in the other cases listed for
            :func:`binaryK_matmul`.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if a format's range exceeds the carrier's, or the
            format the result holds has values a narrower storage dtype cannot
            represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> superfp_matmul(
        ...     a, a, mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=8, mul_bias=7
        ... )
        tensor([[ 7., 10.],
                [15., 22.]])
    """
    # A NAIVE call sets none of the accumulation keywords, and resolves
    # through the NAIVE builder with the same keywords as before the other
    # algorithms existed; this test is all it pays for them.
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_man_bits is None
        and outer_exp_bits is None
        and outer_normal_binades is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _superfp_spec(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            acc_saturation_mode=acc_saturation_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _superfp_accumulated_spec(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            acc_saturation_mode=acc_saturation_mode,
            carrier=carrier,
            block_size=block_size,
            outer_man_bits=outer_man_bits,
            outer_exp_bits=outer_exp_bits,
            outer_normal_binades=outer_normal_binades,
            outer_bias=outer_bias,
            outer_is_signed=outer_is_signed,
            outer_prng_bits=outer_prng_bits,
            outer_saturation_mode=outer_saturation_mode,
        )
    else:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_man_bits is not None
            or outer_exp_bits is not None
            or outer_normal_binades is not None
            or outer_bias is not None
        ):
            spec = _superfp_accumulated_odd_spec(
                mul_man_bits=mul_man_bits,
                mul_exp_bits=mul_exp_bits,
                mul_normal_binades=mul_normal_binades,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                accumulate_quant=accumulate_quant,
                acc_man_bits=acc_man_bits,
                acc_exp_bits=acc_exp_bits,
                acc_normal_binades=acc_normal_binades,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                acc_saturation_mode=acc_saturation_mode,
                carrier=carrier,
                block_size=block_size,
                outer_man_bits=outer_man_bits,
                outer_exp_bits=outer_exp_bits,
                outer_normal_binades=outer_normal_binades,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
            )
        else:
            spec = _superfp_odd_spec(
                mul_man_bits=mul_man_bits,
                mul_exp_bits=mul_exp_bits,
                mul_normal_binades=mul_normal_binades,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                accumulate_quant=accumulate_quant,
                acc_man_bits=acc_man_bits,
                acc_exp_bits=acc_exp_bits,
                acc_normal_binades=acc_normal_binades,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                acc_saturation_mode=acc_saturation_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


def _binaryK_fma_spec(
    *,
    fma_K: int,
    fma_P: int,
    fma_bias: int | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_fma`'s format into a ``_GemmSpec``.

    That function documents the arguments. With ``fma_quant=False`` no format
    rounds anything and no format's values are stored, so the carriers have
    nothing to check.
    """
    if fma_bias is None:
        fma_bias = _binaryK_bias(fma_K, fma_P, fma_is_signed)
    rounded: list[_Format] = []
    stored: tuple[_Stored, ...] = ()
    if fma_quant:
        fma = (fma_K, fma_P, fma_bias, fma_is_signed, saturation_mode, subnormals_mode)
        rounded.append((_binaryK_findings, (*fma, fma_prng_bits)))
        stored = ((check_binaryK_storage, fma),)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_binaryK_fma.default,
        (
            fma_quant,
            fma_K,
            fma_P,
            fma_bias,
            fma_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            fma_prng_bits,
        ),
        stored,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def binaryK_matmul_fma(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_K: int,
    fma_P: int,
    fma_bias: int | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    block_size: int | None = None,
    outer_K: int | None = None,
    outer_P: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    outer_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM with a fused multiply-add in a binaryK format.

    The fused counterpart of :func:`binaryK_matmul`. Each step of a dot
    product is one fused multiply-add, as in hardware: ``s + a * b``, with
    ``s`` the running sum, rounded once, with no separate rounding of the
    product. A hardware FMA unit rounds only its output, so there is a single
    ``fma_K``/``fma_P`` format instead of a ``mul_*``/``acc_*`` pair. With
    ``fma_quant=False`` no format rounds the step, and the GEMM runs entirely
    in the carrier.

    ``a`` and ``b`` follow ``torch.matmul``'s operand rules. The carrier, the
    storage check and resolving the format on every call are as for
    :func:`binaryK_matmul`.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        fma_K (int): width in bits of the fused step's format.
        fma_P (int): precision of the fused step's format.
        fma_bias (int, optional): its exponent bias. Default: ``None``,
            P3109's ``2**(K - P - 1)`` if signed and ``2**(K - P)`` if
            unsigned.
        fma_is_signed (bool): whether the format has a sign bit. Default:
            ``True``
        fma_quant (bool): whether the fused step is rounded to the format at
            all. Default: ``True``
        fma_prng_bits (int): random bits stochastic rounding draws for the fused
            rounding, as :func:`binaryK_quantize`'s ``prng_bits``. Default:
            ``0``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :func:`binaryK_matmul`, with the fused step in place of the split
            one, except ``TREE``, which raises: a fused multiply-add has no
            product term to sum pairwise. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the fused step's rounding mode. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the format's overflow behavior.
            Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the format has below its
            smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        block_size (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_K (int, optional): as for :func:`binaryK_matmul`. Default: ``None``
        outer_P (int, optional): as for :func:`binaryK_matmul`. Default: ``None``
        outer_bias (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, ``fma_is_signed``.
        outer_prng_bits (int): as for :func:`binaryK_matmul`. Default: ``0``
        outer_saturation_mode (SaturationMode, optional): the outer format's
            overflow behavior. Default: ``None``, ``saturation_mode``.
        outer_subnormals_mode (SubnormalsMode, optional): what the outer format
            has below its smallest normal number. Default: ``None``,
            ``subnormals_mode``.
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`;
            ``RoundMode.RO`` needs ``fma_quant=True``. Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: in the cases listed for :func:`binaryK_matmul`, and for
            ``accumulate_algorithm=TREE`` or for ``carrier_rounding=RO`` with
            ``fma_quant=False``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if the format's range exceeds the carrier's, or the
            format has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> binaryK_matmul_fma(a, a, fma_K=8, fma_P=3)
        tensor([[ 7., 10.],
                [16., 24.]])
    """
    # A NAIVE call sets none of the accumulation keywords, and resolves
    # through the NAIVE builder with the same keywords as before the other
    # algorithms existed; this test is all it pays for them.
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_K is None
        and outer_P is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _binaryK_fma_spec(
            fma_K=fma_K,
            fma_P=fma_P,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _binaryK_fma_accumulated_spec(
            fma_K=fma_K,
            fma_P=fma_P,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            carrier=carrier,
            block_size=block_size,
            outer_K=outer_K,
            outer_P=outer_P,
            outer_bias=outer_bias,
            outer_is_signed=outer_is_signed,
            outer_prng_bits=outer_prng_bits,
            outer_saturation_mode=outer_saturation_mode,
            outer_subnormals_mode=outer_subnormals_mode,
        )
    else:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_K is not None
            or outer_P is not None
            or outer_bias is not None
        ):
            spec = _binaryK_fma_accumulated_odd_spec(
                fma_K=fma_K,
                fma_P=fma_P,
                fma_bias=fma_bias,
                fma_is_signed=fma_is_signed,
                fma_quant=fma_quant,
                fma_prng_bits=fma_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
                carrier=carrier,
                block_size=block_size,
                outer_K=outer_K,
                outer_P=outer_P,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
                outer_subnormals_mode=outer_subnormals_mode,
            )
        else:
            spec = _binaryK_fma_odd_spec(
                fma_K=fma_K,
                fma_P=fma_P,
                fma_bias=fma_bias,
                fma_is_signed=fma_is_signed,
                fma_quant=fma_quant,
                fma_prng_bits=fma_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


def _superfp_fma_spec(
    *,
    fma_man_bits: int,
    fma_exp_bits: int,
    fma_normal_binades: int,
    fma_bias: int,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_fma`'s format into a ``_GemmSpec``.

    That function documents the arguments. With ``fma_quant=False`` no format
    rounds anything and no format's values are stored, so the carriers have
    nothing to check.
    """
    rounded: list[_Format] = []
    stored: tuple[_Stored, ...] = ()
    if fma_quant:
        fma = (fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias, saturation_mode)
        rounded.append((_superfp_findings, (*fma, fma_prng_bits)))
        stored = ((check_superfp_storage, fma),)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_superfp_fma.default,
        (
            fma_quant,
            fma_man_bits,
            fma_exp_bits,
            fma_normal_binades,
            fma_bias,
            fma_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            fma_prng_bits,
        ),
        stored,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def superfp_matmul_fma(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_man_bits: int,
    fma_exp_bits: int,
    fma_normal_binades: int,
    fma_bias: int,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    block_size: int | None = None,
    outer_man_bits: int | None = None,
    outer_exp_bits: int | None = None,
    outer_normal_binades: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM with a fused multiply-add in a superfp format.

    The superfp counterpart of :func:`binaryK_matmul_fma`. Each step of a dot
    product is one fused multiply-add, rounded once to the format given by
    ``fma_man_bits``, ``fma_exp_bits``, ``fma_normal_binades`` and
    ``fma_bias`` (see :func:`superfp_quantize`). With ``fma_quant=False`` the
    step is not rounded by any format. ``a`` and ``b`` follow
    ``torch.matmul``'s operand rules. The carrier, the storage check and
    resolving the format on every call are as for :func:`binaryK_matmul`.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        fma_man_bits (int): stored mantissa bits of the fused step's format.
        fma_exp_bits (int): its exponent bits.
        fma_normal_binades (int): the number of its top binades that have a
            mantissa.
        fma_bias (int): its exponent bias.
        fma_is_signed (bool): whether the format has a sign bit. Default:
            ``True``
        fma_quant (bool): whether the fused step is rounded to the format at
            all. Default: ``True``
        fma_prng_bits (int): random bits stochastic rounding draws for the fused
            rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :func:`binaryK_matmul`, with the fused step in place of the split
            one, except ``TREE``, which raises: a fused multiply-add has no
            product term to sum pairwise. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the fused step's rounding mode. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the format's overflow behavior.
            Default: ``SaturationMode.SAT_NONE``
        block_size (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_man_bits (int, optional): mantissa bits of the outer format, the
            format that ``BLOCK`` and ``TREE`` round the total to after each
            block is added (:func:`binaryK_matmul`'s ``outer_K``). The four
            outer fields (``outer_man_bits``, ``outer_exp_bits``,
            ``outer_normal_binades``, ``outer_bias``) are given together, or
            none of them, which leaves the total unrounded, in the carrier.
            Default: ``None``
        outer_exp_bits (int, optional): exponent bits of the outer format.
            Default: ``None``
        outer_normal_binades (int, optional): normal binades of the outer
            format. Default: ``None``
        outer_bias (int, optional): exponent bias of the outer format.
            Default: ``None``
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, ``fma_is_signed``.
        outer_prng_bits (int): random bits stochastic rounding draws for the
            outer rounding. Default: ``0``
        outer_saturation_mode (SaturationMode, optional): the outer format's
            overflow behavior. Default: ``None``, ``saturation_mode``.
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`;
            ``RoundMode.RO`` needs ``fma_quant=True``. Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: in the cases listed for :func:`binaryK_matmul`, and for
            ``accumulate_algorithm=TREE`` or for ``carrier_rounding=RO`` with
            ``fma_quant=False``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if the format's range exceeds the carrier's, or the
            format has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> superfp_matmul_fma(
        ...     a, a, fma_man_bits=2, fma_exp_bits=4, fma_normal_binades=8, fma_bias=7
        ... )
        tensor([[ 7., 10.],
                [16., 24.]])
    """
    # A NAIVE call sets none of the accumulation keywords, and resolves
    # through the NAIVE builder with the same keywords as before the other
    # algorithms existed; this test is all it pays for them.
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_man_bits is None
        and outer_exp_bits is None
        and outer_normal_binades is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _superfp_fma_spec(
            fma_man_bits=fma_man_bits,
            fma_exp_bits=fma_exp_bits,
            fma_normal_binades=fma_normal_binades,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _superfp_fma_accumulated_spec(
            fma_man_bits=fma_man_bits,
            fma_exp_bits=fma_exp_bits,
            fma_normal_binades=fma_normal_binades,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            carrier=carrier,
            block_size=block_size,
            outer_man_bits=outer_man_bits,
            outer_exp_bits=outer_exp_bits,
            outer_normal_binades=outer_normal_binades,
            outer_bias=outer_bias,
            outer_is_signed=outer_is_signed,
            outer_prng_bits=outer_prng_bits,
            outer_saturation_mode=outer_saturation_mode,
        )
    else:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_man_bits is not None
            or outer_exp_bits is not None
            or outer_normal_binades is not None
            or outer_bias is not None
        ):
            spec = _superfp_fma_accumulated_odd_spec(
                fma_man_bits=fma_man_bits,
                fma_exp_bits=fma_exp_bits,
                fma_normal_binades=fma_normal_binades,
                fma_bias=fma_bias,
                fma_is_signed=fma_is_signed,
                fma_quant=fma_quant,
                fma_prng_bits=fma_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                carrier=carrier,
                block_size=block_size,
                outer_man_bits=outer_man_bits,
                outer_exp_bits=outer_exp_bits,
                outer_normal_binades=outer_normal_binades,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
            )
        else:
            spec = _superfp_fma_odd_spec(
                fma_man_bits=fma_man_bits,
                fma_exp_bits=fma_exp_bits,
                fma_normal_binades=fma_normal_binades,
                fma_bias=fma_bias,
                fma_is_signed=fma_is_signed,
                fma_quant=fma_quant,
                fma_prng_bits=fma_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


# --- palette GEMMs, whose format is chosen per output element ----------------
#
# A palette op takes up to eight formats and a `prec_idx` map that chooses one
# of them for each output element. Its builder turns every palette argument
# into a list of the palette's length (repeating a scalar, and filling in each
# entry's default for `None`) and removes duplicate formats before the carrier
# and storage checks, so that a scalar repeated across the palette is checked
# once.


def _binaryK_mixed_spec(
    *,
    mul_K: Sequence[int],
    mul_P: Sequence[int],
    mul_bias: int | Sequence[int] | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_K: int | Sequence[int] | None = None,
    acc_P: int | Sequence[int] | None = None,
    acc_bias: int | Sequence[int] | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_mixed`'s palette into a ``_GemmSpec``.

    That function documents the arguments. This one turns every palette
    argument into a list of the palette's length, with each entry's defaults
    filled in, and raises binary64's errors.
    """
    mul_K_l, mul_P_l, n = _palette_pair(mul_K, mul_P, "mul_P", "binaryK_matmul_mixed")
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if acc_subnormals_mode is None:
        acc_subnormals_mode = subnormals_mode

    mul_bias_l = _binaryK_palette_bias(mul_bias, mul_K_l, mul_P_l, mul_is_signed, "mul_bias")
    if accumulate_quant:
        acc_K_l = _palette_list_or(acc_K, mul_K_l, n, "acc_K")
        acc_P_l = _palette_list_or(acc_P, mul_P_l, n, "acc_P")
        acc_bias_l = _binaryK_palette_bias(acc_bias, acc_K_l, acc_P_l, acc_is_signed, "acc_bias")
    else:
        acc_K_l = [0] * n
        acc_P_l = [0] * n
        acc_bias_l = [0] * n

    rounded = _palette_formats(
        _binaryK_findings,
        zip(mul_K_l, mul_P_l, mul_bias_l, strict=True),
        (mul_is_signed, saturation_mode, subnormals_mode, mul_prng_bits),
    )
    stored: tuple[_Stored, ...] = ()
    if accumulate_quant:
        acc_widths = tuple(zip(acc_K_l, acc_P_l, acc_bias_l, strict=True))
        acc_shared = (acc_is_signed, acc_saturation_mode, acc_subnormals_mode)
        rounded += _palette_formats(_binaryK_findings, acc_widths, (*acc_shared, acc_prng_bits))
        stored = _palette_formats(check_binaryK_storage, acc_widths, acc_shared)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_binaryK_mixed.default,
        (
            mul_K_l,
            mul_P_l,
            mul_bias_l,
            mul_is_signed,
            accumulate_quant,
            acc_K_l,
            acc_P_l,
            acc_bias_l,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            acc_saturation_mode.value,
            acc_subnormals_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        stored,
        mixed=True,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def binaryK_matmul_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_K: Sequence[int],
    mul_P: Sequence[int],
    mul_bias: int | Sequence[int] | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_K: int | Sequence[int] | None = None,
    acc_P: int | Sequence[int] | None = None,
    acc_bias: int | Sequence[int] | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM whose binaryK format varies per output element.

    The palette counterpart of :func:`binaryK_matmul`. The arithmetic is the
    same split multiply-accumulate, a rounded multiply followed by a rounded
    accumulate, but the binaryK formats of each output element's dot product
    are chosen by ``prec_idx`` from a *palette* of up to 8 entries, each a
    pair of multiply and accumulate formats.

    ``mul_K`` and ``mul_P`` are per-entry sequences, and their common length
    is the palette size ``n``. Each of the other palette arguments may be a
    scalar (used for every entry), an ``n``-length sequence, or ``None``.
    ``None`` gives each entry the default of :func:`binaryK_matmul`: an
    ``acc_*`` argument falls back to the entry's ``mul_*`` value, and a bias to
    P3109's. Every format is checked against the call's carrier, and every
    accumulate format against a narrower storage dtype.

    ``prec_idx`` selects a palette entry per output element. Its shape must
    be the GEMM's output shape ``[M, N]`` (dense), ``[M, 1]`` (per row) or
    ``[1, N]`` (per column). It may have a leading batch dimension of ``B``
    (one map per batch element) or 1. The batch is never folded into ``M``
    here, because the map is indexed by output element.

    Any integer dtype and layout is accepted. A map that is already
    ``int32``, contiguous and on ``a``'s device is used as it is; any other
    map is converted on every call. On CUDA the bounds check copies the map's
    smallest and largest entries back to the host, which stalls the device,
    so its result is cached on the map tensor's identity and version (the
    count of its in-place writes). Keep one map and reuse it across calls: a
    map built anew for every call is checked on every call.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        prec_idx (Tensor): the palette index of each output element, in
            ``[0, n)``.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_K (Sequence[int]): width in bits of each entry's multiply format.
        mul_P (Sequence[int]): precision of each entry's multiply format.
        mul_bias (int or Sequence[int], optional): exponent bias of each
            entry's multiply format. Default: ``None``, P3109's per entry.
        mul_is_signed (bool): whether the multiply formats have a sign bit;
            shared by the palette. Default: ``True``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            multiply rounding; shared by the palette. Default: ``0``
        accumulate_quant (bool): whether the running sum is rounded to the
            accumulate format after every step. Default: ``True``
        acc_K (int or Sequence[int], optional): width of each entry's
            accumulate format. Default: ``None``, the ``mul_K`` entry.
        acc_P (int or Sequence[int], optional): precision of each entry's
            accumulate format. Default: ``None``, the ``mul_P`` entry.
        acc_bias (int or Sequence[int], optional): exponent bias of each
            entry's accumulate format. Default: ``None``, P3109's per entry.
        acc_is_signed (bool, optional): whether the accumulate formats have a
            sign bit. Default: ``None``, ``mul_is_signed``.
        acc_prng_bits (int): random bits stochastic rounding draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed; the palette ops implement ``NAIVE`` only.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of every format.
            Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply formats' overflow
            behavior. Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the multiply formats have
            below their smallest normal number.
            Default: ``SubnormalsMode.SUBNORMALS``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            formats' overflow behavior. Default: ``None``, ``saturation_mode``.
        acc_subnormals_mode (SubnormalsMode, optional): what the accumulate
            formats have below their smallest normal number. Default: ``None``,
            ``subnormals_mode``.
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`, applied
            to every palette entry: ``RoundMode.RO`` runs the palette's
            round-to-odd op, and each entry's ``P`` (``P + prng_bits`` under
            the stochastic modes) is checked against the limits.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if the palette is empty or a palette argument's length is
            not ``n``, if a carrier cannot hold an entry, if an entry the
            result holds has more precision than a narrower storage dtype, if
            ``carrier`` names no carrier or is narrower than the operands, if
            ``carrier_rounding`` is neither ``RNE`` nor ``RO`` or an entry
            exceeds the round-to-odd limits, or if the operands' leading
            dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``n`` is above 8, ``prec_idx`` has the wrong shape or
            an entry outside ``[0, n)``, ``accumulate_algorithm`` is not
            ``NAIVE`` (a ``ValueError`` under ``carrier_rounding=RO``), the
            inner dimensions do not match, the operands differ in dtype or
            device, or an operand requires grad under grad mode.

    Warns:
        FormatRangeWarning: if an entry's range exceeds the carrier's, or an
            accumulate entry has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> prec_idx = torch.tensor([[0, 0], [0, 1]])
        >>> binaryK_matmul_mixed(a, a, prec_idx, mul_K=[8, 8], mul_P=[4, 3])
        tensor([[ 7., 10.],
                [15., 24.]])

    Three output elements use the ``P=4`` entry and are exact. The last uses
    the ``P=3`` entry, whose two mantissa bits round the sum 22 to 24.
    """
    build = _binaryK_mixed_spec
    if carrier_rounding is not _RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        build = _binaryK_mixed_odd_spec
    return _gemm_mixed_nd(
        build(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
            carrier=carrier,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


def _superfp_mixed_spec(
    *,
    mul_man_bits: Sequence[int],
    mul_exp_bits: Sequence[int],
    mul_normal_binades: int | Sequence[int],
    mul_bias: int | Sequence[int],
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_man_bits: int | Sequence[int] | None = None,
    acc_exp_bits: int | Sequence[int] | None = None,
    acc_normal_binades: int | Sequence[int] | None = None,
    acc_bias: int | Sequence[int] | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_mixed`'s palette into a ``_GemmSpec``.

    That function documents the arguments. This one turns every palette
    argument into a list of the palette's length, with each entry's defaults
    filled in, and raises binary64's errors.
    """
    mul_mb_l, mul_eb_l, n = _palette_pair(
        mul_man_bits, mul_exp_bits, "mul_exp_bits", "superfp_matmul_mixed"
    )
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode

    mul_nb_l = _palette_list(mul_normal_binades, n, "mul_normal_binades")
    mul_bias_l = _palette_list(mul_bias, n, "mul_bias")

    if accumulate_quant:
        acc_mb_l = _palette_list_or(acc_man_bits, mul_mb_l, n, "acc_man_bits")
        acc_eb_l = _palette_list_or(acc_exp_bits, mul_eb_l, n, "acc_exp_bits")
        acc_nb_l = _palette_list_or(acc_normal_binades, mul_nb_l, n, "acc_normal_binades")
        acc_bias_l = _palette_list_or(acc_bias, mul_bias_l, n, "acc_bias")
    else:
        acc_mb_l = [0] * n
        acc_eb_l = [0] * n
        acc_nb_l = [0] * n
        acc_bias_l = [0] * n

    rounded = _palette_formats(
        _superfp_findings,
        zip(mul_mb_l, mul_eb_l, mul_nb_l, mul_bias_l, strict=True),
        (saturation_mode, mul_prng_bits),
    )
    stored: tuple[_Stored, ...] = ()
    if accumulate_quant:
        acc_widths = tuple(zip(acc_mb_l, acc_eb_l, acc_nb_l, acc_bias_l, strict=True))
        rounded += _palette_formats(
            _superfp_findings, acc_widths, (acc_saturation_mode, acc_prng_bits)
        )
        stored = _palette_formats(check_superfp_storage, acc_widths, (acc_saturation_mode,))

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_superfp_mixed.default,
        (
            mul_mb_l,
            mul_eb_l,
            mul_nb_l,
            mul_bias_l,
            mul_is_signed,
            accumulate_quant,
            acc_mb_l,
            acc_eb_l,
            acc_nb_l,
            acc_bias_l,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            acc_saturation_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        stored,
        mixed=True,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def superfp_matmul_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_man_bits: Sequence[int],
    mul_exp_bits: Sequence[int],
    mul_normal_binades: int | Sequence[int],
    mul_bias: int | Sequence[int],
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_man_bits: int | Sequence[int] | None = None,
    acc_exp_bits: int | Sequence[int] | None = None,
    acc_normal_binades: int | Sequence[int] | None = None,
    acc_bias: int | Sequence[int] | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM whose superfp format varies per output element.

    The superfp counterpart of :func:`binaryK_matmul_mixed`, which documents
    the palette, ``prec_idx`` and the carrier. ``mul_man_bits`` and
    ``mul_exp_bits`` are per-entry sequences, and their common length is the
    palette size ``n``. ``mul_normal_binades``, ``mul_bias`` and every
    ``acc_*`` may be a scalar or an ``n``-length sequence. An ``acc_*`` may
    also be ``None``, which falls back to the corresponding ``mul_*`` entry.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        prec_idx (Tensor): the palette index of each output element, in
            ``[0, n)``.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_man_bits (Sequence[int]): stored mantissa bits of each entry's
            multiply format.
        mul_exp_bits (Sequence[int]): exponent bits of each entry's multiply
            format.
        mul_normal_binades (int or Sequence[int]): the number of top binades
            of each entry's multiply format that have a mantissa.
        mul_bias (int or Sequence[int]): exponent bias of each entry's
            multiply format.
        mul_is_signed (bool): whether the multiply formats have a sign bit;
            shared by the palette. Default: ``True``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            multiply rounding; shared by the palette. Default: ``0``
        accumulate_quant (bool): whether the running sum is rounded to the
            accumulate format after every step. Default: ``True``
        acc_man_bits (int or Sequence[int], optional): the accumulate
            formats' mantissa bits. Default: ``None``, the ``mul_man_bits``
            entry.
        acc_exp_bits (int or Sequence[int], optional): the accumulate
            formats' exponent bits. Default: ``None``, the ``mul_exp_bits``
            entry.
        acc_normal_binades (int or Sequence[int], optional): the accumulate
            formats' normal binades. Default: ``None``, the
            ``mul_normal_binades`` entry.
        acc_bias (int or Sequence[int], optional): the accumulate formats'
            exponent bias. Default: ``None``, the ``mul_bias`` entry.
        acc_is_signed (bool, optional): whether the accumulate formats have a
            sign bit. Default: ``None``, ``mul_is_signed``.
        acc_prng_bits (int): random bits stochastic rounding draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed; the palette ops implement ``NAIVE`` only.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of every format.
            Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply formats' overflow
            behavior. Default: ``SaturationMode.SAT_NONE``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            formats' overflow behavior. Default: ``None``, ``saturation_mode``.
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`, applied
            to every palette entry: ``RoundMode.RO`` runs the palette's
            round-to-odd op, and each entry's ``P`` (``P + prng_bits`` under
            the stochastic modes) is checked against the limits.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if the palette is empty or a palette argument's length is
            not ``n``, if a carrier cannot hold an entry, if an entry the
            result holds has more precision than a narrower storage dtype, if
            ``carrier`` names no carrier or is narrower than the operands, if
            ``carrier_rounding`` is neither ``RNE`` nor ``RO`` or an entry
            exceeds the round-to-odd limits, or if the operands' leading
            dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``n`` is above 8, ``prec_idx`` has the wrong shape or
            an entry outside ``[0, n)``, ``accumulate_algorithm`` is not
            ``NAIVE`` (a ``ValueError`` under ``carrier_rounding=RO``), the
            inner dimensions do not match, the operands differ in dtype or
            device, or an operand requires grad under grad mode.

    Warns:
        FormatRangeWarning: if an entry's range exceeds the carrier's, or an
            accumulate entry has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> prec_idx = torch.tensor([[0, 0], [0, 1]])
        >>> superfp_matmul_mixed(
        ...     a, a, prec_idx, mul_man_bits=[3, 2], mul_exp_bits=[4, 4],
        ...     mul_normal_binades=8, mul_bias=7,
        ... )
        tensor([[ 7., 10.],
                [15., 24.]])
    """
    build = _superfp_mixed_spec
    if carrier_rounding is not _RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        build = _superfp_mixed_odd_spec
    return _gemm_mixed_nd(
        build(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            acc_saturation_mode=acc_saturation_mode,
            carrier=carrier,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


def _binaryK_fma_mixed_spec(
    *,
    fma_K: Sequence[int],
    fma_P: Sequence[int],
    fma_bias: int | Sequence[int] | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_fma_mixed`'s palette into a ``_GemmSpec``.

    That function documents the arguments. ``fma_quant=False`` is passed on,
    for the op to reject.
    """
    fma_K_l, fma_P_l, _ = _palette_pair(fma_K, fma_P, "fma_P", "binaryK_matmul_fma_mixed")
    fma_bias_l = _binaryK_palette_bias(fma_bias, fma_K_l, fma_P_l, fma_is_signed, "fma_bias")
    rounded: tuple[_Format, ...] = ()
    stored: tuple[_Stored, ...] = ()
    if fma_quant:
        widths = tuple(zip(fma_K_l, fma_P_l, fma_bias_l, strict=True))
        shared = (fma_is_signed, saturation_mode, subnormals_mode)
        rounded = _palette_formats(_binaryK_findings, widths, (*shared, fma_prng_bits))
        stored = _palette_formats(check_binaryK_storage, widths, shared)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_binaryK_fma_mixed.default,
        (
            fma_quant,
            fma_K_l,
            fma_P_l,
            fma_bias_l,
            fma_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            fma_prng_bits,
        ),
        stored,
        mixed=True,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def binaryK_matmul_fma_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_K: Sequence[int],
    fma_P: Sequence[int],
    fma_bias: int | Sequence[int] | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized fused-multiply-add GEMM whose binaryK format varies per output element.

    The palette counterpart of :func:`binaryK_matmul_fma`. Each step of a dot
    product is one fused multiply-add, rounded once, and the binaryK format of
    that rounding is chosen for each output element by ``prec_idx``, from a
    palette of up to 8 formats. ``prec_idx`` is as for
    :func:`binaryK_matmul_mixed`.

    ``fma_K`` and ``fma_P`` are per-entry sequences, and their common length
    is the palette size ``n``. ``fma_bias`` may be a scalar, an ``n``-length
    sequence, or ``None`` for P3109's bias per entry.

    ``fma_quant=False`` is rejected. An unrounded fused step has no format,
    so a palette would leave ``prec_idx`` nothing to choose. Use
    :func:`binaryK_matmul_fma` for the unquantized fused step.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        prec_idx (Tensor): the palette index of each output element, in
            ``[0, n)``.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        fma_K (Sequence[int]): width in bits of each entry's format.
        fma_P (Sequence[int]): precision of each entry's format.
        fma_bias (int or Sequence[int], optional): exponent bias of each
            entry's format. Default: ``None``, P3109's per entry.
        fma_is_signed (bool): whether the formats have a sign bit; shared by
            the palette. Default: ``True``
        fma_quant (bool): must be ``True``. Default: ``True``
        fma_prng_bits (int): random bits stochastic rounding draws for the fused
            rounding; shared by the palette. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the fused steps are
            summed; the palette ops implement ``NAIVE`` only. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of every entry.
            Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): the formats' overflow behavior.
            Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the formats have below their
            smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`, applied
            to every palette entry: ``RoundMode.RO`` runs the palette's
            round-to-odd op, and each entry's ``P`` (``P + prng_bits`` under
            the stochastic modes) is checked against the limits.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if the palette is empty or a palette argument's length is
            not ``n``, if a carrier cannot hold an entry, if an entry the
            result holds has more precision than a narrower storage dtype, if
            ``carrier`` names no carrier or is narrower than the operands, if
            ``carrier_rounding`` is neither ``RNE`` nor ``RO`` or an entry
            exceeds the round-to-odd limits, or if the operands' leading
            dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``fma_quant`` is false or ``accumulate_algorithm``
            is not ``NAIVE`` (each a ``ValueError`` under
            ``carrier_rounding=RO``), ``n`` is above 8, ``prec_idx`` has the
            wrong shape or an entry outside ``[0, n)``, the inner dimensions do
            not match, the operands differ in dtype or device, or an operand
            requires grad under grad mode.

    Warns:
        FormatRangeWarning: if an entry's range exceeds the carrier's, or an
            entry has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> prec_idx = torch.tensor([[0, 0], [0, 1]])
        >>> binaryK_matmul_fma_mixed(a, a, prec_idx, fma_K=[8, 8], fma_P=[4, 3])
        tensor([[ 7., 10.],
                [15., 24.]])
    """
    build = _binaryK_fma_mixed_spec
    if carrier_rounding is not _RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        build = _binaryK_fma_mixed_odd_spec
    return _gemm_mixed_nd(
        build(
            fma_K=fma_K,
            fma_P=fma_P,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            carrier=carrier,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


def _superfp_fma_mixed_spec(
    *,
    fma_man_bits: Sequence[int],
    fma_exp_bits: Sequence[int],
    fma_normal_binades: int | Sequence[int],
    fma_bias: int | Sequence[int],
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_fma_mixed`'s palette into a ``_GemmSpec``.

    That function documents the arguments. ``fma_quant=False`` is passed on,
    for the op to reject.
    """
    fma_mb_l, fma_eb_l, n = _palette_pair(
        fma_man_bits, fma_exp_bits, "fma_exp_bits", "superfp_matmul_fma_mixed"
    )
    fma_nb_l = _palette_list(fma_normal_binades, n, "fma_normal_binades")
    fma_bias_l = _palette_list(fma_bias, n, "fma_bias")
    rounded: tuple[_Format, ...] = ()
    stored: tuple[_Stored, ...] = ()
    if fma_quant:
        widths = tuple(zip(fma_mb_l, fma_eb_l, fma_nb_l, fma_bias_l, strict=True))
        rounded = _palette_formats(_superfp_findings, widths, (saturation_mode, fma_prng_bits))
        stored = _palette_formats(check_superfp_storage, widths, (saturation_mode,))

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_superfp_fma_mixed.default,
        (
            fma_quant,
            fma_mb_l,
            fma_eb_l,
            fma_nb_l,
            fma_bias_l,
            fma_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            fma_prng_bits,
        ),
        stored,
        mixed=True,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def superfp_matmul_fma_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_man_bits: Sequence[int],
    fma_exp_bits: Sequence[int],
    fma_normal_binades: int | Sequence[int],
    fma_bias: int | Sequence[int],
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized fused-multiply-add GEMM whose superfp format varies per output element.

    The superfp counterpart of :func:`binaryK_matmul_fma_mixed`, which
    documents the palette, ``prec_idx`` and the carrier. ``fma_quant=False``
    is rejected here too, because an unrounded fused step has no format for
    ``prec_idx`` to choose. ``fma_man_bits`` and ``fma_exp_bits`` are
    per-entry sequences, and their common length is the palette size ``n``.
    ``fma_normal_binades`` and ``fma_bias`` may each be a scalar or an
    ``n``-length sequence.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        prec_idx (Tensor): the palette index of each output element, in
            ``[0, n)``.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        fma_man_bits (Sequence[int]): stored mantissa bits of each entry's
            format.
        fma_exp_bits (Sequence[int]): exponent bits of each entry's format.
        fma_normal_binades (int or Sequence[int]): the number of top binades
            of each entry's format that have a mantissa.
        fma_bias (int or Sequence[int]): exponent bias of each entry's
            format.
        fma_is_signed (bool): whether the formats have a sign bit; shared by
            the palette. Default: ``True``
        fma_quant (bool): must be ``True``. Default: ``True``
        fma_prng_bits (int): random bits stochastic rounding draws for the fused
            rounding; shared by the palette. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the fused steps are
            summed; the palette ops implement ``NAIVE`` only. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of every entry.
            Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): the formats' overflow behavior.
            Default: ``SaturationMode.SAT_NONE``
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`, applied
            to every palette entry: ``RoundMode.RO`` runs the palette's
            round-to-odd op, and each entry's ``P`` (``P + prng_bits`` under
            the stochastic modes) is checked against the limits.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if the palette is empty or a palette argument's length is
            not ``n``, if a carrier cannot hold an entry, if an entry the
            result holds has more precision than a narrower storage dtype, if
            ``carrier`` names no carrier or is narrower than the operands, if
            ``carrier_rounding`` is neither ``RNE`` nor ``RO`` or an entry
            exceeds the round-to-odd limits, or if the operands' leading
            dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``fma_quant`` is false or ``accumulate_algorithm``
            is not ``NAIVE`` (each a ``ValueError`` under
            ``carrier_rounding=RO``), ``n`` is above 8, ``prec_idx`` has the
            wrong shape or an entry outside ``[0, n)``, the inner dimensions do
            not match, the operands differ in dtype or device, or an operand
            requires grad under grad mode.

    Warns:
        FormatRangeWarning: if an entry's range exceeds the carrier's, or an
            entry has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> prec_idx = torch.tensor([[0, 0], [0, 1]])
        >>> superfp_matmul_fma_mixed(
        ...     a, a, prec_idx, fma_man_bits=[3, 2], fma_exp_bits=[4, 4],
        ...     fma_normal_binades=8, fma_bias=7,
        ... )
        tensor([[ 7., 10.],
                [15., 24.]])
    """
    build = _superfp_fma_mixed_spec
    if carrier_rounding is not _RNE:
        _checked_carrier_rounding_call(carrier_rounding)
        build = _superfp_fma_mixed_odd_spec
    return _gemm_mixed_nd(
        build(
            fma_man_bits=fma_man_bits,
            fma_exp_bits=fma_exp_bits,
            fma_normal_binades=fma_normal_binades,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            carrier=carrier,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


# --- split multiply-accumulates across the two families ------------------------
#
# A binaryK multiply with a superfp accumulate, or the reverse
# (`csrc/common/gemm_cross.h`), runs in two NAIVE ops and their
# `*_accumulated` counterparts, laid out like the single-family split ops, so
# that a NAIVE cross-family call costs what a single-family one does. Their
# arguments are the single-family split op's, with the accumulate format given
# in its own family and no `accumulate_quant`: a sum left in the carrier has
# only one family, and is that family's op with `accumulate_quant=False`. So
# the accumulate format does not fall back to the multiply format's fields.
# Its sign and saturation do default to the multiply format's, and a binaryK
# accumulate format under a superfp multiply, which has no subnormals mode to
# pass on, defaults to `SUBNORMALS`.


def _cross_spec(
    op: Callable[..., torch.Tensor],
    args: tuple[Any, ...],
    rounded: tuple[_Format, _Format],
    stored: tuple[_Stored, ...],
    carrier: torch.dtype | None,
) -> _GemmSpec:
    """Build a cross-family op's spec, collecting the findings on its two formats."""
    return _GemmSpec(
        op,
        args,
        stored,
        carrier=_checked_carrier(carrier),
        findings=_format_findings(rounded),
    )


def _binaryK_superfp_spec(
    *,
    mul_K: int,
    mul_P: int,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    acc_man_bits: int,
    acc_exp_bits: int,
    acc_normal_binades: int,
    acc_bias: int,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_superfp_matmul`'s NAIVE formats into a ``_GemmSpec``."""
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if mul_bias is None:
        mul_bias = _binaryK_bias(mul_K, mul_P, mul_is_signed)
    mul = (mul_K, mul_P, mul_bias, mul_is_signed, saturation_mode, subnormals_mode)
    acc = (acc_man_bits, acc_exp_bits, acc_normal_binades, acc_bias, acc_saturation_mode)
    return _cross_spec(
        torch.ops.mptorch.custom_matmul_binaryK_superfp.default,
        (
            mul_K,
            mul_P,
            mul_bias,
            mul_is_signed,
            acc_man_bits,
            acc_exp_bits,
            acc_normal_binades,
            acc_bias,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            acc_saturation_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        ((_binaryK_findings, (*mul, mul_prng_bits)), (_superfp_findings, (*acc, acc_prng_bits))),
        ((check_superfp_storage, acc),),
        carrier,
    )


def _superfp_binaryK_spec(
    *,
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    acc_K: int,
    acc_P: int,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_binaryK_matmul`'s NAIVE formats into a ``_GemmSpec``."""
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if acc_bias is None:
        acc_bias = _binaryK_bias(acc_K, acc_P, acc_is_signed)
    mul = (mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, saturation_mode)
    acc = (acc_K, acc_P, acc_bias, acc_is_signed, acc_saturation_mode, acc_subnormals_mode)
    return _cross_spec(
        torch.ops.mptorch.custom_matmul_superfp_binaryK.default,
        (
            mul_man_bits,
            mul_exp_bits,
            mul_normal_binades,
            mul_bias,
            mul_is_signed,
            acc_K,
            acc_P,
            acc_bias,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            acc_saturation_mode.value,
            acc_subnormals_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        ((_superfp_findings, (*mul, mul_prng_bits)), (_binaryK_findings, (*acc, acc_prng_bits))),
        ((check_binaryK_storage, acc),),
        carrier,
    )


def _binaryK_superfp_accumulated_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    block_size: int | None = None,
    outer_man_bits: int | None = None,
    outer_exp_bits: int | None = None,
    outer_normal_binades: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_binaryK_superfp_spec` for any accumulate algorithm; the outer
    format is a superfp, the accumulate format's family."""
    spec = _binaryK_superfp_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    acc_is_signed, acc_saturation = spec.args[8], spec.args[13]
    accumulation = _superfp_accumulation(
        accumulate_algorithm,
        block_size,
        outer_man_bits,
        outer_exp_bits,
        outer_normal_binades,
        outer_bias,
        acc_is_signed if outer_is_signed is None else outer_is_signed,
        SaturationMode(acc_saturation) if outer_saturation_mode is None else outer_saturation_mode,
        outer_prng_bits,
        has_product=True,
    )
    if accumulation is None:
        return spec
    return _accumulated(
        spec,
        torch.ops.mptorch.custom_matmul_binaryK_superfp_accumulated.default,
        accumulation,
    )


def _superfp_binaryK_accumulated_spec(
    *,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    block_size: int | None = None,
    outer_K: int | None = None,
    outer_P: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    outer_subnormals_mode: SubnormalsMode | None = None,
    **formats: Any,
) -> _GemmSpec:
    """:func:`_superfp_binaryK_spec` for any accumulate algorithm; the outer
    format is a binaryK, the accumulate format's family."""
    spec = _superfp_binaryK_spec(accumulate_algorithm=accumulate_algorithm, **formats)
    acc_is_signed, acc_saturation, acc_subnormals = spec.args[8], spec.args[12], spec.args[13]
    accumulation = _binaryK_accumulation(
        accumulate_algorithm,
        block_size,
        outer_K,
        outer_P,
        outer_bias,
        acc_is_signed if outer_is_signed is None else outer_is_signed,
        SaturationMode(acc_saturation) if outer_saturation_mode is None else outer_saturation_mode,
        SubnormalsMode(acc_subnormals) if outer_subnormals_mode is None else outer_subnormals_mode,
        outer_prng_bits,
        has_product=True,
    )
    if accumulation is None:
        return spec
    return _accumulated(
        spec,
        torch.ops.mptorch.custom_matmul_superfp_binaryK_accumulated.default,
        accumulation,
    )


def binaryK_superfp_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_K: int,
    mul_P: int,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    acc_man_bits: int,
    acc_exp_bits: int,
    acc_normal_binades: int,
    acc_bias: int,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    block_size: int | None = None,
    outer_man_bits: int | None = None,
    outer_exp_bits: int | None = None,
    outer_normal_binades: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM whose products are binaryK and whose sums are superfp.

    A split multiply-accumulate across the two families: each step computes
    ``s = round_acc(s + round_mul(a * b))`` in the carrier, as in
    :func:`binaryK_matmul`, with the product rounded to the binaryK format
    ``mul_K``/``mul_P`` and the running sum to the superfp format given by the
    four ``acc_*`` fields. The accumulate format is required: a sum left in
    the carrier is :func:`binaryK_matmul` with ``accumulate_quant=False``.
    :func:`binaryK_matmul` documents what the two have in common
    (``torch.matmul``'s operand rules, the transpose flags, the carrier, the
    storage check, resolving the formats on every call), and
    :func:`superfp_quantize` describes the accumulate format. The result
    holds the accumulate format's values or, under ``BLOCK`` or ``TREE`` with
    an outer format, the outer format's, which belongs to the accumulate
    format's family.

    :class:`mptorch.quant.SplitMac` with a :class:`mptorch.BinaryK` multiply
    and a :class:`mptorch.SuperFP` accumulate resolves to this op.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_K (int): width in bits of the binaryK multiply format.
        mul_P (int): precision of the multiply format.
        mul_bias (int, optional): exponent bias of the multiply format.
            Default: ``None``, P3109's ``2**(K - P - 1)`` if signed and
            ``2**(K - P)`` if unsigned.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            multiply rounding. Default: ``0``
        acc_man_bits (int): stored mantissa bits of the superfp accumulate
            format.
        acc_exp_bits (int): exponent bits of the accumulate format.
        acc_normal_binades (int): the number of top binades of the
            accumulate format that have a mantissa.
        acc_bias (int): exponent bias of the accumulate format.
        acc_is_signed (bool, optional): whether the accumulate format has a
            sign bit. Default: ``None``, ``mul_is_signed``.
        acc_prng_bits (int): random bits stochastic rounding draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :func:`binaryK_matmul`. Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of both formats. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply format's overflow
            behavior. Default: ``SaturationMode.SAT_NONE``
        subnormals_mode (SubnormalsMode): what the multiply format has below
            its smallest normal number. Default: ``SubnormalsMode.SUBNORMALS``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            format's overflow behavior. Default: ``None``, ``saturation_mode``.
        block_size (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_man_bits (int, optional): mantissa bits of the superfp outer
            format, as for :func:`superfp_matmul`. The four outer fields are
            given together, or none of them. Default: ``None``
        outer_exp_bits (int, optional): exponent bits of the outer format.
            Default: ``None``
        outer_normal_binades (int, optional): normal binades of the outer
            format. Default: ``None``
        outer_bias (int, optional): exponent bias of the outer format.
            Default: ``None``
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, the accumulate format's.
        outer_prng_bits (int): random bits stochastic rounding draws for the
            outer rounding. Default: ``0``
        outer_saturation_mode (SaturationMode, optional): the outer format's
            overflow behavior. Default: ``None``, the accumulate format's.
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`.
            ``torch.float64`` widens narrower operands to binary64 and
            narrows the result back to their dtype with a single rounding.
            ``torch.float32`` refuses float64 operands. Default: ``None``,
            the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`, applied
            to both formats: ``RoundMode.RO`` runs the round-to-odd
            cross-family op (its ``*_accumulated_odd`` counterpart under
            ``KAHAN``, ``BLOCK`` and ``TREE``), and the multiply, accumulate
            and outer formats are each checked against the limits.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold a format, and in the other
            cases listed for :func:`binaryK_matmul`.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if a format's range exceeds the carrier's, or the
            accumulate format has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> binaryK_superfp_matmul(
        ...     a, a, mul_K=8, mul_P=4,
        ...     acc_man_bits=2, acc_exp_bits=4, acc_normal_binades=8, acc_bias=7,
        ... )
        tensor([[ 7., 10.],
                [16., 24.]])

    Every product is exact in Binary8p4. The sums 15 and 22 fall halfway
    between values of the superfp accumulate format, whose two mantissa bits
    space them 2 apart in [8, 16) and 4 apart in [16, 32), and round to the
    even mantissa: 16 and 24.
    """
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_man_bits is None
        and outer_exp_bits is None
        and outer_normal_binades is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _binaryK_superfp_spec(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            acc_saturation_mode=acc_saturation_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _binaryK_superfp_accumulated_spec(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
            acc_saturation_mode=acc_saturation_mode,
            carrier=carrier,
            block_size=block_size,
            outer_man_bits=outer_man_bits,
            outer_exp_bits=outer_exp_bits,
            outer_normal_binades=outer_normal_binades,
            outer_bias=outer_bias,
            outer_is_signed=outer_is_signed,
            outer_prng_bits=outer_prng_bits,
            outer_saturation_mode=outer_saturation_mode,
        )
    else:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_man_bits is not None
            or outer_exp_bits is not None
            or outer_normal_binades is not None
            or outer_bias is not None
        ):
            spec = _binaryK_superfp_accumulated_odd_spec(
                mul_K=mul_K,
                mul_P=mul_P,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                acc_man_bits=acc_man_bits,
                acc_exp_bits=acc_exp_bits,
                acc_normal_binades=acc_normal_binades,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
                acc_saturation_mode=acc_saturation_mode,
                carrier=carrier,
                block_size=block_size,
                outer_man_bits=outer_man_bits,
                outer_exp_bits=outer_exp_bits,
                outer_normal_binades=outer_normal_binades,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
            )
        else:
            spec = _binaryK_superfp_odd_spec(
                mul_K=mul_K,
                mul_P=mul_P,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                acc_man_bits=acc_man_bits,
                acc_exp_bits=acc_exp_bits,
                acc_normal_binades=acc_normal_binades,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                subnormals_mode=subnormals_mode,
                acc_saturation_mode=acc_saturation_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


def superfp_binaryK_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    acc_K: int,
    acc_P: int,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.SAT_NONE,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    block_size: int | None = None,
    outer_K: int | None = None,
    outer_P: int | None = None,
    outer_bias: int | None = None,
    outer_is_signed: bool | None = None,
    outer_prng_bits: int = 0,
    outer_saturation_mode: SaturationMode | None = None,
    outer_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    """Quantized GEMM whose products are superfp and whose sums are binaryK.

    :func:`binaryK_superfp_matmul` with the families swapped: each product is
    rounded to the superfp format given by the four ``mul_*`` fields, and the
    running sum to the binaryK format ``acc_K``/``acc_P``, which is required
    (a sum left in the carrier is :func:`superfp_matmul` with
    ``accumulate_quant=False``). Under ``BLOCK`` or ``TREE`` the outer format
    is a binaryK, as for :func:`binaryK_matmul`.
    :class:`mptorch.quant.SplitMac` with a :class:`mptorch.SuperFP` multiply
    and a :class:`mptorch.BinaryK` accumulate resolves to this op.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_man_bits (int): stored mantissa bits of the superfp multiply
            format.
        mul_exp_bits (int): exponent bits of the multiply format.
        mul_normal_binades (int): the number of top binades of the multiply
            format that have a mantissa.
        mul_bias (int): exponent bias of the multiply format.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            multiply rounding. Default: ``0``
        acc_K (int): width in bits of the binaryK accumulate format.
        acc_P (int): precision of the accumulate format.
        acc_bias (int, optional): exponent bias of the accumulate format.
            Default: ``None``, P3109's for ``acc_K``/``acc_P``.
        acc_is_signed (bool, optional): whether the accumulate format has a
            sign bit. Default: ``None``, ``mul_is_signed``.
        acc_prng_bits (int): random bits stochastic rounding draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :func:`binaryK_matmul`. Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding mode of both formats. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply format's overflow
            behavior. Default: ``SaturationMode.SAT_NONE``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            format's overflow behavior. Default: ``None``, ``saturation_mode``.
        acc_subnormals_mode (SubnormalsMode): what the accumulate format has
            below its smallest normal number. It does not default to the
            multiply format's, since a superfp has no subnormals mode.
            Default: ``SubnormalsMode.SUBNORMALS``
        block_size (int, optional): as for :func:`binaryK_matmul`.
            Default: ``None``
        outer_K (int, optional): width of the binaryK outer format, as
            :func:`binaryK_matmul`'s. Default: ``None``
        outer_P (int, optional): precision of the outer format; given
            together with ``outer_K``. Default: ``None``
        outer_bias (int, optional): exponent bias of the outer format.
            Default: ``None``, P3109's for ``outer_K``/``outer_P``.
        outer_is_signed (bool, optional): whether the outer format has a sign
            bit. Default: ``None``, the accumulate format's.
        outer_prng_bits (int): random bits stochastic rounding draws for the
            outer rounding. Default: ``0``
        outer_saturation_mode (SaturationMode, optional): the outer format's
            overflow behavior. Default: ``None``, the accumulate format's.
        outer_subnormals_mode (SubnormalsMode, optional): what the outer
            format has below its smallest normal number. Default: ``None``,
            the accumulate format's.
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`.
            ``torch.float64`` widens narrower operands to binary64 and
            narrows the result back to their dtype with a single rounding.
            ``torch.float32`` refuses float64 operands. Default: ``None``,
            the operands' own carrier.
        carrier_rounding (RoundMode): as for :func:`binaryK_matmul`, applied
            to both formats: ``RoundMode.RO`` runs the round-to-odd
            cross-family op (its ``*_accumulated_odd`` counterpart under
            ``KAHAN``, ``BLOCK`` and ``TREE``), and the multiply, accumulate
            and outer formats are each checked against the limits.
            Default: ``RoundMode.RNE``

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold a format, and in the other
            cases listed for :func:`binaryK_matmul`.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if a format's range exceeds the carrier's, or the
            accumulate format has values a narrower storage dtype cannot represent.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> superfp_binaryK_matmul(
        ...     a, a, mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=8, mul_bias=7,
        ...     acc_K=8, acc_P=3,
        ... )
        tensor([[ 7., 10.],
                [16., 24.]])
    """
    if (
        accumulate_algorithm is AccumulateAlgorithm.NAIVE
        and block_size is None
        and outer_K is None
        and outer_P is None
        and outer_bias is None
        and carrier_rounding is _RNE
    ):
        spec = _superfp_binaryK_spec(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
            carrier=carrier,
        )
    elif carrier_rounding is _RNE:
        spec = _superfp_binaryK_accumulated_spec(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
            carrier=carrier,
            block_size=block_size,
            outer_K=outer_K,
            outer_P=outer_P,
            outer_bias=outer_bias,
            outer_is_signed=outer_is_signed,
            outer_prng_bits=outer_prng_bits,
            outer_saturation_mode=outer_saturation_mode,
            outer_subnormals_mode=outer_subnormals_mode,
        )
    else:
        _checked_carrier_rounding_call(carrier_rounding)
        if (
            accumulate_algorithm is not AccumulateAlgorithm.NAIVE
            or block_size is not None
            or outer_K is not None
            or outer_P is not None
            or outer_bias is not None
        ):
            spec = _superfp_binaryK_accumulated_odd_spec(
                mul_man_bits=mul_man_bits,
                mul_exp_bits=mul_exp_bits,
                mul_normal_binades=mul_normal_binades,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                acc_K=acc_K,
                acc_P=acc_P,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                acc_saturation_mode=acc_saturation_mode,
                acc_subnormals_mode=acc_subnormals_mode,
                carrier=carrier,
                block_size=block_size,
                outer_K=outer_K,
                outer_P=outer_P,
                outer_bias=outer_bias,
                outer_is_signed=outer_is_signed,
                outer_prng_bits=outer_prng_bits,
                outer_saturation_mode=outer_saturation_mode,
                outer_subnormals_mode=outer_subnormals_mode,
            )
        else:
            spec = _superfp_binaryK_odd_spec(
                mul_man_bits=mul_man_bits,
                mul_exp_bits=mul_exp_bits,
                mul_normal_binades=mul_normal_binades,
                mul_bias=mul_bias,
                mul_is_signed=mul_is_signed,
                mul_prng_bits=mul_prng_bits,
                acc_K=acc_K,
                acc_P=acc_P,
                acc_bias=acc_bias,
                acc_is_signed=acc_is_signed,
                acc_prng_bits=acc_prng_bits,
                accumulate_algorithm=accumulate_algorithm,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                acc_saturation_mode=acc_saturation_mode,
                acc_subnormals_mode=acc_subnormals_mode,
                carrier=carrier,
            )
    return _gemm_nd(spec, a, b, trans_a, trans_b)
