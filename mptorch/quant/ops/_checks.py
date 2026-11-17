"""Each call's checks of its formats against the carrier and the storage dtype.

The carrier. Each call rounds in a carrier, the binary float the kernel
computes in: binary64 for float64 operands, binary32 for float32, float16 and
bfloat16, or the one `carrier=` names, which is never narrower than the
operands. `mptorch.number`'s findings say whether a carrier can hold a
simulated format: a range beyond the carrier's warns (the format still
quantizes correctly over the part that fits), and precision beyond it raises.
Only the call knows its operands' dtype, and so its carrier, so the format is
checked against the carrier here, on each call; `BinaryK` and `SuperFP` raise
only for what no carrier can do. The flat wrappers take a format's parameters
as plain integers and never build a format object, so they raise those errors
themselves when they resolve a format:
`binaryK_matmul(a, b, mul_K=60, mul_P=54)` raises what `BinaryK(60, 54)`
raises.

A GEMM's formats are fixed when its spec (`_GemmSpec`) is built, so both
carriers' findings are looked up then (`mptorch.number` caches them per
format), and a call only indexes a tuple. A quantizer resolves nothing before
the call and looks its format up on each call, which hits the same cache.

`carrier=torch.float64` on narrower operands makes the call the float64 call
on their values: the operands are widened, the binary64 kernel runs, and the
result is narrowed back to the operands' dtype with a single rounding
(`_narrowed`). This happens here in Python, not through an op argument, so
that it is bit-identical to widening the operands by hand and calling the op:
the same binary64 kernel on the same values, drawing the same random words.
Only the final narrowing differs from torch's own `.to()`, which can round
twice (always for bfloat16, and for float16 on some platforms). A carrier
narrower than the operands is refused, because rounding a float64 tensor in
binary32 would round every input once before the format does.

The storage dtype. When the operands are narrower than the carrier (float16
or bfloat16 under binary32; those and float32 under binary64), the result is
converted back to the operands' dtype, the storage dtype, when it is written,
which rounds it a second time. `mptorch.number`'s `check_*_storage` functions
decide whether the storage dtype can hold the simulated format's results.
Only the call knows the storage dtype and the carrier, so the check runs
here, on each call, and only when the storage dtype is narrower than the
carrier: a result stored in the carrier's own dtype keeps every value the
carrier computed. The format checked is the one whose values the result
holds: a GEMM's last rounding (its accumulate or fused format), whose sums can
take every value of that format, or an elementwise quantizer's format, whose
inputs are already values of the storage dtype, so that only the edges of the
format's range matter (`elementwise=True`). A GEMM's multiply format is never
stored, since its products stay in the carrier.

The storage dtype limits only the precision of the values written into it,
not `man_bits + prng_bits`: stochastic rounding draws its random bits in the
carrier, so the carrier check holds that sum against the carrier's 23 or 52
mantissa bits. Limiting it by the storage dtype's mantissa would describe a
kernel that rounds in the storage dtype, which these kernels do not.

The elementwise quantizers' two shared helpers, `_quantizer_operand` and
`_refuse_widening_in_place`, are here too, since both concern the carrier a
quantizer's tensor is rounded in.
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch

from mptorch.number.base import _report_per_call

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
