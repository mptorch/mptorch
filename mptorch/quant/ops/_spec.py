"""Format defaults, the resolved GEMM and quantizer calls, and running them.

Every GEMM wrapper works out the same few things: a binaryK format's default
exponent bias, the accumulate format's fallback to the multiply format, and
which format's values the result holds. The palette (mixed) wrappers do the
same for each palette entry. The helpers here hold that logic, so each flat
wrapper is only a signature, a docstring, and a call that passes its
arguments in the op's order.

A palette op takes up to eight formats and a `prec_idx` map that chooses one
of them for each output element. Its builder turns every palette argument
into a list of the palette's length (repeating a scalar, and filling in each
entry's default for `None`) and removes duplicate formats before the carrier
and storage checks, so that a scalar repeated across the palette is checked
once.

A GEMM call with its formats resolved is a `_GemmSpec`, which
`mptorch.quant.gemm`'s factories build once per layer, and `_run_gemm` and
`_run_gemm_mixed` call its op on operands of rank 2 or 3.

An elementwise quantizer call with its format resolved is a `_QuantSpec`,
which `mptorch.quant.Quant` builds once, when it is made, and `_run_quant`
calls its op. A flat quantizer such as `binaryK_quantize` looks its format's
findings up on every call instead; the two must give the same results, the
same warnings and the same errors, which `tests/test_quant_resolved.py`
checks.
"""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import torch

from mptorch.number.base import _report_per_call

from ._checks import (
    _BELOW_BINARY64,
    _CARRIER,
    _NARROW_STORAGE,
    _call_carrier,
    _check_stored,
    _Findings,
    _narrowed,
    _quantizer_operand,
    _refuse_widening_in_place,
    _report_findings,
    _Stored,
)


def _binaryK_bias(K: int, P: int, is_signed: bool) -> int:
    """Return binaryK's default exponent bias, as IEEE P3109 defines it.

    It is ``2**(K - P - 1)`` for a signed format and ``2**(K - P)`` for an
    unsigned one. The exponent field is ``K - P`` bits wide (one bit wider
    without a sign bit), and this bias puts 1.0 at the middle code of the
    format; it is one more than the IEEE 754 bias for the same field width.
    """
    return 2 ** (K - P - 1) if is_signed else 2 ** (K - P)


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
    # The operand dtypes whose result the op computes into float64 instead:
    # the fixed-point GEMM's, when its accumulate format has more significant
    # bits than the dtype holds (`number.fixedpoint._fixedpoint_gemm_wide`).
    # A call keeps such a result in float64, a named carrier included, and a
    # layer rounds it once with its output rounding (`gemm._spec_matmul`).
    # `wide_label` names the accumulate format, for the refusal.
    wide: frozenset[torch.dtype] = frozenset()
    wide_label: str = ""


_WIDE_PRECISION = {torch.float32: 24, torch.float16: 11, torch.bfloat16: 8}


def _wide_refusal(
    spec: _GemmSpec, dtype: torch.dtype, where: str, device: torch.device
) -> ValueError:
    """The error of a pass whose result would be wide: ``where`` is
    ``"forward"`` (a forward pass with no output rounding) or ``"gradient"``.
    On MPS, which has no float64, the only fixes are a narrower accumulate
    format and the CPU, as the op's own refusal says."""
    name = str(dtype).removeprefix("torch.")
    p = _WIDE_PRECISION[dtype]
    if device.type == "mps":
        return ValueError(
            f"{spec.wide_label}, more than {name}'s {p}, so its product is computed in float64, "
            f"which MPS does not have: use an accumulate format of at most {p} magnitude bits, "
            "or compute on the CPU"
        )
    if where == "forward":
        why = (
            f"this forward pass has no output rounding to store it in {name} with: give the "
            "layer an output_quant (GemmFormats(output_quant=...), or fixedpoint_gemm_formats("
            f"output_quant=...)), use float64 operands or an accumulate format of at most {p} "
            "magnitude bits, or call fixedpoint_matmul, which returns the float64 result"
        )
    else:
        why = (
            f"a gradient pass stores it in the gradient's dtype, {name}, with no rounding of its "
            "own: use float64 gradients (a float64 layer), or an accumulate format of at most "
            f"{p} magnitude bits for this pass"
        )
    return ValueError(
        f"{spec.wide_label}, more than {name}'s {p}, so its product is computed in float64, "
        f"and {why}"
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
    if narrow is None or narrow in spec.wide:
        return out
    return _narrowed(out, narrow)


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


# One finding on one format, as `(error, warning)`: what `mptorch.number`'s
# findings functions return, with `(None, None)` when there is nothing to say.
_Pair = tuple[str | None, str | None]


class _QuantSpec(NamedTuple):
    """One elementwise quantizer call's format, resolved into its op's arguments.

    ``op`` is the quantize op, out of place or in place (``in_place``), and
    ``args`` every op argument after the tensor, in the op's order, with the
    modes already converted to their integer values. ``name`` is the flat
    function the call stands for, which the in-place op's refusal of a
    widening carrier names.

    ``findings`` holds binary32's finding on the format, then binary64's, and
    ``stored`` the findings on storing a result in a dtype narrower than each
    of those carriers, keyed by that dtype. A finding with nothing to say is
    ``None`` in ``findings`` and absent from ``stored``, so a call on a format
    both carriers hold pays two tests and a dictionary lookup. They are found
    once, when the spec is built, from the memoized functions that the flat
    quantizers' ``check_*_carrier`` and ``check_*_storage`` calls read on
    every call, and they are raised or issued on each call, for the carrier
    and the dtype of that call.

    The last three fields serve a pointwise op (``mptorch.quant.pointwise``),
    which rounds a result it has just computed (``_round_result``):
    ``plain_op`` and ``fresh_op`` are the out-of-place and in-place ops,
    whatever ``in_place`` says (the in-place op returns nothing), and
    ``result_stored`` holds the findings on storing that result in a narrower
    dtype, under the rule for a GEMM's result (``elementwise=False``): a
    computed result reaches the format's whole value set, where a quantizer's
    input is already a value of its dtype.
    """

    op: Callable[..., Any]
    args: tuple[Any, ...]
    name: str
    in_place: bool
    carrier: torch.dtype | None
    findings: tuple[_Pair | None, _Pair | None]
    stored: tuple[dict[torch.dtype, _Pair], dict[torch.dtype, _Pair]]
    plain_op: Callable[..., torch.Tensor]
    fresh_op: Callable[..., None]
    result_stored: tuple[dict[torch.dtype, _Pair], dict[torch.dtype, _Pair]]


def _said(pair: _Pair) -> _Pair | None:
    """``pair``, or ``None`` when it holds neither an error nor a warning."""
    return None if pair[0] is None and pair[1] is None else pair


def _quant_spec(
    ops: tuple[Callable[..., torch.Tensor], Callable[..., None]],
    args: tuple[Any, ...],
    name: str,
    in_place: bool,
    carrier: torch.dtype | None,
    finding: Callable[[torch.dtype], _Pair],
    storage: Callable[[torch.dtype, bool], _Pair],
) -> _QuantSpec:
    """Build a ``_QuantSpec``, finding the format's verdicts once.

    ``ops`` is the out-of-place op and the in-place one, which writes its
    argument and returns nothing. ``finding(carrier)`` is the format's finding
    in a carrier dtype, and ``storage(dtype, elementwise)`` its finding on
    storing a result in ``dtype``, under the quantizer's rule
    (``elementwise=True``) or a computed result's; each family's builder
    passes its own two. Every carrier and every storage dtype narrower than it
    is looked up, since a call's dtype is not known yet.
    """
    findings = (_said(finding(_CARRIER[0])), _said(finding(_CARRIER[1])))

    def stored(elementwise: bool) -> tuple[dict[torch.dtype, _Pair], dict[torch.dtype, _Pair]]:
        found = tuple(
            {
                dtype: pair
                for dtype in narrower
                if (pair := _said(storage(dtype, elementwise))) is not None
            }
            for narrower in (_NARROW_STORAGE, _BELOW_BINARY64)
        )
        return found[0], found[1]

    plain_op, fresh_op = ops
    op = fresh_op if in_place else plain_op
    return _QuantSpec(
        op, args, name, in_place, carrier, findings, stored(True), plain_op, fresh_op, stored(False)
    )


def _run_quant(spec: _QuantSpec, x: torch.Tensor) -> torch.Tensor:
    """Call a resolved quantizer on ``x``.

    It makes the flat quantizer's checks in the flat quantizer's order: the
    carrier the call rounds in, the format against that carrier, the format
    against a narrower storage dtype, and, for the in-place op, the refusal of
    a carrier wider than ``x``. Each warning names the caller's line, as the
    flat quantizer's does.
    """
    dtype = x.dtype
    if spec.carrier is None:
        wide, widen = dtype is torch.float64, False
    else:
        wide, widen = _call_carrier(spec.carrier, dtype)
    found = spec.findings[wide]
    if found is not None:
        _report_per_call(*found)
    stored = spec.stored[wide].get(dtype)
    if stored is not None:
        _report_per_call(*stored)
    if spec.in_place:
        if widen:
            _refuse_widening_in_place(spec.name, dtype)
        spec.op(x, *spec.args)
        return x
    out = spec.op(_quantizer_operand(x, widen), *spec.args)
    return _narrowed(out, dtype) if widen else out


def _round_result(
    q: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Round ``t``, a result a pointwise op has just computed in its carrier,
    with the slot ``q``, and store it in ``dtype``.

    ``t`` belongs to the op, so a ``Quant`` rounds it in place when it is
    contiguous, and out of place otherwise (a result keeps its operands'
    layout, which need not be contiguous). Its checks are ``_run_quant``'s,
    made against ``t``'s carrier, except that storing in ``dtype`` is checked
    under a computed result's rule (``_QuantSpec.result_stored``). Any other
    callable (a block format's ``Quant``, a ``ScaledQuant``, the caller's
    own) is called on ``t`` and makes its own checks. The result is stored in
    ``dtype``: from float32 with one conversion, from float64 by
    ``_narrowed``.
    """
    spec = getattr(q, "_spec", None)
    if spec is None:
        out = q(t)
    else:
        wide = t.dtype is torch.float64
        found = spec.findings[wide]
        if found is not None:
            _report_per_call(*found)
        stored = spec.result_stored[wide].get(dtype)
        if stored is not None:
            _report_per_call(*stored)
        if t.is_contiguous():
            spec.fresh_op(t, *spec.args)
            out = t
        else:
            out = spec.plain_op(t, *spec.args)
    return _stored(out, dtype)


def _stored(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``t``, computed in a carrier, stored in ``dtype``: unchanged when it is
    that dtype already, converted once from float32, and narrowed with a
    single rounding from float64 (``_narrowed``)."""
    if t.dtype is dtype:
        return t
    return _narrowed(t, dtype) if t.dtype is torch.float64 else t.to(dtype)
