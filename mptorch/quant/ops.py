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
from mptorch.number import (
    _binaryK_findings,
    _report_per_call,
    _superfp_findings,
    check_binaryK_carrier,
    check_binaryK_storage,
    check_superfp_carrier,
    check_superfp_storage,
)

__all__ = [
    "binaryK_quantize",
    "binaryK_quantize_",
    "superfp_quantize",
    "superfp_quantize_",
    "binaryK_matmul",
    "superfp_matmul",
    "binaryK_matmul_fma",
    "superfp_matmul_fma",
    "binaryK_matmul_mixed",
    "superfp_matmul_mixed",
    "binaryK_matmul_fma_mixed",
    "superfp_matmul_fma_mixed",
]

# --- format defaulting, written once ----------------------------------------
#
# Every GEMM wrapper resolves the same few things: the exponent bias a binaryK
# format defaults to, the accumulate format's fallback to the multiply format,
# and which format's values the result holds. The palette (mixed) wrappers do
# the same per entry. That logic lives in the helpers below, so each of the
# eight wrappers is a signature, a docstring and the schema's argument order.


def _binaryK_bias(K: int, P: int, is_signed: bool) -> int:
    """binaryK's default exponent bias, as IEEE P3109 defines it.

    ``2**(K - P - 1)`` for a signed format and ``2**(K - P)`` for an unsigned
    one. The exponent field is ``K - P`` bits wide (one more without a sign
    bit), and this bias puts 1.0 at the middle code of that field, one more
    than the IEEE 754 bias for the same field width.
    """
    return 2 ** (K - P - 1) if is_signed else 2 ** (K - P)


# --- the format's range against its carrier's ----------------------------------
#
# A call rounds in a carrier: binary64 for float64 operands, binary32 for
# float32, float16 and bfloat16, or the one `carrier=` names, which is never
# narrower than the operands. `mptorch.number`'s findings say what a carrier
# makes of a format: a range past the carrier's warns (the format still
# quantizes correctly over the part that fits), a precision past it raises.
# Only the call knows the dtype, so the format is held to its carrier here,
# per call, while `BinaryK`/`SuperFP` raise only for what no carrier can do.
# The wrappers below take format parameters as plain integers and never see a
# format object, so they raise the same errors themselves when they resolve a
# format: `binaryK_matmul(a, b, mul_K=60, mul_P=54)` says what
# `BinaryK(60, 54)` says.
#
# A GEMM's formats are fixed when its spec is built, so both carriers'
# findings are looked up then (memoized per format in `mptorch.number`) and a
# call pays a tuple index. A quantizer resolves nothing ahead of the call and
# looks its format up per call, which hits the same memo.
#
# `carrier=torch.float64` on narrower operands is the float64 call on their
# values: the operands are widened, the binary64 kernel runs, and the result
# is narrowed back to the operands' dtype with one rounding (`_narrowed`).
# This is done here in Python rather than through a schema argument so that
# it is bit-identical to calling the op on operands widened by hand: the same
# binary64 instantiation on the same values, drawing the same random words,
# up to that last conversion, which torch's own `.to()` performs in two
# roundings for float16 and bfloat16. A carrier narrower than the operands is
# refused, because rounding a float64 tensor in binary32 would round every
# input once before the format does.

# What one carrier says about one GEMM's formats: (error, warning) pairs, one
# of the two set in each, errors first. Empty when the carrier holds them all.
_Findings = tuple[tuple[str | None, str | None], ...]

# `(function, format arguments)`: one format an op rounds with or stores,
# called as `function(*arguments, ...)`. Plain values rather than a `partial`,
# so two spellings of the same GEMM compare equal; `tests/test_number_formats.py`
# asserts that `mac.py`'s vocabulary and the flat wrappers build equal specs.
_Format = tuple[Callable[..., Any], tuple[Any, ...]]


def _checked_carrier(carrier: torch.dtype | None) -> torch.dtype | None:
    """The ``carrier`` argument, validated.

    Returns it unchanged when it is ``None``, ``torch.float32`` or
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
    """What a call on ``dtype`` operands does about its carrier.

    Returns ``(wide, widen)``: whether the call rounds in binary64, and
    whether reaching that carrier means widening the operands to float64
    first. With ``carrier=None`` a float64 tensor rounds in binary64 and every
    other dtype in binary32, and nothing is widened. ``torch.float32`` on
    float64 operands raises, since a carrier is never narrower than its
    operands.
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
    """A float64 result stored in ``dtype``, rounded to nearest-even once.

    float64 to float32 is a single conversion and ``.float()`` is it. To
    float16 and bfloat16 torch converts through float32, which rounds twice:
    a value a hair above a tie on the narrow grid first lands on the tie and
    then goes to even, so 65519.999999 comes back as float16 infinity where a
    single rounding gives 65504. The ``narrow_float64`` op rounds the float64
    word onto the dtype's grid directly, by integer arithmetic on its bits
    (``csrc/common/narrow_binary64.h``), at about the cost of ``.to()``.
    """
    if dtype is torch.float32:
        return x.float()
    return torch.ops.mptorch.narrow_float64.default(x, dtype)


# What a findings function returns for a format its carrier holds entirely,
# and the spec's findings when every format is such.
_NOTHING = (None, None)
_NO_FINDINGS: tuple[_Findings, _Findings] = ((), ())


def _format_findings(formats: Iterable[_Format]) -> tuple[_Findings, _Findings]:
    """What binary32 and binary64 each say about every format a GEMM rounds with.

    Returns ``(binary32's findings, binary64's findings)``, each deduplicated
    with the errors first. They are found once, when the spec is built, and
    reported by each call against the carrier it rounds in. binary64's errors
    are what no carrier can do, so they raise here, which is the plain-integer
    spelling of what ``BinaryK``/``SuperFP`` raise when built.

    A flat wrapper builds its spec on every call, so the common case, a format
    both carriers hold, is kept cheap: binary64 is wider than binary32 at
    every edge, so only a format binary32 finds something in is looked up
    against binary64, and the deduplication (a palette repeats a scalar across
    its entries, a ``SplitMac`` often rounds both halves in one format) runs
    only when there is something to deduplicate.
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
    """``found`` deduplicated in first-seen order, the errors ahead of the warnings."""
    unique = dict.fromkeys(found)
    return tuple([f for f in unique if f[0] is not None] + [f for f in unique if f[0] is None])


def _report_findings(found: _Findings) -> None:
    """Raise ``found``'s first error, or warn once per warning, at the caller's line."""
    for error, warning in found:
        _report_per_call(error, warning)


# --- the format's values against a result narrower than its carrier ----------
#
# A tensor narrower than its carrier (float16 or bfloat16 in binary32, and
# float32 too in binary64) has its result converted back when it is written,
# which rounds it a second time. `mptorch.number`'s `check_*_storage` are that
# rule. Only the call knows the dtype and the carrier, so it runs here, per
# call, and only for the dtypes it has something to say about: a result in the
# carrier's own dtype holds every value of it. The format held against the
# dtype is the one whose values the result actually holds: a GEMM's last
# rounding (the accumulate or fused format), whose sums reach every value the
# format has, or the elementwise quantizer's own, whose inputs are already the
# dtype's values and so reach only the edges of its range, which is what
# `elementwise=True` tells the check. A GEMM's multiply format never reaches
# storage, since its products are intermediates in the carrier.
#
# The storage dtype bounds only the precision of the values written into it,
# not `man_bits + prng_bits`: stochastic rounding draws its bits in the
# carrier, so that sum is the carrier check's to hold against its 23 or 52
# mantissa bits. Bounding it by the storage dtype's mantissa instead would
# describe a kernel that rounds in the storage dtype, which these do not.

# The dtypes stored a second time: binary32's results in the first set,
# binary64's in the second.
_NARROW_STORAGE = frozenset((torch.float16, torch.bfloat16))
_BELOW_BINARY64 = frozenset((torch.float32, torch.float16, torch.bfloat16))

# The carrier as `mptorch.number` keys it, indexed by `_call_carrier`'s first answer.
_CARRIER = (torch.float32, torch.float64)

# The storage check for one format a result may hold, called as
# `check(*arguments, storage=dtype)`.
_Stored = _Format


def _palette_formats(
    fn: Callable[..., Any], widths: Iterable[tuple[Any, ...]], shared: tuple[Any, ...]
) -> tuple[_Format, ...]:
    """One ``_Format`` per distinct palette entry.

    Each entry's arguments are its per-entry widths followed by the fields the
    palette shares. Duplicates collapse, so a scalar broadcast across the
    palette is one format and is checked once.
    """
    return tuple(dict.fromkeys((fn, (*w, *shared)) for w in widths))


def _check_stored(stored: tuple[_Stored, ...], storage: torch.dtype) -> None:
    """Hold each format a result holds against the narrower dtype storing it."""
    for check, fmt in stored:
        check(*fmt, storage=storage)


def _palette_list(val: int | Sequence[int], n: int, name: str) -> list[int]:
    """A scalar broadcast to an ``n``-length list, or a sequence whose length
    is checked to be ``n`` (``name`` is for the error message)."""
    if isinstance(val, int):
        return [val] * n
    out = list(val)
    if len(out) != n:
        raise ValueError(f"{name} must have length {n} (the palette size), got {len(out)}")
    return out


def _palette_list_or(
    val: int | Sequence[int] | None, default: list[int], n: int, name: str
) -> list[int]:
    """:func:`_palette_list`, but ``None`` falls back to ``default``."""
    return default if val is None else _palette_list(val, n, name)


def _palette_pair(
    first: Sequence[int], second: Sequence[int], second_name: str, fn: str
) -> tuple[list[int], list[int], int]:
    """The two per-entry sequences that define a palette, as lists, with its size.

    Every mixed op takes one such pair, ``(K, P)`` or ``(man_bits, exp_bits)``,
    whose common length is the palette size every other palette argument is
    broadcast or defaulted against. An empty palette or a length mismatch
    raises ``ValueError`` naming ``fn``, the wrapper being called.
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
    """The per-entry exponent biases of a binaryK palette: :func:`_binaryK_bias`
    of each entry when ``bias`` is ``None``, else the scalar broadcast or the
    sequence checked for length."""
    if bias is None:
        return [_binaryK_bias(k, p, is_signed) for k, p in zip(K_l, P_l, strict=True)]
    return _palette_list(bias, len(K_l), name)


# --- a GEMM call, resolved ---------------------------------------------------


class _GemmSpec(NamedTuple):
    """One GEMM's formats resolved into exactly what the op takes.

    ``args`` is every schema argument after the operands and the transpose
    flags, in order. ``stored`` is the formats whose values the result holds:
    the last rounding's, one per distinct palette entry, and none when the
    last step is left unrounded. ``findings`` is what each carrier says about
    every format the op rounds with. Those two are all a call still has to
    check, against the operand dtype it is the first to know.

    Resolving is split from calling so a caller with a fixed format can
    resolve once and call many times: `mptorch.quant.gemm`'s factories build
    one spec per layer and bind it into the layer's math hooks, and
    `mptorch.quant.mac` memoizes one per frozen format value. Re-deriving
    every default and re-running every check on each call costs 2.3-6.4 us
    per call, most of a small GEMM's Python overhead.
    """

    op: Callable[..., torch.Tensor]
    args: tuple[Any, ...]
    stored: tuple[_Stored, ...] = ()
    # Whether `op` is one of the four palette ops, which take a `prec_idx`
    # between the operands and the transpose flags. `mptorch.quant.mac` reads
    # this to decide whether a call needs a map, rather than re-deriving it
    # from the format.
    mixed: bool = False
    # The carrier the caller named, as a dtype, or None for the operands' own.
    carrier: torch.dtype | None = None
    # binary32's findings, then binary64's, as `_format_findings` returns them.
    findings: tuple[_Findings, _Findings] = ((), ())


def _packed(x: torch.Tensor, trans: bool) -> tuple[torch.Tensor, bool]:
    """An operand the op can read without a copy, and the flag to read it with.

    The kernel reads ``op(x)`` through ``x``'s own strides, so a tensor that
    is the transpose of a contiguous one (``k.mT`` in ``q @ k.mT``, the shape
    a batched attention GEMM has) needs no copy, only the opposite transpose
    flag. Any other non-contiguous layout is copied. The values are the same
    either way: the kernel visits the same elements, and stochastic rounding
    keys its random stream on the output element, not on the operand's
    layout. A contiguous operand takes the first branch, so the common path
    costs one ``is_contiguous()``.
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
    """The prologue of a call whose spec names its carrier.

    Runs the checks that carrier and the operand dtype decide (the carrier's
    findings, then the storage check when the dtype is narrower than the
    carrier) and returns the operands in that carrier, plus the dtype to
    narrow the result back to when they had to be widened, or ``None``.
    """
    dtype = a.dtype
    wide, widen = _call_carrier(spec.carrier, dtype)
    found = spec.findings[wide]
    if found:
        _report_findings(found)
    if spec.stored and dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        _check_stored(spec.stored, dtype)
    # Operands of two dtypes are the op's to refuse; widening both here would
    # hide the mismatch.
    if widen and b.dtype is dtype:
        return a.double(), b.double(), dtype
    return a, b, None


def _run_gemm(
    spec: _GemmSpec, a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> torch.Tensor:
    """Call a resolved GEMM on rank-2 or rank-3 operands.

    All that is left per call is what the operand dtype decides (the carrier's
    findings, the storage check, widening for a named carrier) and the layout
    fold. ``torch.matmul``'s operand rules (1D promotion, leading-dim
    broadcasting, rank above 3) are :func:`_matmul_operands`' job, above this.
    """
    narrow = None
    if spec.carrier is not None:
        a, b, narrow = _named_carrier_operands(spec, a, b)
    else:
        # The cost of every call without a `carrier=`, spelled out here and in
        # `_run_gemm_mixed` rather than behind a function call (0.25 us): the
        # operands' carrier's findings, indexed by a bool (binary32's, then
        # binary64's), and the storage check, of which a float32 call pays
        # only the membership test.
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
    """:func:`_run_gemm` for the palette ops, whose schema takes ``prec_idx`` third.

    ``prec_idx`` is passed through as given. Narrowing and packing it is the
    C++ side's job, and that side memoizes its bounds check on the identity
    and version of the tensor it is handed, so a ``.to()`` here would hand it
    a fresh tensor every call and make it re-check the map every call.
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


# --- torch.matmul's operand rules, over a rank-2-or-3 op ----------------------
#
# The op boundary is rank 2 or rank 3, strictly: at most one batch dimension,
# which each operand either carries or is read across at stride 0
# (`common/gemm_host.h`). Everything torch.matmul accepts on top of that, a 1D
# operand, leading dims of any rank, broadcasting between them, is expressed
# here, as views wherever torch.matmul itself would use one, so the two agree
# on when a copy happens.


class _MatmulLayout(NamedTuple):
    """What the op is handed, and the shape its result has to come back as.

    ``out_shape`` already has the promoted dimension of a 1D operand removed,
    so the caller reshapes the op's ``[B, M, N]`` (or ``[M, N]``) result to
    it and is done.
    """

    a: torch.Tensor
    b: torch.Tensor
    trans_a: bool
    trans_b: bool
    out_shape: tuple[int, ...]


def _collapse(x: torch.Tensor) -> torch.Tensor:
    """Fold every leading dim of a rank>2 operand into one, without a copy.

    ``reshape`` alone would copy a transposed view, which is the layout
    :func:`_packed` exists to keep, so a transposed view is collapsed through
    its own base and handed back transposed.
    """
    # The batch extent is spelled out rather than inferred: `reshape(-1, r, c)`
    # is ambiguous when the tensor is empty, which a zero batch or a zero M is.
    batch = 1
    for d in x.shape[:-2]:
        batch *= d
    if x.is_contiguous() or not x.transpose(-2, -1).is_contiguous():
        return x.reshape(batch, *x.shape[-2:])
    xt = x.transpose(-2, -1)
    return xt.reshape(batch, *xt.shape[-2:]).transpose(-2, -1)


def _operand_3d(x: torch.Tensor, batch: tuple[int, ...]) -> torch.Tensor:
    """One operand as the rank-2 or rank-3 tensor the op takes.

    No leading dims, or all of them 1, means the op reads it at batch stride
    0: shared across the batch, never expanded. Leading dims that are already
    the broadcast batch collapse to one dim as a view. Anything else is a
    genuine partial broadcast (``[4, 1, M, K] @ [1, 3, K, N]``), and expanding
    it copies exactly where ``torch.matmul`` itself copies.
    """
    lead = x.shape[:-2]
    if all(d == 1 for d in lead):
        while x.dim() > 2:  # squeeze(0) is a view whatever the layout
            x = x.squeeze(0)
        return x
    if tuple(lead) != tuple(batch):
        x = x.expand(*batch, *x.shape[-2:])
    return _collapse(x)


def _broadcast_batch(a_lead: Sequence[int], b_lead: Sequence[int]) -> tuple[int, ...]:
    """The leading dims two operands broadcast to.

    ``torch.broadcast_shapes`` spelled out by hand because it costs 8.7 us,
    most of an entire resolved GEMM call, for a pair of tuples that are
    usually equal and never long. Same rule, same errors, about 0.3 us.
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
    """``torch.matmul``'s operand contract, resolved onto the rank-2/3 op.

    Returns the operands as the op takes them, the transpose flags to read
    them with, and the shape the result must come back as. ``trans_a`` and
    ``trans_b`` apply to the last two dims, as they would on a ``bmm`` of
    transposed views; they are ignored for a 1D operand, which
    ``torch.matmul`` has no transpose flag for.

    ``fold=False`` keeps the batch a batch. Only a palette op needs that: its
    ``prec_idx`` is indexed by output element, so a map shaped for the
    unfolded ``[M, N]`` output does not describe the folded ``[B*M, N]`` one.
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

    # `[..., M, K] @ [K, N]` with a shared, untransposed `a`: folding the batch
    # into M is a view and gives one large GEMM instead of B small ones. It is
    # bit-identical to the batched spelling, not merely equivalent, because
    # stochastic rounding keys each element's random stream on its index into
    # the dense [batch, M, N] output: `(b*M + row)*N + col` batched and
    # `row'*N + col` folded, with `row' = b*M + row`, are the same index. So
    # the fold needs no gate of its own. This is the QLinear shape, which
    # `gemm.py` folds the same way.
    if fold and a3.dim() == 3 and b3.dim() == 2 and not trans_a and a3.is_contiguous():
        a3 = a3.reshape(-1, a3.shape[-1])

    return _MatmulLayout(a3, b3, trans_a, trans_b, out_shape)


def _gemm_nd(
    spec: _GemmSpec, a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> torch.Tensor:
    """:func:`_run_gemm` under ``torch.matmul``'s operand rules.

    This is what the flat wrappers call; the layer hooks in
    `mptorch.quant.gemm` call :func:`_run_gemm` directly on operands they
    have already flattened.
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

    The map's own leading dims are collapsed the same way the operands' are,
    so a per-batch-element map keeps up with a rank>3 call, and the batch is
    never folded into ``M``, because the map is indexed by output element and
    a folded output numbers its elements differently.
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
    """``run``, a resolved 2D/3D GEMM, under ``torch.matmul``'s operand rules.

    Two ranks reach the op untouched, and they are the ones every layer uses:
    a pair of matrices, and a pair of equal batches. Both are already exactly
    what the op takes, and the general path (:func:`_matmul_operands`, mostly
    its shape broadcast) costs about 12 us of Python to hand back the
    operands themselves, several times a resolved call's whole overhead.
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a binaryK floating-point format.

    binaryK is the parameterized family of IEEE P3109, the draft Standard for
    Arithmetic Formats for Machine Learning (see :class:`mptorch.BinaryK`). A
    format has ``K`` bits in total, ``P`` of them precision (``P - 1`` stored
    mantissa bits plus the implicit one), and ``K - P`` exponent bits
    (``K - P + 1`` when ``is_signed`` is false, since there is no sign bit).
    Pass ``bias`` explicitly for a format outside P3109: the OCP 8-bit formats
    are E4M3, ``K=8, P=4, bias=7``, and E5M2, ``K=8, P=3, bias=15``.

    The rounding happens in a *carrier*, the binary format the arithmetic
    runs in: binary64 for a float64 ``x``, and binary32 for float32, float16
    and bfloat16, on their float32 value. The format is held to what that
    carrier can hold on every call, at most 24 bits of precision and seven
    exponent bits in binary32, 53 and ten in binary64 (see :doc:`/concepts`),
    and the random bits of ``RoundMode.SR`` are drawn in the carrier too, so
    ``P - 1 + prng_bits`` is held to its 23 or 52 mantissa bits. A result
    narrower than its carrier (float16 or bfloat16, or float32 under binary64)
    is rounded a second time when it is stored, so the format is held against
    that dtype as well; the inputs are already its values, so only a result
    at an edge of the format's range can land off its grid, and that warns.
    NaN inputs pass through, and so do infinities except under
    ``SaturationMode.SAT_FINITE``, which clamps them to the largest finite
    value.

    Not differentiable: rounding to a coarse grid has a zero derivative almost
    everywhere, so on a tensor that requires grad under grad mode this raises
    and points at :class:`mptorch.quant.Quantizer`, the straight-through
    estimator. See :class:`mptorch.quant.Quant` for the format-object
    spelling of the same call.

    Args:
        x (Tensor): the tensor to round; float32, float64, float16 or
            bfloat16.
        K (int): the format's width in bits.
        P (int): the format's precision: ``P - 1`` stored mantissa bits plus
            the implicit one.
        bias (int, optional): the exponent bias. Default: ``None``, which is
            P3109's ``2**(K - P - 1)`` signed and ``2**(K - P)`` unsigned, one
            more than IEEE 754 gives the same exponent width.
        prng_bits (int): random bits ``RoundMode.SR`` draws below the target
            mantissa; ignored by every other mode. Default: ``0``
        is_signed (bool): whether the format has a sign bit. Default: ``True``
        rounding_mode (RoundMode): how a value between two of the format's is
            rounded. Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): what a value past the largest finite
            one becomes. Default: ``SaturationMode.OVF_INF``
        subnormals_mode (SubnormalsMode): how the bottom of the range is
            filled in. Default: ``SubnormalsMode.SUBNORMALS``
        carrier (torch.dtype, optional): ``torch.float64`` rounds any ``x`` in
            binary64, widening a narrower one first and narrowing the result
            back to its dtype with one rounding. ``torch.float32`` names
            binary32, which a float64 ``x`` refuses, since the carrier is
            never narrower than the tensor. Default: ``None``, ``x``'s own
            carrier.

    Returns:
        Tensor: a new tensor of ``x``'s shape and dtype holding the rounded
        values.

    Raises:
        ValueError: if the carrier cannot hold the format (precision above the
            carrier's, ``P - 1 + prng_bits`` above its mantissa bits, more
            than seven or ten exponent bits, no finite normal value), or if
            ``carrier`` is a dtype that names no carrier or is narrower than
            ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode.

    Warns:
        FormatRangeWarning: if the format's range outruns the carrier's, or a
            result at an edge of the range lands off a narrower storage
            dtype's grid. The format still quantizes correctly over the part
            that fits.

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
    """``x`` as the elementwise op reads it: contiguous, and widened to
    float64 when the call's carrier asks for that, in a single copy."""
    if widen:
        return x.to(torch.float64, memory_format=torch.contiguous_format)
    return x.contiguous()


def _refuse_widening_in_place(op: str, dtype: torch.dtype) -> None:
    """Raise for an in-place quantizer whose carrier is wider than its tensor.

    ``carrier=torch.float64`` on a float32, float16 or bfloat16 tensor rounds
    a float64 copy of it and narrows the result back, which takes a buffer of
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a binaryK format, in place.

    The in-place spelling of :func:`binaryK_quantize`: the same arguments, the
    same per-call format checks and, element for element, the same result
    (under ``RoundMode.SR`` and one seed too, since each element's random word
    is keyed on its index), written over ``x``. There is no output allocation
    and no output-sized footprint, so peak memory is one tensor instead of
    two; the time is the same, since the kernel reads and writes the same
    bytes either way. It is for a caller that owns ``x``: a weight quantized
    once at load, an activation nothing else will read.

    What :func:`binaryK_quantize` handles by copying, this refuses, because a
    copy is what the kernel would then write, leaving ``x`` as it was:

    * a tensor that is not contiguous;
    * on CUDA, a view that does not start on a 16-byte boundary (``x[1:]``),
      which the kernel's vector loads cannot address;
    * ``carrier=torch.float64`` on a float32, float16 or bfloat16 tensor,
      which rounds a float64 copy and narrows the result. A float64 tensor
      rounds in binary64 in place like any other dtype in its carrier.

    It has no MPS kernel yet and raises on an MPS tensor.

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
            contiguous, is a CUDA view off a 16-byte boundary, or is on MPS.

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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a superfp (supernormal) format.

    The format has ``man_bits`` stored mantissa bits and ``exp_bits`` exponent
    bits, but only the top ``normal_binades`` binades carry the mantissa. The
    lower binades' encodings become that many further powers of two below the
    normal region, the supernormals, and values below those flush to zero:
    there are no subnormals, hence no ``subnormals_mode``. ``bias`` is
    required, since the format has no default rule for it. See
    :class:`mptorch.SuperFP`.

    Dtypes, the carrier, the storage check, ``prng_bits`` and
    differentiability are as for :func:`binaryK_quantize`.

    Args:
        x (Tensor): the tensor to round; float32, float64, float16 or
            bfloat16.
        man_bits (int): stored mantissa bits of a normal binade.
        exp_bits (int): exponent bits.
        normal_binades (int): how many of the top binades carry the mantissa;
            each lower binade holds one power of two.
        bias (int): the exponent bias.
        prng_bits (int): random bits ``RoundMode.SR`` draws below the target
            mantissa; ignored by every other mode. Default: ``0``
        is_signed (bool): whether the format has a sign bit. Default: ``True``
        rounding_mode (RoundMode): how a value between two of the format's is
            rounded. Default: ``RoundMode.RNE``
        saturation_mode (SaturationMode): what a value past the largest finite
            one becomes. Default: ``SaturationMode.OVF_INF``
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`.
            Default: ``None``, ``x``'s own carrier.

    Returns:
        Tensor: a new tensor of ``x``'s shape and dtype holding the rounded
        values.

    Raises:
        ValueError: if the carrier cannot hold the format (precision above the
            carrier's, ``man_bits + prng_bits`` above its mantissa bits, an
            over-wide exponent field, a ``normal_binades`` that leaves no
            supernormal code, no finite normal value), or if ``carrier`` is a
            dtype that names no carrier or is narrower than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode.

    Warns:
        FormatRangeWarning: if the format's range outruns the carrier's, or a
            result at an edge of the range lands off a narrower storage
            dtype's grid.

    Example::

        >>> x = torch.tensor([1.1, 3.3, 0.05])
        >>> superfp_quantize(x, man_bits=3, exp_bits=4, normal_binades=8, bias=7)
        tensor([1.0000, 3.2500, 0.0625])

    With ``bias=7`` the eight normal binades start at 2, so 3.3 rounds on a
    grid of step 0.25, while 1.1 and 0.05 fall in the supernormal region and
    round to a power of two.
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a superfp format, in place.

    The in-place spelling of :func:`superfp_quantize`: the same arguments,
    checks and result, written over ``x``, with no output allocation. What it
    is for and what it refuses (a tensor that is not contiguous, a CUDA view
    off a 16-byte boundary, a ``carrier`` wider than ``x``, an MPS tensor) are
    as for :func:`binaryK_quantize_`.

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
            contiguous, is a CUDA view off a 16-byte boundary, or is on MPS.

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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul`'s formats into a ``_GemmSpec``.

    See that function for the contract. The ``acc_*`` fallbacks to the
    multiply format, the default biases and the zeroed accumulate format of
    ``accumulate_quant=False`` are applied here, and binary64's errors raise.
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
    # The result holds values of the accumulate format; the products never
    # reach storage.
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized GEMM with a split multiply-accumulate in binaryK formats.

    Computes ``op(a) @ op(b)``, where ``op(x) = x.T`` when the corresponding
    ``trans_*`` flag is set, else ``op(x) = x``. Unlike quantizing ``a`` and
    ``b`` and then calling ``torch.matmul``, this quantizes the *arithmetic*
    of each dot product: every partial product is rounded to the binaryK
    format given by ``mul_K``/``mul_P``, and, when ``accumulate_quant`` is
    true, the running sum is rounded to the format given by
    ``acc_K``/``acc_P`` after every accumulation step. With
    ``accumulate_quant=False`` the running sum stays at the carrier's
    precision and only the multiply is quantized.

    Everything the formats do not round (a product before its rounding, a sum
    before its) is computed in a carrier, and so are the roundings: binary64
    for float64 operands and binary32 for float32, float16 and bfloat16. Both
    formats are held to that carrier on every call, and the accumulate
    format, whose values the result holds, is also held against a result
    dtype narrower than the carrier, as in :func:`binaryK_quantize`.

    ``a`` and ``b`` follow ``torch.matmul``'s operand rules: 1D operands are
    promoted (and their dimension dropped from the result), leading
    dimensions broadcast against each other, and ``trans_a``/``trans_b``
    apply to the last two dimensions. A shared operand is read at stride 0
    rather than expanded, an operand that is the transpose of a contiguous
    tensor flips the kernel's flag instead of being copied, and
    ``[..., M, K] @ [K, N]`` folds its batch into ``M`` as a view, so the two
    shapes a quantized attention block uses cost no copy. Under
    ``RoundMode.SR`` each output element's random stream is keyed on its
    index into the ``[batch, M, N]`` result, so the folded and the batched
    spelling are bit-identical.

    This function resolves the format on every call. A caller holding one
    format across many calls should use `mptorch.quant.gemm`'s factories,
    which resolve it once per layer, and :func:`mptorch.quant.qmatmul` is the
    differentiable entry point.

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
            Default: ``None``, P3109's ``2**(K - P - 1)`` signed and
            ``2**(K - P)`` unsigned.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits ``RoundMode.SR`` draws for the
            multiply rounding, as :func:`binaryK_quantize`'s ``prng_bits``.
            Default: ``0``
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
        acc_prng_bits (int): random bits ``RoundMode.SR`` draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the partial products
            are folded into the running sum; only ``NAIVE`` is implemented.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding of both formats, one mode per
            op because the kernel is instantiated on it. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply format's overflow
            behavior. Default: ``SaturationMode.OVF_INF``
        subnormals_mode (SubnormalsMode): the multiply format's bottom of
            range. Default: ``SubnormalsMode.SUBNORMALS``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            format's overflow behavior. Default: ``None``, ``saturation_mode``.
        acc_subnormals_mode (SubnormalsMode, optional): the accumulate
            format's bottom of range. Default: ``None``, ``subnormals_mode``.
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`:
            ``torch.float64`` widens narrower operands to binary64 and narrows
            the result back to their dtype with one rounding, and
            ``torch.float32`` refuses float64 operands. Default: ``None``, the
            operands' own carrier.

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold a format (as for
            :func:`binaryK_quantize`), if ``carrier`` names no carrier or is
            narrower than the operands, or if the operands' leading dimensions
            do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode (use :func:`mptorch.quant.qmatmul`).

    Warns:
        FormatRangeWarning: if a format's range outruns the carrier's, or the
            accumulate format's range reaches past a narrower result dtype's.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> binaryK_matmul(a, a, mul_K=8, mul_P=4)
        tensor([[ 7., 10.],
                [15., 22.]])
        >>> binaryK_matmul(a, a, mul_K=8, mul_P=3)
        tensor([[ 7., 10.],
                [16., 24.]])

    With three stored mantissa bits (``P=4``) every product and sum is exact;
    with two (``P=3``) the sums 15 and 22 round to 16 and 24.
    """
    return _gemm_nd(
        _binaryK_spec(
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
        trans_a,
        trans_b,
    )


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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul`'s formats into a ``_GemmSpec``.

    See that function for the contract. The ``acc_*`` fallbacks to the
    multiply format and the zeroed accumulate format of
    ``accumulate_quant=False`` are applied here, and binary64's errors raise.
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized GEMM with a split multiply-accumulate in superfp formats.

    The superfp analog of :func:`binaryK_matmul`: every partial product is
    rounded to the superfp format given by ``mul_man_bits``, ``mul_exp_bits``,
    ``mul_normal_binades`` and ``mul_bias``, and, when ``accumulate_quant`` is
    true, the running sum to the ``acc_*`` format after every step. See
    :func:`binaryK_matmul` for the general contract (``torch.matmul``'s
    operand rules, the transpose flags, the carrier, the storage check and
    the per-call resolution) and :func:`superfp_quantize` for the format.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_man_bits (int): stored mantissa bits of the multiply format.
        mul_exp_bits (int): exponent bits of the multiply format.
        mul_normal_binades (int): binades of the multiply format that carry
            the mantissa.
        mul_bias (int): exponent bias of the multiply format.
        mul_is_signed (bool): whether the multiply format has a sign bit.
            Default: ``True``
        mul_prng_bits (int): random bits ``RoundMode.SR`` draws for the
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
        acc_prng_bits (int): random bits ``RoundMode.SR`` draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the partial products
            are folded into the running sum; only ``NAIVE`` is implemented.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding of both formats. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply format's overflow
            behavior. Default: ``SaturationMode.OVF_INF``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            format's overflow behavior. Default: ``None``, ``saturation_mode``.
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold a format (as for
            :func:`superfp_quantize`), if ``carrier`` names no carrier or is
            narrower than the operands, or if the operands' leading dimensions
            do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if a format's range outruns the carrier's, or the
            accumulate format's range reaches past a narrower result dtype's.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> superfp_matmul(
        ...     a, a, mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=8, mul_bias=7
        ... )
        tensor([[ 7., 10.],
                [15., 22.]])
    """
    return _gemm_nd(
        _superfp_spec(
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
        trans_a,
        trans_b,
    )


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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_fma`'s format into a ``_GemmSpec``.

    See that function for the contract. With ``fma_quant=False`` no format is
    rounded with or stored, so there is nothing for the carriers to find.
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized GEMM with a fused multiply-add in a binaryK format.

    The fused analog of :func:`binaryK_matmul`: every dot-product step
    computes a single hardware-style fused multiply-add, ``a*b + acc`` with
    one rounding, instead of rounding the multiply and the accumulate
    separately. A real FMA unit has one output rounding, so there is a single
    ``fma_K``/``fma_P`` format rather than a ``mul_*``/``acc_*`` pair. With
    ``fma_quant=False`` the fused step runs at the carrier's own precision,
    the fused analog of :func:`binaryK_matmul`'s ``accumulate_quant=False``.

    ``a`` and ``b`` follow ``torch.matmul``'s operand rules, and the carrier,
    the storage check and the per-call resolution are as for
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
            P3109's ``2**(K - P - 1)`` signed and ``2**(K - P)`` unsigned.
        fma_is_signed (bool): whether the format has a sign bit. Default:
            ``True``
        fma_quant (bool): whether the fused step is rounded to the format at
            all. Default: ``True``
        fma_prng_bits (int): random bits ``RoundMode.SR`` draws for the fused
            rounding, as :func:`binaryK_quantize`'s ``prng_bits``. Default:
            ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the fused steps are
            ordered; only ``NAIVE`` is implemented. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the fused step's rounding. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the format's overflow behavior.
            Default: ``SaturationMode.OVF_INF``
        subnormals_mode (SubnormalsMode): the format's bottom of range.
            Default: ``SubnormalsMode.SUBNORMALS``
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold the format, if ``carrier`` names
            no carrier or is narrower than the operands, or if the operands'
            leading dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if the format's range outruns the carrier's or
            reaches past a narrower result dtype's.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> binaryK_matmul_fma(a, a, fma_K=8, fma_P=3)
        tensor([[ 7., 10.],
                [16., 24.]])
    """
    return _gemm_nd(
        _binaryK_fma_spec(
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
        trans_a,
        trans_b,
    )


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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_fma`'s format into a ``_GemmSpec``.

    See that function for the contract. With ``fma_quant=False`` no format is
    rounded with or stored, so there is nothing for the carriers to find.
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized GEMM with a fused multiply-add in a superfp format.

    The superfp analog of :func:`binaryK_matmul_fma`: every dot-product step
    is one fused multiply-add rounded once to the format given by
    ``fma_man_bits``, ``fma_exp_bits``, ``fma_normal_binades`` and
    ``fma_bias`` (see :func:`superfp_quantize`), or left at the carrier's
    precision with ``fma_quant=False``. ``a`` and ``b`` follow
    ``torch.matmul``'s operand rules, and the carrier, the storage check and
    the per-call resolution are as for :func:`binaryK_matmul`.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        fma_man_bits (int): stored mantissa bits of the fused step's format.
        fma_exp_bits (int): its exponent bits.
        fma_normal_binades (int): its binades that carry the mantissa.
        fma_bias (int): its exponent bias.
        fma_is_signed (bool): whether the format has a sign bit. Default:
            ``True``
        fma_quant (bool): whether the fused step is rounded to the format at
            all. Default: ``True``
        fma_prng_bits (int): random bits ``RoundMode.SR`` draws for the fused
            rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the fused steps are
            ordered; only ``NAIVE`` is implemented. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the fused step's rounding. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the format's overflow behavior.
            Default: ``SaturationMode.OVF_INF``
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if a carrier cannot hold the format, if ``carrier`` names
            no carrier or is narrower than the operands, or if the operands'
            leading dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if the format's range outruns the carrier's or
            reaches past a narrower result dtype's.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> superfp_matmul_fma(
        ...     a, a, fma_man_bits=2, fma_exp_bits=4, fma_normal_binades=8, fma_bias=7
        ... )
        tensor([[ 7., 10.],
                [16., 24.]])
    """
    return _gemm_nd(
        _superfp_fma_spec(
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
        trans_a,
        trans_b,
    )


# --- palette (spatially-varying mixed-format) GEMMs --------------------------
#
# A palette op takes up to eight formats and a `prec_idx` map that chooses one
# per output element. Its builder resolves every palette argument to a list of
# the palette's length (scalars broadcast, `None` defaulted per entry) and
# deduplicates the formats before the carrier and storage checks, so a scalar
# repeated across the palette is checked once.


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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_mixed`'s palette into a ``_GemmSpec``.

    See that function for the contract. Every palette argument becomes a list
    of the palette's length here, with the per-entry defaults applied, and
    binary64's errors raise.
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized GEMM whose binaryK format varies per output element.

    The palette analog of :func:`binaryK_matmul`: the same split arithmetic,
    a rounded multiply and then a rounded accumulate, but the binaryK format
    each output element's dot product runs in is chosen from a *palette* of
    up to 8 formats by ``prec_idx``. ``mul_K`` and ``mul_P`` are per-entry
    sequences whose common length is the palette size ``n``; the other
    palette arguments may be a scalar, broadcast to every entry, an
    ``n``-length sequence, or ``None`` with the same per-entry defaulting as
    :func:`binaryK_matmul` (``acc_*`` fall back to the ``mul_*`` entry, a
    bias to P3109's). Every entry is held to the call's carrier, and every
    accumulate entry to a narrower result dtype.

    ``prec_idx`` selects a palette entry per output element. Its shape must
    be the GEMM's output shape ``[M, N]`` (dense), ``[M, 1]`` (per row) or
    ``[1, N]`` (per column), optionally with a leading batch dimension of
    ``B`` (one map per batch element) or 1. The batch is never folded into
    ``M`` here, since the map is indexed by output element. Any integer dtype
    and layout is accepted: a map that is already ``int32``, contiguous and
    on ``a``'s device is passed through untouched, and any other spelling is
    narrowed and packed on every call. On CUDA the bounds check reads the
    map's extremes back to the host, so it is memoized on the map tensor's
    identity and version; hold one map and reuse it across calls rather than
    rebuilding it, which would re-check it each call.

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
        mul_prng_bits (int): random bits ``RoundMode.SR`` draws for the
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
        acc_prng_bits (int): random bits ``RoundMode.SR`` draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the partial products
            are folded into the running sum; only ``NAIVE`` is implemented.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding of every format. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply formats' overflow
            behavior. Default: ``SaturationMode.OVF_INF``
        subnormals_mode (SubnormalsMode): the multiply formats' bottom of
            range. Default: ``SubnormalsMode.SUBNORMALS``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            formats' overflow behavior. Default: ``None``, ``saturation_mode``.
        acc_subnormals_mode (SubnormalsMode, optional): the accumulate
            formats' bottom of range. Default: ``None``, ``subnormals_mode``.
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if the palette is empty or a palette argument's length is
            not ``n``, if a carrier cannot hold an entry, if ``carrier`` names
            no carrier or is narrower than the operands, or if the operands'
            leading dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``n`` is above 8, ``prec_idx`` has the wrong shape or
            an entry outside ``[0, n)``, the inner dimensions do not match,
            the operands differ in dtype or device, or an operand requires
            grad under grad mode.

    Warns:
        FormatRangeWarning: if an entry's range outruns the carrier's, or an
            accumulate entry's range reaches past a narrower result dtype's.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> prec_idx = torch.tensor([[0, 0], [0, 1]])
        >>> binaryK_matmul_mixed(a, a, prec_idx, mul_K=[8, 8], mul_P=[4, 3])
        tensor([[ 7., 10.],
                [15., 24.]])

    Three output elements use the ``P=4`` entry and are exact; the last uses
    the ``P=3`` entry, whose two mantissa bits round the sum 22 to 24.
    """
    return _gemm_mixed_nd(
        _binaryK_mixed_spec(
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_mixed`'s palette into a ``_GemmSpec``.

    See that function for the contract. Every palette argument becomes a list
    of the palette's length here, with the per-entry defaults applied, and
    binary64's errors raise.
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    acc_saturation_mode: SaturationMode | None = None,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized GEMM whose superfp format varies per output element.

    The superfp analog of :func:`binaryK_matmul_mixed`, which holds the
    palette, ``prec_idx`` and carrier contract. ``mul_man_bits`` and
    ``mul_exp_bits`` are per-entry sequences whose common length is the
    palette size ``n``; ``mul_normal_binades``, ``mul_bias`` and every
    ``acc_*`` may be a scalar or an ``n``-length sequence, and an ``acc_*``
    may also be ``None`` to fall back to the corresponding ``mul_*`` entry.

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
        mul_normal_binades (int or Sequence[int]): binades of each entry's
            multiply format that carry the mantissa.
        mul_bias (int or Sequence[int]): exponent bias of each entry's
            multiply format.
        mul_is_signed (bool): whether the multiply formats have a sign bit;
            shared by the palette. Default: ``True``
        mul_prng_bits (int): random bits ``RoundMode.SR`` draws for the
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
        acc_prng_bits (int): random bits ``RoundMode.SR`` draws for the
            accumulate rounding. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the partial products
            are folded into the running sum; only ``NAIVE`` is implemented.
            Default: ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding of every format. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the multiply formats' overflow
            behavior. Default: ``SaturationMode.OVF_INF``
        acc_saturation_mode (SaturationMode, optional): the accumulate
            formats' overflow behavior. Default: ``None``, ``saturation_mode``.
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if the palette is empty or a palette argument's length is
            not ``n``, if a carrier cannot hold an entry, if ``carrier`` names
            no carrier or is narrower than the operands, or if the operands'
            leading dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``n`` is above 8, ``prec_idx`` has the wrong shape or
            an entry outside ``[0, n)``, the inner dimensions do not match,
            the operands differ in dtype or device, or an operand requires
            grad under grad mode.

    Warns:
        FormatRangeWarning: if an entry's range outruns the carrier's, or an
            accumulate entry's range reaches past a narrower result dtype's.

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
    return _gemm_mixed_nd(
        _superfp_mixed_spec(
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_fma_mixed`'s palette into a ``_GemmSpec``.

    See that function for the contract. ``fma_quant=False`` is passed through
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized fused-multiply-add GEMM whose binaryK format varies per element.

    The palette analog of :func:`binaryK_matmul_fma`: each dot-product step
    is a single fused multiply-add rounded once, and the binaryK format that
    rounding uses is chosen per output element from a palette of up to 8
    formats by ``prec_idx``, whose contract is :func:`binaryK_matmul_mixed`'s.
    ``fma_K`` and ``fma_P`` are per-entry sequences whose common length is
    the palette size ``n``; ``fma_bias`` may be a scalar, an ``n``-length
    sequence or ``None`` for P3109's bias per entry. ``fma_quant=False`` is
    rejected: an unrounded fused step carries no format, so a palette of it
    would leave ``prec_idx`` nothing to choose; use
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
        fma_prng_bits (int): random bits ``RoundMode.SR`` draws for the fused
            rounding; shared by the palette. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the fused steps are
            ordered; only ``NAIVE`` is implemented. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding of every entry. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the formats' overflow behavior.
            Default: ``SaturationMode.OVF_INF``
        subnormals_mode (SubnormalsMode): the formats' bottom of range.
            Default: ``SubnormalsMode.SUBNORMALS``
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if the palette is empty or a palette argument's length is
            not ``n``, if a carrier cannot hold an entry, if ``carrier`` names
            no carrier or is narrower than the operands, or if the operands'
            leading dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``fma_quant`` is false, ``n`` is above 8,
            ``prec_idx`` has the wrong shape or an entry outside ``[0, n)``,
            the inner dimensions do not match, the operands differ in dtype
            or device, or an operand requires grad under grad mode.

    Warns:
        FormatRangeWarning: if an entry's range outruns the carrier's or
            reaches past a narrower result dtype's.

    Example::

        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> prec_idx = torch.tensor([[0, 0], [0, 1]])
        >>> binaryK_matmul_fma_mixed(a, a, prec_idx, fma_K=[8, 8], fma_P=[4, 3])
        tensor([[ 7., 10.],
                [15., 24.]])
    """
    return _gemm_mixed_nd(
        _binaryK_fma_mixed_spec(
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_fma_mixed`'s palette into a ``_GemmSpec``.

    See that function for the contract. ``fma_quant=False`` is passed through
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
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized fused-multiply-add GEMM whose superfp format varies per element.

    The superfp analog of :func:`binaryK_matmul_fma_mixed`, which holds the
    palette, ``prec_idx`` and carrier contract and the ``fma_quant=False``
    rejection. ``fma_man_bits`` and ``fma_exp_bits`` are per-entry sequences
    whose common length is the palette size ``n``; ``fma_normal_binades`` and
    ``fma_bias`` may each be a scalar or an ``n``-length sequence.

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
        fma_normal_binades (int or Sequence[int]): binades of each entry's
            format that carry the mantissa.
        fma_bias (int or Sequence[int]): exponent bias of each entry's
            format.
        fma_is_signed (bool): whether the formats have a sign bit; shared by
            the palette. Default: ``True``
        fma_quant (bool): must be ``True``. Default: ``True``
        fma_prng_bits (int): random bits ``RoundMode.SR`` draws for the fused
            rounding; shared by the palette. Default: ``0``
        accumulate_algorithm (AccumulateAlgorithm): how the fused steps are
            ordered; only ``NAIVE`` is implemented. Default:
            ``AccumulateAlgorithm.NAIVE``
        rounding_mode (RoundMode): the rounding of every entry. Default:
            ``RoundMode.RNE``
        saturation_mode (SaturationMode): the formats' overflow behavior.
            Default: ``SaturationMode.OVF_INF``
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.

    Returns:
        Tensor: the product, in the operands' dtype, with the shape
        ``torch.matmul`` would give.

    Raises:
        ValueError: if the palette is empty or a palette argument's length is
            not ``n``, if a carrier cannot hold an entry, if ``carrier`` names
            no carrier or is narrower than the operands, or if the operands'
            leading dimensions do not broadcast.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``fma_quant`` is false, ``n`` is above 8,
            ``prec_idx`` has the wrong shape or an entry outside ``[0, n)``,
            the inner dimensions do not match, the operands differ in dtype
            or device, or an operand requires grad under grad mode.

    Warns:
        FormatRangeWarning: if an entry's range outruns the carrier's or
            reaches past a narrower result dtype's.

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
    return _gemm_mixed_nd(
        _superfp_fma_mixed_spec(
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
