"""Fixed-point GEMMs whose every step is exact, run on a vendor GEMM.

A fixed-point mac rounds each product to its multiply format and each sum to
its accumulate format (`mptorch.number.fixedpoint`). When the operands are
values of fixed-point formats of at most 8 bits, every product is exact in
the multiply format and every partial sum exact in the accumulate format,
nothing is rounded or saturated, and the result is the exact integer sum of
the operands' codes, scaled. Any exact engine gives that sum, in any order,
so a vendor GEMM computes the same words as the fixed-point kernels, far
faster. Two engines serve:

* ``torch._int_mm`` (int8 x int8 -> int32) on CUDA, on MPS, and on a CPU
  whose torch has oneDNN: on a Ryzen 7940HS at 512^3, 3.5 against 954 ps a
  mac step on one thread, on an RTX 4060 0.34 against 10, on an M3 Pro's GPU
  about 2 against 24.
* torch's float matmul on a CPU without oneDNN, where ``_int_mm`` is a
  scalar loop (an M3 Pro's CPU: 513 ps against 746). Every partial sum is
  an integer number of the products' steps, at most ``K * p_max`` of them,
  so float32 sums exactly while that is at most 2^24 and float64 while it is
  at most 2^53; on the M3 Pro (Accelerate) they take 1.3 and 5.0 ps a step.

What decides it, and where:

* The operands' formats. A fixed-point GEMM's spec knows only its multiply
  and accumulate formats, not what it will be given, so the route is chosen
  where the operands' quantizers are known: a recipe's pass
  (`GemmFormats.affine` and `GemmFormats.matmul`), whose operand slots are
  both a `Quant` of a fixed-point format. `exact_route` checks the formats
  once, when the pass is built; `run_exact` checks the call (the length of
  the sum, the dtype, the device) and gives every other call to the kernel.
* The path the kernel would take. The integer accumulator returns the
  carrier's canonical quiet NaN for an output that a NaN operand reaches,
  on the CPU and on CUDA alike, and `run_exact` writes that NaN into the
  same rows and columns. The carrier path returns a NaN that depends on the
  backend (the CPU keeps the operand's payload), so the route takes only the
  calls the integer accumulator computes (`_fixedpoint_gemm`'s ``in_carrier``
  false), which every exact sum of 8-bit codes longer than a few hundred
  steps is.

Its memory: on ``_int_mm``, the operands' codes (one byte an element, a
quarter of a float32 operand) and the int32 product, computed in row blocks
of at most `_BLOCK_BYTES`; on a float matmul, nothing in float32, and the
operands in float64 when the sum needs binary64. Nothing it computes is
saved for backward.
"""

from functools import partial
from typing import Any, NamedTuple

import torch

from mptorch.number import FixedPoint, Overflow
from mptorch.number.fixedpoint import _fixedpoint_gemm

from ..mac import FusedMac, Quant, SplitMac, _resolved
from ..ops._checks import (
    _BELOW_BINARY64,
    _NARROW_STORAGE,
    _call_carrier,
    _check_stored,
    _report_findings,
)
from ..ops._operands import _gemm_nd, _matmul_nd
from ..ops._spec import _GemmSpec, _run_gemm

#: Whether calls take the route at all; the tests and timings compare both.
_ENABLED = True

#: The smallest GEMM, in mac steps (M x N x K), that takes the route on each
#: device, and on the CPU for each engine (``_CPU_ENGINE``). The route's code
#: conversions, NaN masks and scaling are a dozen small tensor operations,
#: which cost more than the kernel below these sizes. Kernel / route at n^3
#: with Q0.7 operands into (25, 14), on a Ryzen 7940HS at 4.8 GHz and an RTX
#: 4060 (2026-11-27), and on an M3 Pro:
#:
#: - the Ryzen's CPU, the int engine: 1.1x at 2^15 steps and 7x at 2^18 on
#:   one thread; on 16 threads 0.75-0.88x at 2^16-2^16.8, where the route's
#:   small operations go multithreaded, then 1.08x at 2^18 and 1.4x at 2^19;
#: - the Ryzen's CPU, the float engine: 1.06x at 2^13.8 and 3.6x at 2^16 on
#:   one thread; 0.99x at 2^15 and 1.2x at 2^16 on 16 threads;
#: - the M3 Pro's CPU, the float engine: 1.9x at 2^15 on 1 and 11 threads;
#: - CUDA: 0.73x at 2^24 and 0.99-1.08x at 2^24.5, since about 200 us of
#:   each call is launching the small operations, then 1.4x at 2^25;
#: - MPS: 0.7x at 2^18, about 1.1x at 2^21 and 1.3x at 2^22.8.
_MIN_STEPS = {"cpu-int": 1 << 18, "cpu-float": 1 << 16, "cuda": 1 << 25, "mps": 1 << 22}

#: The CPU's engine: ``"int"`` (``_int_mm``) where torch has oneDNN, whose
#: int8 GEMM it calls, and ``"float"`` (the float matmul) where it has not.
_CPU_ENGINE = "int" if torch.backends.mkldnn.is_available() else "float"

#: The largest int32 product block, in bytes.
_BLOCK_BYTES = 64 << 20

#: The engines, module attributes so that a test can count their calls.
_int_mm = torch._int_mm
_float_mm = torch.matmul

# The largest code magnitude an int8 holds, and the int32 sum's.
_INT8_MAX = 127
_INT32_MAX = (1 << 31) - 1


class ExactRoute(NamedTuple):
    """What a pass's calls need to take the route.

    ``fl_a`` and ``fl_b`` are the operands' fractional lengths, so their codes
    are ``value * 2^fl``. ``k_max`` is the longest sum the bound allows: no
    partial sum reaches the accumulate format's range or int32's. ``dtypes``
    are the operand dtypes whose calls the integer accumulator computes.
    ``p_max`` is the largest magnitude of a product's code, for the float
    engine's bound."""

    fl_a: int
    fl_b: int
    k_max: int
    dtypes: frozenset[torch.dtype]
    p_max: int


def _operand_format(slot: Any) -> FixedPoint | None:
    """The format an operand slot's values have, when it is a `Quant` of a
    fixed-point format whose codes fit an int8; ``None`` otherwise."""
    if type(slot) is not Quant or not isinstance(slot.fmt, FixedPoint):
        return None
    fmt = slot.fmt
    return fmt if _code_max(fmt) <= _INT8_MAX and _code_min(fmt) >= -_INT8_MAX - 1 else None


def _code_max(fmt: FixedPoint) -> int:
    """The largest code of ``fmt`` (its largest value times ``2^fl``)."""
    return (1 << fmt.mag_bits) - 1


def _code_min(fmt: FixedPoint) -> int:
    """The smallest code of ``fmt``: ``-2^mag``, one less for a symmetric
    format, and 0 for an unsigned one."""
    if not fmt.is_signed:
        return 0
    return -((1 << fmt.mag_bits) - 1) if fmt.symmetric else -(1 << fmt.mag_bits)


def _holds(fmt: FixedPoint, lo: int, hi: int, fl: int) -> bool:
    """Whether ``fmt`` holds every multiple of ``2^-fl`` from ``lo * 2^-fl``
    to ``hi * 2^-fl`` exactly: its grid is as fine and its range covers them."""
    if fmt.fl < fl:
        return False
    shift = fmt.fl - fl
    return _code_min(fmt) <= lo << shift and hi << shift <= _code_max(fmt)


def exact_route(mac: Any, a_slot: Any, b_slot: Any) -> ExactRoute | None:
    """The route of a pass computing ``op(a) @ op(b)`` in ``mac``, with ``a``
    and ``b`` the values of the quantizers ``a_slot`` and ``b_slot``; ``None``
    when its calls cannot all be exact.

    It needs a saturating fixed-point mac that does not round stochastically
    (an SR kernel draws its seed even when nothing rounds, and wrap-around is
    left out until an int32 GEMM's overflow is shown to wrap), both slots a
    `Quant` of a fixed-point format of at most 8 bits, a multiply format
    that holds every product, and an accumulate format whose grid is as fine
    as the products'."""
    if not isinstance(mac, SplitMac | FusedMac):
        return None
    if mac.overflow is not Overflow.SATURATE or mac.rounding.is_stochastic:
        return None
    fmt_a, fmt_b = _operand_format(a_slot), _operand_format(b_slot)
    if fmt_a is None or fmt_b is None:
        return None
    if isinstance(mac, FusedMac):
        if mac.fma is None:
            return None
        mul, acc = None, _resolved(mac.fma)[0]
    else:
        if mac.acc is None:
            return None
        mul, acc = _resolved(mac.mul)[0], _resolved(mac.acc)[0]
    if not isinstance(acc, FixedPoint) or (mul is not None and not isinstance(mul, FixedPoint)):
        return None

    # The products' codes, on the grid of 2^-(fl_a + fl_b).
    corners = [
        x * y
        for x in (_code_min(fmt_a), _code_max(fmt_a))
        for y in (_code_min(fmt_b), _code_max(fmt_b))
    ]
    p_lo, p_hi, fl_p = min(corners), max(corners), fmt_a.fl + fmt_b.fl
    if mul is not None and not _holds(mul, p_lo, p_hi, fl_p):
        return None
    if acc.fl < fl_p:
        return None
    # The longest sum whose partial sums the accumulate format and int32 hold:
    # k * p_lo and k * p_hi must stay within both.
    shift = acc.fl - fl_p
    k_max = _INT32_MAX // max(abs(p_lo), abs(p_hi), 1)
    if p_hi > 0:
        k_max = min(k_max, _code_max(acc) // (p_hi << shift))
    if p_lo < 0:
        if _code_min(acc) == 0:
            return None
        k_max = min(k_max, _code_min(acc) // (p_lo << shift))
    if k_max < 1:
        return None

    # The dtypes whose calls the integer accumulator computes: float32 in
    # the mac's carrier (binary32 unless it names binary64), float64 in
    # binary64, and only where that carrier accepts the formats.
    dtypes = set()
    for dtype in (torch.float32, torch.float64):
        carrier = torch.float64 if dtype is torch.float64 else (mac.carrier or torch.float32)
        plan = _fixedpoint_gemm(mul, acc, False, carrier)
        if plan.error is None and not plan.in_carrier:
            dtypes.add(dtype)
    if not dtypes:
        return None
    return ExactRoute(fmt_a.fl, fmt_b.fl, k_max, frozenset(dtypes), max(abs(p_lo), abs(p_hi)))


def _checked(spec: _GemmSpec, dtype: torch.dtype) -> None:
    """The checks `_run_gemm` makes on a call before it runs the op (the
    carrier's findings and the storage check), in its order."""
    if spec.carrier is not None:
        wide, _ = _call_carrier(spec.carrier, dtype)
        found = spec.findings[wide]
        if found:
            _report_findings(found)
        if spec.stored and dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
            _check_stored(spec.stored, dtype)
        return
    found = spec.findings[dtype is torch.float64]
    if found:
        _report_findings(found)
    if spec.stored and dtype in _NARROW_STORAGE:
        _check_stored(spec.stored, dtype)


def _codes(x: torch.Tensor, fl: int, rows: int, cols: int) -> torch.Tensor:
    """``x``'s codes, ``x * 2^fl`` as int8, in a ``rows`` x ``cols`` buffer
    whose padding is zero (a NaN's code is whatever the conversion gives:
    `run_exact` overwrites the outputs it reaches)."""
    scaled = torch.mul(x, 2.0**fl).to(torch.int8)
    if (rows, cols) == tuple(x.shape) and (x.device.type != "cuda" or scaled.is_contiguous()):
        return scaled
    padded = torch.zeros(rows, cols, dtype=torch.int8, device=x.device)
    padded[: x.shape[0], : x.shape[1]] = scaled
    return padded


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _int_2d(route: ExactRoute, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """``out = a @ b`` for 2D ``a`` ([M, K]) and ``b`` ([K, N]) on ``_int_mm``:
    the int32 sum of the codes, scaled by ``2^-(fl_a + fl_b)``. A NaN's
    outputs are `_nan_filled`'s to write."""
    M, K = a.shape
    N = b.shape[1]
    cuda = a.device.type == "cuda"
    if cuda:
        # cuBLASLt's int8 GEMM takes M in multiples of 32 and K, N in
        # multiples of 8, and some column-major operands not at all; zero
        # rows and columns add nothing to the sums. The CPU's and MPS's take
        # any shape.
        Mp, Kp, Np = max(_round_up(M, 32), 32), _round_up(K, 8), _round_up(N, 8)
    else:
        Mp, Kp, Np = M, K, N
    ca = _codes(a, route.fl_a, Mp, Kp)
    cb = _codes(b, route.fl_b, Kp, Np)
    scale = 2.0 ** -(route.fl_a + route.fl_b)
    step = max(1, _BLOCK_BYTES // (4 * Np))
    if cuda:
        step = max(32, step // 32 * 32)
    for r0 in range(0, M, step):
        r1 = min(M, r0 + step)
        rows = ca[r0 : r0 + _round_up(r1 - r0, 32)] if cuda else ca[r0:r1]
        block = _int_mm(rows, cb)[: r1 - r0, :N]
        # Exact: a sum the accumulate format holds has at most 24 (53)
        # significant bits, and the scale is a power of two that keeps it
        # normal (fl_a + fl_b is at most 125 for a binary32 format). Written
        # in place, not with out=, which under torch.compile can come back
        # with another layout (see `ScaledQuant`).
        out[r0:r1].copy_(block).mul_(scale)


def _float_engine(route: ExactRoute, a: torch.Tensor, b: torch.Tensor, out_dtype) -> torch.Tensor:
    """``a @ b`` on the float matmul, any rank ``torch.matmul`` takes:
    float32 when every partial sum (at most ``K * p_max`` product steps) has
    at most 24 significant bits, float64 otherwise; the words of `_int_2d`.
    A NaN's outputs are `_nan_filled`'s to write."""
    K = a.shape[-1]
    # float32 only at full precision: a lower float32 matmul precision
    # (torch.set_float32_matmul_precision) lets a backend multiply in fewer
    # bits, which is not exact; float64 is never lowered. Dynamo cannot read
    # the setting (a string), so a compiled graph takes float64.
    if (
        a.dtype is torch.float32
        and K * route.p_max <= 1 << 24
        and not torch.compiler.is_compiling()
        and torch.get_float32_matmul_precision() == "highest"
    ):
        prod = _float_mm(a, b)
    else:
        prod = _float_mm(a.double(), b.double())
    out = prod.to(out_dtype)
    # A float sum of zero products can be -0.0, which the integer sum never
    # is (P3109's one zero is unsigned). A compare and a fill, not `+ 0.0`,
    # which a compiler may drop.
    return out.masked_fill_(out == 0, 0.0)


def _nan_filled(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``out`` with the carrier's canonical quiet NaN wherever a NaN operand
    reaches: a row of ``a`` ([..., M, K]) or a column of ``b`` ([..., K, N]),
    which is what the integer accumulator returns there."""
    # A quantizer's values are finite or NaN, so a row's or a column's sum is
    # NaN exactly when it holds one, found in one reduction.
    nan_rows = a.sum(dim=-1).isnan()
    nan_cols = b.sum(dim=-2).isnan()
    # On the CPU outside a compiled graph, testing for a NaN costs no sync,
    # and the two passes over the result are skipped when there is none.
    if a.device.type == "cpu" and not torch.compiler.is_compiling():
        if not (nan_rows.any() or nan_cols.any()):
            return out
    out.masked_fill_(nan_rows.unsqueeze(-1), float("nan"))
    out.masked_fill_(nan_cols.unsqueeze(-2), float("nan"))
    return out


def run_exact(
    route: ExactRoute,
    spec: _GemmSpec,
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    """`_run_gemm` for a pass with a route: the same result, word for word,
    from the device's engine when this call qualifies, and from the kernel
    otherwise."""
    dtype = a.dtype
    K = a.shape[-2] if trans_a else a.shape[-1]
    M = a.shape[-1] if trans_a else a.shape[-2]
    N = b.shape[-2] if trans_b else b.shape[-1]
    batch = a.shape[0] if a.dim() == 3 else b.shape[0] if b.dim() == 3 else 1
    device = f"cpu-{_CPU_ENGINE}" if a.device.type == "cpu" else a.device.type
    if (
        not _ENABLED
        or dtype not in route.dtypes
        or b.dtype is not dtype
        or device not in _MIN_STEPS
        or K > route.k_max
        or M * N * K * batch < _MIN_STEPS[device]
    ):
        return _run_gemm(spec, a, b, trans_a, trans_b)
    _checked(spec, dtype)
    out_dtype = torch.float64 if dtype in spec.wide else dtype
    op_a = a.mT if trans_a else a
    op_b = b.mT if trans_b else b
    if a.device.type == "cpu" and _CPU_ENGINE == "float":
        return _nan_filled(_float_engine(route, op_a, op_b, out_dtype), op_a, op_b)
    if a.dim() == 2 and b.dim() == 2:
        out = torch.empty(M, N, dtype=out_dtype, device=a.device)
        _int_2d(route, op_a, op_b, out)
        return _nan_filled(out, op_a, op_b)
    # A batch (one of the two may be a broadcast view of one matrix).
    out = torch.empty(batch, M, N, dtype=out_dtype, device=a.device)
    for i in range(batch):
        _int_2d(route, op_a[i] if a.dim() == 3 else op_a, op_b[i] if b.dim() == 3 else op_b, out[i])
    return _nan_filled(out, op_a, op_b)


def exact_runner(route: ExactRoute | None) -> Any:
    """The runner a pass binds its spec to (as `_hooks._spec_matmul` does): a
    2D or 3D runner like `_run_gemm`, taking the route when there is one."""
    return _run_gemm if route is None else partial(run_exact, route)


def exact_runner_nd(route: ExactRoute | None) -> Any:
    """`exact_runner` with ``torch.matmul``'s operand rules, like `_gemm_nd`."""
    if route is None:
        return _gemm_nd

    def run(spec: _GemmSpec, a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool):
        return _matmul_nd(partial(run_exact, route, spec), a, b, trans_a, trans_b)

    return run
