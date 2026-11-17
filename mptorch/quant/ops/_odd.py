"""The carrier step rounded to odd: the checks and specs of the *_odd ops.

The carrier step rounded to odd (`csrc/common/carrier_odd.h`). With
`carrier_rounding=RoundMode.RO`, each mac step rounds its carrier operation to
odd instead of to nearest, so that the cast to the simulated format then
rounds the exact value. Each `custom_matmul_*_odd` op takes the same
arguments as its default counterpart, so its spec is the counterpart's with
the op replaced, plus a check of what the guarantee needs of the formats: the
cast of a result rounded to odd rounds the exact value only if every rounding
boundary of the format is a carrier value with an even last bit. With p the
carrier's precision (24 in binary32, 53 in binary64), a format of precision P
needs

  RNE, RNA           P <= p - 2
  RU, RD, RZ, RO     P <= p - 1
  SR (StochasticA)   P + prng_bits <= p - 1
  SR_B, SR_C         P + prng_bits <= p - 2

The nearest modes need one bit more because their boundaries are midpoints
between format values, and SR_B and SR_C because each reads one bit of the
value below the random bits (SR_C to find a midpoint there). A format outside
these limits raises rather than warns, on each call and for each carrier, and
the binary32 error names carrier=torch.float64 when binary64 can hold the
format: binary64's verdict raises when the spec is built, and binary32's when
a call rounds in binary32, as for every other carrier finding.

Under KAHAN, BLOCK and TREE the spec is the accumulated spec with its op
replaced by the `*_accumulated_odd` one (`_odd_accumulated`). When an outer
format rounds the running total, it is checked against the round-to-odd
limits along with the mac's formats, since its cast also rounds a sum that
was rounded to odd. The tail of `args` is the accumulation's
(`_accumulate._binaryK_accumulation`, `_accumulate._superfp_accumulation`):
block_size, outer_quant, the outer format's fields, and its prng_bits last. A
NAIVE call never comes here: `_checked_block_size` rejects one that names a
block size or an outer format, and the callers send a plain NAIVE call to the
NAIVE round-to-odd builders.

The per-family builders (`binaryk._binaryK_odd_spec` and its counterparts)
are in the family modules, since each calls its family's NAIVE builder.
"""

from collections.abc import Callable, Iterable, Sequence

from mptorch import AccumulateAlgorithm, RoundMode

from ._checks import _errors_first, _Findings
from ._spec import _GemmSpec

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


def _palette_slots(name: str, Ps: Sequence[int], prng_bits: int) -> list[tuple[str, int, int]]:
    """Return ``_odd_findings``' slots for a palette: one per entry, named with its index."""
    return [(f"{name} (entry {i})", P, prng_bits) for i, P in enumerate(Ps)]


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


def _checked_carrier_rounding_call(carrier_rounding: RoundMode) -> None:
    """Check a flat wrapper's ``carrier_rounding`` when it is not RNE: it must be RO."""
    if carrier_rounding is not RoundMode.RO:
        raise ValueError(
            "carrier_rounding is RoundMode.RNE (the carrier's nearest step) or RoundMode.RO "
            f"(rounded to odd), got {carrier_rounding!r}"
        )
