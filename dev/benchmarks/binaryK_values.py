#!/usr/bin/env python3
"""Print every value a BinaryK format represents, one per line.

BinaryK is IEEE P3109's binaryK family (see ``docs/source/concepts.rst``), so
the enumeration here is over its *code points*: K bits laid out as a binary
float, decoded the way the standard decodes them, with the reserved codes at
the top spent on the infinities and the NaN. That is a different derivation
from the one the kernels use -- they round a binary32 value onto the format's
grid and never look at an encoding -- which is what makes ``--verify`` worth
running: it quantizes every value this script prints and checks that the
kernel gives it back unchanged, so the two derivations have to agree.

The code points, for a format with exponent width E and precision P:

* magnitude code ``c`` decodes as ``e, t = divmod(c, 2**(P-1))``, a value of
  ``(1 + t*2**(1-P)) * 2**(e-bias)`` when ``e > 0`` and ``t * 2**(1-P-bias+1)``
  when ``e == 0`` -- the subnormals, which ``--subnormals`` can take away
  (``NORMALS``: those codes go unused and the range below the smallest normal
  flushes to zero) or move (``EXTENDED_NORMALS``: code 0's binade becomes one
  more binade of normals, with the full mantissa);
* a signed format spends the code with the sign bit set and no magnitude --
  what a binary float would read as -0.0 -- on its single NaN, which is why
  the kernels never return -0.0; an unsigned format spends its top code on it;
* the extended domain (``OVF_INF``, ``SAT_PROPAGATE``) spends the top
  remaining magnitude code on an infinity, the finite domain (``SAT_FINITE``)
  does not, and that one code is the whole difference between them.

Examples::

    binaryK_values.py -K 8 -P 4                    # Binary8p4se, P3109's bias
    binaryK_values.py -K 8 -P 4 --bias 7           # E4M3's finite values (below)
    binaryK_values.py -K 6 -P 3 --saturation SAT_FINITE --annotate --verify
    binaryK_values.py -K 8 -P 4 --unsigned --order code --style both
    binaryK_values.py -K 40 -P 30 --bias 512 --verify --dtype float64

``--bias 7`` is the closest a BinaryK comes to OCP / NVIDIA E4M3, and is not
E4M3: every finite value is E4M3's, at E4M3's code, but the three special
codes differ. 0x7f and 0xff are the infinities here and E4M3's two NaNs, and
0x80 is the NaN here and E4M3's -0. ``--saturation SAT_FINITE`` does not close
the gap: it turns 0x7f and 0xff into +480 and -480, which E4M3 does not have.

``--dtype float64`` verifies through the binary64 kernels instead, which hold
formats binary32 cannot (up to 53 bits of precision). A format of more than
2**24 code points is not printed but sampled, for ``--verify`` only: every
exponent field's first and last codes and some random ones between, plus
random codes anywhere (`sampled_codes`).

--verify needs the built extension (see README.md, "Installation"); nothing
else here does.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from fractions import Fraction

SATURATIONS = ("OVF_INF", "SAT_FINITE", "SAT_PROPAGATE")
SUBNORMALS = ("SUBNORMALS", "NORMALS", "EXTENDED_NORMALS")


class Entry:
    """One code point: its kind, and, when finite, its value as ``m * 2**e``."""

    __slots__ = ("code", "kind", "m", "e", "negative")

    def __init__(self, code: int, kind: str, m: int = 0, e: int = 0, negative: bool = False):
        self.code = code
        self.kind = kind  # finite | inf | nan | unused
        self.m = m  # significand, an exact integer
        self.e = e  # its power of two
        self.negative = negative

    @property
    def value(self) -> float | None:
        """The value as a float, or None if binary64 cannot hold it."""
        if self.kind == "nan":
            return math.nan
        if self.kind == "inf":
            return -math.inf if self.negative else math.inf
        try:
            v = math.ldexp(float(self.m), self.e)
        except OverflowError:
            return None
        if v == 0.0 and self.m != 0:
            return None  # underflowed binary64, so the decimal would be a lie
        return -v if self.negative else v

    def sort_key(self) -> tuple[float, float]:
        # NaN last, and everything else by value; a None value (outside
        # binary64) still orders by its exact exponent.
        if self.kind == "nan":
            return (2.0, 0.0)
        v = self.value
        if v is not None:
            return (0.0, v)
        sign = -1.0 if self.negative else 1.0
        return (0.0, sign * math.inf)


def decode_magnitude(code: int, P: int, bias: int, subnormals: str) -> tuple[int, int] | None:
    """A magnitude code as ``(m, e)`` with value ``m * 2**e``, or None if unused."""
    man_bits = P - 1
    e_field, t = divmod(code, 1 << man_bits)
    if e_field == 0:
        if subnormals == "SUBNORMALS":
            return (t, 1 - bias - man_bits)
        if subnormals == "EXTENDED_NORMALS":
            # One more binade of normals, at the exponent below the smallest
            # normal's -- what `extended_normals` lowers min_exponent_store
            # for. Not its mantissa-zero code, though: that is the zero, and
            # with the sign bit the NaN, as in every other binaryK format, so
            # the binade's values start one step up. With man_bits == 0 the
            # binade is that one code, and holds nothing else.
            return (0, 0) if t == 0 else ((1 << man_bits) | t, -bias - man_bits)
        # NORMALS: the subnormal codes encode nothing and the range below the
        # smallest normal flushes to zero -- but code 0 is still the zero, the
        # one value every float format spends an exponent-0 code on.
        return (0, 0) if t == 0 else None
    return ((1 << man_bits) | t, e_field - bias - man_bits)


def _top_magnitude(K: int, is_signed: bool) -> tuple[int, int]:
    """The largest magnitude code, and the NaN's code."""
    if is_signed:
        return (1 << (K - 1)) - 1, 1 << (K - 1)  # the NaN is what a binary float reads as -0.0
    return (1 << K) - 2, (1 << K) - 1  # the top code is the NaN


# Past this many code points a format is sampled rather than enumerated.
MAX_ENUMERATED = 1 << 24


def sampled_codes(K: int, P: int, is_signed: bool, per_field: int = 64) -> list[int]:
    """Magnitude codes of a format too wide to enumerate, for ``--verify``.

    Every exponent field's first and last ``per_field`` mantissa codes, where
    the grid meets the binades beside it, and as many random ones -- or, past
    2**12 fields, the fields at both ends and random ones -- plus 2**16 codes
    anywhere. The top code is always among them, so the infinity is too.
    """
    top_magnitude, _ = _top_magnitude(K, is_signed)
    man_bits = P - 1
    exp_bits = K - P if is_signed else K - P + 1
    per = 1 << man_bits
    rng = random.Random(K * 1000 + P)
    n_fields = 1 << exp_bits
    fields = (
        set(range(n_fields))
        if n_fields <= 1 << 12
        else (
            set(range(1 << 11))
            | set(range(n_fields - (1 << 11), n_fields))
            | {rng.randrange(n_fields) for _ in range(1 << 12)}
        )
    )
    codes: set[int] = set()
    for f in fields:
        base = f << man_bits
        codes.update(base + t for t in range(min(per, per_field)))
        codes.update(base + t for t in range(max(0, per - per_field), per))
        codes.update(base + rng.randrange(per) for _ in range(min(per, per_field)))
    codes.update(rng.randrange(top_magnitude + 1) for _ in range(1 << 16))
    return sorted(c for c in codes if c <= top_magnitude)


def enumerate_format(
    K: int,
    P: int,
    bias: int,
    is_signed: bool,
    saturation: str,
    subnormals: str,
    codes: list[int] | None = None,
) -> list[Entry]:
    """Every code point of the format -- or, given ``codes``, those magnitude
    codes and their negatives."""
    extended = saturation != "SAT_FINITE"
    top_magnitude, nan_code = _top_magnitude(K, is_signed)
    inf_magnitude = top_magnitude if extended else None

    entries: list[Entry] = []
    for c in range(top_magnitude + 1) if codes is None else codes:
        if c == inf_magnitude:
            entries.append(Entry(c, "inf"))
            continue
        me = decode_magnitude(c, P, bias, subnormals)
        entries.append(Entry(c, "unused") if me is None else Entry(c, "finite", me[0], me[1]))
    if is_signed:
        # the negatives, in the same order, minus the magnitude-zero code
        for src in list(entries):
            if src.code == 0:
                continue
            if src.kind == "unused":
                entries.append(Entry(nan_code | src.code, "unused"))
            else:
                entries.append(Entry(nan_code | src.code, src.kind, src.m, src.e, negative=True))
    entries.append(Entry(nan_code, "nan"))
    return entries


def held_exactly(m: int, e: int, negative: bool, v: float) -> bool:
    """Whether ``v`` is ``m * 2**e`` (negated) exactly: an entry's ``value``
    rounds one below binary64's subnormal grid without saying so."""
    return Fraction(v) == (-1 if negative else 1) * Fraction(m) * Fraction(2) ** e


def format_exact(entry: Entry) -> str:
    """The hex-float spelling, exact for any exponent, and parseable back."""
    if entry.kind == "nan":
        return "nan"
    if entry.kind == "inf":
        return "-inf" if entry.negative else "inf"
    sign = "-" if entry.negative else ""
    if entry.m == 0:
        return f"{sign}0x0p+0"
    return f"{sign}0x{entry.m:x}p{entry.e:+d}"


def format_value(entry: Entry, style: str) -> tuple[str, bool]:
    """The printed form, and whether binary64 could not hold the value."""
    exact = format_exact(entry)
    if style == "hex":
        return exact, False
    v = entry.value
    dec = exact if v is None else repr(v)
    if style == "both":
        return f"{dec}\t{exact}", v is None
    return dec, v is None


def verify(entries: list[Entry], args: argparse.Namespace) -> int:
    """Quantize every value printed and check the kernel gives it back."""
    import warnings

    import torch

    from mptorch.number import (
        FormatRangeWarning,
        RoundMode,
        SaturationMode,
        SubnormalsMode,
    )
    from mptorch.quant import binaryK_quantize

    # the range warning is the library's view of this format, which is not
    # what is being verified here; the listing's own summary says the rest
    warnings.simplefilter("ignore", FormatRangeWarning)
    dtype = getattr(torch, args.dtype)
    carrier = "binary64" if args.dtype == "float64" else "binary32"
    smallest_normal = 2.0**-1022 if args.dtype == "float64" else 2.0**-126
    finite = [e for e in entries if e.kind == "finite"]
    values, subnormal, skipped = [], [], 0
    for e in finite:
        v = e.value
        if v is None or not math.isfinite(v) or not held_exactly(e.m, e.e, e.negative, v):
            skipped += 1
            continue
        held = float(torch.tensor([v], dtype=dtype).item())
        if held != v:  # the carrier cannot hold it, so the kernel never sees it
            skipped += 1
            continue
        # Below the carrier's smallest normal the cast reads an exponent field
        # every subnormal of it shares, so its idea of the input's binade is
        # one off and its subnormal grid comes out a factor of two coarse.
        # Those values are reported on their own rather than as failures: it is
        # a property of simulating a format in the carrier, not of this format.
        (subnormal if 0.0 < abs(v) < smallest_normal else values).append(v)
    if any(e.kind == "inf" for e in entries):
        values += [math.inf, -math.inf] if args.signed else [math.inf]

    x = torch.tensor(values, dtype=dtype)
    xs = torch.tensor(subnormal, dtype=dtype)
    below = 0
    bad = 0
    for mode in RoundMode:
        if mode is RoundMode.SR:
            continue  # not idempotent by construction: it picks a neighbour at random
        got = binaryK_quantize(
            x,
            args.K,
            args.P,
            bias=args.bias,
            is_signed=args.signed,
            rounding_mode=mode,
            saturation_mode=SaturationMode[args.saturation],
            subnormals_mode=SubnormalsMode[args.subnormals],
        )
        wrong = (got != x) | (torch.signbit(got) != torch.signbit(x))
        if wrong.any():
            bad += int(wrong.sum())
            i = int(wrong.nonzero()[0])
            print(
                f"verify: {mode.name} moved {x[i].item()!r} to {got[i].item()!r}",
                file=sys.stderr,
            )
        if subnormal:
            g = binaryK_quantize(
                xs,
                args.K,
                args.P,
                bias=args.bias,
                is_signed=args.signed,
                rounding_mode=mode,
                saturation_mode=SaturationMode[args.saturation],
                subnormals_mode=SubnormalsMode[args.subnormals],
            )
            below += int(((g != xs) | (torch.signbit(g) != torch.signbit(xs))).sum())
        nan = binaryK_quantize(
            torch.tensor([math.nan], dtype=dtype),
            args.K,
            args.P,
            bias=args.bias,
            is_signed=args.signed,
            rounding_mode=mode,
            saturation_mode=SaturationMode[args.saturation],
            subnormals_mode=SubnormalsMode[args.subnormals],
        )
        if not bool(nan.isnan().item()):
            bad += 1
            print(f"verify: {mode.name} did not keep a NaN", file=sys.stderr)
    print(
        f"verify: {len(values)} values x 6 rounding modes in {carrier}, {skipped} skipped as "
        f"outside it, {bad} wrong",
        file=sys.stderr,
    )
    if subnormal:
        print(
            f"verify: {len(subnormal)} values below {carrier}'s smallest normal, "
            f"where the cast rounds on an exponent field every {args.dtype} subnormal "
            f"shares: {below} of {len(subnormal) * 6} are not fixed points",
            file=sys.stderr,
        )
    return bad


def main() -> int:
    p = argparse.ArgumentParser(
        description="Print every value a BinaryK (IEEE P3109) format represents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples::", 1)[1] if "Examples::" in __doc__ else None,
    )
    p.add_argument("-K", type=int, required=True, help="total bits")
    p.add_argument("-P", type=int, required=True, help="precision, the implicit bit included")
    p.add_argument(
        "--bias",
        type=int,
        default=None,
        help="exponent bias (default: P3109's, 2**(K-P-1) signed and 2**(K-P) unsigned)",
    )
    p.add_argument("--unsigned", action="store_true", help="no sign bit; E = K - P + 1")
    p.add_argument("--saturation", choices=SATURATIONS, default="OVF_INF")
    p.add_argument("--subnormals", choices=SUBNORMALS, default="SUBNORMALS")
    p.add_argument("--order", choices=("value", "code"), default="value")
    p.add_argument("--style", choices=("dec", "hex", "both"), default="dec")
    p.add_argument("--annotate", action="store_true", help="prefix each line with its code point")
    p.add_argument("--unused", action="store_true", help="print the codes that encode nothing")
    p.add_argument("--summary", action="store_true", help="format and counts, on stderr")
    p.add_argument(
        "--verify",
        action="store_true",
        help="round-trip every value through the kernel",
    )
    p.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="the tensor --verify rounds, and so the carrier it rounds in",
    )
    args = p.parse_args()

    if args.P < 1 or args.K < args.P:
        p.error(f"needs 1 <= P <= K, got K={args.K}, P={args.P}")
    args.signed = not args.unsigned
    if args.signed and args.K == args.P:
        p.error("a signed format needs a bit for the sign: P < K")
    exp_bits = args.K - args.P if args.signed else args.K - args.P + 1
    if args.bias is None:
        args.bias = 2 ** (exp_bits - 1)
    sample = (1 << args.K) > MAX_ENUMERATED
    if sample and not args.verify:
        p.error(f"K={args.K} is {2**args.K} code points; this prints them all (--verify samples)")

    entries = enumerate_format(
        args.K,
        args.P,
        args.bias,
        args.signed,
        args.saturation,
        args.subnormals,
        sampled_codes(args.K, args.P, args.signed) if sample else None,
    )
    if sample:
        print(f"{len(entries)} of {2**args.K} code points, sampled", file=sys.stderr)
        return verify(entries, args) and 1
    shown = [e for e in entries if e.kind != "unused" or args.unused]
    if args.order == "value":
        shown.sort(key=Entry.sort_key)

    out, outside64 = [], 0
    for e in shown:
        if e.kind == "unused":
            text = "(unused)"
        else:
            text, over = format_value(e, args.style)
            outside64 += over
        out.append(f"{e.code:#0{(args.K + 3) // 4 + 2}x}\t{text}" if args.annotate else text)
    print("\n".join(out))

    if args.summary or outside64:
        finite = [e for e in entries if e.kind == "finite"]
        mags = [abs(v) for v in (e.value for e in finite) if v not in (None, 0.0)]
        name = f"Binary{args.K}p{args.P}{'s' if args.signed else 'u'}"
        name += "f" if args.saturation == "SAT_FINITE" else "e"
        print(
            f"{name}: E={exp_bits} P-1={args.P - 1} bias={args.bias} "
            f"{args.saturation} {args.subnormals}\n"
            f"  {len(entries)} code points: {len(finite)} finite, "
            f"{sum(e.kind == 'inf' for e in entries)} infinite, "
            f"{sum(e.kind == 'nan' for e in entries)} NaN, "
            f"{sum(e.kind == 'unused' for e in entries)} unused",
            file=sys.stderr,
        )
        if mags:
            print(f"  magnitudes {min(mags)!r} to {max(mags)!r}", file=sys.stderr)
        if outside64:
            print(
                f"  {outside64} values outside binary64, printed in hex",
                file=sys.stderr,
            )
    return verify(entries, args) and 1 if args.verify else 0


if __name__ == "__main__":
    sys.exit(main())
