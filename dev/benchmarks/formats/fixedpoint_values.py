#!/usr/bin/env python3
"""Print every value a FixedPoint format represents, one per line.

``FixedPoint(wl, fl, is_signed, symmetric)`` (``docs/source/concepts.rst``)
holds the multiples ``n * 2**-fl`` of its step that a ``wl``-bit integer word
holds: ``n`` in ``[-2**(wl-1), 2**(wl-1) - 1]`` for two's complement, without
the bottom code when ``symmetric``, and ``[0, 2**wl - 1]`` unsigned. There are
no reserved codes -- no infinity, no NaN, no second zero -- so the enumeration
is the integer range itself. That is a different derivation from the one the
kernels use -- they round a carrier value onto the step's grid on its word and
never look at an integer -- which is what makes ``--verify`` worth running: it
quantizes every value this script prints in the six deterministic rounding
modes and checks that the kernel gives it back unchanged, that an infinity
saturates to the end of the range on its side, and that a NaN passes through.

Examples::

    fixedpoint_values.py --wl 4 --fl 2                    # -2 .. 1.75, step 0.25
    fixedpoint_values.py --wl 4 --fl 2 --symmetric --annotate
    fixedpoint_values.py --wl 8 --fl -3 --unsigned --style both
    fixedpoint_values.py --wl 16 --fl 14 --verify
    fixedpoint_values.py --wl 40 --fl 30 --verify --dtype float64

``--annotate`` prefixes each value with its word, in hex: two's complement for
a signed format, which is what a fixed-point register holds. A format of more
than 2**24 codes is not printed but sampled, for ``--verify`` only: the bottom
and top 4096 codes and 2**16 random ones between.

--verify needs the built extension (see README.md, "Installation"); nothing
else here does.
"""

from __future__ import annotations

import argparse
import math
import random
import sys

MAX_ENUMERATED = 1 << 24
EDGE = 4096


def code_range(wl: int, signed: bool, symmetric: bool) -> tuple[int, int]:
    """The integers the word holds, both ends included."""
    if not signed:
        return 0, (1 << wl) - 1
    top = (1 << (wl - 1)) - 1
    return (-top if symmetric else -top - 1), top


def codes(lo: int, hi: int, sample: bool) -> list[int]:
    if not sample:
        return list(range(lo, hi + 1))
    rng = random.Random(0x5EED)
    picked = set(range(lo, lo + EDGE)) | set(range(hi - EDGE + 1, hi + 1))
    picked |= {rng.randint(lo, hi) for _ in range(1 << 16)}
    picked |= {c for c in (0, 1, -1) if lo <= c <= hi}  # never a set of the whole range
    return sorted(picked)


def format_value(n: int, fl: int, style: str) -> str:
    """Decimal (Python's shortest repr of the binary64 value), the exact
    hex-float spelling, or both."""
    exact = f"{'-' if n < 0 else ''}0x{abs(n):x}p{-fl:+d}" if n else "0x0p+0"
    if style == "hex":
        return exact
    try:
        dec = repr(math.ldexp(float(n), -fl))
    except OverflowError:
        dec = exact
    return f"{dec}\t{exact}" if style == "both" else dec


def verify(values: list[int], args: argparse.Namespace) -> int:
    """Quantize every value printed and check the kernel gives it back."""
    import warnings

    import torch

    from mptorch.number import FormatRangeWarning, RoundMode
    from mptorch.quant import fixedpoint_quantize

    warnings.simplefilter("ignore", FormatRangeWarning)
    dtype = getattr(torch, args.dtype)
    carrier = "binary64" if args.dtype == "float64" else "binary32"
    # every value in binary64, then the ones the carrier holds (the rest the
    # kernel never sees); one tensor each, not one per value
    exact = torch.tensor([math.ldexp(float(n), -args.fl) for n in values], dtype=torch.float64)
    in_carrier = exact.to(dtype).to(torch.float64) == exact
    held = exact[in_carrier].tolist()
    skipped = int((~in_carrier).sum())
    lo, hi = code_range(args.wl, args.signed, args.symmetric)
    top, bottom = math.ldexp(float(hi), -args.fl), math.ldexp(float(lo), -args.fl)
    x = torch.tensor(held + [math.inf, -math.inf, math.nan], dtype=dtype)
    want = torch.tensor(held + [top, bottom if args.signed else 0.0, math.nan], dtype=dtype)

    bad = 0
    for mode in RoundMode:
        if mode is RoundMode.SR:
            continue  # not idempotent by construction: it picks a neighbour at random
        got = fixedpoint_quantize(
            x, args.wl, args.fl, is_signed=args.signed, symmetric=args.symmetric, rounding_mode=mode
        )
        same = (got == want) & (torch.signbit(got) == torch.signbit(want))
        same |= torch.isnan(got) & torch.isnan(want)
        if not bool(same.all()):
            wrong = ~same
            bad += int(wrong.sum())
            i = int(wrong.nonzero()[0])
            print(
                f"verify: {mode.name} moved {x[i].item()!r} to {got[i].item()!r}", file=sys.stderr
            )
    print(
        f"verify: {len(held)} values, two infinities and a NaN x 6 rounding modes in {carrier}, "
        f"{skipped} skipped as outside it, {bad} wrong",
        file=sys.stderr,
    )
    return bad


def main() -> int:
    p = argparse.ArgumentParser(
        description="Print every value a FixedPoint format represents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples::", 1)[1] if __doc__ and "Examples::" in __doc__ else None,
    )
    p.add_argument("--wl", type=int, required=True, help="word length, the sign bit included")
    p.add_argument("--fl", type=int, required=True, help="fractional bits: the step is 2**-fl")
    p.add_argument("--unsigned", action="store_true", help="no sign bit")
    p.add_argument("--symmetric", action="store_true", help="drop the two's complement bottom code")
    p.add_argument("--style", choices=("dec", "hex", "both"), default="dec")
    p.add_argument("--annotate", action="store_true", help="prefix each line with its word")
    p.add_argument(
        "--verify", action="store_true", help="round-trip every value through the kernel"
    )
    p.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="the tensor --verify rounds, and so the carrier it rounds in",
    )
    args = p.parse_args()
    args.signed = not args.unsigned
    if args.wl < (2 if args.signed else 1):
        least = 2 if args.signed else 1
        p.error(f"a {'signed' if args.signed else 'unsigned'} format needs wl >= {least}")
    if args.symmetric and not args.signed:
        p.error("--symmetric is a signed format's option")

    lo, hi = code_range(args.wl, args.signed, args.symmetric)
    sample = hi - lo + 1 > MAX_ENUMERATED
    if sample and not args.verify:
        p.error(f"wl={args.wl} is {hi - lo + 1} codes; this prints them all (--verify samples)")
    values = codes(lo, hi, sample)
    if sample:
        print(f"{len(values)} of {hi - lo + 1} codes, sampled", file=sys.stderr)
        return verify(values, args) and 1

    width = (args.wl + 3) // 4
    mask = (1 << args.wl) - 1
    out = []
    for n in values:
        text = format_value(n, args.fl, args.style)
        out.append(f"{n & mask:#0{width + 2}x}\t{text}" if args.annotate else text)
    print("\n".join(out))
    if args.verify:
        return verify(values, args) and 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
