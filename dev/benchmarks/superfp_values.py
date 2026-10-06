#!/usr/bin/env python3
"""Print every value a SuperFP format represents, one per line.

SuperFP is the second family ``mptorch/csrc/common/cast_superfp.h`` implements
(see ``docs/source/concepts.rst``): a binary float whose top ``normal_binades``
binades carry a mantissa, and whose remaining exponent codes are reinterpreted,
``2**man_bits`` of them per binade, as that many further *powers of two* below
the normal region. There are no subnormals; below the last of those powers
everything flushes to zero.

The enumeration is over code points, and the regions come out of the same two
cutoffs the cast classifies an exponent with
(``superfp_region_cutoffs``), so the boundaries here and the kernel's are the
same arithmetic reached from opposite ends -- ``--verify`` quantizes every
value printed and checks the kernel gives it back unchanged.

With ``2**exp_bits`` exponent codes, ``2**man_bits`` mantissa codes and ``b``
normal binades, writing ``n = (2**exp_bits - b) * 2**man_bits``:

* magnitude code 0 is zero;
* codes 1 to n-1 are the supernormal region, the powers of two from
  ``2**supernormal_cutoff`` up to ``2**(normal_cutoff-1)``;
* codes n and above are the normal region, ``(1 + t*2**-man_bits) *
  2**(e-bias)`` for the top b binades;
* the extended domain (``OVF_INF``, ``SAT_PROPAGATE``) spends the top code on
  an infinity, the finite domain (``SAT_FINITE``) does not.

SuperFP spends no code on a NaN, and none on a negative zero: a signed format's
sign-bit-with-no-magnitude code encodes nothing, which is one of the two
reasons the kernels never return -0.0 (P3109's single unsigned zero is the
other). A NaN still passes through the cast unchanged, since what the kernels
simulate is a value and not an encoding -- ``nan`` is printed for that reason,
with ``(no code point)`` against it under ``--annotate``.

Examples::

    superfp_values.py --man 2 --exp 4 --normal-binades 2 --bias 7
    superfp_values.py --man 3 --exp 4 --normal-binades 1 --bias 7 --verify
    superfp_values.py --man 2 --exp 4 --normal-binades 2 --bias 7 --style both \\
        --order code --annotate
    superfp_values.py --man 30 --exp 4 --normal-binades 15 --bias 7 --verify --dtype float64

``--dtype float64`` verifies through the binary64 kernels instead, and a format
of more than 2**24 code points is sampled rather than printed, for
``--verify`` only (`sampled_codes`), as in ``binaryK_values.py``.

--verify needs the built extension (see README.md, "Installation"); nothing
else here does.
"""

from __future__ import annotations

import argparse
import math
import random
import sys

from binaryK_values import MAX_ENUMERATED, held_exactly

SATURATIONS = ("OVF_INF", "SAT_FINITE", "SAT_PROPAGATE")


class Entry:
    """One code point: its region, and, when finite, its value as ``m * 2**e``."""

    __slots__ = ("code", "kind", "region", "m", "e", "negative")

    def __init__(
        self,
        code: int,
        kind: str,
        region: str = "",
        m: int = 0,
        e: int = 0,
        negative: bool = False,
    ):
        self.code = code
        self.kind = kind  # finite | inf | nan | unused
        self.region = region  # zero | supernormal | normal
        self.m = m
        self.e = e
        self.negative = negative

    @property
    def value(self) -> float | None:
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
        if self.kind == "nan":
            return (2.0, 0.0)
        v = self.value
        if v is not None:
            return (0.0, v)
        sign = -1.0 if self.negative else 1.0
        return (0.0, sign * math.inf)


def region_cutoffs(man_bits: int, exp_bits: int, normal_binades: int, bias: int) -> tuple[int, int]:
    """superfp_region_cutoffs, transcribed: the normal and supernormal floors."""
    normal_cutoff = ((1 << exp_bits) - 1 - bias) - normal_binades + 1
    supernormal_cutoff = normal_cutoff - ((1 << exp_bits) - normal_binades) * (1 << man_bits) + 1
    return normal_cutoff, supernormal_cutoff


def sampled_codes(man_bits: int, exp_bits: int, normal_binades: int, per: int = 64) -> list[int]:
    """Magnitude codes of a format too wide to enumerate, for ``--verify``:
    the first and last 2**12 supernormal codes and 2**12 random ones, each
    normal field's first and last ``per`` codes and ``per`` random ones, and
    2**16 codes anywhere. The top code, and so the infinity, is among them."""
    top_magnitude = (1 << (exp_bits + man_bits)) - 1
    supernormal_codes = ((1 << exp_bits) - normal_binades) << man_bits
    rng = random.Random(man_bits * 1000 + exp_bits)
    n = 1 << 12
    codes = set(range(min(supernormal_codes, n)))
    codes.update(range(max(0, supernormal_codes - n), supernormal_codes))
    codes.update(rng.randrange(max(1, supernormal_codes)) for _ in range(n))
    width = 1 << man_bits
    fields = (
        range(normal_binades)
        if normal_binades <= n
        else (list(range(n // 2)) + list(range(normal_binades - n // 2, normal_binades)))
    )
    for f in fields:
        base = supernormal_codes + (f << man_bits)
        codes.update(base + t for t in range(min(width, per)))
        codes.update(base + t for t in range(max(0, width - per), width))
        codes.update(base + rng.randrange(width) for _ in range(min(width, per)))
    codes.update(rng.randrange(top_magnitude + 1) for _ in range(1 << 16))
    return sorted(c for c in codes if c <= top_magnitude)


def enumerate_format(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    is_signed: bool,
    saturation: str,
    codes: list[int] | None = None,
) -> list[Entry]:
    """Every code point of the format -- or, given ``codes``, those magnitude
    codes and their negatives."""
    normal_cutoff, supernormal_cutoff = region_cutoffs(man_bits, exp_bits, normal_binades, bias)
    supernormal_codes = ((1 << exp_bits) - normal_binades) << man_bits
    top_magnitude = (1 << (exp_bits + man_bits)) - 1
    inf_magnitude = top_magnitude if saturation != "SAT_FINITE" else None

    entries: list[Entry] = []
    for c in range(top_magnitude + 1) if codes is None else codes:
        if c == inf_magnitude:
            entries.append(Entry(c, "inf"))
        elif c == 0:
            entries.append(Entry(c, "finite", "zero", 0, 0))
        elif c < supernormal_codes:
            # the powers of two, from 2**supernormal_cutoff upwards
            entries.append(Entry(c, "finite", "supernormal", 1, supernormal_cutoff + c - 1))
        else:
            e_field, t = divmod(c - supernormal_codes, 1 << man_bits)
            e_field += (1 << exp_bits) - normal_binades
            entries.append(
                Entry(c, "finite", "normal", (1 << man_bits) | t, e_field - bias - man_bits)
            )
    if is_signed:
        for src in list(entries):
            if src.code == 0:
                continue
            entries.append(
                Entry(
                    (1 << (exp_bits + man_bits)) | src.code,
                    src.kind,
                    src.region,
                    src.m,
                    src.e,
                    negative=True,
                )
            )
        # the sign bit with no magnitude: superfp has no -0.0 and no NaN code
        entries.append(Entry(1 << (exp_bits + man_bits), "unused"))
    entries.append(Entry(-1, "nan"))  # a value the cast keeps, with no encoding
    return entries


def format_exact(entry: Entry) -> str:
    if entry.kind == "nan":
        return "nan"
    if entry.kind == "inf":
        return "-inf" if entry.negative else "inf"
    sign = "-" if entry.negative else ""
    if entry.m == 0:
        return f"{sign}0x0p+0"
    return f"{sign}0x{entry.m:x}p{entry.e:+d}"


def format_value(entry: Entry, style: str) -> tuple[str, bool]:
    exact = format_exact(entry)
    if style == "hex":
        return exact, False
    v = entry.value
    dec = exact if v is None else repr(v)
    if style == "both":
        return f"{dec}\t{exact}", v is None
    return dec, v is None


def verify(entries: list[Entry], args: argparse.Namespace) -> int:
    import warnings

    import torch

    from mptorch.number import FormatRangeWarning, RoundMode, SaturationMode
    from mptorch.quant import superfp_quantize

    warnings.simplefilter("ignore", FormatRangeWarning)  # as in binaryK_values.verify
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
        # Below the carrier's smallest normal the supernormal arm has no
        # exponent to round: every subnormal of it carries the same exponent
        # field, so a format whose supernormal region reaches down there has
        # code points the cast cannot land on. Reported on its own rather than
        # as a failure -- it is a property of simulating in the carrier.
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

        # one closure rather than a kwargs dict: a dict's value type is the
        # union of every argument's, which the type checker then refuses at
        # each parameter in turn
        def quant(t, mode=mode):
            return superfp_quantize(
                t,
                args.man,
                args.exp,
                args.normal_binades,
                args.bias,
                is_signed=args.signed,
                rounding_mode=mode,
                saturation_mode=SaturationMode[args.saturation],
            )

        got = quant(x)
        wrong = (got != x) | (torch.signbit(got) != torch.signbit(x))
        if wrong.any():
            bad += int(wrong.sum())
            i = int(wrong.nonzero()[0])
            print(
                f"verify: {mode.name} moved {x[i].item()!r} to {got[i].item()!r}",
                file=sys.stderr,
            )
        if subnormal:
            g = quant(xs)
            below += int(((g != xs) | (torch.signbit(g) != torch.signbit(xs))).sum())
        nan = quant(torch.tensor([math.nan], dtype=dtype))
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
            f"where every {args.dtype} subnormal shares one exponent field: "
            f"{below} of {len(subnormal) * 6} are not fixed points",
            file=sys.stderr,
        )
    return bad


def main() -> int:
    p = argparse.ArgumentParser(
        description="Print every value a SuperFP format represents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples::", 1)[1] if "Examples::" in __doc__ else None,
    )
    p.add_argument("--man", "--man-bits", type=int, required=True, help="stored mantissa bits")
    p.add_argument("--exp", "--exp-bits", type=int, required=True, help="exponent bits")
    p.add_argument(
        "--normal-binades",
        type=int,
        required=True,
        help="how many top binades keep a mantissa; the rest become powers of two",
    )
    p.add_argument("--bias", type=int, required=True, help="exponent bias (no default)")
    p.add_argument("--unsigned", action="store_true", help="no sign bit")
    p.add_argument("--saturation", choices=SATURATIONS, default="OVF_INF")
    p.add_argument("--order", choices=("value", "code"), default="value")
    p.add_argument("--style", choices=("dec", "hex", "both"), default="dec")
    p.add_argument("--annotate", action="store_true", help="prefix each line with its code point")
    p.add_argument("--unused", action="store_true", help="print the codes that encode nothing")
    p.add_argument("--summary", action="store_true", help="format, regions and counts, on stderr")
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

    if args.man < 0 or args.exp < 1:
        p.error(f"needs man_bits >= 0 and exp_bits >= 1, got man={args.man}, exp={args.exp}")
    if not 1 <= args.normal_binades <= (1 << args.exp):
        p.error(f"needs 1 <= normal_binades <= {1 << args.exp}, got {args.normal_binades}")
    sample = (1 << (args.man + args.exp)) > MAX_ENUMERATED
    if sample and not args.verify:
        p.error(
            f"{args.man + args.exp} magnitude bits is {2 ** (args.man + args.exp)} code points "
            f"(--verify samples)"
        )
    args.signed = not args.unsigned

    normal_cutoff, supernormal_cutoff = region_cutoffs(
        args.man, args.exp, args.normal_binades, args.bias
    )
    if supernormal_cutoff > normal_cutoff:
        # normal_binades == 2**exp_bits leaves no code for the supernormal
        # region, and the cast's region test then inverts: `underflow` claims
        # every exponent at or below normal_cutoff, so the lowest normal binade
        # is flushed to zero and those code points are values the kernel never
        # produces. The listing below is the format's; --verify will say so.
        print(
            f"warning: normal_binades={args.normal_binades} leaves no supernormal "
            f"codes, and the cast flushes the binade below 2**{normal_cutoff + 1} "
            f"to zero rather than keeping it normal",
            file=sys.stderr,
        )
    entries = enumerate_format(
        args.man,
        args.exp,
        args.normal_binades,
        args.bias,
        args.signed,
        args.saturation,
        sampled_codes(args.man, args.exp, args.normal_binades) if sample else None,
    )
    if sample:
        print(f"{len(entries) - 1} code points, sampled", file=sys.stderr)
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
        if args.annotate:
            width = (args.exp + args.man + 4) // 4 + 2
            code = "(no code point)" if e.code < 0 else f"{e.code:#0{width}x}"
            out.append(f"{code}\t{e.region or e.kind}\t{text}")
        else:
            out.append(text)
    print("\n".join(out))

    if args.summary or outside64:
        finite = [e for e in entries if e.kind == "finite"]
        mags = [abs(v) for v in (e.value for e in finite) if v not in (None, 0.0)]
        print(
            f"SuperFP m{args.man}e{args.exp}n{args.normal_binades}b{args.bias}"
            f"{'' if args.signed else ' unsigned'} {args.saturation}\n"
            f"  normal region 2**{normal_cutoff} and up, supernormal region "
            f"2**{supernormal_cutoff} to 2**{normal_cutoff - 1}, below that zero\n"
            f"  {len(entries) - 1} code points: "
            f"{sum(e.kind == 'finite' for e in entries)} finite "
            f"({sum(e.region == 'supernormal' for e in entries)} supernormal, "
            f"{sum(e.region == 'normal' for e in entries)} normal), "
            f"{sum(e.kind == 'inf' for e in entries)} infinite, "
            f"{sum(e.kind == 'unused' for e in entries)} unused, no NaN code",
            file=sys.stderr,
        )
        if mags:
            print(f"  magnitudes {min(mags)!r} to {max(mags)!r}", file=sys.stderr)
        if outside64:
            print(f"  {outside64} values outside binary64, printed in hex", file=sys.stderr)
    return verify(entries, args) and 1 if args.verify else 0


if __name__ == "__main__":
    sys.exit(main())
