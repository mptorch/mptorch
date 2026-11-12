"""Statistical gate for the elementwise ``RoundMode.SR`` path.

Commit 64982e6 replaced the int32 tensor of random draws that
``binaryK_quantize`` / ``superfp_quantize`` used to materialize with a Philox
stream generated inside the kernel, keyed on each element's linear index.
Unlike the kernel optimizations made around it, it does not preserve outputs
bit for bit -- every element gets a different random value than it did -- so
the gate for it cannot be a byte comparison. These are the properties that
have to hold instead, checked on both backends:

    A  exactly representable values survive SR untouched
    B  SR(x) is always RD(x) or RU(x), tail elements included
    C  the round-up rate tracks the fractional position, 31 points, 5 sigma
    D  no bias by position mod 4 or mod 8 -- four consecutive elements share
       one Philox block, so a wrong word mapping shows up here and nowhere else
    E  no autocorrelation between neighbouring elements
    F  successive calls draw different, uncorrelated values
    G  torch.manual_seed still reproduces a call exactly
    H  the rounding is unbiased over a broad random input
    I  prng_bits still narrows the draw
    J  a CPU result does not depend on torch.set_num_threads
    K  CPU and CUDA agree on the distribution (they do not, and need not,
       agree element by element -- different generators, as before)

    python3 dev/benchmarks/sweeps/elementwise_sr_stats.py
    python3 dev/benchmarks/sweeps/elementwise_sr_stats.py --dtype float64

``--dtype float64`` runs every property on float64 tensors, which are rounded
in binary64 and draw two Philox words an element -- so two elements share a
block, and D checks position mod 2 as well -- twice: with 30 random bits, and
with the widest draw binary64 leaves below the format's mantissa.

Exits non-zero on the first property that fails, with the measured value and
the tolerance it missed.
"""

import argparse
import math
import sys

import torch

from mptorch import RoundMode
from mptorch.quant import binaryK_quantize, superfp_quantize

FAILURES: list[str] = []
CHECKS = 0

# The tensors C to K round, and what their draw is made in: set by --dtype.
DTYPE = torch.float32
CARRIER_MAN_BITS = 23
PRNG_CAP: int | None = None  # a draw narrower than the widest, when set
LABEL = ""

# e4m3 near 1.0: the neighbouring representable values are 1.0 and 1.125, so
# 1 + f * 0.125 sits a fraction f of the way between them and must round up
# with probability f.
BK_LO, BK_STEP = 1.0, 0.125


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    name = LABEL + name
    if not ok:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}  {detail}")


def full_prng(man_bits: int) -> int:
    """The widest draw the cast will use.

    ``prng_bits`` counts random bits below the kept mantissa;
    ``round_bitwise_stochastic`` masks the draw to the field being discarded,
    so 0 makes SR a plain truncation and every check here would pass
    vacuously. The draw is made in the value the kernel rounds, so the widest
    is the carrier's mantissa less the format's: binary32's 23 bits for
    float32, float16 and bfloat16, binary64's 52 for float64 -- capped under
    ``PRNG_CAP`` when it is set.
    """
    width = CARRIER_MAN_BITS - man_bits
    return width if PRNG_CAP is None else min(width, PRNG_CAP)


def q_bk(x: torch.Tensor, rm: RoundMode, prng_bits: int | None = None) -> torch.Tensor:
    if prng_bits is None:
        prng_bits = full_prng(3)
    return binaryK_quantize(x, 8, 4, bias=7, prng_bits=prng_bits, rounding_mode=rm, is_signed=True)


def q_sfp(x: torch.Tensor, rm: RoundMode, prng_bits: int | None = None) -> torch.Tensor:
    if prng_bits is None:
        prng_bits = full_prng(2)
    return superfp_quantize(
        x,
        2,
        5,
        normal_binades=1,
        bias=15,
        prng_bits=prng_bits,
        rounding_mode=rm,
        is_signed=True,
    )


FAMILIES = {"binaryK e4m3": q_bk, "superfp m2e5": q_sfp}
DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]


def run(device: str) -> None:
    print(f"\n=== {device} {LABEL} ===")
    dtypes = DTYPES if DTYPE is torch.float32 else [DTYPE]

    for fam, quant in FAMILIES.items():
        for dtype in dtypes:
            x = torch.randn(100_003, dtype=dtype, device=device) * 0.5
            grid = quant(x, RoundMode.RD)
            out = quant(grid, RoundMode.SR)
            check(
                f"A grid-identity {fam} {dtype}",
                torch.equal(out, grid),
                f"{(out != grid).sum().item()} of {grid.numel()} moved",
            )

    for fam, quant in FAMILIES.items():
        for dtype in dtypes:
            # sizes on and off the vector width, down to the tail-only cases
            for n in (100_003, 33, 8, 7, 4, 3, 2, 1):
                x = torch.randn(n, dtype=dtype, device=device) * 0.5
                rd, ru = quant(x, RoundMode.RD), quant(x, RoundMode.RU)
                sr = quant(x, RoundMode.SR)
                check(f"B bounds {fam} {dtype} n={n}", bool(((sr == rd) | (sr == ru)).all()))

    n = 1 << 20
    for f in [i / 32 for i in range(1, 32)]:
        x = torch.full((n,), BK_LO + f * BK_STEP, dtype=DTYPE, device=device)
        rate = (q_bk(x, RoundMode.SR) > BK_LO).float().mean().item()
        sigma = math.sqrt(f * (1 - f) / n)
        check(
            f"C round-up rate f={f:.4f}",
            abs(rate - f) < 5 * sigma,
            f"measured {rate:.6f}, want {f:.6f} (5 sigma = {5 * sigma:.6f})",
        )

    n = 1 << 22
    x = torch.full((n,), BK_LO + 0.5 * BK_STEP, dtype=DTYPE, device=device)
    up = (q_bk(x, RoundMode.SR) > BK_LO).float()
    for m in (4, 8) if DTYPE is torch.float32 else (2, 4, 8):
        rates = [up.view(-1, m)[:, k].mean().item() for k in range(m)]
        sigma = math.sqrt(0.25 / (n / m))
        worst = max(abs(r - 0.5) for r in rates)
        check(
            f"D lane uniformity mod {m}",
            worst < 5 * sigma,
            f"worst deviation {worst:.6f} (5 sigma = {5 * sigma:.6f})",
        )

    centered = up - up.mean()
    denom = (centered * centered).sum().item()
    check("E round-up indicator varies", denom > 0.0)
    for lag in (1, 2, 3, 4, 5, 17):
        r = (centered[:-lag] * centered[lag:]).sum().item() / max(denom, 1e-30)
        check(
            f"E autocorrelation lag {lag}",
            abs(r) < 5 / math.sqrt(n),
            f"r = {r:.6f} (bound {5 / math.sqrt(n):.6f})",
        )

    n = 1 << 16
    x = torch.full((n,), BK_LO + 0.5 * BK_STEP, dtype=DTYPE, device=device)
    first, second = q_bk(x, RoundMode.SR), q_bk(x, RoundMode.SR)
    check("F successive calls differ", not torch.equal(first, second))
    agreement = ((first > BK_LO) == (second > BK_LO)).float().mean().item()
    check(
        "F successive calls uncorrelated",
        abs(agreement - 0.5) < 5 * math.sqrt(0.25 / n),
        f"agreement {agreement:.5f}",
    )

    for fam, quant in FAMILIES.items():
        for dtype in dtypes:
            x = torch.randn(50_001, dtype=dtype, device=device) * 0.5
            outs = []
            for _ in range(2):
                torch.manual_seed(9871)
                if device == "cuda":
                    torch.cuda.manual_seed_all(9871)
                outs.append(quant(x, RoundMode.SR))
            check(f"G reproducible {fam} {dtype}", torch.equal(outs[0], outs[1]))

    for fam, quant in FAMILIES.items():
        x = torch.randn(1 << 22, dtype=DTYPE, device=device) * 0.3
        err = (quant(x, RoundMode.SR).double() - x.double()).mean().item()
        scale = (quant(x, RoundMode.RD).double() - x.double()).abs().mean().item()
        check(
            f"H unbiased {fam}",
            abs(err) < 0.02 * max(scale, 1e-12),
            f"mean error {err:.3e} against mean |RD error| {scale:.3e}",
        )

    for prng_bits in (1, 2, 3, 8):
        x = torch.full((1 << 18,), BK_LO + 0.5 * BK_STEP, dtype=DTYPE, device=device)
        rate = (q_bk(x, RoundMode.SR, prng_bits=prng_bits) > BK_LO).float().mean().item()
        check(f"I prng_bits={prng_bits} at f=0.5", abs(rate - 0.5) < 0.02, f"{rate:.5f}")


def run_thread_independence() -> None:
    print("\n=== cpu thread-count independence ===")
    restore = torch.get_num_threads()
    x = torch.randn(200_003, dtype=DTYPE) * 0.5
    results = {}
    for threads in (1, 2, 8):
        torch.set_num_threads(threads)
        torch.manual_seed(4242)
        results[threads] = q_bk(x, RoundMode.SR)
    torch.set_num_threads(restore)
    check("J threads 1 vs 2", torch.equal(results[1], results[2]))
    check("J threads 1 vs 8", torch.equal(results[1], results[8]))


def run_backend_agreement() -> None:
    print("\n=== cpu/cuda distribution agreement ===")
    n = 1 << 20
    for f in (0.25, 0.5, 0.75):
        x = torch.full((n,), BK_LO + f * BK_STEP, dtype=DTYPE)
        on_cpu = (q_bk(x, RoundMode.SR) > BK_LO).float().mean().item()
        on_gpu = (q_bk(x.cuda(), RoundMode.SR) > BK_LO).float().mean().item()
        sigma = math.sqrt(f * (1 - f) / n)
        check(
            f"K cpu/cuda f={f}",
            abs(on_cpu - on_gpu) < 7 * sigma,
            f"cpu {on_cpu:.6f}, cuda {on_gpu:.6f}",
        )


def main() -> int:
    global DTYPE, CARRIER_MAN_BITS, PRNG_CAP, LABEL
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = p.parse_args()
    caps: list[int | None] = [None]
    if args.dtype == "float64":
        DTYPE, CARRIER_MAN_BITS, caps = torch.float64, 52, [30, None]
    for cap in caps:
        PRNG_CAP = cap
        if args.dtype == "float64":
            LABEL = f"[float64 prng={full_prng(3)}] "
        run("cpu")
        if torch.cuda.is_available():
            run("cuda")
        run_thread_independence()
        if torch.cuda.is_available():
            run_backend_agreement()
    print(f"\n{CHECKS} checks, {len(FAILURES)} failed")
    for failure in FAILURES:
        print("  " + failure)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
