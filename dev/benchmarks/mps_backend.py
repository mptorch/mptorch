"""The MPS backend (``mptorch/csrc/mps/``), measured and swept against the CPU.

    python3 dev/benchmarks/mps_backend.py bench     # every op on MPS against the CPU op
    python3 dev/benchmarks/mps_backend.py bench-beyond  # the ops past the NAIVE GEMMs only
    python3 dev/benchmarks/mps_backend.py sweep     # the quantizers over all 2^32 binary32 words
    python3 dev/benchmarks/mps_backend.py control   # the sweep can see a contracted split
    python3 dev/benchmarks/mps_backend.py odd-probe  # what the odd step's Metal primitives assume
    python3 dev/benchmarks/mps_backend.py odd-sweep [SEED]  # those primitives against the CPU's
    python3 dev/benchmarks/mps_backend.py bench-odd  # the odd ops against their default twins
    python3 dev/benchmarks/mps_backend.py bench-counted  # the counted ops against their twins

``sweep`` runs ``binaryK_quantize``, ``superfp_quantize`` and
``fixedpoint_quantize`` on an MPS tensor and on a CPU one over every binary32
word, for each format in BINARYK, SUPERFP and FIXEDPOINT and each rounding mode
(SR from the same seed on both), and counts the
words that differ, the 2^24 subnormal inputs separately. Every count must be
zero: the casts round on the word, and bit_helper.h spells the few operations
they apply to the value itself so that the GPU's flush of binary32
subnormals cannot reach them. Before the SR add was spelled that way, this
sweep counted 283 mismatches, all on subnormal inputs, in three superfp SR
cases. ~6 minutes on an M3 Pro.

``control`` is the sweep's negative control. It compiles the elementwise
kernel through ``torch.mps.compile_shader`` from the backend's own sources
twice, once with the RNE fast path's Veltkamp split replaced by an explicit
FMA (which a contracting compiler would make of it) and once under
``math_mode(fast)``, and sweeps binary8p4 RNE through both. Both must
mismatch, or a clean sweep proves nothing.

``bench`` times each op on MPS against the CPU op on the same operands, and
checks that the results agree (up to a NaN's payload, which the GPU's float
unit canonicalizes in the GEMMs). Each instantiation is compiled before it is
timed.

Measured on an M3 Pro (18-core GPU, torch 2.14, macOS 26), CPU / MPS in ms,
the CPU at torch's default 5 threads of 11, formats of 8 to 12 bits:

    op                   [64,784]@[784,512]    512^3            1024^3
    binaryK              17.7 / 0.73            86 / 2.4        685 / 11.3
    superfp              20.6 / 1.26           102 / 3.6        807 / 20.4
    binaryK_fma           8.6 / 0.94            42 / 1.7        336 / 6.9
    superfp_fma           9.5 / 0.72            47 / 2.5        393 / 11.5
    binaryK_mixed        17.9 / 1.94            87 / 4.1        697 / 19.7
    superfp_mixed        20.7 / 2.98           101 / 3.8        805 / 24.4
    binaryK_fma_mixed     8.7 / 1.87            42 / 3.5        333 / 12.6
    superfp_fma_mixed    18.6 / 2.13            94 / 4.3        714 / 14.8

binaryK at 2048^3 is 5508 / 88.5 ms. The quantizers break even with the CPU
at 2^20 elements (0.3 ms either way) and run 4.3x (RNE) to 9.1x (SR) faster
at 2^25 (2.2-2.5 ms). The first column is mostly launch and synchronization;
at [64,784]@[784,512], min of 30, binaryK_fma is 0.30 ms and binaryK_mixed
0.66. Each format is compiled into its kernel because reading the same
constants from a buffer measured 1.7-1.8x slower (csrc/mps/gemm.metal),
which is also about what a mixed op costs over its single-format twin on the
same formats, 1.2-1.8x at 1024^3: its policy is chosen per element, so its
constants are run-time values.

``bench-beyond`` (the end of ``bench`` too) times the ops past the NAIVE
GEMMs, CPU / MPS in ms (the CPU column drifts by up to 25% in an hour):

    binaryK split mac at 1024^3, RNE / SR   NAIVE 877 / 11.2, 4072 / 60.4
                                            KAHAN 3074 / 24.1, 7755 / 223
                                            BLOCK 32 911 / 11.8, 3171 / 89.0
                                            TREE 64 1216 / 15.9, 2585 / 99.2
    conv 3x3 [8,64,32,32], fwd / igrad / wgrad   250 / 3.9, 255 / 4.3, 254 / 4.6
    block_pack 4096^2, MXFP8 / NVFP4             26 / 1.5, 45 / 2.7
    block_pack 4096^2, 128x128 tiles,            16 / 1.9, 99 / 6.5
      MXFP8 / E2M1 under Four Over Six MSE
    block_matmul MXFP8 1024^3                    456 / 11.5

NVFP4's pack took 26.5 ms on MPS until its default tensor scale stopped
using ``torch.linalg.vector_norm(x, inf)``, which is 25 ms over 2^24 float32
elements there, for ``torch.aminmax`` (1.5 ms).

The 128 x 128 rows took 16.5 and 73 ms on MPS while the pack gave every
tile one thread; a call with few tiles for their size now gives each a
threadgroup (csrc/mps/block_entry.cpp's run_pack has the rule).

``bench-counted`` times each counted op (gemm.metal under MPT_COUNTING)
against its twin on MPS, the two alternated. On the M3 Pro, at 512^3 and
1024^3, a counted op took its twin's time times:

    binaryK split mac, NAIVE / KAHAN / BLOCK 32 / TREE 64   RNE 2.5-3.0x, SR 1.21-1.25x
    binaryK fused mac, NAIVE                                RNE 2.2-2.4x, SR 1.22-1.23x
    superfp split mac, NAIVE RNE                            1.83-1.88x
    fixed point, RNE: carrier path / integer / fused        3.6-4.3x / 1.54-1.57x / 1.21-1.22x
    conv 3x3 [8,64,32,32] fwd, binaryK split RNE            2.3-2.4x

Each cast's classification and count is integer work of a fixed size,
which weighs most beside the cheapest casts (binaryK's RNE fast path, the
fixed-point carrier path) and least beside SR's draws.

The GEMM kernel stages 16 x 16 tiles of A and B in threadgroup memory
(gemm.metal says what that bought and what it did not). Before it did, one
thread per element read its operands from device memory and the 1024^3 row
read 12.2 / 8.7 / 29.8 / 17.1 ms for binaryK / binaryK_fma / binaryK_mixed
/ binaryK_fma_mixed, and a split binaryK mac under SR 82 ms (60 now).
"""

import dataclasses
import re
import struct
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from mptorch import MXFP8_E4M3, NVFP4, AccumulateAlgorithm, BinaryK, FixedPoint, ScaleRounding
from mptorch.number import FormatRangeWarning, RoundMode, SaturationMode, SubnormalsMode, SuperFP
from mptorch.quant import (
    FusedMac,
    SplitMac,
    binaryK_quantize,
    binaryK_quantize_,
    block_matmul,
    block_pack,
    block_quantize,
    conv_formats,
    fixedpoint_quantize,
    superfp_quantize,
)
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_fma_mixed,
    binaryK_matmul_mixed,
    binaryK_superfp_matmul,
    superfp_matmul,
    superfp_matmul_fma,
    superfp_matmul_fma_mixed,
    superfp_matmul_mixed,
)
from mptorch.quant.ops._conv import _FWD, _ConvOp
from mptorch.quant.ops._counted import _counted_conv_twin, _counted_twin
from mptorch.quant.ops._spec import _run_gemm

CSRC = Path(__file__).resolve().parents[2] / "mptorch" / "csrc"
CHUNK = 1 << 26

# (K, P, saturation, subnormals, is_signed): 8-bit formats, a 6-, a 12- and a
# 16-bit one, an unsigned one, the edges of the binaryK fast path's gate
# (P = 1 and SAT_PROPAGATE leave it off), and every saturation mode.
BINARYK = [
    (8, 4, SaturationMode.SAT_NONE, SubnormalsMode.SUBNORMALS, True),
    (8, 3, SaturationMode.SAT_NONE, SubnormalsMode.SUBNORMALS, True),
    (8, 5, SaturationMode.SAT_FINITE, SubnormalsMode.SUBNORMALS, True),
    (8, 1, SaturationMode.SAT_NONE, SubnormalsMode.SUBNORMALS, True),
    (8, 4, SaturationMode.SAT_PROPAGATE, SubnormalsMode.SUBNORMALS, True),
    (8, 4, SaturationMode.SAT_NONE, SubnormalsMode.NORMALS, True),
    (8, 4, SaturationMode.SAT_NONE, SubnormalsMode.EXTENDED_NORMALS, True),
    (8, 4, SaturationMode.SAT_FINITE, SubnormalsMode.SUBNORMALS, False),
    (6, 3, SaturationMode.SAT_NONE, SubnormalsMode.SUBNORMALS, True),
    (12, 8, SaturationMode.SAT_FINITE, SubnormalsMode.SUBNORMALS, True),
    (16, 12, SaturationMode.SAT_NONE, SubnormalsMode.SUBNORMALS, True),
    (8, 4, SaturationMode.SAT_FINITE_EXTENDED, SubnormalsMode.SUBNORMALS, True),
]

# (man_bits, exp_bits, normal_binades, bias, saturation, is_signed); the last
# has no underflow region at all.
SUPERFP = [
    (3, 4, 1, 7, SaturationMode.SAT_NONE, True),
    (2, 5, 2, 15, SaturationMode.SAT_FINITE, True),
    (0, 4, 2, 7, SaturationMode.SAT_NONE, True),
    (4, 3, 1, 3, SaturationMode.SAT_PROPAGATE, False),
    (9, 3, 1, 3, SaturationMode.SAT_NONE, True),
]


# (wl, fl, is_signed, symmetric, prng_bits): the reference format, symmetric,
# unsigned, a 16-bit one, binary32's full precision (mag_bits 24, no room for
# a draw), the step at 2^-125 (inputs there are binary32 subnormals, which
# the GPU flushes in float arithmetic and the cast never does any on), and the
# two's complement bottom at -2^127.
FIXEDPOINT = [
    (8, 6, True, False, 10),
    (8, 6, True, True, 10),
    (12, 0, False, False, 10),
    (16, 14, True, False, 7),
    (25, 20, True, False, 0),
    (8, 125, True, False, 10),
    (8, -120, True, False, 10),
]


def quantizers():
    """(label, fn) for every format and rounding mode the sweep covers."""
    out = []
    for K, P, sat, subn, signed in BINARYK:
        sign = "" if signed else " unsigned"
        for rm in RoundMode:
            kw: dict[str, Any] = dict(
                prng_bits=10, rounding_mode=rm, saturation_mode=sat, subnormals_mode=subn,
                is_signed=signed,
            )  # fmt: skip
            label = f"binaryK K={K} P={P} {sat.name} {subn.name}{sign} {rm.name}"
            out.append((label, lambda x, K=K, P=P, kw=kw: binaryK_quantize(x, K, P, **kw)))
    for man, exp, nb, bias, sat, signed in SUPERFP:
        sign = "" if signed else " unsigned"
        for rm in RoundMode:
            kw: dict[str, Any] = dict(
                prng_bits=10, rounding_mode=rm, saturation_mode=sat, is_signed=signed
            )
            fmt = (man, exp, nb, bias)
            label = f"superfp {man}/{exp}/{nb} bias={bias} {sat.name}{sign} {rm.name}"
            out.append((label, lambda x, fmt=fmt, kw=kw: superfp_quantize(x, *fmt, **kw)))
    for wl, fl, signed, symmetric, prng in FIXEDPOINT:
        kind = " unsigned" if not signed else " symmetric" if symmetric else ""
        for rm in RoundMode:
            kw: dict[str, Any] = dict(
                prng_bits=prng, rounding_mode=rm, is_signed=signed, symmetric=symmetric
            )
            label = f"fixedpoint wl={wl} fl={fl}{kind} {rm.name}"
            out.append((label, lambda x, wl=wl, fl=fl, kw=kw: fixedpoint_quantize(x, wl, fl, **kw)))
    return out


def chunks():
    """Every binary32 word, CHUNK at a time: (index, CPU floats, MPS floats,
    mask of the subnormal inputs on MPS)."""
    for i, base in enumerate(range(-(1 << 31), 1 << 31, CHUNK)):
        words = torch.arange(base, base + CHUNK, dtype=torch.int32)
        x = words.view(torch.float32)
        xm = x.to("mps")
        wm = words.to("mps")
        subnormal = ((wm & 0x7F800000) == 0) & ((wm & 0x007FFFFF) != 0)
        yield i, x, xm, subnormal


def mismatches(ref: torch.Tensor, got: torch.Tensor, subnormal: torch.Tensor) -> tuple[int, int]:
    """(differing words, of which on subnormal inputs), compared on the GPU."""
    differ = ref.to("mps").view(torch.int32) != got.view(torch.int32)
    return int(differ.sum()), int((differ & subnormal).sum())


def sweep():
    cases = quantizers()
    counts = {label: [0, 0] for label, _ in cases}
    t0 = time.perf_counter()
    for i, x, xm, subnormal in chunks():
        for label, fn in cases:
            torch.manual_seed(i)
            ref = fn(x)
            torch.manual_seed(i)
            got = fn(xm)
            bad, bad_sub = mismatches(ref, got, subnormal)
            counts[label][0] += bad
            counts[label][1] += bad_sub
        if (i + 1) % 8 == 0:
            print(f"  {i + 1} of 64 chunks, {time.perf_counter() - t0:.0f} s", flush=True)
    for label, (bad, bad_sub) in counts.items():
        print(f"{label}: {bad} mismatches ({bad_sub} on subnormal inputs)")
    total = sum(b for b, _ in counts.values())
    print(f"{len(cases)} format/mode cases, {total} mismatches in all")


def flatten(path: Path, seen: set[Path], out: list[str]) -> None:
    """A .metal root with its quoted includes resolved, as setup.py's
    embed_metal_sources does for the extension."""
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    for line in path.read_text().splitlines():
        match = re.match(r'^\s*#\s*include\s+"([^"]+)"\s*$', line)
        if match:
            flatten(path.parent / match.group(1), seen, out)
        elif line.strip() != "#pragma once":
            out.append(line)


def control_library(edit):
    """quantize.metal for binary8p4 RNE on float32, with `edit` applied to
    the flattened source, compiled by torch rather than by the extension."""
    lines: list[str] = []
    flatten(CSRC / "mps" / "prelude.metal", set(), lines)
    tail = (
        "using mpt_storage_t = float;\n#define MPT_ROUND_MODE RoundMode::RNE\n"
        "#define MPT_IS_SIGNED true\n#define MPT_PRNG_BITS 0\n#define MPT_FORMATS 0\n"
        "#define MPT_SUBNORMALS SubnormalsMode::SUBNORMALS\ninline BinaryKParams mpt_params()\n"
        "{ return make_binaryK_params<float>(3, 4, 8, true, SaturationMode::SAT_NONE, false); }\n"
    )
    kernel: list[str] = []
    flatten(CSRC / "mps" / "quantize.metal", set(), kernel)
    return torch.mps.compile_shader(edit("\n".join(lines)) + "\n" + tail + "\n".join(kernel))


def control():
    split = "F::rn_sub(t, F::rn_sub(t, xc))"
    safe = "#pragma METAL fp math_mode(safe)"

    def contracted(src: str) -> str:
        assert split in src, "the Veltkamp split moved; update the control"
        return src.replace(split, "(t - metal::fma(-p.fast_split_c, xc, t))")

    def fast_math(src: str) -> str:
        assert safe in src, "prelude.metal's math pragma moved; update the control"
        return src.replace(safe, "#pragma METAL fp math_mode(fast)")

    libs = {"explicit FMA in the split": control_library(contracted)}
    libs["math_mode(fast)"] = control_library(fast_math)
    counts = dict.fromkeys(libs, 0)
    for _, x, xm, subnormal in chunks():
        ref = binaryK_quantize(x, 8, 4)
        params = torch.tensor([CHUNK, 0], dtype=torch.int64, device="mps")
        for name, lib in libs.items():
            got = torch.empty_like(xm)
            lib.mpt_quantize(xm, got, params, threads=(CHUNK // 4, 1, 1))
            counts[name] += mismatches(ref, got, subnormal)[0]
    for name, bad in counts.items():
        print(f"{name}: {bad} mismatches (must be nonzero)")


def _rn32(x: Fraction) -> int:
    """The binary32 word of x rounded to nearest even, subnormals included
    (gradual underflow), infinity past the top."""
    if x == 0:
        return 0
    sign, ax = (0x80000000 if x < 0 else 0), abs(x)
    e = ax.numerator.bit_length() - ax.denominator.bit_length()
    if Fraction(2) ** e > ax:
        e -= 1
    q = Fraction(2) ** max(e - 23, -149)
    k, rest = divmod(ax, q)
    k = int(k)
    if rest * 2 > q or (rest * 2 == q and k & 1):
        k += 1
    value = k * q
    if value >= Fraction(2) ** 128:
        return sign | 0x7F800000
    return sign | struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _words32(t: torch.Tensor) -> list[int]:
    return t.cpu().contiguous().view(torch.int32).tolist()


def odd_probe():
    """What the odd carrier step's Metal primitives may assume of the GPU
    (common/carrier_odd_metal.h).

    1. metal::fma is one rounding to nearest even: equal to the exact
       a*b + c rounded once, wherever no operand is a binary32 subnormal and
       the exact result is not below 2^-126. The flush is decided before
       rounding: 2^-126 (1 - 2^-24), which rounds to 2^-126, comes back 0
       (found 2026-11-15, M3 Pro, macOS 26).
    2. Whether a subnormal x * 1.0f folds to x or flushes to zero, through
       BinaryKMultiplierT's own call under RU, as SplitMacOdd hands it an odd
       product (carrier_odd_metal.h's cast helper is there either way).
    3. Whether -x flushes a subnormal, and the sign of a flushed result.
    4. Whether the Metal compiler takes a noinline attribute.
    """
    head = (
        "#include <metal_stdlib>\n"
        "#pragma METAL fp math_mode(safe)\n"
        "#pragma METAL fp contract(off)\n"
    )
    plain = torch.mps.compile_shader(
        head
        + """
inline float times(float a, float b) { return a * b; }
kernel void probe_fma(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                      device const float *c [[buffer(2)]], device uint *o [[buffer(3)]],
                      uint i [[thread_position_in_grid]])
{
    o[i] = as_type<uint>(metal::fma(a[i], b[i], c[i]));
}
kernel void probe_ops(device const float *x [[buffer(0)]], device uint *o [[buffer(1)]],
                      uint i [[thread_position_in_grid]])
{
    const float v = x[i];
    o[4 * i + 0] = as_type<uint>(times(v, 1.0f));
    o[4 * i + 1] = as_type<uint>(-v);
    o[4 * i + 2] = as_type<uint>(v * 0.5f);
    o[4 * i + 3] = as_type<uint>(v);
}
"""
    )

    # 1. metal::fma against the exact sum rounded once.
    def is_sub(w: int) -> bool:
        return (w & 0x7F800000) == 0 and (w & 0x007FFFFF) != 0

    g = torch.Generator().manual_seed(0)
    n = 1 << 14
    exps = torch.randint(-60, 60, (3, n), generator=g)
    vals = torch.randn(3, n, generator=g, dtype=torch.float64) * torch.pow(2.0, exps.double())
    a, b, c = (v.float() for v in vals)
    # the binary32 review vectors' operands too, and some products that land
    # in or near the subnormal range
    extra = [
        (1, 1, 2**-30),
        (1, 1, -(2**-30)),
        (1 + 2**-12, 1 + 2**-12, -(2**-11)),
        (2**-50, 2**-50, 1),
        (2**-75, 2**-75, 0),
        (2**-63 * (1 - 2**-24), 2**-63, 0),
        (2**-60, 2**-60, 2**-110),
        (2**-63, 2**-64, 0),
        (2**64, 2**64, -3.4028234663852886e38),
    ]
    a = torch.cat([a, torch.tensor([e[0] for e in extra], dtype=torch.float32)])
    b = torch.cat([b, torch.tensor([e[1] for e in extra], dtype=torch.float32)])
    c = torch.cat([c, torch.tensor([e[2] for e in extra], dtype=torch.float32)])
    got = torch.empty(a.numel(), dtype=torch.int32, device="mps")
    plain.probe_fma(a.to("mps"), b.to("mps"), c.to("mps"), got, threads=(a.numel(), 1, 1))
    torch.mps.synchronize()
    clean_bad = flushed = flushed_other = 0
    for wa, wb, wc, x, y, z, w in zip(
        _words32(a),
        _words32(b),
        _words32(c),
        a.tolist(),
        b.tolist(),
        c.tolist(),
        _words32(got),
        strict=True,
    ):
        want = _rn32(Fraction(x) * Fraction(y) + Fraction(z))
        mask = lambda v: v & 0xFFFFFFFF  # noqa: E731
        exact = Fraction(x) * Fraction(y) + Fraction(z)
        tiny = 0 < abs(exact) < Fraction(2) ** -126
        sub = any(is_sub(mask(v)) for v in (wa, wb, wc)) or tiny
        if not sub:
            if mask(w) != want:
                clean_bad += 1
                print(f"   differs: fma({x!r}, {y!r}, {z!r}) = {mask(w):#010x}, RN {want:#010x}")
        elif mask(w) != want:
            flushed += 1
        else:
            flushed_other += 1
    print(f"1. metal::fma: {clean_bad} of the cases without subnormals differ from RN(a*b + c)")
    print(
        f"   (cases with a subnormal operand or a result below 2^-126: {flushed} differ, "
        f"{flushed_other} agree)"
    )

    # 2-3. x * 1.0f, -x and a flushed result, on subnormals of both signs.
    sub = torch.tensor([1, 0x00400000, 0x007FFFFF, 0x80000001, 0x80400000]).to(torch.int32)
    out = torch.empty(4 * sub.numel(), dtype=torch.int32, device="mps")
    plain.probe_ops(sub.view(torch.float32).to("mps"), out, threads=(sub.numel(), 1, 1))
    torch.mps.synchronize()
    rows = torch.tensor(_words32(out)).view(-1, 4).tolist()
    for w, (times1, neg, half, copy) in zip(sub.tolist(), rows, strict=True):
        print(
            f"2-3. x = {w & 0xFFFFFFFF:#010x}: x*1 {times1 & 0xFFFFFFFF:#010x}, "
            f"-x {neg & 0xFFFFFFFF:#010x}, x*0.5 {half & 0xFFFFFFFF:#010x}, "
            f"copy {copy & 0xFFFFFFFF:#010x}"
        )

    lines: list[str] = []
    flatten(CSRC / "mps" / "prelude.metal", set(), lines)
    policy = torch.mps.compile_shader(
        "\n".join(lines)
        + """
kernel void probe_mul1(device const float *x [[buffer(0)]], device uint *o [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    const BinaryKMultiplierT<float, RoundMode::RU> m(3, 4, 8, true, SaturationMode::SAT_NONE,
                                                     SubnormalsMode::SUBNORMALS, 0);
    PhiloxEngine rng;
    o[i] = as_type<uint>(m(x[i], 1.0f, rng));
}
"""
    )
    x = torch.tensor([2.0**-140, 2.0**-149, 2.0**-127], dtype=torch.float32)
    out = torch.empty(x.numel(), dtype=torch.int32, device="mps")
    policy.probe_mul1(x.to("mps"), out, threads=(x.numel(), 1, 1))
    torch.mps.synchronize()
    print(
        "2. BinaryKMultiplierT<RU>(x, 1) for binary8p4, x subnormal: "
        + ", ".join(f"{v:#010x}" for v in _words32(out))
        + " (0x3a800000 = 2^-10 if x * 1 folds, 0 if it flushes)"
    )

    # 4. noinline spellings.
    for spelling in ("__attribute__((noinline))", "[[clang::noinline]]", "[[gnu::noinline]]"):
        src = (
            head
            + spelling
            + " float cold(float v) { return v * 3.0f; }\n"
            + "kernel void probe_call(device float *o [[buffer(0)]],\n"
            + "                       uint i [[thread_position_in_grid]])"
            + " { o[i] = cold(o[i]); }\n"
        )
        try:
            torch.mps.compile_shader(src)
            print(f"4. {spelling}: accepted")
        except RuntimeError as e:
            print(f"4. {spelling}: refused ({str(e).splitlines()[0][:120]})")


def _floats(g: torch.Generator, n: int, e_lo: int, e_hi: int) -> torch.Tensor:
    """n binary32 values of random sign and mantissa whose unbiased exponent
    is uniform in [e_lo, e_hi]; -127 gives a subnormal (or a zero, for a
    zero mantissa) and 128 an infinity or a NaN."""
    sign = torch.randint(0, 2, (n,), generator=g) << 31
    field = torch.randint(e_lo + 127, e_hi + 128, (n,), generator=g)
    mant = torch.randint(0, 1 << 23, (n,), generator=g)
    return _as_float(sign | (field << 23) | mant)


def _as_float(w: torch.Tensor) -> torch.Tensor:
    """int64 words in [0, 2^32) as binary32 values."""
    return (w - ((w >> 31) << 32)).to(torch.int32).view(torch.float32)


def _words(g: torch.Generator, n: int) -> torch.Tensor:
    return _as_float(torch.randint(0, 1 << 32, (n,), generator=g))


def _pick(g: torch.Generator, *options: torch.Tensor) -> torch.Tensor:
    """Each element from one of `options`, chosen uniformly."""
    k = torch.randint(0, len(options), options[0].shape, generator=g)
    out = options[0].clone()
    for i, o in enumerate(options[1:], 1):
        out[k == i] = o[k == i]
    return out


def _zeros(g: torch.Generator, n: int) -> torch.Tensor:
    return _as_float(torch.randint(0, 2, (n,), generator=g) << 31)


def _jitter(g: torch.Generator, x: torch.Tensor, ulps: int) -> torch.Tensor:
    """x moved by up to `ulps` units in the last place, on the word."""
    w = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    d = torch.randint(-ulps, ulps + 1, x.shape, generator=g)
    return _as_float((w + d) & 0xFFFFFFFF)


def _odd_regimes(g: torch.Generator, n: int):
    """(op, regime, a, b, c) for odd-sweep: each primitive's operands in the
    regimes its gate and its cold path tell apart (common/carrier_odd_metal.h)."""
    F = lambda lo, hi: _floats(g, n, lo, hi)  # noqa: E731
    sub = lambda: F(-127, -127)  # noqa: E731
    for op, name in ((0, "fma"), (1, "add"), (2, "mul")):
        yield op, f"{name} words", _words(g, n), _words(g, n), _words(g, n)
        yield op, f"{name} normal", F(-20, 20), F(-20, 20), F(-40, 40)
        yield (
            op,
            f"{name} subnormal",
            _pick(g, sub(), F(-30, 30), F(100, 127)),
            _pick(g, sub(), F(-30, 30), F(100, 127)),
            _pick(g, sub(), F(-30, 30)),
        )
        yield (
            op,
            f"{name} zeros",
            _pick(g, _zeros(g, n), sub(), F(-140, 127)),
            _pick(g, _zeros(g, n), sub(), F(-140, 127)),
            _pick(g, _zeros(g, n), sub(), F(-140, 127)),
        )
    # fma
    a, b = F(-20, 20), F(-20, 20)
    yield 0, "fma cancel", a, b, _jitter(g, -(a.double() * b.double()).float(), 4)
    yield (
        0,
        "fma tiny product",
        F(-90, -30),
        F(-90, -30),
        _pick(g, _zeros(g, n), sub(), F(-126, -60), F(-30, 30)),
    )
    a, b = F(55, 70), F(55, 70)
    yield (
        0,
        "fma huge",
        a,
        b,
        _pick(
            g, F(100, 127), _jitter(g, -(a.double() * b.double()).clamp(-3.4e38, 3.4e38).float(), 4)
        ),
    )
    ea = torch.randint(-50, -27, (n,), generator=g)
    eb = torch.randint(-80, -75, (n,), generator=g) - ea
    a = F(0, 0) * torch.pow(2.0, ea.float())
    b = F(0, 0) * torch.pow(2.0, eb.float())
    yield 0, "fma gate edges", a, b, _pick(g, _zeros(g, n), F(-102, -98), F(123, 127))
    a, b = F(-20, 20), F(-20, 20)
    shift = _pick(
        g,
        torch.randint(54, 67, (n,), generator=g).float(),
        -torch.randint(54, 67, (n,), generator=g).float(),
    )
    e_ab = torch.floor(torch.log2((a.double() * b.double()).abs())).float()
    c = F(0, 0) * torch.pow(2.0, (e_ab + shift).clamp(-125, 126))
    yield 0, "fma far terms", a, b, c
    # add
    a = F(-20, 20)
    yield 1, "add cancel", a, _jitter(g, -a, 4), a
    yield 1, "add tiny", _pick(g, sub(), F(-126, -90)), _pick(g, sub(), F(-126, -90), F(-30, 30)), a
    yield 1, "add huge", F(120, 127), _pick(g, F(120, 127), _jitter(g, -F(120, 127), 4)), a
    yield (
        1,
        "add gate edges",
        _pick(g, F(-105, -101), F(123, 127)),
        _pick(g, F(-105, -101), F(123, 127), F(-20, 20)),
        a,
    )
    e = torch.randint(-60, 60, (n,), generator=g).float()
    gap = torch.randint(54, 67, (n,), generator=g).float()
    yield 1, "add far terms", F(0, 0) * torch.pow(2.0, e), F(0, 0) * torch.pow(2.0, e - gap), a
    # mul
    yield 2, "mul tiny", F(-90, -30), F(-90, -30), a
    yield 2, "mul huge", F(55, 70), F(55, 70), a
    ea = torch.randint(-50, -27, (n,), generator=g)
    eb = torch.randint(-80, -75, (n,), generator=g) - ea
    yield (
        2,
        "mul gate edges",
        F(0, 0) * torch.pow(2.0, ea.float()),
        F(0, 0) * torch.pow(2.0, eb.float()),
        a,
    )


def odd_sweep():
    """_carrier_odd on MPS (common/carrier_odd_metal.h) against the CPU's
    (common/carrier_odd.h's host binary32 primitives) over 2^24 operand
    triples per regime and primitive: every word must agree, a NaN with any
    NaN. The regimes are the gate's and the cold path's: subnormal, zero and
    non-finite operands, cancellation, tiny products, huge terms, the gate's
    bounds, and terms 54 to 66 binades apart, either side of the cold path's
    stand-ins at 60."""
    odd_op = torch.ops.mptorch._carrier_odd.default
    n = 1 << 24
    g = torch.Generator().manual_seed(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    total = 0
    t0 = time.perf_counter()
    for op, name, a, b, c in _odd_regimes(g, n):
        ref = odd_op(a, b, c, op)
        got = odd_op(a.to("mps"), b.to("mps"), c.to("mps"), op).cpu()
        same = (ref.view(torch.int32) == got.view(torch.int32)) | (ref.isnan() & got.isnan())
        bad = int((~same).sum())
        total += bad
        print(f"{name:<20} {bad} mismatches", flush=True)
        for i in (~same).nonzero().flatten()[:4].tolist():
            w = [int(t[i].view(torch.int32)) & 0xFFFFFFFF for t in (a, b, c, ref, got)]
            print("    a b c {:08x} {:08x} {:08x}  cpu {:08x}  mps {:08x}".format(*w))
    print(f"{total} mismatches in all ({time.perf_counter() - t0:.0f} s)")


def best_of(repeats: int, fn, *args) -> float:
    """The fastest of `repeats` calls, in milliseconds, synchronized."""
    times = []
    for _ in range(repeats):
        torch.mps.synchronize()
        t = time.perf_counter()
        fn(*args)
        torch.mps.synchronize()
        times.append(time.perf_counter() - t)
    return min(times) * 1e3


GEMMS = {
    "binaryK": lambda a, b, p: binaryK_matmul(a, b, mul_K=8, mul_P=4, acc_K=12, acc_P=6),
    "superfp": lambda a, b, p: superfp_matmul(
        a, b, mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=1, mul_bias=7,
        acc_man_bits=5, acc_exp_bits=5, acc_normal_binades=2, acc_bias=15,
    ),
    "binaryK_fma": lambda a, b, p: binaryK_matmul_fma(a, b, fma_K=10, fma_P=5),
    "superfp_fma": lambda a, b, p: superfp_matmul_fma(
        a, b, fma_man_bits=4, fma_exp_bits=5, fma_normal_binades=1, fma_bias=15
    ),
    "binaryK_mixed": lambda a, b, p: binaryK_matmul_mixed(
        a, b, p % 3, mul_K=[8, 8, 6], mul_P=[4, 3, 3], acc_K=[12, 10, 8], acc_P=[6, 5, 4]
    ),
    "superfp_mixed": lambda a, b, p: superfp_matmul_mixed(
        a, b, p % 3, mul_man_bits=[3, 2, 1], mul_exp_bits=[4, 5, 4], mul_normal_binades=1,
        mul_bias=[7, 15, 7], acc_man_bits=[5, 4, 3], acc_exp_bits=[5, 5, 5],
        acc_normal_binades=2, acc_bias=[15, 15, 15],
    ),
    "binaryK_fma_mixed": lambda a, b, p: binaryK_matmul_fma_mixed(
        a, b, p % 2, fma_K=[10, 8], fma_P=[5, 4]
    ),
    "superfp_fma_mixed": lambda a, b, p: superfp_matmul_fma_mixed(
        a, b, p % 2, fma_man_bits=[4, 2], fma_exp_bits=[5, 5], fma_normal_binades=1,
        fma_bias=[15, 15],
    ),
}  # fmt: skip


def bench():
    shapes = [(64, 784, 512), (512, 512, 512), (1024, 1024, 1024)]
    print(f"CPU: {torch.get_num_threads()} threads")
    for op, fn in GEMMS.items():
        row = f"{op:18}"
        for M, K, N in shapes:
            a, b = torch.randn(M, K), torch.randn(K, N)
            p = torch.randint(0, 6, (M, N), dtype=torch.int32)
            am, bm, pm = a.to("mps"), b.to("mps"), p.to("mps")
            ref = fn(a, b, p)
            got = fn(am, bm, pm)  # compiles the instantiation
            same = torch.equal(ref.view(torch.int32), got.cpu().view(torch.int32))
            cpu = best_of(1 if M >= 1024 else 3, fn, a, b, p)
            mps = best_of(3, fn, am, bm, pm)
            row += f"  [{M},{K}]@[{K},{N}] {cpu:7.1f} / {mps:6.2f} ms{'' if same else ' DIFFER'}"
        print(row, flush=True)
    a, b = torch.randn(2048, 2048), torch.randn(2048, 2048)
    fn = GEMMS["binaryK"]
    fn(a.to("mps"), b.to("mps"), None)
    mps = best_of(3, fn, a.to("mps"), b.to("mps"), None)
    print(f"binaryK 2048^3: {best_of(1, fn, a, b, None):.0f} / {mps:.1f} ms")
    sr: dict[str, Any] = dict(rounding_mode=RoundMode.SR, prng_bits=10)
    quantize = [
        ("binaryK_quantize", lambda t: binaryK_quantize(t, 8, 4)),
        ("superfp_quantize", lambda t: superfp_quantize(t, 3, 4, 1, 7)),
        ("binaryK_quantize SR", lambda t: binaryK_quantize(t, 8, 4, **sr)),
        ("fixedpoint_quantize", lambda t: fixedpoint_quantize(t, 8, 6)),
        ("fixedpoint_quantize SR", lambda t: fixedpoint_quantize(t, 8, 6, **sr)),
    ]
    for n in [1 << 20, 1 << 25]:
        x = torch.randn(n)
        xm = x.to("mps")
        for name, q in quantize:
            q(xm)
            cpu, mps = best_of(3, q, x), best_of(5, q, xm)
            print(f"{name} over 2^{n.bit_length() - 1}: {cpu:.2f} / {mps:.2f} ms")
    bench_beyond_naive()


def _row(label, fn, args_cpu, args_mps, repeats_cpu=1):
    """One timed op on both devices, CPU / MPS in ms, and whether they agree."""
    torch.manual_seed(0)
    ref = fn(*args_cpu)
    torch.manual_seed(0)
    got = fn(*args_mps)  # compiles the instantiation
    refs = ref if isinstance(ref, tuple | list) else (ref,)
    gots = got if isinstance(got, tuple | list) else (got,)
    same = all(
        torch.equal(r.contiguous().view(torch.uint8), g.cpu().contiguous().view(torch.uint8))
        for r, g in zip(refs, gots, strict=True)
    )
    cpu = best_of(repeats_cpu, fn, *args_cpu)
    mps = best_of(5, fn, *args_mps)
    print(f"{label:44} {cpu:8.1f} / {mps:7.2f} ms{'' if same else '  DIFFER'}", flush=True)


def _ms(fn, *args) -> float:
    fn(*args)  # compiles the instantiation
    return best_of(5, fn, *args)


def bench_odd():
    """The odd carrier step on the GPU (carrier_rounding=RoundMode.RO,
    common/carrier_odd_metal.h): each odd op against its default twin on MPS,
    in ms, with the ratio, at 512^3 and 1024^3; DIFFER where the odd op's
    result at 512^3 is not the CPU's word for word."""
    print("odd carrier step on MPS: default / odd ms, ratio")
    RO = RoundMode.RO
    sr: dict[str, Any] = dict(mul_prng_bits=8, acc_prng_bits=8)
    kahan = dict(accumulate_algorithm=AccumulateAlgorithm.KAHAN)
    rows: list[tuple[str, Any]] = []
    for rm in (RoundMode.RNE, RoundMode.RU, RoundMode.SR):
        kw: dict[str, Any] = dict(rounding_mode=rm)
        s = dict(kw, **sr) if rm is RoundMode.SR else kw
        rows += [
            (
                f"binaryK {rm.name}",
                lambda a, b, cr, s=s: binaryK_matmul(
                    a, b, mul_K=8, mul_P=4, acc_K=12, acc_P=6, carrier_rounding=cr, **s
                ),
            ),
            (
                f"superfp {rm.name}",
                lambda a, b, cr, s=s: superfp_matmul(
                    a, b, mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=1, mul_bias=7,
                    acc_man_bits=5, acc_exp_bits=5, acc_normal_binades=2, acc_bias=15,
                    carrier_rounding=cr, **s,
                ),
            ),
            (
                f"binaryK_fma {rm.name}",
                lambda a, b, cr, kw=kw: binaryK_matmul_fma(
                    a, b, fma_K=10, fma_P=5, carrier_rounding=cr, **kw
                ),
            ),
            (
                f"superfp_fma {rm.name}",
                lambda a, b, cr, kw=kw: superfp_matmul_fma(
                    a, b, fma_man_bits=4, fma_exp_bits=5, fma_normal_binades=1, fma_bias=15,
                    carrier_rounding=cr, **kw,
                ),
            ),
        ]  # fmt: skip
    for rm in (RoundMode.RNE, RoundMode.SR):
        s: dict[str, Any] = dict(kahan, rounding_mode=rm, **(sr if rm is RoundMode.SR else {}))
        rows.append(
            (
                f"binaryK KAHAN {rm.name}",
                lambda a, b, cr, s=s: binaryK_matmul(
                    a, b, mul_K=8, mul_P=4, acc_K=12, acc_P=6, carrier_rounding=cr, **s
                ),
            )
        )
    rows.append(
        (
            "binaryK unquantized sum RU",
            lambda a, b, cr: binaryK_matmul(
                a, b, mul_K=8, mul_P=4, accumulate_quant=False, rounding_mode=RoundMode.RU,
                carrier_rounding=cr,
            ),
        )
    )  # fmt: skip
    for n in (512, 1024):
        a, b = torch.randn(n, n), torch.randn(n, n)
        am, bm = a.to("mps"), b.to("mps")
        p = torch.randint(0, 3, (n, n), dtype=torch.int32)
        pm = p.to("mps")
        palette = (
            "binaryK_mixed RNE",
            lambda x, y, cr, q: binaryK_matmul_mixed(
                x, y, q, mul_K=[8, 8, 6], mul_P=[4, 3, 3], acc_K=[12, 10, 8], acc_P=[6, 5, 4],
                carrier_rounding=cr,
            ),
        )  # fmt: skip
        for label, fn in rows + [palette]:
            extra = (pm,) if fn is palette[1] else ()
            base = _ms(fn, am, bm, RoundMode.RNE, *extra)
            odd = _ms(fn, am, bm, RO, *extra)
            same = ""
            if n == 512:
                torch.manual_seed(0)
                ref = fn(a, b, RO, *((p,) if extra else ()))
                torch.manual_seed(0)
                got = fn(am, bm, RO, *extra)
                same = (
                    ""
                    if torch.equal(ref.view(torch.int32), got.cpu().view(torch.int32))
                    else ("  DIFFER")
                )
            print(f"{label + f' {n}^3':36} {base:7.2f} / {odd:7.2f} ms  {odd / base:4.2f}x{same}",
                  flush=True)  # fmt: skip
    # A convolution layer's forward pass, the conv ops' gathered GEMM.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 64, 32, 32, generator=g).to("mps")
    w = torch.randn(64, 64, 3, 3, generator=g).to("mps")
    geo: dict[str, Any] = dict(stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, nd=2)
    times = []
    for cr in (RoundMode.RNE, RO):
        fwd = conv_formats(SplitMac(BinaryK(8, 4), BinaryK(12, 6), carrier_rounding=cr)).fwd_math
        assert fwd
        times.append(_ms(lambda u, v, fwd=fwd: fwd(u, v, None, **geo), x, w))
    print(f"{'conv 3x3 [8,64,32,32] fwd':36} {times[0]:7.2f} / {times[1]:7.2f} ms  "
          f"{times[1] / times[0]:4.2f}x")  # fmt: skip


def bench_counted():
    """The counted ops (common/gemm_counted.h, gemm.metal under MPT_COUNTING)
    against their twins on MPS: twin / counted ms and the ratio, the two
    alternated, the best of five rounds each; DIFFER where the counted op's
    result is not its twin's word for word. Needs a build with
    MPTORCH_COUNTING=1."""
    print("counted ops on MPS: twin / counted ms, ratio")
    warnings.simplefilter("ignore", FormatRangeWarning)
    b8, b12, b16 = BinaryK(8, 4), BinaryK(12, 6), BinaryK(16, 11)
    sr8, sr12 = BinaryK(8, 4, prng_bits=8), BinaryK(12, 6, prng_bits=8)
    A = AccumulateAlgorithm
    macs: list[tuple[str, Any]] = []
    for rm in (RoundMode.RNE, RoundMode.SR):
        m, acc = (sr8, sr12) if rm is RoundMode.SR else (b8, b12)
        macs += [
            (f"binaryK NAIVE {rm.name}", SplitMac(m, acc, rounding=rm)),
            (f"binaryK KAHAN {rm.name}", SplitMac(
                m, acc, rounding=rm, accumulate_algorithm=A.KAHAN)),
            (f"binaryK BLOCK 32 {rm.name}", SplitMac(
                m, acc, rounding=rm, accumulate_algorithm=A.BLOCK, block_size=32, outer=b16)),
            (f"binaryK TREE 64 {rm.name}", SplitMac(
                m, acc, rounding=rm, accumulate_algorithm=A.TREE, block_size=64, outer=b16)),
            (f"binaryK_fma NAIVE {rm.name}", FusedMac(acc, rounding=rm)),
        ]  # fmt: skip
    macs += [
        # GEMMS' superfp formats, whose bottoms outrun binary32 (a warning)
        ("superfp NAIVE RNE", SplitMac(SuperFP(3, 4, 1, 7), SuperFP(5, 5, 2, 15))),
        ("fixedpoint carrier path RNE", SplitMac(FixedPoint(12, 6), FixedPoint(16, 6))),
        ("fixedpoint integer RNE", SplitMac(FixedPoint(16, 12), FixedPoint(20, 4))),
        ("fixedpoint_fma RNE", FusedMac(FixedPoint(10, 4))),
    ]

    def alternated(twin, counted, *args):
        twin(*args), counted(*args)  # compiles both instantiations
        t, c = float("inf"), float("inf")
        for r in range(5):
            for which in (0, 1) if r % 2 == 0 else (1, 0):
                if which == 0:
                    t = min(t, best_of(1, twin, *args))
                else:
                    c = min(c, best_of(1, counted, *args))
        return t, c

    for n in (512, 1024):
        g = torch.Generator().manual_seed(0)
        a = torch.randn(n, n, generator=g).to("mps")
        b = torch.randn(n, n, generator=g).to("mps")
        for label, mac in macs:
            spec = spec_for_mac(mac)
            op, args = _counted_twin(spec)  # ty: ignore[not-iterable]
            twin = lambda x, y, spec=spec: _run_gemm(spec, x, y, False, False)  # noqa: E731
            counted = lambda x, y, op=op, args=args: op(x, y, False, False, *args)[0]  # noqa: E731
            torch.manual_seed(0)
            ref = twin(a, b)
            torch.manual_seed(0)
            same = torch.equal(ref.view(torch.int32), counted(a, b).view(torch.int32))
            t, c = alternated(twin, counted, a, b)
            print(f"{label + f' {n}^3':36} {t:7.2f} / {c:7.2f} ms  {c / t:4.2f}x"
                  f"{'' if same else '  DIFFER'}", flush=True)  # fmt: skip
    # A convolution layer's forward pass, the conv ops' gathered GEMM.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 64, 32, 32, generator=g).to("mps")
    w = torch.randn(64, 64, 3, 3, generator=g).to("mps")
    spec = spec_for_mac(SplitMac(b8, b12))
    conv, args = _counted_conv_twin(spec)  # ty: ignore[not-iterable]
    geometry = ((32, 32), (1, 1), (1, 1), (1, 1), 1)
    plain = _ConvOp(spec, None)
    t, c = alternated(
        lambda u, v: plain(_FWD, u, v, *geometry),
        lambda u, v: conv(u, v, _FWD, *geometry, *args)[0],
        w, x,
    )  # fmt: skip
    print(f"{'conv 3x3 [8,64,32,32] fwd':36} {t:7.2f} / {c:7.2f} ms  {c / t:4.2f}x")


def bench_beyond_naive():
    """The ops past the NAIVE GEMMs and the out-of-place quantizers: the
    accumulate algorithms, a cross-family mac, the three conv passes, the
    block formats' elementwise ops and their GEMM, and an in-place quantizer."""
    print("\nbeyond NAIVE, CPU / MPS:")
    kahan = dict(accumulate_algorithm=AccumulateAlgorithm.KAHAN)
    block = dict(
        accumulate_algorithm=AccumulateAlgorithm.BLOCK, block_size=32, outer_K=16, outer_P=11
    )
    tree = dict(
        accumulate_algorithm=AccumulateAlgorithm.TREE, block_size=64, outer_K=16, outer_P=11
    )
    for n in (512, 1024):
        a, b = torch.randn(n, n), torch.randn(n, n)
        am, bm = a.to("mps"), b.to("mps")
        for name, kw in [("NAIVE", {}), ("KAHAN", kahan), ("BLOCK 32", block), ("TREE 64", tree)]:
            for rm in (RoundMode.RNE, RoundMode.SR):
                kws: dict[str, Any] = dict(kw, rounding_mode=rm, mul_prng_bits=8, acc_prng_bits=8)

                def fn(x, y, kws=kws):
                    return binaryK_matmul(x, y, mul_K=8, mul_P=4, acc_K=12, acc_P=6, **kws)

                _row(f"binaryK {name} {rm.name} {n}^3", fn, (a, b), (am, bm))
        cross = lambda x, y: binaryK_superfp_matmul(  # noqa: E731
            x,
            y,
            mul_K=8,
            mul_P=4,
            acc_man_bits=5,
            acc_exp_bits=5,
            acc_normal_binades=2,
            acc_bias=15,
        )
        _row(f"binaryK_superfp {n}^3", cross, (a, b), (am, bm))

    # A convolution layer's three passes, and a strided one's input gradient
    # (four residue classes).
    formats = conv_formats(SplitMac(BinaryK(8, 4), BinaryK(12, 6)))
    fwd, igrad, wgrad = formats.fwd_math, formats.bwd_igrad_math, formats.bwd_wgrad_math
    assert fwd and igrad and wgrad
    for stride in (1, 2):
        x, w = torch.randn(8, 64, 32, 32), torch.randn(64, 64, 3, 3)
        geo = dict(stride=stride, padding=1, dilation=1, groups=1, nd=2)
        y = fwd(x, w, None, **geo)
        dy = torch.randn_like(y)
        cpu_args, mps_args = (x, w, dy), (x.to("mps"), w.to("mps"), dy.to("mps"))
        label = f"conv 3x3 [8,64,32,32] stride {stride}"
        _row(
            f"{label} fwd",
            lambda x, w, dy, geo=geo, f=fwd: f(x, w, None, **geo),
            cpu_args,
            mps_args,
        )
        _row(
            f"{label} igrad",
            lambda x, w, dy, geo=geo, f=igrad: f(dy, w, input_size=x.shape, **geo),
            cpu_args, mps_args,
        )  # fmt: skip
        _row(
            f"{label} wgrad",
            lambda x, w, dy, geo=geo, f=wgrad: f(dy, x, weight_size=w.shape, **geo),
            cpu_args, mps_args,
        )  # fmt: skip

    # The block formats over a 4096 x 4096 tensor, and their GEMM.
    sq = dataclasses.replace(NVFP4, block_rows=16)
    fos = dataclasses.replace(NVFP4, scale_rounding=ScaleRounding.FOUR_OVER_SIX_MSE)
    big = dataclasses.replace(MXFP8_E4M3, block_size=128, block_rows=128)
    big_fos = dataclasses.replace(fos, block_size=128, block_rows=128)
    x = torch.randn(4096, 4096)
    xm = x.to("mps")
    for name, fmt in [
        ("MXFP8", MXFP8_E4M3),
        ("NVFP4", NVFP4),
        ("NVFP4 16x16", sq),
        ("NVFP4 4/6 MSE", fos),
        ("MXFP8 128x128", big),
        ("E2M1 128x128 4/6 MSE", big_fos),
    ]:
        pack = lambda t, fmt=fmt: (lambda q: (q.data, q.scales))(block_pack(t, fmt))  # noqa: E731
        _row(f"block_pack {name} 4096^2", pack, (x,), (xm,), 3)
        _row(
            f"block_quantize {name} 4096^2",
            lambda t, fmt=fmt: block_quantize(t, fmt),
            (x,),
            (xm,),
            3,
        )
        p, pm = block_pack(x, fmt), block_pack(xm, fmt)
        _row(f"block_unpack {name} 4096^2", lambda q: q.unpack(), (p,), (pm,), 3)
    a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
    for name, fmt in [("MXFP8", MXFP8_E4M3), ("NVFP4", NVFP4)]:
        pa, pb = block_pack(a, fmt), block_pack(b, fmt, -2)
        pam, pbm = block_pack(a.to("mps"), fmt), block_pack(b.to("mps"), fmt, -2)
        gemm = lambda u, v: block_matmul(u, v, acc=BinaryK(12, 6))  # noqa: E731
        _row(f"block_matmul {name} 1024^3", gemm, (pa, pb), (pam, pbm))

    x = torch.randn(1 << 25)
    _row(
        "binaryK_quantize_ over 2^25",
        lambda t: binaryK_quantize_(t, 8, 4),
        (x.clone(),),
        (x.to("mps"),),
        3,
    )


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        raise SystemExit("No MPS device found.")
    {
        "bench": bench,
        "bench-beyond": bench_beyond_naive,
        "sweep": sweep,
        "control": control,
        "odd-probe": odd_probe,
        "odd-sweep": odd_sweep,
        "bench-odd": bench_odd,
        "bench-counted": bench_counted,
    }[sys.argv[1] if len(sys.argv) > 1 else "bench"]()
