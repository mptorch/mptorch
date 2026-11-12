"""The MPS backend (``mptorch/csrc/mps/``), measured and swept against the CPU.

    python3 dev/benchmarks/mps_backend.py bench     # every op on MPS against the CPU op
    python3 dev/benchmarks/mps_backend.py bench-beyond  # the ops past the NAIVE GEMMs only
    python3 dev/benchmarks/mps_backend.py sweep     # the quantizers over all 2^32 binary32 words
    python3 dev/benchmarks/mps_backend.py control   # the sweep can see a contracted split

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

The GEMM kernel stages 16 x 16 tiles of A and B in threadgroup memory
(gemm.metal says what that bought and what it did not). Before it did, one
thread per element read its operands from device memory and the 1024^3 row
read 12.2 / 8.7 / 29.8 / 17.1 ms for binaryK / binaryK_fma / binaryK_mixed
/ binaryK_fma_mixed, and a split binaryK mac under SR 82 ms (60 now).
"""

import dataclasses
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

from mptorch import MXFP8_E4M3, NVFP4, AccumulateAlgorithm, BinaryK, ScaleRounding
from mptorch.number import RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import (
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
    (0, 4, 1, 7, SaturationMode.SAT_NONE, True),
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
    {"bench": bench, "bench-beyond": bench_beyond_naive, "sweep": sweep, "control": control}[
        sys.argv[1] if len(sys.argv) > 1 else "bench"
    ]()
