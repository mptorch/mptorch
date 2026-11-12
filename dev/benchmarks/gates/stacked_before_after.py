"""Two builds of the extension compared: op outputs word for word, time and memory.

`capture` and `compare` are the op-level gate, run before and after any
change: `capture` saves the raw bytes of the GEMM and elementwise ops'
outputs (and with `--block` the block formats' ops) over a fixed grid of
formats, shapes, transpose flags, rounding modes and, for the quantizers,
dtypes and saturation modes, and `compare` checks two captures byte for byte,
so NaN payloads and signed zeros are held to it too, and exits non-zero if
anything moves outside the one exception below.

`time` and `memory` measure a build against an older one. The other timing
scripts here each measure one change against the build it landed on, and
those numbers do not compose -- several changes overlap, several are confined
to one path, and some move no kernel's clock at all (since 02855b2, two: the
two device syncs each mixed-format call took to validate its precision map,
and the GEMM's `double` instantiations, which narrowed on load and computed
in float). These two modes answer the other question: what the tree is worth
today against the tree as it stood before all of them.

Every mode needs two builds of the extension. For a change in progress that
is a capture, a rebuild and a second capture; against an older commit the
script is run from two worktrees. One possible baseline is 02855b2, the last
commit before the performance work on the GEMM and quantizer kernels:

    git worktree add /tmp/base 02855b2                  # the baseline build
    cd /tmp/base && python setup.py build_ext --inplace # its own _C.abi3.so
    cd -

    # alternate the two, flipping the order each round -- two builds timed one
    # after the other cannot resolve anything under ~15% on a laptop GPU
    for r in 1 2 3; do
      for t in /tmp/base $PWD; do
        PYTHONPATH=$t python dev/benchmarks/gates/stacked_before_after.py time --group gpu_gemm
      done
    done

    # and the same two builds compared as values, not as time
    PYTHONPATH=/tmp/base python dev/benchmarks/gates/stacked_before_after.py capture --out base.pt
    python dev/benchmarks/gates/stacked_before_after.py capture --out head.pt
    python dev/benchmarks/gates/stacked_before_after.py compare base.pt head.pt

`compare` allows one exception, by design: since 64982e6 the elementwise SR
quantizers (`binaryK_quantize` and `superfp_quantize` under `RoundMode.SR`)
draw from an in-kernel Philox stream (`common/philox.h`'s `philox_block`)
where they used to materialize a `randint_like` tensor of draws, so against a
baseline from before it their SR rows differ, and every other row must not.
Against 02855b2 they were the only rows to move: 16 of 952 outputs per
backend (RTX 4060 Laptop and its 16-thread CPU, 2026-09-02). `compare` prints
the deterministic and SR rows separately so that exception is visible rather
than averaged away.

The mixed-format ops need `41ac289` as their baseline instead -- they did not
exist before it. Pass `--group gpu_mixed` against that worktree.
"""

import argparse
import collections
import functools
import json
import statistics
import time
from typing import Any

import torch

from mptorch.number import RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import (
    QAffineFormats,
    QLinear,
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_quantize,
    superfp_matmul,
    superfp_matmul_fma,
    superfp_quantize,
)

RNE, SR = RoundMode.RNE, RoundMode.SR
MODES = list(RoundMode)
SATS = [SaturationMode.SAT_NONE, SaturationMode.SAT_FINITE]
SHAPES = [(64, 64, 64), (70, 33, 51), (1, 128, 96), (96, 5, 1), (33, 33, 33)]
MIB = 1024 * 1024

# The two formats this script uses throughout: an e5m2-shaped binaryK and an
# fp8-shaped superfp, the latter at both normal_binades settings because that
# is what decides which of its three regions ordinary operands land in.
BK: dict[str, Any] = dict(mul_K=8, mul_P=4, acc_K=8, acc_P=4)
SFP1: dict[str, Any] = dict(
    mul_man_bits=3,
    mul_exp_bits=4,
    mul_normal_binades=1,
    mul_bias=7,
    acc_man_bits=3,
    acc_exp_bits=4,
    acc_normal_binades=1,
    acc_bias=7,
)
SFP8: dict[str, Any] = dict(SFP1, mul_normal_binades=8, acc_normal_binades=8)


# --------------------------------------------------------------------- timing


def timed(fn, reps, warmup, cuda):
    for _ in range(warmup):
        fn()
    ts = []
    if cuda:
        torch.cuda.synchronize()
        for _ in range(reps):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            ts.append(start.elapsed_time(end))
    else:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
    return {"min": min(ts), "median": statistics.median(ts)}


def gemm_cases(dev, n, dtype=torch.float32):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    sr: dict[str, Any] = dict(rounding_mode=SR, mul_prng_bits=8, acc_prng_bits=8)
    return {
        "binaryK split RNE": lambda: binaryK_matmul(a, b, **BK, rounding_mode=RNE),
        "binaryK split SR": lambda: binaryK_matmul(a, b, **BK, **sr),
        "binaryK fma RNE": lambda: binaryK_matmul_fma(a, b, fma_K=8, fma_P=4, rounding_mode=RNE),
        "superfp split RNE nb=1": lambda: superfp_matmul(a, b, **SFP1, rounding_mode=RNE),
        "superfp split RNE nb=8": lambda: superfp_matmul(a, b, **SFP8, rounding_mode=RNE),
        "superfp split SR nb=1": lambda: superfp_matmul(a, b, **SFP1, **sr),
        "superfp split SR nb=8": lambda: superfp_matmul(a, b, **SFP8, **sr),
        "superfp fma RNE nb=1": lambda: superfp_matmul_fma(
            a,
            b,
            fma_man_bits=3,
            fma_exp_bits=4,
            fma_normal_binades=1,
            fma_bias=7,
            rounding_mode=RNE,
        ),
    }


def mixed_cases(dev, n, dtype=torch.float32):
    from mptorch.quant import binaryK_matmul_mixed, superfp_matmul_mixed

    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    # one palette entry per row, which is the shape the mixed kernels' palette
    # prologue (the per-element format lookup, compiled in under MIXED only)
    # was paid for
    idx = (torch.arange(n, device=dev) % 3).to(torch.int32).view(n, 1)
    bk: dict[str, Any] = dict(mul_K=[8, 6, 5], mul_P=[4, 3, 2])
    sf: dict[str, Any] = dict(
        mul_man_bits=[3, 2, 4],
        mul_exp_bits=[4, 5, 3],
        mul_normal_binades=[1, 1, 1],
        mul_bias=[7, 15, 3],
    )
    # the third slot has three exponent bits, so seven normal binades is its most
    sf8: dict[str, Any] = dict(sf, mul_normal_binades=[8, 8, 7])
    sr: dict[str, Any] = dict(rounding_mode=SR, mul_prng_bits=8, acc_prng_bits=8)
    return {
        "binaryK mixed RNE": lambda: binaryK_matmul_mixed(a, b, idx, **bk, rounding_mode=RNE),
        "binaryK mixed SR": lambda: binaryK_matmul_mixed(a, b, idx, **bk, **sr),
        "superfp mixed RNE nb=1": lambda: superfp_matmul_mixed(a, b, idx, **sf, rounding_mode=RNE),
        "superfp mixed RNE nb=8": lambda: superfp_matmul_mixed(a, b, idx, **sf8, rounding_mode=RNE),
        "superfp mixed SR nb=1": lambda: superfp_matmul_mixed(a, b, idx, **sf, **sr),
        "superfp mixed SR nb=8": lambda: superfp_matmul_mixed(a, b, idx, **sf8, **sr),
    }


def cpu_gemm_cases(dtype=torch.float32):
    out = {}
    for m, k, n in ((256, 256, 256), (32, 1024, 1024)):
        a, b = torch.randn(m, k, dtype=dtype), torch.randn(k, n, dtype=dtype)
        out[f"binaryK split RNE {m}x{k}x{n}"] = functools.partial(
            binaryK_matmul, a, b, **BK, rounding_mode=RNE
        )
    return out


def elem_cases(dev, nelem):
    out = {}
    for dt, tag in ((torch.float32, "f32"), (torch.bfloat16, "bf16")):
        x = torch.randn(nelem, device=dev, dtype=dt)
        pb = 8 if dt is torch.float32 else 4
        out[f"binaryK_quantize RNE {tag}"] = functools.partial(binaryK_quantize, x, 8, 4)
        out[f"binaryK_quantize SR {tag}"] = functools.partial(
            binaryK_quantize, x, 8, 4, prng_bits=pb, rounding_mode=SR
        )
        out[f"superfp_quantize RNE {tag}"] = functools.partial(superfp_quantize, x, 3, 4, 1, 7)
        out[f"superfp_quantize SR {tag}"] = functools.partial(
            superfp_quantize, x, 3, 4, 1, 7, prng_bits=pb, rounding_mode=SR
        )
    return out


def run_timing(args):
    torch.manual_seed(1234)
    # --dtype reaches the GEMM groups only: float64 is the binary64 carrier's
    # row (the GEMMs round a float64 operand in binary64 since 87f653c), and
    # against a build that predates it, the narrow-round-widen path it replaced
    dtype = getattr(torch, args.dtype)
    groups = {
        "gpu_gemm": (lambda: gemm_cases("cuda", args.n, dtype), True),
        "gpu_mixed": (lambda: mixed_cases("cuda", args.n, dtype), True),
        "cpu_gemm": (lambda: cpu_gemm_cases(dtype), False),
        "gpu_elem": (lambda: elem_cases("cuda", 33_554_432), True),
        "cpu_elem": (lambda: elem_cases("cpu", 4_194_304), False),
    }
    build, cuda = groups[args.group]
    res = {name: timed(fn, args.reps, args.warmup, cuda) for name, fn in build().items()}
    print(
        json.dumps(
            {
                "tree": __import__("mptorch").__file__,
                "threads": torch.get_num_threads(),
                "group": args.group,
                "dtype": args.dtype,
                "results": res,
            }
        )
    )


# --------------------------------------------------------------------- memory


def run_memory(args):
    """Twelve stacked QLinear, all six quantizers on: what forward holds for
    backward and the step's high-water, which move with the operands saved, a
    gradient quantized for a path that is not requested and the bias add's
    output -- and, in a build from before 64982e6, with the elementwise SR
    quantizers' tensor of draws."""
    fields = (
        "weight_quant",
        "input_quant",
        "bias_quant",
        "wgrad_quant",
        "igrad_quant",
        "bgrad_quant",
    )
    depth, width, batch = 12, args.width, args.batch
    out = {}
    for mode, prng, tag in ((RNE, 0, "RNE"), (SR, 8, "SR")):
        for name, trainable in (
            ("all trainable", lambda i: True),
            ("first+head trainable", lambda i: i in (0, depth - 1)),
        ):
            torch.manual_seed(1234)
            fmt = QAffineFormats()
            q = functools.partial(binaryK_quantize, K=8, P=4, rounding_mode=mode, prng_bits=prng)
            for f in fields:
                setattr(fmt, f, q)
            layers = [
                QLinear(width, width, bias=True, formats=fmt).to("cuda") for _ in range(depth)
            ]
            for i, layer in enumerate(layers):
                layer.weight.requires_grad_(trainable(i))
                layer.bias.requires_grad_(trainable(i))
            net = torch.nn.Sequential(*[m for lay in layers for m in (lay, torch.nn.ReLU())][:-1])
            x = torch.randn(batch, width, device="cuda")

            net(x).sum().backward()  # warm the allocator and build the graph once
            net.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            y = net(x)
            torch.cuda.synchronize()
            held = torch.cuda.memory_allocated() - base - y.numel() * y.element_size()
            fwd = torch.cuda.max_memory_allocated() - base
            y.sum().backward()
            torch.cuda.synchronize()
            out[f"{name}, {tag}"] = {
                "held_MiB": held / MIB,
                "fwd_MiB": fwd / MIB,
                "step_MiB": (torch.cuda.max_memory_allocated() - base) / MIB,
            }
            net.zero_grad(set_to_none=True)
            del y, net, layers, x
            torch.cuda.empty_cache()
    print(json.dumps({"tree": __import__("mptorch").__file__, "results": out}))


# --------------------------------------------------------------------- values


def raw(t):
    """Compare as bytes, so NaN payloads and signed zeros are held to it too."""
    return t.detach().cpu().contiguous().view(torch.uint8).clone()


def gemm_outputs(dev):
    out = {}
    for m, k, n in SHAPES:
        for ta in (False, True):
            for tb in (False, True):
                a = torch.randn(k, m, device=dev) if ta else torch.randn(m, k, device=dev)
                b = torch.randn(n, k, device=dev) if tb else torch.randn(k, n, device=dev)
                for mode in MODES:
                    pb = 4 if mode is SR else 0
                    tag = f"{m}x{k}x{n} ta{int(ta)} tb{int(tb)} {mode.name}"
                    torch.manual_seed(1234)
                    out[f"bk split {tag}"] = raw(
                        binaryK_matmul(
                            a,
                            b,
                            trans_a=ta,
                            trans_b=tb,
                            **BK,
                            rounding_mode=mode,
                            mul_prng_bits=pb,
                            acc_prng_bits=pb,
                        )
                    )
                    torch.manual_seed(1234)
                    out[f"bk fma {tag}"] = raw(
                        binaryK_matmul_fma(
                            a,
                            b,
                            trans_a=ta,
                            trans_b=tb,
                            fma_K=8,
                            fma_P=4,
                            rounding_mode=mode,
                            fma_prng_bits=pb,
                        )
                    )
                    for nb, fmt in ((1, SFP1), (8, SFP8)):
                        torch.manual_seed(1234)
                        out[f"sfp split nb{nb} {tag}"] = raw(
                            superfp_matmul(
                                a,
                                b,
                                trans_a=ta,
                                trans_b=tb,
                                **fmt,
                                rounding_mode=mode,
                                mul_prng_bits=pb,
                                acc_prng_bits=pb,
                            )
                        )
                        torch.manual_seed(1234)
                        out[f"sfp fma nb{nb} {tag}"] = raw(
                            superfp_matmul_fma(
                                a,
                                b,
                                trans_a=ta,
                                trans_b=tb,
                                fma_man_bits=3,
                                fma_exp_bits=4,
                                fma_normal_binades=nb,
                                fma_bias=7,
                                rounding_mode=mode,
                                fma_prng_bits=pb,
                            )
                        )
    return out


def elem_outputs(dev):
    out = {}
    x0 = torch.randn(4099, device=dev) * torch.logspace(-20, 20, 4099, device=dev)
    x0 = torch.cat(
        [x0, torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan")], device=dev)]
    )
    man = {torch.float32: 23, torch.float64: 23, torch.bfloat16: 7, torch.float16: 10}
    for dt, mb in man.items():
        x = x0.to(dt)
        for mode in MODES:
            pb = min(4, mb - 4) if mode is SR else 0
            for sat in SATS:
                tag = f"{dt} {mode.name} {sat.name}"
                torch.manual_seed(1234)
                out[f"bk quant {tag}"] = raw(
                    binaryK_quantize(
                        x,
                        8,
                        4,
                        prng_bits=pb,
                        rounding_mode=mode,
                        saturation_mode=sat,
                        subnormals_mode=SubnormalsMode.SUBNORMALS,
                    )
                )
                torch.manual_seed(1234)
                out[f"sfp quant {tag}"] = raw(
                    superfp_quantize(
                        x,
                        3,
                        4,
                        1,
                        7,
                        prng_bits=pb,
                        rounding_mode=mode,
                        saturation_mode=sat,
                    )
                )
    return out


def block_outputs(dev):
    """The block formats (``mptorch.BlockFormat``, ``block_pack``,
    ``block_matmul``): every preset and two 2D tiles packed and quantized along
    both axes in every mode, and the block GEMM over three pairings, both macs,
    every mode and the three accumulate algorithms past NAIVE. Captured under
    ``--block`` only, so captures of a tree without the block ops (one from
    before 6f4b3fd) still compare."""
    import dataclasses

    from mptorch import (
        MXFP4_E2M1,
        MXFP6_E2M3,
        MXFP6_E3M2,
        MXFP8_E4M3,
        MXFP8_E5M2,
        NVFP4,
        AccumulateAlgorithm,
        BinaryK,
    )
    from mptorch.quant import block_matmul, block_pack, block_quantize

    formats = {
        "mxfp8_e4m3": MXFP8_E4M3,
        "mxfp8_e5m2": MXFP8_E5M2,
        "mxfp6_e2m3": MXFP6_E2M3,
        "mxfp6_e3m2": MXFP6_E3M2,
        "mxfp4_e2m1": MXFP4_E2M1,
        "nvfp4": NVFP4,
        "nvfp4_16x16": dataclasses.replace(NVFP4, block_rows=16),
        "mxfp8_32x32": dataclasses.replace(MXFP8_E4M3, block_rows=32),
    }
    out = {}
    x = torch.randn(96, 160, device=dev) * torch.logspace(-6, 6, 160, device=dev)
    x[3, :7] = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-40, -1e-40])
    for name, fmt in formats.items():
        for mode in MODES:
            f = fmt
            if mode is SR:
                f = dataclasses.replace(fmt, elem=dataclasses.replace(fmt.elem, prng_bits=4))
            for axis in (0, 1):
                tag = f"{name} axis{axis} {mode.name}"
                torch.manual_seed(1234)
                out[f"quant {tag}"] = raw(block_quantize(x, f, axis, rounding=mode))
                torch.manual_seed(1234)
                p = block_pack(x, f, axis, rounding=mode)
                out[f"pack data {tag}"] = raw(p.data)
                out[f"pack scales {tag}"] = raw(p.scales)
    a = torch.randn(37, 200, device=dev)
    b = torch.randn(200, 29, device=dev)
    acc = BinaryK(12, 7, prng_bits=4)
    for pair, (fa, fb) in {
        "mxfp8": (MXFP8_E4M3, MXFP8_E4M3),
        "mxfp4_mxfp6": (MXFP4_E2M1, MXFP6_E3M2),
        "nvfp4_16x16": (NVFP4, formats["nvfp4_16x16"]),
    }.items():
        pa, pb = block_pack(a, fa), block_pack(b, fb, 0)
        for fused in (False, True):
            for mode in MODES:
                tag = f"{pair} {'fused' if fused else 'split'} {mode.name}"
                torch.manual_seed(1234)
                out[f"gemm {tag}"] = raw(block_matmul(pa, pb, acc=acc, fused=fused, rounding=mode))
            for alg, bs in ((AccumulateAlgorithm.KAHAN, None), (AccumulateAlgorithm.BLOCK, 8)):
                tag = f"{pair} {'fused' if fused else 'split'} {alg.name} {RoundMode.RNE.name}"
                out[f"gemm {tag}"] = raw(
                    block_matmul(
                        pa, pb, acc=acc, fused=fused, accumulate_algorithm=alg, block_size=bs
                    )
                )
        tag = f"{pair} split TREE {RoundMode.RNE.name}"
        out[f"gemm {tag}"] = raw(
            block_matmul(
                pa, pb, acc=acc, accumulate_algorithm=AccumulateAlgorithm.TREE, block_size=16
            )
        )
    return out


def run_capture(args):
    torch.manual_seed(1234)
    d = {f"gemm | {k}": v for k, v in gemm_outputs(args.device).items()}
    d.update({f"elem | {k}": v for k, v in elem_outputs(args.device).items()})
    if args.block:
        d.update({f"block | {k}": v for k, v in block_outputs(args.device).items()})
    torch.save(d, args.out)
    print(f"{len(d)} outputs -> {args.out}  ({__import__('mptorch').__file__})")


def run_compare(args):
    before = torch.load(args.before)
    after = torch.load(args.after)
    if before.keys() != after.keys():
        raise SystemExit("the two captures do not cover the same configurations")
    same: collections.Counter = collections.Counter()
    diff: collections.Counter = collections.Counter()
    moved = []
    for key in before:
        kind = (
            "GEMM"
            if key.startswith("gemm")
            else "block"
            if key.startswith("block")
            else "elementwise"
        )
        bucket = f"{kind}, {'SR' if f' {SR.name} ' in f' {key} ' else 'deterministic'}"
        if torch.equal(before[key], after[key]):
            same[bucket] += 1
        else:
            diff[bucket] += 1
            moved.append(key)
    for bucket in sorted(set(same) | set(diff)):
        total = same[bucket] + diff[bucket]
        tail = f"   ({diff[bucket]} moved)" if diff[bucket] else ""
        print(f"{bucket:28s} {same[bucket]:4d}/{total:4d} bit-identical{tail}")
    # The elementwise SR rows are the one exception by design (a build from
    # before 64982e6 draws them from a materialized tensor); anything else
    # moving is a bug.
    unexpected = [k for k in moved if not (k.startswith("elem") and SR.name in k)]
    if unexpected:
        print(f"\nUNEXPECTED: {len(unexpected)} moved beyond elementwise SR, e.g. {unexpected[:3]}")
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("time")
    t.add_argument(
        "--group",
        required=True,
        choices=["gpu_gemm", "gpu_mixed", "cpu_gemm", "gpu_elem", "cpu_elem"],
    )
    t.add_argument("--reps", type=int, default=7)
    t.add_argument("--warmup", type=int, default=2)
    t.add_argument("--n", type=int, default=1024)
    t.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    t.set_defaults(fn=run_timing)

    m = sub.add_parser("memory")
    m.add_argument("--width", type=int, default=2048)
    m.add_argument("--batch", type=int, default=8192)
    m.set_defaults(fn=run_memory)

    c = sub.add_parser("capture")
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="cuda")
    c.add_argument(
        "--block",
        action="store_true",
        help="also capture the block formats' ops, which a tree before 6f4b3fd lacks",
    )
    c.set_defaults(fn=run_capture)

    d = sub.add_parser("compare")
    d.add_argument("before")
    d.add_argument("after")
    d.set_defaults(fn=run_compare)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
