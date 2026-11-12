#!/usr/bin/env python3
"""Block formats (dev/continuation_plan.md, phase D): what the kernels cost.

``elementwise``
    block_pack, block_unpack and block_quantize against the elementwise
    quantizer of the element format (binaryK_quantize, superfp_quantize or
    fixedpoint_quantize),
    in GB/s of float32 input, for every preset, three superfp formats, the
    2D tiles (NVFP4 at 16 x 16, MXFP8 at 32 x 32 and 128 x 128), where the
    tile kernel's two passes are measured against the 1D kernel, and the scale
    rules other than each scale's default (ScaleRounding), the three Four
    Over Six rules among them, which quantize every block twice, on 1D
    blocks and on 2D tiles.
``gemm``
    custom_matmul_block against the flat binaryK GEMM on the unpacked
    operands (split and fused, RNE and SR), at 1024^3 and in each of the four
    orientations of the operands.
``layer``
    QLinear forward and backward with block_gemm_formats against
    binaryK_gemm_formats at 4096 x 4096, with a 1D weight format packed per
    hook and a 16 x 16 one packed once per step, and the peak memory of each;
    then, under no_grad, the forward and the memory a layer holds, packed
    once or frozen to its codes (``freeze_packed_``, phase L, on a tree that
    has it).

Each row is the minimum of ``--rounds`` timed runs after a warm-up. Run on a
CUDA device (``--device cpu`` for the CPU kernels, at smaller sizes).
"""

import argparse
import dataclasses
import time
import warnings
from functools import partial

import torch

from mptorch import (
    E2M1,
    E4M3,
    E8M0,
    MXFP4_E2M1,
    MXFP6_E2M3,
    MXFP6_E3M2,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
    BinaryK,
    BlockFormat,
    FixedPoint,
    RoundMode,
    ScaleRounding,
    SuperFP,
    bfp,
)
from mptorch.quant import (
    BlockQuant,
    QLinear,
    binaryK_gemm_formats,
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_quantize,
    block_gemm_formats,
    block_matmul,
    block_pack,
    block_quantize,
    block_unpack,
    fixedpoint_quantize,
    superfp_quantize,
)

warnings.simplefilter("ignore")

PRESETS = {
    "MXFP8_E4M3": MXFP8_E4M3,
    "MXFP8_E5M2": MXFP8_E5M2,
    "MXFP6_E2M3": MXFP6_E2M3,
    "MXFP6_E3M2": MXFP6_E3M2,
    "MXFP4_E2M1": MXFP4_E2M1,
    "NVFP4": NVFP4,
    "NVFP4 16x16": dataclasses.replace(NVFP4, block_rows=16),
    "MXFP8 32x32": dataclasses.replace(MXFP8_E4M3, block_rows=32),
    "MXFP8 128x128": dataclasses.replace(MXFP8_E4M3, block_size=128, block_rows=128),
    "SuperFP 4-bit": BlockFormat(SuperFP(1, 2, 1, 1), E8M0, 32),
    "SuperFP 8-bit": BlockFormat(SuperFP(3, 4, 2, 7), E8M0, 32, elem_max=448.0, nan_code=0x7F),
    "E2M1, SuperFP scale": BlockFormat(E2M1, SuperFP(3, 4, 2, 7), 16),
    "  ...NEAREST": BlockFormat(
        E2M1, SuperFP(3, 4, 2, 7), 16, scale_rounding=ScaleRounding.NEAREST
    ),
    "MXFP4_E2M1 UP": dataclasses.replace(MXFP4_E2M1, scale_rounding=ScaleRounding.UP),
    "MXFP8_E4M3 SELECTIVE": dataclasses.replace(MXFP8_E4M3, scale_rounding=ScaleRounding.SELECTIVE),
    "NVFP4 SELECTIVE": dataclasses.replace(NVFP4, scale_rounding=ScaleRounding.SELECTIVE),
    "NVFP4 4/6 MSE": dataclasses.replace(NVFP4, scale_rounding=ScaleRounding.FOUR_OVER_SIX_MSE),
    "NVFP4 4/6 MAE": dataclasses.replace(NVFP4, scale_rounding=ScaleRounding.FOUR_OVER_SIX_MAE),
    "NVFP4 4/6 ABS_MAX": dataclasses.replace(
        NVFP4, scale_rounding=ScaleRounding.FOUR_OVER_SIX_ABS_MAX
    ),
    "NVFP4 16x16 4/6 MSE": dataclasses.replace(
        NVFP4, block_rows=16, scale_rounding=ScaleRounding.FOUR_OVER_SIX_MSE
    ),
    "NVFP4 16x16 4/6 ABS_MAX": dataclasses.replace(
        NVFP4, block_rows=16, scale_rounding=ScaleRounding.FOUR_OVER_SIX_ABS_MAX
    ),
    "E2M1/E4M3 128x128 4/6 MSE": BlockFormat(
        E2M1, E4M3, 128, block_rows=128, scale_rounding=ScaleRounding.FOUR_OVER_SIX_MSE
    ),
    "bfp8 (MXINT8 shape)": bfp(8, 32),
    "bfp4": bfp(4, 16),
    "bfp12": bfp(12, 16),
}


def timed(f, device, rounds):
    f()
    if device == "cuda":
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        f()
        if device == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def elementwise(args):
    n = args.elements
    x = torch.randn(n // 4096, 4096, device=args.device)
    gb = x.numel() * 4 / 1e9
    print(
        f"{'format':<22}{'pack':>10}{'unpack':>10}{'quantize':>10}{'elem q.':>10}   GB/s of float32"
    )
    for name, fmt in PRESETS.items():
        # A static tensor scale, so no amax pass over the tensor, and one that
        # keeps the largest tile's scale inside E4M3's range: at 1e-3 a tile of
        # 256 Gaussians or more asks for a scale past 448, both of Four Over
        # Six's candidates clamp to it, and the rule's pass is skipped.
        ts = 1e-2 if fmt.has_tensor_scale else None
        e = fmt.elem
        p = block_pack(x, fmt, tensor_scale=ts)
        if isinstance(e, SuperFP):
            quant = partial(superfp_quantize, x, e.man_bits, e.exp_bits, e.normal_binades, e.bias)
        elif isinstance(e, FixedPoint):
            quant = partial(fixedpoint_quantize, x, e.wl, e.fl, symmetric=e.symmetric)
        else:
            quant = partial(binaryK_quantize, x, e.K, e.P, e.bias)
        calls = [
            partial(block_pack, x, fmt, tensor_scale=ts),
            partial(block_unpack, p),
            partial(block_quantize, x, fmt, tensor_scale=ts),
            quant,
        ]
        rows = [timed(f, args.device, args.rounds) for f in calls]
        print(f"{name:<22}" + "".join(f"{gb / t:>10.1f}" for t in rows))


def gemm(args):
    n = args.size
    torch.manual_seed(0)
    a = torch.randn(n, n, device=args.device)
    b = torch.randn(n, n, device=args.device)
    acc = BinaryK(16, 11)
    print(f"{'mac':<14}{'mode':<6}{'orientation':<14}{'block ms':>10}{'flat ms':>10}")
    for ta in (False, True):
        for tb in (False, True):
            pa = block_pack(a, MXFP8_E4M3, -2 if ta else -1)
            pb = block_pack(b, MXFP8_E4M3, -1 if tb else -2)
            ua, ub = pa.unpack(), pb.unpack()
            for fused in (False, True):
                for rm in (RoundMode.RNE, RoundMode.SR):
                    blk = timed(
                        partial(block_matmul, pa, pb, acc=acc, fused=fused, rounding=rm),
                        args.device,
                        args.rounds,
                    )
                    if fused:
                        f = partial(
                            binaryK_matmul_fma, ua, ub, fma_K=16, fma_P=11, rounding_mode=rm
                        )
                    else:
                        f = partial(
                            binaryK_matmul,
                            ua,
                            ub,
                            mul_K=32,
                            mul_P=24,
                            mul_bias=127,
                            acc_K=16,
                            acc_P=11,
                            rounding_mode=rm,
                        )
                    flat = timed(f, args.device, args.rounds)
                    orient = f"{'T' if ta else 'N'}{'T' if tb else 'N'}"
                    mac = "fused" if fused else "split"
                    print(f"{mac:<14}{rm.name:<6}{orient:<14}{blk * 1e3:>10.2f}{flat * 1e3:>10.2f}")


def layer(args):
    n = args.features
    acc = BinaryK(16, 11)
    w16 = dataclasses.replace(NVFP4, block_rows=16)
    x = torch.randn(args.batch, n, device=args.device)
    configs = {
        "binaryK (float operands)": binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=16, acc_P=11),
        "block, 1D weight": block_gemm_formats(NVFP4, NVFP4, acc=acc),
        "block, 16x16 weight packed once": block_gemm_formats(NVFP4, w16, acc=acc),
    }
    configs["block, 16x16 weight packed once"].weight_quant = BlockQuant(w16, axis=1).pack
    print(f"{'layer':<34}{'fwd+bwd ms':>12}{'peak MB':>10}")
    for name, formats in configs.items():
        lin = QLinear(n, n, formats=formats).to(args.device)
        xi = x.clone().requires_grad_()

        def step(lin=lin, xi=xi):
            lin(xi).sum().backward()

        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = timed(step, args.device, args.rounds)
        peak = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else float("nan")
        print(f"{name:<34}{t * 1e3:>12.2f}{peak:>10.1f}")
        del lin, xi, step

    # Inference. "held MB" is what the layer itself holds (its parameters, its
    # buffers and a BlockQuant's memoized packing), counted after a forward;
    # "peak MB" the CUDA allocator's peak over the forwards, from a reset
    # after the layer was built.
    frozen = hasattr(QLinear, "freeze_packed_")
    infer: dict[str, tuple[BlockFormat, bool]] = {
        "1D weight, packed once": (NVFP4, False),
        "16x16 weight, packed once": (w16, False),
    }
    if frozen:
        infer["1D weight, frozen"] = (NVFP4, True)
        infer["16x16 weight, frozen"] = (w16, True)
    print(f"\n{'inference (no_grad)':<34}{'fwd ms':>12}{'held MB':>10}{'peak MB':>10}")
    for name, (wfmt, freeze) in infer.items():
        formats = block_gemm_formats(NVFP4, wfmt, acc=acc)
        quant = BlockQuant(wfmt, axis=1)
        formats.weight_quant = quant.pack
        lin = QLinear(n, n, formats=formats).to(args.device)
        if freeze:
            lin.freeze_packed_()
        with torch.no_grad():
            lin(x)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t = timed(lambda lin=lin: lin(x), args.device, args.rounds)
        tensors = [*lin.parameters(), *lin.buffers()]
        memo = quant._memo[0]
        held = sum(v.nbytes for v in tensors) + (memo[-1].nbytes if memo else 0)
        peak = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else float("nan")
        print(f"{name:<34}{t * 1e3:>12.2f}{held / 2**20:>10.1f}{peak:>10.1f}")
        del lin, formats, quant, tensors, memo  # each row's peak starts from nothing of the last


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("what", choices=["elementwise", "gemm", "layer", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--elements", type=int, default=1 << 26)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--features", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    for what in ("elementwise", "gemm", "layer") if args.what == "all" else (args.what,):
        print(f"== {what} ({args.device})")
        globals()[what](args)


if __name__ == "__main__":
    main()
