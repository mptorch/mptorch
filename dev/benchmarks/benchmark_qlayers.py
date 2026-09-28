"""Layer timings: QLinear and QConv2d, and the conv passes on the GEMM core.

``python dev/benchmarks/benchmark_qlayers.py`` times the two layers with
elementwise quantizers (the default, ``layers``). ``conv-gemm`` times each
pass of a convolution three ways, with the peak memory each allocates beyond
its inputs: the gathered conv ops (``mptorch.quant.conv_formats``), the same
arithmetic over an explicit ``unfold`` (the forward and the weight gradient as
one batched GEMM over the unfolded input, the input gradient as a GEMM into
the column space followed by ``F.fold``, whose sum is float32 and outside the
simulated arithmetic), and cuDNN in float32. Min of ``--rounds`` rounds of
``--iters`` calls each, the three alternating (``dev/gemm_roadmap.md``,
*Timing on this machine*).
"""

import argparse
import time

import torch
import torch.nn.functional as F
import torch.nn.grad as grad

from mptorch import BinaryK
from mptorch.number import RoundMode
from mptorch.quant import (
    QAffineFormats,
    QConv2d,
    QLinear,
    SplitMac,
    binaryK_quantize,
    conv_formats,
)
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops import _run_gemm


def benchmark_layer(layer_fn, input_shape, dtype, device="cuda", num_iters=100, warmup_iters=20):
    if device == "cuda" and not torch.cuda.is_available():
        return None, None

    layer = layer_fn(dtype=dtype, device=device)
    x = torch.randn(*input_shape, dtype=dtype, device=device, requires_grad=True)

    # Warmup
    for _ in range(warmup_iters):
        out = layer(x)
        out.sum().backward()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.perf_counter()

    for _ in range(num_iters):
        out = layer(x)
        out.sum().backward()

    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event) / num_iters
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        end_time = time.perf_counter()
        time_ms = (end_time - start_time) * 1000 / num_iters
        mem_mb = 0  # Cannot easily track peak memory on CPU in this way

    return time_ms, mem_mb


def run_benchmarks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {device.upper()}")

    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    # We will use binaryK_quantize for everything, but configure it for lower precision
    # when testing the half-precision dtypes to simulate a true mixed precision setup
    # actually, binaryK_quantize processes at the same precision regardless of container dtype,
    # but the container dtype determines what tensor cores are used by the fallback F.linear.

    def get_formats():
        def quant_fn(x):
            return binaryK_quantize(x, K=8, P=4, rounding_mode=RoundMode.RNE)

        return QAffineFormats(
            weight_quant=quant_fn,
            input_quant=quant_fn,
            igrad_quant=quant_fn,
            wgrad_quant=quant_fn,
        )

    print("\n--- QLinear Benchmark (Batch=1024, In=1024, Out=1024) ---")

    def linear_factory(dtype, device):
        return QLinear(1024, 1024, bias=False, formats=get_formats(), device=device, dtype=dtype)

    linear_input = (1024, 1024)

    print(f"{'Dtype':<15} | {'Time (ms/iter)':<15} | {'Peak Memory (MB)':<15}")
    print("-" * 50)
    for dtype in dtypes:
        t_ms, mem_mb = benchmark_layer(linear_factory, linear_input, dtype, device)
        if t_ms is not None:
            print(f"{dtype!s:<15} | {t_ms:<15.4f} | {mem_mb:<15.2f}")

    print("\n--- QConv2d Benchmark (Batch=64, C=64, H=32, W=32) ---")

    def conv_factory(dtype, device):
        return QConv2d(
            64,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
            formats=get_formats(),
            device=device,
            dtype=dtype,
        )

    conv_input = (64, 64, 32, 32)

    print(f"{'Dtype':<15} | {'Time (ms/iter)':<15} | {'Peak Memory (MB)':<15}")
    print("-" * 50)
    for dtype in dtypes:
        t_ms, mem_mb = benchmark_layer(conv_factory, conv_input, dtype, device)
        if t_ms is not None:
            print(f"{dtype!s:<15} | {t_ms:<15.4f} | {mem_mb:<15.2f}")


# (batch, in channels, out channels, spatial, kernel, stride, padding)
CONV_SHAPES = [
    (32, 64, 64, 56, 3, 1, 1),
    (32, 64, 128, 56, 3, 2, 1),
    (32, 128, 128, 28, 3, 1, 1),
    (32, 256, 256, 14, 3, 1, 1),
    (32, 256, 64, 14, 1, 1, 0),
]


def _unfold_passes(mac):
    """The three passes as explicit unfolds plus the flat GEMM op."""
    spec = spec_for_mac(mac)

    def fwd(x, w, s, p):
        B, _, H, W = x.shape
        cout, k = w.shape[0], w.shape[2]
        cols = F.unfold(x, k, padding=p, stride=s)  # [B, C*k*k, L]
        a = w.reshape(1, cout, -1).expand(B, -1, -1)
        y = _run_gemm(spec, a.contiguous(), cols, False, False)
        oh = (H + 2 * p - k) // s + 1
        return y.reshape(B, cout, oh, -1)

    def igrad(gy, w, size, s, p):
        B, cout = gy.shape[:2]
        k = w.shape[2]
        wt = w.reshape(1, cout, -1).expand(B, -1, -1)
        cols = _run_gemm(spec, wt.contiguous(), gy.reshape(B, cout, -1), True, False)
        return F.fold(cols, size[2:], k, padding=p, stride=s)

    def wgrad(gy, x, k, s, p):
        B, cout = gy.shape[:2]
        cols = F.unfold(x, k, padding=p, stride=s)  # [B, C*k*k, L]
        a = gy.reshape(B, cout, -1).permute(1, 0, 2).reshape(cout, -1)
        b = cols.permute(0, 2, 1).reshape(-1, cols.shape[1])
        return _run_gemm(spec, a, b, False, False).reshape(cout, x.shape[1], k, k)

    return fwd, igrad, wgrad


def _time(fn, iters):
    """ms per call over `iters` calls, and the peak allocation beyond the start."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
        del out
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, (torch.cuda.max_memory_allocated() - base) / 2**20


def _conv_runs(passes, x, w, gy, k, s, p):
    """The three implementations of each pass, as thunks over one shape."""
    (gfwd, gigrad, gwgrad), (ufwd, uigrad, uwgrad) = passes
    geo = dict(stride=s, padding=p, dilation=1, groups=1, nd=2)
    return {
        "fwd": (
            lambda: gfwd(x, w, None, **geo),
            lambda: ufwd(x, w, s, p),
            lambda: F.conv2d(x, w, None, s, p),
        ),
        "igrad": (
            lambda: gigrad(gy, w, input_size=x.shape, **geo),
            lambda: uigrad(gy, w, x.shape, s, p),
            lambda: grad.conv2d_input(x.shape, w, gy, s, p),
        ),
        "wgrad": (
            lambda: gwgrad(gy, x, weight_size=w.shape, **geo),
            lambda: uwgrad(gy, x, k, s, p),
            lambda: grad.conv2d_weight(x, w.shape, gy, s, p),
        ),
    }


def run_conv_gemm(rounds: int, iters: int, mode: RoundMode):
    if not torch.cuda.is_available():
        print("conv-gemm needs a CUDA device")
        return
    mac = SplitMac(BinaryK(8, 4), BinaryK(12, 7), rounding=mode)
    f = conv_formats(mac)
    passes = ((f.fwd_math, f.bwd_igrad_math, f.bwd_wgrad_math), _unfold_passes(mac))
    print(f"conv passes, {mac}, min of {rounds} rounds x {iters} calls: ms / peak MB")
    head = f"{'shape':<34} {'pass':<6} {'gathered':>16} {'unfold':>16} {'cuDNN fp32':>16}"
    print(head)
    print("-" * len(head))
    for B, C, Cout, S, k, s, p in CONV_SHAPES:
        x = torch.randn(B, C, S, S, device="cuda")
        w = torch.randn(Cout, C, k, k, device="cuda")
        o = (S + 2 * p - k) // s + 1
        gy = torch.randn(B, Cout, o, o, device="cuda")
        runs = _conv_runs(passes, x, w, gy, k, s, p)
        label = f"{B}x{C}x{S}x{S} -> {Cout}, {k}x{k} s{s}"
        for name, fns in runs.items():
            for f in fns:  # warm up, and compile the cuDNN plan
                f()
            best = [(float("inf"), 0.0)] * 3
            for _ in range(rounds):
                for i, f in enumerate(fns):
                    t, m = _time(f, iters)
                    best[i] = (min(best[i][0], t), max(best[i][1], m))
            cells = " ".join(f"{t:8.2f} /{m:6.0f}" for t, m in best)
            print(f"{label:<34} {name:<6} {cells}")
            label = ""
        del x, w, gy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", nargs="?", default="layers", choices=["layers", "conv-gemm"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--mode", default="RNE", choices=[m.name for m in RoundMode])
    args = parser.parse_args()
    if args.what == "layers":
        run_benchmarks()
    else:
        run_conv_gemm(args.rounds, args.iters, RoundMode[args.mode])
