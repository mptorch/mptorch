import time

import torch

from mptorch.number import RoundMode
from mptorch.quant import QAffineFormats, QConv2d, QLinear, binaryK_quantize


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


if __name__ == "__main__":
    run_benchmarks()
