import torch

from mptorch.quant import binaryK_quantize


def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("CUDA not available. Benchmark requires CUDA for performance measurement.")
        return

    # Benchmark sizes
    sizes = [10_000, 1_000_000, 10_000_000, 50_000_000]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    K, P = 8, 4

    for size in sizes:
        print(f"\n--- Tensor size: {size} elements ---")
        for dtype in dtypes:
            x = torch.randn(size, dtype=dtype, device=device)

            # Warmup
            for _ in range(10):
                binaryK_quantize(x, K, P, is_signed=True)

            torch.cuda.synchronize()

            # Benchmark
            iters = 100
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(iters):
                binaryK_quantize(x, K, P, is_signed=True)
            end_event.record()

            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event) / iters
            # Read + Write throughput
            throughput_gbps = (size * x.element_size() * 2) / (elapsed_time_ms * 1e-3) / (1024**3)

            print(
                f"dtype: {dtype!s:<15} | Time: {elapsed_time_ms:8.4f} ms | "
                f"Throughput: {throughput_gbps:8.2f} GB/s"
            )


if __name__ == "__main__":
    benchmark()
