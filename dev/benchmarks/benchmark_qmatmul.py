"""
Perf sanity check for the custom-arithmetic GEMM core
(mptorch.quant.binaryK_matmul / superfp_matmul, see dev/gemm_core_roadmap.md)
against plain torch.matmul on the same shapes. Not a hard gate -- just a way
to see the fused kernel is in the right ballpark, and to compare tile-size
choices when iterating on mptorch/csrc/cuda/custom_matmul_kernel.cu.

Run with: python3 dev/benchmarks/benchmark_qmatmul.py
"""

import time

import torch

from mptorch.quant import binaryK_matmul, superfp_matmul

SHAPES = [
    (256, 256, 256),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 512, 4096),
]


def _time_cpu(fn, iters=5):
    for _ in range(2):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters * 1e3


def _time_cuda(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    timer = _time_cuda if device == "cuda" else _time_cpu

    for M, K, N in SHAPES:
        a = torch.randn(M, K, device=device)
        b = torch.randn(N, K, device=device)

        t_ref = timer(lambda a=a, b=b: a @ b.t())
        t_binaryK_full_acc = timer(
            lambda a=a, b=b: binaryK_matmul(
                a, b, trans_b=True, mul_K=33, mul_P=24, accumulate_quant=False
            )
        )
        t_binaryK_quant_acc = timer(
            lambda a=a, b=b: binaryK_matmul(
                a, b, trans_b=True, mul_K=8, mul_P=4, accumulate_quant=True
            )
        )
        t_superfp = timer(
            lambda a=a, b=b: superfp_matmul(
                a,
                b,
                trans_b=True,
                mul_man_bits=3,
                mul_exp_bits=4,
                mul_normal_binades=8,
                mul_bias=7,
                accumulate_quant=True,
            )
        )

        print(
            f"M={M:<5} K={K:<5} N={N:<5} | "
            f"torch.matmul: {t_ref:8.3f} ms | "
            f"binaryK (mul-only quant): {t_binaryK_full_acc:8.3f} ms | "
            f"binaryK (mul+acc quant): {t_binaryK_quant_acc:8.3f} ms | "
            f"superfp (mul+acc quant): {t_superfp:8.3f} ms"
        )


if __name__ == "__main__":
    benchmark()
