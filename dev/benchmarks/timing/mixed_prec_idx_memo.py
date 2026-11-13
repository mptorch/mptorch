"""Cost of the mixed-format `prec_idx` bounds check, by map dtype and layout.

The check is memoized (since a6048fe) so a reused map costs nothing after the
first call. Until 6d41f35 the memo keyed on the *converted* map, which
`prec_idx.to(device, int32).contiguous()` reallocates for anything that is not
already int32-contiguous on the operand device -- so an int64 map (what
`torch.zeros`, `torch.randint` and `torch.arange` hand out without an explicit
dtype) missed on every call and paid back the `aminmax` + `.cpu()` sync. It now
keys on the caller's tensor.

Run: python3 dev/benchmarks/timing/mixed_prec_idx_memo.py
"""

import time

import torch

from mptorch.quant import binaryK_matmul_mixed

MUL_K = [8, 6]
MUL_P = [4, 3]


def run(a, b, prec_idx):
    return binaryK_matmul_mixed(a, b, prec_idx, mul_K=MUL_K, mul_P=MUL_P, acc_prng_bits=0)


def bench(a, b, prec_idx, iters=200):
    for _ in range(20):
        run(a, b, prec_idx)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run(a, b, prec_idx)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def launch_only(a, b, prec_idx, iters=200):
    """Wall time of the host half alone: no final sync, so a memo hit returns
    before the kernel does and a memo miss cannot."""
    for _ in range(20):
        run(a, b, prec_idx)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run(a, b, prec_idx)
    dt = (time.perf_counter() - t0) / iters * 1e3
    torch.cuda.synchronize()
    return dt


def main():
    dev = "cuda"
    torch.manual_seed(0)
    for m, k, n in ((64, 64, 64), (512, 512, 512)):
        a = torch.randn(m, k, device=dev)
        b = torch.randn(k, n, device=dev)
        maps = {
            "int32 dense [M,N]": torch.randint(0, 2, (m, n), device=dev, dtype=torch.int32),
            "int64 dense [M,N]": torch.randint(0, 2, (m, n), device=dev, dtype=torch.int64),
            "int64 per-row [M,1]": torch.randint(0, 2, (m, 1), device=dev, dtype=torch.int64),
            "int32 non-contig": torch.randint(0, 2, (m, 2 * n), device=dev, dtype=torch.int32)[
                :, ::2
            ],
            "cpu int32 dense": torch.randint(0, 2, (m, n), dtype=torch.int32),
        }
        print(f"\n{m}x{k}x{n}   (ms/call)")
        print(f"  {'map':<22}{'end-to-end':>12}{'host-only':>12}")
        for name, pidx in maps.items():
            print(f"  {name:<22}{bench(a, b, pidx):>12.3f}{launch_only(a, b, pidx):>12.3f}")


if __name__ == "__main__":
    main()
