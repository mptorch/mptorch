"""What the batched GEMM is worth against the loop of 2D calls it replaces.

X1 gave the kernel one batch dimension with per-operand strides. This measures
the three things that decision was made on:

    batched      one call, one launch (or one parallel region) over the whole
                 batch -- what `qmatmul` does now
    loop of 2D   the same arithmetic as B separate 2D calls, which is what a
                 caller had to write before, and what the layer hooks still do
                 for the shapes they can fold
    torch.matmul the unquantized reference, for the ratio that says what the
                 simulation costs

on attention-shaped operands, where the batch is real (`batch * heads`) and
the per-element GEMM is small enough that the launch and the parallel region
dominate -- which is the case the batch dimension exists for.

The last row is the one that decides whether a transposed operand needs a
copy: `q @ k.mT` against `q @ k.mT.contiguous()`, which is the same values
either way (tests/test_qmatmul_batched.py) and should now be the same time as
well, minus the copy.

Run: python3 dev/benchmarks/benchmark_qmatmul.py [--device cuda]
"""

import argparse
import time
from typing import Any

import torch

from mptorch import BinaryK
from mptorch.quant import SplitMac, binaryK_matmul, qmatmul

BK: dict[str, Any] = dict(mul_K=8, mul_P=4, acc_K=8, acc_P=4)
MAC = SplitMac(BinaryK(8, 4), BinaryK(8, 4))

# (name, a shape, b shape). The first two are one attention head's two GEMMs
# at batch 32 x 12 heads; the third is the QLinear shape, which folds its
# batch into M rather than using the batch dimension at all; the fourth is a
# broadcast weight against a batched activation.
SHAPES = [
    ("attention q@k^T [32,12,128,64]", (32, 12, 128, 64), (32, 12, 64, 128)),
    ("attention p@v   [32,12,128,128]", (32, 12, 128, 128), (32, 12, 128, 64)),
    ("linear-shaped   [32,128,512]", (32, 128, 512), (512, 512)),
    ("broadcast b     [64,64,128]", (64, 64, 128), (128, 128)),
]

# The same four shapes an order of magnitude down: the CPU kernel is ~500x
# slower per element than the GPU one, and the point of the comparison is the
# ratio, which the smaller batch shows just as well.
SHAPES_CPU = [
    ("attention q@k^T [4,4,64,32]", (4, 4, 64, 32), (4, 4, 32, 64)),
    ("attention p@v   [4,4,64,64]", (4, 4, 64, 64), (4, 4, 64, 32)),
    ("linear-shaped   [8,64,128]", (8, 64, 128), (128, 128)),
    ("broadcast b     [16,32,64]", (16, 32, 64), (64, 64)),
]


def timed(fn, *, device: str, reps: int = 5, warmup: int = 2) -> float:
    """Milliseconds per call, min of `reps`."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def loop_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The same GEMM as a Python loop of 2D calls, batch element by element."""
    a2 = a.reshape(-1, *a.shape[-2:])
    b2 = b.reshape(-1, *b.shape[-2:]) if b.dim() > 2 else None
    out = [binaryK_matmul(a2[i], b2[i] if b2 is not None else b, **BK) for i in range(a2.shape[0])]
    return torch.stack(out).reshape(*a.shape[:-1], out[0].shape[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = ap.parse_args()
    dev = args.device

    print(f"{dev}, ms/call (min of 5)\n")
    print(f"{'shape':<34}{'batched':>10}{'loop of 2D':>12}{'speedup':>9}{'torch':>9}")
    for name, sa, sb in SHAPES if dev == "cuda" else SHAPES_CPU:
        a = torch.randn(*sa, device=dev)
        b = torch.randn(*sb, device=dev)
        batched = timed(lambda x=a, y=b: binaryK_matmul(x, y, **BK), device=dev)
        looped = timed(lambda x=a, y=b: loop_2d(x, y), device=dev)
        plain = timed(lambda x=a, y=b: x @ y, device=dev)
        print(f"{name:<34}{batched:>10.2f}{looped:>12.2f}{looped / batched:>8.1f}x{plain:>9.3f}")

    # The transposed-operand fold: same values, and now no copy.
    print(f"\n{'transposed operand':<34}{'view':>10}{'contiguous':>12}{'ratio':>9}")
    head = (32, 12, 128, 64) if dev == "cuda" else (4, 4, 64, 32)
    q = torch.randn(*head, device=dev)
    k = torch.randn(*head, device=dev)
    view = timed(lambda: binaryK_matmul(q, k.mT, **BK), device=dev)
    copied = timed(lambda: binaryK_matmul(q, k.mT.contiguous(), **BK), device=dev)
    print(f"{f'q @ k.mT {list(head)}':<34}{view:>10.2f}{copied:>12.2f}{copied / view:>8.2f}x")

    # And what the differentiable entry point adds on top of the op.
    print(f"\n{'entry point':<34}{'fwd':>10}{'fwd+bwd':>12}")
    a = torch.randn(*head, device=dev, requires_grad=True)
    b = torch.randn(*head[:-2], head[-1], head[-2], device=dev, requires_grad=True)

    def fwd_bwd():
        out = qmatmul(a, b, MAC)
        out.sum().backward()

    print(
        f"{f'qmatmul {list(head)}':<34}"
        f"{timed(lambda: qmatmul(a.detach(), b.detach(), MAC), device=dev):>10.2f}"
        f"{timed(fwd_bwd, device=dev):>12.2f}"
    )


if __name__ == "__main__":
    main()
