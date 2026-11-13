"""The narrow_float64 op against its references, over far more words than pytest can.

`carrier=torch.float64` on a float32, float16 or bfloat16 tensor stores its
binary64 result with `torch.ops.mptorch.narrow_float64` (since 323e0fe), whose
rounding is `csrc/common/narrow_binary64.h`. This checks the op, CPU and CUDA,
against two references written apart from it:

    float32              the hardware conversion, `Tensor.float()`
    float16, bfloat16    `tests/test_float64_carrier.py`'s torch spelling
                         (step from the exponent field, `x / step`,
                         `torch.round`), itself held to a Fraction reference

on two input sets: every exponent field with structured mantissas (random,
ties at the target's last kept bit on odd and even significands, and a word
either side) and uniformly random words. A word matches when it is the same
word, or when both are NaN; CPU against CUDA must match word for word, NaN
payloads included.

    python3 dev/benchmarks/sweeps/narrow_float64_sweep.py short   # ~2^19 structured, 2^24 random
    python3 dev/benchmarks/sweeps/narrow_float64_sweep.py sweep   # ~2^25 structured, 2^32 random
                                                           # on CUDA and 2^28 on the CPU
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import mptorch  # noqa: E402, F401  (registers the op)
from tests.test_float64_carrier import _every_field, _narrowed_by_torch  # noqa: E402

OP = torch.ops.mptorch.narrow_float64.default
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
BITS = {4: torch.int32, 2: torch.int16}


def mismatches(x: torch.Tensor, dtype: torch.dtype) -> int:
    got = OP(x, dtype)
    want = x.float() if dtype is torch.float32 else _narrowed_by_torch(x, dtype)
    bits = BITS[got.element_size()]
    same = (got.view(bits) == want.view(bits)) | (got.isnan() & want.isnan())
    return int((~same).sum())


def device_mismatches(x: torch.Tensor, dtype: torch.dtype) -> int:
    bits = BITS[torch.empty(0, dtype=dtype).element_size()]
    return int((OP(x.cuda(), dtype).cpu().view(bits) != OP(x, dtype).view(bits)).sum())


def random_words(n: int, device: str, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(
        -(1 << 63), (1 << 63) - 1, (n,), dtype=torch.int64, device=device, generator=gen
    ).view(torch.float64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("mode", choices=("short", "sweep"))
    args = p.parse_args()
    per_field = 64 if args.mode == "short" else 4096
    random_log2 = {"short": (24, 24), "sweep": (32, 28)}[args.mode]  # (CUDA, CPU)
    chunk = 1 << 24
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    failed = 0
    for dtype in DTYPES:
        t0 = time.perf_counter()
        structured = _every_field(dtype, per_field, torch.Generator().manual_seed(1))
        row = {"structured": structured.numel()}
        for device in devices:
            row[f"{device} structured"] = sum(
                mismatches(structured[i : i + chunk].to(device), dtype)
                for i in range(0, structured.numel(), chunk)
            )
        if "cuda" in devices:
            row["cpu == cuda"] = sum(
                device_mismatches(structured[i : i + chunk], dtype)
                for i in range(0, structured.numel(), chunk)
            )
        for device, log2 in zip(("cuda", "cpu"), random_log2, strict=True):
            if device not in devices:
                continue
            gen = torch.Generator(device=device).manual_seed(2)
            bad = 0
            for _ in range(0, 1 << log2, chunk):
                bad += mismatches(random_words(min(chunk, 1 << log2), device, gen), dtype)
            row[f"{device} random 2^{log2}"] = bad
        bad = sum(v for k, v in row.items() if k != "structured")
        failed += bad
        detail = ", ".join(f"{k} {v}" for k, v in row.items())
        verdict = "PASS" if bad == 0 else "FAIL"
        print(f"{str(dtype):<15} {verdict}  {detail}  ({time.perf_counter() - t0:.1f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
