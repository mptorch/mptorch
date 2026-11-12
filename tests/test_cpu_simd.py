"""The CPU GEMM's vectorized RNE tiles return the scalar kernel's words (phase P).

``cpu/gemm_rne_tile.cpp`` runs a binaryK RNE GEMM's packed tiles through a
vector loop over the shipped ``cast_binaryK_rne_fast`` when its formats admit
it, and ``MPTORCH_CPU_SIMD``, read when the extension loads, picks the scalar
loop (``0``) or caps the ISA (``avx2``). So each case here runs in an
interpreter per setting, and their results must agree bit for bit, NaNs
included. The cases are the tile's
(split, product only, fused, conv), its edges (ragged tiles, batches,
transposes, zeros, binary32 subnormals, overflow, infinities, NaNs, float16
and bfloat16 operands), and formats it hands back to the scalar loop
(unsigned, NORMALS, SAT_PROPAGATE), which must agree trivially.
"""

import os
import subprocess
import sys
import textwrap

import torch

_CASES = textwrap.dedent(
    """
    import sys
    import torch
    from mptorch import BinaryK, RoundMode, SaturationMode, SubnormalsMode
    from mptorch.quant import (
        QConv2d, SplitMac, FusedMac, binaryK_matmul, binaryK_matmul_fma, conv_formats,
    )

    def operands(m, k, n, dtype, seed):
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(m, k, generator=g) * 3
        b = torch.randn(k, n, generator=g) * 3
        flat = a.view(-1)
        specials = [0.0, -0.0, 1e-41, -1e-41, 1e-30, 3e38, -3e38, float("inf"),
                    float("-inf"), float("nan"), 65504.0, 1e5, 2.0**-20]
        for i, v in enumerate(specials):
            flat[(i * 37) % flat.numel()] = v
        return a.to(dtype), b.to(dtype)

    out = {}
    shapes = [(64, 64, 64), (37, 70, 45), (5, 33, 9)]
    formats = [(8, 4, 16, 11), (16, 11, 16, 11), (12, 7, 16, 8), (6, 3, 10, 6), (24, 16, 24, 16)]
    # a result holds the accumulate format's precision (mptorch refuses more)
    precision = {torch.float32: 24, torch.float16: 11, torch.bfloat16: 8}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for s, (m, k, n) in enumerate(shapes):
            a, b = operands(m, k, n, dtype, s)
            for mk, mp, ak, ap in formats:
                if ap > precision[dtype]:
                    continue
                kw = dict(mul_K=mk, mul_P=mp, acc_K=ak, acc_P=ap)
                key = f"{dtype}-{m}x{k}x{n}-{mk}p{mp}-{ak}p{ap}"
                out["split-" + key] = binaryK_matmul(a, b, **kw)
                out["split-sat-" + key] = binaryK_matmul(
                    a, b, **kw, saturation_mode=SaturationMode.SAT_FINITE
                )
                out["mul-only-" + key] = binaryK_matmul(
                    a, b, mul_K=mk, mul_P=mp, accumulate_quant=False
                )
                out["fused-" + key] = binaryK_matmul_fma(a, b, fma_K=ak, fma_P=ap)
            # the scalar loop's formats, which must agree trivially
            out[f"unsigned-{dtype}-{s}"] = binaryK_matmul(
                a.abs(), b.abs(), mul_K=8, mul_P=4, acc_K=16, acc_P=8, mul_is_signed=False,
                acc_is_signed=False,
            )
            out[f"normals-{dtype}-{s}"] = binaryK_matmul(
                a, b, mul_K=8, mul_P=4, acc_K=16, acc_P=8,
                subnormals_mode=SubnormalsMode.NORMALS,
            )
            out[f"propagate-{dtype}-{s}"] = binaryK_matmul_fma(
                a, b, fma_K=16, fma_P=8, saturation_mode=SaturationMode.SAT_PROPAGATE
            )
    g = torch.Generator().manual_seed(9)
    a = torch.randn(3, 33, 40, generator=g)
    b = torch.randn(40, 17, generator=g)
    out["batched"] = binaryK_matmul(a, b, mul_K=8, mul_P=4, acc_K=16, acc_P=11)
    out["transposed"] = binaryK_matmul(
        a[0].t().contiguous().t(), b.t().contiguous().t(), mul_K=8, mul_P=4, acc_K=16, acc_P=11
    )
    torch.manual_seed(3)
    split_mac = SplitMac(BinaryK(8, 4), BinaryK(16, 11))
    conv = QConv2d(3, 5, 3, padding=1, formats=conv_formats(split_mac))
    fconv = QConv2d(3, 5, 3, stride=2, formats=conv_formats(FusedMac(BinaryK(16, 11))))
    x = torch.randn(2, 3, 11, 13, generator=g)
    with torch.no_grad():
        out["conv-split"] = conv(x)
        out["conv-fused"] = fconv(x)
    torch.save(out, sys.argv[1])
    """
)


def _run(tmp_path, simd: str | None) -> dict:
    env = dict(os.environ)
    env.pop("MPTORCH_CPU_SIMD", None)
    if simd is not None:
        env["MPTORCH_CPU_SIMD"] = simd
    target = tmp_path / f"out_{simd}.pt"
    script = tmp_path / "cases.py"
    script.write_text(_CASES)
    subprocess.run(
        [sys.executable, "-W", "ignore", str(script), str(target)],
        check=True,
        env=env,
        cwd=tmp_path,
    )
    return torch.load(target)


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.element_size() == 4 else t.view(torch.int16)


def test_the_vector_tiles_return_the_scalar_kernels_words(tmp_path):
    """The scalar loop, the widest tile the CPU has, and AVX2 at most (the
    same as the widest on a CPU without AVX-512, and on arm64)."""
    scalar = _run(tmp_path, "0")
    for simd in (None, "avx2"):
        vector = _run(tmp_path, simd)
        assert scalar.keys() == vector.keys()
        differ = [k for k in scalar if not torch.equal(_words(scalar[k]), _words(vector[k]))]
        assert not differ, (simd, differ)
