"""The CPU's vector code returns the scalar loops' words.

The vector code, each run where its formats admit it:

- ``cpu/gemm_rne_tile.cpp``, a binaryK RNE GEMM's packed tiles, over the
  shipped ``cast_binaryK_rne_fast``, and ``cpu/gemm_odd_tile.cpp``, the same
  with the odd carrier step (``carrier_rounding=RoundMode.RO``);
- ``cpu/gemm_word_tile.cpp`` and ``cpu/gemm_word_odd_tile.cpp``, the same under
  RNA, RU, RD, RZ and RO, over ``common/cast_binaryK_word.h``;
- ``cpu/gemm_palette_tile.h``, the RNE tiles per palette entry;
- ``cpu/pointwise_rne.cpp``, the fused pointwise ops (``qadd`` and its
  siblings);
- ``cpu/binaryK_quant_vector.cpp``, the elementwise quantizer under RNE
  (over ``cast_binaryK_rne_fast``) and the five modes the word casts serve.

``MPTORCH_CPU_SIMD``, read when the extension loads, picks the scalar loop
(``0``) or caps the ISA (``avx2``), so each case here runs in an interpreter
per setting, and their results must agree bit for bit, NaNs included. The
cases are the tiles' (split, product only, fused, conv), their edges (ragged
tiles, batches, transposes, zeros, binary32 subnormals, overflow, infinities,
NaNs, float16 and bfloat16 operands), the quantizer's (random words, parallel
chunks, unaligned starts, the in-place op), and formats they hand back to the
scalar loop (unsigned, NORMALS, SAT_PROPAGATE), which must agree trivially.
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

    def has_odd():  # a build with MPTORCH_NO_ODD=1 has no odd kernels to compare
        x = torch.ones(1)
        try:
            torch.ops.mptorch._carrier_odd(x, x, x, 0)
        except RuntimeError:
            return False
        return True

    ODD = has_odd()
    # the word tiles' modes (cpu/gemm_word_tile.h)
    WORD_MODES = (RoundMode.RNA, RoundMode.RU, RoundMode.RD, RoundMode.RZ, RoundMode.RO)
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
                # the word tiles, in each of their modes, and their odd twins
                for mode in WORD_MODES:
                    wkey = f"{mode.name}-{key}"
                    w = dict(rounding_mode=mode)
                    out["word-split-" + wkey] = binaryK_matmul(a, b, **kw, **w)
                    out["word-split-sat-" + wkey] = binaryK_matmul(
                        a, b, **kw, saturation_mode=SaturationMode.SAT_FINITE, **w
                    )
                    out["word-mul-only-" + wkey] = binaryK_matmul(
                        a, b, mul_K=mk, mul_P=mp, accumulate_quant=False, **w
                    )
                    out["word-fused-" + wkey] = binaryK_matmul_fma(a, b, fma_K=ak, fma_P=ap, **w)
                    if ODD:
                        o = dict(carrier_rounding=RoundMode.RO, **w)
                        out["word-odd-split-" + wkey] = binaryK_matmul(a, b, **kw, **o)
                        out["word-odd-mul-only-" + wkey] = binaryK_matmul(
                            a, b, mul_K=mk, mul_P=mp, accumulate_quant=False, **o
                        )
                        out["word-odd-fused-" + wkey] = binaryK_matmul_fma(
                            a, b, fma_K=ak, fma_P=ap, **o
                        )
                # the odd carrier step's tiles
                odd = dict(carrier_rounding=RoundMode.RO)
                if not ODD:
                    continue
                out["odd-split-" + key] = binaryK_matmul(a, b, **kw, **odd)
                out["odd-split-sat-" + key] = binaryK_matmul(
                    a, b, **kw, saturation_mode=SaturationMode.SAT_FINITE, **odd
                )
                out["odd-mul-only-" + key] = binaryK_matmul(
                    a, b, mul_K=mk, mul_P=mp, accumulate_quant=False, **odd
                )
                out["odd-fused-" + key] = binaryK_matmul_fma(a, b, fma_K=ak, fma_P=ap, **odd)
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
            # ... and the same under a word tile's mode
            out[f"word-unsigned-{dtype}-{s}"] = binaryK_matmul(
                a.abs(), b.abs(), mul_K=8, mul_P=4, acc_K=16, acc_P=8, mul_is_signed=False,
                acc_is_signed=False, rounding_mode=RoundMode.RU,
            )
            out[f"word-normals-{dtype}-{s}"] = binaryK_matmul(
                a, b, mul_K=8, mul_P=4, acc_K=16, acc_P=8,
                subnormals_mode=SubnormalsMode.NORMALS, rounding_mode=RoundMode.RZ,
            )
    # tiles on both sides of the cut-offs below which a tile goes back to the
    # scalar loop whole (four outputs for the RNE and odd tiles, eight for
    # the word tiles)
    for m, k, n in [(1, 70, 1), (1, 40, 3), (2, 40, 2), (1, 40, 7), (8, 40, 1), (3, 40, 3)]:
        a, b = operands(m, k, n, torch.float32, 30)
        for mode in (RoundMode.RNE, RoundMode.RZ):
            kw = dict(mul_K=8, mul_P=4, acc_K=16, acc_P=11, rounding_mode=mode)
            fkw = dict(fma_K=16, fma_P=11, rounding_mode=mode)
            key = f"narrow-{m}x{k}x{n}-{mode.name}"
            out["split-" + key] = binaryK_matmul(a, b, **kw)
            out["fused-" + key] = binaryK_matmul_fma(a, b, **fkw)
            if ODD:
                odd = dict(carrier_rounding=RoundMode.RO)
                out["odd-split-" + key] = binaryK_matmul(a, b, **kw, **odd)
                out["odd-fused-" + key] = binaryK_matmul_fma(a, b, **fkw, **odd)
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
    odd_split = SplitMac(BinaryK(8, 4), BinaryK(16, 11), carrier_rounding=RoundMode.RO)
    odd_fused = FusedMac(BinaryK(16, 11), carrier_rounding=RoundMode.RO)
    oconv = QConv2d(3, 5, 3, padding=1, formats=conv_formats(odd_split))
    ofconv = QConv2d(3, 5, 3, stride=2, formats=conv_formats(odd_fused))
    wconv = QConv2d(3, 5, 3, padding=1, formats=conv_formats(
        SplitMac(BinaryK(8, 4), BinaryK(16, 11), rounding=RoundMode.RZ)
    ))  # fmt: skip
    wfconv = QConv2d(3, 5, 3, stride=2, formats=conv_formats(
        FusedMac(BinaryK(16, 11), rounding=RoundMode.RU)
    ))  # fmt: skip
    x = torch.randn(2, 3, 11, 13, generator=g)
    with torch.no_grad():
        out["conv-split"] = conv(x)
        out["conv-fused"] = fconv(x)
        out["word-conv-split"] = wconv(x)
        out["word-conv-fused"] = wfconv(x)
        if ODD:
            out["odd-conv-split"] = oconv(x)
            out["odd-conv-fused"] = ofconv(x)
    # the palette GEMMs' vector tiles (cpu/gemm_palette_tile.h): every map
    # structure the tile tells apart (one entry, per column, per row, per
    # element, three entries), the three macs and the odd ones, ragged tiles,
    # the specials, an entry the vector tile does not take, and a conv
    from mptorch.quant import Palette, binaryK_matmul_fma_mixed, binaryK_matmul_mixed

    def palette_maps(m, n, g):
        return {
            "one": torch.zeros(1, n, dtype=torch.int32),
            "columns": torch.randint(0, 2, (1, n), generator=g, dtype=torch.int32),
            "halves": (torch.arange(n) >= n // 2).to(torch.int32).view(1, n),
            "rows": torch.randint(0, 2, (m, 1), generator=g, dtype=torch.int32),
            "elements": torch.randint(0, 2, (m, n), generator=g, dtype=torch.int32),
            "three": torch.randint(0, 3, (m, n), generator=g, dtype=torch.int32),
        }

    g = torch.Generator().manual_seed(11)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        ap = min(11, precision[dtype])
        for s, (m, k, n) in enumerate([(64, 64, 64), (37, 70, 45)]):
            a, b = operands(m, k, n, dtype, 20 + s)
            split = dict(mul_K=[8, 16, 12], mul_P=[4, ap, 7], acc_K=[16, 16, 12], acc_P=[ap, ap, 7])
            fused = dict(fma_K=[16, 12, 8], fma_P=[ap, 7, 4])
            for name, p in palette_maps(m, n, g).items():
                key = f"{dtype}-{m}x{k}x{n}-{name}"
                out["pal-split-" + key] = binaryK_matmul_mixed(a, b, p, **split)
                out["pal-mul-only-" + key] = binaryK_matmul_mixed(
                    a, b, p, mul_K=[8, 16, 12], mul_P=[4, ap, 7], accumulate_quant=False
                )
                out["pal-fused-" + key] = binaryK_matmul_fma_mixed(a, b, p, **fused)
                if ODD:
                    odd = dict(carrier_rounding=RoundMode.RO)
                    out["pal-odd-split-" + key] = binaryK_matmul_mixed(a, b, p, **split, **odd)
                    out["pal-odd-fused-" + key] = binaryK_matmul_fma_mixed(a, b, p, **fused, **odd)
            # an unsigned entry, which the vector tile does not take
            p = palette_maps(m, n, g)["columns"]
            out[f"pal-unsigned-{dtype}-{s}"] = binaryK_matmul_mixed(
                a.abs(), b.abs(), p, mul_K=[8, 8], mul_P=[4, 4], acc_K=[16, 16], acc_P=[ap, ap],
                mul_is_signed=False,
            )  # fmt: skip
    # an odd palette whose second entry holds two columns of a one-row tile: a
    # column group narrower than the cut-off, which hands its K-block back
    if ODD:
        a, b = operands(1, 64, 32, torch.float32, 40)
        p = torch.zeros(1, 32, dtype=torch.int32)
        p[0, 5:7] = 1
        for name, call in (("split", binaryK_matmul_mixed), ("fused", binaryK_matmul_fma_mixed)):
            fmts = (
                dict(mul_K=[8, 12], mul_P=[4, 7], acc_K=[16, 16], acc_P=[11, 11])
                if name == "split"
                else dict(fma_K=[16, 12], fma_P=[11, 7])
            )
            out[f"pal-odd-narrow-group-{name}"] = call(
                a, b, p, **fmts, carrier_rounding=RoundMode.RO
            )
    pal_mac = SplitMac(
        Palette([BinaryK(8, 4), BinaryK(12, 7)]), Palette([BinaryK(16, 11), BinaryK(16, 11)])
    )
    # a per-channel map, [N, C, positions]: per row of the forward GEMM
    channels = torch.randint(0, 2, (1, 5, 1), generator=g, dtype=torch.int32)
    pconv = QConv2d(3, 5, 3, padding=1, formats=conv_formats(
        pal_mac, prec_idx=channels.expand(2, 5, 11 * 13).contiguous()
    ))  # fmt: skip
    with torch.no_grad():
        out["pal-conv"] = pconv(x)
    # the fused pointwise ops' vector blocks (cpu/pointwise_rne.cpp): every
    # op, tensor and number operands, several blocks and a ragged tail, and a
    # broadcast operand and formats that keep the scalar loop
    from mptorch.quant import Quant, qadd, qdiv, qmul, qsqrt, qsub

    specials = [0.0, -0.0, 1e-41, -1e-41, 1e-30, 3e38, -3e38, float("inf"),
                float("-inf"), float("nan"), 65504.0, 1e5, 2.0**-20]
    g = torch.Generator().manual_seed(10)
    x0, y0 = torch.randn(3001, generator=g) * 3, torch.randn(3001, generator=g) * 3
    for i, v in enumerate(specials):
        x0[(i * 37) % x0.numel()] = v
        y0[(i * 53) % y0.numel()] = v
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x, y = x0.to(dtype), y0.to(dtype)
        for k, p in [(8, 4), (16, 11), (12, 7), (6, 3), (24, 16)]:
            if p > precision[dtype]:
                continue
            q = Quant(BinaryK(k, p))
            key = f"pointwise-{dtype}-{k}p{p}"
            out[key + "-add"] = qadd(x, y, q)
            out[key + "-sub"] = qsub(x, y, q)
            out[key + "-mul"] = qmul(x, y, q)
            out[key + "-div"] = qdiv(x, y, q)
            out[key + "-sqrt"] = qsqrt(x, q)
            out[key + "-add-number"] = qadd(x, 0.7, q)
            out[key + "-rsub"] = qsub(0.3, x, q)
            out[key + "-rdiv"] = qdiv(0.3, x, q)
            out[key + "-broadcast"] = qmul(x[:3000].view(30, 100), y[:100], q)
        out[f"pointwise-{dtype}-unsigned"] = qadd(
            x.abs(), y.abs(), Quant(BinaryK(8, 4, is_signed=False))
        )
        out[f"pointwise-{dtype}-normals"] = qadd(
            x, y, Quant(BinaryK(8, 4, subnormals=SubnormalsMode.NORMALS))
        )
    # the elementwise quantizer's vector loops (cpu/binaryK_quant_vector.cpp):
    # random words (NaN payloads, subnormals and infinities among them) over
    # several parallel chunks and a ragged tail, a start off every vector
    # boundary, the in-place op, the edges of the formats the loops take
    # (one and 22 stored mantissa bits, a bias, saturation), and formats and
    # dtypes that keep the scalar loop, which must agree trivially
    from mptorch.quant import binaryK_quantize, binaryK_quantize_

    g = torch.Generator().manual_seed(12)
    words = torch.randint(-(2**31), 2**31 - 1, (20003,), generator=g, dtype=torch.int32)
    w0 = torch.cat([words.view(torch.float32), x0, torch.tensor(specials)])
    quant_formats = {
        torch.float32: [
            (8, 4, {}), (16, 11, {}), (12, 7, {}), (6, 3, {}), (8, 2, {}), (30, 23, {}),
            (8, 4, dict(bias=3)), (8, 4, dict(saturation_mode=SaturationMode.SAT_FINITE)),
            (31, 24, {}), (8, 4, dict(saturation_mode=SaturationMode.SAT_PROPAGATE)),
            (8, 4, dict(is_signed=False)), (8, 4, dict(subnormals_mode=SubnormalsMode.NORMALS)),
            (8, 4, dict(subnormals_mode=SubnormalsMode.EXTENDED_NORMALS)),
        ],
        torch.float16: [(8, 4, {})],
        torch.bfloat16: [(8, 4, {})],
    }  # fmt: skip
    for dtype, fmts in quant_formats.items():
        w = w0.to(dtype)
        for k, p, kw in fmts:
            for mode in (RoundMode.RNE, *WORD_MODES):
                q = dict(rounding_mode=mode, **kw)
                key = f"quant-{dtype}-{k}p{p}-{sorted(kw.items())}-{mode.name}"
                out[key] = binaryK_quantize(w, k, p, **q)
                out[key + "-offset"] = binaryK_quantize(w[3:], k, p, **q)
                y = w.clone()
                binaryK_quantize_(y[1:], k, p, **q)
                out[key + "-in-place"] = y
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
