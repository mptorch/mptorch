# dev/benchmarks

The tools that check and measure the library outside the test suite:
exhaustive sweeps against independent references, gates that compare two
builds, format enumerators, and timings. None of them is part of `pytest` or of
the build. Each file's header says what it is for, how to build and run it from
the repository root, its modes and how long they take, and, where it has
them, dated results.

## What belongs here

**A tool that checks or measures shipped code**, meaning either of:
- the installed ops, called through `mptorch`;
- the shipped kernels and headers, instantiated from an `-I` tree, as
  `timing/gemm_sr_ab.cu` does.

A one-off experiment on a hand-written stand-in (a copy of a kernel, a
re-spelled cast) is not a tool: it goes in `temp/` or a scratch directory,
and its result goes with the decision it informed: in the commit message,
or in a comment at the code it decided. Its numbers stop meaning anything as
soon as the code it copied moves on, and keeping it compiling is a tax on
every header change. Eight such tools, and the old `dev/cuda` prototypes,
were taken out of this folder on 2026-11-12 for this reason.

## Layout

| folder | what is in it |
|---|---|
| `sweeps/` | correctness checks over far more inputs than pytest can afford, against references written apart from the code under test |
| `formats/` | what a format represents, and whether a carrier can simulate it |
| `gates/` | before and after comparisons of two builds: machine code, op outputs, layer outputs |
| `timing/` | what things cost: native A/B harnesses on the shipped kernels, and timings through the ops |
| top level | `mps_backend.py` (the Apple GPU backend, all four roles), `check_native.sh` |

## Index

### sweeps/

| tool | checks | run |
|---|---|---|
| `cast_all_modes_sweep.cu` | every cast in every rounding mode against `reference_casts.h`, over all 2^32 binary32 inputs: mismatches and `-0.0` results per mode; fixed point; `image64*` carries it to the binary64 instantiation; `sr-mean` holds SR's distribution to P3109 exactly, `sr-mean-old` is its negative control (`old_sr_casts.h`) | `short` on CUDA (42 configurations, ~3 min) and `short-host` on the host are the gate for a change; `spot-pairs` and `sweep` when the work is finished |
| `cast_double_sweep.cu` | the binary64 casts against the reference on structured inputs (every exponent field, the format's grid ± an ulp, ties, random words) | `short` (38 configurations, ~2 min), `short-host` |
| `gemm_cast_float_arith.cu` | binaryK's float-arithmetic RNE fast path, three ways: reference, integer path, shipped; `gemm` times the two paths | `spot` on the host and the GPU; `coarse`/`sweep` for the format grid |
| `gemm_cast_superfp_arith.cu` | the same for superfp; the device and the host spell it differently | `coarse` on the GPU, `spot` and `coarse` on the host |
| `gemm_cast_sr_arith.cu` | SR's magnitude-compare clip against the integer clip and the reference | `spot` |
| `cpu_rne_tile_sweep.cpp` | every binary32 word through the CPU's vector RNE tiles (`cpu/gemm_rne_tile.cpp`, compiled in) against the scalar step, one K-step a call; then whole tiles (32 x 32 x 32, the copy with fixed sizes) over a sample of the words, whole and as one-step calls, against the scalar steps | ~10 s per format on 16 threads (~1.5 min); again with `MPTORCH_CPU_SIMD=avx2` |
| `cpu_odd_tile_sweep.cpp` | every binary32 word through the CPU's vector tiles of the odd carrier step (`cpu/gemm_odd_tile.cpp`, compiled in: the directed pair by AVX-512's embedded rounding and under AVX2's MXCSR, NEON's binary32 error-free transforms), fused, split and product-only, against spreads of products and moderate ones, plus the odd step's special cases and whole tiles as in `cpu_rne_tile_sweep.cpp`, against the scalar step | ~25-30 s per format on 16 threads (~4.5 min); again with `MPTORCH_CPU_SIMD=avx2` and `=eft` |
| `narrow_float64_sweep.py` | the `narrow_float64` op against the hardware conversion and the tests' torch spelling | `short` (seconds), `sweep` (2^32 words on CUDA) |
| `elementwise_sr_stats.py` | eleven statistical properties of the elementwise SR path (exact values survive, RD-or-RU, rates, position bias, correlation, seeding, CPU vs CUDA) | no argument, then `--dtype float64`; seconds |
| `reference_casts.h` | the independent reference: every cast in the `(man_bits, exp_bits, bias, ...)` spelling, deriving its constants inline, apart from the shipped casts, which read them from a struct built once | included by the sweeps |
| `old_sr_casts.h` | the stochastic casts as they were at `4164cee`, before every region was rounded exactly on the word, frozen as `sr-mean-old`'s negative control | included by `cast_all_modes_sweep.cu` |

### formats/

| tool | does | run |
|---|---|---|
| `binaryK_values.py`, `superfp_values.py`, `fixedpoint_values.py` | print every value a format represents, from its code points; `--verify` round-trips each through the kernel (`--dtype float64` through the binary64 one) | e.g. `binaryK_values.py -K 8 -P 4 --verify` |
| `format_limits.py` | whether a carrier can carry a format at all, and where the casts stop being faithful; `sweep --audit` measures the boundaries `tests/test_format_limits.py` pins | `format_limits.py binaryK -K 8 -P 4`; `sweep --audit`, `--carrier binary64`, `--dtype float16` |

### gates/

| tool | compares | run |
|---|---|---|
| `binary_identity.py` | two builds' machine code: every ELF section, every `.nv_fatbin` member, every device function's SASS and register report, every host function | `so BASE.so NEW.so`; `obj BASE.o NEW.o` to bisect one unit; `cubin` for a benchmark built with `nvcc -cubin` |
| `stacked_before_after.py` | two builds' op outputs (GEMMs, quantizers, SR included) word for word; `--block` adds the block ops. Its `time` and `memory` modes time two builds alternated, and measure their peak memory | `capture --out before.pt` before a change, `compare before.pt after.pt` after; seconds |
| `layer_output_capture.py` | two builds' layer outputs, gradients and generator state, over formats, `requires_grad` patterns and biases | `capture cpu before.pt`, then `compare`; on cuda too |

### timing/

| tool | measures | run |
|---|---|---|
| `cpu_gemm_ab.cpp` | the shipped CPU GEMM kernel outside torch, built against two header trees (`git archive <commit> mptorch/csrc`) with the extension's host flags; `MPTORCH_CPU_SIMD=0` keeps the scalar loop; `-DMPTORCH_AB_ODD` (with `cpu/gemm_odd_tile.cpp` linked) adds the odd carrier step's GEMMs; on arm64, `cpu/gemm_odd_tile.cpp` compiled with `-DMPTORCH_ODD_NEON_FPCR` runs NEON's odd tiles as the directed pair under FPCR, the A/B against its transforms | `--threads N --reps R [--n SIZE] [--only NAME]`, the two binaries alternated |
| `gemm_sr_ab.cu` | the shipped CUDA GEMM under SR, against two header trees | the two binaries alternated |
| `gemm_f64_unroll.cu` | the shipped binary64 GEMM kernels at other K-loop unrolls (`MPTORCH_GEMM_F64_UNROLL`) | one build per factor |
| `benchmark_qmatmul.py` | the batched GEMM against a loop of 2D calls, and the accumulators against NAIVE | `--device cuda`, `--n` |
| `benchmark_qlayers.py` | `QLinear`/`QConv2d` with quantizers; `conv-gemm`: the gathered conv ops against `unfold` and cuDNN | `layers`, `conv-gemm` |
| `benchmark_block.py` | the block-format ops: pack and unpack, the block GEMM, layers, frozen weights | `elementwise`, `gemm`, `layer` |
| `cpu_quantize_elementwise.py`, `cuda_quantize_elementwise.py` | quantizer throughput per mode and dtype, in place and out of place, against a copy's ceiling | `[elements] [threads]` |
| `cpu_gemm_tile_state.py` | CPU GEMMs through the ops at ragged and thin shapes, as JSON, for comparing two builds | `--threads 8`, under each build's `PYTHONPATH` in turn |
| `python_call_overhead.py` | what the Python wrappers cost per call: the bare op, a resolved spec, per-call resolution, memoized format objects | `--device cpu` or `cuda` |
| `sr_variants.py` | SR's cost per rounding in StochasticA, B and C, quantizers and GEMMs | `bench` |
| `carrier_double_rounding.py` | `rates`: how often the binary32 carrier rounds a mac step wrongly, against the binary64 carrier and the round-to-odd ops; `bench`: what the round-to-odd step costs | `rates --device cuda`, `bench` |
| `layer_bwd_memory.py` | a layer's peak and held memory, and step time, by which gradients are asked for; `recompute`: the layers' `recompute_input`/`recompute_weight` flags | `cuda`, `cuda recompute` |
| `mixed_prec_idx_memo.py` | the palette ops' `prec_idx` bounds-check memo, by map dtype and layout: an int64 map must hit it too | CUDA, no argument |

### top level

| tool | does | run |
|---|---|---|
| `mps_backend.py` | `bench`: every op on MPS against the CPU op, results checked; `sweep`: the quantizers over all 2^32 binary32 words against the CPU (~6 min on an M3 Pro); `control`: the sweep's negative control | on a Mac |
| `check_native.sh` | compiles every native tool here against the working tree (host syntax checks, then nvcc at `-O0`) | no argument; ~1 min |

## Which tools a change needs

These come on top of what every change runs: the suite, and ruff, ty and pre-commit.

| the change touches | run |
|---|---|
| any header a native tool includes | `check_native.sh` |
| a cast (`cast_binaryK.h`, `cast_superfp.h`, `cast_fixedpoint.h`, `stochastic.h`, `bit_helper.h`) | `cast_all_modes_sweep` `short` and `short-host`, `cast_double_sweep short`, and the `gemm_cast_*_arith` program whose fast path it reaches, in its gate mode; `format_limits.py` if a range moves |
| GEMM, conv or block kernels, policies or drivers | `binary_identity.py so` against the build before; `stacked_before_after.py capture`/`compare`, with `--block` for anything that reaches `block_decode.h` or the pack kernels |
| layers, their autograd Functions, `_saved.py` | `layer_output_capture.py` on cpu and cuda |
| the CPU's vector tiles | `cpu_rne_tile_sweep.cpp` and `cpu_odd_tile_sweep.cpp` at each ISA; `cpu_gemm_ab.cpp` for the cost |
| the float64 narrowing | `narrow_float64_sweep.py short` |
| the elementwise SR path | `elementwise_sr_stats.py`, both dtypes; `cast_all_modes_sweep sr-mean` |
| a bound in `mptorch/number.py` | `format_limits.py sweep --audit`, and with `--carrier binary64` |
| MPS code, or a header the Metal prelude includes | on the Mac: the suite, `mps_backend.py sweep` and `bench` |
| a cost you are going to claim | the matching `timing/` tool, alternated as below |

Run longer grids once, when a piece of work is finished, trimmed to under 20 minutes.

## Conventions for a new tool

- **Where it goes:** in the folder of its role.
- **The header** says, in this order:
  - what the tool answers, in terms of the library code it exercises (not of a plan or a ticket, which will not outlive it);
  - the build and run lines, from the repository root;
  - its modes and how long each takes;
  - the results it produced, dated, with the machine named.
- **An exhaustive check** has a `short` mode of a few minutes, which is its gate for a change. Its full runs stay under 20 minutes per grid.
- **A sweep that can pass needs a negative control that fails** (`sr-mean-old`, `mps_backend.py control`). Otherwise a clean sweep proves nothing.
- **A native harness** instantiates the shipped kernel from the tree `-I` names (`launch_as`, as in `gemm_sr_ab.cu` and `cpu_gemm_ab.cpp`), with the extension's own flags: `-ffp-contract=off -DMPTORCH_FAST_CAST=1` on the host, from `build/temp.*/build.ninja`.
  - Never a copy of a kernel body.
  - Where it needs a knob the kernel does not have, add a macro to the shipped kernel that nothing in the build defines, as `MPTORCH_GEMM_F64_UNROLL` does.
  - Then add the tool to `check_native.sh`.
- **A timing** alternates the two variants, call by call or build by build, flips the order each round, and keeps the minimum. On this laptop two builds timed one after the other cannot resolve anything under ~15%: re-running the same build moves the short kernels by 15-29%, and the drift is upward within a session.
  - Compare a header change through the shipped kernels, never through a scratch loop.
- **A Python tool:**
  - uses `argparse`;
  - runs from a `__main__` guard;
  - passes `ruff check`, `ruff format` and `ty check` like the library (CI runs ruff on this folder, pre-commit runs ty);
  - imports nothing from another tool outside its own folder.
- **Heavy runs** cap their memory (on the WSL laptop, `systemd-run --user --scope -p MemoryMax=8G`) and never run beside a build or the suite.
