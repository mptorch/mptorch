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
| `cast_all_modes_sweep.cu` | every cast in every rounding mode against `reference_casts.h`, over all 2^32 binary32 inputs: mismatches and `-0.0` results per mode; fixed point; `image64*` carries it to the binary64 instantiation; `sr-mean` holds SR's distribution to P3109 exactly, `sr-mean-old` is its negative control (`old_sr_casts.h`) | `short` on CUDA (72 configurations, ~5 min) and `short-host` on the host (15, ~19 min) are the gate for a change; `spot-pairs` and `sweep` when the work is finished |
| `cast_double_sweep.cu` | the binary64 casts against the reference on structured inputs (every exponent field, the format's grid ± an ulp, ties, random words) | `short` (58 configurations, ~3 min), `short-host` (9, ~4 min) |
| `cast_ubsan.cpp` | every cast in every rounding mode, in both carriers, under the undefined behaviour sanitizer (a shift by the word's width or more, a signed overflow, an out-of-range float-to-int conversion), over a sample that reaches every arm, with formats at the edges of what the ops accept (`common/format_host.h`'s `check_format`); a negative control in a child process fails the run when the sanitizer is off | no argument (~3 s), the gate; `--recover` on a build without `-fno-sanitize-recover` lists every report |
| `gemm_cast_float_arith.cu` | binaryK's float-arithmetic RNE fast path, three ways: reference, integer path, shipped; `gemm` times the two paths | `spot` on the host and the GPU; `coarse`/`sweep` for the format grid |
| `gemm_cast_superfp_arith.cu` | the same for superfp; the device and the host spell it differently | `coarse` on the GPU, `spot` and `coarse` on the host |
| `gemm_cast_sr_arith.cu` | SR's magnitude-compare clip against the integer clip and the reference | `spot` |
| `gemm_cast_fixedpoint_arith.cu` | the fixed-point GEMM's float-arithmetic RNE cast (`common/fixedpoint_gemm.h`), three ways: reference, the shipped bitwise cast, the fast path, over every binary32 input for each format of a grid past the gate's ends (1,034 formats, 845 admitted), reporting whether the gate is honest; `sr`: the GEMM's stochastic cast against the shipped one the quantizers run, over every binary32 input under a set of draws, for a grid of formats, random bits and the three variants (3,240 cases), with a negative control (`sr-control`); `det`: the GEMM's deterministic casts (RNE, RNA, RU, RD, RZ, RO) against the shipped ones over every binary32 input for the same grid (1,296 cases), with `det-control` | no argument on the GPU (~2 min), `spot` on the host (~25 s); `sr` on the GPU (~15 min), `sr-short` (~10 s on the GPU, ~10 min on 16 host threads), `sr-control`; `det` on the GPU (~1.5 min), `det-short` (~5 s on the GPU, ~1.3 min on 16 host threads), `det-control` |
| `fixedpoint_int_sweep.cpp` | the fixed-point GEMM's integer path (`common/fixedpoint_gemm.h`) against the shipped casts: its rounding over every exact sum of small spans in every mode and draw, saturated and wrapped, and under SR the rounding from per-launch constants too (`round`); the fused step's guard and sticky bits against the binary64 cast of the exact sum (`fused`); the integer path against the carrier path over K-chains, with a negative control (`chain`); a wide accumulator's float64 result (`wide`) | no argument (~5 s on 16 threads); again with `-fsanitize=undefined -fno-sanitize-recover` |
| `cpu_rne_tile_sweep.cpp` | every binary32 word through the CPU's vector RNE tiles (`cpu/gemm_rne_tile.cpp`, compiled in) against the scalar step, one K-step a call; then whole tiles (32 x 32 x 32, the copy with fixed sizes) over a sample of the words, whole and as one-step calls, against the scalar steps | ~10 s per format on 16 threads (~1.5 min); again with `MPTORCH_CPU_SIMD=avx2` |
| `cpu_palette_tile_sweep.cpp` | every binary32 word through the palette GEMMs' lane tiles (`cpu/gemm_palette_lanes.cpp`, compiled in: `cast_binaryK_rne_fast` with each lane's constants), each lane on its own format (every format in turn, per element and per column), fused and split, against the scalar cast; then whole tiles of the three macs with a palette of three per block; `control` gives each lane its neighbour's format and must mismatch | ~2 min on 16 threads; again with `MPTORCH_CPU_SIMD=avx2` |
| `cpu_odd_tile_sweep.cpp` | every binary32 word through the CPU's vector tiles of the odd carrier step (`cpu/gemm_odd_tile.cpp`, compiled in: the directed pair by AVX-512's embedded rounding and under AVX2's MXCSR and NEON's FPCR rounding mode), fused, split and product-only, against spreads of products and moderate ones, plus the odd step's special cases and whole tiles as in `cpu_rne_tile_sweep.cpp`, against the scalar step | ~25-30 s per format on 16 threads (~4.5 min); again with `MPTORCH_CPU_SIMD=avx2` |
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
| `source_identity.py` | the C++ sources against a revision with comments removed: whether a change touched only comments, or moved code without changing it, before anything is built | `diff HEAD` (`--resolve-includes` for files moved into a subdirectory); `split HEAD OLD NEW...` for one file split into several; seconds |

### timing/

| tool | measures | run |
|---|---|---|
| `cpu_gemm_ab.cpp` | the shipped CPU GEMM kernel outside torch, built against two header trees (`git archive <commit> mptorch/csrc`) with the extension's host flags; `MPTORCH_CPU_SIMD=0` keeps the scalar loop; `-DMPTORCH_AB_ODD` (with `cpu/gemm_odd_tile.cpp` linked) adds the odd carrier step's GEMMs | `--threads N --reps R [--n SIZE] [--only NAME]`, the two binaries alternated |
| `gemm_sr_ab.cu` | the shipped CUDA GEMM under SR, against two header trees: binaryK and superfp macs, and the fixed-point GEMM's carrier path and integer accumulator (`-DMPTORCH_AB_NO_FIXEDPOINT` for a tree from before it) | the two binaries alternated |
| `gemm_f64_unroll.cu` | the shipped binary64 GEMM kernels at other K-loop unrolls (`MPTORCH_GEMM_F64_UNROLL`) | one build per factor |
| `benchmark_qmatmul.py` | the batched GEMM against a loop of 2D calls, and the accumulators against NAIVE | `--device cuda`, `--n` |
| `benchmark_qlayers.py` | `QLinear`/`QConv2d` with quantizers; `conv-gemm`: the gathered conv ops against `unfold` and cuDNN | `layers`, `conv-gemm` |
| `benchmark_block.py` | the block-format ops: pack and unpack, the block GEMM, layers, frozen weights | `elementwise`, `gemm`, `layer` |
| `cpu_quantize_elementwise.py`, `cuda_quantize_elementwise.py` | quantizer throughput per mode and dtype, in place and out of place, against a copy's ceiling | `[elements] [threads]` |
| `cpu_gemm_tile_state.py` | CPU GEMMs through the ops at ragged and thin shapes, as JSON, for comparing two builds | `--threads 8`, under each build's `PYTHONPATH` in turn |
| `python_call_overhead.py` | what the Python wrappers cost per call: the bare op, a resolved spec, per-call resolution, memoized format objects | `--device cpu` or `cuda` |
| `instrument_overhead.py` | what `simulate` costs a model it leaves in PyTorch's arithmetic: `tests/models.py`'s `MiniLlama`, plain, with every call passed through, and with every product routed through mptorch's Functions, then the same with the activation, elementwise and embedding kinds named; the gate is at most 5-10% on the small size | `--device cpu` or `cuda` |
| `cpu_palette_gemm.py` | a CPU palette GEMM (`binaryK_matmul_mixed`, `binaryK_matmul_fma_mixed`) against its single-format twin, for the maps a model meets: one entry, per column, two halves, per row, per element | `--threads 1` (or 16); `MPTORCH_CPU_SIMD=0` for the scalar loop; ~1 min |
| `pointwise_ops.py` | what `mptorch.quant.pointwise`'s ops cost against torch's: add, mul, div, sqrt, exp, GELU and SiLU, forward and with backward, float32 and bfloat16, with no slots, RNE, SR, binary64 and rounded to odd | `--device cpu --threads 1` (or 16) or `--device cuda`; ~2 min |
| `reduction_ops.py` | what `qsum`, `qsoftmax`, `qlayernorm` and `qrmsnorm` cost against torch's, over a row's last dimension at four shapes: forward with no formats, RNE, SR and KAHAN sums on the GEMM kernels; forward and backward with the backward rounded or torch's | `--device cpu --threads 1` (or 16) or `--device cuda`; ~1-3 min |
| `attention_ops.py` | what `qattention` costs against `F.scaled_dot_product_attention` (its default kernel and its math kernel), causal, at two or three shapes: forward with no formats, RNE and SR on the GEMM kernels; forward and backward under RNE | `--device cpu --threads 1` (or 16) or `--device cuda`; ~1 min on CUDA, ~4-20 min on the CPU |
| `gpt2_simulated.py` | what simulating a pretrained GPT-2 small costs: tokens per second and peak memory, plain and under an fp8 recipe with the attention kind, where one window's time goes (mptorch's GEMMs, its quantizers, torch's operators, the rest), and with `--compile` what `torch.compile(sim.wrap())` costs and gains; needs the `hf` group and downloads the model once | `--device cuda --compile`, or `--device cpu --threads 16` (or 1); ~2 min on CUDA, ~3-10 min on the CPU |
| `compile_scale.py` | compiling a model under `simulate` (`torch.compile(sim.wrap())`) with formats per layer, at 4 to 32 layers: compile time, frames, Functions in the graph, guards and their check time | `--layers 4 8 16 32` |
| `sr_variants.py` | SR's cost per rounding in StochasticA, B and C, quantizers and GEMMs | `bench` |
| `carrier_double_rounding.py` | `rates`: how often the binary32 carrier rounds a mac step wrongly, against the binary64 carrier and the round-to-odd ops; `bench`: what the round-to-odd step costs; `rates-accumulated`, `bench-accumulated`: the same under KAHAN, BLOCK and TREE (a build with `MPTORCH_ODD_ACCUMULATED=1`) | `rates --device cuda`, `bench`, `rates-accumulated`, `bench-accumulated` |
| `layer_bwd_memory.py` | a layer's peak and held memory, and step time, by which gradients are asked for; `recompute`: the layers' `recompute_input`/`recompute_weight` flags | `cuda`, `cuda recompute` |
| `fixedpoint_gemm.py` | the fixed-point GEMM ops per mac step, against the binaryK split GEMM (a format the fast cast admits, and one below its floor): the carrier path, the integer accumulator split and fused, a wide result, under RNE, SR and SR_C; `layer`: a `QLinear` step on fixed-point formats against binaryK formats and plain PyTorch | `ops --device cuda` (~1 min), `ops --device cpu --threads 1` (~3 min), `layer --device cuda` |
| `mixed_prec_idx_memo.py` | the palette ops' `prec_idx` bounds-check memo, by map dtype and layout: an int64 map must hit it too | CUDA, no argument |

### top level

| tool | does | run |
|---|---|---|
| `mps_backend.py` | `bench`: every op on MPS against the CPU op, results checked; `sweep`: the quantizers over all 2^32 binary32 words against the CPU (~6 min on an M3 Pro); `control`: the sweep's negative control; `odd-probe`: what the odd carrier step's Metal primitives assume of the GPU (`metal::fma`, the flush of operands and results); `odd-sweep [SEED]`: those primitives (`_carrier_odd`) against the CPU's over 25 regimes x 2^24 triples (~18 s, must be zero); `bench-odd`: each odd op against its default twin on MPS | on a Mac |
| `check_native.sh` | compiles every native tool here against the working tree (host syntax checks, then nvcc at `-O0`) | no argument; ~1 min |

## Which tools a change needs

These come on top of what every change runs: the suite, and ruff, ty and pre-commit.

| the change touches | run |
|---|---|
| any header a native tool includes | `check_native.sh` |
| a cast (`cast_binaryK.h`, `cast_superfp.h`, `cast_fixedpoint.h`, `stochastic.h`, `bit_helper.h`) | `cast_all_modes_sweep` `short` and `short-host`, `cast_double_sweep short`, `cast_ubsan.cpp`, and the `gemm_cast_*_arith` program whose fast path it reaches, in its gate mode; `format_limits.py` if a range moves |
| GEMM, conv or block kernels, policies or drivers | `binary_identity.py so` against the build before; `stacked_before_after.py capture`/`compare`, with `--block` for anything that reaches `block_decode.h` or the pack kernels |
| layers, their autograd Functions, `_saved.py` | `layer_output_capture.py` on cpu and cuda |
| the CPU's vector tiles | `cpu_rne_tile_sweep.cpp`, `cpu_odd_tile_sweep.cpp` and `cpu_palette_tile_sweep.cpp` at each ISA; `cpu_gemm_ab.cpp` for the cost, `cpu_palette_gemm.py` for the palette GEMMs' |
| the fixed-point GEMM's header (`common/fixedpoint_gemm.h`) | `gemm_cast_fixedpoint_arith.cu` on the GPU (with `sr-short`/`sr-control` for its stochastic cast and `det`/`det-control` for its deterministic ones) and `spot` and `det-short` on the host; `fixedpoint_int_sweep.cpp`, plain and under the sanitizer; `binary_identity.py so`; `gemm_sr_ab.cu` for an SR cost |
| the float64 narrowing | `narrow_float64_sweep.py short` |
| the elementwise SR path | `elementwise_sr_stats.py`, both dtypes; `cast_all_modes_sweep sr-mean` |
| a bound in `mptorch/number/` | `format_limits.py sweep --audit`, and with `--carrier binary64` |
| MPS code, or a header the Metal prelude includes | on the Mac: the suite, `mps_backend.py sweep` and `bench` |
| The odd carrier step's headers (`common/carrier_odd*.h`) | on the Mac: the suite and `mps_backend.py odd-sweep` (the Metal side); on Linux: `gates/binary_identity.py so` (their host and CUDA text) |
| only comments in C++, or code moved between files | `gates/source_identity.py diff` (or `split`) first; then the build's own gates for what a comment edit still moves (line numbers in host code, the Metal text on the Mac) |
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
