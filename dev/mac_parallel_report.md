# Mac report for `dev/mac_parallel_handoff.md`

One section per task, as the handoff's section 2.3 asks. The machine is an
Apple M3 Pro (5 performance and 6 efficiency cores, 18 GB), macOS 26.7,
Apple clang 21.0.0, torch 2.15.0, Python 3.14.10. Every build has
`MPTORCH_ODD_ACCUMULATED=1`. Every timing ran on AC power, with `pmset -g
therm` reporting no thermal or performance warning, and nothing else
running. The changes are committed on `feature-refactor`, one commit per
task (the last section lists them).

## M1: the verification round (at `d612464`)

### Build and tests (4.2)

`MPTORCH_ODD_ACCUMULATED=1 uv sync --group hf` after clearing Inductor's
cache. The compile lines show `-ffp-contract=off -DMPTORCH_FAST_CAST=1`,
and the build printed no new warning.

| command | result |
|---|---|
| `pytest tests/test_fixedpoint_mps.py tests/test_fixedpoint_layers.py tests/test_fixedpoint_gemm.py tests/test_entry_copies.py` | 838 passed, 362 skipped (every skip is a CUDA case, or `a fused mac always runs the integer accumulator`) |
| `85d1d31`'s new MPS cases in `test_fixedpoint_layers.py` | all 12 pass: the flat wrappers in RNE, RD, SR and SR_C, tier 2, tier 3 for the linear (RNE, RD), conv and matmul, and the three wide-refusal tests |
| `pytest tests/` | **19,759 passed, 11,430 skipped, 0 failed**, in 14:38 |

The suite's total is 31,189, which is Linux's 21,324 + 9,865. It ran
alongside the sweeps below and a second build, so its time says nothing.

The MPS cases' skips (`pytest tests/ -k mps -rs`: 8,646 passed, 850
skipped):

| skips | reason |
|---|---|
| 789 | `MPS has no float64 tensors` (and one `MPS has no float64`) |
| 56 | `TREE sums products pairwise; a fused mac has none` |
| 2 | `An unsigned format cannot be symmetric.` |
| 1 | `OCP is a power-of-two scale's rule` |
| 2 | `bfloat16 GEMMs are tested on the CPU and CUDA` (`test_simulate.py::test_sdpa_is_its_eager_twin`, llama in bfloat16) |

Only the last two are not a missing dtype or an impossible combination.
They are a test's own rule from before these commits. I left them as they
are.

### The host sweeps under Apple clang (4.3)

These binaries were built as the handoff gives them, with Homebrew's libomp
(standalone binaries only), and ran on 11 threads.

| sweep | result | time |
|---|---|---|
| `det-short` | `6 formats, 36 cases (format, mode); 0 cases mismatch` | 85 s |
| `det-control` | `RNE against the shipped RNA 68478 mismatches, RU against RD 8321159945`: Linux's counts exactly | 30 s |
| `sr-short` | `6 formats, 54 cases (format, N, variant); 0 cases mismatch` | 500 s |
| `sr-control` | `1941258420 mismatches`: Linux's count exactly | 493 s |
| `spot` | `5 admitted by the gate; 0 admitted formats mismatch; 0 formats inside the cast's domain where the bitwise cast and the reference disagree` | 27 s |
| `fixedpoint_int_sweep.cpp` | `every comparison agrees`; the two `chain` controls differ (10555 and 812) | 5 s |
| the same, `-O2 -fsanitize=undefined -fno-sanitize-recover` | the same lines, exit 0, no `runtime error` | 7 s |
| `check_native.sh` | `all native tools build` (the `.cu` builds skipped: no nvcc) | 9 s |

### The host's choice of modes on arm64 (4.4)

Method: the committed build ("new", `d612464`) against `85d1d31`'s ("base"),
each built with this checkout's venv, timed through the ops by the
handoff's `ab_ops.py` at 384^3. There were four rounds, the order rotated
each round, and each row's minimum is reported in ps per mac step. A third
build ("arm") is a local experiment and is not in the working tree: the
committed tree plus one line, an `__aarch64__` arm in
`fixedpoint_word_call_v` that gives the line every deterministic mode but
RNE:

```diff
 #if defined(__CUDA_ARCH__)
       fixedpoint_word_mode_v<RM>;
+#elif defined(__aarch64__)
+      fixedpoint_word_mode_v<RM> && RM != RoundMode::RNE;
 #else
       RM == RoundMode::RU || RM == RoundMode::RD || RM == RoundMode::RZ;
 #endif
```

One thread (spread within a build, max/min - 1, at most 5.5%, mostly under
1%):

| row | mode | base | new | arm | new/base | arm/new |
|---|---|---|---|---|---|---|
| carrier | RNE | 1,195 | 1,196 | 1,193 | 1.00 | 1.00 |
| carrier | RNA | 11,409 | 11,095 | 3,211 | 0.97 | **0.29** |
| carrier | RU | 13,138 | 2,981 | 2,931 | **0.23** | 0.98 |
| carrier | RD | 12,935 | 2,299 | 2,303 | **0.18** | 1.00 |
| carrier | RZ | 11,709 | 2,302 | 2,296 | **0.20** | 1.00 |
| carrier | RO | 12,248 | 12,093 | 3,592 | 0.99 | **0.30** |
| carrier | SR | 10,673 | 9,725 | 9,704 | 0.91 | 1.00 |
| int split | RNE | 1,552 | 1,540 | 1,555 | 0.99 | 1.01 |
| int split | RNA | 5,397 | 5,544 | 3,756 | 1.03 | **0.68** |
| int split | RU | 3,508 | 2,116 | 2,114 | **0.60** | 1.00 |
| int split | RD | 3,041 | 1,658 | 1,647 | **0.55** | 0.99 |
| int split | RZ | 3,218 | 1,779 | 1,778 | **0.55** | 1.00 |
| int split | RO | 3,574 | 3,645 | 2,406 | 1.02 | **0.66** |
| int split | SR | 11,993 | 8,355 | 8,336 | **0.70** | 1.00 |
| int fused | RNE, RNA, RU, RD, RZ, RO | 2,144-3,279 | the same | the same | 1.00 | 1.00 |
| int fused | SR | 6,412 | 5,267 | 5,287 | **0.82** | 1.00 |

Eight threads (same method; spreads up to 20%, since a call takes a few ms
at 384^3):

| row | mode | base | new | arm\* | new/base | arm\*/new |
|---|---|---|---|---|---|---|
| carrier | RNE | 231 | 260 | (1,164) | 1.13 | (4.48) |
| carrier | RNA | 1,710 | 1,703 | 594 | 1.00 | **0.35** |
| carrier | RU, RD, RZ | 1,777-2,006 | 415-543 | 418-555 | **0.22-0.27** | 1.00-1.02 |
| carrier | RO | 1,893 | 1,888 | 670 | 1.00 | **0.35** |
| carrier | SR | 1,842 | 1,666 | 1,655 | 0.90 | 0.99 |
| int split | RNA | 901 | 913 | 648 | 1.01 | **0.71** |
| int split | RU, RD, RZ | 539-625 | 322-399 | 342-399 | **0.60-0.64** | 1.00-1.06 |
| int split | RO | 665 | 695 | 447 | 1.05 | **0.64** |
| int split | SR | 2,062 | 1,403 | 1,421 | 0.68 | 1.01 |
| int fused | SR | 1,157 | 917 | 917 | 0.79 | 1.00 |

\* At eight threads the third build was "wide": the line in *every*
deterministic mode, RNE included (`fixedpoint_word_mode_v<RM>`). That build
also ran on one thread, where its RNE carrier path took 6,372 ps, **5.4x**
the shipped 1,189, though the RNE fast path comes first in the call. I did
not look into why. Its other modes matched the arm build's to within 2%.

**Findings:**
- **The committed choice holds where it acts.** RU, RD and RZ gain on arm64
  as on the Ryzen, more on the carrier path (0.18-0.23x against the
  Ryzen's 0.23-0.81x). RNE is unchanged, and SR gains (0.70-0.91x).
- **RNA and RO disagree with the Ryzen.** On arm64 the shipped RNA and RO
  casts cost as much as the shipped directed ones (11-12k ps), and the line
  takes them to 0.29-0.30x on the carrier path and 0.66-0.68x on the
  integer split. On the Ryzen they measured 1.0-1.3x. So the arm above is
  the proposal for SP-1. With it, `tests/test_fixedpoint_gemm.py`,
  `test_fixedpoint_layers.py` and `test_fixedpoint_mps.py` pass (826
  passed, 362 skipped). The `det` sweeps already hold the line itself to
  the shipped casts in every mode under clang.
- **RNE must stay off the line on the host.** The wide build's 5.4x shows
  this.
- **The new build slows a few rows slightly**: the integer split under RNA
  (1.03) and RO (1.02) on one thread, the modes where the host keeps the
  shipped casts. The arm removes both.

### MPS timings, for the docs (4.5)

**A problem with `--device mps`:** `ops` stops at its last row, `fixed point,
split into (40, 8), wide`. MPS refuses a wide result, as designed: `the
accumulate format has 39 magnitude bits, more than float32's 24, so the
result is float64, which MPS does not have`. The tool is Linux's, so I left
it alone and timed a scratch copy that drops that row on MPS:

```diff
-    return {
+    rows = {
 ...
+    if a.device.type == "mps":  # MPS refuses a wide (float64) result
+        del rows["fixed point, split into (40, 8), wide"]
+    return rows
```

`layer --device mps` runs as committed.

`ops --device mps` at 1024^3, in ps per mac step (6 rounds, minimum):

| row | RNE | RNA | RU | RD | RZ | RO | SR | SR_C |
|---|---|---|---|---|---|---|---|---|
| binaryK split, Binary16p9 into Binary16p12 | 10.4 | 17.8 | 82.1 | 82.2 | 109.9 | 16.8 | 62.2 | 65.7 |
| binaryK split, Binary16p8 (bias 127) | 16.2 | 19.0 | 82.3 | 82.4 | 109.5 | 18.1 | 62.5 | 66.4 |
| fixed point, the carrier path | **5.0** | 24.5 | 48.4 | 48.4 | 57.8 | 25.3 | 75.3 | 79.6 |
| fixed point, integer accumulator, split | 36.0 | 46.9 | 52.2 | 47.8 | 57.3 | 41.2 | 98.5 | 101.4 |
| fixed point, integer accumulator, fused | 49.8 | 50.5 | 45.9 | 40.8 | 46.1 | 45.2 | 77.8 | 79.1 |

`layer --device mps`, a `QLinear(1024, 1024)` step on 4096 rows: plain 7.0
ms; binaryK 134 ms (RNE) and 784 ms (SR); fixed point 63 ms (RNE) and 993
ms (SR).

For the docs' cost paragraph, these give, on the M3 Pro's GPU:
- under RNE the carrier path takes 0.48x the binaryK split's time;
- the integer accumulator takes 7.2x the carrier path (split) and 10x
  (fused), which is the "7 to 10 times" the paragraph quotes;
- under SR the carrier path takes 1.21x the binaryK split, since Metal
  keeps the shipped casts (M4's subject);
- under RZ, RU and RD the carrier path takes 0.53-0.59x the binaryK split,
  but still 10-12x its own RNE time.

### Files changed

None in the tree. The base and experiment builds are worktrees outside the
repository (`../mptorch-base` at `85d1d31`, `../mptorch-wide` at `d612464`
with the arm). `git worktree remove` deletes them when they are no longer
needed.

### Open

- SP-1: the `__aarch64__` arm above, for Linux to take or not. An x86-64
  host and the CUDA device preprocess it away, so `binary_identity.py`
  should find their objects unchanged.
- The timing tool's wide row on MPS, for Linux to fix.

## M2: the fused `ScaledQuant` by blocks on MPS (phase AQ, at `d612464`)

### Diagnosis

First I timed each pass alone at 4096 x 4096, BinaryK(8, 4) under SAT_FINITE
and RNE. Each figure is the mean of 20 calls, minimum of 5, and the GPU
events agreed within 4%. The fused call is the amax pass plus one kernel;
the four-pass call is the amax pass plus torch's product, the in-place
rounding and torch's quotient.

| float32 | tensor | row | blocks of 32 |
|---|---|---|---|
| amax + scale (both calls) | 1.06 ms | 1.05 ms | 1.35 ms |
| **the fused kernel alone** | **1.14 ms** | **3.48 ms** | **4.96 ms** |
| four passes: product, rounding, quotient | 0.99 + 0.99 + 1.00 | 1.84 + 0.98 + 1.97 | 1.84 + 0.99 + 1.97 |

bfloat16 gave the same picture: the fused kernel alone took 1.06, 3.52 and
5.00 ms.

The fused kernel's arithmetic per element is the same at every
granularity; only the index arithmetic differs. It computed `row = i /
cols` and `col / block` in 64 bits for every element. For one scale,
`cols` is the whole tensor, so both quotients are 0. By rows and by blocks
they are real divisions, and Apple GPUs emulate 64-bit integers and have
no integer divider. **Suspect 1 holds.** The amax pass costs both calls
about the same (it grows only 1.3x by blocks), so it is not the cause.

### Variants

These were scratch switches (an environment variable picked the kernel
body through a tail define), alternated in one process, 6 rounds. Each was
first held to the shipped words by `test_scaled_quant.py`'s 463 MPS cases.
The fused kernel alone, ms, under RNE:

| variant | f32 tensor | f32 row | f32 block | bf16 tensor | bf16 row | bf16 block |
|---|---|---|---|---|---|---|
| shipped | 1.11 | 3.48 | 4.96 | 1.05 | 3.52 | 5.00 |
| 32-bit indices | 1.07 | 1.08 | 1.10 | 0.97 | 0.95 | 0.94 |
| one 64-bit division per thread, then steps | 1.06 | 1.57 | 1.92 | 1.05 | 1.62 | 1.97 |
| **both** | **1.02** | **1.02** | **1.06** | **0.75** | **0.75** | **0.76** |

SR gave the same order (both: 1.06, 1.04, 1.07, 0.96, 0.96, 0.96 ms).
Suspect 2, the scale load, was not worth pursuing once its row's time
matched the tensor's. Suspect 3, the elements per thread, was not tried
either: the kernel now runs at the plain quantizer's pass time (0.99 ms
in float32), which suspect 3 cannot beat.

### What changed

The kept variant is "both". Its 32-bit indices are chosen at launch: the
tail gains `using mpt_index_t = uint32_t;` when the last index fits, and
`uint64_t` otherwise. A thread divides only for its first element's row
and block, and steps along the row for the other three. I checked the
64-bit path by forcing it in a throwaway build: 463 passed.

- `mptorch/csrc/mps/scaled_quant.metal`: the kernel body and a paragraph
  of its header comment.
- `mptorch/csrc/mps/quantize.cpp`: `quantize_scaled` adds the
  `mpt_index_t` line to the tail; `#include <limits>`.

Neither is a shared file, and no Linux object moves. `quantize.metal` and
the prelude are unchanged, so only the scaled kernels recompile.

### Result

Committed (`../mptorch-base`, whose MPS sources and Python for these ops
match `d612464`) against the new kernels (`fc2caad`, `4d5e60a`), alternated across processes,
4 rounds. Spreads are at most 2.5%.

| call | committed fused | new fused | new/old | four passes | speedup now (was) |
|---|---|---|---|---|---|
| f32, per tensor | 2.23 ms | 2.13 ms | 0.96 | 4.06 ms | 1.91x (1.83x) |
| f32, per row | 4.55 ms | 2.11 ms | 0.46 | 5.86 ms | 2.78x (1.29x) |
| f32, blocks of 32 | 6.29 ms | 2.43 ms | **0.39** | 6.14 ms | **2.53x (0.97x)** |
| bf16, per tensor | 1.68 ms | 1.35 ms | 0.81 | 5.07 ms | 3.75x (3.03x) |
| bf16, per row | 4.11 ms | 1.36 ms | 0.33 | 6.91 ms | 5.10x (1.68x) |
| bf16, blocks of 32 | 5.83 ms | 1.62 ms | 0.28 | 7.18 ms | 4.44x (1.23x) |

Under SR, new over old: 0.88, 0.44, 0.37, 0.80, 0.36 and 0.30. The four
passes stay at 1.00 (the control). The new kernel is faster in every row.

## M3: `block_project` on MPS (phase AR, at `d612464`)

### Diagnosis

The kernel gave each thread one group of codes that fill whole bytes:
one element for an 8-bit code, two for a 4-bit one. Per thread it divided
twice in 64 bits, copied the format's parameters, decoded its block's
scale and built the casts. `block_pack` does that once per tile of 32.
The ratios followed the threads per element (MXFP8 4.2x, the FP4 formats
2.2-2.6x), so suspects 1 and 2 both held.

### Variants

Scratch switches as in M2, alternated in one process, 6 rounds, each held
to the shipped words first by the block_project cases of
`test_block_ops.py` and `test_mps.py` (127, SR included). Wall clock in ms
(GPU events within 6%):

| variant | MXFP8 RNE | MXFP4 RNE | NVFP4 RNE | MXFP8 SR | MXFP4 SR | NVFP4 SR |
|---|---|---|---|---|---|---|
| `block_pack` | 1.15 | 1.13 | 1.41 | 1.14 | 1.32 | 1.60 |
| shipped | 4.78 | 2.96 | 3.17 | 4.78 | 3.09 | 3.32 |
| a whole block per thread | 0.95 | 1.17 | 1.45 | 0.97 | 1.34 | 1.62 |
| **that, with 32-bit divisions** | **0.89** | **1.09** | **1.30** | **0.90** | **1.27** | **1.51** |

I did not try four MXFP8 codes per 32-bit store (suspect 3). With a block
per thread the kernel already beats `block_pack` for all three formats.

### What changed

The kept variant is a whole block per thread, with 32-bit divisions. A
thread per block needs no new rule against two threads writing one byte,
since a block's codes fill whole bytes. The divisions can always be 32-bit
ones: the entry point already kept the thread count below 2^32, so every
quotient and divisor is below 2^32 too. No choice at launch is needed.

- `mptorch/csrc/mps/block_project.metal`: `G` is the block size, the two
  divisions are 32-bit, and the header comment says why.
- `mptorch/csrc/mps/block_entry.cpp`: `L.per_row = p.n_blocks`, and the
  thread-count check's message names blocks (nothing cites it).

Neither is shared. One comment is now stale and I left it:
`launch_params.h` still describes `BlockProjectLaunch::per_row` as byte
groups. That header is part of `prelude.metal`, so editing it would
recompile every kernel; it can go with the next prelude edit.

### Result

The same cross-process alternation as M2:

| format | `block_pack` | committed `block_project` | new | new/old | new / `block_pack` |
|---|---|---|---|---|---|
| MXFP8_E4M3 | 1.12 ms | 4.75 ms | 0.89 ms | 0.19 | **0.79x** (was 4.23x) |
| MXFP4_E2M1 | 1.10 ms | 2.93 ms | 1.07 ms | 0.36 | **0.97x** (was 2.66x) |
| NVFP4 | 1.40 ms | 3.17 ms | 1.30 ms | 0.41 | **0.93x** (was 2.26x) |

Under SR: 0.79x, 0.98x and 0.91x `block_pack`. MXFP4 and NVFP4 still wait
on the refusal flag (GPU time 1.03 and 1.24 ms against 1.09 and 1.30 ms of
wall clock).

## M5: `torch.compile` on MPS (phase AJ's question, at `d612464`)

This was an investigation only; no library change. The scratch files are
in the session's scratchpad: `m5/test_compile_mps.py`,
`llama_compile.py`, `plain_compile.py` and `bitcast_probe.py`.

**The compile tests on MPS.** These are the compile cases of
`test_fake_kernels.py` and `test_freeze_packed.py`, run on MPS under
`aot_eager` and `inductor`. `test_compile_inplace.py` is already
parametrized over MPS, and its cases pass in the suite. Every case
compiles in one graph with no break, and every result an mptorch op rounds
equals eager's word for word, under both backends:
- `test_compiled_ops_equal_eager` (RNE, SR), `test_the_flat_wrappers_trace_into_one_graph`,
  `test_the_layers_trace_into_one_graph` (all six models),
  `test_compile_takes_a_dynamic_batch_without_recompiling`,
  `test_compile_keeps_two_identical_sr_calls_apart` and
  `test_a_frozen_layer_compiles`: pass.
- `test_a_compiled_model_equals_eager`: passes under `aot_eager` for all
  six models. Under `inductor`, four models' bias gradients differ in the
  last bits (up to 2.9e-6) and everything else is equal. The original test
  allows exactly this: Inductor sums the bias gradient in its own order. So
  all 28 cases pass under the original tests' rules. My copy was strict on
  the bias, to see the difference.

**MiniLlama under `presets.fp8()`, through `simulate(...).wrap()`, with
`fullgraph=True`.** Batch 4 x 64 tokens; the small size is vocab 256,
hidden 128, 2 layers; the larger is vocab 1024, hidden 512, 4 layers.

| run | backend | compiles (first call) | words equal eager's | time |
|---|---|---|---|---|
| small, forward | aot_eager | yes (1.7 s) | 1/1 | 0.88x eager |
| small, forward | inductor | yes (4.3 s) | 1/1 | **2.95x faster** |
| small, training step | aot_eager | yes (4.2 s) | 21/22 (one gradient off by 1.5e-11) | 1.24x faster |
| small, training step | inductor | yes (8.2 s) | 5/22 (max 2.7e-6) | 2.75x faster |
| larger, forward | aot_eager | yes (2.7 s) | 1/1 | 1.00x |
| larger, forward | inductor | yes (6.6 s) | **0/1 (max 0.052)** | 1.30x faster |
| larger, training step | aot_eager | yes (7.3 s) | 40/40 | 1.07x faster |
| larger, training step | inductor | yes (13.4 s) | 0/40 (max 0.052) | 1.30x faster |

**Why Inductor's words differ.** The same model *without* mptorch, compiled
by Inductor, differs from eager in the last bits. On MPS 93% of the logits
differ, by at most 1.4e-6; on the CPU, 93% by at most 1.6e-6. So Inductor's
own kernels for the plain torch arithmetic (norms, softmax, rotary, SiLU)
round differently from eager's, on both devices. The fp8 roundings then
turn some of those last-bit differences into a different E4M3 value. The
mptorch ops compute their inputs' words faithfully. Nothing here is
specific to MPS.

**A probe for AJ itself.** I wrote a toy binary32 -> BinaryK(8, 4) RNE cast
as torch ops on the word: `view(torch.int32)`, shifts, masks, adds,
`view(torch.float32)`, `where`. Inductor on MPS compiles it inside `x * 1.5
... + 1.0` into **one** Metal kernel, and its words equal eager's on 2^24
N(0, 64) elements. It runs in 1.01 ms. The same graph through mptorch's
`binaryK_quantize` takes 3.03 ms compiled (3.01 eager), and the word-op
version runs eagerly in 15.5 ms. So the plan's approach works on MPS.

One question stays open for AJ's plan: whether Inductor's Metal kernels
flush binary32 subnormals and under which math mode they compile. Their
float operations, such as the `x * 1.5` around the cast, would then
differ from the CPU's below 2^-126, as mptorch's MPS kernels avoid by
working on the word. My probe's inputs had no subnormals, so it does not
answer this.

## M6a: binary32 round-to-odd division and square root on Metal (phase AI's pre-work, at `d612464`)

### What was added

`mptorch::gemm::odd::div(float, float)` and `mptorch::gemm::odd::sqrt(float)`
are in `common/carrier_odd_metal.h`, beside `fma`, `add` and `mul`. The
handoff writes the interface as `mptorch::odd`. The existing primitives,
and so these, are in `mptorch::gemm::odd`, so Linux should match that
namespace. The header is Metal-only: `carrier_odd.h` includes it only
under `__METAL_VERSION__`, and nothing else includes it. So no host or
CUDA object can move. Since the file is under `common/`, the hunk still
needs Linux's `binary_identity.py` run (expected: nothing changes).

Each primitive has the three parts of the others:
- **core.** For div, `q = a / b` and `r = fma(-q, b, a)`; the odd fix of `q`
  (the existing `fix`) takes a residual of `r`'s magnitude whose sign is
  `r`'s times `b`'s. For sqrt, `s = sqrt(x)` and `r = fma(-s, s, x)`, with
  the residual `r`. The fma rounds `r` once, which keeps its sign and
  whether it is zero, so `r` need not be representable. `q` and `s` need
  only be faithful.
- **gate.** These go cold: a dividend or radicand below 2^-79 (zero and
  subnormals included), an infinite or NaN one, a negative radicand, and
  a quotient below 2^-126 or not finite. The quotient test also catches a
  zero, infinite, subnormal (flushed) or NaN divisor. Inside the gate,
  `a - q b` is a multiple of 2^(E_a - 47), and `x - s^2` one of
  2^(E_x - 47), so a nonzero residual is at least 2^-126 and the flush
  cannot reach it.
- **cold** (`noinline`, as for the others). The specials are answered from
  the words, with IEEE's results (`sqrt(-0) = -0`; a negative subnormal's
  root is NaN, where the unit would flush it to `-0`). Every other input is
  rescaled on its words onto [1, 2) (the radicand onto [1, 4)), run
  through the core, and scaled back by the existing `descale`. That rounds
  onto the subnormal grid to odd and gives Omega past the top, as
  `carrier_odd.h`'s table requires.

### Test method and counts

The harness is scratch and not in the tree. For this test only, I added
four ops to `_carrier_odd`'s Metal root (`mps/carrier_odd.metal`):
`odd::div`, `odd::sqrt`, and the unit's own `a / b` and `sqrt`. The MPS
entry skipped its op check for them. Both were reverted afterwards. The
reference is exact: integer quotients and integer square roots in numpy
int64, rounded to odd on the binary32 grid. It agreed with a `Fraction`
bisection on 48,000 pairs, and an RZ-for-RO control of that check found
3,795 and 2,992 differences.

| division set | pairs | `odd::div` mismatches | cold share |
|---|---|---|---|
| uniform random words | 4,194,304 | 0 | 36.9% |
| normal operands, exponents in [-60, 60] | 4,194,304 | 0 | 0.0% |
| quotients in 2^-160 .. 2^-120 | 1,048,576 | 0 | 45.2% |
| dividends in 2^-82 .. 2^-77 (the gate's end) | 1,048,576 | 0 | 50.1% |
| quotients in 2^-129 .. 2^-124 (the gate's end) | 1,048,576 | 0 | 34.4% |
| divisors in 2^122 .. 2^127 | 1,048,576 | 0 | 0.0% |
| quotients in 2^120 .. 2^130 (overflow) | 1,048,576 | 0 | 15.0% |
| subnormal operands | 1,048,576 | 0 | 100% |
| exact quotients (12-bit by 12-bit) | 1,048,576 | 0 | 0.0% |
| one code off an exact quotient | 1,048,576 | 0 | 0.0% |
| exact quotients below 2^-126 | 1,048,576 | 0 | 71.5% |
| 17 specials x 17 specials | 289 | 0 | 88.2% |

| square-root set | radicands | `odd::sqrt` mismatches | cold share |
|---|---|---|---|
| uniform random words | 4,194,304 | 0 | 59.5% |
| normal, exponents in [-60, 60] | 4,194,304 | 0 | 0.0% |
| radicands in 2^-82 .. 2^-77 | 1,048,576 | 0 | 50.0% |
| normals below 2^-79 | 1,048,576 | 0 | 100% |
| subnormals | 1,048,576 | 0 | 100% |
| the top binades | 1,048,576 | 0 | 0.0% |
| exact squares, and one code off them | 2 x 1,048,576 | 0 | 16.9% |
| negative words | 1,048,576 | 0 | 100% |
| specials | 17 | 0 | 76.5% |

The cold share is the gate recomputed from the words and the unit's
quotient. In the ranges a pointwise op normally meets it is zero.

The method assumes the unit's `a / b` and `sqrt` are faithful in safe
math. The harness compared both with the CPU's correctly rounded results
wherever operands and result are normal: no mismatch in 14.2 M quotients
(divisors up to 2^127 included) or 11.5 M roots. So they are correctly
rounded there.

The negative controls: the unit's round-to-nearest results against the
RO reference mismatch on 50% of inexact inputs and on none of the exact
ones, and `odd::div(b, a)` against RO(a / b) mismatches everywhere. So
the comparison can fail.

With the header changed, `tests/test_carrier_rounding.py` and every odd
MPS test pass: the existing primitives and the odd GEMMs compile and agree
as before. The odd kernels recompile once, since the `odd` root's text
changed.

### Open

- SP-AI: Linux confirms the interface (`mptorch::gemm::odd`) and writes
  the host and CUDA twins. M6b then calls these from the pointwise kernels.
- Their speed is unmeasured, since nothing calls them yet.

## Not started

- **M4** (the fixed-point GEMM's new layouts on Metal) waits for the user's
  go-ahead, as the handoff says. Section 4.5's MPS table shows what it
  would act on: under SR the carrier path takes 1.21x the binaryK split's
  time, and under RZ, RU and RD it takes 10-12x its own RNE time.
- **M6b, M7, M8, M9** wait for their sync points (SP-AI, SP-AK, SP-S,
  SP-AL).

## The commits, by task

Each task's changes are a commit of their own, on top of `d612464`:

| task | commit | files | shared? |
|---|---|---|---|
| M2 | `fc2caad` MPS: the fused scaled quantizer divides once per thread, in 32 bits when it can | `mptorch/csrc/mps/scaled_quant.metal`, `mptorch/csrc/mps/quantize.cpp` | no |
| M3 | `4d5e60a` MPS: block_project runs a block per thread | `mptorch/csrc/mps/block_project.metal`, `mptorch/csrc/mps/block_entry.cpp` | no |
| M6a | `482a291` MPS: binary32 round-to-odd division and square root | `mptorch/csrc/common/carrier_odd_metal.h` (one hunk, Metal-only) | yes: **Linux reruns `binary_identity.py`** (expected: no object moves) |
| this report | the commit that adds this file | `dev/mac_parallel_report.md` | |

M1 and M5 changed no file.

**What Linux is asked to do:**
1. Rerun `binary_identity.py` for `482a291`'s header. Nothing should move,
   since only Metal compiles that header.
2. SP-1: decide on M1's `__aarch64__` arm in `fixedpoint_word_call_v`
   (section 4.4 above), and push it if taken; the Mac then reruns 4.4.
3. Let `dev/benchmarks/timing/fixedpoint_gemm.py ops --device mps` skip
   its wide row on MPS (section 4.5 above).
4. At SP-AI, match M6a's interface on the host and CUDA, in
   `mptorch::gemm::odd`.
5. Fold M5's findings into AJ's plan.

Outside the repository there are two worktrees, `../mptorch-base` (at
`85d1d31`, the committed MPS kernels for M2/M3's A/B) and `../mptorch-wide`
(`d612464` with M1's `__aarch64__` arm). `git worktree remove` deletes
each when it is no longer needed.

## The full suite on the final tree

With M2's, M3's and M6a's changes, at `d612464`: **19,759 passed, 11,430
skipped, 0 failed**, in 9:19. The cache was partly cold, since the odd,
scaled and block_project kernels recompiled.
