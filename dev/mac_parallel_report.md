# Mac report for `dev/mac_parallel_handoff.md`

One section per task, as the handoff's section 2.3 asks. The machine is an
Apple M3 Pro (5 performance and 6 efficiency cores, 18 GB), macOS 26.7,
Apple clang 21.0.0, torch 2.15.0, Python 3.14.10. Every build has
`MPTORCH_ODD_ACCUMULATED=1`. Every timing ran on AC power, with `pmset -g
therm` reporting no thermal or performance warning, and nothing else
running. The changes are committed on `feature-refactor`, one commit per
task (each round's last section lists them). Round 2, the handoff's
section 0, starts at "Round 2" below.

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

# Round 2 (the handoff's section 0, at `6afa44a`)

Same machine, state and build flag as round 1. `git pull` brought
`7d879b9` (SP-1), `5376e5f` and `73052e1` (phase AI's primitives and ops)
and `6afa44a` (the handoff's round 2). Each task below is a commit of its
own; the last section of this round lists them.

## R2.0: verification of AI's CPU code on arm64

### Build

`MPTORCH_ODD_ACCUMULATED=1 uv sync --group hf` after clearing Inductor's
cache: 28 s, with ccache. This is the first arm64 build of
`cpu/pointwise_*.cpp`, `cpu/carrier_odd_kernel.cpp`'s op codes 3 and 4 and
`common/carrier_odd_div_sqrt.h`'s host primitives (`std::fma`,
`std::frexp`, `std::bit_cast`) under Apple clang 21. The new units
(`cpu/pointwise_{binaryK,superfp,fixedpoint,rne}.cpp`,
`cpu/carrier_odd_kernel.cpp`, `meta/pointwise_meta.cpp`), compiled with
the build's flags, print no warning.

**The NEON blocks vectorize.** `cpu/pointwise_rne.cpp` with the build's
flags plus `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize`:

```
14 x pointwise_rne.cpp:44:1: remark: vectorized loop (vectorization width: 4, interleaved count: 1)
 1 x pointwise_rne.cpp:106:1: remark: vectorized loop (vectorization width: 4, interleaved count: 4)
```

There are no missed-loop remarks. Line 44 is `block<OP, HAS_Y>`'s loop,
seven ops times the two operand kinds, the root included. Line 106 is the
NaN reduction. Clang needs no pragma for this: its defaults on Darwin are
no-trapping-math and no-math-errno, which the GCC pragma sets on Linux. The
timings below agree: on one thread the fused RNE root takes 1.89 ms, the
time of the add (1.87 ms). The tool's docstring gives the Ryzen 6.76 ms for
the root against 1.54 for the add.

### Tests

| command | result |
|---|---|
| `pytest tests/test_pointwise_fused.py tests/test_cpu_simd.py tests/test_carrier_rounding.py tests/test_fake_kernels.py tests/test_pointwise.py tests/test_pwl.py` | **5,937 passed, 9,479 skipped, 0 failed** |
| the same, `-k "division_and_the_root and cpu"` | 55 passed (op codes 3 and 4 on the CPU against the exact reference) |
| `pytest tests/` | **23,999 passed, 19,843 skipped, 0 failed**, in 7:41 |

`test_cpu_simd.py` runs its cases in an interpreter per setting. On arm64
those settings are `MPTORCH_CPU_SIMD=0` (the scalar loop), the default
(NEON), and `avx2` (NEON again, since the cap names an x86 ISA). Its
pointwise cases agree word for word with the scalar loop.

Skips other than CUDA's, in the six files:

| skips | reason |
|---|---|
| 4,131 | `test_pointwise_fused.py`: `no MPS kernel yet` (M6b) |
| 75 | `test_carrier_rounding.py`: `the MPS _carrier_odd has no op codes 3 and 4 yet` (M6b) |
| 287 | `MPS has no float64 tensors` |
| 9 | `Inductor's CPU code generation is exercised on Linux` |
| 2 | the odd kernels' build-flag cases (`the odd kernels are built`, ...) |

The full suite's skips other than CUDA's are float64 on MPS (788, and
1), M6b's missing kernels (4,131 and 75), Inductor's CPU code generation
(9), the odd kernels' build-flag cases (3), and the tests' own rules for
combinations that do not exist (56 TREE without products, 8 fused without
a tree, 6 unsigned symmetric, 2 OCP, 2 KAHAN, 2 bfloat16 GEMMs, 1 fused
always integer).

### Timings (`dev/benchmarks/timing/pointwise_ops.py`)

AC power, no thermal warning before or after, 2048 x 2048 on the CPU and
4096 x 4096 on MPS, float32 then bfloat16 forward, ms. `2pass` is the path
these ops took before their fused op. The glibc variables do not apply on
macOS, so the CPU rows ran without them.

| float32, forward | torch | rne | 2pass | 2pass/rne | sr | f64 | ro |
|---|---|---|---|---|---|---|---|
| CPU x1 add | 0.46 | 1.87 | 6.09 | 3.3x | 13.88 | 9.08 | 11.77 |
| CPU x1 mul | 0.46 | 1.88 | 6.11 | 3.3x | 13.86 | 9.11 | 12.20 |
| CPU x1 div | 0.46 | 1.96 | 6.11 | 3.1x | 13.87 | 9.09 | 12.36 |
| CPU x1 sqrt | 0.56 | 1.89 | 7.82 | 4.1x | 14.02 | 8.96 | 12.39 |
| CPU x1 pwl | 3.54 | 13.45 | 22.48 | 1.7x | 21.35 | 15.66 | |
| CPU x11 add | 0.39 | 0.52 | 1.61 | 3.1x | 2.78 | 2.64 | 2.45 |
| CPU x11 mul | 0.39 | 0.52 | 1.62 | 3.1x | 2.78 | 2.64 | 2.52 |
| CPU x11 div | 0.39 | 0.53 | 1.62 | 3.1x | 2.78 | 2.76 | 2.58 |
| CPU x11 sqrt | 0.21 | 0.47 | 2.66 | 5.7x | 2.80 | 2.23 | 2.56 |
| CPU x11 pwl | 0.80 | 2.75 | 8.26 | 3.0x | 4.31 | 3.66 | |
| MPS add | 1.52 | 2.51 | 2.53 | 1.0x | 2.55 | | |
| MPS sqrt | 1.02 | 2.05 | 2.00 | 1.0x | 2.01 | | |
| MPS pwl | 1.02 | 24.23 | 24.28 | 1.0x | 24.21 | | |

| bfloat16, forward | torch | rne | 2pass | 2pass/rne | sr | f64 | ro |
|---|---|---|---|---|---|---|---|
| CPU x1 add | 1.31 | 2.44 | 7.83 | 3.2x | 15.15 | 15.05 | 13.57 |
| CPU x1 sqrt | 0.65 | 2.24 | 9.09 | 4.1x | 15.40 | 14.68 | 13.51 |
| CPU x1 pwl | 4.40 | 14.58 | 23.78 | 1.6x | 22.74 | 21.33 | |
| CPU x11 add | 0.31 | 0.54 | 2.32 | 4.3x | 2.88 | 3.81 | 2.90 |
| CPU x11 sqrt | 0.17 | 0.52 | 3.14 | 6.0x | 2.91 | 3.18 | 2.83 |
| CPU x11 pwl | 0.94 | 3.05 | 8.83 | 2.9x | 4.53 | 4.54 | |
| MPS add | 0.78 | 4.85 | 4.83 | 1.0x | 4.84 | | |
| MPS sqrt | 0.51 | 3.53 | 3.53 | 1.0x | 3.55 | | |
| MPS pwl | 0.52 | 25.69 | 25.70 | 1.0x | 25.66 | | |

The full outputs, with the backward rows and the other activations, are
the tool's own; the numbers above are its lines. On the CPU the fused ops
take 1/3 to 1/6 of the two passes' time. MPS still takes the two passes
for every variant, which is what M6b is to beat.

**The tool has a bug on MPS.** `_timer` synchronizes CUDA and not MPS, so
`--device mps` times the launches only: every row read 0.00 ms. The MPS
rows above are from a scratch copy that adds `torch.mps.synchronize()`
beside each `torch.cuda.synchronize()`:

```diff
     if device == "cuda":
         torch.cuda.synchronize()
+    elif device == "mps":
+        torch.mps.synchronize()
     t = time.perf_counter()
     for _ in range(n):
         fn()
     if device == "cuda":
         torch.cuda.synchronize()
+    elif device == "mps":
+        torch.mps.synchronize()
```

The tool is Linux's, so I did not change it.

## R2.1: SP-1's rerun of the arm64 timing (section 4.4)

Three builds, each with this checkout's venv: "base" `85d1d31`
(`../mptorch-base`), "new" `d612464` (`../mptorch-wide`, reset to the
commit), and "arm", the committed tree, which carries `7d879b9`'s arm. Same
script and method as round 1: one thread, 384^3, four rounds with the
order rotated, minimum in ps per mac step.

| row | mode | base | new | arm | new/base | arm/new | arm/base |
|---|---|---|---|---|---|---|---|
| carrier | RNE | 1,193 | 1,186 | 1,192 | 0.99 | 1.01 | 1.00 |
| carrier | RNA | 11,140 | 11,270 | 3,251 | 1.01 | **0.29** | 0.29 |
| carrier | RU | 13,055 | 2,977 | 2,808 | 0.23 | 0.94 | 0.22 |
| carrier | RD | 12,910 | 2,296 | 2,297 | 0.18 | 1.00 | 0.18 |
| carrier | RZ | 11,624 | 2,288 | 2,292 | 0.20 | 1.00 | 0.20 |
| carrier | RO | 12,148 | 11,821 | 3,588 | 0.97 | **0.30** | 0.30 |
| carrier | SR | 10,649 | 9,714 | 9,691 | 0.91 | 1.00 | 0.91 |
| int split | RNE | 1,554 | 1,554 | 1,552 | 1.00 | 1.00 | 1.00 |
| int split | RNA | 5,385 | 5,534 | 3,714 | 1.03 | **0.67** | 0.69 |
| int split | RU | 3,502 | 2,111 | 2,112 | 0.60 | 1.00 | 0.60 |
| int split | RD | 3,054 | 1,656 | 1,650 | 0.54 | 1.00 | 0.54 |
| int split | RZ | 3,208 | 1,775 | 1,772 | 0.55 | 1.00 | 0.55 |
| int split | RO | 3,571 | 3,637 | 2,400 | 1.02 | **0.66** | 0.67 |
| int split | SR | 11,960 | 8,329 | 8,329 | 0.70 | 1.00 | 0.70 |
| int fused | RNE, RNA, RU, RD, RO | 2,137-3,272 | the same | the same | 1.00 | 1.00 | 1.00 |
| int fused | RZ | 2,646 | 2,646 | 2,753 | 1.00 | 1.04 | 1.04 |
| int fused | SR | 6,391 | 5,251 | 5,285 | 0.82 | 1.01 | 0.83 |

The spread within a build is at most 6.2% (the carrier's RO), and mostly
under 1%.

**Findings:**
- **The committed tree is round 1's arm build.** RNA and RO take
  0.29-0.30x on the carrier path and 0.66-0.67x on the integer split, as
  round 1 measured. Every other row is unchanged within 1%.
- **The fused mac under RZ is a placement effect.** It takes 4% longer in
  the committed build, stable over the four rounds (2,753-2,756 against
  2,646-2,650). The fused mac never calls `fixedpoint_word_call_v`, and
  `cpu/inst/custom_matmul_fixedpoint_fma.o` is byte-identical between the
  two builds (`cmp`). So the only difference is where the linker puts it:
  the committed `.so` holds phase AI's units too.

## M4: the fixed-point GEMM's K-loop layouts on Metal (at `6afa44a`)

### What changed

`mptorch/csrc/common/fixedpoint_gemm.h` alone. The host and CUDA layouts
(`cast_fixedpoint_stochastic_gemm`, `fixedpoint_round_sr`,
`fixedpoint_word_round`, their constants and the policies' members) were
behind `#if !defined(__METAL_VERSION__)`. They were already almost valid
MSL, so the guards come off and Metal compiles the same text. Its
differences:

- **MSL 3.1 accepts `[[no_unique_address]]`, `std::conditional_t` (the
  prelude's shim) and `inline` variable templates.** A probe compiled MSL
  3.1 source with `MTLCompileOptions` as `metal_runtime.mm` sets them, with
  `-Weverything` as an error except the pre-C++14 and C++17-extension
  groups. It showed that a namespace-scope variable needs the `constant`
  address space ("program scope variable must reside in constant address
  space"). So `fixedpoint_word_mode_v` and `fixedpoint_word_call_v` are
  now `inline constexpr MPTORCH_CONSTANT bool`. `MPTORCH_CONSTANT` is empty
  on the host and CUDA, so their token streams are unchanged.
- **Two locals were named `half`**, an MSL type, in
  `cast_fixedpoint_stochastic_gemm` and `fixedpoint_word_round`. They are
  now `half_unit`, as in `stochastic.h`.
- **`fixedpoint_word_call_v` gets a Metal arm of its own**, ahead of the
  `__aarch64__` one, as the handoff asks: every deterministic mode
  (`fixedpoint_word_mode_v<RM>`).
- **`fixedpoint_word_round`'s choice of spelling.** Metal takes the
  device's logical form: `#if defined(__CUDA_ARCH__)` becomes `#if
  defined(__CUDA_ARCH__) || defined(__METAL_VERSION__)`.
- The comments say where each layout runs, with Metal's measurements.

Under SR the carrier path and the integer accumulator take the new
layouts on Metal as on the other backends. The Metal flush rules hold. The
line reads only the word of the value it casts. The constants are built
from the literal Args (folded by the compiler), and with the line on, the
step and the range's ends are normal values. A format whose step is a
binary32 subnormal (`FixedPoint(12, 140)`, `FixedPoint(24, 135)`) has
`span = 0`, so every input takes the shipped cast, as before.

**The hunks, all in `common/fixedpoint_gemm.h`, for Linux's
`binary_identity.py`.** Eleven `#if !defined(__METAL_VERSION__)`/`#endif`
pairs or `#else` arms are removed around:
- the SR section (`FixedPointStochasticT` ... `FixedPointIntSrMember`);
- the deterministic section (`fixedpoint_bitwise_cast` ...
  `FixedPointWordMember`);
- the two policies' members, constructors and calls;
- `FixedPointIntMac::isr`, `round_sum` (whose Metal `#else` arm goes) and
  `make_mac`'s `isr`.

The token changes on the host and CUDA:
- `half` becomes `half_unit` (two functions);
- `MPTORCH_CONSTANT` is added (empty there);
- `__METAL_VERSION__` is added to two `#if`/`#elif` lines;
- the comments change.

`git diff 6afa44a -- mptorch/csrc/common/fixedpoint_gemm.h` shows each
one. The changed lines are clang-formatted.

**Host objects under Apple clang: identical.** I rebuilt this tree with
`6afa44a`'s header and saved its 325 objects, rebuilt it with M4's, and
compared with `cmp`: only `mps/metal_runtime.o` differs, and that unit
embeds the Metal sources. The rename moves nothing here. Linux's x86-64 and
CUDA run is still the gate. `bash dev/benchmarks/check_native.sh`: `all
native tools build`.

### The gate

| check | result |
|---|---|
| `pytest tests/test_fixedpoint_mps.py` | **257 passed** (every mode, the subnormal-step formats included) |
| a GPU port of `gemm_cast_fixedpoint_arith.cu det-short` (the line against `fixedpoint_bitwise_cast`, both on the GPU, all 2^32 words) | `6 formats, 36 cases (format, mode); 0 cases mismatch` |
| its `det-control` | `RNE against the shipped RNA 68478 mismatches, RU against RD 8321159945`: Linux's CUDA counts exactly, so every word ran |
| the same for `sr-short` (`cast_fixedpoint_stochastic_gemm` against `cast_fixedpoint_stochastic`) | `6 formats, 54 cases (format, N, variant); 0 cases mismatch` |
| its `sr-control` | `1941258420 mismatches`: Linux's count exactly |
| `pytest tests/` | **23,999 passed, 19,843 skipped, 0 failed**, in 6:19 (R2.0's counts) |

The GPU sweep is a scratch script, not committed. It compiles the build's
embedded prelude and `fixedpoint` root (`build/generated/mps_metal_sources.h`)
with sweep kernels through `torch.mps.compile_shader`, and builds the
constants on the GPU from runtime values, with the handoff's six formats
`(8, 4)`, `(16, 8)` symmetric, `(16, 4)`, `(12, -20)` unsigned,
`(9, 125)` unsigned and `(10, 127)`. It takes 5 s for the deterministic
modes and 30 s for SR. If it would help, it could become a mode of
`dev/benchmarks/mps_backend.py`.

### Which modes take the line on Metal

Method: three builds alternated for four rounds, the order reversed every
other round, through the ops at 1024^3 (min of 5 calls, ps per mac step):
- "base", `6afa44a` (`../mptorch-m4base`);
- "bit", every mode on the line in the bitwise (host) form;
- "log", the same in the logical (device) form.

The rows are the handoff's carrier path ((16, 8) into (16, 4), inside the
RNE fast gate), the same past the gate ("carrier 24": (24, 12) into
(24, 12), 23 magnitude bits), the integer split (the op's `integer` hook)
and the fused mac. The spread within a build is at most 1.9%, except the
5 ps carrier RNE row (6%).

| row | mode | base | bit | log | bit/base | log/base |
|---|---|---|---|---|---|---|
| carrier | RNE | 5.0 | 5.0 | 5.0 | 1.00 | 1.00 |
| carrier | RNA | 24.5 | 22.8 | 22.5 | 0.93 | 0.92 |
| carrier | RU | 48.4 | 23.6 | 22.2 | 0.49 | 0.46 |
| carrier | RD | 48.3 | 22.8 | 22.1 | 0.47 | 0.46 |
| carrier | RZ | 57.5 | 18.8 | 18.8 | 0.33 | 0.33 |
| carrier | RO | 24.7 | 23.7 | 22.6 | 0.96 | 0.92 |
| carrier | SR | 75.1 | 60.5 | 60.4 | 0.81 | 0.80 |
| carrier 24 | RNE | 30.6 | 26.2 | 25.0 | 0.86 | 0.82 |
| carrier 24 | RNA | 24.5 | 22.4 | 22.0 | 0.92 | 0.90 |
| carrier 24 | RU | 48.3 | 22.9 | 21.5 | 0.47 | 0.44 |
| carrier 24 | RD | 48.4 | 22.1 | 21.4 | 0.46 | 0.44 |
| carrier 24 | RZ | 57.6 | 18.1 | 18.2 | 0.31 | 0.32 |
| carrier 24 | RO | 24.7 | 23.4 | 22.2 | 0.95 | 0.90 |
| int split | RNE | 36.0 | 36.0 | 36.0 | 1.00 | 1.00 |
| int split | RNA | 46.9 | 46.5 | 46.1 | 0.99 | 0.98 |
| int split | RU | 52.1 | 42.2 | 41.1 | 0.81 | 0.79 |
| int split | RD | 47.8 | 37.3 | 36.6 | 0.78 | 0.77 |
| int split | RZ | 57.3 | 39.6 | 39.5 | 0.69 | 0.69 |
| int split | RO | 41.2 | 41.6 | 40.1 | 1.01 | 0.97 |
| int split | SR | 98.5 | 81.8 | 81.7 | 0.83 | 0.83 |
| int fused | RNE, RNA, RU, RD, RZ, RO | 40.8-50.4 | the same | the same | 1.00 | 1.00 |
| int fused | SR | 77.7 | 63.4 | 63.6 | 0.82 | 0.82 |

So Metal takes **every deterministic mode on the line, in the logical
form**, and the SR layouts. No row is slower than the base. The logical
form is at least as fast as the bitwise one in every row, and faster
where the bitwise one gained least (RO on the integer split, 1.01x against
0.97x; RNE past the gate, 0.86x against 0.82x). Unlike the arm64 host's
5.4x, RNE on the line costs Metal nothing inside the fast gate, since the
fast path comes first, and gains 0.82-0.83x past it.

The committed tree, timed again against the base in the same way, gives
the numbers the header's comment quotes:
- the carrier path: RZ 0.32-0.33x, RU and RD 0.44-0.46x, RNA and RO
  0.90-0.92x, RNE 0.83x past the gate, SR 0.81x;
- the integer split: RZ, RU and RD 0.69-0.79x, RNA 0.98x, RO 0.97x, SR
  0.83x;
- the fused mac: SR 0.82x.

### `fixedpoint_gemm.py` on the committed tree, for the docs

`ops --device mps --modes RNE RNA RU RD RZ RO SR`, ps per mac step (round
1's numbers, before M4, in parentheses):

| row | RNE | RNA | RU | RD | RZ | RO | SR |
|---|---|---|---|---|---|---|---|
| binaryK split, Binary16p9 into Binary16p12 | 10.4 | 18.0 | 82.1 | 82.2 | 109.4 | 16.8 | 62.3 |
| fixed point, the carrier path | 5.0 (5.0) | 22.5 (24.5) | 22.1 (48.4) | 22.1 (48.4) | 19.4 (57.8) | 22.6 (25.3) | 60.7 (75.3) |
| fixed point, integer accumulator, split | 36.0 (36.0) | 46.1 (46.9) | 41.1 (52.2) | 36.6 (47.8) | 39.6 (57.3) | 40.1 (41.2) | 81.6 (98.5) |
| fixed point, integer accumulator, fused | 49.9 (49.8) | 50.5 (50.5) | 45.7 (45.9) | 40.8 (40.8) | 46.1 (46.1) | 45.3 (45.2) | 63.8 (77.8) |

`layer --device mps`, a step:
- plain: 6.99 ms;
- binaryK: 133 ms (RNE) and 783 ms (SR);
- fixed point: 63 ms (RNE) and **842 ms (SR, 993 before)**.

Under SR the carrier path now takes 0.97x the binaryK split mac's time
(1.21x before). Under RZ, RU and RD it takes 3.9-4.4x its own RNE time,
down from 10-12x. These rows are what `fixedpoint_gemm.py`'s M3 Pro table
and `docs/source/kernels/arithmetic.rst`'s cost paragraph could take. Both
are Linux's.

## M6b: the fused pointwise and table ops on MPS (at `ba9d01d`, rebased on `502eb15`)

### What was built

The six ops (`binaryK_pointwise`, `superfp_pointwise`,
`fixedpoint_pointwise` and their `_pwl` siblings) on an Apple GPU, the MPS
`_carrier_odd`'s op codes 3 and 4, and the switch that sends
`mptorch.quant.pointwise`'s calls on MPS to them.

**The kernel.** One kernel source, `KernelSource::Pointwise`, compiles from
three parts:
- `mps/pointwise.metal`, the head: `common/carrier_odd.h` (so
  `carrier_odd_metal.h`'s primitives, division and the root included),
  `common/pointwise.h` and `mps/pointwise_params.h`, flattened against the
  prelude's headers as `fixedpoint.metal` is;
- the instantiation's lines;
- `mps/pointwise_kernel.metal`, which includes `quantize.metal`.

The design choices:
- **The cast is `quantize.metal`'s `quantize_one`.** That is the MPS
  quantizer's own cast, under the quantizer's own tail. `quantize_tail` and
  the three format-line builders moved from `quantize.cpp` into
  `mps/quantize_tail.h`, unchanged, so the quantizers' tails, and their
  kernels' text, are byte-identical.
- **The step is `common/pointwise.h`'s `ArithStep` or `PwlStep`** (`MPT_PWL`
  in the tail).
- **The op, the carrier rounding, the number operand and a table's
  constants are launch parameters** (`PointwiseLaunch`), as the CPU's step
  takes them, so a format compiles once for all seven ops and both carrier
  roundings.
- **The geometry is `PointwiseGeom`, a constant buffer** copied into the
  thread, since `pointwise::offsets` takes a thread reference. It is
  size-asserted at 240 bytes on the host. `make_geom(..., device=true)`
  keeps a non-flat geometry under 2^31 elements.
- **Each thread takes four elements**, element j taking word j & 3 of
  Philox block j >> 2, as `quantize.metal` does. The index type is 32 bits
  when every index fits, as in M2.
- **The entry (`mps/pointwise_entry.cpp`) is the CPU entries' body in their
  order:** the format checks and the canonical round mode; then the
  operands' checks, the result and the geometry; under SR the seed from the
  CPU generator, drawn before the empty test; then the launch.
- **No `PointwiseLaunch` in `launch_params.h`.** The prelude includes that
  header, and adding a struct there would have changed every kernel's
  source.

**`mptorch/quant/pointwise.py`:** `_FUSED_DEVICES` gains `"mps"` (it serves
`pwl._rounded` too). So `qadd`, `qsub`, `qmul`, `qdiv`, `qsqrt` and the
table call sites run one fused op on MPS, and round to odd runs there,
where it raised before.

### Two Metal arms in `common/pointwise.h` (for `binary_identity.py`)

Both are behind `#if defined(__METAL_VERSION__)`, with the host and CUDA
statements verbatim in the `#else`:
1. **`PwlStep::e` is `device const T *` on Metal.**
2. **The table's range and relu tests compare on the words.** I added
   `metal_less` and `metal_order`, Metal-only, ahead of `PwlStep`, and they
   are used in `interpolate`'s clamp (`x < lo`, `x > hi`) and its relu tails
   (`x < lo || x > hi`, `x > 0`). The GPU's compare flushes a subnormal
   operand to zero. So under clamp tails at `lo = 0`, a negative subnormal
   was not clamped. Under relu tails with a positive `lo`, a positive
   subnormal's relu became 0 instead of itself.

The rest of the header's Metal text, Linux's, needed no change:
- the `rn::` arms;
- the casts' factories (unused, since the kernel casts with
  `quantize_one`);
- `floor_of`, `word_of`, `float_of`, `power_of_two` and `exponent_of`.

The `rn::add` arm's `metal_sub_rn(x, -y)` relies on negation keeping a
subnormal. A GPU probe confirmed it: `-x` flips the sign bit, subnormals
and NaN payloads included, while `0.0f - x` and the compares flush. Torch's
own MPS `neg` keeps subnormals too, and its `+` flushes them.

### The MPS `_carrier_odd`

`mps/carrier_odd.metal` gains `MPT_ODD_OP` 3 (`odd::div`) and 4
(`odd::sqrt`), M6a's primitives. `mps/carrier_odd_entry.cpp` passes `5` to
`check_carrier_odd`, as the CPU and CUDA entries do.

### Tests

These are the shared test lines, with the reason for each:
- **`tests/test_pointwise_fused.py`:**
  - `DEVICES` loses its skip.
  - The references move to the CPU on MPS (`_reference_device`): the
    two passes' torch operations on MPS flush binary32 subnormals, and
    2^-140 is among the operands. The MPS ops return the CPU's words, so
    the references are the CPU's two passes, the odd step's float64 (which
    MPS lacks), and `pwl._evaluate`.
  - The generator checks read the generator the ops draw from (the CPU's
    on MPS). They are unchanged on the CPU and CUDA.
  - `test_the_pointwise_ops_run_one_fused_op` drops its
    `carrier=torch.float64` calls on MPS.
- **`tests/test_carrier_rounding.py`:** `QUOTIENT_DEVICES` and
  `QUOTIENT_ACCELERATORS` lose their MPS skip.
- **`tests/test_pointwise.py`:** `test_round_to_odd_and_binary64_fail_on_mps`
  pinned the old refusal. It becomes
  `test_round_to_odd_runs_and_binary64_fails_on_mps`: RO on MPS gives the
  CPU's words for add, sub, mul, div (under SR) and sqrt, and a binary64
  carrier still fails.
- **`tests/test_mps.py`, new section "The fused pointwise and table ops"**,
  every case against the CPU op word for word:
  - every op, both carrier roundings and every mode, on operands built so
    that sums, differences, products, quotients and roots land among
    binary32's subnormals and near 2^-126;
  - formats that reach them (binaryK `(12, 5)` at bias 127, superfp
    `(3, 4, 1)` at bias 22, `FixedPoint(8, 125)`, binaryK `(28, 20)`);
  - every 16-bit word of float16 and bfloat16 as the first operand;
  - tables clamped at 0, with relu tails from 2^-120, extended, and the
    1/sqrt table;
  - strided, broadcast and over-six-dimension geometries.

  The new tests accept any NaN where the CPU has one. The GPU's float unit
  hands a NaN step out as its canonical NaN, where the CPU keeps the
  operand's payload. This is the payload exception the docs state for the
  fused ops ("A NaN is a NaN, whatever its payload"), and CUDA behaves the
  same.

| check | result |
|---|---|
| `pytest tests/test_pointwise_fused.py -k mps` | **3,105 passed**, 1,026 skipped (`MPS has no float64 tensors`) |
| `pytest tests/test_carrier_rounding.py -k mps` | 380 passed (the 75 quotient cases included), 321 skipped (float64) |
| `pytest tests/test_mps.py -k "pointwise or pwl"` | **223 passed** |
| negative control: the Metal arms replaced by the GPU's plain operations (my two compares, and `rn::add`/`rn::mul` as `x + y`, `x * y`), rebuilt | 173 of the 223 fail, 62 of the 72 clamp/relu table cases among them; restored afterwards |
| `pytest tests/` | **27,364 passed, 16,701 skipped, 0 failed**, in 7:56, and the same on the tree rebased on `502eb15` (phase AK's tiles in). The `no MPS kernel yet` (4,131) and quotient (75) skips are gone; the MPS skips left are float64's and the tests' own rules (R2.0's list) |
| lint: `ruff check`, `ruff format --check`, `ty check mptorch` | pass |

**Host objects under Apple clang:** compared with M4's build (`cmp` over
326 objects), only `mps/quantize.o`, `mps/metal_runtime.o`,
`mps/carrier_odd_entry.o` and `mps/mps_ops.o` differ, plus the new
`mps/pointwise_entry.o`. Every CPU and Meta object is byte-identical:
`cpu/pointwise_*.o` and `meta/pointwise_meta.o`, which include
`pointwise.h`, and every unit that includes `quant_ops.h`.

### Timing

`pointwise_ops.py --device mps`, with R2.0's scratch fix for its timer.
4096 x 4096, AC power, no thermal warning. ms, with R2.0's two-pass base
beside:

| float32, forward | torch | before (two passes) | fused rne | fused sr | `2pass` column |
|---|---|---|---|---|---|
| add | 1.53 | 2.51 | **1.68** | 1.59 | 2.56 |
| mul | 1.53 | 2.57 | 1.67 | 1.59 | 2.55 |
| div | 1.52 | 2.54 | 1.62 | 1.61 | 2.50 |
| sqrt | 1.02 | 2.05 | **1.06** | 1.06 | 2.01 |
| pwl (`qpwl`, `silu_table()`) | 1.02 (SiLU) | 24.23 | **1.46** | 1.69 | 24.26 |

| bfloat16, forward | torch | before | fused rne | fused sr |
|---|---|---|---|---|
| add | 0.77 | 4.85 | **0.91** | 1.06 |
| sqrt | 0.51 | 3.53 | 0.76 | 1.00 |
| pwl | 0.52 | 25.69 | **1.49** | 1.72 |

The fused ops cost 1.0-1.2x torch's own op in float32 and 1.2-1.5x in
bfloat16 under RNE. That is 1.5-1.9x faster than the two passes in float32
and 4.7-5.3x in bfloat16. A table is 16-17x faster. With backward, float32
add, mul, div and sqrt take about 1 ms less (8.74 against 9.64 for add).
The ops that do not fuse (exp, gelu, silu) are unchanged.

Round to odd costs nothing on MPS (a scratch timing of `qadd`, `qmul`,
`qdiv` and `qsqrt` at the same size): 1.65-1.68 ms against 1.63-1.67 under
RNE, and 1.05 against 1.06 for the root. The tool drops `ro` (and `f64`)
on MPS because RO used to raise there.

### Files

Mac-owned:
- new: `mptorch/csrc/mps/pointwise.metal`, `pointwise_kernel.metal`,
  `pointwise_entry.cpp`, `pointwise_params.h`, `quantize_tail.h`;
- changed: `quantize.cpp` (the builders moved out), `carrier_odd.metal`,
  `carrier_odd_entry.cpp`, `metal_runtime.h`, `metal_runtime.mm`,
  `mps_ops.cpp`, and `tests/test_mps.py`.

Shared, the hunks as described above:
- `mptorch/csrc/common/pointwise.h` (the two Metal arms);
- `setup.py` (`embed_metal_sources` gains `pointwise` and `pointwise_kernel`,
  the first flattened against the prelude's headers, and the docstring says
  so);
- `mptorch/csrc/quant_ops.h` (the six `_mps` declarations and the comment
  naming the MPS file);
- `mptorch/quant/pointwise.py` (`_FUSED_DEVICES`);
- `tests/test_pointwise_fused.py`, `tests/test_carrier_rounding.py` and
  `tests/test_pointwise.py`.

## Round 2's commits

| task | commit | files | shared? |
|---|---|---|---|
| R2.0, R2.1 | none | | |
| M4 | `e72a53b` MPS: the fixed-point GEMM's K-loop layouts on Metal | `mptorch/csrc/common/fixedpoint_gemm.h` | yes: merged; Linux's `binary_identity.py` IDENTICAL |
| R2.0, R2.1, M4's report | `ba9d01d` | `dev/mac_parallel_report.md` | |
| M6b | the commit "MPS: the fused pointwise and table ops" | the files listed under M6b | yes: `common/pointwise.h` needs **`binary_identity.py`** (expected: nothing moves; under Apple clang nothing did) |
| this report's M6b section | the commit after it | `dev/mac_parallel_report.md` | |

**What Linux is asked to do.** Linux has already merged M4 (its
`binary_identity.py` run was IDENTICAL), fixed `pointwise_ops.py`'s timer on
MPS, and taken M4's numbers into the docs (`502eb15`). What is left:
1. Rerun `binary_identity.py` for M6b's `common/pointwise.h`.
2. `dev/benchmarks/timing/pointwise_ops.py`'s `ro` variant can now run on
   MPS. The tool still drops it there; `f64` still cannot run.
3. Update the text that still says the fused ops run on the CPU and CUDA
   only:
   - `docs/source/kernels/pointwise.rst`: the round-to-odd paragraph
     ("On the CPU and CUDA the fused ops ...", and "Elsewhere (MPS) the two
     passes compute it in float64 ..."), "One pass" ("On the CPU and CUDA,
     ...", and "and MPS run the two passes"), and "What a table costs";
   - `mptorch/quant/pointwise.py`'s docstrings of the `carrier_rounding`
     argument ("on the CPU and CUDA, any float operands; elsewhere float32,
     float16 and bfloat16 ones") and of `qsqrt` ("with a `Quant` slot on
     the CPU and CUDA the fused op computes it"). On MPS the fused op now
     computes both, in binary32.

   M6b's tables above give the MPS costs.

Worktrees outside the repository:
- `../mptorch-base` (`85d1d31`) and `../mptorch-wide` (`6afa44a` with the
  logical-form experiment) served R2.1 and M4;
- `../mptorch-m4base` (`6afa44a`) was M4's base.

`git worktree remove` deletes them.

## M7: AK's NEON palette tiles on the M3 Pro (at `714aa11`, AK's `64ac91e` in)

No file changed except this report: M7 is verification and timing, with no
proposal for the tile files (see "NEON layout" below).

### Build and tests

The extension built with the rest of the merged tree. `cpu/gemm_palette_lanes.cpp`
compiled again with the build's flags plus `-Rpass=loop-vectorize
-Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize`:
- every cast loop vectorizes at width 4 (`lane_casts`' four, lines 87, 90,
  96 and 102, four instantiations each: per column and per element, the
  fixed 32-size tile and the run-time one);
- so do the products' inner loop (line 67, nine) and the NaN reduction
  (line 179, three);
- the only loop that does not is `lane_products`' outer row loop (line 62,
  "not beneficial"), as expected.

Clang needs no pragma for this, as for the pointwise blocks (R2.0).

| command | result |
|---|---|
| `pytest tests/test_cpu_simd.py tests/test_gemm_mixed_palette.py tests/test_palette_recipes.py` | 143 passed, 64 skipped (CUDA) |
| `pytest tests/test_qmatmul.py -k palette` | 1 passed |
| `pytest tests/test_carrier_rounding.py -k "palette or mixed"` (the odd palettes) | 132 passed, 130 skipped (CUDA, float64) |
| `pytest tests/` on the merged tree | 27,364 passed, 16,701 skipped, 0 failed (M6b's run) |
| `bash dev/benchmarks/check_native.sh` | `OK    sweeps/cpu_palette_tile_sweep.cpp`, `all native tools build` |

`test_cpu_simd.py`'s settings on arm64 are `0` (the scalar loop), the
default (NEON) and `avx2` (NEON again). Its palette cases agree word for
word with the scalar loop.

### The tile sweep under Apple clang

Built as section 4.3 builds the others, with Homebrew's libomp for the
standalone binary:

```
clang++ -O3 -std=c++20 -ffp-contract=off -DMPTORCH_FAST_CAST=1 -Xpreprocessor -fopenmp \
    -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp -I mptorch/csrc \
    dev/benchmarks/sweeps/cpu_palette_tile_sweep.cpp mptorch/csrc/cpu/gemm_palette_lanes.cpp \
    mptorch/csrc/cpu/gemm_rne_tile.cpp -o cpu_palette_tile_sweep
```

| run | output | time (11 threads) |
|---|---|---|
| `cpu_palette_tile_sweep` | `isa neon, 9 formats, a format per lane`; `every word, one-step tiles, per element and per column: 0 mismatches`; `whole tiles, a palette of three per block: 0 mismatches (172032 of 196608 blocks ran whole)`; `PASS` | 2:36 |
| `cpu_palette_tile_sweep control` | `83111807712 mismatches`; `PASS: the control mismatches, so the comparison can fail` | 2:33 |
| `MPTORCH_CPU_SIMD=0 cpu_palette_tile_sweep` | `no lane tile on this CPU or build (isa none): nothing to check` | |

### Timing (`dev/benchmarks/timing/cpu_palette_gemm.py`)

AC power, no thermal warning before or after; each palette against its
single-format twin, at 256^3 and 32 x 128 x 128 (the range over the two
shapes), with the Ryzen's numbers from the tool's docstring in brackets:

| | mac | one | columns | halves | rows | elements |
|---|---|---|---|---|---|---|
| 1 thread | split | 1.0x [1.0-1.1] | 1.5-1.6x [1.5] | 1.0x [1.1] | 1.0-1.1x [1.1] | 1.5x [1.5-1.6] |
| | fused | 1.0-1.1x [1.1] | 1.5x [1.4] | 1.0-1.1x [1.1] | 1.1x [1.2] | 1.7x [1.5-1.6] |
| 11 threads [16] | split | 1.0-1.2x [1.0-1.2] | 1.5-1.6x [1.3-1.5] | 1.0-1.1x [1.0-1.2] | 1.1x [1.1-1.3] | 1.4-1.5x [1.5-1.8] |
| | fused | 1.0-1.1x [1.2] | 1.3-1.5x [1.4-1.5] | 1.1-1.2x [1.2-1.3] | 1.1x [1.2-1.3] | 1.4-1.7x [1.5-1.7] |
| 1 thread, RO | split | 1.0x [1.0-1.1] | 1.3-1.4x [1.3-1.4] | 1.0x [1.0-1.1] | 1.2x [1.1-1.2] | 2.1x [2.2] |
| | fused | 1.0-1.1x [1.1] | 1.6-1.7x [1.6] | 1.0-1.1x [1.1] | 1.2x [1.3] | 2.1-2.2x [2.4] |
| 11 threads, RO | split | 1.0-1.1x | 1.3x | 1.0x | 1.1-1.2x | 1.8-1.9x |
| | fused | 1.0-1.1x | 1.3-1.4x | 1.0x | 1.1-1.2x | 1.7-1.9x |

The absolute times at 256^3 on one thread:
- **under RNE:** the split twin takes 11.29 ms and its palettes
  11.05-16.94; the fused twin 5.44 and its palettes 5.67-9.12;
- **under RO:** the twins 14.96 and 7.49, and the palettes 15.22-31.61 and
  7.72-16.56;
- **`MPTORCH_CPU_SIMD=0`**, the scalar loop for twin and palette alike:
  every map is 1.0x its twin, and the split twin takes 50.31 ms under RNE
  and 112.61 under RO.

So the palettes, which ran that scalar loop before AK, are now 2.8-4.6x
faster under RNE and 3.2-7.5x under RO on one thread. The M3 Pro's ratios
are within 0.1-0.2 of the Ryzen's:
- the fused per-element map on one thread is a little slower (1.7x against
  1.5-1.6x);
- the odd macs' per-element maps are a little faster (2.1-2.2x against
  2.2-2.4x).

### NEON layout

The handoff asks whether NEON prefers another layout. AA's single-format
tiles kept NEON's product-only and fused tiles row by row, while the lane
tiles run passes over a band of `LANE_BAND = 8` rows, sized for the Ryzen's
L1 (12 KB of per-element constants for the split mac). The M3 Pro's L1 is
128 KB, so I timed bands of 16 and 32 (the whole tile's 48 KB) and of 1
(row by row). Each was a local one-line change in a worktree, the builds
alternated through the tool, three rounds with the order rotated, on one
thread, under RNE and RO.

The per-element map against band 8:
- **16 and 32:** 0.98-0.99x under RNE, and 1.00-1.01x under RO, whose odd
  macs do not take the lane tiles;
- **1 (row by row):** 1.02x for the split mac and **1.10x** for the fused
  one.

**Placement is as large as these effects.** The per-column map, which never
reads the band, moved too: 0.94-0.95x (split) and 1.11x (fused) in the
band-32 and band-1 builds, and 1.00x in band 16's. The per-column and
per-element paths are inlined into one function per mac (`split_neon`,
`fused_neon`), so a change to one moves the other's loops. R2.1 saw the
same sensitivity: a 4% move with byte-identical objects.

**No proposal.** Passes over a band suit the M3 Pro's lane tiles (row by
row is 10% slower for the fused mac). The 1-2% that a wider band gains is
within what code placement moves on this machine.

### Open

The worktrees are back on their commits' text:
- `../mptorch-wide` and `../mptorch-m4base` are at `714aa11`, holding the
  band experiments' builds;
- `../mptorch-base` is at `85d1d31`.

`git worktree remove` deletes all three.

## M8: S's MPS part, the counted ops on Metal (at `127f83d`, rebased on `e57c983`)

The twelve counted ops run on MPS. Their results are the CPU's word for word
and their counts the CPU's count for count, SR included.

### The counting build under Apple clang

The first `MPTORCH_COUNTING=1` build failed at its first counted unit
(`cpu/custom_conv_counted_entry.cpp`; every counted unit includes the
header). Darwin's `<math.h>` defines `OVERFLOW` (3) and `UNDERFLOW` (4),
System V's `matherr` codes, under `__DARWIN_C_LEVEL >= __DARWIN_C_FULL`,
which a default compile meets. They replaced the enumerators
`events::OVERFLOW` and `events::UNDERFLOW` in `common/event_counts.h` and
their uses in `fixedpoint_counted.h`. Linux's `<math.h>` defines neither,
so Linux never saw it.

The fix is an Apple-host-only block after `event_counts.h`'s includes (which
have brought in `<cmath>` by then, through `bit_helper.h`):

```cpp
#if defined(__APPLE__) && !defined(__METAL_VERSION__)
#undef OVERFLOW
#undef UNDERFLOW
#endif
```

On Linux it preprocesses to nothing. Renaming the two enumerators, as
`NAN_RESULT` dodges `NAN`, would also work and would not depend on undefining
a system macro; that is Linux's call.

With it, the CPU counted kernels build and pass under Apple clang (before
any Metal work): `test_counted_gemm.py`, `test_simulate_events.py`,
`test_event_counts.py`, `test_entry_copies.py` gave 424 passed, 394 skipped
(CUDA, and the no-flag test). The task flush's `__atomic_fetch_add` builds
under Apple clang's libc++.

### What was built

**The kernel.** `gemm.metal` under `MPT_COUNTING`, compiled after one of two
new roots, as the odd and fixed-point kernels are compiled after theirs:
- `mps/counted.metal` (`KernelSource::GemmCounted`) holds
  `common/gemm_counted.h`, for the binaryK and superfp ops;
- `mps/fixedpoint_counted.metal` (`GemmFixedPointCounted`) holds
  `common/fixedpoint_counted.h`, which brings `fixedpoint_gemm.h` with it.

Both are flattened against the prelude's headers (`setup.py`), and both
spell `std::remove_const_t` (`using metal::remove_const_t;`), which the
counted Args use and the prelude's `std` lacks. A binaryK or superfp counted
kernel therefore never compiles the fixed-point header.

The tail is the Args' `mpt_args()` (`CountingArgs<Base, ALG>` or
`FixedPointCountingArgs<Args>`, written field by field by
`gemm_backend.cpp`, size-asserted) plus `#define MPT_COUNTING 1`, and, for a
conv pass, the usual `MPT_GATHERED` lines. The kernel builds the counted
policies itself, from the literal Args, through the Args' own
`with_accumulator`, as for every other Args. So the casts' constants are
immediates, as in the twin.

**Where the counts go.** Each thread's counts are its policies' `tally`
fields (32-bit, as on CUDA). After the store, `count_tile` sums them over
the threadgroup:
1. each SIMD group sums each of the 18 cells with `simd_sum`, over the
   cells' 16-bit halves, so no partial sum can wrap (32 x 65535 < 2^21);
2. after one barrier, thread c (< 18) adds the SIMD groups' halves into a
   64-bit total;
3. it adds the total to the call's int64 count with two 32-bit atomic adds:
   the low word, then its carry plus the high word into the high word.

The one add that wraps the low word sees the wrap in the value
`atomic_fetch_add` returns, so the count is exact in any order. Threads
outside C contribute zeros, and every thread reaches the barrier.

The handoff proposed a per-threadgroup row of partial counts, summed by the
entry point. The atomics need no partial buffer and no second kernel or
reduction, and they keep the entry points the CPU's text: the driver
(`gemm_counted_host.h`) allocates the counts and hands the backend only
their data pointer. On MPS that pointer is the tensor's `MTLBuffer` (what
`metal_runtime.mm`'s `buffer_of` reads), so `KernelArg::storage_at` binds
it as is, after the variant's buffers (5, or 6 for a conv pass). A conv
pass's per-class GEMMs (the input gradient) all add to the one tensor, as on
the other backends.

### The Metal spellings in `common/` (for `binary_identity.py`)

MSL 3.1 rejects a derived class outright: "derived classes are not
supported in Metal", from a compile probe as the runtime compiles (MSL 3.1,
safe math). `mutable` members compile.

**`common/gemm_counted.h`:**
1. **`Counting<Acc>`** on Metal holds the accumulator as a member, `acc`,
   and passes on the four calls `gemm.metal` makes (`seed_rng`,
   `accumulate`, `end_slab`, `finalize`). It declares `counting` and
   `positional`, and has an unused `counts` (a `uint32_t`), so that
   `CountingArgs`' initializers, `{{mac}, args.counts}`, are the host's
   text. The host's derived `Counting` is verbatim in the `#else`.
2. **`CountingArgs`** is now outside the host-only block. Its `counts` is a
   Metal arm (the `uint32_t`; the host's pointer verbatim in the `#else`).
   Its static members, the `Builder`'s two references and `operator()`'s
   parameter, and `with_accumulator`'s parameter gained `MPTORCH_CONSTANT` and
   `MPTORCH_THREAD`, the spellings `AccumulateArgs` already uses. They
   expand to nothing on the host and CUDA.
3. **`counting_task_end` and the CUDA flush functions** keep their guards,
   now an inner `#if !defined(__METAL_VERSION__)`.
4. **A Metal-only block at the end:** `counting_gather`,
   `counting_gather_mac`, `counting_gather_outer` (BLOCK's and TREE's
   `outer`, chosen by the accumulator's type, since Metal has no
   requires-expression) and `counting_tallies`, the Metal counterpart of
   `counting_thread_end`: they add each wrapper's tally into a row per role.
5. A bullet in the top comment says where the counts go on an Apple GPU.

**`common/fixedpoint_counted.h`:**
1. **`CountedFixedPointIntMac`** on Metal holds the integer mac as a member,
   `m`. Its `sum_bits` and `step` are the host's statements with `m.` for
   `this->` (the fused step's mantissa is `mant`, since `m` is taken), except
   one: the fused step's test of a huge product is on the words, as in
   `FixedPointIntMac::step`'s own Metal arm (`p_top` is a binary32 subnormal
   for a format whose top is below 2^-126). It has `value()`. The host's
   derived struct is verbatim in the `#else`.
2. **`PositionalFixedPointIntAccumulator`**: `tile_type`, `finalize_wide` and
   the tile functions behind `!defined(__METAL_VERSION__)`, as the other
   accumulators have them; `MPTORCH_CONSTANT` on its static members (the
   `int_result` comment moved to its own line).
3. **`counted_int_mac::fused` and `FixedPointCountingArgs`**:
   `MPTORCH_CONSTANT` and `MPTORCH_THREAD` as above. `counts` is a Metal arm.
4. **A Metal-only `counting_gather_mac`** for the integer mac's two tallies.

**`common/event_counts.h`:** the Apple-host `#undef` above.

**Evidence that the host text did not move.** I preprocessed
`common/fixedpoint_counted_host.h`, which includes all three headers, as a
host unit (`clang++ -std=c++20 -E -P -DMPTORCH_FAST_CAST=1
-DMPTORCH_COUNTING=1`, with `-D_POSIX_C_SOURCE=200809L` so that Darwin's two
macros are absent from both trees), at `127f83d` and on the working tree,
each from its own root. The two outputs differ in two lines only, by the
space an empty `MPTORCH_THREAD` leaves (`with_accumulator( F &&f)`). So the
host token streams are identical. On Linux the `#undef` block preprocesses
away too. CUDA's arms were not preprocessed here (no CUDA headers), but no
CUDA statement changed: they sit under the same conditions, nested one level
deeper.

Linux: please run `binary_identity.py` for these three headers. Only the
counted units include them, so the expected result is the counted objects
identical and nothing else touched.

### The host side

- `mps/gemm_backend.h`: `gemm_kernel_source` for the two counted Args;
  `variant_source` adds `MPT_COUNTING`; `counts_of(args)` is the counts
  pointer for a counted Args and null otherwise; it is passed from the
  generic `launch` and the conv `launch` to `launch_source`/`launch_conv`.
  No new `launch` overload was needed, and `CountingBackend<MpsBackend>`
  (the fixed-point drivers' wrapper) works unchanged.
- `mps/gemm_backend.cpp`: the two `args_source` templates (every field but
  `counts`, size-asserted), instantiated for the 16 `CountingArgs` (four
  macs, four algorithms) and the 5 `FixedPointCountingArgs`, all under
  `#if defined(MPTORCH_COUNTING)`; `launch_gemm` appends the counts.
- `mps/metal_runtime.{h,mm}`: the two `KernelSource`s (at the end of the
  enum) and `KernelArg::storage_at`.
- `mps/custom_{matmul,conv}_counted_entry.cpp`: the CPU files' text with the
  backend and the `_mps` suffix; `mps/mps_ops.cpp`: twelve lines.

Without the flag: the three MPS units compiled by hand without
`-DMPTORCH_COUNTING=1` build, and reference no counted kernel source (`nm
-u`: no `CountingArgs`, no `args_source`); the entries raise from the shared
driver, as on the CPU.

### The arithmetic the counts see

The events are decided on the words (`event_counts.h`), which the GPU's
flush cannot reach. But they describe the cast the GPU made: a product or
sum below 2^-126, which the GPU flushes to zero before the cast (gemm.metal's
"What the GPU's arithmetic cannot match"), is counted as the zero the cast
received. Where the CPU counts an underflow (or an inexact subnormal) for
such a cast, MPS counts nothing. `gemm.metal`'s new paragraph says so. None
of the tests' operands reach that range (`test_event_counts`' rows have no
binary32 subnormal, and the GEMM operands are `randn` scaled).

### Files

Mac-owned:
- new: `mptorch/csrc/mps/counted.metal`, `mps/fixedpoint_counted.metal`,
  `mps/custom_matmul_counted_entry.cpp`, `mps/custom_conv_counted_entry.cpp`;
- changed: `mps/gemm.metal`, `mps/gemm_backend.{h,cpp}`,
  `mps/metal_runtime.{h,mm}`, `mps/mps_ops.cpp`,
  `dev/benchmarks/mps_backend.py` (`bench-counted`).

Shared, with the hunks above:
- `mptorch/csrc/common/event_counts.h`, `common/gemm_counted.h`,
  `common/fixedpoint_counted.h` (**`binary_identity.py`**);
- `mptorch/csrc/quant_ops.h`: the twelve `_mps` declarations, each after its
  CPU twin, and the two section comments naming the MPS files;
- `setup.py`: the roots `counted` and `fixedpoint_counted`, flattened after
  the prelude (the tuple of such roots is now named `after_prelude`), and
  the docstring;
- `tests/test_counted_gemm.py`, `tests/test_simulate_events.py`,
  `tests/test_entry_copies.py` (below).

### Tests

The shared test lines, with the reason for each:
- **`tests/test_counted_gemm.py`:**
  - `counted_devices` gains `pytest.param("mps", marks=requires_mps)`, so
    every device-parametrized test runs on MPS against the same references
    (Fractions, `_recount`, the unfolded GEMMs);
  - `test_fixedpoint_paths_count_alike` lost its wide half, which is now
    `test_a_wide_fixedpoint_result_is_counted` over `float64_devices` (a
    wide result is float64, which MPS refuses), and
    `test_on_mps_a_wide_counted_result_is_refused` holds the MPS refusal;
  - `_fx_sr_mac(which)` factors `test_fixedpoint_stochastic_rounding`'s SR
    mac, which `test_mps_is_the_cpu` uses too;
  - **`test_mps_is_the_cpu`** (144 cases): MPS against the CPU, word for
    word and count for count, under one seed: every `CASES_32` case
    (four macs, both families, every schedule) under RNE, RO and SR; the
    eight fixed-point macs under RNE and SR; the four conv macs' three
    passes under RNE and SR. Under SR the per-device tests hold only the
    casts' totals, so this is where MPS's SR counts are checked;
  - `test_without_the_flag_the_ops_raise` gains MPS (it skips on this
    flag-on build);
  - the module docstring says MPS is held to the CPU.
- **`tests/test_simulate_events.py`:** `devices` gains MPS.
- **`tests/test_entry_copies.py`:** `PARTIAL_ENTRY_FILES` is empty again;
  the MPS entry files are held to the CPU's token for token.

| command (the flag-on build: `MPTORCH_ODD_ACCUMULATED=1 MPTORCH_COUNTING=1`) | result |
|---|---|
| the seven kinds of counted call (binaryK split NAIVE, fused KAHAN, superfp BLOCK, binaryK TREE, the fixed-point carrier path, integer accumulator and fused mac) at 37 x 41 x 29, MPS against the CPU | results and counts equal, first try |
| `pytest tests/test_counted_gemm.py tests/test_simulate_events.py tests/test_event_counts.py` | 896 passed, 399 skipped; every skip is a CUDA case, the no-flag test or one test's own rule ("fixed point has its own counted ops"); no MPS case skipped (489 collected) |
| the same plus `test_entry_copies.py` and `test_fake_kernels.py`, on the final tree | 1,406 passed, 413 skipped |
| `pytest tests/` | 28,286 passed, 17,101 skipped, 0 failed (13 min 50 s, the GEMM kernels compiling cold; M6b's run: 27,364 / 16,701) |
| `ruff check`, `ruff format --check` (mptorch, tests, dev/benchmarks), `ty check mptorch`, `ty check` on the changed tests and tool | clean |

The full suite ran on the build before a formatting-only pass over the
C++ lines I changed; the counted modules above ran on the rebuilt tree.

Rebased on `d314a01` (the opt-in kinds' counting, Python only) and
`e57c983`: its new tests in `test_simulate_events.py` run on MPS through
the same `devices` line and pass (the module: 21 passed, 7 skipped, all
CUDA); `test_counted_gemm.py`, `test_simulate_events.py` and
`test_entry_copies.py` together: 864 passed, 367 skipped.

### Timing (`dev/benchmarks/mps_backend.py bench-counted`, new)

AC power, no thermal warning before or after, two runs (they agree within
5%); twin / counted ms, the two alternated, best of five rounds each, and
no counted result differed from its twin's:

| op | 512^3 | 1024^3 |
|---|---|---|
| binaryK split NAIVE RNE | 1.64 / 4.10 (2.50x) | 11.33 / 30.66 (2.71x) |
| binaryK split KAHAN RNE | 3.25 / 8.83 (2.72x) | 24.06 / 68.82 (2.86x) |
| binaryK split BLOCK 32 RNE | 1.69 / 4.74 (2.81x) | 11.85 / 35.96 (3.03x) |
| binaryK split TREE 64 RNE | 2.23 / 5.90 (2.64x) | 15.91 / 45.13 (2.84x) |
| binaryK fused NAIVE RNE | 1.01 / 2.24 (2.23x) | 6.90 / 16.29 (2.36x) |
| binaryK split NAIVE SR | 8.20 / 10.16 (1.24x) | 64.14 / 79.50 (1.24x) |
| binaryK split KAHAN SR | 20.86 / 25.65 (1.23x) | 166.1 / 203.4 (1.22x) |
| binaryK split BLOCK 32 SR | 9.14 / 11.39 (1.25x) | 71.28 / 88.62 (1.24x) |
| binaryK split TREE 64 SR | 10.48 / 12.69 (1.21x) | 81.72 / 99.43 (1.22x) |
| binaryK fused NAIVE SR | 4.16 / 5.07 (1.22x) | 31.58 / 38.88 (1.23x) |
| superfp split NAIVE RNE | 2.75 / 5.03 (1.83x) | 20.64 / 38.50 (1.86x) |
| fixed point, carrier path, RNE | 0.86 / 3.09 (3.60x) | 5.35 / 22.84 (4.27x) |
| fixed point, integer accumulator, RNE | 5.04 / 7.77 (1.54x) | 38.67 / 60.56 (1.57x) |
| fixed point, fused, RNE | 6.95 / 8.42 (1.21x) | 53.61 / 65.61 (1.22x) |

A binaryK split conv 3x3 [8,64,32,32] forward pass: 3.96 / 9.22 ms (2.33x).

So the cost of counting on the M3 Pro's GPU is 2.2-3.0x under RNE and
1.21-1.25x under SR, against Linux's 1.7-3.4x on CUDA. The classification
is integer work of a fixed size per cast, which weighs most beside the
cheapest casts (binaryK's RNE fast path, the fixed-point carrier path) and
least beside SR's draws. The end-of-tile reduction is once per threadgroup
and does not show at these sizes. The numbers are in `mps_backend.py`'s
docstring.

### What Linux is asked to do

1. **`binary_identity.py`** for `common/event_counts.h`,
   `common/gemm_counted.h` and `common/fixedpoint_counted.h`, against
   `127f83d`, flag off and on. Expected: nothing moves; the host token
   streams are identical (above), and only the counted units include the
   three headers.
2. **Decide on `OVERFLOW`/`UNDERFLOW`**: keep the Apple-host `#undef`, or
   rename the two enumerators.
3. `dev/benchmarks/README.md`'s `mps_backend.py` row: add `bench-counted`
   ("each counted op against its twin on MPS").
4. CLAUDE.md's macOS section (untracked, yours): the counted ops run on MPS
   (`KernelSource::GemmCounted`/`GemmFixedPointCounted`, `mps/counted.metal`
   and `mps/fixedpoint_counted.metal`, `gemm.metal` under `MPT_COUNTING`,
   the counts added with 32-bit atomics), and Darwin's `OVERFLOW`/`UNDERFLOW`
   macros, a trap for any new identifier of those names.

### Open

- The worktrees `../mptorch-base`, `../mptorch-wide` and `../mptorch-m4base`
  are still there (M7's report lists them).

### M8's commits

| task | commit | files | shared? |
|---|---|---|---|
| M8 | "MPS: the counted GEMM and conv ops" | the files listed above | yes: the three `common/` headers need **`binary_identity.py`**; `quant_ops.h`, `setup.py` and the three test files are listed above |
| this section | the commit after it | `dev/mac_parallel_report.md` | |

## M9's baseline: flash attention on MPS before AL (at `3937f31`)

SP-AL is not delivered yet, so this is the "before" for AL's MPS round: the
current flash path (`FlashFormats`, FA2's tile loop in Python) against the
standard path on MPS, eager and compiled. No file changed but this report.
The scratch script is the session's `m9/flash_mps.py`.

**Method.** Causal self-attention over `[batch, heads, length, 64]` float32,
in the formats of `dev/benchmarks/timing/attention_ops.py`: both products
`SplitMac(BinaryK(8, 4), BinaryK(14, 8))`, the operands and probabilities
`BinaryK(8, 4)`, the softmax's steps too; RNE, or SR with its random bits.
The flash rows add `FlashFormats(block_q=b, block_kv=b)`, its three slots
unset. Minimum over three rounds of two calls, the variants alternated,
`torch.mps.synchronize()` around each batch. AC power, no thermal warning
before or after.

`tests/test_flash_attention.py` passes on this tree: 57 passed, 28 skipped
(CUDA, float64), 27 of the cases on MPS.

### Eager: ms, and against the standard path in the same mode

| shape | math | std RNE | flash 64 RNE | flash 128 RNE | std SR | flash 64 SR | flash 128 SR |
|---|---|---|---|---|---|---|---|
| 8x8x256 | 0.4 | 9.5 | 7.2 (0.76x) | 7.0 (0.74x) | 38.2 | 25.6 (0.67x) | 28.7 (0.75x) |
| 4x16x1024 | 3.1 | 143.4 | 83.9 (0.58x) | 75.9 (0.53x) | 608.4 | 333.2 (0.55x) | 333.3 (0.55x) |
| 1x16x4096 | 11.6 | 568.6 | 478.6 (0.84x) | 301.7 (0.53x) | 2490.5 | 1485.8 (0.60x) | 1286.6 (0.52x) |

Forward and backward, RNE:

| shape | math | std | flash 64 | flash 128 |
|---|---|---|---|---|
| 8x8x256 | 4.0 | 24.6 | 21.8 (0.89x) | 22.1 (0.90x) |
| 4x16x1024 | 49.4 | 379.1 | 258.5 (0.68x) | 242.1 (0.64x) |
| 1x16x4096 | 181.3 | 1497.6 | 1354.9 (0.90x) | 947.1 (0.63x) |

**On MPS flash is faster than the standard path, not 5-8x slower as on
CUDA.** The likely reason: flash skips the causal tiles above the diagonal,
about half of the simulated GEMMs' work, and on the M3 Pro's GPU those GEMMs
are slow enough to outweigh the loop's launches. The loop's cost still
shows. At 1x16x4096, block 64 runs 2,080 tile steps and block 128 runs 528,
and the two take 478.6 and 301.7 ms. That puts the loop's overhead near
0.11 ms per tile step, about half of the block-64 time (my estimate from
the difference, not a profile). So batching the query tiles (AL's step 2)
should pay on MPS too, most at small blocks and long sequences.

### Compiled (forward, RNE, `torch.compile(fullgraph=True)`)

| shape | path | backend | compile | eager | compiled | words |
|---|---|---|---|---|---|---|
| 8x8x256 | std | inductor | 2.3 s | 9.4 ms | 8.0 ms (1.17x faster) | equal |
| 8x8x256 | flash 64 | inductor | 1.7 s | 7.2 ms | 5.3 ms (1.36x faster) | equal |
| 4x16x1024 | std | inductor | 0.4 s | 143.8 ms | 123.8 ms (1.16x faster) | equal |
| 4x16x1024 | flash 64 | inductor | **fails** | 83.3 ms | | |
| 4x16x1024 | flash 64 | inductor, `max_fusion_size = 2` | 16.5 s | 83.3 ms | 66.7 ms (1.25x faster) | equal |
| 4x16x1024 | flash 64 | aot_eager | 6.8 s | 83.3 ms | 98.2 ms (0.85x) | equal |

**Inductor fails on the flash loop at 4x16x1024.** It fuses so many inputs
into one generated Metal kernel that the kernel needs more than Metal's 31
buffer arguments, and the Metal compiler refuses it: "no 'buffer' resource
location available for 'in_ptr8'", and the same for other inputs.
`torch._inductor.config.max_fusion_size = 8` still fails; `2` compiles and
gains 1.25x. That is a torch bug, Inductor's MPS backend not capping a
kernel's buffers. AL's step 1, if it compiles flash, needs a workaround on
MPS, or a test that the fallback is eager. Every compiled result equals
eager's word for word. The gains are small because each mptorch op stays
one opaque dispatch: Inductor fuses only the torch operations between them.

### For Linux

- **`dev/benchmarks/timing/attention_ops.py --device mps` times the
  enqueueing, not the work.** Its `_timer` synchronizes CUDA only, the
  pointwise tool's bug of R2.0. At 4x8x256 it reports 0.01 / 0.09 / 0.09 ms
  for math / rne / sr, which take 0.45 / 4.68 / 19.12 ms synchronized. It
  also has no MPS shapes (it falls back to the CPU's).
- If AL's tools time flash on MPS, the numbers above are the base.

## M9: AL step 2 on MPS (at `99232dc`)

The batched path gives the loop's words on MPS in every case. It is faster
than the loop where the tiles are many or the products cheap, and 4-6.5%
slower at 128 x 128 tiles with the baseline's simulated products. No file
changed but this report.

The flag-on build (`MPTORCH_ODD_ACCUMULATED=1 MPTORCH_COUNTING=1`) rebuilt
the counted units for `36f0c44`'s rename (`OVERFLOWED`, `UNDERFLOWED`) under
Apple clang without the `#undef`, and their tests are in the suite below.

### 1. Tests on MPS

| command | result |
|---|---|
| `pytest tests/test_flash_attention.py` | 91 passed, 45 skipped (CUDA); 43 cases on MPS, among them the 16 of `test_the_batched_path_is_the_loop` (four recipes x four masks, both geometries in each), all passed |
| `pytest tests/test_attention.py` | 219 passed, 118 skipped (CUDA, and float64 on MPS) |
| `pytest tests/test_simulate_events.py` | 21 passed, 7 skipped (CUDA) |
| `pytest tests/test_profile.py` | 45 passed |
| `pytest tests/` | 28,327 passed, 17,120 skipped, 0 failed (8 min) |

`test_what_the_batched_path_takes` has no device parameter (it runs once).
No batched case differed, so there is no first-differing tensor to report.

### 2. The baseline again

The baseline's script, shapes, formats and method (section "M9's
baseline"), with a column per block size for the loop forced
(`mock.patch.object(attention, "_batchable", lambda f: (False, False))`).
Under RNE these formats batch both passes (`_batchable` returns `(True,
True)`); under SR they batch neither, so the SR columns are the control. AC
power, no thermal warning before or after. ms, and against the standard
path in the same mode:

Forward:

| shape | std RNE | flash 64 | loop 64 | flash 128 | loop 128 | std SR | flash 64 SR | flash 128 SR |
|---|---|---|---|---|---|---|---|---|
| 8x8x256 | 9.4 | 6.7 (0.71x) | 7.3 (0.78x) | 7.2 (0.77x) | 7.0 (0.75x) | 37.7 | 25.3 (0.67x) | 28.3 (0.75x) |
| 4x16x1024 | 141.7 | 82.8 (0.58x) | 83.1 (0.59x) | 80.2 (0.57x) | 75.8 (0.53x) | 601.1 | 330.8 (0.55x) | 330.7 (0.55x) |
| 1x16x4096 | 563.2 | 312.1 (0.55x) | 481.9 (0.86x) | 295.3 (0.52x) | 299.8 (0.53x) | 2470.3 | 1480.0 (0.60x) | 1274.9 (0.52x) |

Forward and backward, RNE:

| shape | std | flash 64 | loop 64 | flash 128 | loop 128 |
|---|---|---|---|---|---|
| 8x8x256 | 24.6 | 20.9 (0.85x) | 21.4 (0.87x) | 22.6 (0.92x) | 22.2 (0.90x) |
| 4x16x1024 | 375.4 | 260.9 (0.70x) | 259.6 (0.69x) | 255.4 (0.68x) | 240.2 (0.64x) |
| 1x16x4096 | 1483.6 | 998.1 (0.67x) | 1365.3 (0.92x) | 944.5 (0.64x) | 944.3 (0.64x) |

**The control holds.** The SR columns are the baseline's within 1-2%
(330.8 against 333.2 ms, 1480.0 against 1485.8, 1274.9 against 1286.6), and
the forced loop is the baseline's flash within 1%.

**Where batching pays.** At 1x16x4096 with 64 x 64 tiles (2,080 tile
pairs), the batched forward takes 312.1 ms against the loop's 481.9 (1.54x),
and forward and backward 998.1 against 1365.3 (1.37x). The 170 ms saved
is most of the loop overhead the baseline estimated there (about 230 ms, at
0.11 ms per tile step). With
128 x 128 tiles, or at L = 1,024, there are too few tile pairs for it to
show.

**Where it costs.** With 128 x 128 tiles the batched path is slower than
the loop. Rechecked over seven rounds, batched against loop:

| shape | tiles | fwd | fwd+bwd |
|---|---|---|---|
| 4x16x1024 | 64 | 82.9 / 82.9 (1.000) | 261.1 / 255.7 (1.021) |
| 4x16x1024 | 128 | 80.2 / 75.4 (1.065) | 255.7 / 240.4 (1.064) |
| 2x16x2048 | 64 | 159.2 / 185.0 (0.860) | 503.4 / 554.6 (0.908) |
| 2x16x2048 | 128 | 149.4 / 143.3 (1.042) | 477.9 / 458.2 (1.043) |

A lead, from counting the dispatched ops (`TorchDispatchMode`) of one
forward at 4x16x1024, 128 x 128 tiles. The batched path makes 521 ops
against the loop's 1,332 (24 GEMMs against 108), but it moves more data:
`copy_` writes 27.9M elements (the loop's none), `clone` 23.1M against
12.6M, `add_` 8.4M against none, and its `slice`s and `transpose`s cover
72.3M and 33.6M elements against 12.6M and 4.2M (views, which the copies
then read strided). With 36 tile pairs there are only about 3 ms of
launches to save at the baseline's 0.11 ms per step, so the extra copies
evidently cost more than that here (I did not time them apart). Writing the running state in place, or
keeping the suffix of tiles contiguous, might remove the difference. I did
not try: `attention.py` is Linux's.

### 3. The tool

`flash_attention.py --device mps --recipes carrier torch --lengths 512 1024
2048` (`[1, 8, L, 64]`, causal, 128 x 128 tiles, the tool's own method), ms;
no `[differs]` mark anywhere:

| recipe | L | fwd: standard / loop / flash | fwd+bwd: standard / loop / flash |
|---|---|---|---|
| carrier | 512 | 2.49 / 2.92 / 2.34 | 5.52 / 7.20 / 5.94 |
| carrier | 1024 | 8.90 / 8.89 / 6.10 | 20.19 / 22.47 / 17.93 |
| carrier | 2048 | 34.23 / 28.92 / 20.93 | 79.44 / 79.39 / 59.73 |
| torch | 512 | 1.06 / 1.64 / 1.62 | 1.98 / 3.39 / 3.44 |
| torch | 1024 | 3.58 / 3.90 / 3.86 | 6.99 / 10.38 / 10.41 |
| torch | 2048 | 14.28 / 12.38 / 12.33 | 26.58 / 36.32 / 36.41 |

With the carrier recipe, whose products are cheap (`FusedMac(None)`), the
batched path runs the forward 1.25-1.46x faster than the loop, and from
L = 1,024 faster than the standard path, as on CUDA. The `torch` recipe
keeps the loop, as designed, so its two flash columns agree.

### 4. Compile, short

Inductor at 4x16x1024, flash 64 x 64, forward, RNE, on the batched path as
shipped (eager 82.7 ms):

| config | result |
|---|---|
| default | fails after 6.5 s: "number of constant buffers exceeds maximum supported (31)" |
| `max_fusion_size = 2` | compiles in 5.0 s (the loop's took 16.5); 67.8 ms (1.22x faster); words equal eager's |

So the batched graph still exceeds Metal's 31 buffers under Inductor's
default fusion. It compiles three times faster than the loop's, and runs as
fast as the compiled loop (66.7 ms in the baseline).

## M9 follow-up: `580c512` (fewer copies of the running state) on MPS

`test_flash_attention.py` and `test_attention.py`: 310 passed, 163 skipped
(CUDA, float64); the 16 batched cases on MPS still equal the loop. AC
power, no thermal warning.

**The copies fell as the commit says** (one forward at 4x16x1024, 128 x 128
tiles): `copy_` 4.3M elements, against 27.9M before. **The 128-tile gap
barely moved.** Batched against loop, seven rounds, the report's method:

| shape | tiles | fwd: before, now | fwd+bwd: before, now |
|---|---|---|---|
| 4x16x1024 | 64 | 1.000, 0.987 | 1.021, 1.010 |
| 4x16x1024 | 128 | 1.065, 1.053 | 1.064, 1.056 |
| 2x16x2048 | 64 | 0.860, 0.836 | 0.908, 0.898 |
| 2x16x2048 | 128 | 1.042, 1.040 | 1.043, 1.033 |
| 1x16x4096 | 64 | 0.648, 0.636 | 0.731, 0.729 |
| 1x16x4096 | 128 | 0.985, 0.945 | 1.000, 0.974 |

(The "before" ratios at 4096 are from the M9 table, three rounds; the rest
from its seven-round recheck.)

**Both paths are GPU-bound here, with no host syncs.** At 4x16x1024 with
128 x 128 tiles, neither path dispatches `_local_scalar_dense`. The host
finishes enqueuing in 1.5 ms (batched) and 3.3 ms (loop) of the 80.4 and
76.0 ms, so the 4.4 ms difference is GPU work.

**The remaining extra work is GEMM operand copies.** Grouped by caller
(`TorchDispatchMode`, the copying ops' elements), batched against loop:

| op, caller | batched | loop |
|---|---|---|
| `clone`, `ops/_spec.py:213 _packed` (a GEMM operand made contiguous) | 14.68M | 0 |
| `clone`, `ops/_checks.py:243 _quantizer_operand` | 8.39M | 12.58M |
| `add_` on a non-contiguous view, `attention.py:885 _batched_scores` (the causal bias) | 7.34M | 0 |
| `copy_`, `attention.py:898 _retire` | 3.78M | 0 |

So the batched path hands its GEMMs operands that are neither contiguous
nor a transposed contiguous tensor, which `_packed` copies; in the loop no
GEMM operand needed that copy. Laying the
batched operands out so `_packed` passes them through (contiguous, or the
transpose of a contiguous tensor) would likely close most of the gap. The
non-contiguous causal-bias `add_` is the next largest. I did not try
either, since `attention.py` is Linux's.

## AQ re-measured at HEAD: closed, nothing to build (at `f714e5e`)

Phase AQ was M2 above, fixed in `fc2caad`. I re-measured it at the current
tree with M2's own per-pass script (the session's `m2/sq_parts.py`): 4096 x
4096, `BinaryK(8, 4)` under `SAT_FINITE`, RNE, AC power. Each figure is the
minimum over rounds of the mean of 20 calls, in ms.

| float32 | fused call | amax + scale | fused kernel | four passes | speedup |
|---|---|---|---|---|---|
| per tensor | 2.08 | 1.05 | 1.02 | 3.99 | 1.92x |
| per row | 2.06 | 1.05 | 1.03 | 5.80 | 2.81x |
| blocks of 32 | 2.40 | 1.33 | 1.04 | 6.05 | **2.52x** (0.97x before `fc2caad`) |

bfloat16 per tensor: 1.35 against 5.03 (3.7x).

**The kernel has nothing left to gain.** The fused kernel takes 1.02-1.04 ms
at every granularity, a plain quantizer pass's time (0.98 ms). That is one
read and one write of 64 MB, about 129 GB/s, near the M3 Pro's memory
bandwidth.

**What remains is the amax pass**, torch's, about half the call (1.05-1.33
ms). Folding it into the kernel (a block's maximum taken inside the
threadgroup) could take a block-of-32 call from 2.4 to about 1.1 ms. I do
not propose it:
- the scale rule (the largest magnitude, the format's maximum, scale 1 for
  a bad amax by `torch.where`) is Python in `mptorch/quant/scaled.py`, so
  that every backend agrees, nothing syncs and it compiles; moving it into
  kernels is CPU, CUDA, MPS and Meta work, with new word-for-word gates;
- `f714e5e` measures all quantizers at 1-3% of GPT-2 small's GPU time, so
  halving `ScaledQuant` moves a model by well under 2%.

**For Linux:** mark AQ done in the live `dev/continuation_plan.md` (done as
M2 in `fc2caad`; 0.97x to 2.52x by blocks of 32 in float32). If the amax
fusion is worth keeping as an idea, it belongs with the cross-backend work,
with the numbers above.

## M10a: `torch._int_mm` on the Mac (at `4d78016`)

The handoff's probe (section 14.1), saved unchanged as the session's
`m10a/probe.py`, under `uv run --no-sync python`. The machine is the M3 Pro
on AC power, torch 2.15.0, torch's default 5 threads. No file changed but
this report.

### The probe's output

```
cpu exact: True torch.int32
  cpu (1, 64, 64) ok
  cpu (16, 64, 64) ok
  cpu (17, 64, 64) ok
  cpu (24, 64, 64) ok
  cpu (64, 60, 64) ok
  cpu (64, 64, 60) ok
  cpu transposed weight (Linear layout): True
mps exact: True torch.int32
  mps (1, 64, 64) ok
  mps (16, 64, 64) ok
  mps (17, 64, 64) ok
  mps (24, 64, 64) ok
  mps (64, 60, 64) ok
  mps (64, 64, 60) ok
  mps transposed weight (Linear layout): True
words equal, fixedpoint_matmul, wide (32, 14): True
words equal, fixedpoint_matmul, (25, 14): True
cpu 1 thr, fixedpoint_matmul, wide (32, 14): 740.1 ps/step
cpu 1 thr, fixedpoint_matmul, (25, 14): 745.9 ps/step
cpu 1 thr, torch._int_mm: 512.6 ps/step
cpu 5 thr, fixedpoint_matmul, wide (32, 14): 169.9 ps/step
cpu 5 thr, fixedpoint_matmul, (25, 14): 169.9 ps/step
cpu 5 thr, torch._int_mm: 118.4 ps/step
mps, fixedpoint_matmul, wide (32, 14): refused (custom_matmul_fixedpoint: the accumulate format has 31 magnitude bits, more than)
mps, fixedpoint_matmul, (25, 14): 24.37 ps/step
mps, torch._int_mm: 1.928 ps/step
```

### What it says

- **MPS: the route can cover it.** `_int_mm` runs, is exact (the K = 65,536
  check), takes every shape tried (M = 1, 16, 17, 24; K and N of 60; none of
  CUDA's multiples), and takes the transposed weight. It runs at about 2 ps a
  step (1.9-3.9 across rounds), against 24.4 for the fixed-point GEMM on MPS:
  about 10x. The wide row refuses, as expected.
- **The arm64 CPU: `_int_mm` is a slow fallback.** It is exact and takes
  every shape, but costs 513 ps a step on one thread, against 3.7 on the
  Ryzen, about 140x slower. That is only 1.45x faster than the fixed-point
  GEMM (740-746 ps), and 1.43x on 5 threads. This build has no oneDNN
  (`torch.backends.mkldnn.is_available()` is False); its BLAS is Accelerate.
  So the route needs a platform gate, or another engine on arm64.

### Another exact engine on arm64: torch's float matmul

Any exact engine gives the route's words, and on this machine torch's float
matmuls (Accelerate) are far faster than `_int_mm`. An 8-bit product is at
most 2^14 in magnitude, so every partial sum is an integer of magnitude at
most K x 2^14. That makes a float32 sum exact in any order for K <= 1,024
(2^24), and a float64 one for K <= 2^39. The session's `m10a/float_route.py`
and `m10a/bound.py`, 512^3, ps a step, the median of 7-9 calls:

| engine | CPU 1 thread | CPU 5 | CPU 11 | MPS |
|---|---|---|---|---|
| `torch._int_mm` | 510 | 117 | 95 | 1.9-3.9 |
| float32 matmul (K <= 1,024) | 1.29 | 1.28 | 1.28 | 2.0-2.4 |
| int8 -> float32, matmul, -> int32 | 1.59 | 1.88 | 2.44 | |
| float64 matmul | 4.99 | 4.95 | 4.94 | (no float64) |

Every row's words equal `_int_mm`'s. One 26 ps reading of float64 at 5
threads did not repeat in five later runs. At the bound, `bound.py` held
float32 (CPU and MPS) and float64 (CPU) to an int64 reference at K = 1,024
in three cases, all exact:
- every product 2^14, so the sum is exactly 2^24;
- large same-sign products;
- random operands.

A K = 65,536 case of equal products was exact too, but it is no test of
float32 past the bound, since its sums are multiples of 2^14.

So on the arm64 CPU, float64 is an exact engine for any K the route will see,
about 150x the fixed-point GEMM on one thread and 100x `_int_mm`. float32
gains another 4x for K <= 1,024. The fixed-point operands are already floats
on their grid, so the route could scale them by 2^fl and multiply in float
without any int8 conversion.

### For Linux (AN-b's device gate)

1. MPS: `_int_mm` qualifies (exact, every shape, about 10x the fixed-point
   GEMM); so does float32 matmul under the K bound.
2. arm64 CPU: not `_int_mm` (a scalar-speed fallback in this build). The
   float64 matmul, or float32 under K <= 1,024, is exact and about 150-580x
   the fixed-point GEMM. Whether the route takes a float engine everywhere,
   or `_int_mm` on x86 and CUDA and float elsewhere, is Linux's call.
3. The Ryzen's float matmuls were not measured here; the comparison there
   decides whether one float engine could serve every backend. On CUDA a
   float32 matmul is not exact when TF32 is allowed
   (`torch.backends.cuda.matmul.allow_tf32`), so a float engine there would
   need it off, or float64.

## M10b: the exact int8 route on the Mac (at `53610bc`)

The route gives the kernel's words on the M3 Pro's CPU and GPU, its NaN case
included. On the CPU its float engine runs a 512^3 call 250x faster than the
kernel on one thread. On MPS the route wins from about 128^3, well below its
threshold of 2^25 steps. No file changed but this report.

### Tests

| command | result |
|---|---|
| `pytest tests/test_fixedpoint_exact_route.py` | 101 passed, 33 skipped (CUDA, and four wide cases MPS refuses by design); `cpu-int`, `cpu-float` and `mps` all ran |
| `pytest tests/test_fixedpoint_layers.py` | 51 passed, 16 skipped (CUDA) |
| `pytest tests/test_fixedpoint_mps.py` | 257 passed |
| `pytest tests/` | 28,430 passed, 17,155 skipped, 0 failed |

**The NaN test passes on MPS** (`test_nan_operands_reach_the_same_outputs[mps]`).
So MPS's integer-accumulator NaN is the canonical quiet `0x7fc00000` the
route writes (`FixedPointIntMac::value` builds `INF_BITS | TIE` on Metal
too). MPS needs no NaN of its own.

### Timing (`fixedpoint_gemm.py exact`, the kernel and the route alternated)

AC power, no thermal warning; ps a mac step, the kernel -> the route.
`(25, 14)` holds sums of up to 1,023 terms, so at n = 1,024 its route takes
the kernel (MPS's equal 1,024 row).

| | 256^3 | 512^3 |
|---|---|---|
| CPU 1 thread, float engine (the default here), (25, 14) | 765 -> 4.9 | 767 -> 3.1 |
| same, wide (32, 14) | 763 -> 5.4 | 771 -> 2.9 |
| CPU 1 thread, int engine forced, (25, 14) | 760 -> 15.8 | 761 -> 15.2 |
| CPU 11 threads, float engine, (25, 14) | 157 -> 14.9 | 146 -> 3.5 |
| same, wide (32, 14) | 156 -> 18.1 | 149 -> 3.4 |
| CPU 11 threads, int engine forced, (25, 14) | 157 -> 28.7 | 136 -> 6.6 |
| MPS, `--all-sizes`, (25, 14) | 46.1 -> 25.4 | 23.8 -> 7.1 |

At 1,024^3 on MPS: 22.3 -> 22.3 (the kernel both ways, as above).

**The float engine is the right default on this CPU.** It is 2-5x the
forced `_int_mm`. At 11 threads the route is slower than at one below
512^3 (14.9 against 4.9 ps at 256^3), likely because its elementwise
steps then spread over the efficiency cores too (torch's default here is 5
threads; I did not profile it).

**A correction to M10a: arm64's `_int_mm` depends on B's layout.** M10a
timed it with a row-major B (`[K, N]` contiguous): 523 ps a step on one
thread. With B the transpose of a contiguous `[N, K]` weight, which is the
Linear layout and what the tool's rows pass (`trans_b=True`), it is 37x
faster:

| `torch._int_mm(a, b)`, 512^3 | 1 thread | 5 | 11 |
|---|---|---|---|
| b row-major `[K, N]` | 523 | 117 | 97 |
| b = `w.t()`, w `[N, K]` contiguous | 14.2 | 4.4 | 14.5 |

Blocking A's rows changes nothing (523-548 ps for blocks of 1 to 512 rows),
which suggests a fallback loop that wants both operands contiguous along K. The
float engine is still faster here (3.1 ps on one thread), so the gate
stands. But a pass whose B is row-major (a `matmul` pass, `trans_b=False`)
would hit the 523 ps path if the int engine were taken on such a CPU, and
making B K-contiguous first would avoid it.

### The thresholds (`--all-sizes`, ms a call, the kernel -> the route)

CPU, the float engine, (25, 14):

| n | 32 | 48 | 64 | 96 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|
| steps | 2^15 | 2^16.8 | 2^18 | 2^19.8 | 2^21 | 2^22.8 | 2^24 |
| 1 thread | 0.028 -> 0.015 | 0.093 -> 0.017 | 0.198 -> 0.018 | 0.663 -> 0.024 | 1.556 -> 0.033 | 5.41 -> 0.054 | 12.96 -> 0.083 |
| 11 threads | 0.028 -> 0.015 | 0.069 -> 0.019 | 0.083 -> 0.020 | 0.195 -> 0.030 | 0.423 -> 0.033 | 1.17 -> 0.17 | 2.62 -> 0.22 |

The route wins at every size measured, from 32^3 (2^15 steps, 1.9x) up. The
CPU's threshold of 2^20 is safe, but it gives up 1.9-27x between 2^15 and
2^20 on this machine. Nothing below 32^3 was measured.

MPS, (25, 14):

| n | 64 | 128 | 192 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|
| steps | 2^18 | 2^21 | 2^22.8 | 2^24 | 2^25.8 | 2^27 |
| ms | 0.230 -> 0.322 | 0.400 -> 0.363 | 0.353 -> 0.274 | 0.581 -> 0.352 | 1.48 -> 0.558 | 3.20 -> 1.01 |
| route / kernel | 1.40 | 0.91 | 0.78 | 0.61 | 0.38 | 0.31 |

The route loses at 64^3 and breaks even near 128^3 (2^21). It wins 1.3x at
192^3 and 1.65x at 256^3, both below today's threshold of 2^25 (322^3).
The small calls are launch-bound on both paths (the kernel's 128^3 took
longer than its 192^3), so the crossing is not sharp.

### For Linux

1. **MPS's NaN:** nothing to do; the canonical NaN is MPS's too.
2. **`_MIN_STEPS`:** on the M3 Pro the route wins from 2^15 steps on the
   CPU (the smallest size measured) and from about 2^21 on MPS. Suggested:
   MPS 2^22, which keeps a margin above the break-even and takes the 1.3-1.65x
   at 192^3-256^3. The CPU's 2^20 could drop to 2^15 on this machine; whether
   one CPU value suits the Ryzen too is Linux's measurement.
3. **arm64's `_int_mm` and layout:** if the int engine is ever taken on an
   arm64 CPU without oneDNN, give it a K-contiguous B (the Linear layout).
   With a row-major B it runs 37x slower.

## M11: the NEON word tiles (at `8ef5cdd`)

The NEON word tiles run on the M3 Pro and give the scalar loop's words on
every binary32 input, in all five modes, with both carriers. That is their
first run anywhere. Every vector loop in both tile files vectorizes under
Apple clang. At 512^3 the tiles take the split mac 2.7-5.7x faster than the
scalar loop on one thread (3.7-9.6x with the odd carrier). The Ryzen's
AVX-512 tiles are 1.4-2.3x faster still. The row loop that the RNE tiles use
on NEON doesn't help these tiles, so there is no layout to propose. No file
changed but this report.

Machine: the M3 Pro, on AC power, with no thermal or performance warning
recorded before or after any timing. Built flag-on
(`MPTORCH_ODD_ACCUMULATED=1 MPTORCH_COUNTING=1`) with Apple clang 21.0.0.

### Tests

| command | result |
|---|---|
| `pytest tests/test_cpu_simd.py` | 1 passed: the settings `0` (the scalar loop), the default (NEON) and `avx2` (NEON again on arm64) agree word for word, the five modes' cases included |
| `pytest tests/test_cpu_simd.py tests/test_carrier_rounding.py tests/test_accumulators.py` | 1,789 passed, 1,521 skipped (CUDA and MPS-float64 cases) |
| `pytest tests/` | 28,430 passed, 17,155 skipped, 0 failed (6:25); the counts are M10b's, since `test_cpu_simd.py`'s new cases are inside its one test |
| `bash dev/benchmarks/check_native.sh` | `OK    sweeps/cpu_word_cast_sweep.cpp`, `OK    sweeps/cpu_word_tile_sweep.cpp`, `all native tools build` |

### The sweeps under Apple clang

Built as section 4.3 builds the others, with Homebrew's libomp for the
standalone binaries (the build lines in the files' headers, `g++` and
`-fopenmp` replaced):

```
FLAGS="-O3 -std=c++20 -ffp-contract=off -DMPTORCH_FAST_CAST=1 -Xpreprocessor -fopenmp \
    -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp -I mptorch/csrc"
clang++ $FLAGS dev/benchmarks/sweeps/cpu_word_cast_sweep.cpp -o cpu_word_cast_sweep
clang++ $FLAGS dev/benchmarks/sweeps/cpu_word_tile_sweep.cpp \
    mptorch/csrc/cpu/gemm_word_tile.cpp mptorch/csrc/cpu/gemm_word_odd_tile.cpp -o cpu_word_tile_sweep
```

Both compile with no warning.

| run | output | time (11 threads) |
|---|---|---|
| `cpu_word_cast_sweep` | 0 mismatches in each of the 65 comparisons (13 formats x 5 modes, all 2^32 words); `ok: 0 mismatches in all` | 3:00 |
| `cpu_word_cast_sweep control` | a mismatch count in every comparison (166,723,584 to 2,279,603,970 in the first format); `control ok: 0 of the comparisons found no mismatch` | 2:52 |
| `cpu_word_tile_sweep` | `the neon word tiles`; 0 mismatches in each of the 60 comparisons (6 formats x 5 modes, the default and the odd carriers); `ok: 0 mismatches in all` | 13:18 |
| `cpu_word_tile_sweep control` | a mismatch count in every comparison (453,690,833 to 12,834,374,784); `control ok: 0 of the comparisons found no mismatch` | 14:05 |
| `MPTORCH_CPU_SIMD=0 cpu_word_tile_sweep` | `no word tiles on this CPU or under MPTORCH_CPU_SIMD=0: nothing to check` | |

The odd tiles' FPCR switches therefore give the scalar odd step's words under
clang too.

### Vectorization

I recompiled both tile files with the extension's own flags (`build.ninja`'s
`cflags` and `post_cflags`) plus clang's vectorizer remarks
(`-Rpass=loop-vectorize -Rpass-missed=loop-vectorize
-Rpass-analysis=loop-vectorize`). **Every `#pragma omp simd` loop
vectorized, at width 4, in all 15 functions of each file** (5 modes x split,
product-only and fused), in both copies of the step code (the fixed 32^3
tile and the run-time sizes):

| file | loop | vectorized |
|---|---|---|
| `gemm_word_tile.cpp` | the products (line 54) | 30 of 30 |
| | the split tile's two casts (65, 68), the product-only cast (74), the fused cast (80) | 10 of 10 each |
| | the NaN check (105) | 15 of 15 |
| `gemm_word_odd_tile.cpp` | the casts (187 fused, 194 product-only, 201 and 205 split) | 10 of 10 each |
| | the odd sums' down and up passes (116, 120) | 10 of 10 each |
| | the odd products' down and up passes (86, 99) | every instance: see below |
| | the NaN check (48) | 15 of 15 |

In the odd products' fixed 32^3 copy, clang fully unrolled each row's
32-lane loop before the loop vectorizer saw it (`FullyUnrolled` in
`-fsave-optimization-record`). The SLP vectorizer then packed each unrolled
row into eight four-lane stores. That is 40 store groups per tile kind and
pass, 5 functions x 8 (lines 90 and 92 the down pass, 105 the up pass). The
loop vectorizer took the run-time-size copy, and both copies of the fused
tiles' up pass.

The only loops not vectorized are the outer row loops (`for i`, lines 49,
83 and 96: "the cost-model indicates that vectorization is not
beneficial"), and their inner loops vectorized. So the RU and RD tiles
vectorize under clang too. On x86 GCC left them scalar in their first
form, which computed both helpers on every lane; the shipped
direction-selected helper (`absolute_directed`) replaced it.

In the built objects (`objdump -d`), 28-40% of each tile function's
instructions are NEON ALU operations (`cmhi.4s`, `cmeq.4s`, `bit.16b`,
`and.16b`, `ushr.4s`, `fmul.4s`, ...). The rest are loads, stores, loop
control and the scalar remainders of the run-time-size copy.

### Timing (the GEMM, the tiles against `MPTORCH_CPU_SIMD=0`)

The brief's calls at 512^3 on N(0, 1) inputs:
- split: `binaryK_matmul(a, b, mul_K=8, mul_P=4, acc_K=16, acc_P=11,
  rounding_mode=m)`;
- fused: `binaryK_matmul_fma(a, b, fma_K=16, fma_P=11, rounding_mode=m)`;
- each also with `carrier_rounding=RoundMode.RO`.

`MPTORCH_CPU_SIMD` is read at load, so each setting ran in its own process,
alternated with the other in three rounds (the order reversed every round).
Each process made one warm-up call and timed two more; the table gives the
minimum over the rounds. RNE is the reference row: the RNE tiles under the
default setting, and the scalar loop under `0`. The script is
a scratch script, `word_timing.py`.

ps a step, scalar -> tiles (speedup):

| | 1 thread | 11 threads |
|---|---|---|
| split RNE | 2,967 -> 645 (4.6x) | 519 -> 116 (4.5x) |
| split RNA | 6,186 -> 2,267 (2.7x) | 869 -> 375 (2.3x) |
| split RU | 11,281 -> 2,918 (3.9x) | 1,381 -> 494 (2.8x) |
| split RD | 10,796 -> 2,938 (3.7x) | 1,333 -> 495 (2.7x) |
| split RZ | 13,049 -> 2,298 (5.7x) | 1,561 -> 391 (4.0x) |
| split RO | 5,770 -> 2,022 (2.9x) | 824 -> 351 (2.3x) |
| fused RNE | 1,351 -> 323 (4.2x) | 237 -> 63 (3.8x) |
| fused RNA | 2,855 -> 1,111 (2.6x) | 373 -> 192 (1.9x) |
| fused RU | 3,403 -> 1,400 (2.4x) | 447 -> 241 (1.9x) |
| fused RD | 3,251 -> 1,420 (2.3x) | 441 -> 241 (1.8x) |
| fused RZ | 3,428 -> 1,148 (3.0x) | 462 -> 210 (2.2x) |
| fused RO | 2,855 -> 1,014 (2.8x) | 376 -> 179 (2.1x) |
| odd split RNE | 7,042 -> 893 (7.9x) | 1,168 -> 157 (7.4x) |
| odd split RNA | 9,154 -> 2,442 (3.7x) | 1,335 -> 390 (3.4x) |
| odd split RU | 20,858 -> 3,044 (6.9x) | 2,538 -> 506 (5.0x) |
| odd split RD | 20,738 -> 3,096 (6.7x) | 2,489 -> 509 (4.9x) |
| odd split RZ | 23,990 -> 2,511 (9.6x) | 2,838 -> 402 (7.1x) |
| odd split RO | 9,721 -> 2,246 (4.3x) | 1,452 -> 371 (3.9x) |
| odd fused RNE | 3,041 -> 445 (6.8x) | 516 -> 81 (6.4x) |
| odd fused RNA | 4,563 -> 1,220 (3.7x) | 677 -> 218 (3.1x) |
| odd fused RU | 4,923 -> 1,536 (3.2x) | 741 -> 259 (2.9x) |
| odd fused RD | 4,809 -> 1,557 (3.1x) | 730 -> 254 (2.9x) |
| odd fused RZ | 5,229 -> 1,256 (4.2x) | 774 -> 214 (3.6x) |
| odd fused RO | 4,579 -> 1,123 (4.1x) | 718 -> 193 (3.7x) |

Against Linux's Ryzen numbers (1 thread, scalar -> tiles):

| | Ryzen (AVX-512) | M3 Pro (NEON) |
|---|---|---|
| split RZ | 11,550 -> 1,458 (7.9x) | 13,049 -> 2,298 (5.7x) |
| split RZ, odd carrier | 21,827 -> 1,626 (13.4x) | 23,990 -> 2,511 (9.6x) |
| split RU and RD | about 9,990 -> 2,050 (4.9x) | 10,796-11,281 -> 2,918-2,938 (3.7-3.9x) |
| split RNA and RO | 3,390-3,770 -> 980-1,030 | 5,770-6,186 -> 2,022-2,267 (2.7-2.9x) |
| the fused macs | 3-5x | 2.3-3.0x (3.1-4.2x odd) |

- **The scalar loop costs the same on both machines** under RZ, RU and RD
  (within 13%). The NEON tiles are 1.4-2.3x slower than AVX-512's, with 4
  lanes against 16.
- **The split word tiles cost 3.1-4.6x the RNE tile** on the M3 Pro (645
  ps), against about 3.3x for RZ on the Ryzen.
- **RU and RD are the slowest tiles** here as there: their direction-selected
  helper costs 27% over RZ's single helper (2,918 against 2,298).
- On 11 threads the speedups are smaller than on one (1.8-7.1x). In these
  five modes the scalar loop scales 7.0-8.5x from 1 to 11 threads, and the
  tiles 5.5-6.3x. A guess I didn't profile: the six efficiency cores keep up
  better with branchy scalar code than with vector code.

### NEON layout: the row loop, measured, nothing to propose

On NEON the RNE tiles run their product-only and fused tiles as a row loop
(`gemm_rne_tile.cpp`'s `tile_rows`), because whole-tile passes cost the M3
Pro 2-4% there. The word tiles use passes on every ISA. So I tried the row
loop over the word casts in a scratch copy of `csrc`. It is
`gemm_rne_tile.cpp`'s `tile_rows` with `cast_binaryK_word<RM>` in place of
`cast_binaryK_rne_fast`, chosen by an environment variable read at load.
Two variants:
- the row loop for the product-only and fused tiles, as the RNE tiles do;
- the row loop for all three.

I timed both with `dev/benchmarks/timing/cpu_gemm_ab.cpp`, compiled against
the shipped tree and the scratch tree. A scratch copy of the tool runs RNE,
RNA, RU and RZ in place of its RNE, RU and SR. The two binaries were
alternated in five rounds of five reps, at 256^3 and a ragged 100^3, on 1
and 11 threads. Every output hash was the same across the variants and the
scalar loop.

Variant / passes, ns a mac:

| | product-only and fused rows | all rows |
|---|---|---|
| split, 1 thread | 0.94-1.03 (unchanged code: see below) | **1.06-1.24** |
| split, 11 threads | 0.95-1.02 (unchanged code) | **1.05-1.26** |
| product-only, 1 thread | 0.98-1.05 | 0.98-1.05 |
| fused, 1 thread | 0.96-1.02 | 0.97-1.03 |
| product-only and fused, 11 threads | 0.94-1.05 | 0.95-1.06 |

- **For the split tiles the passes win, by 6-26%.** The word casts are
  long chains, and in one loop per row the product's cast feeds the sum's
  (the reason `gemm_rne_tile.cpp` gives for its passes).
- **For the product-only and fused tiles there is no preference.** The
  effects are mixed in sign, within 6% either way, and that is this A/B's
  resolution. The first variant leaves the split tiles' code unchanged, yet
  their times moved 0.94-1.03x between the two binaries, and the RNE rows
  (whose code no variant touches) moved 0.99-1.03x on one thread.

So the shipped layout stands on NEON, and there is no hunk for Linux.

### Open

- AS step 3, the elementwise quantizers' word loop: their timing, when it
  lands.
- M12 waits on SP-AT.

## M11 step 3 and M12: the NEON quantizer loops, the narrow tiles, the exact route on MPS (at `a507483`)

Linux's code for both tasks runs on the M3 Pro and gives the scalar code's
words: the elementwise quantizer's six NEON loops over all 2^32 binary32
inputs, the RNE and odd tiles' narrow path, and the word tiles' hand-back.
The six loops all vectorize at width 4, and on one thread they take E4M3
1.4-4.9x faster than the scalar casts. The narrow path fixes qsum's odd
chain: 77.7 -> 16.9 ns a step (4.6x) against `daa9304`.

**NEON wants other cut-offs in two of the four tile files**, and a hunk for
Linux follows:
- the word tiles: 4 outputs, not 8;
- the odd tiles: 40 outputs, not 4. Their passes set the directed roundings
  through FPCR, at 11-13 ns a write, a fixed cost per K-step. The hunk also
  pads the narrow path to a multiple of four lanes, because clang leaves 5-7,
  9-11 and 13-15 lanes scalar.

With the hunk, odd tiles of 4 to 39 outputs run 1.1-4.4x faster, and the odd
palette edge 1.19-1.21x faster. The hunk gives the scalar loop's words on
7,752 calls, and the odd tile sweep and the tests pass on it. On MPS the exact
route's new threshold, 2^22, sits at the crossing. No file changed but this
report.

Machine: the M3 Pro, on AC power, with no thermal or performance warning
recorded before or after any timing. Built flag-on
(`MPTORCH_ODD_ACCUMULATED=1 MPTORCH_COUNTING=1`) with Apple clang 21.0.0.

### Tests

| command | result |
|---|---|
| `pytest tests/test_cpu_simd.py tests/test_carrier_rounding.py tests/test_accumulators.py tests/test_fixedpoint_exact_route.py` | 1,892 passed, 1,554 skipped (CUDA and MPS-float64 cases) |
| `pytest tests/` | 28,432 passed, 17,155 skipped, 0 failed (6:35) |
| `bash dev/benchmarks/check_native.sh` | `OK    sweeps/cpu_quant_vector_sweep.cpp` among the others, `all native tools build` |

### The sweeps under Apple clang

Built as for M11: each file's header build line, with clang++ and Homebrew's
libomp in place of `g++` and `-fopenmp`:

```
FLAGS="-O3 -std=c++20 -ffp-contract=off -DMPTORCH_FAST_CAST=1 -Xpreprocessor -fopenmp \
    -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp -I mptorch/csrc"
clang++ $FLAGS dev/benchmarks/sweeps/cpu_quant_vector_sweep.cpp \
    mptorch/csrc/cpu/binaryK_quant_vector.cpp -o cpu_quant_vector_sweep
clang++ $FLAGS dev/benchmarks/sweeps/cpu_rne_tile_sweep.cpp \
    mptorch/csrc/cpu/gemm_rne_tile.cpp -o cpu_rne_tile_sweep
clang++ $FLAGS dev/benchmarks/sweeps/cpu_odd_tile_sweep.cpp \
    mptorch/csrc/cpu/gemm_odd_tile.cpp -o cpu_odd_tile_sweep
clang++ $FLAGS dev/benchmarks/sweeps/cpu_word_tile_sweep.cpp \
    mptorch/csrc/cpu/gemm_word_tile.cpp mptorch/csrc/cpu/gemm_word_odd_tile.cpp -o cpu_word_tile_sweep
```

All four compile with no warning. The times are on 11 threads.

| run | output | time |
|---|---|---|
| `cpu_quant_vector_sweep` | `the neon loops, all 2^32 binary32 inputs against the scalar casts`; 0 mismatches in each of the 76 comparisons (14 formats x 6 modes, less RNE for the two formats the fast gate leaves to the scalar loop: `kept by the scalar loop, skipped`); `ok: 0 mismatches in 76 comparisons` | 2:23 |
| `cpu_quant_vector_sweep control` | a mismatch count in every comparison (90,177,504 to 4,278,124,928); `control ok: 0 of the 76 comparisons found no mismatch` | 2:40 |
| `cpu_rne_tile_sweep` | `isa neon`; 0 mismatches in each of 9 formats (whole tiles: 172,032 of 196,608 blocks ran whole); `PASS: 9 formats checked, 0 mismatches` | 5:42 |
| `cpu_odd_tile_sweep` | `isa neon`; 0 mismatches in each of 10 formats (the same whole-tile count); `PASS: 10 formats checked, 0 mismatches` | 14:29 |
| `cpu_word_tile_sweep` | `the neon word tiles`; 0 mismatches in each of the 60 comparisons (6 formats x 5 modes, both carriers); `ok: 0 mismatches in all` | 13:31 |
| `cpu_word_tile_sweep control` | a mismatch count in every comparison (453,690,833 to 12,834,374,784); `control ok: 0 of the comparisons found no mismatch` | 14:09 |

**The narrow sweeps bite on NEON too.** I built both tile sweeps against a
copy of their tile file whose `narrow_lanes` reads every lane's B from
column 0 (`bj[l] = 0` in place of `l % tj`):

| run | output |
|---|---|
| broken `cpu_rne_tile_sweep` | mismatches in all 9 formats (235,704 to 1,789,524,340); `FAIL: 9 formats checked, 1792023655 mismatches` (5:54) |
| broken `cpu_odd_tile_sweep` | 1,022,852,050 to 1,433,113,616 mismatches in each of its first 6 formats, when I stopped it at 10 min |

### The elementwise quantizer's loops

**Vectorization.** I recompiled `cpu/binaryK_quant_vector.cpp` with the
extension's own flags (`build.ninja`'s `cflags` and `post_cflags`) plus
clang's vectorizer remarks. The one `#pragma omp simd` loop (line 46)
vectorized at width 4, interleave 1, in each of its six instantiations. In
the object (`objdump -d`), NEON operations (`.4s` and `.16b` mnemonics, or
`q` operands) are 41-43% of each function's instructions:

| function | instructions | NEON |
|---|---|---|
| `rne_neon` (`ltmp0` in objdump; `nm` names it) | 71 | 30 |
| `rna_neon` | 205 | 85 |
| `ru_neon` | 237 | 99 |
| `rd_neon` | 238 | 99 |
| `rz_neon` | 200 | 86 |
| `ro_neon` | 189 | 85 |

The rest are loads, stores, saved registers and the scalar remainder.

**Timing.** `cpu_quantize_elementwise.py` at 2^22 elements, one process per
setting (`MPTORCH_CPU_SIMD=0` against the default), alternated in three
rounds with the order reversed. The table gives the minimum over the rounds
of the tool's median. E4M3 on float32, ms, scalar -> vector (speedup):

| mode | 1 thread | 11 threads | Ryzen, 1 thread |
|---|---|---|---|
| RNE | 5.59 -> 1.15 (4.86x) | 1.17 -> 0.31 (3.77x) | 5.57 -> 0.93 |
| RNA | 5.57 -> 4.09 (1.36x) | 1.42 -> 0.90 (1.58x) | 5.73 -> 2.12 |
| RU | 20.43 -> 5.25 (3.89x) | 3.18 -> 1.16 (2.74x) | 19.2-20.6 -> 4.2 |
| RD | 19.89 -> 5.31 (3.75x) | 3.18 -> 1.18 (2.69x) | |
| RZ | 21.36 -> 4.41 (4.84x) | 3.40 -> 0.98 (3.47x) | 19.2 -> 2.93 |
| RO | 7.98 -> 3.94 (2.03x) | 1.36 -> 0.89 (1.53x) | 6.06 -> 1.85 |

- **The in-place rows agree.** I timed `y.copy_(x)` then
  `binaryK_quantize_(y)`, the copy subtracted, in three rounds: RNE 4.59x,
  RNA 1.36x, RU 4.00x, RD 3.77x, RZ 4.91x and RO 2.01x on one thread, and
  1.51-3.73x on 11. So macOS's allocator doesn't distort the out-of-place
  rows.
- **The tool's other rows move by 0.97-1.07x**, as they should: superfp,
  fixed point, SR, float64 and the copy don't take the loops.
- **RNA gains least.** Its scalar cast is as cheap as RNE's (5.57 ms), and
  its word cast has about three times the instructions of RNE's fast cast
  (205 against 71). With 4 lanes that leaves 1.36x, against 2.7x with the
  Ryzen's 16.
- **On 11 threads the word modes stay well above a copy's time**, unlike on
  the Ryzen's 16. They take 0.89-1.18 ms, 5-7x a copy (0.17 ms). From 1 to
  11 threads the vector loops scale 3.7-4.6x on the 5 performance and 6
  efficiency cores, so here they are compute-bound.

### The narrow tiles at HEAD

**`cpu_narrow_gemm.py`, one thread.** The tool alternates
`MPTORCH_CPU_SIMD=0` with the default. I ran it on HEAD and on a build whose
four tile files are `daa9304`'s, whose tiles take every shape. That build is
those files compiled with the extension's flags and linked into a copy of
the package, imported through `PYTHONPATH`. Each build ran twice,
alternated. Each cell gives the scalar loop's ns a mac step (the minimum
over both builds), then `daa9304`'s tiles / HEAD's tiles over it (< 1 is
faster than the scalar loop):

| M x N | outputs | RNE | RNE odd | RZ | RZ odd | fused RNE odd |
|---|---|---|---|---|---|---|
| 1x1 | 1 | 9.79: 1.04 / 1.02 | 18.56: 4.30 / 1.01 | 11.83: 1.38 / 1.01 | 19.20: 4.80 / 1.01 | 18.08: 2.28 / 1.00 |
| 1x2 | 2 | 4.56: 1.02 / 1.06 | 9.02: 4.54 / 1.01 | 9.55: 1.16 / 1.00 | 14.60: 3.30 / 1.02 | 8.72: 2.44 / 1.03 |
| 2x1 | 2 | 10.09: 0.52 / 0.53 | 16.48: 2.55 / 0.59 | 10.38: 1.14 / 1.02 | 17.59: 2.82 / 1.00 | 18.06: 1.24 / 0.53 |
| 1x4 | 4 | 3.44: 0.75 / 0.75 | 7.26: 2.75 / 2.74 | 9.28: 0.49 / 1.00 | 14.50: 1.71 / 1.02 | 4.72: 2.21 / 2.21 |
| 4x1 | 4 | 8.98: 0.36 / 0.36 | 15.33: 1.38 / 1.36 | 9.57: 0.54 / 1.02 | 16.76: 1.54 / 1.00 | 16.82: 0.71 / 0.71 |
| 2x2 | 4 | 5.38: 0.54 / 0.55 | 10.02: 2.07 / 2.05 | 9.16: 0.54 / 1.00 | 14.08: 1.81 / 1.03 | 9.34: 1.22 / 1.23 |
| 1x8 | 8 | 3.14: 0.44 / 0.44 | 6.97: 1.53 / 1.51 | 9.42: 0.32 / 0.32 | 15.14: 0.92 / 0.89 | 3.28: 1.72 / 1.72 |
| 8x1 | 8 | 8.47: 0.22 / 0.23 | 14.89: 0.79 / 0.77 | 9.25: 0.36 / 0.37 | 16.34: 0.91 / 0.89 | 16.23: 0.42 / 0.42 |
| 1x16 | 16 | 2.83: 0.30 / 0.30 | 6.66: 0.83 / 0.82 | 9.75: 0.24 / 0.24 | 16.34: 0.49 / 0.49 | 2.98: 0.95 / 0.95 |
| 4x4 | 16 | 3.65: 0.27 / 0.29 | 8.34: 0.70 / 0.70 | 8.82: 0.28 / 0.28 | 14.01: 0.59 / 0.61 | 5.52: 0.58 / 0.58 |
| 1x32 | 32 | 2.88: 0.24 / 0.24 | 6.66: 0.48 / 0.48 | 10.34: 0.22 / 0.22 | 18.13: 0.28 / 0.29 | 2.98: 0.57 / 0.57 |
| 8x8 | 64 | 2.89: 0.26 / 0.25 | 6.62: 0.33 / 0.32 | 10.37: 0.21 / 0.21 | 17.68: 0.22 / 0.22 | 2.94: 0.41 / 0.41 |

The tool's 16x1 and 32x1 rows follow 8x1's pattern.

- **At one output the passes cost NEON up to 4.8x the scalar loop**
  (`daa9304`: RNE odd 4.30x, RZ odd 4.80x, fused odd 2.28x), against up to
  2x on the Ryzen. HEAD brings every family to the scalar loop's speed
  there (1.00-1.02x).
- **Below the cut-off HEAD's narrow path beats the scalar loop on column
  tiles** (2x1: 0.53-0.59x).
- **The odd tiles still lose to the scalar loop from 4 outputs up to 8 or
  16** (1x4: 2.74x for the split, 2.21x for the fused). The cut-offs below
  deal with that.

**qsum and the odd palette edge.** `narrow_ab.py` is a scratch script: one
process per build, alternated, each making one warm-up call and keeping the
minimum of 7, and the table gives the minimum over the rounds. ns a mac
step, one thread:
- qsum: `qsum(x, mac=SplitMac(BinaryK(16, 11), BinaryK(16, 11)[,
  carrier_rounding=RO]))`, x 2^20 N(0, 1) values;
- the palette edge: `binaryK_matmul_mixed` (and `_fma_mixed`) of a
  [1, 4096] row by [4096, 256], entry 1 on two columns of every 32,
  `carrier_rounding=RO`.

| | `daa9304` -> HEAD (3 rounds) | HEAD -> the hunk below (6 rounds) |
|---|---|---|
| qsum, nearest carrier | 8.46 -> 8.18 (1.03x) | 8.17 -> 8.20 (1.00x) |
| qsum, odd carrier | **77.66 -> 16.85 (4.61x)** | 16.84 -> 16.87 (1.00x) |
| palette edge, odd split | 6.23 -> 4.43 (1.40x) | **4.17 -> 3.46 (1.21x)** |
| palette edge, odd fused | 3.27 -> 2.74 (1.19x) | **2.55 -> 2.14 (1.19x)** |

HEAD's palette rows ran 6-7% faster in the second run than in the first, so
each ratio holds within its own alternated run. On the Ryzen the odd chain
went 28.8 -> 13.75 ns. Neither palette edge is slower than `daa9304`'s,
unlike the first form of the narrow path that the brief mentions. The hunk
speeds up the palette edge's other groups, of 5 and 25 columns.

### NEON's cut-offs

The tool jumps from 4 outputs to 8 and 16, so I timed more shapes with
scratch copies of it: `narrow_ext.py`, `narrow_ext2.py`, and for the odd
tiles `lanes.py` and `wide.py` (every size from 1 x 1 to 33 x 1, then 2-D
shapes from 28 to 64 outputs). Each alternates the builds, in two to four
rounds.

**Word tiles: 4, not 8.** Each cell gives the scalar loop's ns, then
`daa9304`'s passes over it. HEAD hands these tiles back below 8 (1.00-1.08x
there).

| M x N | outputs | RNA | RU | RO | RZ | RZ, odd carrier |
|---|---|---|---|---|---|---|
| 1x3 | 3 | 8.45: 0.91 | 8.67: 1.26 | 8.51: 0.90 | 9.62: 0.91 | 14.53: 2.39 |
| 3x1 | 3 | 8.66: 0.91 | 8.94: 1.27 | 8.88: 0.87 | 9.88: 0.92 | 17.01: 2.06 |
| 1x4 | 4 | 7.97: 0.43 | 7.93: 0.61 | 7.85: 0.44 | 9.17: 0.50 | 14.47: 1.70 |
| 4x1 | 4 | 8.44: 0.51 | 8.77: 0.63 | 8.46: 0.50 | 9.59: 0.54 | 16.76: 1.55 |
| 2x2 | 4 | 7.91: 0.50 | 8.13: 0.63 | 7.91: 0.50 | 9.08: 0.54 | 14.18: 1.81 |
| 1x6 | 6 | 7.64: 0.51 | 7.75: 0.69 | 7.36: 0.49 | 9.76: 0.42 | 15.89: 1.16 |
| 6x1 | 6 | 8.09: 0.54 | 8.41: 0.69 | 8.29: 0.51 | 9.49: 0.52 | 16.54: 1.17 |
| 1x8 | 8 | 7.50: 0.37 | 7.75: 0.47 | 7.42: 0.35 | 9.61: 0.31 | 15.41: 0.90 |
| 8x1 | 8 | 8.16: 0.38 | 8.36: 0.46 | 8.33: 0.36 | 9.17: 0.36 | 16.34: 0.91 |
| 4x2 | 8 | | | | 8.85: 0.35 | 13.88: 1.07 |
| 1x12 | 12 | 7.49: 0.33 | 7.79: 0.41 | 7.32: 0.32 | 9.76: 0.27 | 15.91: 0.65 |

- **The word tiles** run 0.42-0.69x of the scalar loop at 4 and 6 outputs
  in all four modes, so NEON's cut-off is 4. At 3 outputs they run 0.87-1.27x,
  and RU loses.
- **The odd word tiles keep 8** (the last column, from `narrow_ext.py`):
  1.55-1.81x of the scalar loop at 4 outputs, 1.16-1.36x at 6, 0.90-1.07x
  at 8 and 0.65-0.81x at 12. 8 is a tie, and 12 would gain at most 7%, at
  4x2.
- **The RNE tiles keep 4.** Their passes run 0.36-0.75x of the scalar loop
  at 4 outputs (the tool's table). Below 4, HEAD's narrow path (0.39-1.06x)
  and `daa9304`'s passes (0.43-1.04x) are within a few percent of each
  other.

**Odd tiles: 40, not 4.** On NEON the odd passes take the directed
roundings from the FP unit's mode (`gemm_odd_tile.cpp`'s `Fpcr`): three
FPCR writes a K-step for the products, and three more for the split tile's
`odd_sums`. That cost is fixed per K-step, whatever the tile's width. The
passes' time per K-step (ns a mac step x outputs, `daa9304`, 1 x N tiles and
8x8 for 64) shows it:

| outputs | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| odd split, ns a K-step | 79.8 | 81.9 | 79.9 | 85.3 | 88.4 | 102.3 | 139.8 |
| odd fused, ns a K-step | 41.2 | 42.6 | 41.7 | 45.1 | 45.3 | 54.4 | 77.1 |
| RNE split (no FPCR), ns a K-step | 10.2 | 9.3 | 10.3 | 11.1 | 13.6 | 22.1 | 48.1 |

Up to 16 outputs the odd split pays 70-75 ns a K-step more than the RNE
split: six writes at 11-13 ns. The fused tile, with three writes, pays about
half. A scratch loop of writes (`fpcr.cpp`, alternating the RMode bits as
the passes do) measures 7.6-8.4 ns a write on its own. A K-step of the
narrow path costs about 3 ns an output, so the passes need some 35-40
outputs to amortize the writes. On x86, AVX-512 rounds by instruction, with
no mode switch, which suits the Ryzen's cut-off of 4. AVX2 switches MXCSR,
so its crossover may be higher too; I couldn't measure that here.

**Clang vectorizes the narrow path only at some sizes.** With one
`narrow_lanes` per size, the split tile cost 2.7-3.0 ns a step at 8, 12,
..., 28 lanes (4.6-4.9 at 4), but 6.9-7.4 ns at 5-7, 9-11 and 13-15 lanes,
the scalar chain's speed. The fused tile behaved the same way. A probe (one
`noinline` function per lane count, compiled with the extension's flags and
`-fsave-optimization-record`) shows why. Clang's loop vectorizer takes the
lane loop at width 4 at 4, 8, 12 and 16 lanes and at every size measured
from 17 up. It rejects 5-7, 9-11 and 13-15 as `VectorizationNotBeneficial`.
At 1-3 lanes the loop is fully unrolled. So the hunk pads a tile to the next
multiple of four lanes: the extra lanes repeat the last output and aren't
stored. Against unpadded sizes (one `narrow_lanes` per size) the padding
measured:
- 1.56-2.19x faster for the split tile and 1.04-1.98x for the fused at 5-7,
  9-11 and 13-15 outputs, and 1.17-1.19x for the split at 3;
- the same at multiples of four;
- 0.88-1.16x from 17 to 31;
- 0.93-0.96x for the fused tile at one or two outputs, so those keep
  `narrow_lanes<KIND, 1>` and `<KIND, 2>`.

**The crossover.** Here is HEAD against the hunk (narrow below 40, padded),
ns a mac step, one thread, four rounds:

| outputs | split, 1 x N | split, N x 1 | fused, 1 x N | fused, N x 1 |
|---|---|---|---|---|
| 1 | 18.78 -> 18.75 (1.00x) | | 18.21 -> 18.27 (1.00x) | |
| 2 | 9.12 -> 9.16 (1.00x) | 9.63 -> 9.65 (1.00x) | 9.01 -> 9.01 (1.00x) | 9.54 -> 9.51 (1.00x) |
| 3 | 7.43 -> 6.43 (1.16x) | 7.60 -> 6.38 (1.19x) | 6.42 -> 6.52 (0.98x) | 6.34 -> 6.52 (0.97x) |
| 4 | 20.11 -> 4.67 (4.30x) | 21.10 -> 4.81 (4.39x) | 10.39 -> 4.75 (2.19x) | 12.17 -> 4.91 (2.48x) |
| 5 | 17.50 -> 4.73 (3.70x) | 18.09 -> 4.63 (3.91x) | 8.94 -> 4.03 (2.22x) | 10.05 -> 3.92 (2.57x) |
| 6 | 15.00 -> 3.80 (3.95x) | 15.77 -> 3.90 (4.05x) | 7.57 -> 3.19 (2.38x) | 9.06 -> 3.28 (2.76x) |
| 8 | 10.66 -> 2.90 (3.68x) | 11.61 -> 2.95 (3.94x) | 5.69 -> 2.44 (2.33x) | 6.79 -> 2.49 (2.73x) |
| 11 | 9.03 -> 3.31 (2.72x) | 9.66 -> 3.28 (2.94x) | 4.72 -> 2.00 (2.36x) | 5.81 -> 1.92 (3.02x) |
| 16 | 5.49 -> 2.78 (1.98x) | 6.66 -> 2.96 (2.25x) | 2.82 -> 1.46 (1.92x) | 4.23 -> 1.53 (2.76x) |
| 24 | 4.12 -> 2.74 (1.50x) | 5.22 -> 2.90 (1.80x) | 2.10 -> 1.49 (1.41x) | 3.52 -> 1.53 (2.30x) |
| 31 | 3.61 -> 2.83 (1.28x) | 4.83 -> 2.95 (1.64x) | 1.90 -> 1.52 (1.25x) | 3.23 -> 1.58 (2.04x) |
| 32 | 3.27 -> 2.72 (1.20x) | 4.55 -> 2.88 (1.58x) | 1.69 -> 1.47 (1.15x) | 2.94 -> 1.50 (1.95x) |

The sizes between follow the same pattern. Some 2-D shapes:
- at 32 outputs (2x16, 16x2, 4x8): 1.17-1.58x for the split, 1.23-1.95x for
  the fused;
- at 36 (2x18, 3x12): 1.14-1.15x and 1.20-1.24x;
- at 40 and up both builds run the passes, and they agree within 0.93-1.09x,
  which is this A/B's noise.

A variant that ran the narrow path up to 64 outputs located the cut-off.
Narrow over the passes:
- 1.12-1.49x for the split and 1.19-2.09x for the fused at 32-36 outputs;
- 1.01-1.09x and 1.08-1.16x at 40-45;
- 0.91-0.94x and 0.90-1.03x on row tiles at 48 (2x24, 4x12, 6x8), while
  24x2 still ran 1.29x and 1.72x;
- 0.79-0.95x and 0.86-0.98x at 56-60;
- 0.30-0.47x and 0.38-0.76x at 61-64 outputs, padded to 64 lanes.

So 40 takes every size where the narrow path wins clearly, and the
matrix-vector tile (1 x 32) with them.

### The hunk for Linux: NEON's cut-offs and the padded narrow path

Both cut-offs are file-local constants in Linux's files. The change is
NEON-only, behind `__aarch64__`, with the x86 statements verbatim in each
`#else`. `git apply --check` takes it at `a507483`:

```diff
diff --git a/mptorch/csrc/cpu/gemm_word_tile.cpp b/mptorch/csrc/cpu/gemm_word_tile.cpp
--- a/mptorch/csrc/cpu/gemm_word_tile.cpp
+++ b/mptorch/csrc/cpu/gemm_word_tile.cpp
@@ -35,8 +35,14 @@ namespace mptorch::gemm_cpu
     // their setup costs more than the lanes save: on one thread the scalar
     // loop was 1.2-2x faster for a single output (a sum into one scalar),
     // and these tiles first won at eight outputs
-    // (dev/benchmarks/timing/cpu_narrow_gemm.py).
+    // (dev/benchmarks/timing/cpu_narrow_gemm.py). NEON's tiles, of four
+    // lanes, first win at four outputs (on an M3 Pro, 0.42-0.69x of the
+    // scalar loop at four and six outputs under RNA, RU, RO and RZ).
+#if defined(__aarch64__)
+    constexpr int64_t NARROW_TILE = 4;
+#else
     constexpr int64_t NARROW_TILE = 8;
+#endif
 
     // One K-step after another, each element's steps in K order (see
     // cpu/gemm_rne_tile.cpp's tile_steps, whose structure this is). FTI, FTJ
diff --git a/mptorch/csrc/cpu/gemm_odd_tile.cpp b/mptorch/csrc/cpu/gemm_odd_tile.cpp
--- a/mptorch/csrc/cpu/gemm_odd_tile.cpp
+++ b/mptorch/csrc/cpu/gemm_odd_tile.cpp
@@ -62,6 +62,9 @@
 #include <cmath>
 #include <cstdlib>
 #include <cstring>
+#if defined(__aarch64__)
+#include <utility>
+#endif
 
 #if defined(__x86_64__)
 #include <immintrin.h>
@@ -76,8 +79,17 @@ namespace mptorch::gemm_cpu
     // the passes, for gemm_rne_tile.cpp's reasons (its NARROW_TILE): the
     // passes on one lane or a few cost up to 2x the scalar loop, and the
     // palette tiles call these functions on column and row groups, so the
-    // tile is not handed back.
+    // tile is not handed back. On NEON the passes set the directed
+    // roundings through FPCR, three writes a K-step for the products and
+    // three more for the split tile's sums, a cost per K-step whatever the
+    // tile's width. On an M3 Pro the narrow path ran 1.1-4.4x faster than
+    // the passes from 4 to 36 outputs, about as fast at 40-45, slower on
+    // rows from 48 on, and 1.3-3.3x slower at 64 lanes.
+#if defined(__aarch64__)
+    constexpr int64_t NARROW_TILE = 40;
+#else
     constexpr int64_t NARROW_TILE = 4;
+#endif
 
     namespace odd = mptorch::gemm::odd;
 
@@ -289,6 +301,36 @@ namespace mptorch::gemm_cpu
         sums[l] = s[l];
     }
 
+#if defined(__aarch64__)
+    // The same for n <= N outputs, N a multiple of four; the lanes from n
+    // on repeat output n - 1 and are not stored. Clang vectorizes the lane
+    // loop four lanes at a time at a multiple of four and from 17 lanes on,
+    // and finds 5-7, 9-11 and 13-15 not worth it: those ran 1.6-2.2x
+    // slower a step on an M3 Pro than their size padded.
+    template <int KIND, int N>
+    [[gnu::always_inline]] inline void narrow_lanes_padded(const float *__restrict__ a_pack,
+                                                           const float *__restrict__ b_pack,
+                                                           float *__restrict__ sums, int64_t n,
+                                                           int64_t tk, int64_t tj,
+                                                           const BinaryKParams &pm,
+                                                           const BinaryKParams &pa)
+    {
+      float s[N];
+      int64_t ai[N], bj[N];
+      for (int l = 0; l < N; ++l)
+      {
+        const int64_t o = l < n ? l : n - 1;
+        s[l] = sums[o], ai[l] = (o / tj) * tk, bj[l] = o % tj;
+      }
+      for (int64_t k = 0; k < tk; ++k)
+        for (int l = 0; l < N; ++l)
+          s[l] = narrow_step<KIND>(a_pack[ai[l] + k], b_pack[k * tj + bj[l]], s[l], pm, pa);
+      for (int l = 0; l < N; ++l)
+        if (l < n)
+          sums[l] = s[l];
+    }
+#endif
+
     template <int KIND>
     [[gnu::always_inline]] inline void narrow_steps(const float *__restrict__ a_pack,
                                                     const float *__restrict__ b_pack,
@@ -296,6 +338,25 @@ namespace mptorch::gemm_cpu
                                                     int64_t tk, int64_t tj, const BinaryKParams &pm,
                                                     const BinaryKParams &pa)
     {
+#if defined(__aarch64__)
+      // one output or two in lanes of their own (padded to four, they ran
+      // up to 7% slower); from three, the narrowest padded size that holds
+      // them
+      static_assert(NARROW_TILE % 4 == 0, "lanes in fours");
+      const int64_t n = ti * tj;
+      if (n == 1)
+        narrow_lanes<KIND, 1>(a_pack, b_pack, sums, tk, tj, pm, pa);
+      else if (n == 2)
+        narrow_lanes<KIND, 2>(a_pack, b_pack, sums, tk, tj, pm, pa);
+      else
+        [&]<int... Q>(std::integer_sequence<int, Q...>)
+        {
+          (void)((n <= 4 * (Q + 1) &&
+                  (narrow_lanes_padded<KIND, 4 * (Q + 1)>(a_pack, b_pack, sums, n, tk, tj, pm, pa),
+                   true)) ||
+                 ...);
+        }(std::make_integer_sequence<int, NARROW_TILE / 4>{});
+#else
       static_assert(NARROW_TILE == 4, "a narrow_lanes per size below the cut-off");
       const int64_t n = ti * tj;
       if (n == 1)
@@ -304,6 +365,7 @@ namespace mptorch::gemm_cpu
         narrow_lanes<KIND, 2>(a_pack, b_pack, sums, tk, tj, pm, pa);
       else
         narrow_lanes<KIND, 3>(a_pack, b_pack, sums, tk, tj, pm, pa);
+#endif
     }
 
     template <int KIND, class Ops>
```

What I ran on it. The hunk's two files were compiled with the extension's
flags, linked into a copy of the package, and imported through
`PYTHONPATH`; the final comment edits leave both objects byte-identical to
the ones tested.
- **Words.** A scratch script (`prop_check.py`) hashes every result of 7,752
  calls. It covers every M x N with M N < 40 (the whole narrow range) at
  K = 5, 37 and 300, some with an inf (so NaN hand-backs). The calls are the
  split, product-only and fused macs, under the odd carrier and under RNA,
  RU, RD, RZ and RO, plus an odd palette. The hash under the default setting
  equals the hash under `MPTORCH_CPU_SIMD=0`, and HEAD's scalar loop's.
- **The odd tile sweep** against the hunk's `gemm_odd_tile.cpp`: `isa neon`,
  0 mismatches in each of the 10 formats, `PASS: 10 formats checked, 0
  mismatches` (15:00). Its narrow passes now run the padded 4-lane path
  for its rows and columns of three and its 1 x 3 and 3 x 1 tiles.
- **Tests.** `test_cpu_simd.py`, `test_carrier_rounding.py`,
  `test_accumulators.py`, `test_reductions.py`, `test_number_formats.py`,
  `test_gemm_mixed_palette.py`, `test_cpu_tile_state.py`,
  `test_gemm_dispatch.py` and `test_fixedpoint_exact_route.py`: 3,096
  passed, 2,252 skipped. A pytest plugin confirmed that the copy's
  `_C.abi3.so` was the one loaded.
- **Timing:** the tables above.
- **Size:** `gemm_odd_tile.o` grows from 19 KB to 49 KB: 12 sizes x 3
  kinds. One `narrow_lanes` per size, unpadded, made it 139 KB.

### AN-b's threshold on MPS

`fixedpoint_gemm.py exact --device mps --all-sizes --rounds 15 --sizes 96
128 144 160 176 192 224 256`, three runs, (25, 14). The kernel and the route
in ms a call, the range over the runs:

| n | steps | the kernel | the route | route / kernel |
|---|---|---|---|---|
| 144 | 2^21.5 | 0.205-0.232 | 0.219-0.234 | 1.01-1.07 |
| 160 | 2^22.0 | 0.238-0.270 | 0.225-0.257 | 0.90-0.97 |
| 176 | 2^22.4 | 0.295-0.305 | 0.236-0.261 | 0.79-0.88 |
| 192 | 2^22.8 | 0.315-0.332 | 0.243-0.268 | 0.77-0.83 |
| 224 | 2^23.4 | 0.417-0.437 | 0.281-0.296 | 0.66-0.68 |
| 256 | 2^24 | 0.543-0.555 | 0.299-0.312 | 0.55-0.56 |

The route breaks even between 144^3 and 160^3, so 2^22 sits at the crossing.
The gains match M10b's (192^3: 0.78; 256^3: 0.61). The 96^3 and 128^3 rows
come first in each run, and launch noise swamps them (the kernel's 128^3
took 0.20-0.41 ms). Without `--all-sizes` both rows take the kernel below
2^22, as they should.

### For Linux

1. **The hunk above.** It sets NEON's cut-offs to 4 for the word tiles and
   40 for the odd tiles, and pads the odd narrow path to fours. Odd tiles of
   4 to 39 outputs run 1.1-4.4x faster, the odd palette edge 1.19-1.21x
   faster, and qsum is unchanged. The one loss is the fused tile at 3
   outputs, 2-3% slower.
2. **The sweep's coverage, if you take it.** `cpu_odd_tile_sweep.cpp`'s
   narrow passes (1 x 1 to 3 x 1, and the rows of eight) reach only 4 of the
   12 lane counts on arm64 (1 and 2, and 4 and 8 padded). Here only
   `prop_check.py`'s calls through the ops cover the sizes from 4 to 39. A sweep pass over those sizes, padded lanes
   included, would gate them.
3. **AVX2** switches MXCSR per pass as NEON switches FPCR, so its odd tiles'
   crossover may also lie above 4 outputs. If it does, GCC's vectorizer may
   need the same padding.
4. **The odd word tiles** hand narrow tiles back to a scalar loop that costs
   14-19 ns a step at 1 to 4 outputs here. A narrow path in registers, as
   the odd tiles have, would likely help there too. I didn't build one.

### Open

- Nothing waits on the Mac for M11 or M12.

## M12 follow-up: the odd tile sweep's new pass on NEON (at `de279ae`)

The sweep's new pass gives 0 mismatches on NEON, so every padded lane count
of the odd narrow path now gives the scalar step's words in a sweep. A
column-major bug in the padded path gives 77,560,575 mismatches on the first
format here, the count Linux found on x86. The shipped hunk is the one I
tested: the two tile files differ from it in comments only, and their objects
are byte-identical to the ones M12 timed. No file changed but this report.

Machine: the M3 Pro, on AC power, with no thermal or performance warning
recorded. Built flag-on (`MPTORCH_ODD_ACCUMULATED=1 MPTORCH_COUNTING=1`) with
Apple clang 21.0.0.

| command | result |
|---|---|
| `cmp` of `gemm_odd_tile.o` and `gemm_word_tile.o` against the M12 hunk's | identical |
| `pytest` on the tile, palette and carrier files (`test_cpu_simd.py`, `test_carrier_rounding.py`, `test_accumulators.py`, `test_reductions.py`, `test_number_formats.py`, `test_gemm_mixed_palette.py`, `test_cpu_tile_state.py`, `test_gemm_dispatch.py`, `test_fixedpoint_exact_route.py`) | 3,096 passed, 2,252 skipped |
| `pytest tests/` | 28,432 passed, 17,155 skipped, 0 failed (6:28) |
| `cpu_odd_tile_sweep` (built as in M12) | `isa neon`; 0 mismatches in each of the 10 formats (whole tiles: 172,032 of 196,608 blocks ran whole); `PASS: 10 formats checked, 0 mismatches` (15:43 on 11 threads, against 14:29 before the new pass) |
| the same against a broken copy of `gemm_odd_tile.cpp` | 77,560,575, 78,316,534 and 87,525,712 mismatches on the first three formats, when it was stopped at 5 min |

The broken copy maps each padded lane column-major: `ai[l] = (o % ti) * tk`
and `bj[l] = o / ti` (written `n / tj` for `ti`) in `narrow_lanes_padded`.
That is correct on rows and columns and wrong on 2-D tiles, so the old sweep,
which cut only 1 x 1 to 3 x 1, could not see it.

To keep the broken run's lines when the time limit stops it, I built that
copy with stdout line-buffered (`-include` of a header with a static
`setvbuf(stdout, nullptr, _IOLBF, 0)`). The sweep's own build is the header's
line.

### Open

- Nothing waits on the Mac for M12.

## M13: superfp's NaN on P3109's codes, on MPS (at `01da482`, the code at `d2b40a9`)

The MPS source generators that now pass the sign work. The superfp
quantizers, in-place ops, GEMMs and block ops give the CPU's words on MPS,
and the new code-point test passes on MPS, signed and unsigned, in both
domains. The quantizer sweep gives 0 mismatches over all 2^32 words,
including the unsigned row's moved top and the new `(0, 4, 2, 7)` row. A
generator that wrote the wrong sign would be caught: a scratch build whose
superfp tail always writes `true` fails exactly the unsigned cases, and the
sweep's unsigned row then mismatches on about 1.03 billion words. No file
changed but this report.

Machine: the M3 Pro, on AC power, with no thermal or performance warning
recorded. Built with `MPTORCH_ODD_ACCUMULATED=1 MPTORCH_COUNTING=1 uv sync
--group hf` (1:37; `build_config.txt` shows `-DMPTORCH_ODD_ACCUMULATED=1
-DMPTORCH_COUNTING=1`), Apple clang 21.0.0.

### What I read

Every `make_superfp_params` call that Metal compiles passes the sign:
- the tails from `mps/quantize_tail.h`'s `superfp_format_lines` (its callers
  in `mps/quantize.cpp`, three, and `mps/pointwise_entry.cpp`);
- `mps/block_entry.cpp`'s two calls;
- the header calls in `common/gemm_policy.h`, `gemm_counted.h`,
  `block_decode.h` and `pointwise.h`.

`grep` finds no other generator of a superfp builder call under `mps/`.

### Tests

| command | result |
|---|---|
| `bash dev/benchmarks/check_native.sh` | every host tool `OK`, `all native tools build` |
| `pytest tests/test_mps.py tests/test_superfp_quantize.py tests/test_block_pack.py tests/test_unsigned_zero.py tests/test_gemm_mixed_palette.py tests/test_event_counts.py tests/test_counted_gemm.py tests/test_pointwise_fused.py` | 13,018 passed, 7,045 skipped, 0 failed (8:36, the first run after the prelude changed, so every kernel it touched compiled cold) |
| `pytest tests/test_superfp_quantize.py -k code_points` | the 4 MPS cases (signed and unsigned, `SAT_NONE` and `SAT_FINITE`) and the 4 CPU cases pass; the 4 CUDA cases skip |
| `pytest tests/test_block_pack.py -k 'sfp5_unsigned or e4m3_sfp_m0_scale'` | 48 passed (24 of them on MPS), 24 skipped (CUDA) |
| `pytest tests/` | 28,471 passed, 17,165 skipped, 0 failed (17:12, the rest of the kernels compiling cold) |

### The sweep

| command | result | time |
|---|---|---|
| `mps_backend.py sweep` | `216 format/mode cases, 0 mismatches in all`; every one of the 45 superfp cases (5 formats x 9 modes) 0, including `superfp 4/3/1 bias=3 SAT_PROPAGATE unsigned` and the new `superfp 0/4/2 bias=7 SAT_NONE` | 11:55 |
| `mps_backend.py control` | `explicit FMA in the split: 248512514 mismatches`, `math_mode(fast): 2261778181 mismatches` (both must be nonzero) | 0:07 |

The sweep took about twice the brief's 6 min. It prints one
`FormatRangeWarning`, for `SuperFP(9, 3, 1, 3)`, whose bottom lies below
binary32's smallest subnormal. That is a property of that format's bottom,
unrelated to this change.

### Would a wrong sign be caught?

The brief puts the likeliest failure in a generated tail. I built a scratch
copy of `mps/quantize_tail.h` whose `superfp_format_lines` writes `true` for
every format, compiled `mps/quantize.cpp` and `mps/pointwise_entry.cpp`
against it with the extension's flags, and linked them into a copy of the
package, imported through `PYTHONPATH`. A pytest plugin confirmed that
copy's `_C.abi3.so` was the one loaded.
- **The tests.** `pytest tests/test_mps.py tests/test_superfp_quantize.py -k
  'superfp or sfp or code_points'` gives 11 failed, 1,117 passed. The
  failures are exactly the unsigned cases: `test_superfp_quantize` on
  `(4, 3, 1, 3, unsigned, SAT_PROPAGATE)` in all 9 modes, and
  `test_the_code_points_are_p3109s`'s two unsigned MPS cases. Every signed
  case passes, as it should, since the copy writes their sign right.
- **The sweep.** The sweep's unsigned row alone over all 2^32 words (its
  `quantizers()` and `chunks()`, through a scratch script) gives 1,032,847,360
  to 1,033,371,647 mismatches in each of the 9 modes, none on subnormal
  inputs. The same script at HEAD gives 0 in all 9.

### Open

- Nothing waits on the Mac for M13.
