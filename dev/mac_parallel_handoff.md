# Mac/MPS work in parallel with the remaining phases

Written on Linux on 2026-11-26 for the Mac session, which runs it while the
Linux session works through the remaining phases' CPU and CUDA parts. It is
self-contained: the plan it draws on, `dev/continuation_plan.md`, is
untracked and local to Linux, and the Mac's copy of it is older than most of
what follows. Phase letters (AI, AK, S, ...) are that plan's names; this
document says what each one means wherever it uses one.

## 1. In short

**Where things stand.** The last Mac round passed at `137dc14` (phase T's
step 6, the fixed-point GEMM on MPS). Since then Linux committed and pushed
five commits, none of them run on a Mac:

| commit | what | touches Metal? |
|---|---|---|
| `85d1d31` | T, step 7: the fixed-point GEMM's Python API (`Overflow`, fixed-point `SplitMac`/`FusedMac`, `fixedpoint_matmul(_fma)`, `fixedpoint_gemm_formats(_fma)`, wide results in the layers) and its docs | no; its new tests have MPS cases that have never run |
| `d83b4fe` | T, timing gate: stochastic rounding in the fixed-point GEMM laid out for the K-loop, host and CUDA | `common/fixedpoint_gemm.h`'s text, behind `!__METAL_VERSION__` |
| `cbcdda4` | plan labels removed from tracked comments, docstrings and tests; one test renamed | comments only (`mps/fixedpoint.metal`'s first line) |
| `71099aa` | T: the deterministic modes laid out the same way, on the device in every mode, on the host under RU, RD and RZ | `common/fixedpoint_gemm.h`'s text, behind `!__METAL_VERSION__` |
| the commit adding this file | this handoff, and `--device mps` for `dev/benchmarks/timing/fixedpoint_gemm.py` | no |

**What Linux does next**, in this order, on the CPU and CUDA: AI (fused
pointwise kernels), AK (vector tiles for the CPU palette GEMMs), S (numeric
event counters), AL (flash attention's speed). Each has an MPS part, which
this document assigns to the Mac with a sync point.

**The Mac's tasks:**

| task | what | when | depends on Linux? |
|---|---|---|---|
| M1 | the verification round for the five commits above, the arm64 host timing of the new layouts, MPS timings | now, first | no |
| M2 | phase AQ: why the fused `ScaledQuant` by blocks of 32 gains nothing in float32 on MPS | now | no |
| M3 | phase AR: why `block_project` takes 2.2-4.2x `block_pack`'s time on MPS | now | no |
| M4 | optional, only with the user's go-ahead: the fixed-point GEMM's new layouts on Metal | after M1 | must be merged before Linux starts S |
| M5 | phase AJ's open question: what `torch.compile` does on MPS | now | no |
| M6a | AI's pre-work: binary32 round-to-odd division and square root on Metal | now | no (an interface is fixed below) |
| M6b | AI's MPS part: the fused pointwise, table and (if built) row-reduction ops on Metal | at SP-AI | yes |
| M7 | AK's NEON tiles: build, test, sweep and tune on the M3 Pro | at SP-AK | yes |
| M8 | S's MPS part: the counting variant of `gemm.metal` and its entries | at SP-S | yes |
| M9 | AL's Mac round, and its Metal kernel if Linux builds one | at SP-AL | yes |

M2, M3 and M5 are independent of each other and of Linux; run them in any
order after M1.

## 2. Ground rules for working in parallel

### 2.1 Who edits what

| owner | files |
|---|---|
| **the Mac** | `mptorch/csrc/mps/**`; `tests/test_mps.py`, `tests/test_fixedpoint_mps.py` and any new `tests/test_*mps*.py`; `dev/benchmarks/mps_backend.py`; `dev/mac_parallel_report.md` |
| **shared, edited only at a sync point and listed in the report** | `setup.py` (a new root in `embed_metal_sources`); `mptorch/csrc/quant_ops.h` (MPS declarations); `mptorch/csrc/common/**` (Metal arms only: behind `#if defined(__METAL_VERSION__)`, with the host and CUDA statements verbatim in the `#else`); MPS parameters in the other test files |
| **Linux** | everything else |

**Every edit under `mptorch/csrc/common/`** needs a Linux rerun of
`dev/benchmarks/gates/binary_identity.py`. Name each such hunk in the report.

### 2.2 Commits and pushes

The Mac session commits nothing and pushes nothing. At each sync point, or
when a task is done, the user commits the Mac's changes and pushes them, and
Linux pulls. Keep each task's changes apart (one task, one set of files), so
that the user can make one commit per task. Before starting a task that
waits on a sync point, pull.

### 2.3 Reports

Append one section per task to **`dev/mac_parallel_report.md`** (new): what
was done, the commands and their output lines that matter, the measurements
with the machine's state (power, thermal), the files changed with the
shared ones' hunks, and anything left open. The user pushes it with the
task's commit.

### 2.4 The bar

- **Word for word with the CPU.** An MPS op returns the CPU op's result bit
  for bit, SR included: MPS draws its SR seed from the CPU generator
  (`draw_cpu_seed`) and keys the Philox streams as the CPU kernels do. Tests
  use `torch.equal`, never a tolerance, wherever the docs promise this. The
  documented exceptions: a NaN's payload, and the GEMM's arithmetic between
  roundings, which flushes binary32 subnormals on the GPU (products and sums
  below 2^-126 become zero before they are rounded).
- **binary32 only.** MPS has no float64; `tests/conftest.py` skips the MPS
  cases of a test parametrized over float64 or a binary64 carrier.
- **Rules for Metal sources** (the repository's `CLAUDE.md`, macOS section):
  - every reference and pointer in a shared header needs an address space,
    through `MPTORCH_THREAD` and `MPTORCH_CONSTANT`;
  - nothing in a header the prelude includes may be named after an MSL
    built-in type or address-space keyword (`half`, `bfloat`, `uchar`,
    `ushort`, `uint`, `ulong`, the vector types, `device`, `constant`,
    `thread`, `threadgroup`), not even a local in an uninstantiated template;
  - MSL 3.1 (pinned in `metal_runtime.mm`) is C++14 with no lambdas; `if
    constexpr` and nested namespaces are accepted as extensions;
  - Metal has no derived classes and no `std::` traits; the host-only parts
    of a header sit behind `!defined(__METAL_VERSION__)`;
  - a new float operation on a cast's input value needs the flush treatment
    (`bit_helper.h`'s `metal_mul_rn`/`metal_div_rn`/`metal_sub_rn`, or the
    word), proven in `tests/test_mps.py::test_subnormal_inputs_round_exactly`.
- **The Metal cache.** Editing a root (`gemm.metal`, `quantize.metal`,
  `block.metal`, the prelude) changes every kernel's source text that
  includes it, and the next suite recompiles thousands of kernels (~10 min
  cold instead of a few warm). Do not edit them without need, and batch such
  edits.

### 2.5 Before each task

```bash
git pull                                         # at the commit the task names
rm -rf "${TMPDIR:-/tmp}/torchinductor_$USER"     # after any schema change (Inductor's cache does not key on it)
MPTORCH_ODD_ACCUMULATED=1 uv sync --group hf     # this Mac's builds use the flag; export it for the session
```

Check that the machine is on AC power, with no thermal warning, before any
timing. Lint any Python you change, as CI does:

```bash
.venv/bin/ruff check mptorch tests dev/benchmarks && .venv/bin/ruff format --check mptorch tests dev/benchmarks
.venv/bin/ty check mptorch && .venv/bin/pre-commit run --all-files
```

## 3. Sync points

| point | Linux delivers | the Mac then does |
|---|---|---|
| **SP-0** (now) | the commit adding this file, pushed | M1 first; then M2, M3, M5 and M6a in any order; M4 only with the user's go-ahead |
| **SP-1** | after M1's report: if the arm64 timing disagrees with the host's choice of modes (section 4.4), Linux adjusts `fixedpoint_word_call_v` and pushes | rerun section 4.4 on the new commit |
| **SP-AI** | AI's ops on the CPU, CUDA and Meta, with `common/pointwise.h` (and `common/pwl.h`) written as valid Metal, the schemas, and tests with MPS parameters that skip without an MPS kernel; the interface of M6a's primitives confirmed | M6b; M6a's primitives merged into it |
| **SP-AK** | AK's palette tiles, NEON included, written on Linux and checked on x86 | M7 |
| **SP-S** | S's counted ops on the CPU, CUDA and Meta, `common/event_counts.h`, and the MPS tests that hold counts to the CPU's | M8. **M4, if done, must be merged before Linux starts S**, since S wraps the GEMM policies, the fixed-point ones included |
| **SP-AL** | AL's steps 1 and 2 (compiled flash, batched query tiles), and step 4's ops if Linux builds them | M9 |

Linux will say at each point which commit it is, and the user relays it.

## 4. M1: the verification round

### 4.1 What changed, for the Mac

- **`85d1d31`** is Python and docs. New MPS cases that have never run:
  `tests/test_fixedpoint_layers.py`'s float32 tests on MPS (`FLOAT32_DEVICES`:
  the flat wrappers against the ops in four modes, a `QLinear`'s tiers 2 and
  3, a `QConv2d`'s and a `qmatmul`'s tier 3) and
  `test_on_mps_a_wide_result_is_refused_in_every_pass` (a wide fixed-point
  result raises on MPS, by the Python check without an output rounding and by
  the op's own check with one).
- **`d83b4fe` and `71099aa`** change `common/fixedpoint_gemm.h`, but only
  inside `#if !defined(__METAL_VERSION__)`: Metal compiles the same code as at
  `137dc14`, and the fixed-point kernels recompile once because the flattened
  `fixedpoint` root's text changed. The host gets new code, which Apple clang
  compiles for arm64 for the first time:
  - under SR, `cast_fixedpoint_stochastic_gemm` (one range test of the word,
    then a straight line from per-launch constants) and
    `fixedpoint_round_sr`;
  - under the deterministic modes, `fixedpoint_word_round`, which the host
    takes under RU, RD and RZ only (`fixedpoint_word_call_v`), in its host
    spelling (0/1 words with bitwise operations);
  - Linux measured these on x86-64 with GCC; arm64 with clang may choose
    differently (section 4.4).
- **`cbcdda4`** renamed `test_what_phase_y_added_runs` to
  `test_the_accumulated_odd_steps_run` (it runs on builds with
  `MPTORCH_ODD_ACCUMULATED=1`, as this Mac's are).

### 4.2 Build and tests

```bash
MPTORCH_ODD_ACCUMULATED=1 uv sync --group hf
uv run --no-sync pytest tests/test_fixedpoint_mps.py tests/test_fixedpoint_layers.py tests/test_fixedpoint_gemm.py tests/test_entry_copies.py -q
uv run --no-sync pytest tests/ -q                      # the full suite
```

The bar: no failure; no MPS case skipped except for float64, a CUDA device,
or a test's own rule for a combination that does not exist. On Linux the
suite gives 21,324 passed and 9,865 skipped (the MPS cases among the skips).

### 4.3 The host sweeps under Apple clang

The two fixed-point sweeps hold the host code to the shipped casts over
every binary32 input. They need OpenMP to finish in minutes. Homebrew's
libomp is fine for these standalone binaries, which are separate processes;
the extension itself must never link one (CLAUDE.md, macOS):

```bash
OMP=$(brew --prefix libomp 2>/dev/null)    # empty if not installed: then build without the two OMP flags, one thread
FLAGS="-O3 -std=c++20 -ffp-contract=off -DMPTORCH_FAST_CAST=1 -I mptorch/csrc -I dev/benchmarks/sweeps"
[ -n "$OMP" ] && FLAGS="$FLAGS -Xpreprocessor -fopenmp -I$OMP/include -L$OMP/lib -lomp"
clang++ $FLAGS -x c++ dev/benchmarks/sweeps/gemm_cast_fixedpoint_arith.cu -o /tmp/fx_arith
/tmp/fx_arith det-short      # ~1.3 min on 16 x86 threads; tens of minutes on one
/tmp/fx_arith det-control
/tmp/fx_arith sr-short       # ~10 min on 16 x86 threads; skip it without OpenMP (hours)
/tmp/fx_arith sr-control
/tmp/fx_arith spot
clang++ $FLAGS dev/benchmarks/sweeps/fixedpoint_int_sweep.cpp -o /tmp/fx_int && /tmp/fx_int
clang++ ${FLAGS/-O3/-O2} -fsanitize=undefined -fno-sanitize-recover dev/benchmarks/sweeps/fixedpoint_int_sweep.cpp -o /tmp/fx_int_ub && /tmp/fx_int_ub
bash dev/benchmarks/check_native.sh
```

The bar:
- `det-short`: `6 formats, 36 cases (format, mode); 0 cases mismatch`;
- `det-control`: two nonzero counts (on Linux 68,478 and 8,321,159,945);
- `sr-short`: `6 formats, 54 cases (format, N, variant); 0 cases mismatch`;
- `sr-control`: a nonzero count (on Linux 1,941,258,420);
- `spot`: `0 admitted formats mismatch`;
- `fx_int` and `fx_int_ub` end with `every comparison agrees`, the two
  `chain` negative controls differ, and the sanitizer prints no `runtime
  error`;
- `check_native.sh` ends with `all native tools build`.

A mismatch is a finding about the shared header under clang: report the
line the sweep prints (it names the format, the mode and a word), and do not
edit the header; Linux owns it.

### 4.4 Does the host's choice of modes hold on arm64?

The host runs the new line under RU, RD and RZ, where it measured
0.23-0.81x of the shipped casts on the Ryzen, and keeps the shipped casts
under RNE, RNA and RO, where the line measured 1.0-1.3x. Measure the same on
the M3 Pro's CPU: the committed build against `85d1d31`'s (before both
layouts), alternated, through the ops.

1. Build the base in a second worktree:

   ```bash
   git worktree add ../mptorch-base 85d1d31
   (cd ../mptorch-base && ../mptorch/.venv/bin/python setup.py build_ext --inplace)
   ```

   (`../mptorch` stands for this checkout; use the venv of this checkout, so
   that torch is the same.)
2. Save this script outside the repository, as `/tmp/ab_ops.py`:

   ```python
   """One round of the fixed-point ops under one build: name|mode|ps per step (min of 3)."""
   import sys
   import time
   import warnings

   import torch

   import mptorch
   from mptorch import RoundMode as R
   from mptorch.quant import fixedpoint_matmul, fixedpoint_matmul_fma
   from mptorch.quant.ops.fixedpoint import _fixedpoint_spec

   warnings.simplefilter("ignore")
   device, root = sys.argv[1], sys.argv[2]
   assert mptorch.__file__.startswith(root), mptorch.__file__  # the build meant, not the editable one
   torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
   n = 1024 if device == "mps" else 384
   g = torch.Generator().manual_seed(0)
   a = (torch.randn(n, n, generator=g) * 4).to(device)
   b = (torch.randn(n, n, generator=g) * 4).to(device)


   def sync():
       if device == "mps":
           torch.mps.synchronize()


   def t(f):
       f()
       sync()
       best = float("inf")
       for _ in range(3):
           t0 = time.perf_counter()
           f()
           sync()
           best = min(best, time.perf_counter() - t0)
       return best


   for mode in (R.RNE, R.RNA, R.RU, R.RD, R.RZ, R.RO, R.SR):
       prng = 4 if mode is R.SR else 0
       kw = dict(mul_wl=16, mul_fl=8, acc_wl=16, acc_fl=4, mul_prng_bits=prng, acc_prng_bits=prng)
       spec = _fixedpoint_spec(**kw, rounding_mode=mode)
       rows = {
           "carrier": lambda: fixedpoint_matmul(a, b, **kw, rounding_mode=mode),
           "int split": lambda: spec.op(a, b, False, False, *spec.args, True),
           "int fused": lambda: fixedpoint_matmul_fma(
               a, b, fma_wl=16, fma_fl=4, fma_prng_bits=prng, rounding_mode=mode
           ),
       }
       for name, f in rows.items():
           print(f"{name}|{mode.name}|{t(f) / n**3 * 1e12:.3f}")
   ```

3. Alternate the two builds, four rounds, the order flipped each round, from
   a directory outside both checkouts:

   ```bash
   cd /tmp
   PY=<this checkout>/.venv/bin/python
   for r in 1 2 3 4; do
     if [ $((r % 2)) = 1 ]; then o="base new"; else o="new base"; fi
     for v in $o; do
       root=$([ $v = base ] && echo <the base worktree> || echo <this checkout>)
       PYTHONPATH=$root $PY /tmp/ab_ops.py cpu $root/mptorch 1 | sed "s/^/$v|/"
     done
   done > /tmp/ab_cpu.txt
   ```

   and report each row's minimum per build and the ratio new/base. Repeat
   with `8` threads in place of `1` if time allows.

**What to look for.** Under RU, RD and RZ the new build should be faster on
the carrier path and the integer split; under RNE, RNA, RO and SR, and on
the fused mac, within a few percent. If a mode the host leaves to the
shipped casts would gain from the line on arm64, or one it gives the line
loses, report the numbers; Linux then adds an `__aarch64__` arm to
`fixedpoint_word_call_v` (SP-1). The line's host spelling can be timed under
every mode by building the new checkout with the host's choice widened
locally (`fixedpoint_word_call_v` returning `fixedpoint_word_mode_v<RM>` in
the `#else`): a local experiment, not a change to report as a diff.

### 4.5 MPS timings, for the docs

```bash
uv run --no-sync python dev/benchmarks/timing/fixedpoint_gemm.py ops --device mps --modes RNE RZ SR
uv run --no-sync python dev/benchmarks/timing/fixedpoint_gemm.py layer --device mps
```

`--device mps` is new in this commit; report any problem with it. The docs'
cost paragraph (`docs/source/kernels/arithmetic.rst`, "Fixed-point macs")
quotes the M3 Pro only as "the integer accumulator takes 7 to 10 times the
carrier path under RNE"; the rows above against the `Binary16p9` reference,
whose casts take the fast path, are what it lacks.

## 5. M2: phase AQ, the fused `ScaledQuant` by blocks on MPS

**The observation** (the last Mac round, 4096 x 4096, `BinaryK(8, 4)` under
`SAT_FINITE`, two runs within 1%): AG's fused op (`binaryK_quant_scaled` and
its siblings, a call of `ScaledQuant` whose scaled axis is last) against the
four separate passes it replaces:

| | fused | four passes | speedup | RTX 4060 |
|---|---|---|---|---|
| float32, per tensor | 2.18 ms | 3.99 ms | 1.83x | 1.9-2.2x |
| float32, per row | 4.51 ms | 5.80 ms | 1.29x | |
| float32, blocks of 32 | 6.23 ms | 6.05 ms | **0.97x** | |
| bfloat16, per tensor | 1.65 ms | 5.01 ms | 3.03x | 2.0-3.3x |
| bfloat16, per row | 4.06 ms | 6.81 ms | 1.68x | |
| bfloat16, blocks of 32 | 5.76 ms | 7.09 ms | 1.23x | |

**Why it is a question.** The four passes read and write about 2.3 times the
memory the fused call does, so a fused kernel no faster than them is bound
by its arithmetic. Its work per element is the same at every granularity;
part of the shrinking gain may be the largest-magnitude pass, which both
calls run and which costs most by blocks.

**Suspects, in this order:**
1. **64-bit integer index arithmetic per element.** `mps/scaled_quant.metal`
   computes `row = i / q.cols` and `col / q.block` on `uint64_t` for every
   element, and Apple GPUs emulate 64-bit integers (the fixed-point integer
   accumulator ran 3.4-4.7x the carrier path on the M3 Pro).
2. **The scale load and the exact multiply and divide per element**
   (`metal_mul_rn`/`metal_div_rn`'s subnormal tests), where a thread's four
   elements share a block's scale.
3. **The grid:** four elements a thread, against torch's tuned elementwise
   kernels.

**Steps:**
1. Time the kernels alone (GPU timestamps, `torch.mps.Event`s around one
   launch, or an Xcode capture), the amax pass apart.
2. A/B variants of `mps/scaled_quant.metal` and its launch only, each held to
   the shipped words by `tests/test_scaled_quant.py`'s MPS cases: 32-bit
   indices when the tensor allows (chosen at launch), or no division (a 2D
   grid); the scale loaded once per thread; then, if needed, the elements per
   thread.
3. Keep a variant only if it gives the same words and is faster in every
   row of the table; record the measurements either way.

**Do not touch** `quantize.metal` (every elementwise quantizer compiles it)
or any header under `common/`. A change confined to `mps/scaled_quant.metal`
and its entry (`mps/quantize.cpp`) moves no Linux object. If suspect 1 holds,
M3's first suspect is the same.

## 6. M3: phase AR, `block_project` on MPS

**The observation** (the last Mac round, 4096 x 4096 float32):
`block_project(x, fmt, p.scales)` onto `block_pack`'s own scales, against
`block_pack(x, fmt)`:

| format | `block_pack` | `block_project` | ratio | RTX 4060 |
|---|---|---|---|---|
| MXFP8_E4M3 | 1.12 ms | 4.72 ms | **4.20x** | 1.09-1.44x |
| MXFP4_E2M1 | 1.11 ms | 2.91 ms | 2.63x | |
| NVFP4 | 1.40 ms | 3.13 ms | 2.23x | |

On the CPU it is 0.81-0.94x `block_pack`; the projection does less than the
pack (no largest magnitude, no scale rounding). The refusal flag's `.item()`
is not the main cause (MXFP8, whose E4M3 element has a NaN code, reads no
flag and is the slowest), but it costs MXFP4 and NVFP4 a device wait per
call.

**Suspects, in this order:**
1. **64-bit divisions per thread.** `mps/block_project.metal` finds its row
   and group with `int64_t(t) / L.per_row` and `g / L.rows`. The ratios
   follow the threads per element: `project_group` gives an 8-bit code a
   group of one, so MXFP8 runs one thread, and its divisions, per element;
   an FP4 code a group of two.
2. **The work per thread:** each thread decodes its block's scale
   (`project_scale`) and builds the casts (`mpt_block_cast()`) for one or two
   elements, where the pack gives a tile of 32 one thread or a threadgroup.
3. **One-byte stores**, one per thread for MXFP8.

**Steps:**
1. Time the kernel alone, apart from the flag's wait and the wrapper.
2. A/B variants of `mps/block_project.metal` and `block_project_mps` (at the
   end of `mps/block_entry.cpp`) only, each held to the shipped words by
   `tests/test_block_ops.py`'s and `tests/test_mps.py`'s MPS cases, SR
   included (draws are keyed per element, so a thread may take more elements
   without moving them): 32-bit indices; a whole block per thread, or a fixed
   number of whole bytes per thread (four MXFP8 codes as one 32-bit store),
   with the scale decoded once.
3. Keep a variant only if it gives the same words and is faster for the
   three formats; record the measurements either way.

**Do not touch** `block.metal`, `common/block_project.h` (the CPU, CUDA and
Meta kernels share it) or any other header under `common/`.

## 7. M4 (optional): the fixed-point GEMM's new layouts on Metal

**Only with the user's go-ahead.** Not in the plan yet; Linux left it out of
T's timing gate.

**Why.** On the host and CUDA, the fixed-point GEMM's casts now take a
layout of their own (one unsigned range test of the word, then a straight
line of integer operations from constants built once per launch; every
other input goes to the shipped cast): under SR
`cast_fixedpoint_stochastic_gemm` and `fixedpoint_round_sr`, under the
deterministic modes `fixedpoint_word_round`. On CUDA they took SR to
0.73-0.82x of its time and the deterministic modes to 0.16-0.78x. Metal
still runs the shipped casts (the new code is behind
`!defined(__METAL_VERSION__)`), and on the M3 Pro the fixed-point SR carrier
path costs 1.23x the binaryK split mac.

**What it would be.** The same code on Metal, made MSL-valid: the constants'
structs and builders, the casts, and the policies' members
(`FixedPointStochasticMember`, `FixedPointWordMember`, `FixedPointIntSrMember`
are `std::conditional_t` with `[[no_unique_address]]`, which MSL 3.1 lacks:
a Metal spelling is needed, for example the members always present on Metal,
since Metal builds the policies on the GPU from literal Args and no host
layout has to agree). All in `common/fixedpoint_gemm.h`, so:
- every Metal-only spelling goes behind `__METAL_VERSION__`, the host and CUDA
  text unchanged;
- keep each mode only where it measures faster on the M3 Pro (`ops --device
  mps`, the base being the commit before the change, alternated), as Linux
  did per side;
- the gate: `tests/test_fixedpoint_mps.py` (257 cases, every mode, the
  subnormal-step formats `FixedPoint(12, 140)` and `FixedPoint(24, 135)`
  included), and the Metal flush rules of section 2.4: the line works on
  words, and its only float input is the value being cast, whose word it
  reads;
- Linux reruns `binary_identity.py` (the host and CUDA objects must not move)
  and merges it **before starting S**.

## 8. M5: what `torch.compile` does on MPS (phase AJ's open question)

**Context.** AJ (Python, unordered) would let Inductor fuse mptorch's casts
into its own kernels, by spelling them as torch ops under compile; on CUDA
that measured 1.7-20x faster than today's compiled code, where each rounding
is an opaque custom op. Its plan lists one unknown: MPS's Inductor backend,
untested.

**The task, an investigation only** (no library change):
- run the compile tests on MPS where they exist (`tests/test_fake_kernels.py`'s
  compile cases, `tests/test_compile_inplace.py`, `tests/test_freeze_packed.py`'s
  compile case) by parametrizing locally, and record what passes, what
  graph-breaks and what fails, with the messages;
- compile a small model on MPS under a recipe (`tests/models.py`'s
  `MiniLlama` under `presets.fp8()`, through `simulate(...).wrap()`), with
  `backend="inductor"` and `"aot_eager"`, and record whether it compiles,
  whether its words equal eager's, and its time against eager;
- report; Linux folds the findings into AJ's plan.

## 9. M6: phase AI's MPS part

**What AI is.** Fused pointwise ops, one per family
(`binaryK_pointwise`, `superfp_pointwise`, `fixedpoint_pointwise`): load the
operands, compute add, sub, mul, div or sqrt in the carrier, round once,
store, in one pass, where phase I's Python ops take two or more. On MPS they
would also round to odd (phase I's ops raise there) and compute the carrier
step without the subnormal flush (through `metal_*_rn`), so MPS results equal
the CPU's word for word, subnormal results included. AI's scope also holds a
fused table evaluation (`binaryK_pwl` and its siblings, for `PWLTable`) and,
if profiles call for it, a dedicated row-reduction kernel for the sums,
softmax and norms. Its design (broadcasting and strides: TensorIterator or a
strided index of mptorch's own) is settled on Linux when the phase starts.

### 9.1 M6a, now: binary32 round-to-odd division and square root on Metal

Round to odd needs division and square root primitives that phase Z's
(`common/carrier_odd_metal.h`: TwoProd, 2Sum, Fast2Sum behind a gate of word
tests, with a `noinline` cold path) do not have. On the host and on Metal the
plan's method is the exact remainder by FMA: for `q = RN(a / b)`,
`r = fma(-q, b, a)` is exact where nothing underflows, and its sign says
whether `a / b` lies above or below `q`; for `s = RN(sqrt(x))`,
`fma(-s, s, x)` does the same. The odd result is `q` when the remainder is
zero, and otherwise `q`'s neighbour toward the exact value if `q`'s code is
even, else `q`.

**The interface** (Linux will match it on the host and CUDA, and confirm it
at SP-AI): in `common/carrier_odd_metal.h`, beside the existing primitives,

```cpp
namespace mptorch::odd {
  float div(float a, float b);   // binary32 round to odd of the exact a / b
  float sqrt(float x);           // binary32 round to odd of the exact sqrt(x)
}
```

with IEEE results for the specials (NaN, infinities, zeros, division by
zero, the root of a negative number), and exact behaviour where a remainder
would underflow (a cold path that rescales on the word, as Z's does), since
Apple GPUs flush subnormal operands of `fma`.

**How to test before AI's ops exist:** a scratch harness, not committed, for
example `torch.mps.compile_shader` over the needed header text, or a local
change to the test-only `_carrier_odd` op's Metal root. Hold both functions
to a reference computed on the CPU (exact rational arithmetic, or float64
with an exactness test) over random pairs in every regime: normal results,
results below 2^-126, operands near the gate's ends, the specials. Report
the method, the counts and the cold path's share.

**Where it lands:** the header is shared, so the hunk goes to Linux at
SP-AI (section 2.1), Metal-only behind `__METAL_VERSION__`.

### 9.2 M6b, at SP-AI: the MPS kernels

After Linux lands AI's ops on the CPU, CUDA and Meta (with `common/pointwise.h`
and `common/pwl.h` written as valid Metal):
- a new root, `mps/pointwise.metal` (and `mps/pwl.metal`), flattened against
  the prelude's headers as `odd.metal` and `fixedpoint.metal` are
  (`setup.py`'s `embed_metal_sources`), with a `KernelSource` each;
- the MPS entries, the CPU entries' text over the MPS backend
  (`tests/test_entry_copies.py` holds the copies token for token), and the
  `mps_ops.cpp` lines; the `_mps` declarations in `quant_ops.h`;
- the carrier step through `metal_mul_rn`/`metal_div_rn`/`metal_sub_rn` (the
  sum through an exact 2Sum or the word, as Linux's design says), so results
  equal the CPU's with subnormals; round to odd from M6a's primitives and
  Z's `odd::add`/`odd::mul`;
- the row-reduction kernel's Metal variant, if Linux builds it;
- tests: `tests/test_mps.py` parity over every op, mode, dtype and the
  subnormal-heavy inputs, and the MPS cases of AI's own tests, which the
  Linux commit adds skipping without an MPS kernel;
- timing: the fused ops against phase I's ops on MPS
  (`dev/benchmarks/timing/pointwise_ops.py --device mps`, which gains a
  fused variant in AI).

Estimate on the Linux plan: 1-1.5 d, plus the table evaluation's Metal part.

## 10. M7, at SP-AK: the NEON palette tiles

**What AK is.** On the CPU a palette GEMM (a format per output element, or
per neuron in a recipe) runs the scalar loop, because phase P's vector RNE
tiles (`cpu/gemm_rne_tile.cpp`) serve single-format macs only; it costs
9-15x its single-format twin on the Ryzen. AK gives palettes vector tiles,
NEON included, with results word for word the scalar loop's.

**The Mac's part** (as for P and AA, whose NEON code Linux wrote and the Mac
measured):
- build, `tests/test_cpu_simd.py` under each `MPTORCH_CPU_SIMD` setting the
  test covers on arm64, the palette tests (`test_gemm_mixed_palette.py`,
  `test_palette_recipes.py`, `test_qmatmul.py`'s palette tier);
- the tile sweep Linux adds (in `cpu_rne_tile_sweep.cpp`'s manner) under
  Apple clang, as section 4.3 builds the others;
- timing with `dev/benchmarks/timing/cpu_gemm_ab.cpp` on 1 and 11 threads
  against the scalar loop and against the single-format twin;
- NEON-specific layout choices, if the M3 Pro prefers one (AA kept NEON's
  product-only and fused tiles row by row, `tile_rows`, because passes over
  the whole tile cost the M3 Pro 2-4%): measure, report, and propose; the
  tile file is Linux's, so a NEON-only change goes to Linux as a hunk.

## 11. M8, at SP-S: the counting variant of `gemm.metal`

**What S is.** Numeric event counters: opt-in GEMM and conv ops
(`MPTORCH_COUNTING=1`) that count, per rounding role (a split mac's product,
its sum, ...), events such as underflow to zero, saturation and subnormal
results, reported per layer through `simulate` and `profile`. Linux builds
the policy wrappers (`common/event_counts.h`, valid Metal), the CPU and CUDA
kernels and the Python.

**The Mac's part** (the plan's decision S-d: parity):
- `mps/gemm.metal` under `MPT_COUNTING`: each thread's counts reduced over
  its threadgroup into a per-threadgroup row of a partial-counts buffer
  (Metal has no 64-bit atomics on every GPU), summed by the entry point;
- the MPS entries of the counted ops and their `mps_ops.cpp` lines;
- tests: results **and** counts equal the CPU's, in every family and mode
  the counted ops cover;
- this edit changes `gemm.metal`'s text, so every GEMM kernel recompiles
  once; fold into it the comment the last round deferred for that reason
  (its header comment, naming the fixed-point ops among those it runs).

Estimate on the Linux plan: 1 d plus the round.

## 12. M9, at SP-AL: flash attention

**What AL is.** Phase AE's flash attention (FlashAttention-2's order, tile by
tile, in Python) is 5-8x the standard path's time on CUDA, because its tile
loop is launch-bound. AL first measures `torch.compile` of it, then batches
the query tiles of each step where the arithmetic cannot tell (every slot
elementwise, nothing stochastic), and only then, if needed, writes a fused
kernel for a tile's online-softmax steps on CUDA, the CPU and Metal.

**The Mac's part:** the round after steps 1 and 2 (`tests/test_flash_attention.py`
on MPS, which holds every path word for word to an independent FA2;
timings of the standard path, the loop and the batched path on MPS, eager
and, after M5, compiled). If Linux builds step 4's ops, their Metal kernels
follow M6b's pattern.

## 13. Later, when their phases are refined

None of these is planned in detail; each will get its own brief:
- **AN** (integer operands for the fixed-point GEMM): a code-tile branch in
  the Metal GEMM kernel, beside the CPU's and CUDA's.
- **AO** (CPU vector tiles for the fixed-point GEMM): NEON, as M7.
- **AP** (a fast RNE path for the elementwise fixed-point quantizer): the
  Metal quantizer (`quantize.metal`, a root: cold cache).
- **AM** (exact accumulation of float products in a wide fixed-point
  accumulator): possibly a 128-bit accumulator, which Metal would emulate
  further or refuse.

## 14. What to report, per task

- the commit the task ran at, the machine and its state (AC power, thermal);
- each command, and the lines of its output that the task's bar names;
- the measurements, with the method (alternation, rounds, minimum);
- every file changed, and for the shared ones (section 2.1) the hunks;
- what failed and what was done about it (diagnose before changing: which
  elements differ, by how much, what the inputs and intermediates are);
- what is left open, and which sync point it waits on.
