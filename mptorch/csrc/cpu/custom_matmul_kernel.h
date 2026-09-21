#pragma once

// The CPU GEMM kernel and CpuBackend::launch_as, the template that the eight
// custom_matmul_*.cpp translation units explicitly instantiate: one per
// (format family x mac mode) in the binary32 carrier, and its *_f64.cpp twin
// in binary64. Nothing else includes this header. The kernel body is heavy
// to compile (one instantiation per Accumulator policy per round mode), and
// with every instantiation in one object that object took about 67 s on its
// own and was the build's critical path; over eight objects the largest is
// about 31 s and ninja compiles them side by side. The entry points live in
// cpu/custom_matmul_entry.cpp, which sees only cpu/gemm_backend.h and so
// instantiates no kernel. cuda/custom_matmul_kernel.cuh is split the same
// way. Six more files, custom_matmul_*_{kahan,block,tree}.cpp, instantiate
// launch_accumulated_as for the accumulate algorithms past NAIVE
// (common/gemm_accumulate.h), in binary32.

#include "../common/gemm_accumulate.h"
#include "../common/gemm_args.h"
#include "../common/gemm_policy.h"
#include "../common/modes.h"
#include "gemm_backend.h"
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h> // at::internal::GRAIN_SIZE lives here
#include <algorithm>

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::GemmShape;

  // Tile packing. The operands' storage dtype is a runtime tag
  // (mptorch::GemmDtype, common/gemm_dtype.h) rather than a template
  // parameter of the kernel, so the kernel body is instantiated once per
  // carrier rather than once per dtype. On the GPU the tag is a switch
  // inside each load; here a switch in the innermost loop would cost a
  // branch per multiply-add, so the kernel does what a BLAS does instead:
  // each 32x32 operand tile is converted into a buffer of the carrier type
  // once, and the K-loop runs over those buffers with no dtype in sight.
  // Three further consequences, all wanted: every element converts once per
  // tile instead of once per K-step of every row (a c10::Half to float
  // conversion is a function call on the host, not an instruction); both
  // operands are stride-1 in the innermost loop whatever trans_a and trans_b
  // say, since the transposition is resolved by the pack's index
  // arithmetic; and only these small routines are templated on the storage
  // dtype. The accumulate order is unchanged by the packing (same values,
  // same sequence of Accumulator::accumulate calls), so results are
  // bit-identical to an unpacked loop.
  //
  // pack_a_impl copies the ti x tk block of op(A) at (i0, k0) into dst, row
  // major with row stride tk; pack_b_impl copies the tk x tj block of op(B)
  // at (k0, j0) into dst with row stride tj; store_tile_impl writes the
  // ti x tj block of C at (i0, j0) from src, converting back to the storage
  // dtype. `off` is the batch element's base offset into the operand, in
  // elements: 0 for a 2D call and for an operand broadcast across the batch.
  // It is added to the base pointer once per tile, not per element. T is
  // the carrier the tile is packed in (common/gemm_policy.h): float for
  // float32, float16 and bfloat16 operands, double for float64, whose pack
  // is a plain copy.
  template <typename scalar_t, class T>
  void pack_a_impl(const void *A, T *__restrict__ dst, int64_t M, int64_t K, bool trans_a,
                   int64_t i0, int64_t k0, int64_t ti, int64_t tk, int64_t off)
  {
    const scalar_t *__restrict__ src = static_cast<const scalar_t *>(A) + off;
    for (int64_t i = 0; i < ti; ++i)
      for (int64_t k = 0; k < tk; ++k)
        dst[i * tk + k] = trans_a ? static_cast<T>(src[(k0 + k) * M + (i0 + i)])
                                  : static_cast<T>(src[(i0 + i) * K + (k0 + k)]);
  }

  template <typename scalar_t, class T>
  void pack_b_impl(const void *B, T *__restrict__ dst, int64_t K, int64_t N, bool trans_b,
                   int64_t k0, int64_t j0, int64_t tk, int64_t tj, int64_t off)
  {
    const scalar_t *__restrict__ src = static_cast<const scalar_t *>(B) + off;
    for (int64_t k = 0; k < tk; ++k)
      for (int64_t j = 0; j < tj; ++j)
        dst[k * tj + j] = trans_b ? static_cast<T>(src[(j0 + j) * K + (k0 + k)])
                                  : static_cast<T>(src[(k0 + k) * N + (j0 + j)]);
  }

  template <typename scalar_t, class T>
  void store_tile_impl(void *C, const T *__restrict__ src, int64_t N,
                       int64_t i0, int64_t j0, int64_t ti, int64_t tj, int64_t off)
  {
    scalar_t *__restrict__ dst = static_cast<scalar_t *>(C) + off;
    for (int64_t i = 0; i < ti; ++i)
      for (int64_t j = 0; j < tj; ++j)
        dst[(i0 + i) * N + (j0 + j)] = static_cast<scalar_t>(src[i * tj + j]);
  }

  // Expands CALL(scalar_t) for the storage dtype named by DT. Used once per
  // tile per operand and never inside the K-loop, so the switch stays off
  // the hot path.
#define MPTORCH_GEMM_BY_DTYPE(DT, CALL)      \
  switch (DT)                                \
  {                                          \
  case mptorch::GemmDtype::Half:             \
    CALL(at::Half);                          \
    break;                                   \
  case mptorch::GemmDtype::BFloat16:         \
    CALL(at::BFloat16);                      \
    break;                                   \
  default:                                   \
    CALL(float);                             \
    break;                                   \
  }

  // Dispatch from the runtime dtype tag to the packing templates. The
  // primary templates are binary32's, over its three storage dtypes; the
  // double specializations below skip the switch, since binary64 has one
  // storage dtype and the tag is already known to be Double there.
  template <class T>
  inline void pack_a(const void *A, mptorch::GemmDtype dt, T *dst, int64_t M, int64_t K,
                     bool trans_a, int64_t i0, int64_t k0, int64_t ti, int64_t tk, int64_t off)
  {
#define MPTORCH_PACK_A(S) pack_a_impl<S, T>(A, dst, M, K, trans_a, i0, k0, ti, tk, off)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_PACK_A)
#undef MPTORCH_PACK_A
  }

  template <class T>
  inline void pack_b(const void *B, mptorch::GemmDtype dt, T *dst, int64_t K, int64_t N,
                     bool trans_b, int64_t k0, int64_t j0, int64_t tk, int64_t tj, int64_t off)
  {
#define MPTORCH_PACK_B(S) pack_b_impl<S, T>(B, dst, K, N, trans_b, k0, j0, tk, tj, off)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_PACK_B)
#undef MPTORCH_PACK_B
  }

  template <class T>
  inline void store_tile(void *C, mptorch::GemmDtype dt, const T *src, int64_t N,
                         int64_t i0, int64_t j0, int64_t ti, int64_t tj, int64_t off)
  {
#define MPTORCH_STORE_TILE(S) store_tile_impl<S, T>(C, src, N, i0, j0, ti, tj, off)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_STORE_TILE)
#undef MPTORCH_STORE_TILE
  }

#undef MPTORCH_GEMM_BY_DTYPE

  template <>
  inline void pack_a<double>(const void *A, mptorch::GemmDtype, double *dst, int64_t M, int64_t K,
                             bool trans_a, int64_t i0, int64_t k0, int64_t ti, int64_t tk,
                             int64_t off)
  {
    pack_a_impl<double, double>(A, dst, M, K, trans_a, i0, k0, ti, tk, off);
  }

  template <>
  inline void pack_b<double>(const void *B, mptorch::GemmDtype, double *dst, int64_t K, int64_t N,
                             bool trans_b, int64_t k0, int64_t j0, int64_t tk, int64_t tj,
                             int64_t off)
  {
    pack_b_impl<double, double>(B, dst, K, N, trans_b, k0, j0, tk, tj, off);
  }

  template <>
  inline void store_tile<double>(void *C, mptorch::GemmDtype, const double *src, int64_t N,
                                 int64_t i0, int64_t j0, int64_t ti, int64_t tj, int64_t off)
  {
    store_tile_impl<double, double>(C, src, N, i0, j0, ti, tj, off);
  }

  // Cache-tiled batched GEMM: C[b] = op(A[b]) @ op(B[b]) for each batch
  // element b, where op(X) is X.T when the matching trans flag is set. A is
  // [M, K] ([K, M] when trans_a), B is [K, N] ([N, K] when trans_b), and C
  // is written densely as [batch, M, N]. stride_a/stride_b are the per-batch
  // element strides into A and B, where 0 broadcasts that operand across the
  // batch. A and B are read through their own storage layout; the trans
  // flags only change the index arithmetic, no transpose is materialized.
  //
  // The Accumulator owns both the multiply and the accumulate step through
  // its Mac policy (common/gemm_policy.h), so this kernel only ever calls
  // seed_rng()/accumulate()/finalize() and never multiplies operands itself.
  // The per-output-element reduction state (a running sum and, for SR, a
  // Philox stream) lives in the Accumulator's tile type as dense arrays
  // rather than in one Accumulator object per element, so what the K-loop
  // strides through is a plain array of carrier values. use_rng/seed drive
  // RoundMode::SR: when set, each output element's stream is seeded before
  // its K-reduction starts, keyed by its global linear index into
  // [batch, M, N], so the draws it sees depend on neither the tiling nor the
  // thread count.
  //
  // MIXED selects the spatially-varying-format instantiation, in which each
  // output element's Mac policy is looked up in a FormatPalette through
  // prec_idx[b, i, j] (read with the three idx_* element strides). It is a
  // template parameter rather than a runtime "is there a palette?" test
  // because the single-format path wants its one Mac held in registers for
  // the whole call, and a runtime test would turn that into a per-element
  // pointer load in the innermost loop. The single-format instantiation
  // takes no palette at all: PaletteArg<false, Mac> is the empty NoPalette.
  template <bool MIXED = false, class Accumulator>
  void matmul_cpu_kernel_impl(const void *A, const void *B, void *C, mptorch::GemmDtype dt,
                              int64_t M, int64_t K, int64_t N,
                              bool trans_a, bool trans_b,
                              int64_t batch, int64_t stride_a, int64_t stride_b,
                              Accumulator acc_proto,
                              bool use_rng, uint64_t seed,
                              PaletteArg<MIXED, typename Accumulator::mac_type> pal = {},
                              const int32_t *__restrict__ prec_idx = nullptr,
                              int64_t idx_row_stride = 0, int64_t idx_col_stride = 0,
                              int64_t idx_batch_stride = 0)
  {
    constexpr int64_t TI = 32, TJ = 32, TK = 32;
    using Mac = typename Accumulator::mac_type;
    using T = typename Accumulator::value_t; // the carrier (common/gemm_policy.h)

    const int64_t n_tiles_i = (M + TI - 1) / TI;
    const int64_t n_tiles_j = (N + TJ - 1) / TJ;
    const int64_t n_tiles = n_tiles_i * n_tiles_j;

    // The parallel axis is the whole (i0, j0) output tile grid, not just its
    // rows: splitting on M alone would give ceil(M / TI) tasks, so a typical
    // QLinear batch (M = 32) would run on one thread however many are
    // available. Output tiles are disjoint and each element's K-reduction
    // stays sequential inside one task, so results are independent of the
    // thread count, RoundMode::SR included, since its stream is keyed by the
    // output element's global linear index (NaiveTile::seed_rng) rather than
    // by the tiling. The grain is the number of tiles that add up to one of
    // ATen's default grains of elementwise work, so a small GEMM is not cut
    // finer than the thread pool's dispatch cost is worth.
    //
    // at::parallel_for rather than a raw `#pragma omp parallel for`: it runs
    // on ATen's own thread pool, so torch.set_num_threads applies, and it
    // degrades to serial inside an enclosing parallel region instead of
    // nesting. Under the AT_PARALLEL_OPENMP backend it still needs setup.py's
    // -fopenmp, because parallel_for is a header template whose `#pragma omp
    // parallel` is compiled in this translation unit; without the flag the
    // loop is silently serial.
    const int64_t work_per_tile = TI * TJ * std::max<int64_t>(K, 1);
    const int64_t grain = std::max<int64_t>(1, at::internal::GRAIN_SIZE / work_per_tile);

    // The batch is on the same axis as the tiles rather than a loop around
    // them: the tasks are (batch element, output tile) pairs, so a batched
    // call with a small M*N per element (an attention head's 128x128 output
    // is 16 tiles) still has work for every thread, where a loop of 2D calls
    // would leave most threads idle on each call. batch = 1 is the plain 2D
    // decomposition.
    at::parallel_for(0, batch * n_tiles, grain, [&](int64_t task_begin, int64_t task_end)
    {
      // One tile's worth of reduction state per worker task, reused across
      // every tile the task takes rather than rebuilt per tile: TI*TJ running
      // sums and TI*TJ Philox streams (gemm_policy.h's NaiveTile; the streams
      // are allocated for every round mode and only seeded and drawn under
      // SR, which keeps the tile type one type). The Mac policy is not part
      // of it: one copy for the whole call below, or a pointer per element
      // into the palette on the mixed path.
      typename Accumulator::tile_type tile;
      tile.resize(TI * TJ);
      const auto tv = tile.view();

      // The single-format path's one Mac, held by value in the task's frame
      // rather than re-read per output element. By value and not through a
      // reference to the caller's acc_proto: the reference measured 5-8%
      // slower on the split macs, since the compiler could not keep the
      // policy's fields in registers across the loop.
      const Mac mac_single = acc_proto.mac;
      // A positional accumulator (KAHAN, BLOCK, TREE;
      // common/gemm_accumulate.h) is configured by more than its Mac, a block
      // size and an outer format, so its tile functions take the prototype
      // itself. Every use of it is behind `if constexpr`, and it is read
      // through the caller's acc_proto rather than through a second local:
      // even an empty local of a NaiveAccumulator's moved the register
      // allocation of two of that kernel's existing instantiations, and the
      // NAIVE objects are held byte-identical (dev/gemm_roadmap.md, R-2).
      constexpr bool POSITIONAL = is_positional_accumulator_v<Accumulator>;
      const Mac *slot_of[MIXED ? TI * TJ : 1]; // the mixed path's per-element slots

      // Packed operand tiles and the finalized output tile, one set per
      // worker task and reused across every tile it takes: 12 KB of frame in
      // binary32 (24 KB in binary64) against a conversion per K-step saved.
      // 64-byte aligned so the innermost loop's reads of b_pack start on a
      // cache line.
      alignas(64) T a_pack[TI * TK];
      alignas(64) T b_pack[TK * TJ];
      alignas(64) T c_pack[TI * TJ];

      for (int64_t task_id = task_begin; task_id < task_end; ++task_id)
      {
        const int64_t bId = task_id / n_tiles;    // batch element of this task
        const int64_t tile_id = task_id % n_tiles; // output tile within it
        const int64_t a_off = bId * stride_a;
        const int64_t b_off = bId * stride_b;
        const int64_t c_off = bId * M * N;
        const int64_t i0 = (tile_id / n_tiles_j) * TI;
        const int64_t j0 = (tile_id % n_tiles_j) * TJ;
        const int64_t ti = std::min<int64_t>(TI, M - i0);
        const int64_t tj = std::min<int64_t>(TJ, N - j0);
        tile.begin(ti * tj);

        // Each element's SR stream is keyed by its global linear index into
        // the whole [batch, M, N] output, so batch element 0 of a batched
        // call draws exactly what the 2D call draws, and no two elements
        // share a subsequence.
        if (use_rng)
        {
          for (int64_t i = 0; i < ti; ++i)
            for (int64_t j = 0; j < tj; ++j)
              tile.seed_rng(i * tj + j, seed,
                            static_cast<uint64_t>(c_off + (i0 + i) * N + (j0 + j)));
        }

        // Mixed format: resolve each output element's Mac policy from the
        // palette (gemm_policy.h's FormatPalette) before its K-reduction.
        // Pointers into the palette rather than copies of it: the palette
        // outlives the call and an element's slot never changes
        // mid-reduction, so a copy would only cost frame space.
        if constexpr (MIXED)
        {
          for (int64_t i = 0; i < ti; ++i)
            for (int64_t j = 0; j < tj; ++j)
              slot_of[i * tj + j] =
                  &pal.slot(prec_idx[bId * idx_batch_stride + (i0 + i) * idx_row_stride +
                                     (j0 + j) * idx_col_stride]);
        }

        for (int64_t k0 = 0; k0 < K; k0 += TK)
        {
          int64_t tk = std::min<int64_t>(TK, K - k0);
          pack_a(A, dt, a_pack, M, K, trans_a, i0, k0, ti, tk, a_off);
          pack_b(B, dt, b_pack, K, N, trans_b, k0, j0, tk, tj, b_off);
          for (int64_t i = 0; i < ti; ++i)
          {
            for (int64_t k = 0; k < tk; ++k)
            {
              const T aVal = a_pack[i * tk + k];
              const T *__restrict__ b_row = b_pack + k * tj;
              if constexpr (POSITIONAL)
              {
                // The position in the 16-step slab the algorithms count in
                // (the device's tile depth, which is theirs on every
                // backend), and the slab's end after step 15 or the last.
                const int64_t k_end = k0 + k + 1;
                const int pos = static_cast<int>((k0 + k) & 15);
                for (int64_t j = 0; j < tj; ++j)
                  Accumulator::accumulate(tv, i * tj + j, acc_proto, aVal, b_row[j], pos);
                if ((k_end & 15) == 0 || k_end == K)
                  for (int64_t j = 0; j < tj; ++j)
                    Accumulator::end_slab(tv, i * tj + j, acc_proto, k_end, K);
              }
              else
              {
                for (int64_t j = 0; j < tj; ++j)
                {
                  const int64_t idx = i * tj + j;
                  Accumulator::accumulate(tv, idx, MIXED ? *slot_of[MIXED ? idx : 0] : mac_single,
                                          aVal, b_row[j]);
                }
              }
            }
          }
        }

        for (int64_t i = 0; i < ti; ++i)
          for (int64_t j = 0; j < tj; ++j)
            c_pack[i * tj + j] = tile.finalize(i * tj + j);
        store_tile(C, dt, c_pack, N, i0, j0, ti, tj, c_off);
      }
    });
  }

  // One body for all eight ops in either carrier T (float or double): the
  // Args type names the Accumulator policy, which is the only thing that
  // differs between the ops, and the round mode is resolved to a template
  // parameter here so the accumulate step compiles to straight-line code
  // rather than a switch per multiply-add. Declared in cpu/gemm_backend.h
  // and explicitly instantiated in the eight custom_matmul_*.cpp files.
  template <class T, class Args>
  void CpuBackend::launch_as(const GemmShape &s, const Args &args, const LaunchContext &ctx)
  {
    mptorch::dispatch_round_mode(s.rm, [&](auto rm_c)
    {
      constexpr RoundMode RM = decltype(rm_c)::value;
      if constexpr (Args::mixed)
      {
        args.template with_palette<T, RM>([&](auto acc, const auto &pal)
        {
          matmul_cpu_kernel_impl<true>(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b,
                                       s.batch, s.stride_a, s.stride_b,
                                       acc, s.use_rng, ctx.seed, pal, s.prec_idx,
                                       s.idx_row_stride, s.idx_col_stride, s.idx_batch_stride);
        });
      }
      else
      {
        args.template with_accumulator<T, RM>([&](auto acc)
        {
          matmul_cpu_kernel_impl(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b,
                                 s.batch, s.stride_a, s.stride_b, acc, s.use_rng, ctx.seed);
        });
      }
    });
  }

  // The same body for KAHAN, BLOCK and TREE, whose Args also names the
  // algorithm its accumulator is built for. Instantiated by the six
  // custom_matmul_{binaryK,superfp}_{kahan,block,tree}.cpp files.
  template <class T, AccumulateAlgorithm ALG, class Base>
  void CpuBackend::launch_accumulated_as(const GemmShape &s,
                                         const mptorch::gemm::AccumulateArgs<Base> &args,
                                         const LaunchContext &ctx)
  {
    mptorch::dispatch_round_mode(s.rm, [&](auto rm_c)
    {
      constexpr RoundMode RM = decltype(rm_c)::value;
      args.template with_accumulator<T, RM, ALG>([&](auto acc)
      {
        matmul_cpu_kernel_impl(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b,
                               s.batch, s.stride_a, s.stride_b, acc, s.use_rng, ctx.seed);
      });
    });
  }
} // namespace mptorch::gemm_cpu
