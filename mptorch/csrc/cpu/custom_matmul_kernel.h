#pragma once

// The CPU GEMM kernel and the host-side prologue every entry point repeats --
// shared by the four custom_matmul_*.cpp translation units that hold the eight
// entry points themselves. The CUDA twin (cuda/custom_matmul_kernel.cuh) is
// split the same way and for the same reasons; see its comment and
// dev/gemm_roadmap.md (finding B2). This file was the build's second-longest
// TU, 92.7 s of a 111.2 s build (67.1 s compiled on its own, against 30.8 s
// for the largest of the four that replaced it).

#include "../common/dispatch.h"
#include "../common/gemm_policy.h"
#include "../common/modes.h"
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h> // at::internal::GRAIN_SIZE lives here
#include <ATen/ops/aminmax.h>
#include <algorithm>
#include <initializer_list>

namespace mptorch::gemm_cpu
{
  // Confined to this namespace, rather than dumped into the file scope of
  // everything that includes this header.
  using namespace at;

  // The operands' storage dtype is a runtime argument rather than a template
  // parameter of the kernel (mptorch::GemmDtype, common/dispatch.h -- finding
  // K2). On the GPU that is a switch inside the load; here it would be a
  // switch in the innermost loop, which is not free, so the kernel does what
  // a BLAS does instead and packs each 32x32 operand tile into a float buffer
  // once, then runs a dtype-free K-loop over packed floats. Three consequences
  // beyond the instantiation cut, all of them wanted:
  //
  //   * every element converts once per tile instead of once per K-step of
  //     every row -- today's inner loop re-reads and re-converts B[k, j] for
  //     each of the TI values of i, and c10::Half -> float is a real function
  //     on the host, not an instruction;
  //   * both operands are stride-1 in the innermost loop whatever trans_a and
  //     trans_b say, and the trans_* branches leave that loop entirely;
  //   * only these little pack routines stay templated on the storage dtype.
  //
  // The accumulate order is untouched -- same floats, same sequence of
  // Accumulator::accumulate calls -- so the results are bit-identical.
  template <typename scalar_t>
  void pack_a_impl(const void *A, float *__restrict__ dst, int64_t M, int64_t K, bool trans_a,
                   int64_t i0, int64_t k0, int64_t ti, int64_t tk)
  {
    const scalar_t *__restrict__ src = static_cast<const scalar_t *>(A);
    for (int64_t i = 0; i < ti; ++i)
      for (int64_t k = 0; k < tk; ++k)
        dst[i * tk + k] = trans_a ? static_cast<float>(src[(k0 + k) * M + (i0 + i)])
                                  : static_cast<float>(src[(i0 + i) * K + (k0 + k)]);
  }

  template <typename scalar_t>
  void pack_b_impl(const void *B, float *__restrict__ dst, int64_t K, int64_t N, bool trans_b,
                   int64_t k0, int64_t j0, int64_t tk, int64_t tj)
  {
    const scalar_t *__restrict__ src = static_cast<const scalar_t *>(B);
    for (int64_t k = 0; k < tk; ++k)
      for (int64_t j = 0; j < tj; ++j)
        dst[k * tj + j] = trans_b ? static_cast<float>(src[(j0 + j) * K + (k0 + k)])
                                  : static_cast<float>(src[(k0 + k) * N + (j0 + j)]);
  }

  template <typename scalar_t>
  void store_tile_impl(void *C, const float *__restrict__ src, int64_t N,
                       int64_t i0, int64_t j0, int64_t ti, int64_t tj)
  {
    scalar_t *__restrict__ dst = static_cast<scalar_t *>(C);
    for (int64_t i = 0; i < ti; ++i)
      for (int64_t j = 0; j < tj; ++j)
        dst[(i0 + i) * N + (j0 + j)] = static_cast<scalar_t>(src[i * tj + j]);
  }

  // One switch per tile per operand, not one per element: MPTORCH_GEMM_BY_DTYPE
  // is deliberately not a loop-body construct.
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

  inline void pack_a(const void *A, mptorch::GemmDtype dt, float *dst, int64_t M, int64_t K,
                     bool trans_a, int64_t i0, int64_t k0, int64_t ti, int64_t tk)
  {
#define MPTORCH_PACK_A(T) pack_a_impl<T>(A, dst, M, K, trans_a, i0, k0, ti, tk)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_PACK_A)
#undef MPTORCH_PACK_A
  }

  inline void pack_b(const void *B, mptorch::GemmDtype dt, float *dst, int64_t K, int64_t N,
                     bool trans_b, int64_t k0, int64_t j0, int64_t tk, int64_t tj)
  {
#define MPTORCH_PACK_B(T) pack_b_impl<T>(B, dst, K, N, trans_b, k0, j0, tk, tj)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_PACK_B)
#undef MPTORCH_PACK_B
  }

  inline void store_tile(void *C, mptorch::GemmDtype dt, const float *src, int64_t N,
                         int64_t i0, int64_t j0, int64_t ti, int64_t tj)
  {
#define MPTORCH_STORE_TILE(T) store_tile_impl<T>(C, src, N, i0, j0, ti, tj)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_STORE_TILE)
#undef MPTORCH_STORE_TILE
  }

#undef MPTORCH_GEMM_BY_DTYPE

  // Cache-tiled MxKxN GEMM: C = op(A) @ op(B), where op(X) = X.T if the
  // corresponding trans flag is set. A/B are always read through their own
  // (untransposed) storage layout -- trans_a/trans_b only change the index
  // arithmetic used to read them, no physical transpose is materialized.
  // Accumulator owns both the multiply and the accumulate step (via its Mac
  // policy -- see gemm_policy.h), so this kernel only ever calls
  // seed_rng()/accumulate()/finalize() and never multiplies operands itself.
  // The per-output-element reduction state lives in the Accumulator's tile
  // type rather than in one Accumulator per element (finding C3), so what
  // the K-loop strides through is a dense array of floats. use_rng/seed
  // drive RoundMode::SR: when set, each output element's stream is seeded
  // (keyed by its own global linear index, so the result is independent of
  // tiling and thread count) before its K-reduction starts.

  // MIXED selects whether this instantiation carries the spatially-varying
  // FormatPalette prologue, exactly as it does on the GPU (finding G4) and
  // for the same reason: with the Mac no longer stored per output element
  // (finding C3), the single-format path wants one Mac held in registers for
  // the whole call, which a runtime `pal.n > 0` test would turn back into a
  // per-element pointer load in the innermost loop.
  template <bool MIXED = false, class Accumulator>
  void matmul_cpu_kernel_impl(const void *A, const void *B, void *C, mptorch::GemmDtype dt,
                              int64_t M, int64_t K, int64_t N,
                              bool trans_a, bool trans_b,
                              Accumulator acc_proto,
                              bool use_rng, uint64_t seed,
                              FormatPalette<typename Accumulator::mac_type> pal = {},
                              const int32_t *__restrict__ prec_idx = nullptr,
                              int64_t idx_row_stride = 0, int64_t idx_col_stride = 0)
  {
    constexpr int64_t TI = 32, TJ = 32, TK = 32;
    using Mac = typename Accumulator::mac_type;

    const int64_t n_tiles_i = (M + TI - 1) / TI;
    const int64_t n_tiles_j = (N + TJ - 1) / TJ;
    const int64_t n_tiles = n_tiles_i * n_tiles_j;

    // The parallel axis is the whole (i0, j0) tile grid, not just its rows:
    // splitting on M alone yields only ceil(M / TI) tasks, so a typical
    // QLinear batch (M = 32) would run on a single thread no matter how many
    // are available. Output tiles are disjoint and each element's K-reduction
    // stays sequential inside one task, so results are independent of the
    // thread count -- RoundMode::SR included, since its stream is keyed by
    // the output element's global linear index (NaiveAccumulator::seed_rng)
    // rather than by tiling. at::parallel_for (rather than a raw
    // `#pragma omp parallel for`) runs on ATen's own thread pool, so
    // torch.set_num_threads applies, and it degrades to serial inside an
    // enclosing parallel region instead of nesting. Note it still needs
    // setup.py's -fopenmp under the AT_PARALLEL_OPENMP backend: ATen's
    // parallel_for is a header template whose `#pragma omp parallel` is
    // inlined here, so without the flag this loop is silently serial -- which
    // is exactly how the `#ifdef _OPENMP`-guarded pragma this replaced ended
    // up never running in parallel at all.
    const int64_t work_per_tile = TI * TJ * std::max<int64_t>(K, 1);
    const int64_t grain = std::max<int64_t>(1, at::internal::GRAIN_SIZE / work_per_tile);

    at::parallel_for(0, n_tiles, grain, [&](int64_t tile_begin, int64_t tile_end)
    {
      // One tile's worth of reduction state per worker task, reused across
      // every tile the task takes rather than rebuilt per tile: TI*TJ running
      // sums and TI*TJ Philox streams (gemm_policy.h's NaiveTile, finding
      // C3 -- see there for why the streams are allocated even for the six
      // round modes that never draw). The Mac policy stays out of it: one
      // copy for the whole call below, or a pointer per element into the
      // palette on the mixed path.
      typename Accumulator::tile_type tile;
      tile.resize(TI * TJ);
      const auto tv = tile.view();

      // The single-format path's one Mac, held by value in the task's frame
      // rather than re-read per output element. By value and not by
      // reference: taking a reference to the caller's acc_proto measured
      // 5-8% slower on the split macs.
      const Mac mac_single = acc_proto.mac;
      const Mac *slot_of[MIXED ? TI * TJ : 1]; // the mixed path's per-element slots

      // Packed tiles and the finalized output tile, one set per worker task
      // and reused across every tile it takes: 12 KB of frame against a
      // conversion per K-step saved. 64-byte aligned so the innermost loop's
      // reads of b_pack start on a cache line.
      alignas(64) float a_pack[TI * TK];
      alignas(64) float b_pack[TK * TJ];
      alignas(64) float c_pack[TI * TJ];

      for (int64_t tile_id = tile_begin; tile_id < tile_end; ++tile_id)
      {
        const int64_t i0 = (tile_id / n_tiles_j) * TI;
        const int64_t j0 = (tile_id % n_tiles_j) * TJ;
        const int64_t ti = std::min<int64_t>(TI, M - i0);
        const int64_t tj = std::min<int64_t>(TJ, N - j0);
        tile.begin(ti * tj);

        if (use_rng)
        {
          for (int64_t i = 0; i < ti; ++i)
            for (int64_t j = 0; j < tj; ++j)
              tile.seed_rng(i * tj + j, seed, static_cast<uint64_t>((i0 + i) * N + (j0 + j)));
        }

        // Spatially-varying mixed format: resolve each output element's Mac
        // policy from the palette before its K-reduction, in the same place
        // and the same order the whole slot used to be copied into that
        // element's accumulator (see gemm_policy.h's FormatPalette). Pointers
        // into the palette rather than copies of it: the palette outlives the
        // call and an element's slot never changes mid-reduction.
        if constexpr (MIXED)
        {
          for (int64_t i = 0; i < ti; ++i)
            for (int64_t j = 0; j < tj; ++j)
              slot_of[i * tj + j] =
                  &pal.slot(prec_idx[(i0 + i) * idx_row_stride + (j0 + j) * idx_col_stride]);
        }

        for (int64_t k0 = 0; k0 < K; k0 += TK)
        {
          int64_t tk = std::min<int64_t>(TK, K - k0);
          pack_a(A, dt, a_pack, M, K, trans_a, i0, k0, ti, tk);
          pack_b(B, dt, b_pack, K, N, trans_b, k0, j0, tk, tj);
          for (int64_t i = 0; i < ti; ++i)
          {
            for (int64_t k = 0; k < tk; ++k)
            {
              const float aVal = a_pack[i * tk + k];
              const float *__restrict__ b_row = b_pack + k * tj;
              for (int64_t j = 0; j < tj; ++j)
              {
                const int64_t idx = i * tj + j;
                Accumulator::accumulate(tv, idx, MIXED ? *slot_of[MIXED ? idx : 0] : mac_single,
                                        aVal, b_row[j]);
              }
            }
          }
        }

        for (int64_t i = 0; i < ti; ++i)
          for (int64_t j = 0; j < tj; ++j)
            c_pack[i * tj + j] = tile.finalize(i * tj + j);
        store_tile(C, dt, c_pack, N, i0, j0, ti, tj);
      }
    });
  }

  inline void check_matmul_inputs(const Tensor &a, const Tensor &b, const char *op_name,
                           int64_t round_mode, int64_t accumulate_algorithm)
  {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, op_name, " expects 2D tensors, got ",
               a.dim(), "D and ", b.dim(), "D");
    TORCH_CHECK(static_cast<AccumulateAlgorithm>(accumulate_algorithm) == AccumulateAlgorithm::NAIVE,
               op_name, ": only AccumulateAlgorithm.NAIVE is supported in this build");
  }

  inline void matmul_output_shape(const Tensor &a, const Tensor &b, bool trans_a, bool trans_b,
                           const char *op_name, int64_t &M, int64_t &K, int64_t &N)
  {
    M = trans_a ? a.size(1) : a.size(0);
    K = trans_a ? a.size(0) : a.size(1);
    int64_t K_b = trans_b ? b.size(1) : b.size(0);
    N = trans_b ? b.size(0) : b.size(1);
    TORCH_CHECK(K == K_b, op_name, ": inner dimensions must match (got ", K, " vs ", K_b, ")");
  }

  // Validates a spatially-varying mixed-format op's per-output-element
  // precision index and derives the (row_stride, col_stride) pair the
  // kernel reads it with -- accepting a dense [M, N] map, a per-row [M, 1]
  // map, or a per-column [1, N] map (see gemm_policy.h's FormatPalette).
  // Bounds-checks every entry against the palette size and hands back the
  // contiguous int32 tensor to keep alive across the kernel call.
  inline void resolve_prec_idx(const Tensor &prec_idx, const Tensor &ref, int64_t M, int64_t N,
                        const char *op_name, int64_t n_formats, Tensor &pidx_out,
                        int64_t &idx_row_stride, int64_t &idx_col_stride)
  {
    TORCH_CHECK(prec_idx.dim() == 2, op_name, ": prec_idx must be 2D, got ", prec_idx.dim(), "D");
    Tensor pidx = prec_idx.to(ref.device(), at::kInt).contiguous();
    int64_t r = pidx.size(0), c = pidx.size(1);
    if (r == M && c == N)
    {
      idx_row_stride = N;
      idx_col_stride = 1;
    }
    else if (r == M && c == 1)
    {
      idx_row_stride = 1;
      idx_col_stride = 0;
    }
    else if (r == 1 && c == N)
    {
      idx_row_stride = 0;
      idx_col_stride = 1;
    }
    else
    {
      TORCH_CHECK(false, op_name, ": prec_idx shape must be [M, N], [M, 1] or [1, N] (M=", M,
                 ", N=", N, "), got [", r, ", ", c, "]");
    }
    // one pass over the map rather than two; there is no device sync to
    // save here, which is why the CUDA twin's memo has no counterpart
    auto bounds = at::aminmax(pidx);
    int64_t lo = std::get<0>(bounds).item<int64_t>();
    int64_t hi = std::get<1>(bounds).item<int64_t>();
    TORCH_CHECK(lo >= 0 && hi < n_formats, op_name, ": prec_idx entries must be in [0, ", n_formats,
               "), got range [", lo, ", ", hi, "]");
    pidx_out = pidx;
  }

  inline void check_palette_lengths(int64_t n, const char *op_name, std::initializer_list<int64_t> other_lens)
  {
    TORCH_CHECK(n >= 1 && n <= MAX_GEMM_FORMATS, op_name, ": expected 1..", MAX_GEMM_FORMATS,
               " palette formats, got ", n);
    for (int64_t l : other_lens)
      TORCH_CHECK(l == n, op_name, ": every palette parameter list must have length ", n,
                 " (got one of length ", l, ")");
  }
} // namespace mptorch::gemm_cpu
