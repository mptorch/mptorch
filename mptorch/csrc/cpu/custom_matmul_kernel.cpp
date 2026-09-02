#include "../common/gemm_policy.h"
#include "../common/dispatch.h"
#include "../common/modes.h"
#include "../quant_ops.h"
#include "utils.h"
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <initializer_list>
#include <mutex>
#include <type_traits>
#include <vector>

using namespace at;

namespace
{

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
  template <typename scalar_t, bool MIXED = false, class Accumulator>
  void matmul_cpu_kernel_impl(const scalar_t *__restrict__ A, const scalar_t *__restrict__ B,
                              scalar_t *__restrict__ C,
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
          for (int64_t i = 0; i < ti; ++i)
          {
            for (int64_t k = 0; k < tk; ++k)
            {
              float aVal = trans_a
                              ? static_cast<float>(A[(k0 + k) * M + (i0 + i)])
                              : static_cast<float>(A[(i0 + i) * K + (k0 + k)]);
              for (int64_t j = 0; j < tj; ++j)
              {
                float bVal = trans_b
                                ? static_cast<float>(B[(j0 + j) * K + (k0 + k)])
                                : static_cast<float>(B[(k0 + k) * N + (j0 + j)]);
                const int64_t idx = i * tj + j;
                Accumulator::accumulate(tv, idx, MIXED ? *slot_of[MIXED ? idx : 0] : mac_single,
                                        aVal, bVal);
              }
            }
          }
        }

        for (int64_t i = 0; i < ti; ++i)
          for (int64_t j = 0; j < tj; ++j)
            C[(i0 + i) * N + (j0 + j)] = static_cast<scalar_t>(tile.finalize(i * tj + j));
      }
    });
  }

  void check_matmul_inputs(const Tensor &a, const Tensor &b, const char *op_name,
                           int64_t round_mode, int64_t accumulate_algorithm)
  {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, op_name, " expects 2D tensors, got ",
               a.dim(), "D and ", b.dim(), "D");
    TORCH_CHECK(static_cast<AccumulateAlgorithm>(accumulate_algorithm) == AccumulateAlgorithm::NAIVE,
               op_name, ": only AccumulateAlgorithm.NAIVE is supported in this build");
  }

  void matmul_output_shape(const Tensor &a, const Tensor &b, bool trans_a, bool trans_b,
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
  void resolve_prec_idx(const Tensor &prec_idx, const Tensor &ref, int64_t M, int64_t N,
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

  void check_palette_lengths(int64_t n, const char *op_name, std::initializer_list<int64_t> other_lens)
  {
    TORCH_CHECK(n >= 1 && n <= MAX_GEMM_FORMATS, op_name, ": expected 1..", MAX_GEMM_FORMATS,
               " palette formats, got ", n);
    for (int64_t l : other_lens)
      TORCH_CHECK(l == n, op_name, ": every palette parameter list must have length ", n,
                 " (got one of length ", l, ")");
  }

} // namespace

Tensor binaryK_matmul_cpu(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
    bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  check_matmul_inputs(a, b, "custom_matmul_binaryK", round_mode, accumulate_algorithm);

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK", M, K, N);

  // float64 in, float64 out, narrowed here instead of on every load so the
  // dispatch below need not instantiate for double -- same values, see
  // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
  const bool widen_f64 = mptorch::narrow_float64(a, b);
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return mptorch::widen_float64(c, widen_f64);

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
  RoundMode rm = static_cast<RoundMode>(round_mode);

  int mul_man_bits = static_cast<int>(mul_P - 1);
  int mul_exp_bits = static_cast<int>(mul_is_signed ? mul_K - mul_P : mul_K - mul_P + 1);
  int acc_man_bits = static_cast<int>(acc_P - 1);
  int acc_exp_bits = static_cast<int>(acc_is_signed ? acc_K - acc_P : acc_K - acc_P + 1);

  bool use_rng = (rm == RoundMode::SR);
  uint64_t seed = use_rng ? draw_cpu_seed() : 0;

  MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_matmul_cpu", [&]
                                  {
    BinaryKMultiplier mul{mul_man_bits, mul_exp_bits, static_cast<int>(mul_bias), mul_is_signed, sat, rm, sub,
                          static_cast<int>(mul_prng_bits)};
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    if (accumulate_quant)
    {
      BinaryKAdder add{acc_man_bits, acc_exp_bits, static_cast<int>(acc_bias), acc_is_signed, sat, rm, sub,
                       static_cast<int>(acc_prng_bits)};
      using Mac = SplitMac<BinaryKMultiplier, BinaryKAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{mul, add}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, seed);
    }
    else
    {
      using Mac = SplitMac<BinaryKMultiplier, IdentityAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{mul, IdentityAdder{}}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, seed);
    } });

  return mptorch::widen_float64(c, widen_f64);
}

Tensor superfp_matmul_cpu(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
    bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades,
    int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  check_matmul_inputs(a, b, "custom_matmul_superfp", round_mode, accumulate_algorithm);

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp", M, K, N);

  // float64 in, float64 out, narrowed here instead of on every load so the
  // dispatch below need not instantiate for double -- same values, see
  // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
  const bool widen_f64 = mptorch::narrow_float64(a, b);
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return mptorch::widen_float64(c, widen_f64);

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  RoundMode rm = static_cast<RoundMode>(round_mode);

  bool use_rng = (rm == RoundMode::SR);
  uint64_t seed = use_rng ? draw_cpu_seed() : 0;

  MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "superfp_matmul_cpu", [&]
                                  {
    SuperfpMultiplier mul{static_cast<int>(mul_man_bits), static_cast<int>(mul_exp_bits),
                          static_cast<int>(mul_normal_binades), static_cast<int>(mul_bias), mul_is_signed, sat, rm,
                          static_cast<int>(mul_prng_bits)};
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    if (accumulate_quant)
    {
      SuperfpAdder add{static_cast<int>(acc_man_bits), static_cast<int>(acc_exp_bits),
                       static_cast<int>(acc_normal_binades), static_cast<int>(acc_bias), acc_is_signed, sat, rm,
                       static_cast<int>(acc_prng_bits)};
      using Mac = SplitMac<SuperfpMultiplier, SuperfpAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{mul, add}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, seed);
    }
    else
    {
      using Mac = SplitMac<SuperfpMultiplier, IdentityAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{mul, IdentityAdder{}}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, seed);
    } });

  return mptorch::widen_float64(c, widen_f64);
}

Tensor binaryK_matmul_fma_cpu(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode,
    int64_t fma_prng_bits)
{
  check_matmul_inputs(a, b, "custom_matmul_binaryK_fma", round_mode, accumulate_algorithm);

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK_fma", M, K, N);

  // float64 in, float64 out, narrowed here instead of on every load so the
  // dispatch below need not instantiate for double -- same values, see
  // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
  const bool widen_f64 = mptorch::narrow_float64(a, b);
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return mptorch::widen_float64(c, widen_f64);

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
  RoundMode rm = static_cast<RoundMode>(round_mode);

  int fma_man_bits = static_cast<int>(fma_P - 1);
  int fma_exp_bits = static_cast<int>(fma_is_signed ? fma_K - fma_P : fma_K - fma_P + 1);

  bool use_rng = (rm == RoundMode::SR);
  uint64_t seed = use_rng ? draw_cpu_seed() : 0;

  MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_matmul_fma_cpu", [&]
                                  {
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    if (fma_quant)
    {
      BinaryKAdder add{fma_man_bits, fma_exp_bits, static_cast<int>(fma_bias), fma_is_signed, sat, rm, sub,
                       static_cast<int>(fma_prng_bits)};
      using Mac = FusedMac<BinaryKAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{add}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, seed);
    }
    else
    {
      using Mac = FusedMac<IdentityAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{IdentityAdder{}}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, seed);
    } });

  return mptorch::widen_float64(c, widen_f64);
}

Tensor superfp_matmul_fma_cpu(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades,
    int64_t fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t fma_prng_bits)
{
  check_matmul_inputs(a, b, "custom_matmul_superfp_fma", round_mode, accumulate_algorithm);

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp_fma", M, K, N);

  // float64 in, float64 out, narrowed here instead of on every load so the
  // dispatch below need not instantiate for double -- same values, see
  // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
  const bool widen_f64 = mptorch::narrow_float64(a, b);
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return mptorch::widen_float64(c, widen_f64);

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  RoundMode rm = static_cast<RoundMode>(round_mode);

  bool use_rng = (rm == RoundMode::SR);
  uint64_t seed = use_rng ? draw_cpu_seed() : 0;

  MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "superfp_matmul_fma_cpu", [&]
                                  {
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    if (fma_quant)
    {
      SuperfpAdder add{static_cast<int>(fma_man_bits), static_cast<int>(fma_exp_bits),
                       static_cast<int>(fma_normal_binades), static_cast<int>(fma_bias), fma_is_signed, sat, rm,
                       static_cast<int>(fma_prng_bits)};
      using Mac = FusedMac<SuperfpAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{add}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, seed);
    }
    else
    {
      using Mac = FusedMac<IdentityAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{IdentityAdder{}}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, seed);
    } });

  return mptorch::widen_float64(c, widen_f64);
}

// Spatially-varying (per-output-element) mixed-format binaryK GEMM: same
// SplitMac arithmetic as binaryK_matmul_cpu, but the multiply/accumulate
// format for each output element C[i, j] is picked from a palette of up to
// MAX_GEMM_FORMATS entries by prec_idx[i, j] (see gemm_policy.h's
// FormatPalette). mul_K/mul_P/mul_bias and acc_K/acc_P/acc_bias are
// per-palette-entry lists (all the same length); round_mode/saturation/
// sign/prng_bits are shared across the palette.
Tensor binaryK_matmul_mixed_cpu(
    Tensor a, Tensor b, Tensor prec_idx, bool trans_a, bool trans_b,
    c10::IntArrayRef mul_K, c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias, bool mul_is_signed,
    bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias,
    bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  check_matmul_inputs(a, b, "custom_matmul_binaryK_mixed", round_mode, accumulate_algorithm);

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK_mixed", M, K, N);

  int64_t n_fmt = static_cast<int64_t>(mul_K.size());
  check_palette_lengths(n_fmt, "custom_matmul_binaryK_mixed",
                        {static_cast<int64_t>(mul_P.size()), static_cast<int64_t>(mul_bias.size()),
                         static_cast<int64_t>(acc_K.size()), static_cast<int64_t>(acc_P.size()),
                         static_cast<int64_t>(acc_bias.size())});

  // float64 in, float64 out, narrowed here instead of on every load so the
  // dispatch below need not instantiate for double -- same values, see
  // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
  const bool widen_f64 = mptorch::narrow_float64(a, b);
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return mptorch::widen_float64(c, widen_f64);

  Tensor pidx;
  int64_t sr = 0, sc = 0;
  resolve_prec_idx(prec_idx, a, M, N, "custom_matmul_binaryK_mixed", n_fmt, pidx, sr, sc);
  const int32_t *p_idx = pidx.data_ptr<int32_t>();

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
  RoundMode rm = static_cast<RoundMode>(round_mode);

  bool use_rng = (rm == RoundMode::SR);
  uint64_t seed = use_rng ? draw_cpu_seed() : 0;

  MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_matmul_mixed_cpu", [&]
                                  {
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    auto make_mul = [&](int64_t i) {
      int mmb = static_cast<int>(mul_P[i] - 1);
      int meb = static_cast<int>(mul_is_signed ? mul_K[i] - mul_P[i] : mul_K[i] - mul_P[i] + 1);
      return BinaryKMultiplier{mmb, meb, static_cast<int>(mul_bias[i]), mul_is_signed, sat, rm, sub,
                               static_cast<int>(mul_prng_bits)};
    };

    if (accumulate_quant)
    {
      using Mac = SplitMac<BinaryKMultiplier, BinaryKAdder>;
      static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
      FormatPalette<Mac> pal;
      pal.n = static_cast<int>(n_fmt);
      for (int64_t i = 0; i < n_fmt; ++i)
      {
        int amb = static_cast<int>(acc_P[i] - 1);
        int aeb = static_cast<int>(acc_is_signed ? acc_K[i] - acc_P[i] : acc_K[i] - acc_P[i] + 1);
        BinaryKAdder add{amb, aeb, static_cast<int>(acc_bias[i]), acc_is_signed, sat, rm, sub,
                         static_cast<int>(acc_prng_bits)};
        pal.slots[i] = Mac{make_mul(i), add};
      }
      NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
      matmul_cpu_kernel_impl<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                             use_rng, seed, pal, p_idx, sr, sc);
    }
    else
    {
      using Mac = SplitMac<BinaryKMultiplier, IdentityAdder>;
      static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
      FormatPalette<Mac> pal;
      pal.n = static_cast<int>(n_fmt);
      for (int64_t i = 0; i < n_fmt; ++i)
        pal.slots[i] = Mac{make_mul(i), IdentityAdder{}};
      NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
      matmul_cpu_kernel_impl<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                             use_rng, seed, pal, p_idx, sr, sc);
    } });

  return mptorch::widen_float64(c, widen_f64);
}

// superfp analogue of binaryK_matmul_mixed_cpu -- see its comment.
Tensor superfp_matmul_mixed_cpu(
    Tensor a, Tensor b, Tensor prec_idx, bool trans_a, bool trans_b,
    c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits, c10::IntArrayRef mul_normal_binades,
    c10::IntArrayRef mul_bias, bool mul_is_signed,
    bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits,
    c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  check_matmul_inputs(a, b, "custom_matmul_superfp_mixed", round_mode, accumulate_algorithm);

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp_mixed", M, K, N);

  int64_t n_fmt = static_cast<int64_t>(mul_man_bits.size());
  check_palette_lengths(n_fmt, "custom_matmul_superfp_mixed",
                        {static_cast<int64_t>(mul_exp_bits.size()),
                         static_cast<int64_t>(mul_normal_binades.size()),
                         static_cast<int64_t>(mul_bias.size()),
                         static_cast<int64_t>(acc_man_bits.size()),
                         static_cast<int64_t>(acc_exp_bits.size()),
                         static_cast<int64_t>(acc_normal_binades.size()),
                         static_cast<int64_t>(acc_bias.size())});

  // float64 in, float64 out, narrowed here instead of on every load so the
  // dispatch below need not instantiate for double -- same values, see
  // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
  const bool widen_f64 = mptorch::narrow_float64(a, b);
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return mptorch::widen_float64(c, widen_f64);

  Tensor pidx;
  int64_t sr = 0, sc = 0;
  resolve_prec_idx(prec_idx, a, M, N, "custom_matmul_superfp_mixed", n_fmt, pidx, sr, sc);
  const int32_t *p_idx = pidx.data_ptr<int32_t>();

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  RoundMode rm = static_cast<RoundMode>(round_mode);

  bool use_rng = (rm == RoundMode::SR);
  uint64_t seed = use_rng ? draw_cpu_seed() : 0;

  MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "superfp_matmul_mixed_cpu", [&]
                                  {
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    auto make_mul = [&](int64_t i) {
      return SuperfpMultiplier{static_cast<int>(mul_man_bits[i]), static_cast<int>(mul_exp_bits[i]),
                               static_cast<int>(mul_normal_binades[i]), static_cast<int>(mul_bias[i]),
                               mul_is_signed, sat, rm, static_cast<int>(mul_prng_bits)};
    };

    if (accumulate_quant)
    {
      using Mac = SplitMac<SuperfpMultiplier, SuperfpAdder>;
      static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
      FormatPalette<Mac> pal;
      pal.n = static_cast<int>(n_fmt);
      for (int64_t i = 0; i < n_fmt; ++i)
      {
        SuperfpAdder add{static_cast<int>(acc_man_bits[i]), static_cast<int>(acc_exp_bits[i]),
                         static_cast<int>(acc_normal_binades[i]), static_cast<int>(acc_bias[i]),
                         acc_is_signed, sat, rm, static_cast<int>(acc_prng_bits)};
        pal.slots[i] = Mac{make_mul(i), add};
      }
      NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
      matmul_cpu_kernel_impl<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                             use_rng, seed, pal, p_idx, sr, sc);
    }
    else
    {
      using Mac = SplitMac<SuperfpMultiplier, IdentityAdder>;
      static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
      FormatPalette<Mac> pal;
      pal.n = static_cast<int>(n_fmt);
      for (int64_t i = 0; i < n_fmt; ++i)
        pal.slots[i] = Mac{make_mul(i), IdentityAdder{}};
      NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
      matmul_cpu_kernel_impl<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                             use_rng, seed, pal, p_idx, sr, sc);
    } });

  return mptorch::widen_float64(c, widen_f64);
}

// Spatially-varying (per-output-element) mixed-format binaryK FMA GEMM: same
// FusedMac arithmetic as binaryK_matmul_fma_cpu (one rounding per K-step),
// but the FMA format for each output element C[i, j] is picked from a
// palette of up to MAX_GEMM_FORMATS entries by prec_idx[i, j] (see
// gemm_policy.h's FormatPalette). fma_K/fma_P/fma_bias are per-palette-entry
// lists (all the same length); round_mode/saturation/sign/prng_bits are
// shared across the palette. fma_quant=false is rejected -- see the CUDA
// twin for why.
Tensor binaryK_matmul_fma_mixed_cpu(
    Tensor a, Tensor b, Tensor prec_idx, bool trans_a, bool trans_b,
    bool fma_quant, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias,
    bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t subnormals_mode, int64_t fma_prng_bits)
{
  check_matmul_inputs(a, b, "custom_matmul_binaryK_fma_mixed", round_mode, accumulate_algorithm);
  TORCH_CHECK(fma_quant, "custom_matmul_binaryK_fma_mixed: fma_quant=false has no per-element "
                         "format to vary; use custom_matmul_binaryK_fma");

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK_fma_mixed", M, K, N);

  int64_t n_fmt = static_cast<int64_t>(fma_K.size());
  check_palette_lengths(n_fmt, "custom_matmul_binaryK_fma_mixed",
                        {static_cast<int64_t>(fma_P.size()), static_cast<int64_t>(fma_bias.size())});

  // float64 in, float64 out, narrowed here instead of on every load so the
  // dispatch below need not instantiate for double -- same values, see
  // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
  const bool widen_f64 = mptorch::narrow_float64(a, b);
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return mptorch::widen_float64(c, widen_f64);

  Tensor pidx;
  int64_t sr = 0, sc = 0;
  resolve_prec_idx(prec_idx, a, M, N, "custom_matmul_binaryK_fma_mixed", n_fmt, pidx, sr, sc);
  const int32_t *p_idx = pidx.data_ptr<int32_t>();

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
  RoundMode rm = static_cast<RoundMode>(round_mode);

  bool use_rng = (rm == RoundMode::SR);
  uint64_t seed = use_rng ? draw_cpu_seed() : 0;

  MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_matmul_fma_mixed_cpu", [&]
                                  {
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    using Mac = FusedMac<BinaryKAdder>;
    static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
    FormatPalette<Mac> pal;
    pal.n = static_cast<int>(n_fmt);
    for (int64_t i = 0; i < n_fmt; ++i)
    {
      int mb = static_cast<int>(fma_P[i] - 1);
      int eb = static_cast<int>(fma_is_signed ? fma_K[i] - fma_P[i] : fma_K[i] - fma_P[i] + 1);
      BinaryKAdder add{mb, eb, static_cast<int>(fma_bias[i]), fma_is_signed, sat, rm, sub,
                       static_cast<int>(fma_prng_bits)};
      pal.slots[i] = Mac{add};
    }
    NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
    matmul_cpu_kernel_impl<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                           use_rng, seed, pal, p_idx, sr, sc); });

  return mptorch::widen_float64(c, widen_f64);
}

// superfp analogue of binaryK_matmul_fma_mixed_cpu -- see its comment.
Tensor superfp_matmul_fma_mixed_cpu(
    Tensor a, Tensor b, Tensor prec_idx, bool trans_a, bool trans_b,
    bool fma_quant, c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
    c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t fma_prng_bits)
{
  check_matmul_inputs(a, b, "custom_matmul_superfp_fma_mixed", round_mode, accumulate_algorithm);
  TORCH_CHECK(fma_quant, "custom_matmul_superfp_fma_mixed: fma_quant=false has no per-element "
                         "format to vary; use custom_matmul_superfp_fma");

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp_fma_mixed", M, K, N);

  int64_t n_fmt = static_cast<int64_t>(fma_man_bits.size());
  check_palette_lengths(n_fmt, "custom_matmul_superfp_fma_mixed",
                        {static_cast<int64_t>(fma_exp_bits.size()),
                         static_cast<int64_t>(fma_normal_binades.size()),
                         static_cast<int64_t>(fma_bias.size())});

  // float64 in, float64 out, narrowed here instead of on every load so the
  // dispatch below need not instantiate for double -- same values, see
  // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
  const bool widen_f64 = mptorch::narrow_float64(a, b);
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return mptorch::widen_float64(c, widen_f64);

  Tensor pidx;
  int64_t sr = 0, sc = 0;
  resolve_prec_idx(prec_idx, a, M, N, "custom_matmul_superfp_fma_mixed", n_fmt, pidx, sr, sc);
  const int32_t *p_idx = pidx.data_ptr<int32_t>();

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  RoundMode rm = static_cast<RoundMode>(round_mode);

  bool use_rng = (rm == RoundMode::SR);
  uint64_t seed = use_rng ? draw_cpu_seed() : 0;

  MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "superfp_matmul_fma_mixed_cpu", [&]
                                  {
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    using Mac = FusedMac<SuperfpAdder>;
    static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
    FormatPalette<Mac> pal;
    pal.n = static_cast<int>(n_fmt);
    for (int64_t i = 0; i < n_fmt; ++i)
    {
      SuperfpAdder add{static_cast<int>(fma_man_bits[i]), static_cast<int>(fma_exp_bits[i]),
                       static_cast<int>(fma_normal_binades[i]), static_cast<int>(fma_bias[i]),
                       fma_is_signed, sat, rm, static_cast<int>(fma_prng_bits)};
      pal.slots[i] = Mac{add};
    }
    NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
    matmul_cpu_kernel_impl<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                           use_rng, seed, pal, p_idx, sr, sc); });

  return mptorch::widen_float64(c, widen_f64);
}
