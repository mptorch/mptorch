#include "custom_matmul_kernel.h"
#include "../common/dispatch.h"
#include "../quant_ops.h"
#include <ATen/ops/empty.h>
#include "utils.h"
#include <type_traits>

using namespace at;
using namespace mptorch::gemm_cpu;

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

  const mptorch::GemmDtype dt = mptorch::gemm_dtype_of(a_c, b_c, "custom_matmul_binaryK");
  const void *p_a = a_c.data_ptr();
  const void *p_b = b_c.data_ptr();
  void *p_c = c.data_ptr();

  mptorch::dispatch_round_mode(rm, [&](auto rm_c)
  {
    constexpr RoundMode RM = decltype(rm_c)::value;

    BinaryKMultiplierT<RM> mul{mul_man_bits, mul_exp_bits, static_cast<int>(mul_bias), mul_is_signed, sat, sub,
                               static_cast<int>(mul_prng_bits)};

    if (accumulate_quant)
    {
      BinaryKAdderT<RM> add{acc_man_bits, acc_exp_bits, static_cast<int>(acc_bias), acc_is_signed, sat, sub,
                            static_cast<int>(acc_prng_bits)};
      using Mac = SplitMac<BinaryKMultiplierT<RM>, BinaryKAdderT<RM>>;
      NaiveAccumulator<Mac> acc_proto{Mac{mul, add}, 0.f};
      matmul_cpu_kernel_impl(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto, use_rng, seed);
    }
    else
    {
      using Mac = SplitMac<BinaryKMultiplierT<RM>, IdentityAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{mul, IdentityAdder{}}, 0.f};
      matmul_cpu_kernel_impl(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto, use_rng, seed);
    }
  });

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

  const mptorch::GemmDtype dt = mptorch::gemm_dtype_of(a_c, b_c, "custom_matmul_binaryK_mixed");
  const void *p_a = a_c.data_ptr();
  const void *p_b = b_c.data_ptr();
  void *p_c = c.data_ptr();

  mptorch::dispatch_round_mode(rm, [&](auto rm_c)
  {
    constexpr RoundMode RM = decltype(rm_c)::value;

    auto make_mul = [&](int64_t i) {
      int mmb = static_cast<int>(mul_P[i] - 1);
      int meb = static_cast<int>(mul_is_signed ? mul_K[i] - mul_P[i] : mul_K[i] - mul_P[i] + 1);
      return BinaryKMultiplierT<RM>{mmb, meb, static_cast<int>(mul_bias[i]), mul_is_signed, sat, sub,
                               static_cast<int>(mul_prng_bits)};
    };

    if (accumulate_quant)
    {
      using Mac = SplitMac<BinaryKMultiplierT<RM>, BinaryKAdderT<RM>>;
      static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
      FormatPalette<Mac> pal;
      pal.n = static_cast<int>(n_fmt);
      for (int64_t i = 0; i < n_fmt; ++i)
      {
        int amb = static_cast<int>(acc_P[i] - 1);
        int aeb = static_cast<int>(acc_is_signed ? acc_K[i] - acc_P[i] : acc_K[i] - acc_P[i] + 1);
        BinaryKAdderT<RM> add{amb, aeb, static_cast<int>(acc_bias[i]), acc_is_signed, sat, sub,
                              static_cast<int>(acc_prng_bits)};
        pal.slots[i] = Mac{make_mul(i), add};
      }
      NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
      matmul_cpu_kernel_impl<true>(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto,
                                   use_rng, seed, pal, p_idx, sr, sc);
    }
    else
    {
      using Mac = SplitMac<BinaryKMultiplierT<RM>, IdentityAdder>;
      static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
      FormatPalette<Mac> pal;
      pal.n = static_cast<int>(n_fmt);
      for (int64_t i = 0; i < n_fmt; ++i)
        pal.slots[i] = Mac{make_mul(i), IdentityAdder{}};
      NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
      matmul_cpu_kernel_impl<true>(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto,
                                   use_rng, seed, pal, p_idx, sr, sc);
    }
  });

  return mptorch::widen_float64(c, widen_f64);
}
