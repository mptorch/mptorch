#pragma once

// The host side of the odd palette and cross-family GEMM ops
// (custom_matmul_*_mixed_odd, custom_matmul_{binaryK_superfp,superfp_binaryK}_odd;
// carrier_rounding=RoundMode.RO): one body per op, written once on the
// backend, which {cpu,cuda}/custom_matmul_odd_entry.cpp name. Each is its
// twin's entry point with the packed Args wrapped, OddPaletteArgs for a
// palette (common/carrier_odd_mixed.h) and OddArgs for a cross-family mac
// (common/carrier_odd.h): the schema, the checks and the driver are the
// twin's, so a call differs from the twin's only in the kernel it runs. The
// drivers refuse every algorithm but NAIVE, as for the twins.
//
// A header of its own rather than more of carrier_odd_host.h, which the Meta
// units include: those keep compiling what they did.

#include "carrier_odd_host.h"
#include "carrier_odd_mixed.h"
#include "gemm_cross_host.h"

namespace mptorch::gemm
{
  template <class Backend>
  at::Tensor run_binaryK_matmul_mixed_odd(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                          bool trans_a, bool trans_b, c10::IntArrayRef mul_K,
                                          c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias,
                                          bool mul_is_signed, bool accumulate_quant,
                                          c10::IntArrayRef acc_K, c10::IntArrayRef acc_P,
                                          c10::IntArrayRef acc_bias, bool acc_is_signed,
                                          int64_t accumulate_algorithm, int64_t round_mode,
                                          int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                          int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                          int64_t mul_prng_bits, int64_t acc_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_matmul_binaryK_mixed_odd");
#else
    constexpr const char *op = "custom_matmul_binaryK_mixed_odd";
    return run_custom_matmul_mixed<Backend>(
        op,
        [&] {
          return odd_palette(pack_binaryK_split_mixed(op, mul_K, mul_P, mul_bias, mul_is_signed,
                                                      accumulate_quant, acc_K, acc_P, acc_bias,
                                                      acc_is_signed, mul_saturation_mode,
                                                      mul_subnormals_mode, acc_saturation_mode,
                                                      acc_subnormals_mode, mul_prng_bits,
                                                      acc_prng_bits));
        },
        a, b, prec_idx,
        trans_a, trans_b,
        accumulate_algorithm, round_mode);
#endif
  }

  template <class Backend>
  at::Tensor run_superfp_matmul_mixed_odd(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                          bool trans_a, bool trans_b, c10::IntArrayRef mul_man_bits,
                                          c10::IntArrayRef mul_exp_bits,
                                          c10::IntArrayRef mul_normal_binades,
                                          c10::IntArrayRef mul_bias, bool mul_is_signed,
                                          bool accumulate_quant, c10::IntArrayRef acc_man_bits,
                                          c10::IntArrayRef acc_exp_bits,
                                          c10::IntArrayRef acc_normal_binades,
                                          c10::IntArrayRef acc_bias, bool acc_is_signed,
                                          int64_t accumulate_algorithm, int64_t round_mode,
                                          int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                          int64_t mul_prng_bits, int64_t acc_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_matmul_superfp_mixed_odd");
#else
    constexpr const char *op = "custom_matmul_superfp_mixed_odd";
    return run_custom_matmul_mixed<Backend>(
        op,
        [&] {
          return odd_palette(pack_superfp_split_mixed(op, mul_man_bits, mul_exp_bits,
                                                      mul_normal_binades, mul_bias, mul_is_signed,
                                                      accumulate_quant, acc_man_bits, acc_exp_bits,
                                                      acc_normal_binades, acc_bias, acc_is_signed,
                                                      mul_saturation_mode, acc_saturation_mode,
                                                      mul_prng_bits, acc_prng_bits));
        },
        a, b, prec_idx,
        trans_a, trans_b,
        accumulate_algorithm, round_mode);
#endif
  }

  template <class Backend>
  at::Tensor run_binaryK_matmul_fma_mixed_odd(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                              bool trans_a, bool trans_b, bool fma_quant,
                                              c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                              c10::IntArrayRef fma_bias, bool fma_is_signed,
                                              int64_t accumulate_algorithm, int64_t round_mode,
                                              int64_t fma_saturation_mode,
                                              int64_t fma_subnormals_mode, int64_t fma_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_matmul_binaryK_fma_mixed_odd");
#else
    constexpr const char *op = "custom_matmul_binaryK_fma_mixed_odd";
    return run_custom_matmul_mixed<Backend>(
        op,
        [&] {
          return odd_palette(pack_binaryK_fused_mixed(op, fma_K, fma_P, fma_bias, fma_is_signed,
                                                      fma_saturation_mode, fma_subnormals_mode,
                                                      fma_prng_bits));
        },
        a, b, prec_idx,
        trans_a, trans_b,
        accumulate_algorithm, round_mode,
        [&] { check_fma_quant(fma_quant, op); });
#endif
  }

  template <class Backend>
  at::Tensor run_superfp_matmul_fma_mixed_odd(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                              bool trans_a, bool trans_b, bool fma_quant,
                                              c10::IntArrayRef fma_man_bits,
                                              c10::IntArrayRef fma_exp_bits,
                                              c10::IntArrayRef fma_normal_binades,
                                              c10::IntArrayRef fma_bias, bool fma_is_signed,
                                              int64_t accumulate_algorithm, int64_t round_mode,
                                              int64_t fma_saturation_mode, int64_t fma_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_matmul_superfp_fma_mixed_odd");
#else
    constexpr const char *op = "custom_matmul_superfp_fma_mixed_odd";
    return run_custom_matmul_mixed<Backend>(
        op,
        [&] {
          return odd_palette(pack_superfp_fused_mixed(op, fma_man_bits, fma_exp_bits,
                                                      fma_normal_binades, fma_bias, fma_is_signed,
                                                      fma_saturation_mode, fma_prng_bits));
        },
        a, b, prec_idx,
        trans_a, trans_b,
        accumulate_algorithm, round_mode,
        [&] { check_fma_quant(fma_quant, op); });
#endif
  }

  template <class Backend>
  at::Tensor run_binaryK_superfp_matmul_odd(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                            int64_t mul_K, int64_t mul_P, int64_t mul_bias,
                                            bool mul_is_signed, int64_t acc_man_bits,
                                            int64_t acc_exp_bits, int64_t acc_normal_binades,
                                            int64_t acc_bias, bool acc_is_signed,
                                            int64_t accumulate_algorithm, int64_t round_mode,
                                            int64_t mul_saturation_mode,
                                            int64_t mul_subnormals_mode,
                                            int64_t acc_saturation_mode, int64_t mul_prng_bits,
                                            int64_t acc_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_matmul_binaryK_superfp_odd");
#else
    constexpr const char *op = "custom_matmul_binaryK_superfp_odd";
    return run_custom_matmul<Backend>(
        op,
        OddArgs<BinaryKSuperfpArgs>{pack_binaryK_superfp(mul_K, mul_P, mul_bias, mul_is_signed,
                                                         acc_man_bits, acc_exp_bits,
                                                         acc_normal_binades, acc_bias,
                                                         acc_is_signed, mul_saturation_mode,
                                                         mul_subnormals_mode, acc_saturation_mode,
                                                         mul_prng_bits, acc_prng_bits)},
        a, b, trans_a, trans_b,
        accumulate_algorithm, round_mode);
#endif
  }

  template <class Backend>
  at::Tensor run_superfp_binaryK_matmul_odd(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                            int64_t mul_man_bits, int64_t mul_exp_bits,
                                            int64_t mul_normal_binades, int64_t mul_bias,
                                            bool mul_is_signed, int64_t acc_K, int64_t acc_P,
                                            int64_t acc_bias, bool acc_is_signed,
                                            int64_t accumulate_algorithm, int64_t round_mode,
                                            int64_t mul_saturation_mode,
                                            int64_t acc_saturation_mode,
                                            int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                                            int64_t acc_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_matmul_superfp_binaryK_odd");
#else
    constexpr const char *op = "custom_matmul_superfp_binaryK_odd";
    return run_custom_matmul<Backend>(
        op,
        OddArgs<SuperfpBinaryKArgs>{pack_superfp_binaryK(mul_man_bits, mul_exp_bits,
                                                         mul_normal_binades, mul_bias,
                                                         mul_is_signed, acc_K, acc_P, acc_bias,
                                                         acc_is_signed, mul_saturation_mode,
                                                         acc_saturation_mode, acc_subnormals_mode,
                                                         mul_prng_bits, acc_prng_bits)},
        a, b, trans_a, trans_b,
        accumulate_algorithm, round_mode);
#endif
  }
} // namespace mptorch::gemm
