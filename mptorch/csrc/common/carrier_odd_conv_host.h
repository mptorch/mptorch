#pragma once

// The host side of the ten odd conv ops (custom_conv_*_odd;
// carrier_rounding=RoundMode.RO): one body per op, which
// {cpu,cuda}/custom_conv_odd_entry.cpp name, each its twin's conv entry
// point with the packed Args wrapped as in carrier_odd_gemm_host.h.
//
// The single-format and cross-family ones take their twins' schema, the
// *_accumulated one, but run NAIVE only, through run_custom_conv_odd below
// rather than run_custom_conv_single: that one also builds an AccumulateArgs
// for KAHAN, BLOCK and TREE, which an OddArgs has no AccumulateFamily for.
// The palette ones are NAIVE only like their twins.

#include "carrier_odd_gemm_host.h"

namespace mptorch::gemm
{
  // run_custom_conv_single's checks for a NAIVE call, in its order and with
  // its messages, then the conv driver on the odd Args. run_custom_conv checks
  // no algorithm, so the refusal of the others is this function's.
  template <class Backend, class Base, class Widths, class Common>
  at::Tensor run_custom_conv_odd(const char *op_name, const OddArgs<Base> &args,
                                 const AccumulateTail<Widths, Common> &tail, at::Tensor a,
                                 at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
                                 c10::IntArrayRef stride, c10::IntArrayRef padding,
                                 c10::IntArrayRef dilation, int64_t groups,
                                 int64_t accumulate_algorithm, int64_t round_mode)
  {
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    TORCH_CHECK(tail.block_size >= 0 && tail.block_size <= (int64_t(1) << 30), op_name,
                ": block_size ", tail.block_size, " is out of range");
    check_naive_only(op_name, accumulate_algorithm);
    TORCH_CHECK(tail.block_size == 0 && !tail.outer_quant, op_name,
                ": AccumulateAlgorithm.NAIVE takes no block_size and no outer format");
    return run_custom_conv<Backend>(op_name, args, a, b, nullptr, conv_pass, out_size, stride,
                                    padding, dilation, groups, round_mode);
  }

  template <class Backend>
  at::Tensor run_binaryK_conv_odd(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                  c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                  c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                  int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias,
                                  bool mul_is_signed, bool accumulate_quant, int64_t acc_K,
                                  int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                  int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                  int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size,
                                  bool outer_quant, int64_t outer_K, int64_t outer_P,
                                  int64_t outer_bias, bool outer_is_signed,
                                  int64_t outer_saturation_mode, int64_t outer_subnormals_mode,
                                  int64_t outer_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_conv_binaryK_odd");
#else
    constexpr const char *op = "custom_conv_binaryK_odd";
    return run_custom_conv_odd<Backend>(
        op,
        OddArgs<BinaryKSplitArgs>{pack_binaryK_split(mul_K, mul_P, mul_bias, mul_is_signed,
                                                     accumulate_quant, acc_K, acc_P, acc_bias,
                                                     acc_is_signed, mul_saturation_mode,
                                                     mul_subnormals_mode, acc_saturation_mode,
                                                     acc_subnormals_mode, mul_prng_bits,
                                                     acc_prng_bits)},
        pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                           outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
        a, b, conv_pass, out_size, stride, padding, dilation, groups,
        accumulate_algorithm, round_mode);
#endif
  }

  template <class Backend>
  at::Tensor run_superfp_conv_odd(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                  c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                  c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                  int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits,
                                  int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
                                  bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                                  int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                  int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size,
                                  bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits,
                                  int64_t outer_normal_binades, int64_t outer_bias,
                                  bool outer_is_signed, int64_t outer_saturation_mode,
                                  int64_t outer_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_conv_superfp_odd");
#else
    constexpr const char *op = "custom_conv_superfp_odd";
    return run_custom_conv_odd<Backend>(
        op,
        OddArgs<SuperfpSplitArgs>{pack_superfp_split(mul_man_bits, mul_exp_bits, mul_normal_binades,
                                                     mul_bias, mul_is_signed, accumulate_quant,
                                                     acc_man_bits, acc_exp_bits, acc_normal_binades,
                                                     acc_bias, acc_is_signed, mul_saturation_mode,
                                                     acc_saturation_mode, mul_prng_bits,
                                                     acc_prng_bits)},
        pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                           outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                           outer_prng_bits),
        a, b, conv_pass, out_size, stride, padding, dilation, groups,
        accumulate_algorithm, round_mode);
#endif
  }

  template <class Backend>
  at::Tensor run_binaryK_conv_fma_odd(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                      c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                      c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                      int64_t groups, bool fma_quant, int64_t fma_K, int64_t fma_P,
                                      int64_t fma_bias, bool fma_is_signed,
                                      int64_t accumulate_algorithm, int64_t round_mode,
                                      int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                      int64_t fma_prng_bits, int64_t block_size, bool outer_quant,
                                      int64_t outer_K, int64_t outer_P, int64_t outer_bias,
                                      bool outer_is_signed, int64_t outer_saturation_mode,
                                      int64_t outer_subnormals_mode, int64_t outer_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_conv_binaryK_fma_odd");
#else
    constexpr const char *op = "custom_conv_binaryK_fma_odd";
    check_fma_quant(fma_quant, op);
    return run_custom_conv_odd<Backend>(
        op,
        OddArgs<BinaryKFusedArgs>{pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias,
                                                     fma_is_signed, fma_saturation_mode,
                                                     fma_subnormals_mode, fma_prng_bits)},
        pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                           outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
        a, b, conv_pass, out_size, stride, padding, dilation, groups,
        accumulate_algorithm, round_mode);
#endif
  }

  template <class Backend>
  at::Tensor run_superfp_conv_fma_odd(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                      c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                      c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                      int64_t groups, bool fma_quant, int64_t fma_man_bits,
                                      int64_t fma_exp_bits, int64_t fma_normal_binades,
                                      int64_t fma_bias, bool fma_is_signed,
                                      int64_t accumulate_algorithm, int64_t round_mode,
                                      int64_t fma_saturation_mode, int64_t fma_prng_bits,
                                      int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                      int64_t outer_exp_bits, int64_t outer_normal_binades,
                                      int64_t outer_bias, bool outer_is_signed,
                                      int64_t outer_saturation_mode, int64_t outer_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_conv_superfp_fma_odd");
#else
    constexpr const char *op = "custom_conv_superfp_fma_odd";
    check_fma_quant(fma_quant, op);
    return run_custom_conv_odd<Backend>(
        op,
        OddArgs<SuperfpFusedArgs>{pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits,
                                                     fma_normal_binades, fma_bias, fma_is_signed,
                                                     fma_saturation_mode, fma_prng_bits)},
        pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                           outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                           outer_prng_bits),
        a, b, conv_pass, out_size, stride, padding, dilation, groups,
        accumulate_algorithm, round_mode);
#endif
  }

  template <class Backend>
  at::Tensor run_binaryK_conv_mixed_odd(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                        int64_t conv_pass, c10::IntArrayRef out_size,
                                        c10::IntArrayRef stride, c10::IntArrayRef padding,
                                        c10::IntArrayRef dilation, int64_t groups,
                                        c10::IntArrayRef mul_K, c10::IntArrayRef mul_P,
                                        c10::IntArrayRef mul_bias, bool mul_is_signed,
                                        bool accumulate_quant, c10::IntArrayRef acc_K,
                                        c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias,
                                        bool acc_is_signed, int64_t accumulate_algorithm,
                                        int64_t round_mode, int64_t mul_saturation_mode,
                                        int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
                                        int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                                        int64_t acc_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_conv_binaryK_mixed_odd");
#else
    constexpr const char *op = "custom_conv_binaryK_mixed_odd";
    return run_custom_conv_mixed<Backend>(
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
        conv_pass, out_size, stride, padding, dilation, groups,
        accumulate_algorithm, round_mode);
#endif
  }

  template <class Backend>
  at::Tensor run_superfp_conv_mixed_odd(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                        int64_t conv_pass, c10::IntArrayRef out_size,
                                        c10::IntArrayRef stride, c10::IntArrayRef padding,
                                        c10::IntArrayRef dilation, int64_t groups,
                                        c10::IntArrayRef mul_man_bits,
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
    odd_unavailable("custom_conv_superfp_mixed_odd");
#else
    constexpr const char *op = "custom_conv_superfp_mixed_odd";
    return run_custom_conv_mixed<Backend>(
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
        conv_pass, out_size, stride, padding, dilation, groups,
        accumulate_algorithm, round_mode);
#endif
  }

  template <class Backend>
  at::Tensor run_binaryK_conv_fma_mixed_odd(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                            int64_t conv_pass, c10::IntArrayRef out_size,
                                            c10::IntArrayRef stride, c10::IntArrayRef padding,
                                            c10::IntArrayRef dilation, int64_t groups,
                                            bool fma_quant, c10::IntArrayRef fma_K,
                                            c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias,
                                            bool fma_is_signed, int64_t accumulate_algorithm,
                                            int64_t round_mode, int64_t fma_saturation_mode,
                                            int64_t fma_subnormals_mode, int64_t fma_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_conv_binaryK_fma_mixed_odd");
#else
    constexpr const char *op = "custom_conv_binaryK_fma_mixed_odd";
    return run_custom_conv_mixed<Backend>(
        op,
        [&] {
          return odd_palette(pack_binaryK_fused_mixed(op, fma_K, fma_P, fma_bias, fma_is_signed,
                                                      fma_saturation_mode, fma_subnormals_mode,
                                                      fma_prng_bits));
        },
        a, b, prec_idx,
        conv_pass, out_size, stride, padding, dilation, groups,
        accumulate_algorithm, round_mode,
        [&] { check_fma_quant(fma_quant, op); });
#endif
  }

  template <class Backend>
  at::Tensor run_superfp_conv_fma_mixed_odd(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                            int64_t conv_pass, c10::IntArrayRef out_size,
                                            c10::IntArrayRef stride, c10::IntArrayRef padding,
                                            c10::IntArrayRef dilation, int64_t groups,
                                            bool fma_quant, c10::IntArrayRef fma_man_bits,
                                            c10::IntArrayRef fma_exp_bits,
                                            c10::IntArrayRef fma_normal_binades,
                                            c10::IntArrayRef fma_bias, bool fma_is_signed,
                                            int64_t accumulate_algorithm, int64_t round_mode,
                                            int64_t fma_saturation_mode, int64_t fma_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_conv_superfp_fma_mixed_odd");
#else
    constexpr const char *op = "custom_conv_superfp_fma_mixed_odd";
    return run_custom_conv_mixed<Backend>(
        op,
        [&] {
          return odd_palette(pack_superfp_fused_mixed(op, fma_man_bits, fma_exp_bits,
                                                      fma_normal_binades, fma_bias, fma_is_signed,
                                                      fma_saturation_mode, fma_prng_bits));
        },
        a, b, prec_idx,
        conv_pass, out_size, stride, padding, dilation, groups,
        accumulate_algorithm, round_mode,
        [&] { check_fma_quant(fma_quant, op); });
#endif
  }

  template <class Backend>
  at::Tensor run_binaryK_superfp_conv_odd(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                          c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                          c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                          int64_t groups, int64_t mul_K, int64_t mul_P,
                                          int64_t mul_bias, bool mul_is_signed,
                                          int64_t acc_man_bits, int64_t acc_exp_bits,
                                          int64_t acc_normal_binades, int64_t acc_bias,
                                          bool acc_is_signed, int64_t accumulate_algorithm,
                                          int64_t round_mode, int64_t mul_saturation_mode,
                                          int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
                                          int64_t mul_prng_bits, int64_t acc_prng_bits,
                                          int64_t block_size, bool outer_quant,
                                          int64_t outer_man_bits, int64_t outer_exp_bits,
                                          int64_t outer_normal_binades, int64_t outer_bias,
                                          bool outer_is_signed, int64_t outer_saturation_mode,
                                          int64_t outer_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_conv_binaryK_superfp_odd");
#else
    constexpr const char *op = "custom_conv_binaryK_superfp_odd";
    return run_custom_conv_odd<Backend>(
        op,
        OddArgs<BinaryKSuperfpArgs>{pack_binaryK_superfp(mul_K, mul_P, mul_bias, mul_is_signed,
                                                         acc_man_bits, acc_exp_bits,
                                                         acc_normal_binades, acc_bias,
                                                         acc_is_signed, mul_saturation_mode,
                                                         mul_subnormals_mode, acc_saturation_mode,
                                                         mul_prng_bits, acc_prng_bits)},
        pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                           outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                           outer_prng_bits),
        a, b, conv_pass, out_size, stride, padding, dilation, groups,
        accumulate_algorithm, round_mode);
#endif
  }

  template <class Backend>
  at::Tensor run_superfp_binaryK_conv_odd(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                          c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                          c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                          int64_t groups, int64_t mul_man_bits,
                                          int64_t mul_exp_bits, int64_t mul_normal_binades,
                                          int64_t mul_bias, bool mul_is_signed, int64_t acc_K,
                                          int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
                                          int64_t accumulate_algorithm, int64_t round_mode,
                                          int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                          int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                                          int64_t acc_prng_bits, int64_t block_size,
                                          bool outer_quant, int64_t outer_K, int64_t outer_P,
                                          int64_t outer_bias, bool outer_is_signed,
                                          int64_t outer_saturation_mode,
                                          int64_t outer_subnormals_mode, int64_t outer_prng_bits)
  {
#if defined(MPTORCH_NO_ODD)
    odd_unavailable("custom_conv_superfp_binaryK_odd");
#else
    constexpr const char *op = "custom_conv_superfp_binaryK_odd";
    return run_custom_conv_odd<Backend>(
        op,
        OddArgs<SuperfpBinaryKArgs>{pack_superfp_binaryK(mul_man_bits, mul_exp_bits,
                                                         mul_normal_binades, mul_bias,
                                                         mul_is_signed, acc_K, acc_P, acc_bias,
                                                         acc_is_signed, mul_saturation_mode,
                                                         acc_saturation_mode, acc_subnormals_mode,
                                                         mul_prng_bits, acc_prng_bits)},
        pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                           outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
        a, b, conv_pass, out_size, stride, padding, dilation, groups,
        accumulate_algorithm, round_mode);
#endif
  }
} // namespace mptorch::gemm
