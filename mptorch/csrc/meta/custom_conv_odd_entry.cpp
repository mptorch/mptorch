// The Meta kernels of the conv ops whose carrier step rounds to odd, in the
// order of their CPU entry points in cpu/custom_conv_odd_entry.cpp. Each CPU
// entry point forwards to a body in common/carrier_odd_conv_host.h, and each
// Meta kernel here is that body with the driver replaced by its meta_* twin
// from common/meta_host.h.
// tests/test_entry_copies.py checks that they stay copies. meta_ops.cpp
// registers them.

#include "../common/carrier_odd_accumulate.h"
#include "../common/carrier_odd_mixed.h"
#include "../common/meta_host.h"
#include "meta_ops.h"

using at::Tensor;
using namespace mptorch::gemm;
using namespace mptorch::meta;

Tensor binaryK_conv_odd_meta(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
                             c10::IntArrayRef stride, c10::IntArrayRef padding,
                             c10::IntArrayRef dilation, int64_t groups, int64_t mul_K,
                             int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                             bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias,
                             bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
                             int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                             int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                             int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size,
                             bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias,
                             bool outer_is_signed, int64_t outer_saturation_mode,
                             int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
#if defined(MPTORCH_NO_ODD)
  odd_unavailable("custom_conv_binaryK_odd");
#else
  constexpr const char *op = "custom_conv_binaryK_odd";
  return meta_custom_conv_odd(
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

Tensor superfp_conv_odd_meta(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
                             c10::IntArrayRef stride, c10::IntArrayRef padding,
                             c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits,
                             int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias,
                             bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits,
                             int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias,
                             bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
                             int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                             int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size,
                             bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits,
                             int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed,
                             int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
#if defined(MPTORCH_NO_ODD)
  odd_unavailable("custom_conv_superfp_odd");
#else
  constexpr const char *op = "custom_conv_superfp_odd";
  return meta_custom_conv_odd(
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

Tensor binaryK_conv_fma_odd_meta(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
                                 c10::IntArrayRef stride, c10::IntArrayRef padding,
                                 c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                 int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed,
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
  return meta_custom_conv_odd(
      op,
      OddArgs<BinaryKFusedArgs>{pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias, fma_is_signed,
                                                   fma_saturation_mode, fma_subnormals_mode,
                                                   fma_prng_bits)},
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
#endif
}

Tensor superfp_conv_fma_odd_meta(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
                                 c10::IntArrayRef stride, c10::IntArrayRef padding,
                                 c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                 int64_t fma_man_bits, int64_t fma_exp_bits,
                                 int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
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
  return meta_custom_conv_odd(
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

Tensor binaryK_conv_mixed_odd_meta(Tensor a, Tensor b, Tensor prec_idx, int64_t conv_pass,
                                   c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                   c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                   int64_t groups, c10::IntArrayRef mul_K, c10::IntArrayRef mul_P,
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
  return meta_custom_conv_mixed(
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

Tensor superfp_conv_mixed_odd_meta(Tensor a, Tensor b, Tensor prec_idx, int64_t conv_pass,
                                   c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                   c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                   int64_t groups, c10::IntArrayRef mul_man_bits,
                                   c10::IntArrayRef mul_exp_bits,
                                   c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias,
                                   bool mul_is_signed, bool accumulate_quant,
                                   c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits,
                                   c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias,
                                   bool acc_is_signed, int64_t accumulate_algorithm,
                                   int64_t round_mode, int64_t mul_saturation_mode,
                                   int64_t acc_saturation_mode, int64_t mul_prng_bits,
                                   int64_t acc_prng_bits)
{
#if defined(MPTORCH_NO_ODD)
  odd_unavailable("custom_conv_superfp_mixed_odd");
#else
  constexpr const char *op = "custom_conv_superfp_mixed_odd";
  return meta_custom_conv_mixed(
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

Tensor binaryK_conv_fma_mixed_odd_meta(Tensor a, Tensor b, Tensor prec_idx, int64_t conv_pass,
                                       c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                       c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                       int64_t groups, bool fma_quant, c10::IntArrayRef fma_K,
                                       c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias,
                                       bool fma_is_signed, int64_t accumulate_algorithm,
                                       int64_t round_mode, int64_t fma_saturation_mode,
                                       int64_t fma_subnormals_mode, int64_t fma_prng_bits)
{
#if defined(MPTORCH_NO_ODD)
  odd_unavailable("custom_conv_binaryK_fma_mixed_odd");
#else
  constexpr const char *op = "custom_conv_binaryK_fma_mixed_odd";
  return meta_custom_conv_mixed(
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

Tensor superfp_conv_fma_mixed_odd_meta(Tensor a, Tensor b, Tensor prec_idx, int64_t conv_pass,
                                       c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                       c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                       int64_t groups, bool fma_quant,
                                       c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                       c10::IntArrayRef fma_normal_binades,
                                       c10::IntArrayRef fma_bias, bool fma_is_signed,
                                       int64_t accumulate_algorithm, int64_t round_mode,
                                       int64_t fma_saturation_mode, int64_t fma_prng_bits)
{
#if defined(MPTORCH_NO_ODD)
  odd_unavailable("custom_conv_superfp_fma_mixed_odd");
#else
  constexpr const char *op = "custom_conv_superfp_fma_mixed_odd";
  return meta_custom_conv_mixed(
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

Tensor binaryK_superfp_conv_odd_meta(Tensor a, Tensor b, int64_t conv_pass,
                                     c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                     c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                     int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias,
                                     bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits,
                                     int64_t acc_normal_binades, int64_t acc_bias,
                                     bool acc_is_signed, int64_t accumulate_algorithm,
                                     int64_t round_mode, int64_t mul_saturation_mode,
                                     int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
                                     int64_t mul_prng_bits, int64_t acc_prng_bits,
                                     int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                     int64_t outer_exp_bits, int64_t outer_normal_binades,
                                     int64_t outer_bias, bool outer_is_signed,
                                     int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
#if defined(MPTORCH_NO_ODD)
  odd_unavailable("custom_conv_binaryK_superfp_odd");
#else
  constexpr const char *op = "custom_conv_binaryK_superfp_odd";
  return meta_custom_conv_odd(
      op,
      OddArgs<BinaryKSuperfpArgs>{pack_binaryK_superfp(mul_K, mul_P, mul_bias, mul_is_signed,
                                                       acc_man_bits, acc_exp_bits,
                                                       acc_normal_binades, acc_bias, acc_is_signed,
                                                       mul_saturation_mode, mul_subnormals_mode,
                                                       acc_saturation_mode, mul_prng_bits,
                                                       acc_prng_bits)},
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                         outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
#endif
}

Tensor superfp_binaryK_conv_odd_meta(Tensor a, Tensor b, int64_t conv_pass,
                                     c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                     c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                     int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits,
                                     int64_t mul_normal_binades, int64_t mul_bias,
                                     bool mul_is_signed, int64_t acc_K, int64_t acc_P,
                                     int64_t acc_bias, bool acc_is_signed,
                                     int64_t accumulate_algorithm, int64_t round_mode,
                                     int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                     int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                                     int64_t acc_prng_bits, int64_t block_size, bool outer_quant,
                                     int64_t outer_K, int64_t outer_P, int64_t outer_bias,
                                     bool outer_is_signed, int64_t outer_saturation_mode,
                                     int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
#if defined(MPTORCH_NO_ODD)
  odd_unavailable("custom_conv_superfp_binaryK_odd");
#else
  constexpr const char *op = "custom_conv_superfp_binaryK_odd";
  return meta_custom_conv_odd(
      op,
      OddArgs<SuperfpBinaryKArgs>{pack_superfp_binaryK(mul_man_bits, mul_exp_bits,
                                                       mul_normal_binades, mul_bias, mul_is_signed,
                                                       acc_K, acc_P, acc_bias, acc_is_signed,
                                                       mul_saturation_mode, acc_saturation_mode,
                                                       acc_subnormals_mode, mul_prng_bits,
                                                       acc_prng_bits)},
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
#endif
}
