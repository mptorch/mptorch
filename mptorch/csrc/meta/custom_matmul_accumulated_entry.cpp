// The Meta kernels of the *_accumulated GEMM ops (KAHAN, BLOCK and TREE), in
// the order of their CPU entry points in
// cpu/custom_matmul_accumulated_entry.cpp. Each body is the CPU entry point's
// with the driver replaced by its meta_* twin from common/meta_host.h.
// meta_ops.cpp registers them.

#include "../common/meta_host.h"
#include "meta_ops.h"

using at::Tensor;
using namespace mptorch::gemm;
using namespace mptorch::meta;

Tensor binaryK_matmul_accumulated_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                       int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                                       bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                                       int64_t acc_bias, bool acc_is_signed,
                                       int64_t accumulate_algorithm, int64_t round_mode,
                                       int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                       int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                       int64_t mul_prng_bits, int64_t acc_prng_bits,
                                      int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                      int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                      int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return meta_custom_matmul_accumulated(
      "custom_matmul_binaryK_accumulated",
      pack_binaryK_split(mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P,
                         acc_bias, acc_is_signed, mul_saturation_mode, mul_subnormals_mode,
                         acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_matmul_fma_accumulated_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                           bool fma_quant, int64_t fma_K, int64_t fma_P,
                                           int64_t fma_bias, bool fma_is_signed,
                                           int64_t accumulate_algorithm, int64_t round_mode,
                                           int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                           int64_t fma_prng_bits,
                                          int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                          int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                          int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return meta_custom_matmul_accumulated(
      "custom_matmul_binaryK_fma_accumulated",
      pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias, fma_is_signed, fma_saturation_mode,
                         fma_subnormals_mode, fma_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_matmul_accumulated_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                       int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                                       int64_t mul_bias, bool mul_is_signed,
                                       bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                                       int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                                       int64_t accumulate_algorithm, int64_t round_mode,
                                       int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                       int64_t mul_prng_bits, int64_t acc_prng_bits,
                                      int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                      int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                      bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return meta_custom_matmul_accumulated(
      "custom_matmul_superfp_accumulated",
      pack_superfp_split(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                         accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades,
                         acc_bias, acc_is_signed, mul_saturation_mode, acc_saturation_mode,
                         mul_prng_bits, acc_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_matmul_fma_accumulated_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                           bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                           int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                           int64_t accumulate_algorithm, int64_t round_mode,
                                           int64_t fma_saturation_mode, int64_t fma_prng_bits,
                                          int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                          int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                          bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return meta_custom_matmul_accumulated(
      "custom_matmul_superfp_fma_accumulated",
      pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias,
                         fma_is_signed, fma_saturation_mode, fma_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}
