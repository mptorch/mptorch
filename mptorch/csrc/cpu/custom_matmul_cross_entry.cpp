// The CPU entry points of the six cross-family ops (common/gemm_cross.h): a
// split mac whose products are rounded to a format of one family and whose
// partial sums to a format of the other. The NAIVE pair, its *_accumulated
// twins (KAHAN, BLOCK and TREE) and the conv pair, each one packer call and
// one call of a driver every other op shares: run_custom_matmul,
// run_custom_matmul_accumulated and run_custom_conv_single.
//
// A file of its own, so that the entry points of the other ops are the text
// they were. Its kernels are instantiated by custom_matmul_cross{,_kahan,
// _block,_tree}.cpp and custom_conv_cross{,_kahan,_block,_tree}.cpp.

#include "../common/gemm_cross_host.h"
#include "../quant_ops.h"
#include "gemm_backend.h"

using at::Tensor;
using namespace mptorch::gemm;

namespace
{
  using Backend = mptorch::gemm_cpu::CpuBackend;
}

Tensor binaryK_superfp_matmul_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                  int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                                  int64_t acc_man_bits, int64_t acc_exp_bits,
                                  int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                  int64_t acc_saturation_mode, int64_t mul_prng_bits,
                                  int64_t acc_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_binaryK_superfp",
      pack_binaryK_superfp(mul_K, mul_P, mul_bias, mul_is_signed, acc_man_bits, acc_exp_bits,
                           acc_normal_binades, acc_bias, acc_is_signed, mul_saturation_mode,
                           mul_subnormals_mode, acc_saturation_mode, mul_prng_bits, acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_binaryK_matmul_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                  int64_t mul_man_bits, int64_t mul_exp_bits,
                                  int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
                                  int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                  int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                                  int64_t acc_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_superfp_binaryK",
      pack_superfp_binaryK(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                           acc_K, acc_P, acc_bias, acc_is_signed, mul_saturation_mode,
                           acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_superfp_matmul_accumulated_cpu(
    Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias,
    bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades,
    int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
    int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant,
    int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades,
    int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
    int64_t outer_prng_bits)
{
  return run_custom_matmul_accumulated<Backend>(
      "custom_matmul_binaryK_superfp_accumulated",
      pack_binaryK_superfp(mul_K, mul_P, mul_bias, mul_is_signed, acc_man_bits, acc_exp_bits,
                           acc_normal_binades, acc_bias, acc_is_signed, mul_saturation_mode,
                           mul_subnormals_mode, acc_saturation_mode, mul_prng_bits, acc_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                         outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_binaryK_matmul_accumulated_cpu(
    Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits,
    int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P,
    int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
    int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant,
    int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed,
    int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return run_custom_matmul_accumulated<Backend>(
      "custom_matmul_superfp_binaryK_accumulated",
      pack_superfp_binaryK(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                           acc_K, acc_P, acc_bias, acc_is_signed, mul_saturation_mode,
                           acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_superfp_conv_cpu(
    Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride,
    c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K,
    int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits,
    int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode,
    int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits,
    int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits,
    int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
    bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return run_custom_conv_single<Backend>(
      "custom_conv_binaryK_superfp",
      pack_binaryK_superfp(mul_K, mul_P, mul_bias, mul_is_signed, acc_man_bits, acc_exp_bits,
                           acc_normal_binades, acc_bias, acc_is_signed, mul_saturation_mode,
                           mul_subnormals_mode, acc_saturation_mode, mul_prng_bits, acc_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                         outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups, accumulate_algorithm,
      round_mode);
}

Tensor superfp_binaryK_conv_cpu(
    Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride,
    c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits,
    int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
    int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode,
    int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits,
    int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K,
    int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
    int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return run_custom_conv_single<Backend>(
      "custom_conv_superfp_binaryK",
      pack_superfp_binaryK(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                           acc_K, acc_P, acc_bias, acc_is_signed, mul_saturation_mode,
                           acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups, accumulate_algorithm,
      round_mode);
}
