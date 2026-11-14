// The CUDA entry points of the ten odd conv ops (custom_conv_*_odd;
// carrier_rounding=RoundMode.RO): their twins' conv entry points with the
// mac's carrier steps rounded to odd. The bodies are
// common/carrier_odd_conv_host.h's, shared with the CPU file; the kernels are
// instantiated by the custom_conv_{binaryK,superfp}{_fma,}_odd{,_f64} and
// custom_conv_cross_odd{,_f64} files.

#include "../common/carrier_odd_conv_host.h"
#include "../quant_ops.h"
#include "gemm_backend.h"

using at::Tensor;

namespace
{
  using Backend = mptorch::gemm_cuda::CudaBackend;
}

Tensor binaryK_conv_odd_cuda(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
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
  return mptorch::gemm::run_binaryK_conv_odd<Backend>(a, b, conv_pass, out_size, stride, padding,
                                                      dilation, groups, mul_K, mul_P, mul_bias,
                                                      mul_is_signed, accumulate_quant, acc_K, acc_P,
                                                      acc_bias, acc_is_signed, accumulate_algorithm,
                                                      round_mode, mul_saturation_mode,
                                                      mul_subnormals_mode, acc_saturation_mode,
                                                      acc_subnormals_mode, mul_prng_bits,
                                                      acc_prng_bits, block_size, outer_quant,
                                                      outer_K, outer_P, outer_bias, outer_is_signed,
                                                      outer_saturation_mode, outer_subnormals_mode,
                                                      outer_prng_bits);
}

Tensor superfp_conv_odd_cuda(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
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
  return mptorch::gemm::run_superfp_conv_odd<Backend>(a, b, conv_pass, out_size, stride, padding,
                                                      dilation, groups, mul_man_bits, mul_exp_bits,
                                                      mul_normal_binades, mul_bias, mul_is_signed,
                                                      accumulate_quant, acc_man_bits, acc_exp_bits,
                                                      acc_normal_binades, acc_bias, acc_is_signed,
                                                      accumulate_algorithm, round_mode,
                                                      mul_saturation_mode, acc_saturation_mode,
                                                      mul_prng_bits, acc_prng_bits, block_size,
                                                      outer_quant, outer_man_bits, outer_exp_bits,
                                                      outer_normal_binades, outer_bias,
                                                      outer_is_signed, outer_saturation_mode,
                                                      outer_prng_bits);
}

Tensor binaryK_conv_fma_odd_cuda(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
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
  return mptorch::gemm::run_binaryK_conv_fma_odd<Backend>(a, b, conv_pass, out_size, stride,
                                                          padding, dilation, groups, fma_quant,
                                                          fma_K, fma_P, fma_bias, fma_is_signed,
                                                          accumulate_algorithm, round_mode,
                                                          fma_saturation_mode, fma_subnormals_mode,
                                                          fma_prng_bits, block_size, outer_quant,
                                                          outer_K, outer_P, outer_bias,
                                                          outer_is_signed, outer_saturation_mode,
                                                          outer_subnormals_mode, outer_prng_bits);
}

Tensor superfp_conv_fma_odd_cuda(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
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
  return mptorch::gemm::run_superfp_conv_fma_odd<Backend>(a, b, conv_pass, out_size, stride,
                                                          padding, dilation, groups, fma_quant,
                                                          fma_man_bits, fma_exp_bits,
                                                          fma_normal_binades, fma_bias,
                                                          fma_is_signed, accumulate_algorithm,
                                                          round_mode, fma_saturation_mode,
                                                          fma_prng_bits, block_size, outer_quant,
                                                          outer_man_bits, outer_exp_bits,
                                                          outer_normal_binades, outer_bias,
                                                          outer_is_signed, outer_saturation_mode,
                                                          outer_prng_bits);
}

Tensor binaryK_conv_mixed_odd_cuda(Tensor a, Tensor b, Tensor prec_idx, int64_t conv_pass,
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
  return mptorch::gemm::run_binaryK_conv_mixed_odd<Backend>(a, b, prec_idx, conv_pass, out_size,
                                                            stride, padding, dilation, groups,
                                                            mul_K, mul_P, mul_bias, mul_is_signed,
                                                            accumulate_quant, acc_K, acc_P,
                                                            acc_bias, acc_is_signed,
                                                            accumulate_algorithm, round_mode,
                                                            mul_saturation_mode,
                                                            mul_subnormals_mode,
                                                            acc_saturation_mode,
                                                            acc_subnormals_mode, mul_prng_bits,
                                                            acc_prng_bits);
}

Tensor superfp_conv_mixed_odd_cuda(Tensor a, Tensor b, Tensor prec_idx, int64_t conv_pass,
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
  return mptorch::gemm::run_superfp_conv_mixed_odd<Backend>(a, b, prec_idx, conv_pass, out_size,
                                                            stride, padding, dilation, groups,
                                                            mul_man_bits, mul_exp_bits,
                                                            mul_normal_binades, mul_bias,
                                                            mul_is_signed, accumulate_quant,
                                                            acc_man_bits, acc_exp_bits,
                                                            acc_normal_binades, acc_bias,
                                                            acc_is_signed, accumulate_algorithm,
                                                            round_mode, mul_saturation_mode,
                                                            acc_saturation_mode, mul_prng_bits,
                                                            acc_prng_bits);
}

Tensor binaryK_conv_fma_mixed_odd_cuda(Tensor a, Tensor b, Tensor prec_idx, int64_t conv_pass,
                                       c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                       c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                       int64_t groups, bool fma_quant, c10::IntArrayRef fma_K,
                                       c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias,
                                       bool fma_is_signed, int64_t accumulate_algorithm,
                                       int64_t round_mode, int64_t fma_saturation_mode,
                                       int64_t fma_subnormals_mode, int64_t fma_prng_bits)
{
  return mptorch::gemm::run_binaryK_conv_fma_mixed_odd<Backend>(a, b, prec_idx, conv_pass, out_size,
                                                                stride, padding, dilation, groups,
                                                                fma_quant, fma_K, fma_P, fma_bias,
                                                                fma_is_signed, accumulate_algorithm,
                                                                round_mode, fma_saturation_mode,
                                                                fma_subnormals_mode, fma_prng_bits);
}

Tensor superfp_conv_fma_mixed_odd_cuda(Tensor a, Tensor b, Tensor prec_idx, int64_t conv_pass,
                                       c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                       c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                       int64_t groups, bool fma_quant,
                                       c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                       c10::IntArrayRef fma_normal_binades,
                                       c10::IntArrayRef fma_bias, bool fma_is_signed,
                                       int64_t accumulate_algorithm, int64_t round_mode,
                                       int64_t fma_saturation_mode, int64_t fma_prng_bits)
{
  return mptorch::gemm::run_superfp_conv_fma_mixed_odd<Backend>(a, b, prec_idx, conv_pass, out_size,
                                                                stride, padding, dilation, groups,
                                                                fma_quant, fma_man_bits,
                                                                fma_exp_bits, fma_normal_binades,
                                                                fma_bias, fma_is_signed,
                                                                accumulate_algorithm, round_mode,
                                                                fma_saturation_mode, fma_prng_bits);
}

Tensor binaryK_superfp_conv_odd_cuda(Tensor a, Tensor b, int64_t conv_pass,
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
  return mptorch::gemm::run_binaryK_superfp_conv_odd<Backend>(a, b, conv_pass, out_size, stride,
                                                              padding, dilation, groups, mul_K,
                                                              mul_P, mul_bias, mul_is_signed,
                                                              acc_man_bits, acc_exp_bits,
                                                              acc_normal_binades, acc_bias,
                                                              acc_is_signed, accumulate_algorithm,
                                                              round_mode, mul_saturation_mode,
                                                              mul_subnormals_mode,
                                                              acc_saturation_mode, mul_prng_bits,
                                                              acc_prng_bits, block_size,
                                                              outer_quant, outer_man_bits,
                                                              outer_exp_bits, outer_normal_binades,
                                                              outer_bias, outer_is_signed,
                                                              outer_saturation_mode,
                                                              outer_prng_bits);
}

Tensor superfp_binaryK_conv_odd_cuda(Tensor a, Tensor b, int64_t conv_pass,
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
  return mptorch::gemm::run_superfp_binaryK_conv_odd<Backend>(a, b, conv_pass, out_size, stride,
                                                              padding, dilation, groups,
                                                              mul_man_bits, mul_exp_bits,
                                                              mul_normal_binades, mul_bias,
                                                              mul_is_signed, acc_K, acc_P, acc_bias,
                                                              acc_is_signed, accumulate_algorithm,
                                                              round_mode, mul_saturation_mode,
                                                              acc_saturation_mode,
                                                              acc_subnormals_mode, mul_prng_bits,
                                                              acc_prng_bits, block_size,
                                                              outer_quant, outer_K, outer_P,
                                                              outer_bias, outer_is_signed,
                                                              outer_saturation_mode,
                                                              outer_subnormals_mode,
                                                              outer_prng_bits);
}
