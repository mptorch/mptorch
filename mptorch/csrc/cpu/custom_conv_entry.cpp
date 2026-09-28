// The CPU entry points of the eight conv ops: one pass of a convolution
// (forward, input gradient or weight gradient) as the GEMM of the op each is
// named after, with the operands gathered from the convolution's tensors in
// the kernel's tile loads rather than unfolded (common/gemm_gather.h).
//
// Each is its GEMM twin's packer handed to the conv driver
// (common/gemm_conv_host.h): the single-format ones take the *_accumulated
// twin's arguments, so every AccumulateAlgorithm, and the palette ones the
// *_mixed twin's. A file of its own, like custom_matmul_accumulated_entry.cpp,
// so that the GEMM entry points compile to what they did.

#include "../common/gemm_conv_host.h"
#include "../quant_ops.h"
#include "gemm_backend.h"

using at::Tensor;
using namespace mptorch::gemm;

namespace
{
  using Backend = mptorch::gemm_cpu::CpuBackend;
}

Tensor binaryK_conv_cpu(Tensor a, Tensor b, int64_t conv_pass,
                        c10::IntArrayRef out_size, c10::IntArrayRef stride,
                        c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                        int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                        bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias,
                        bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
                        int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                        int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                        int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size,
                        bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias,
                        bool outer_is_signed, int64_t outer_saturation_mode,
                        int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return run_custom_conv_single<Backend>(
      "custom_conv_binaryK",
      pack_binaryK_split(mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P,
                         acc_bias, acc_is_signed, mul_saturation_mode, mul_subnormals_mode,
                         acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor binaryK_conv_mixed_cpu(Tensor a, Tensor b, Tensor prec_idx,
                              int64_t conv_pass, c10::IntArrayRef out_size,
                              c10::IntArrayRef stride, c10::IntArrayRef padding,
                              c10::IntArrayRef dilation, int64_t groups, c10::IntArrayRef mul_K,
                              c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias,
                              bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_K,
                              c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias,
                              bool acc_is_signed, int64_t accumulate_algorithm,
                              int64_t round_mode, int64_t mul_saturation_mode,
                              int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
                              int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                              int64_t acc_prng_bits)
{
  constexpr const char *op = "custom_conv_binaryK_mixed";
  return run_custom_conv_mixed<Backend>(
      op,
      [&] {
        return pack_binaryK_split_mixed(op, mul_K, mul_P, mul_bias, mul_is_signed,
                                        accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed,
                                        mul_saturation_mode, mul_subnormals_mode,
                                        acc_saturation_mode, acc_subnormals_mode, mul_prng_bits,
                                        acc_prng_bits);
      },
      a, b, prec_idx, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor binaryK_conv_fma_cpu(Tensor a, Tensor b, int64_t conv_pass,
                            c10::IntArrayRef out_size, c10::IntArrayRef stride,
                            c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                            bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias,
                            bool fma_is_signed, int64_t accumulate_algorithm,
                            int64_t round_mode, int64_t fma_saturation_mode,
                            int64_t fma_subnormals_mode, int64_t fma_prng_bits,
                            int64_t block_size, bool outer_quant, int64_t outer_K,
                            int64_t outer_P, int64_t outer_bias, bool outer_is_signed,
                            int64_t outer_saturation_mode, int64_t outer_subnormals_mode,
                            int64_t outer_prng_bits)
{
  return run_custom_conv_single<Backend>(
      "custom_conv_binaryK_fma",
      pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias, fma_is_signed, fma_saturation_mode,
                         fma_subnormals_mode, fma_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor binaryK_conv_fma_mixed_cpu(Tensor a, Tensor b, Tensor prec_idx,
                                  int64_t conv_pass, c10::IntArrayRef out_size,
                                  c10::IntArrayRef stride, c10::IntArrayRef padding,
                                  c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                  c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                  c10::IntArrayRef fma_bias, bool fma_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                  int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_conv_binaryK_fma_mixed";
  return run_custom_conv_mixed<Backend>(
      op,
      [&] {
        return pack_binaryK_fused_mixed(op, fma_K, fma_P, fma_bias, fma_is_signed,
                                        fma_saturation_mode, fma_subnormals_mode, fma_prng_bits);
      },
      a, b, prec_idx, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode,
      [&] {
        TORCH_CHECK(fma_quant, op, ": fma_quant=false has no per-element format to vary; "
                                   "use custom_conv_binaryK_fma");
      });
}

Tensor superfp_conv_cpu(Tensor a, Tensor b, int64_t conv_pass,
                        c10::IntArrayRef out_size, c10::IntArrayRef stride,
                        c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                        int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                        int64_t mul_bias, bool mul_is_signed, bool accumulate_quant,
                        int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades,
                        int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm,
                        int64_t round_mode, int64_t mul_saturation_mode,
                        int64_t acc_saturation_mode, int64_t mul_prng_bits,
                        int64_t acc_prng_bits, int64_t block_size, bool outer_quant,
                        int64_t outer_man_bits, int64_t outer_exp_bits,
                        int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed,
                        int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return run_custom_conv_single<Backend>(
      "custom_conv_superfp",
      pack_superfp_split(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                         accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades,
                         acc_bias, acc_is_signed, mul_saturation_mode, acc_saturation_mode,
                         mul_prng_bits, acc_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                         outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor superfp_conv_mixed_cpu(Tensor a, Tensor b, Tensor prec_idx,
                              int64_t conv_pass, c10::IntArrayRef out_size,
                              c10::IntArrayRef stride, c10::IntArrayRef padding,
                              c10::IntArrayRef dilation, int64_t groups,
                              c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits,
                              c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias,
                              bool mul_is_signed, bool accumulate_quant,
                              c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits,
                              c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias,
                              bool acc_is_signed, int64_t accumulate_algorithm,
                              int64_t round_mode, int64_t mul_saturation_mode,
                              int64_t acc_saturation_mode, int64_t mul_prng_bits,
                              int64_t acc_prng_bits)
{
  constexpr const char *op = "custom_conv_superfp_mixed";
  return run_custom_conv_mixed<Backend>(
      op,
      [&] {
        return pack_superfp_split_mixed(op, mul_man_bits, mul_exp_bits, mul_normal_binades,
                                        mul_bias, mul_is_signed, accumulate_quant, acc_man_bits,
                                        acc_exp_bits, acc_normal_binades, acc_bias, acc_is_signed,
                                        mul_saturation_mode, acc_saturation_mode, mul_prng_bits,
                                        acc_prng_bits);
      },
      a, b, prec_idx, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor superfp_conv_fma_cpu(Tensor a, Tensor b, int64_t conv_pass,
                            c10::IntArrayRef out_size, c10::IntArrayRef stride,
                            c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                            bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                            int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                            int64_t accumulate_algorithm, int64_t round_mode,
                            int64_t fma_saturation_mode, int64_t fma_prng_bits,
                            int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                            int64_t outer_exp_bits, int64_t outer_normal_binades,
                            int64_t outer_bias, bool outer_is_signed,
                            int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return run_custom_conv_single<Backend>(
      "custom_conv_superfp_fma",
      pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias,
                         fma_is_signed, fma_saturation_mode, fma_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                         outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor superfp_conv_fma_mixed_cpu(Tensor a, Tensor b, Tensor prec_idx,
                                  int64_t conv_pass, c10::IntArrayRef out_size,
                                  c10::IntArrayRef stride, c10::IntArrayRef padding,
                                  c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                  c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                  c10::IntArrayRef fma_normal_binades,
                                  c10::IntArrayRef fma_bias, bool fma_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t fma_saturation_mode, int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_conv_superfp_fma_mixed";
  return run_custom_conv_mixed<Backend>(
      op,
      [&] {
        return pack_superfp_fused_mixed(op, fma_man_bits, fma_exp_bits, fma_normal_binades,
                                        fma_bias, fma_is_signed, fma_saturation_mode,
                                        fma_prng_bits);
      },
      a, b, prec_idx, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode,
      [&] {
        TORCH_CHECK(fma_quant, op, ": fma_quant=false has no per-element format to vary; "
                                   "use custom_conv_superfp_fma");
      });
}
