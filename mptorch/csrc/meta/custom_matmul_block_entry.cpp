// The Meta kernel of the block GEMM, whose CPU entry point is in
// cpu/custom_matmul_block_entry.cpp. Its body is the CPU entry point's with
// the driver replaced by its meta_* twin from common/meta_host.h.
// meta_ops.cpp registers it.

#include "../common/meta_host.h"
#include "meta_ops.h"

using at::Tensor;
using namespace mptorch::gemm;
using namespace mptorch::meta;

Tensor block_matmul_meta(Tensor a_data, Tensor a_scales, int64_t a_cols, bool trans_a,
                         Tensor b_data, Tensor b_scales, int64_t b_cols, bool trans_b,
                         c10::IntArrayRef a_fmt, double a_elem_max, double a_scale_max,
                         double a_tensor_scale, c10::IntArrayRef b_fmt, double b_elem_max,
                         double b_scale_max, double b_tensor_scale, bool fused,
                         bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias,
                         bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
                         int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                         int64_t acc_prng_bits, int64_t block_size, bool outer_quant,
                         int64_t outer_K, int64_t outer_P, int64_t outer_bias,
                         bool outer_is_signed, int64_t outer_saturation_mode,
                         int64_t outer_subnormals_mode, int64_t outer_prng_bits, bool binary64)
{
  const BlockGemmCall call{a_data,     a_scales,    a_cols,         trans_a,  b_data,
                           b_scales,   b_cols,      trans_b,        a_fmt,    a_elem_max,
                           a_scale_max, a_tensor_scale, b_fmt,      b_elem_max, b_scale_max,
                           b_tensor_scale, binary64};
  return meta_custom_matmul_block_entry(
      call, fused, accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed, accumulate_algorithm,
      round_mode, acc_saturation_mode, acc_subnormals_mode, acc_prng_bits,
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits));
}
