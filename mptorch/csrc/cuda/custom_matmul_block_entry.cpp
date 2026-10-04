// The CUDA entry point of the block GEMM, custom_matmul_block
// (common/gemm_block.h): its schema handed to the driver in
// common/gemm_block_host.h. A file of its own, like
// custom_matmul_accumulated_entry.cpp, so that the other GEMM entry points
// compile to what they did.

#include "../common/gemm_block_host.h"
#include "../quant_ops.h"
#include "gemm_backend.h"

using at::Tensor;
using namespace mptorch::gemm;

Tensor block_matmul_cuda(Tensor a_data, Tensor a_scales, int64_t a_cols, bool trans_a, Tensor b_data,
                        Tensor b_scales, int64_t b_cols, bool trans_b, c10::IntArrayRef a_fmt,
                        double a_elem_max, double a_scale_max, double a_tensor_scale,
                        c10::IntArrayRef b_fmt, double b_elem_max, double b_scale_max,
                        double b_tensor_scale, bool fused, bool accumulate_quant, int64_t acc_K,
                        int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
                        int64_t accumulate_algorithm, int64_t round_mode,
                        int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                        int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K,
                        int64_t outer_P, int64_t outer_bias, bool outer_is_signed,
                        int64_t outer_saturation_mode, int64_t outer_subnormals_mode,
                        int64_t outer_prng_bits)
{
  const BlockGemmCall call{a_data,     a_scales,    a_cols,         trans_a,  b_data,
                           b_scales,   b_cols,      trans_b,        a_fmt,    a_elem_max,
                           a_scale_max, a_tensor_scale, b_fmt,      b_elem_max, b_scale_max,
                           b_tensor_scale};
  return custom_matmul_block_entry<mptorch::gemm_cuda::CudaBackend>(
      call, fused, accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed, accumulate_algorithm,
      round_mode, acc_saturation_mode, acc_subnormals_mode, acc_prng_bits,
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits));
}
