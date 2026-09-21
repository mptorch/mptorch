// The MPS entry points of the four accumulated GEMM ops (the single-format
// ops under KAHAN, BLOCK or TREE; common/gemm_accumulate.h): functions that
// raise. The Metal kernels are dev/continuation_plan.md's phase H. They are
// registered anyway (mps_ops.cpp) because an op with no MPS kernel fails at
// dispatch with a message that names nothing a caller can act on, where this
// one names the two ways out. No argument is read, so none is named.

#include "../quant_ops.h"
#include <c10/util/Exception.h>

namespace
{
  at::Tensor no_metal_kernel(const char *op_name)
  {
    TORCH_CHECK(false, op_name, ": AccumulateAlgorithm KAHAN, BLOCK and TREE have no MPS kernel "
                                "yet (dev/continuation_plan.md, phase H); run the GEMM on a CPU or "
                                "CUDA tensor, or use AccumulateAlgorithm.NAIVE");
    return at::Tensor();
  }
} // namespace

at::Tensor binaryK_matmul_accumulated_mps(
    at::Tensor, at::Tensor, bool, bool, int64_t, int64_t, int64_t, bool, bool, int64_t,
    int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t, bool, int64_t, int64_t,
    int64_t)
{
  return no_metal_kernel("custom_matmul_binaryK_accumulated");
}

at::Tensor superfp_matmul_accumulated_mps(
    at::Tensor, at::Tensor, bool, bool, int64_t, int64_t, int64_t, int64_t, bool, bool,
    int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t, bool, int64_t,
    int64_t)
{
  return no_metal_kernel("custom_matmul_superfp_accumulated");
}

at::Tensor binaryK_matmul_fma_accumulated_mps(
    at::Tensor, at::Tensor, bool, bool, bool, int64_t, int64_t, int64_t, bool, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t, bool,
    int64_t, int64_t, int64_t)
{
  return no_metal_kernel("custom_matmul_binaryK_fma_accumulated");
}

at::Tensor superfp_matmul_fma_accumulated_mps(
    at::Tensor, at::Tensor, bool, bool, bool, int64_t, int64_t, int64_t, int64_t, bool,
    int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t,
    int64_t, bool, int64_t, int64_t)
{
  return no_metal_kernel("custom_matmul_superfp_fma_accumulated");
}
