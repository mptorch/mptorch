// The MPS entry points of the eight conv ops (common/gemm_gather.h):
// functions that raise. The gathered tile loads for Metal are
// dev/continuation_plan.md's phase H. They are registered anyway
// (mps_ops.cpp) because an op with no MPS kernel fails at dispatch with a
// message that names nothing a caller can act on, where this one names the
// way out. No argument is read, so none is named.

#include "../quant_ops.h"
#include <c10/util/Exception.h>

namespace
{
  at::Tensor no_metal_kernel(const char *op_name)
  {
    TORCH_CHECK(false, op_name, ": the conv ops have no MPS kernel yet "
                                "(dev/continuation_plan.md, phase H); run the convolution on a "
                                "CPU or CUDA tensor");
    return at::Tensor();
  }
} // namespace

at::Tensor binaryK_conv_mps(
    at::Tensor, at::Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, int64_t, int64_t, int64_t, int64_t, bool, bool, int64_t, int64_t, int64_t,
    bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool,
    int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t)
{
  return no_metal_kernel("custom_conv_binaryK");
}

at::Tensor binaryK_conv_mixed_mps(
    at::Tensor, at::Tensor, at::Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, c10::IntArrayRef, int64_t, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, bool, bool, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, bool,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)
{
  return no_metal_kernel("custom_conv_binaryK_mixed");
}

at::Tensor binaryK_conv_fma_mps(
    at::Tensor, at::Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, int64_t, bool, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t)
{
  return no_metal_kernel("custom_conv_binaryK_fma");
}

at::Tensor binaryK_conv_fma_mixed_mps(
    at::Tensor, at::Tensor, at::Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, c10::IntArrayRef, int64_t, bool, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, bool, int64_t, int64_t, int64_t, int64_t, int64_t)
{
  return no_metal_kernel("custom_conv_binaryK_fma_mixed");
}

at::Tensor superfp_conv_mps(
    at::Tensor, at::Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool, int64_t, int64_t,
    int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool,
    int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t)
{
  return no_metal_kernel("custom_conv_superfp");
}

at::Tensor superfp_conv_mixed_mps(
    at::Tensor, at::Tensor, at::Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, c10::IntArrayRef, int64_t, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, c10::IntArrayRef, bool, bool, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, c10::IntArrayRef, bool, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t)
{
  return no_metal_kernel("custom_conv_superfp_mixed");
}

at::Tensor superfp_conv_fma_mps(
    at::Tensor, at::Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, int64_t, bool, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t,
    int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t)
{
  return no_metal_kernel("custom_conv_superfp_fma");
}

at::Tensor superfp_conv_fma_mixed_mps(
    at::Tensor, at::Tensor, at::Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, c10::IntArrayRef, int64_t, bool, c10::IntArrayRef, c10::IntArrayRef,
    c10::IntArrayRef, c10::IntArrayRef, bool, int64_t, int64_t, int64_t, int64_t)
{
  return no_metal_kernel("custom_conv_superfp_fma_mixed");
}
