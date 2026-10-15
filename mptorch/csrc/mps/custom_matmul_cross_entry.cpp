// The MPS entry points of the six cross-family ops (common/gemm_cross.h; a
// split mac whose products and sums are rounded to formats of two different
// families): functions that raise. The Metal kernels are
// dev/continuation_plan.md's phase H, which writes their Args into the Metal
// prelude and gemm_backend.cpp's args_source. They are registered anyway
// (mps_ops.cpp) because an op with no MPS kernel fails at dispatch with a
// message that names nothing a caller can act on, where this one names the
// way out. No argument is read, so none is named.
//
// A file of its own, like custom_matmul_accumulated_entry.cpp, so that
// custom_matmul_entry.cpp is the text it was.

#include "../quant_ops.h"
#include <c10/util/Exception.h>

namespace
{
  at::Tensor no_metal_kernel(const char *op_name)
  {
    TORCH_CHECK(false, op_name, ": the cross-family ops have no MPS kernel yet "
                                "(dev/continuation_plan.md, phase H); run them on a CPU or CUDA "
                                "tensor");
    return at::Tensor();
  }
} // namespace

at::Tensor binaryK_superfp_matmul_mps(at::Tensor, at::Tensor, bool, bool, int64_t, int64_t,
                                      int64_t, bool, int64_t, int64_t, int64_t, int64_t, bool,
                                      int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                      int64_t)
{
  return no_metal_kernel("custom_matmul_binaryK_superfp");
}

at::Tensor superfp_binaryK_matmul_mps(at::Tensor, at::Tensor, bool, bool, int64_t, int64_t,
                                      int64_t, int64_t, bool, int64_t, int64_t, int64_t, bool,
                                      int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                      int64_t)
{
  return no_metal_kernel("custom_matmul_superfp_binaryK");
}

at::Tensor binaryK_superfp_matmul_accumulated_mps(
    at::Tensor, at::Tensor, bool, bool, int64_t, int64_t, int64_t, bool, int64_t, int64_t,
    int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, bool, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t)
{
  return no_metal_kernel("custom_matmul_binaryK_superfp_accumulated");
}

at::Tensor superfp_binaryK_matmul_accumulated_mps(
    at::Tensor, at::Tensor, bool, bool, int64_t, int64_t, int64_t, int64_t, bool, int64_t,
    int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, bool, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t)
{
  return no_metal_kernel("custom_matmul_superfp_binaryK_accumulated");
}

at::Tensor binaryK_superfp_conv_mps(at::Tensor, at::Tensor, int64_t, c10::IntArrayRef,
                                    c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t,
                                    int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t,
                                    int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t,
                                    int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t,
                                    int64_t, bool, int64_t, int64_t)
{
  return no_metal_kernel("custom_conv_binaryK_superfp");
}

at::Tensor superfp_binaryK_conv_mps(at::Tensor, at::Tensor, int64_t, c10::IntArrayRef,
                                    c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t,
                                    int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t,
                                    int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t,
                                    int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t,
                                    bool, int64_t, int64_t, int64_t)
{
  return no_metal_kernel("custom_conv_superfp_binaryK");
}
