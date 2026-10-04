// The MPS entry points of the five block-format ops (block_pack,
// block_unpack, block_quant, block_quant_ and custom_matmul_block): functions
// that raise. The Metal kernels are dev/continuation_plan.md's phase H; the
// decode they will run is common/block_decode.h, written MSL-legal for it.
// They are registered anyway (mps_ops.cpp) because an op with no MPS kernel
// fails at dispatch with a message that names nothing a caller can act on,
// where this one names the way out. No argument is read, so none is named.

#include "../quant_ops.h"
#include <c10/util/Exception.h>

namespace
{
  [[noreturn]] void no_metal_kernel(const char *op_name)
  {
    TORCH_CHECK(false, op_name, ": block formats have no MPS kernel yet "
                                "(dev/continuation_plan.md, phase H); run it on a CPU or CUDA tensor");
    __builtin_unreachable(); // TORCH_CHECK(false, ...) throws
  }
} // namespace

std::tuple<at::Tensor, at::Tensor> block_pack_mps(at::Tensor, c10::IntArrayRef, double, double, double,
                                                  int64_t)
{
  no_metal_kernel("block_pack");
}

at::Tensor block_unpack_mps(at::Tensor, at::Tensor, int64_t, c10::IntArrayRef, double, double, double,
                            at::ScalarType)
{
  no_metal_kernel("block_unpack");
}

at::Tensor block_quantize_mps(at::Tensor, c10::IntArrayRef, double, double, double, int64_t)
{
  no_metal_kernel("block_quant");
}

at::Tensor &block_quantize_mps_(at::Tensor &, c10::IntArrayRef, double, double, double, int64_t)
{
  no_metal_kernel("block_quant_");
}

at::Tensor block_matmul_mps(at::Tensor, at::Tensor, int64_t, bool, at::Tensor, at::Tensor, int64_t,
                            bool, c10::IntArrayRef, double, double, double, c10::IntArrayRef, double,
                            double, double, bool, bool, int64_t, int64_t, int64_t, bool, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t,
                            int64_t, bool, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_matmul_block");
}
