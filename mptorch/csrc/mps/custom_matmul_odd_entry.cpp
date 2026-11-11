// The custom_matmul_*_odd ops and _carrier_odd on an Apple GPU (phase N of
// dev/continuation_plan.md): not yet. Metal has no directed rounding and
// flushes binary32 subnormals, so the odd step would be error-free
// transforms with integer paths for tiny magnitudes, which phase N left for
// later (decision N-c). The ops are registered here all the same, so that an
// MPS tensor gets this message rather than a dispatch error that names
// nothing, and CI's check that every schema has an MPS kernel holds.

#include "../quant_ops.h"
#include <c10/util/Exception.h>

using at::Tensor;

namespace
{
  [[noreturn]] void no_metal_kernel(const char *op_name)
  {
    TORCH_CHECK(false, op_name,
                ": the carrier step rounded to odd has no MPS kernel (dev/continuation_plan.md, "
                "phase N); run it on a CPU or CUDA tensor");
  }
} // namespace

Tensor binaryK_matmul_odd_mps(Tensor, Tensor, bool, bool, int64_t, int64_t, int64_t, bool, bool, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_matmul_binaryK_odd");
}

Tensor superfp_matmul_odd_mps(Tensor, Tensor, bool, bool, int64_t, int64_t, int64_t, int64_t, bool, bool, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_matmul_superfp_odd");
}

Tensor binaryK_matmul_fma_odd_mps(Tensor, Tensor, bool, bool, bool, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_matmul_binaryK_fma_odd");
}

Tensor superfp_matmul_fma_odd_mps(Tensor, Tensor, bool, bool, bool, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_matmul_superfp_fma_odd");
}

Tensor carrier_odd_mps(Tensor, Tensor, Tensor, int64_t)
{
  no_metal_kernel("_carrier_odd");
}
