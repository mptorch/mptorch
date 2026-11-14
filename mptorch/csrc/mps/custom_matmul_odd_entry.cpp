// The custom_matmul_*_odd ops and _carrier_odd on an Apple GPU: not yet.
// Metal has no directed rounding and flushes binary32 subnormals, so the odd
// step would be error-free transforms with integer paths for tiny
// magnitudes, which a13760a, the commit that added the odd ops, left for
// later, choosing to raise on MPS. The ops are registered here all the same,
// so that an MPS tensor gets this message rather than a dispatch error that
// names nothing, and CI's check that every schema has an MPS kernel holds.

#include "../quant_ops.h"
#include <c10/util/Exception.h>

using at::Tensor;

namespace
{
  [[noreturn]] void no_metal_kernel(const char *op_name)
  {
    TORCH_CHECK(false, op_name,
                ": the carrier step rounded to odd has no MPS kernel: Metal has no directed rounding, "
                "and the error-free-transform version is not written yet; run it on a CPU or CUDA tensor");
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

// _stochastic_cast (common/stochastic_cast_host.h) is test-only, and has no
// Metal kernel: tests/test_mps.py holds every MPS stochastic op to the CPU's
// word for word instead, and the CPU's casts to P3109 through this op.
Tensor stochastic_cast_mps(Tensor, Tensor, int64_t, c10::IntArrayRef, int64_t)
{
  TORCH_CHECK(false, "_stochastic_cast: a test-only op with no MPS kernel; run it on a CPU or CUDA tensor");
}

Tensor binaryK_matmul_mixed_odd_mps(Tensor, Tensor, Tensor, bool, bool, c10::IntArrayRef,
                                    c10::IntArrayRef, c10::IntArrayRef, bool, bool,
                                    c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, bool,
                                    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                    int64_t)
{
  no_metal_kernel("custom_matmul_binaryK_mixed_odd");
}

Tensor superfp_matmul_mixed_odd_mps(Tensor, Tensor, Tensor, bool, bool, c10::IntArrayRef,
                                    c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, bool,
                                    bool, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef,
                                    c10::IntArrayRef, bool, int64_t, int64_t, int64_t, int64_t,
                                    int64_t, int64_t)
{
  no_metal_kernel("custom_matmul_superfp_mixed_odd");
}

Tensor binaryK_matmul_fma_mixed_odd_mps(Tensor, Tensor, Tensor, bool, bool, bool, c10::IntArrayRef,
                                        c10::IntArrayRef, c10::IntArrayRef, bool, int64_t, int64_t,
                                        int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_matmul_binaryK_fma_mixed_odd");
}

Tensor superfp_matmul_fma_mixed_odd_mps(Tensor, Tensor, Tensor, bool, bool, bool, c10::IntArrayRef,
                                        c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, bool,
                                        int64_t, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_matmul_superfp_fma_mixed_odd");
}

Tensor binaryK_superfp_matmul_odd_mps(Tensor, Tensor, bool, bool, int64_t, int64_t, int64_t, bool,
                                      int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t,
                                      int64_t, int64_t, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_matmul_binaryK_superfp_odd");
}

Tensor superfp_binaryK_matmul_odd_mps(Tensor, Tensor, bool, bool, int64_t, int64_t, int64_t,
                                      int64_t, bool, int64_t, int64_t, int64_t, bool, int64_t,
                                      int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_matmul_superfp_binaryK_odd");
}

Tensor binaryK_conv_odd_mps(Tensor, Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef,
                            c10::IntArrayRef, c10::IntArrayRef, int64_t, int64_t, int64_t, int64_t,
                            bool, bool, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t,
                            int64_t, int64_t, bool, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_conv_binaryK_odd");
}

Tensor superfp_conv_odd_mps(Tensor, Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef,
                            c10::IntArrayRef, c10::IntArrayRef, int64_t, int64_t, int64_t, int64_t,
                            int64_t, bool, bool, int64_t, int64_t, int64_t, int64_t, bool, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t,
                            int64_t, int64_t, int64_t, bool, int64_t, int64_t)
{
  no_metal_kernel("custom_conv_superfp_odd");
}

Tensor binaryK_conv_fma_odd_mps(Tensor, Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef,
                                c10::IntArrayRef, c10::IntArrayRef, int64_t, bool, int64_t, int64_t,
                                int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                bool, int64_t, int64_t, int64_t, bool, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_conv_binaryK_fma_odd");
}

Tensor superfp_conv_fma_odd_mps(Tensor, Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef,
                                c10::IntArrayRef, c10::IntArrayRef, int64_t, bool, int64_t, int64_t,
                                int64_t, int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t,
                                bool, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t)
{
  no_metal_kernel("custom_conv_superfp_fma_odd");
}

Tensor binaryK_conv_mixed_odd_mps(Tensor, Tensor, Tensor, int64_t, c10::IntArrayRef,
                                  c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t,
                                  c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, bool, bool,
                                  c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, bool,
                                  int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                  int64_t)
{
  no_metal_kernel("custom_conv_binaryK_mixed_odd");
}

Tensor superfp_conv_mixed_odd_mps(Tensor, Tensor, Tensor, int64_t, c10::IntArrayRef,
                                  c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t,
                                  c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef,
                                  c10::IntArrayRef, bool, bool, c10::IntArrayRef, c10::IntArrayRef,
                                  c10::IntArrayRef, c10::IntArrayRef, bool, int64_t, int64_t,
                                  int64_t, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_conv_superfp_mixed_odd");
}

Tensor binaryK_conv_fma_mixed_odd_mps(Tensor, Tensor, Tensor, int64_t, c10::IntArrayRef,
                                      c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t,
                                      bool, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef,
                                      bool, int64_t, int64_t, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_conv_binaryK_fma_mixed_odd");
}

Tensor superfp_conv_fma_mixed_odd_mps(Tensor, Tensor, Tensor, int64_t, c10::IntArrayRef,
                                      c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t,
                                      bool, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef,
                                      c10::IntArrayRef, bool, int64_t, int64_t, int64_t, int64_t)
{
  no_metal_kernel("custom_conv_superfp_fma_mixed_odd");
}

Tensor binaryK_superfp_conv_odd_mps(Tensor, Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef,
                                    c10::IntArrayRef, c10::IntArrayRef, int64_t, int64_t, int64_t,
                                    int64_t, bool, int64_t, int64_t, int64_t, int64_t, bool,
                                    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                    int64_t, bool, int64_t, int64_t, int64_t, int64_t, bool,
                                    int64_t, int64_t)
{
  no_metal_kernel("custom_conv_binaryK_superfp_odd");
}

Tensor superfp_binaryK_conv_odd_mps(Tensor, Tensor, int64_t, c10::IntArrayRef, c10::IntArrayRef,
                                    c10::IntArrayRef, c10::IntArrayRef, int64_t, int64_t, int64_t,
                                    int64_t, int64_t, bool, int64_t, int64_t, int64_t, bool,
                                    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                    int64_t, bool, int64_t, int64_t, int64_t, bool, int64_t,
                                    int64_t, int64_t)
{
  no_metal_kernel("custom_conv_superfp_binaryK_odd");
}
