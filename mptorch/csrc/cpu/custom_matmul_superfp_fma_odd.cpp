#include "custom_matmul_kernel.h"
#include "../common/carrier_odd.h"

// superfp formats, fused mac, carrier step rounded to odd, in binary32: the
// kernels behind custom_matmul_superfp_fma_odd (carrier_rounding=RoundMode.RO,
// common/carrier_odd.h) for float32, float16 and bfloat16 operands. The kernel template is the
// NAIVE ops' own; only the Mac it is instantiated with is new, and no
// translation unit that existed before a13760a includes carrier_odd.h, so
// their objects are unchanged. The binary64 kernels are custom_matmul_superfp_fma_odd_f64.cpp; the
// entry points are in custom_matmul_odd_entry.cpp.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::SuperfpFusedArgs;
  using mptorch::gemm::OddArgs;

  template void CpuBackend::launch_as<float, OddArgs<SuperfpFusedArgs>>(
      const GemmShape &, const OddArgs<SuperfpFusedArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
