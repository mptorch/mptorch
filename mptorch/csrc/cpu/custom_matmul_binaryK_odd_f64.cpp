#include "custom_matmul_kernel.h"
#include "../common/carrier_odd.h"

// binaryK formats, split mac, carrier step rounded to odd, in binary64: the
// kernels behind custom_matmul_binaryK_odd (phase N of dev/continuation_plan.md,
// common/carrier_odd.h) for float64 operands. The kernel template is the
// NAIVE ops' own; only the Mac it is instantiated with is new, and no
// translation unit that existed before phase N includes carrier_odd.h, so
// their objects are unchanged. The binary32 kernels are custom_matmul_binaryK_odd.cpp; the
// entry points are in custom_matmul_odd_entry.cpp.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::OddArgs;

  template void CpuBackend::launch_as<double, OddArgs<BinaryKSplitArgs>>(
      const GemmShape &, const OddArgs<BinaryKSplitArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
