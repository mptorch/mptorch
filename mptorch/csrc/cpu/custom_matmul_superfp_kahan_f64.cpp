#include "custom_matmul_kernel.h"

// The binary64 twin of custom_matmul_superfp_kahan.cpp: its kernels in the
// binary64 carrier, for float64 operands and for calls that name
// carrier=torch.float64. A file of its own, like the NAIVE *_f64.cpp files,
// so that MPTORCH_NO_FP64=1 leaves these kernels out by leaving the file out,
// and no other translation unit grows.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::SuperfpFusedArgs;
  using mptorch::gemm::AccumulateArgs;

  template void CpuBackend::launch_accumulated_as<double, AccumulateAlgorithm::KAHAN, SuperfpSplitArgs>(
    const GemmShape &, const AccumulateArgs<SuperfpSplitArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_accumulated_as<double, AccumulateAlgorithm::KAHAN, SuperfpFusedArgs>(
    const GemmShape &, const AccumulateArgs<SuperfpFusedArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
