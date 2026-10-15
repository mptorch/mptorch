#include "custom_matmul_kernel.h"

// The binary64 twin of custom_matmul_cross_kahan.cpp: its kernels in the
// binary64 carrier, for float64 operands and for calls that name
// carrier=torch.float64. A file of its own, like the NAIVE *_f64.cpp files,
// so that MPTORCH_NO_FP64=1 leaves these kernels out by leaving the file out,
// and no other translation unit grows.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_accumulated_as<double, AccumulateAlgorithm::KAHAN, BinaryKSuperfpArgs>(
    const GemmShape &, const AccumulateArgs<BinaryKSuperfpArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_accumulated_as<double, AccumulateAlgorithm::KAHAN, SuperfpBinaryKArgs>(
    const GemmShape &, const AccumulateArgs<SuperfpBinaryKArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
