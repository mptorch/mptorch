#include "custom_matmul_kernel.h"

// The binary64 twin of custom_conv_binaryK_fma.cpp: its kernels in the
// binary64 carrier, for float64 operands and for calls that name
// carrier=torch.float64. A file of its own, like the NAIVE *_f64.cpp files,
// so that MPTORCH_NO_FP64=1 leaves these kernels out by leaving the file out,
// and no other translation unit grows.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BinaryKFusedMixedArgs;

  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, BinaryKFusedArgs>(
    const GemmShape &, const ConvArgs<BinaryKFusedArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, BinaryKFusedMixedArgs>(
    const GemmShape &, const ConvArgs<BinaryKFusedMixedArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
