#include "custom_matmul_kernel.h"

// The conv ops over binaryK formats under AccumulateAlgorithm::KAHAN
// (Kahan-compensated summation; common/gemm_accumulate.h), in binary32: the
// gathered kernels (common/gemm_gather.h) of custom_conv_binaryK and custom_conv_binaryK_fma.
// One of ten conv instantiation files (custom_conv_binaryK.cpp says why).

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::BinaryKFusedArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::KAHAN, AccumulateArgs<BinaryKSplitArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSplitArgs>> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::KAHAN, AccumulateArgs<BinaryKFusedArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKFusedArgs>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
