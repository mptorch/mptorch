#include "custom_matmul_kernel.h"

// The conv ops over superfp formats under AccumulateAlgorithm::BLOCK
// (blocked summation with an outer format; common/gemm_accumulate.h), in binary32: the
// gathered kernels (common/gemm_gather.h) of custom_conv_superfp and custom_conv_superfp_fma.
// One of ten conv instantiation files (custom_conv_superfp.cpp says why).

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::SuperfpFusedArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::BLOCK, AccumulateArgs<SuperfpSplitArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpSplitArgs>> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::BLOCK, AccumulateArgs<SuperfpFusedArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpFusedArgs>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
