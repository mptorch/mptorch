// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_binaryK_fma and custom_conv_binaryK_fma_mixed. They accumulate
// under AccumulateAlgorithm::NAIVE (one running sum) in the binary64 carrier
// (float64 operands). The entry points are in ../custom_conv_entry.cpp, and
// ../custom_matmul_kernel.h explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.h"

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
