// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for custom_conv_superfp
// and custom_conv_superfp_fma. They accumulate under AccumulateAlgorithm::BLOCK
// (a sum per block of products, added to the running total) in the binary64
// carrier (float64 operands). The entry points are in ../custom_conv_entry.cpp,
// and ../custom_matmul_kernel.h explains how the instantiations are divided
// among the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::SuperfpFusedArgs;

  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::BLOCK, AccumulateArgs<SuperfpSplitArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpSplitArgs>> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::BLOCK, AccumulateArgs<SuperfpFusedArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpFusedArgs>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
