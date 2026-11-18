// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for custom_conv_binaryK
// and custom_conv_binaryK_fma. They accumulate under AccumulateAlgorithm::BLOCK
// (a sum per block of products, added to the running total) in the binary64
// carrier (float64 operands). The entry points are in ../custom_conv_entry.cpp,
// and ../custom_matmul_kernel.h explains how the instantiations are divided
// among the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::BinaryKFusedArgs;

  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::BLOCK, AccumulateArgs<BinaryKSplitArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSplitArgs>> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::BLOCK, AccumulateArgs<BinaryKFusedArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKFusedArgs>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
