// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_binaryK_superfp and custom_conv_superfp_binaryK. They accumulate
// under AccumulateAlgorithm::NAIVE (one running sum) in the binary64 carrier
// (float64 operands). The entry points are in ../custom_matmul_cross_entry.cpp,
// and ../custom_matmul_kernel.h explains how the instantiations are divided
// among the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, BinaryKSuperfpArgs>(
    const GemmShape &, const ConvArgs<BinaryKSuperfpArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, SuperfpBinaryKArgs>(
    const GemmShape &, const ConvArgs<SuperfpBinaryKArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
