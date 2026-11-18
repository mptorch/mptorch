// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for custom_conv_superfp
// and custom_conv_superfp_mixed. They accumulate under
// AccumulateAlgorithm::NAIVE (one running sum) in the binary32 carrier
// (float32, float16 and bfloat16 operands). The entry points are in
// ../custom_conv_entry.cpp, and ../custom_matmul_kernel.h explains how the
// instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::SuperfpSplitMixedArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, SuperfpSplitArgs>(
    const GemmShape &, const ConvArgs<SuperfpSplitArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, SuperfpSplitMixedArgs>(
    const GemmShape &, const ConvArgs<SuperfpSplitMixedArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
