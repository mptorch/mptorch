// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for custom_conv_superfp.
// They accumulate under AccumulateAlgorithm::TREE (pairwise sums within each
// block of products) in the binary32 carrier (float32, float16 and bfloat16
// operands). The entry points are in ../custom_conv_entry.cpp, and
// ../custom_matmul_kernel.h explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::SuperfpSplitArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::TREE, AccumulateArgs<SuperfpSplitArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpSplitArgs>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
