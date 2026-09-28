#include "custom_matmul_kernel.h"

// The conv ops over binaryK formats with a split mac under AccumulateAlgorithm::NAIVE,
// in binary32: the gathered kernels (common/gemm_gather.h) of custom_conv_binaryK
// and custom_conv_binaryK_mixed, the conv twins of custom_matmul_binaryK and its
// palette op.
//
// One of ten such files, one per (format family x mac mode) for NAIVE and one
// per (format family x algorithm) for KAHAN, BLOCK and TREE, mirroring the
// GEMM's: a gathered kernel is its GEMM kernel's instantiation over a
// Gathered policy, so it costs what its GEMM kernel costs to compile, and
// keeping these apart from the GEMM files is what leaves those compiling
// exactly what they did. binary32 only until dev/continuation_plan.md's
// phase G.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::BinaryKSplitMixedArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, BinaryKSplitArgs>(
    const GemmShape &, const ConvArgs<BinaryKSplitArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, BinaryKSplitMixedArgs>(
    const GemmShape &, const ConvArgs<BinaryKSplitMixedArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
