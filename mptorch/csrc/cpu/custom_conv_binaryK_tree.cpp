#include "custom_matmul_kernel.h"

// The conv ops over binaryK formats under AccumulateAlgorithm::TREE
// (pairwise summation over blocks; common/gemm_accumulate.h), in binary32: the
// gathered kernels (common/gemm_gather.h) of custom_conv_binaryK (a fused mac has no tree).
// One of ten conv instantiation files (custom_conv_binaryK.cpp says why).

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::BinaryKSplitArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::TREE, AccumulateArgs<BinaryKSplitArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSplitArgs>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
