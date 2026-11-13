#include "custom_matmul_kernel.h"

// The conv ops over the cross-family split macs (common/gemm_cross.h)
// under AccumulateAlgorithm::TREE (pairwise summation in blocks;
// common/gemm_accumulate.h), in binary32: the gathered kernels of
// custom_conv_binaryK_superfp and custom_conv_superfp_binaryK. One of four
// cross-family conv instantiation files (custom_conv_cross.cpp says why).

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::TREE, AccumulateArgs<BinaryKSuperfpArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSuperfpArgs>> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::TREE, AccumulateArgs<SuperfpBinaryKArgs>>(
    const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpBinaryKArgs>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
