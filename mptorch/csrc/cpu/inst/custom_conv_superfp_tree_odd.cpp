// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_superfp_odd. They accumulate under AccumulateAlgorithm::TREE
// (pairwise sums within each block of products) in the binary32 carrier
// (float32, float16 and bfloat16 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_conv_odd_entry.cpp, and ../custom_matmul_kernel.h explains how the
// instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpSplitArgs;
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::TREE, AccumulateArgs<OddArgs<SuperfpSplitArgs>>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<OddArgs<SuperfpSplitArgs>>> &,
      const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
