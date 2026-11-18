// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_binaryK_odd and custom_conv_binaryK_mixed_odd. They accumulate
// under AccumulateAlgorithm::NAIVE (one running sum) in the binary32 carrier
// (float32, float16 and bfloat16 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_conv_odd_entry.cpp, and ../custom_matmul_kernel.h explains how the
// instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::BinaryKSplitMixedArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::OddPaletteArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, OddArgs<BinaryKSplitArgs>>(
      const GemmShape &, const ConvArgs<OddArgs<BinaryKSplitArgs>> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, OddPaletteArgs<BinaryKSplitMixedArgs>>(
      const GemmShape &, const ConvArgs<OddPaletteArgs<BinaryKSplitMixedArgs>> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
