// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_superfp_odd and custom_conv_superfp_mixed_odd. They accumulate
// under AccumulateAlgorithm::NAIVE (one running sum) in the binary64 carrier
// (float64 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_conv_odd_entry.cpp, and ../custom_matmul_kernel.h explains how the
// instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::OddPaletteArgs;
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::SuperfpSplitMixedArgs;

  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, OddArgs<SuperfpSplitArgs>>(
      const GemmShape &, const ConvArgs<OddArgs<SuperfpSplitArgs>> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, OddPaletteArgs<SuperfpSplitMixedArgs>>(
      const GemmShape &, const ConvArgs<OddPaletteArgs<SuperfpSplitMixedArgs>> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
