// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_binaryK_superfp_odd and custom_conv_superfp_binaryK_odd. They
// accumulate under AccumulateAlgorithm::NAIVE (one running sum) in the binary64
// carrier (float64 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_conv_odd_entry.cpp, and ../custom_matmul_kernel.h explains how the
// instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, OddArgs<BinaryKSuperfpArgs>>(
      const GemmShape &, const ConvArgs<OddArgs<BinaryKSuperfpArgs>> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, OddArgs<SuperfpBinaryKArgs>>(
      const GemmShape &, const ConvArgs<OddArgs<SuperfpBinaryKArgs>> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
