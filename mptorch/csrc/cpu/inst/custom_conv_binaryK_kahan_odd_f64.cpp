// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_binaryK_odd and custom_conv_binaryK_fma_odd. They accumulate
// under AccumulateAlgorithm::KAHAN (compensated summation) in the binary64
// carrier (float64 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_conv_odd_entry.cpp, and ../custom_matmul_kernel.h explains how the
// instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::KAHAN, AccumulateArgs<OddArgs<BinaryKSplitArgs>>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<OddArgs<BinaryKSplitArgs>>> &,
      const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::KAHAN, AccumulateArgs<OddArgs<BinaryKFusedArgs>>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<OddArgs<BinaryKFusedArgs>>> &,
      const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
