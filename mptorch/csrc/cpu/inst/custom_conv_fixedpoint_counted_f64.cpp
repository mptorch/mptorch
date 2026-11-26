// Explicit instantiations of the CPU GEMM kernel for the counted op
// custom_conv_fixedpoint_counted (common/fixedpoint_counted.h), every
// Args its twin's plan can run, in the binary64 carrier (float64 operands).
// Built with MPTORCH_COUNTING=1 only. The entry points are in
// ../custom_conv_counted_entry.cpp, and ../custom_matmul_kernel.h explains how the
// instantiations are divided among the files in this directory.

#include "../../common/fixedpoint_counted.h"
#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::FixedPointIntSplitArgs;
  using mptorch::gemm::FixedPointSplitArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::FixedPointCountingArgs;

  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, FixedPointCountingArgs<FixedPointSplitArgs>>(
    const GemmShape &, const ConvArgs<FixedPointCountingArgs<FixedPointSplitArgs>> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, FixedPointCountingArgs<FixedPointIntSplitArgs<false>>>(
    const GemmShape &, const ConvArgs<FixedPointCountingArgs<FixedPointIntSplitArgs<false>>> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, FixedPointCountingArgs<FixedPointIntSplitArgs<true>>>(
    const GemmShape &, const ConvArgs<FixedPointCountingArgs<FixedPointIntSplitArgs<true>>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
