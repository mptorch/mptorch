// Explicit instantiations of the CPU GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_fixedpoint in the binary32 carrier (float32, float16 and
// bfloat16 operands): the carrier path (SplitMac over the fixed-point float
// policies) and the integer accumulator's split mac, saturating and wrapping
// around (common/fixedpoint_gemm.h). NAIVE only. The entry points are in
// ../custom_conv_fixedpoint_entry.cpp, and ../custom_matmul_kernel.h
// explains how the instantiations are divided among the files in this
// directory.

#include "../../common/fixedpoint_gemm.h"
#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::FixedPointIntSplitArgs;
  using mptorch::gemm::FixedPointSplitArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, FixedPointSplitArgs>(
      const GemmShape &, const ConvArgs<FixedPointSplitArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE,
                                           FixedPointIntSplitArgs<false>>(
      const GemmShape &, const ConvArgs<FixedPointIntSplitArgs<false>> &,
      const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE,
                                           FixedPointIntSplitArgs<true>>(
      const GemmShape &, const ConvArgs<FixedPointIntSplitArgs<true>> &,
      const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
