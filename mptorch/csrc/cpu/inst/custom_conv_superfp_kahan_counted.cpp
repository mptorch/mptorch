// Explicit instantiations of the CPU GEMM kernel for the counted conv ops
// custom_conv_superfp_counted and custom_conv_superfp_fma_counted
// (common/gemm_counted.h) under AccumulateAlgorithm::KAHAN, in
// the binary32 carrier (float32, float16 and bfloat16 operands). Built with MPTORCH_COUNTING=1
// only. The entry points are in ../custom_conv_counted_entry.cpp, and
// ../custom_matmul_kernel.h explains how the instantiations are divided among
// the files in this directory.
#include "../../common/gemm_counted.h"
#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::SuperfpFusedArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::CountingArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE,
                                       CountingArgs<SuperfpSplitArgs, AccumulateAlgorithm::KAHAN>>(
    const GemmShape &, const ConvArgs<CountingArgs<SuperfpSplitArgs, AccumulateAlgorithm::KAHAN>> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE,
                                       CountingArgs<SuperfpFusedArgs, AccumulateAlgorithm::KAHAN>>(
    const GemmShape &, const ConvArgs<CountingArgs<SuperfpFusedArgs, AccumulateAlgorithm::KAHAN>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
