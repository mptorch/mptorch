// Explicit instantiations of the CPU GEMM kernel for the counted conv ops
// custom_conv_binaryK_counted and custom_conv_binaryK_fma_counted
// (common/gemm_counted.h) under AccumulateAlgorithm::KAHAN, in
// the binary32 carrier (float32, float16 and bfloat16 operands). Built with MPTORCH_COUNTING=1
// only. The entry points are in ../custom_conv_counted_entry.cpp, and
// ../custom_matmul_kernel.h explains how the instantiations are divided among
// the files in this directory.
#include "../../common/gemm_counted.h"
#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::CountingArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE,
                                       CountingArgs<BinaryKSplitArgs, AccumulateAlgorithm::KAHAN>>(
    const GemmShape &, const ConvArgs<CountingArgs<BinaryKSplitArgs, AccumulateAlgorithm::KAHAN>> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE,
                                       CountingArgs<BinaryKFusedArgs, AccumulateAlgorithm::KAHAN>>(
    const GemmShape &, const ConvArgs<CountingArgs<BinaryKFusedArgs, AccumulateAlgorithm::KAHAN>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
