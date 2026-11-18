// Explicit instantiations of the CPU GEMM kernel for
// custom_matmul_superfp_accumulated and custom_matmul_superfp_fma_accumulated.
// They accumulate under AccumulateAlgorithm::KAHAN (compensated summation) in
// the binary32 carrier (float32, float16 and bfloat16 operands). The entry
// points are in ../custom_matmul_accumulated_entry.cpp, and
// ../custom_matmul_kernel.h explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::SuperfpFusedArgs;
  using mptorch::gemm::AccumulateArgs;

  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::KAHAN, SuperfpSplitArgs>(
    const GemmShape &, const AccumulateArgs<SuperfpSplitArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::KAHAN, SuperfpFusedArgs>(
    const GemmShape &, const AccumulateArgs<SuperfpFusedArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
