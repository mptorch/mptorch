// Explicit instantiations of the CPU GEMM kernel for the counted ops
// custom_matmul_superfp_counted and custom_matmul_superfp_fma_counted
// (common/gemm_counted.h) under AccumulateAlgorithm::TREE, in
// the binary32 carrier (float32, float16 and bfloat16 operands). Built with MPTORCH_COUNTING=1
// only. The entry points are in ../custom_matmul_counted_entry.cpp, and
// ../custom_matmul_kernel.h explains how the instantiations are divided among
// the files in this directory.
// Under TREE the fused op has no kernel (TREE sums products, which a fused
// mac has none of): its instantiation only refuses the call's algorithm.

#include "../../common/gemm_counted.h"
#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::SuperfpFusedArgs;
  using mptorch::gemm::CountingArgs;

  template void CpuBackend::launch_as<float, CountingArgs<SuperfpSplitArgs, AccumulateAlgorithm::TREE>>(
    const GemmShape &, const CountingArgs<SuperfpSplitArgs, AccumulateAlgorithm::TREE> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, CountingArgs<SuperfpFusedArgs, AccumulateAlgorithm::TREE>>(
    const GemmShape &, const CountingArgs<SuperfpFusedArgs, AccumulateAlgorithm::TREE> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
