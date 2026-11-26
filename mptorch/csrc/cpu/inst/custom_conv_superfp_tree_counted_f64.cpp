// Explicit instantiations of the CPU GEMM kernel for the counted conv ops
// custom_conv_superfp_counted and custom_conv_superfp_fma_counted
// (common/gemm_counted.h) under AccumulateAlgorithm::TREE, in
// the binary64 carrier (float64 operands). Built with MPTORCH_COUNTING=1
// only. The entry points are in ../custom_conv_counted_entry.cpp, and
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
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::CountingArgs;

  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE,
                                       CountingArgs<SuperfpSplitArgs, AccumulateAlgorithm::TREE>>(
    const GemmShape &, const ConvArgs<CountingArgs<SuperfpSplitArgs, AccumulateAlgorithm::TREE>> &,
    const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE,
                                       CountingArgs<SuperfpFusedArgs, AccumulateAlgorithm::TREE>>(
    const GemmShape &, const ConvArgs<CountingArgs<SuperfpFusedArgs, AccumulateAlgorithm::TREE>> &,
    const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
