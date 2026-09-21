#include "custom_matmul_kernel.h"

// superfp formats under AccumulateAlgorithm::TREE (pairwise summation within a
// block; common/gemm_accumulate.h), in binary32: the float32, float16 and
// bfloat16 kernels of custom_matmul_superfp. There is no fused twin: a tree
// sums products pairwise, and a fused multiply-add has no product term.
//
// One of six such files, one per (format family x algorithm). It is
// instantiation-only, like the eight NAIVE files, behind
// custom_matmul_entry.cpp. They are separate from the NAIVE files so that
// those compile exactly what they compiled before the three algorithms
// existed, and from each other because the build is throughput-bound: this one
// holds fourteen kernels (seven round modes, with and without an accumulate
// format). There is no *_f64 twin yet: the three algorithms are binary32 only
// until dev/continuation_plan.md's phase G.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::AccumulateArgs;

  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::TREE, SuperfpSplitArgs>(
    const GemmShape &, const AccumulateArgs<SuperfpSplitArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
