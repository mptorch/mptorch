// Explicit instantiations of the CUDA GEMM kernel for the counted ops
// custom_matmul_superfp_counted and custom_matmul_superfp_fma_counted
// (common/gemm_counted.h) under AccumulateAlgorithm::TREE, in
// the binary32 carrier (float32, float16 and bfloat16 operands). Built with MPTORCH_COUNTING=1
// only. The entry points are in ../custom_matmul_counted_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided among
// the files in this directory.
// Under TREE the fused op has no kernel (TREE sums products, which a fused
// mac has none of): its instantiation only refuses the call's algorithm.

#include "../../common/gemm_counted.h"
#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::SuperfpSplitArgs;
    using mptorch::gemm::SuperfpFusedArgs;
    using mptorch::gemm::CountingArgs;

    template void CudaBackend::launch_as<float, CountingArgs<SuperfpSplitArgs, AccumulateAlgorithm::TREE>>(
        const GemmShape &, const CountingArgs<SuperfpSplitArgs, AccumulateAlgorithm::TREE> &,
        const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<float, CountingArgs<SuperfpFusedArgs, AccumulateAlgorithm::TREE>>(
        const GemmShape &, const CountingArgs<SuperfpFusedArgs, AccumulateAlgorithm::TREE> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
