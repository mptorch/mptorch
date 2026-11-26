// Explicit instantiations of the CUDA GEMM kernel for the counted ops
// custom_matmul_superfp_counted and custom_matmul_superfp_fma_counted
// (common/gemm_counted.h) under AccumulateAlgorithm::BLOCK, in
// the binary64 carrier (float64 operands). Built with MPTORCH_COUNTING=1
// only. The entry points are in ../custom_matmul_counted_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided among
// the files in this directory.
#include "../../common/gemm_counted.h"
#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::SuperfpSplitArgs;
    using mptorch::gemm::SuperfpFusedArgs;
    using mptorch::gemm::CountingArgs;

    template void CudaBackend::launch_as<double, CountingArgs<SuperfpSplitArgs, AccumulateAlgorithm::BLOCK>>(
        const GemmShape &, const CountingArgs<SuperfpSplitArgs, AccumulateAlgorithm::BLOCK> &,
        const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<double, CountingArgs<SuperfpFusedArgs, AccumulateAlgorithm::BLOCK>>(
        const GemmShape &, const CountingArgs<SuperfpFusedArgs, AccumulateAlgorithm::BLOCK> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
