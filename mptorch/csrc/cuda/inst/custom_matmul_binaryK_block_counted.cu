// Explicit instantiations of the CUDA GEMM kernel for the counted ops
// custom_matmul_binaryK_counted and custom_matmul_binaryK_fma_counted
// (common/gemm_counted.h) under AccumulateAlgorithm::BLOCK, in
// the binary32 carrier (float32, float16 and bfloat16 operands). Built with MPTORCH_COUNTING=1
// only. The entry points are in ../custom_matmul_counted_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided among
// the files in this directory.
#include "../../common/gemm_counted.h"
#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSplitArgs;
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::CountingArgs;

    template void CudaBackend::launch_as<float, CountingArgs<BinaryKSplitArgs, AccumulateAlgorithm::BLOCK>>(
        const GemmShape &, const CountingArgs<BinaryKSplitArgs, AccumulateAlgorithm::BLOCK> &,
        const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<float, CountingArgs<BinaryKFusedArgs, AccumulateAlgorithm::BLOCK>>(
        const GemmShape &, const CountingArgs<BinaryKFusedArgs, AccumulateAlgorithm::BLOCK> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
