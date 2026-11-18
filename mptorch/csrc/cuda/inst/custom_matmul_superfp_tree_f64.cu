// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_superfp_accumulated. They accumulate under
// AccumulateAlgorithm::TREE (pairwise sums within each block of products) in
// the binary64 carrier (float64 operands). The entry points are in
// ../custom_matmul_accumulated_entry.cpp, and ../custom_matmul_kernel.cuh
// explains how the instantiations are divided among the files in this
// directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::SuperfpSplitArgs;
    using mptorch::gemm::AccumulateArgs;

    template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::TREE, SuperfpSplitArgs>(
        const GemmShape &, const AccumulateArgs<SuperfpSplitArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
