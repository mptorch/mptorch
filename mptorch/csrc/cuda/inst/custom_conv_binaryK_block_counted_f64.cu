// Explicit instantiations of the CUDA GEMM kernel for the counted conv ops
// custom_conv_binaryK_counted and custom_conv_binaryK_fma_counted
// (common/gemm_counted.h) under AccumulateAlgorithm::BLOCK, in
// the binary64 carrier (float64 operands). Built with MPTORCH_COUNTING=1
// only. The entry points are in ../custom_conv_counted_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided among
// the files in this directory.
#include "../../common/gemm_counted.h"
#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSplitArgs;
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::CountingArgs;

    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE,
                                         CountingArgs<BinaryKSplitArgs, AccumulateAlgorithm::BLOCK>>(
      const GemmShape &, const ConvArgs<CountingArgs<BinaryKSplitArgs, AccumulateAlgorithm::BLOCK>> &,
      const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE,
                                         CountingArgs<BinaryKFusedArgs, AccumulateAlgorithm::BLOCK>>(
      const GemmShape &, const ConvArgs<CountingArgs<BinaryKFusedArgs, AccumulateAlgorithm::BLOCK>> &,
      const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
