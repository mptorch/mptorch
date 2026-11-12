#include "custom_matmul_kernel.cuh"

// The cross-family split macs (common/gemm_cross.h) under
// AccumulateAlgorithm::BLOCK (blocked summation with an outer
// rounding; common/gemm_accumulate.h), in binary32: the kernels of
// custom_matmul_binaryK_superfp_accumulated and
// custom_matmul_superfp_binaryK_accumulated, seven each. One of four
// cross-family GEMM instantiation files (custom_matmul_cross.cu says why).

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::BinaryKSuperfpArgs;
    using mptorch::gemm::SuperfpBinaryKArgs;

    template void CudaBackend::launch_accumulated_as<float, AccumulateAlgorithm::BLOCK, BinaryKSuperfpArgs>(
      const GemmShape &, const AccumulateArgs<BinaryKSuperfpArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_accumulated_as<float, AccumulateAlgorithm::BLOCK, SuperfpBinaryKArgs>(
      const GemmShape &, const AccumulateArgs<SuperfpBinaryKArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
