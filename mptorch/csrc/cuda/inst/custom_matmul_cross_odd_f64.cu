// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_binaryK_superfp_odd and custom_matmul_superfp_binaryK_odd. They
// accumulate under AccumulateAlgorithm::NAIVE (one running sum) in the binary64
// carrier (float64 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.cuh explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"
#include "../../common/carrier_odd_mixed.h"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSuperfpArgs;
    using mptorch::gemm::OddArgs;
    using mptorch::gemm::SuperfpBinaryKArgs;

    template void CudaBackend::launch_as<double, OddArgs<BinaryKSuperfpArgs>>(
        const GemmShape &, const OddArgs<BinaryKSuperfpArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<double, OddArgs<SuperfpBinaryKArgs>>(
        const GemmShape &, const OddArgs<SuperfpBinaryKArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
