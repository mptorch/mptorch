// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_binaryK_superfp_odd and custom_conv_superfp_binaryK_odd. They
// accumulate under AccumulateAlgorithm::NAIVE (one running sum) in the binary32
// carrier (float32, float16 and bfloat16 operands), and round each carrier step
// to odd (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_conv_odd_entry.cpp, and ../custom_matmul_kernel.cuh explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"
#include "../../common/carrier_odd_mixed.h"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSuperfpArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::OddArgs;
    using mptorch::gemm::SuperfpBinaryKArgs;

    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, OddArgs<BinaryKSuperfpArgs>>(
        const GemmShape &, const ConvArgs<OddArgs<BinaryKSuperfpArgs>> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, OddArgs<SuperfpBinaryKArgs>>(
        const GemmShape &, const ConvArgs<OddArgs<SuperfpBinaryKArgs>> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
