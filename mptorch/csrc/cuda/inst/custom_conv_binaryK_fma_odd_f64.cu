// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_binaryK_fma_odd and custom_conv_binaryK_fma_mixed_odd. They
// accumulate under AccumulateAlgorithm::NAIVE (one running sum) in the binary64
// carrier (float64 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_conv_odd_entry.cpp, and ../custom_matmul_kernel.cuh explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"
#include "../../common/carrier_odd_mixed.h"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::BinaryKFusedMixedArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::OddArgs;
    using mptorch::gemm::OddPaletteArgs;

    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, OddArgs<BinaryKFusedArgs>>(
        const GemmShape &, const ConvArgs<OddArgs<BinaryKFusedArgs>> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, OddPaletteArgs<BinaryKFusedMixedArgs>>(
        const GemmShape &, const ConvArgs<OddPaletteArgs<BinaryKFusedMixedArgs>> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
