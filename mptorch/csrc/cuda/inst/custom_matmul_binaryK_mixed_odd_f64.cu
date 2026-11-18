// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_binaryK_mixed_odd. They accumulate under
// AccumulateAlgorithm::NAIVE (one running sum) in the binary64 carrier (float64
// operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.cuh explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"
#include "../../common/carrier_odd_mixed.h"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSplitMixedArgs;
    using mptorch::gemm::OddPaletteArgs;

    template void CudaBackend::launch_as<double, OddPaletteArgs<BinaryKSplitMixedArgs>>(
        const GemmShape &, const OddPaletteArgs<BinaryKSplitMixedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
