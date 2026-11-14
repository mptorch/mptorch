#include "custom_matmul_kernel.cuh"
#include "../common/carrier_odd_mixed.h"

// superfp formats, fused mac, a palette, carrier step rounded to odd, in
// binary64: the kernels behind custom_matmul_superfp_fma_mixed_odd
// (carrier_rounding=RoundMode.RO) for float64 operands. The kernel template is
// the NAIVE ops' own; only the Args it is instantiated with are new, and no
// unit that existed before these ops includes the headers they come from, so
// their objects are unchanged. Its binary32 twin is
// custom_matmul_superfp_fma_mixed_odd.cu. The entry points are in
// custom_matmul_odd_entry.cpp.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::OddPaletteArgs;
    using mptorch::gemm::SuperfpFusedMixedArgs;

    template void CudaBackend::launch_as<double, OddPaletteArgs<SuperfpFusedMixedArgs>>(
        const GemmShape &, const OddPaletteArgs<SuperfpFusedMixedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
