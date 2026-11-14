#include "custom_matmul_kernel.cuh"
#include "../common/carrier_odd_mixed.h"

// The conv ops over superfp formats, split mac, carrier step rounded to odd, in
// binary32: the kernels behind custom_conv_superfp_odd and
// custom_conv_superfp_mixed_odd (carrier_rounding=RoundMode.RO) for float32,
// float16 and bfloat16 operands. The kernel template is the NAIVE ops' own;
// only the Args it is instantiated with are new, and no unit that existed
// before these ops includes the headers they come from, so their objects are
// unchanged. Its binary64 twin is custom_conv_superfp_odd_f64.cu. The entry
// points are in custom_conv_odd_entry.cpp.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::OddArgs;
    using mptorch::gemm::OddPaletteArgs;
    using mptorch::gemm::SuperfpSplitArgs;
    using mptorch::gemm::SuperfpSplitMixedArgs;

    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, OddArgs<SuperfpSplitArgs>>(
        const GemmShape &, const ConvArgs<OddArgs<SuperfpSplitArgs>> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, OddPaletteArgs<SuperfpSplitMixedArgs>>(
        const GemmShape &, const ConvArgs<OddPaletteArgs<SuperfpSplitMixedArgs>> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
