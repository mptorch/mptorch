#include "custom_matmul_kernel.cuh"
#include "../common/carrier_odd_mixed.h"

// The cross-family conv ops, carrier step rounded to odd, in binary32: the
// kernels behind custom_conv_binaryK_superfp_odd and
// custom_conv_superfp_binaryK_odd (carrier_rounding=RoundMode.RO) for float32,
// float16 and bfloat16 operands. The kernel template is the NAIVE ops' own;
// only the Args it is instantiated with are new, and no unit that existed
// before these ops includes the headers they come from, so their objects are
// unchanged. Its binary64 twin is custom_conv_cross_odd_f64.cu. The entry
// points are in custom_conv_odd_entry.cpp.

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
