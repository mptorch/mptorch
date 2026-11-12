#include "custom_matmul_kernel.cuh"

// superfp formats, fused mac, in binary32: the float32, float16 and bfloat16
// kernels of custom_matmul_superfp_fma and its mixed-format twin.
// custom_matmul_superfp_fma_f64.cu holds the same two ops' binary64 kernels.
//
// This file is one of eight, one per (format family x mac mode x carrier), and
// the two explicit instantiations below are all it contains. The GEMM kernels
// compile in a file that includes no ATen tensor headers because nvcc's fixed
// cost for those is about 25 s per translation unit against about 3 s without;
// the tensors stay in custom_matmul_entry.cpp, and what crosses into here is a
// raw-pointer GemmShape plus an Args struct. Holding one family's single-format
// and mixed-format ops together instantiates that family's ~150-instruction
// cast template once, and every kernel specialization is compiled into exactly
// one object. setup.py globs this directory, so the split needs no build-script
// entry.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::SuperfpFusedArgs;
    using mptorch::gemm::SuperfpFusedMixedArgs;

    template void CudaBackend::launch_as<float, SuperfpFusedArgs>(
        const GemmShape &, const SuperfpFusedArgs &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<float, SuperfpFusedMixedArgs>(
        const GemmShape &, const SuperfpFusedMixedArgs &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
