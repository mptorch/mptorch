#include "custom_matmul_kernel.cuh"

// binaryK formats, fused mac: custom_matmul_binaryK_fma and its mixed-format twin.
//
// The cut is one translation unit per (format family x mac mode), which keeps
// that family's cast template (cast_binaryK.h / cast_superfp.h -- a ~150
// -instruction body) instantiated once rather than twice, and keeps the
// single-format and mixed-format ops whose Mac types are the same together.
// No kernel specialization is therefore compiled into two objects. setup.py
// globs this directory, so the split is a matter of adding files; see
// dev/gemm_roadmap.md (finding B2).
//
// Since H1 that is all a .cu is: the entry points that used to sit here are
// custom_matmul_entry.cpp, and what crosses the boundary is a raw-pointer
// GemmShape plus an Args (common/gemm_args.h), neither of which mentions
// at::Tensor. These two lines are what put this file's kernels in this
// object -- in binary32; custom_matmul_binaryK_fma_f64.cu holds the same two
// ops' binary64 kernels (dev/binary64_carrier_plan.md, phase 4).

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::BinaryKFusedMixedArgs;

    template void CudaBackend::launch_as<float, BinaryKFusedArgs>(
        const GemmShape &, const BinaryKFusedArgs &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<float, BinaryKFusedMixedArgs>(
        const GemmShape &, const BinaryKFusedMixedArgs &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
