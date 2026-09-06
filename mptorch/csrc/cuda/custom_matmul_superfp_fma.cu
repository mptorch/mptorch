#include "custom_matmul_kernel.cuh"

// superfp formats, fused mac: custom_matmul_superfp_fma and its mixed-format twin.
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
// object.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::SuperfpFusedArgs;
    using mptorch::gemm::SuperfpFusedMixedArgs;

    template void CudaBackend::launch<SuperfpFusedArgs>(
        const GemmShape &, const SuperfpFusedArgs &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch<SuperfpFusedMixedArgs>(
        const GemmShape &, const SuperfpFusedMixedArgs &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
