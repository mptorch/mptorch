// The headers of the counted fixed-point GEMMs (common/fixedpoint_counted.h,
// which includes common/fixedpoint_gemm.h and common/gemm_counted.h), for the
// kernels that need them: gemm.metal under a FixedPointCountingArgs, which
// the counted fixed-point GEMM and conv ops build, compiled under
// MPT_COUNTING. The backend compiles such a kernel as prelude.metal, this
// file, the instantiation's lines and then gemm.metal (metal_runtime.mm's
// compile), so no other kernel's source contains these headers. setup.py's
// embed_metal_sources inlines this file's includes skipping the headers the
// prelude already holds, so each header appears once.

// The counted Args name the type of the mac they build through
// std::remove_const_t, which the prelude's std has no spelling of.
namespace std
{
using metal::remove_const_t;
} // namespace std

#include "../common/fixedpoint_counted.h"
