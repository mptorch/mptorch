// The header of the counted GEMMs (common/gemm_counted.h), for the kernels
// that need it: gemm.metal under a CountingArgs, which the counted binaryK
// and superfp GEMM and conv ops build, compiled under MPT_COUNTING. The
// backend compiles such a kernel as prelude.metal, this file, the
// instantiation's lines and then gemm.metal (metal_runtime.mm's compile), so
// no other kernel's source contains the header. setup.py's
// embed_metal_sources inlines this file's includes skipping the headers the
// prelude already holds, so each header appears once.

// CountingArgs names the type of the mac it builds through
// std::remove_const_t, which the prelude's std has no spelling of.
namespace std
{
using metal::remove_const_t;
} // namespace std

#include "../common/gemm_counted.h"
