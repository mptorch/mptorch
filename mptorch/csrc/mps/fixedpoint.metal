// The header of the fixed-point GEMM (phase T; common/fixedpoint_gemm.h), for
// the kernels that need it: gemm.metal under a fixed-point Args
// (FixedPointSplitArgs, the carrier path, or a FixedPointIntArgs, the
// integer accumulator), which the fixed-point GEMM and conv ops build. The
// backend compiles such a kernel as prelude.metal, this file, the
// instantiation's lines and then gemm.metal (metal_runtime.mm's compile), so
// no other kernel's source contains the header. setup.py's
// embed_metal_sources inlines this file's includes skipping the headers the
// prelude already holds, so each header appears once.

#include "../common/fixedpoint_gemm.h"
