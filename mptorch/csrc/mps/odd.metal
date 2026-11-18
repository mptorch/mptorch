// The headers of the odd carrier step (carrier_rounding=RoundMode.RO), for
// the kernels that need them: the odd GEMMs (gemm.metal under an odd Args)
// and _carrier_odd (carrier_odd.metal). The backend compiles such a kernel
// as prelude.metal, this file, the instantiation's lines and then the kernel
// file (metal_runtime.mm's compile), so no other kernel's source contains
// these headers. setup.py's embed_metal_sources inlines this file's
// includes skipping the headers the prelude already holds, so each header
// appears once.

#include "../common/carrier_odd_mixed.h"
#include "../common/carrier_odd_accumulate.h"
