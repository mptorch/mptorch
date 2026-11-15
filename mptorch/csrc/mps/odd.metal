// The odd carrier step's headers (carrier_rounding=RoundMode.RO, phase Z),
// for the kernels that need them: the odd GEMMs (gemm.metal under an odd
// Args) and _carrier_odd (carrier_odd.metal). The backend compiles them as
// prelude.metal, this, the instantiation's lines, then the kernel
// (metal_runtime.mm), so every other kernel's source is what it was.
// setup.py's embed_metal_sources flattens this file against the headers the
// prelude already holds, so each is inlined once.

#include "../common/carrier_odd_mixed.h"
#include "../common/carrier_odd_accumulate.h"
