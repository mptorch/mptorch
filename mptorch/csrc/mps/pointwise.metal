// The headers of the fused pointwise and table ops (common/pointwise.h),
// for their kernel, pointwise_kernel.metal: the backend compiles such a
// kernel as prelude.metal, this file, the instantiation's lines and then
// pointwise_kernel.metal (metal_runtime.mm's compile), so no other kernel's
// source contains them. carrier_odd.h brings the binary32 round-to-odd
// primitives the step calls under carrier_rounding=RO (carrier_odd_metal.h,
// division and the root included). setup.py's embed_metal_sources inlines
// this file's includes skipping the headers the prelude already holds, so
// each header appears once.

#include "../common/carrier_odd.h"
#include "../common/pointwise.h"
#include "pointwise_params.h"
