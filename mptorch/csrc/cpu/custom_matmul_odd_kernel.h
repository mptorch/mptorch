#pragma once

// The CPU kernel as every odd unit instantiates it (carrier_rounding=
// RoundMode.RO): the shared kernel template, then the odd macs and Args
// (common/carrier_odd.h, common/carrier_odd_mixed.h) and the odd macs' vector
// tiles (cpu/gemm_odd_tile.h).
//
// Every CPU unit that instantiates the kernel over an odd accumulator
// includes this one header and nothing else of the kind, so every such unit
// sees the same declarations: the kernel names a trait of its accumulator
// (cpu/gemm_rne_tile.h's rne_tile_kind) that an odd specialization may
// answer, and two units that saw different answers for the same type would
// break the one-definition rule without a diagnostic.

#include "custom_matmul_kernel.h"
#include "../common/carrier_odd.h"
#include "../common/carrier_odd_mixed.h"
#include "gemm_odd_tile.h"
