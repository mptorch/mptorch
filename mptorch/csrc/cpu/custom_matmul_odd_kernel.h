#pragma once

// The CPU kernel as every odd unit (carrier_rounding=RoundMode.RO)
// instantiates it: the shared kernel template, then the odd macs and Args
// (common/carrier_odd.h, common/carrier_odd_mixed.h) and the odd macs'
// vector tiles (cpu/gemm_odd_tile.h, and cpu/gemm_word_odd_tile.h for the
// modes other than RNE).
//
// Every CPU unit that instantiates the kernel over an odd accumulator
// includes this one header and no other header of its kind, so all such
// units see the same declarations. That matters because the kernel reads a
// trait of its accumulator (cpu/gemm_rne_tile.h's rne_tile_kind,
// cpu/gemm_word_tile.h's word_tile_kind) that an odd
// specialization may answer, and two units that saw different answers for
// the same type would break the one-definition rule without a diagnostic.

#include "custom_matmul_kernel.h"
#include "../common/carrier_odd.h"
#include "../common/carrier_odd_mixed.h"
#include "gemm_odd_tile.h"
#include "gemm_word_odd_tile.h"
