#pragma once

// The CPU kernel of block_pack, block_quant and block_quant_ under a Four Over
// Six rule (block_kernel_fos.cpp), as block_kernel.cpp's run_pack calls it.
// It is in a file of its own so that every other rule's kernel compiles to
// the code it would without the rule.

#include "../common/block_kernels.h"
#include <cstdint>

namespace mptorch::block_cpu
{
  // `seed` is the call's seed for stochastic rounding (draw_cpu_seed), read
  // only under RoundMode::SR.
  void run_pack_fos(const mptorch::block::BlockPackJob &j, const mptorch::block::BlockFormatParams &p,
                    const mptorch::block::BlockCast &c, uint64_t seed);
} // namespace mptorch::block_cpu
