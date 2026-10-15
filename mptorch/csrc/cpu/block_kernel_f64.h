#pragma once

// The CPU kernels of the elementwise block ops in binary64, for a float64
// tensor (block_kernel_f64.cpp), as block_kernel.cpp's entry points call
// them. A file of their own, like the GEMM's *_f64 twins, so that
// MPTORCH_NO_FP64=1 leaves them out by leaving the file out.

#include "../common/block_kernels.h"
#include <cstdint>

namespace mptorch::block_cpu
{
  // block_pack, block_quant and block_quant_ under every scale rule, Four
  // Over Six's included. `seed` is the call's stochastic seed
  // (draw_cpu_seed), read only under RoundMode::SR.
  void run_pack_f64(const mptorch::block::BlockPackJob &j, const mptorch::block::BlockFormatParamsT<double> &p,
                    const mptorch::block::BlockCastT<double> &c, uint64_t seed);

  // block_unpack into a float64 result.
  void run_unpack_f64(const mptorch::block::BlockUnpackJob &j,
                      const mptorch::block::BlockFormatParamsT<double> &p);
} // namespace mptorch::block_cpu
