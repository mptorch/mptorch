// The CPU kernel of block_pack, block_quant and block_quant_ under a Four Over
// Six rule (common/block_decode.h defines the rule; block_kernel.cpp holds
// the kernel of every other rule, and the entry points).
//
// It is pack_tiles (cpu/block_pack_tiles.h) with one more pass. A tile,
// block_rows rows of one block, has two candidate scales, tile_scale's and
// four_over_six_low's. Where their codes differ, a pass between the amax pass
// and the encode pass rounds every element of the tile under both, each with
// the random draw the encode pass would give it, adds up each candidate's
// error in the fixed-point arithmetic of common/block_decode.h, and takes the
// second candidate if its error is strictly smaller. The errors are
// integers, so the choice does not depend on the order of the sum: it is the
// CUDA kernel's choice, and for a square tile it is the same whichever of
// the tile's two axes the tensor is packed along. The pass keeps both
// candidates' codes, and the encode pass writes the chosen ones without
// casting a third time.
//
// The new pass only reads, so the in-place quantizer is as sound as under
// every other rule: every read of a tile completes before its first write.
//
// This is a file of its own, and the loops it shares with pack_tiles are
// written twice, because the rule inside pack_tiles changed every other
// rule's instantiation (block_kernel.cpp). The kernel is a template in
// cpu/block_pack_tiles_fos.h, which block_kernel_f64.cpp instantiates for
// float64.

#include "block_kernel_fos.h"
#include "block_pack_tiles_fos.h"

using namespace mptorch::block;

namespace mptorch::block_cpu
{
  void run_pack_fos(const BlockPackJob &j, const BlockFormatParams &p, const BlockCast &c, uint64_t seed)
  {
    switch (j.dt)
    {
    case mptorch::GemmDtype::Half:
      return pack_fos_by_mode<c10::Half>(j, p, c, seed);
    case mptorch::GemmDtype::BFloat16:
      return pack_fos_by_mode<c10::BFloat16>(j, p, c, seed);
    default:
      return pack_fos_by_mode<float>(j, p, c, seed);
    }
  }
} // namespace mptorch::block_cpu
