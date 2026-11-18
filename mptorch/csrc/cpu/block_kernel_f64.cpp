// The CPU kernels of block_pack, block_unpack, block_quant and block_quant_
// for float64 tensors, which round in binary64: the templates of
// cpu/block_pack_tiles.h and cpu/block_pack_tiles_fos.h instantiated for
// double. Their binary32 instantiations are in block_kernel.cpp and
// block_kernel_fos.cpp, and the entry points in block_kernel.cpp.

#include "block_kernel_f64.h"
#include "block_pack_tiles.h"
#include "block_pack_tiles_fos.h"

namespace mptorch::block_cpu
{
  using namespace mptorch::block;

  void run_pack_f64(const BlockPackJob &j, const BlockFormatParamsT<double> &p, const BlockCastT<double> &c,
                    uint64_t seed)
  {
    if (is_four_over_six(c.scale_rounding))
      return pack_fos_by_mode<double>(j, p, c, seed);
    pack_by_mode<double>(j, p, c, seed);
  }

  void run_unpack_f64(const BlockUnpackJob &j, const BlockFormatParamsT<double> &p)
  {
    unpack_rows<double>(j, p);
  }
} // namespace mptorch::block_cpu
