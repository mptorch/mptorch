#pragma once

// The word tiles of the odd macs (carrier_rounding=RoundMode.RO;
// common/carrier_odd.h) under RNA, RU, RD, RZ and RO: cpu/gemm_word_tile.h's
// tiles with the carrier step rounded to odd, as cpu/gemm_odd_tile.h's are
// gemm_rne_tile.h's. The contract is the same: the scalar kernel's words, a
// K-block that leaves a NaN handed back, and the scalar loop for a format the
// word casts do not admit. cpu/gemm_word_odd_tile.cpp holds the bodies.
//
// Only the odd units include this header (through
// cpu/custom_matmul_odd_kernel.h): they alone instantiate the odd macs, so
// every unit that sees one of these types sees the same specializations.

#include "../common/carrier_odd.h"
#include "gemm_word_tile.h"

namespace mptorch::gemm_cpu
{
  // Chosen as word_tiles is (MPTORCH_CPU_SIMD included), when the extension
  // loads.
  extern const WordTiles word_odd_tiles;

  // Kinds 4 to 6, as cpu/gemm_odd_tile.h numbers the odd macs.
  template <RoundMode RM>
  struct word_tile_kind<NaiveAccumulator<
      mptorch::gemm::SplitMacOdd<BinaryKMultiplierT<float, RM>, BinaryKAdderT<float, RM>>>>
      : std::integral_constant<int, word_mode_index(RM) >= 0 ? 4 : 0>
  {
  };
  template <RoundMode RM>
  struct word_tile_kind<NaiveAccumulator<
      mptorch::gemm::SplitMacOdd<BinaryKMultiplierT<float, RM>, IdentityAdder<float>>>>
      : std::integral_constant<int, word_mode_index(RM) >= 0 ? 5 : 0>
  {
  };
  template <RoundMode RM>
  struct word_tile_kind<NaiveAccumulator<mptorch::gemm::FusedMacOdd<BinaryKAdderT<float, RM>>>>
      : std::integral_constant<int, word_mode_index(RM) >= 0 ? 6 : 0>
  {
  };

  template <RoundMode RM>
  struct WordTileRunner<
      mptorch::gemm::SplitMacOdd<BinaryKMultiplierT<float, RM>, BinaryKAdderT<float, RM>>>
  {
    template <class Mac>
    static bool run(const Mac &m, const float *a_pack, const float *b_pack, float *sums, int64_t ti,
                    int64_t tk, int64_t tj)
    {
      const RneTileFn f = word_odd_tiles.split[word_mode_index(RM)];
      if (!f || !word_tile_format(m.mul) || !word_tile_format(m.add))
        return false;
      return f(a_pack, b_pack, sums, ti, tk, tj, m.mul.params, m.add.params);
    }
  };
  template <RoundMode RM>
  struct WordTileRunner<
      mptorch::gemm::SplitMacOdd<BinaryKMultiplierT<float, RM>, IdentityAdder<float>>>
  {
    template <class Mac>
    static bool run(const Mac &m, const float *a_pack, const float *b_pack, float *sums, int64_t ti,
                    int64_t tk, int64_t tj)
    {
      const RneTileFn f = word_odd_tiles.mul_only[word_mode_index(RM)];
      if (!f || !word_tile_format(m.mul))
        return false;
      return f(a_pack, b_pack, sums, ti, tk, tj, m.mul.params, m.mul.params);
    }
  };
  template <RoundMode RM>
  struct WordTileRunner<mptorch::gemm::FusedMacOdd<BinaryKAdderT<float, RM>>>
  {
    template <class Mac>
    static bool run(const Mac &m, const float *a_pack, const float *b_pack, float *sums, int64_t ti,
                    int64_t tk, int64_t tj)
    {
      const RneTileFn f = word_odd_tiles.fused[word_mode_index(RM)];
      if (!f || !word_tile_format(m.add))
        return false;
      return f(a_pack, b_pack, sums, ti, tk, tj, m.add.params, m.add.params);
    }
  };
} // namespace mptorch::gemm_cpu
