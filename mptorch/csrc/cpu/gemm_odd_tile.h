#pragma once

// The CPU GEMM's binaryK RNE steps with the carrier rounded to odd,
// vectorized: cpu/gemm_rne_tile.h's tiles for the odd macs
// (carrier_rounding=RoundMode.RO; common/carrier_odd.h).
//
// The same contract as gemm_rne_tile.h's: a packed tile's K-block runs here
// in place of the scalar loop when that computes the same thing, which is a
// NAIVE single-format binaryK GEMM in binary32 under RNE whose formats are
// signed, under SUBNORMALS and admitted by the fast gate. Each lane computes
// the scalar step, carrier_odd.h's odd primitive and cast_binaryK_rne_fast,
// in K order, and a K-block whose tile ends with a NaN is handed back with
// its sums as they were. cpu/gemm_odd_tile.cpp holds the bodies.
//
// A header of its own, which only the odd units include (through
// cpu/custom_matmul_odd_kernel.h), so gemm_rne_tile.h, and with it every
// default object, is what it was. Each odd kind is a specialization of
// gemm_rne_tile.h's trait and of its run_rne_tile, both named by one alias:
// run_rne_tile's primary would take any other kind for the fused RNE tile,
// which an odd Mac, holding the same policies, would compile against.

#include "../common/carrier_odd.h"
#include "gemm_rne_tile.h"

namespace mptorch::gemm_cpu
{
  struct OddTiles
  {
    RneTileFn split = nullptr;    // SplitMacOdd: add(odd::add(acc, mul(odd::mul(a, b))))
    RneTileFn mul_only = nullptr; // SplitMacOdd over an IdentityAdder: acc + mul(odd::mul(a, b))
    RneTileFn fused = nullptr;    // FusedMacOdd: add(odd::fma(a, b, acc))
    const char *isa = "none";
  };

  // Chosen as rne_tiles is (MPTORCH_CPU_SIMD included), when the extension loads.
  extern const OddTiles odd_tiles;

  using OddSplitTileMac = mptorch::gemm::SplitMacOdd<BinaryKMultiplierT<float, RoundMode::RNE>,
                                                     BinaryKAdderT<float, RoundMode::RNE>>;
  using OddMulOnlyTileMac =
      mptorch::gemm::SplitMacOdd<BinaryKMultiplierT<float, RoundMode::RNE>, IdentityAdder<float>>;
  using OddFusedTileMac = mptorch::gemm::FusedMacOdd<BinaryKAdderT<float, RoundMode::RNE>>;

  template <>
  struct rne_tile_kind<NaiveAccumulator<OddSplitTileMac>> : std::integral_constant<int, 4>
  {
  };
  template <>
  inline bool run_rne_tile<4, OddSplitTileMac>(const OddSplitTileMac &m, const float *a_pack,
                                               const float *b_pack, float *sums, int64_t ti,
                                               int64_t tk, int64_t tj)
  {
    if (!odd_tiles.split || !rne_tile_format(m.mul) || !rne_tile_format(m.add))
      return false;
    return odd_tiles.split(a_pack, b_pack, sums, ti, tk, tj, m.mul.params, m.add.params);
  }

  template <>
  struct rne_tile_kind<NaiveAccumulator<OddMulOnlyTileMac>> : std::integral_constant<int, 5>
  {
  };
  template <>
  inline bool run_rne_tile<5, OddMulOnlyTileMac>(const OddMulOnlyTileMac &m, const float *a_pack,
                                                 const float *b_pack, float *sums, int64_t ti,
                                                 int64_t tk, int64_t tj)
  {
    if (!odd_tiles.mul_only || !rne_tile_format(m.mul))
      return false;
    return odd_tiles.mul_only(a_pack, b_pack, sums, ti, tk, tj, m.mul.params, m.mul.params);
  }

  template <>
  struct rne_tile_kind<NaiveAccumulator<OddFusedTileMac>> : std::integral_constant<int, 6>
  {
  };
  template <>
  inline bool run_rne_tile<6, OddFusedTileMac>(const OddFusedTileMac &m, const float *a_pack,
                                               const float *b_pack, float *sums, int64_t ti,
                                               int64_t tk, int64_t tj)
  {
    if (!odd_tiles.fused || !rne_tile_format(m.add))
      return false;
    return odd_tiles.fused(a_pack, b_pack, sums, ti, tk, tj, m.add.params, m.add.params);
  }
} // namespace mptorch::gemm_cpu
