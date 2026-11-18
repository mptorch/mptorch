#pragma once

// Vector code for the CPU GEMM's binaryK steps under RNE when the carrier
// step rounds to odd (carrier_rounding=RoundMode.RO; common/carrier_odd.h):
// cpu/gemm_rne_tile.h's tiles for the odd macs.
//
// The contract is cpu/gemm_rne_tile.h's: a packed tile's K-block runs here
// in place of the scalar loop when the two compute the same thing, which is
// when the GEMM is a NAIVE single-format binaryK GEMM in binary32 under RNE
// whose formats are signed, under SUBNORMALS and admitted by the fast gate.
// Each lane computes the scalar step (carrier_odd.h's odd primitive, then
// cast_binaryK_rne_fast) in K order, and a K-block that leaves a NaN in its
// tile is handed back with its sums restored. cpu/gemm_odd_tile.cpp holds
// the bodies.
//
// Only the odd units include this header (through
// cpu/custom_matmul_odd_kernel.h), so gemm_rne_tile.h and every default
// object do not depend on it. Each odd kind (4 to 6) is a specialization of
// both rne_tile_kind and run_rne_tile, and the two name the Mac through one
// alias. Both are needed: run_rne_tile's primary template treats every kind
// other than 1 and 2 as the fused RNE tile, and an odd Mac, which holds the
// same policies as the default one, would compile against it.

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

  // Chosen as rne_tiles is (MPTORCH_CPU_SIMD included), when the extension
  // loads.
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
