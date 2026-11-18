#pragma once

// Vector code for the CPU GEMM's binaryK steps under RNE, which the kernel
// (cpu/custom_matmul_kernel.h) runs in place of its scalar loop when the two
// compute the same thing.
//
// The kernel runs a packed tile's K-block (the tk steps of the current tile)
// through one of the functions below when the GEMM is a NAIVE, single-format
// binaryK GEMM in binary32 under RNE whose formats are signed, under
// SUBNORMALS, and admitted by make_binaryK_params' fast gate: exactly the
// case in which the scalar step's cast_binaryK_nearest_even takes its fast
// path, cast_binaryK_rne_fast. The vector functions (cpu/gemm_rne_tile.cpp)
// call that same cast, one output element per lane, with every element's
// steps in K order, so the result is the scalar kernel's word for word.
// There are three: the split mac's two casts, its product cast alone over a
// sum kept in the carrier, and the fused mac's one cast. One case cannot be
// predicted: a sum of two NaNs (an overflow's inf - inf meeting a NaN input,
// say) returns one of them according to the operand order, which the
// vectorizer does not preserve. So a K-block that leaves a NaN in its tile is
// handed back with its sums restored, and the scalar loop runs it. Which
// instruction set runs is chosen once, when the extension loads (rne_tiles);
// a null entry, or MPTORCH_CPU_SIMD=0, keeps the scalar loop.
//
// Every other kernel instantiation sees only declarations here: rne_tile_kind
// is 0 for it, and the kernel's branch is an `if constexpr` that names
// nothing the lambda does not already capture or declare, so its code is
// what it would be without the branch.

#include "../common/gemm_gather.h"
#include "../common/gemm_policy.h"

#include <cstdint>
#include <type_traits>

namespace mptorch::gemm_cpu
{
  // One packed tile's K-block: sums[i * tj + j] takes the tk steps of a_pack
  // (ti x tk, row major) against b_pack (tk x tj), in K order, and the
  // function returns true; or a NaN came out, the sums are put back as they
  // were, and it returns false (see the top of the file). `mul` is the
  // product's format and `add` the sum's; a tile with one format is passed it
  // as both. ti * tj is at most RNE_TILE_MAX.
  constexpr int64_t RNE_TILE_MAX = 32 * 32;
  using RneTileFn = bool (*)(const float *a_pack, const float *b_pack, float *sums, int64_t ti,
                             int64_t tk, int64_t tj, const BinaryKParams &mul,
                             const BinaryKParams &add);

  struct RneTiles
  {
    RneTileFn split = nullptr;    // SplitMac: add(acc + mul(a * b))
    RneTileFn mul_only = nullptr; // SplitMac over an IdentityAdder: acc + mul(a * b)
    RneTileFn fused = nullptr;    // FusedMac: add(fma(a, b, acc))
    const char *isa = "none";
  };

  // AVX-512, or AVX2 with FMA, on x86-64, by what the CPU reports, and NEON
  // on arm64. All null without MPTORCH_FAST_CAST, on any other CPU, or under
  // MPTORCH_CPU_SIMD=0, and AVX2 at most under MPTORCH_CPU_SIMD=avx2 (the
  // variable is read once, when the extension loads).
  extern const RneTiles rne_tiles;

  // Which tile a kernel's accumulator can run: 1 the split mac's, 2 the
  // product-only one's, 3 the fused mac's, 0 none (cpu/gemm_odd_tile.h adds
  // kinds for the odd macs). The conv ops' Gathered<Acc> runs the tile of the
  // accumulator it wraps: only the packing differs, and it happens before the
  // tile runs.
  template <class Acc>
  struct rne_tile_kind : std::integral_constant<int, 0>
  {
  };
  template <>
  struct rne_tile_kind<NaiveAccumulator<
      SplitMac<BinaryKMultiplierT<float, RoundMode::RNE>, BinaryKAdderT<float, RoundMode::RNE>>>>
      : std::integral_constant<int, 1>
  {
  };
  template <>
  struct rne_tile_kind<
      NaiveAccumulator<SplitMac<BinaryKMultiplierT<float, RoundMode::RNE>, IdentityAdder<float>>>>
      : std::integral_constant<int, 2>
  {
  };
  template <>
  struct rne_tile_kind<NaiveAccumulator<FusedMac<BinaryKAdderT<float, RoundMode::RNE>>>>
      : std::integral_constant<int, 3>
  {
  };
  template <class Acc>
  struct rne_tile_kind<mptorch::gemm::Gathered<Acc>> : rne_tile_kind<Acc>
  {
  };

  // Whether a format's scalar RNE cast goes straight to cast_binaryK_rne_fast:
  // the format is signed (the vector tile has no test for unsigned formats),
  // under SUBNORMALS, and passes the fast gate (cast_binaryK_nearest_even's
  // two tests).
  template <class Policy>
  inline bool rne_tile_format(const Policy &q)
  {
    return q.is_signed && q.subnormals_mode == SubnormalsMode::SUBNORMALS && q.params.fast_rne;
  }

  // Runs the K-block in the vector tile and returns true, or returns false
  // and leaves the K-block, unchanged, to the scalar loop.
  template <int KIND, class Mac>
  inline bool run_rne_tile(const Mac &m, const float *a_pack, const float *b_pack, float *sums,
                           int64_t ti, int64_t tk, int64_t tj)
  {
    if constexpr (KIND == 1)
    {
      if (!rne_tiles.split || !rne_tile_format(m.mul) || !rne_tile_format(m.add))
        return false;
      return rne_tiles.split(a_pack, b_pack, sums, ti, tk, tj, m.mul.params, m.add.params);
    }
    else if constexpr (KIND == 2)
    {
      if (!rne_tiles.mul_only || !rne_tile_format(m.mul))
        return false;
      return rne_tiles.mul_only(a_pack, b_pack, sums, ti, tk, tj, m.mul.params, m.mul.params);
    }
    else
    {
      if (!rne_tiles.fused || !rne_tile_format(m.add))
        return false;
      return rne_tiles.fused(a_pack, b_pack, sums, ti, tk, tj, m.add.params, m.add.params);
    }
  }
} // namespace mptorch::gemm_cpu
