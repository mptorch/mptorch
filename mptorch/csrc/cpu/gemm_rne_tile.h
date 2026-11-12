#pragma once

// The CPU GEMM's binaryK RNE steps, vectorized (phase P of
// dev/continuation_plan.md, step 1).
//
// cpu/custom_matmul_kernel.h runs a packed tile's K-block through one of the
// functions below in place of its scalar loop when that computes the same
// thing: a NAIVE, single-format binaryK GEMM in binary32 under RNE whose
// formats are signed, under SUBNORMALS, and admitted by make_binaryK_params'
// fast gate, which is exactly when the scalar step's cast_binaryK_nearest_even
// is cast_binaryK_rne_fast. The vector functions call that same cast
// (cpu/gemm_rne_tile.cpp), one output element per lane and every element's
// K order unchanged, so the result is the scalar kernel's word for word; the
// split mac's two casts, its product cast alone over a carrier sum, and the
// fused mac's one. With one exception the tile cannot see coming: a sum of two
// NaNs (an overflow's inf - inf, say, meeting a NaN input) returns one of
// them by operand order, which the vectorizer does not keep. So a K-block
// whose tile holds a NaN at its end is handed back, its sums as they were,
// and the scalar loop runs it. Which ISA runs is chosen once, when the extension loads
// (rne_tiles); a null entry, or MPTORCH_CPU_SIMD=0, keeps the scalar loop.
//
// Every other kernel instantiation sees only declarations here: the trait is
// 0 for it, and the kernel's branch is an `if constexpr` that names nothing
// the lambda does not already capture or declare (rule 11), so its code is
// what it was.

#include "../common/gemm_gather.h"
#include "../common/gemm_policy.h"

#include <cstdint>
#include <type_traits>

namespace mptorch::gemm_cpu
{
  // One packed tile's K-block: sums[i * tj + j] takes the tk steps of a_pack
  // (ti x tk, row-major) against b_pack (tk x tj), in K order, and true is
  // returned; or a NaN came out, the sums are put back as they were, and false
  // is (above). `mul` is the product's format and `add` the sum's; a kind
  // that has one format takes it for both. ti * tj is at most RNE_TILE_MAX.
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

  // AVX-512 or AVX2 + FMA on x86-64, by what the CPU reports, NEON on arm64;
  // all null without MPTORCH_FAST_CAST, on any other CPU, or under
  // MPTORCH_CPU_SIMD=0, and AVX2 at most under MPTORCH_CPU_SIMD=avx2 (read
  // once, when the extension loads).
  extern const RneTiles rne_tiles;

  // Which tile a kernel's accumulator can run: 1 split, 2 product only,
  // 3 fused, 0 none. The conv ops' Gathered<Acc> runs its inner one's: only
  // the packing differs, and it happens before the tile.
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

  // Whether a format's scalar RNE cast is cast_binaryK_rne_fast, with nothing
  // before it: signed (the vector tile has no unsigned test), SUBNORMALS and
  // the fast gate (cast_binaryK_nearest_even's two tests).
  template <class Policy>
  inline bool rne_tile_format(const Policy &q)
  {
    return q.is_signed && q.subnormals_mode == SubnormalsMode::SUBNORMALS && q.params.fast_rne;
  }

  // Runs the K-block through the vector tile and returns true, or returns
  // false and leaves it, as it was, to the scalar loop.
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
