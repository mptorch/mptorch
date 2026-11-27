#pragma once

// Vector code for the CPU GEMM's binaryK steps under RNA, RU, RD, RZ and RO,
// which the kernel (cpu/custom_matmul_kernel.h) runs in place of its scalar
// loop when the two compute the same thing: the companion of
// cpu/gemm_rne_tile.h for the deterministic modes that have no
// float-arithmetic fast path.
//
// The scalar step casts through cast_binaryK_up, _down, _zero, _nearest_away
// or _odd, which branch on the input's sign and range: on a GEMM's sums the
// sign is a coin flip, and the loop mispredicts on most steps (10.9 ns a
// step under RZ on one Ryzen thread, against 0.4 for the RNE tile). The
// tiles run common/cast_binaryK_word.h's branchless forms of the same casts,
// one output element per lane, every element's steps in K order, so the
// result is the scalar kernel's word for word. The contract is
// cpu/gemm_rne_tile.h's: a NAIVE single-format binaryK GEMM in binary32
// whose formats the word casts admit (signed, SUBNORMALS, 1 to 22 stored mantissa bits) runs a packed
// tile's K-block here; a K-block that leaves a NaN in its tile is handed back
// with its sums restored (a sum of two NaNs returns one by operand order,
// which the vectorizer does not keep); a null table entry or
// MPTORCH_CPU_SIMD=0 keeps the scalar loop.
//
// Its own trait, word_tile_kind, with its own branch in the kernel: the
// palette ops dispatch on rne_tile_kind too, and a kind added there would
// move their instantiations. Every instantiation other than the five modes'
// single-format split, product-only and fused macs (and their conv twins)
// sees word_tile_kind as 0 and compiles to what it did without the branch.

#include "../common/cast_binaryK_word.h"
#include "gemm_rne_tile.h"

namespace mptorch::gemm_cpu
{
  // A mode's index in WordTiles' arrays, or -1 for a mode the word casts do
  // not serve.
  constexpr int word_mode_index(RoundMode rm)
  {
    return rm == RoundMode::RNA  ? 0
           : rm == RoundMode::RU ? 1
           : rm == RoundMode::RD ? 2
           : rm == RoundMode::RZ ? 3
           : rm == RoundMode::RO ? 4
                                 : -1;
  }

  struct WordTiles
  {
    RneTileFn split[5] = {};    // SplitMac: add(acc + mul(a * b))
    RneTileFn mul_only[5] = {}; // SplitMac over an IdentityAdder: acc + mul(a * b)
    RneTileFn fused[5] = {};    // FusedMac: add(fma(a, b, acc))
    const char *isa = "none";
  };

  // Chosen as rne_tiles is (MPTORCH_CPU_SIMD included), when the extension
  // loads.
  extern const WordTiles word_tiles;

  // Which tile a kernel's accumulator can run: 1 split, 2 product only, 3
  // fused, 0 none (as rne_tile_kind numbers them).
  template <class Acc>
  struct word_tile_kind : std::integral_constant<int, 0>
  {
  };
  template <RoundMode RM>
  struct word_tile_kind<
      NaiveAccumulator<SplitMac<BinaryKMultiplierT<float, RM>, BinaryKAdderT<float, RM>>>>
      : std::integral_constant<int, word_mode_index(RM) >= 0 ? 1 : 0>
  {
  };
  template <RoundMode RM>
  struct word_tile_kind<
      NaiveAccumulator<SplitMac<BinaryKMultiplierT<float, RM>, IdentityAdder<float>>>>
      : std::integral_constant<int, word_mode_index(RM) >= 0 ? 2 : 0>
  {
  };
  template <RoundMode RM>
  struct word_tile_kind<NaiveAccumulator<FusedMac<BinaryKAdderT<float, RM>>>>
      : std::integral_constant<int, word_mode_index(RM) >= 0 ? 3 : 0>
  {
  };
  template <class Acc>
  struct word_tile_kind<mptorch::gemm::Gathered<Acc>> : word_tile_kind<Acc>
  {
  };

  // Whether a format's scalar cast in these modes is the word cast's.
  template <class Policy>
  inline bool word_tile_format(const Policy &q)
  {
    return word_cast_format(q.is_signed, q.subnormals_mode, q.params);
  }

  // Runs the K-block in the vector tile and returns true, or returns false
  // and leaves the K-block, unchanged, to the scalar loop: a specialization
  // per mac, the mode from its policies. A class template rather than
  // overloads, so that cpu/gemm_word_odd_tile.h can add the odd macs'
  // specializations, which only the odd units see (the kernel's call is
  // dependent, and argument-dependent lookup would not find overloads in this
  // namespace for macs declared in mptorch::gemm).
  template <class Mac>
  struct WordTileRunner
  {
    static bool run(const Mac &, const float *, const float *, float *, int64_t, int64_t, int64_t)
    {
      return false;
    }
  };
  template <RoundMode RM>
  struct WordTileRunner<SplitMac<BinaryKMultiplierT<float, RM>, BinaryKAdderT<float, RM>>>
  {
    template <class Mac>
    static bool run(const Mac &m, const float *a_pack, const float *b_pack, float *sums, int64_t ti,
                    int64_t tk, int64_t tj)
    {
      const RneTileFn f = word_tiles.split[word_mode_index(RM)];
      if (!f || !word_tile_format(m.mul) || !word_tile_format(m.add))
        return false;
      return f(a_pack, b_pack, sums, ti, tk, tj, m.mul.params, m.add.params);
    }
  };
  template <RoundMode RM>
  struct WordTileRunner<SplitMac<BinaryKMultiplierT<float, RM>, IdentityAdder<float>>>
  {
    template <class Mac>
    static bool run(const Mac &m, const float *a_pack, const float *b_pack, float *sums, int64_t ti,
                    int64_t tk, int64_t tj)
    {
      const RneTileFn f = word_tiles.mul_only[word_mode_index(RM)];
      if (!f || !word_tile_format(m.mul))
        return false;
      return f(a_pack, b_pack, sums, ti, tk, tj, m.mul.params, m.mul.params);
    }
  };
  template <RoundMode RM>
  struct WordTileRunner<FusedMac<BinaryKAdderT<float, RM>>>
  {
    template <class Mac>
    static bool run(const Mac &m, const float *a_pack, const float *b_pack, float *sums, int64_t ti,
                    int64_t tk, int64_t tj)
    {
      const RneTileFn f = word_tiles.fused[word_mode_index(RM)];
      if (!f || !word_tile_format(m.add))
        return false;
      return f(a_pack, b_pack, sums, ti, tk, tj, m.add.params, m.add.params);
    }
  };

  template <class Mac>
  inline bool run_word_tile(const Mac &m, const float *a_pack, const float *b_pack, float *sums,
                            int64_t ti, int64_t tk, int64_t tj)
  {
    return WordTileRunner<Mac>::run(m, a_pack, b_pack, sums, ti, tk, tj);
  }
} // namespace mptorch::gemm_cpu
