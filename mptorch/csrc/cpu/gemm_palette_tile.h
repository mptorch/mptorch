#pragma once

// The CPU palette GEMMs' vector tiles: cpu/gemm_rne_tile.h's tiles (and
// cpu/gemm_odd_tile.h's, for the odd macs) run once per palette entry over
// the elements of a packed tile that take that entry.
//
// A palette GEMM (MIXED in cpu/custom_matmul_kernel.h) looks up each output
// element's Mac in its palette before the element's reduction starts
// (slot_of). The Macs are the single-format kernel's policies, so where every
// Mac of a tile is one the single-format kernel would run in a vector tile,
// the K-block runs here, one group of elements per entry, the groups chosen
// from the tile's slots:
//   - one entry: the tile itself;
//   - an entry per column (every row of the tile holds the same entries, a
//     linear layer's per-neuron map): each entry's columns of b_pack and of
//     the sums gathered into a narrower tile, run, and scattered back;
//   - an entry per row (a conv layer's per-channel map, whose channels are
//     the GEMM's rows): each entry's rows of a_pack and of the sums likewise;
//   - any other map: each entry's tile run on the whole tile from the
//     K-block's starting sums, of which the entry's elements keep their own
//     (the work times the number of entries the tile holds).
// The split, product-only and fused macs (kinds 1-3) have a faster spelling
// of the last two, the lane tiles (cpu/gemm_palette_lanes.cpp): the tile runs
// once, 32 wide, each lane's cast reading its own entry's constants from
// arrays laid out by column (per-column maps) or by element (any other map),
// which cast_binaryK_rne_fast reads from its params. The odd macs take the
// groups. Every element's steps are the scalar loop's, in K order, whatever
// its group or lane, so the results are the scalar loop's word for word,
// under SR too (whose streams these tiles never draw: they serve RNE). When
// a group or the lane tile hands its K-block back (a NaN in its tile, or a
// format the vector tile does not take), the K-block's sums are restored and
// the scalar loop runs it.
//
// A header of its own, whose function the kernel calls only in its palette
// instantiations (behind `if constexpr (MIXED && ...)`), so the single-format
// objects are what they were.

#include "gemm_rne_tile.h"
#include <cstdint>
#include <cstring>

namespace mptorch::gemm_cpu
{
  // The kernel's tiles are at most this on a side (TI, TJ and TK).
  constexpr int64_t PALETTE_TILE = 32;

  // cast_binaryK_rne_fast's constants (BinaryKParams' fast_* fields), one per
  // lane: per column, or per element of a tile, row major. Each array is a
  // cache line longer than a tile: at 4,096 bytes apart, a lane's six
  // constants (twelve for the split mac's two formats) fell in one L1 set,
  // more than its eight ways, and filling and reading them took 1.8x the
  // tile's own time.
  constexpr int64_t LANE_STRIDE = RNE_TILE_MAX + 16;
  struct LaneParams
  {
    float clamp_hi[LANE_STRIDE];
    float magic[LANE_STRIDE];
    float split_c[LANE_STRIDE];
    float min_normal[LANE_STRIDE];
    float max_finite[LANE_STRIDE];
    float ovf[LANE_STRIDE];
  };

  inline void set_lane(LaneParams &l, int64_t i, const BinaryKParams &p)
  {
    l.clamp_hi[i] = p.fast_clamp_hi;
    l.magic[i] = p.fast_magic;
    l.split_c[i] = p.fast_split_c;
    l.min_normal[i] = p.fast_min_normal;
    l.max_finite[i] = p.fast_max_finite;
    l.ovf[i] = p.fast_ovf;
  }

  // A K-block of a packed tile with each lane's constants: by column when
  // `cols`, by element otherwise. The contract is RneTileFn's
  // (cpu/gemm_rne_tile.h): true, or false with the sums restored.
  using LaneTileFn = bool (*)(const float *a_pack, const float *b_pack, float *sums, int64_t ti,
                              int64_t tk, int64_t tj, bool cols, const LaneParams &mul,
                              const LaneParams &add);
  struct LaneTiles
  {
    LaneTileFn split = nullptr;
    LaneTileFn mul_only = nullptr;
    LaneTileFn fused = nullptr;
    const char *isa = "none";
  };
  // Chosen as rne_tiles is (MPTORCH_CPU_SIMD included), when the extension
  // loads; all null without MPTORCH_FAST_CAST.
  extern const LaneTiles lane_tiles;

  // The lane tile for kinds 1-3: LANES_RAN when it ran the K-block,
  // LANES_HANDED_BACK when it found a NaN (the sums restored), LANES_NONE when
  // it cannot run (no lane tiles, or an entry the vector tile does not take).
  enum LaneOutcome
  {
    LANES_NONE,
    LANES_RAN,
    LANES_HANDED_BACK,
  };

  template <int KIND, class Mac>
  LaneOutcome run_lane_tile(const Mac *const *slot_of, const Mac *const *macs, int count, bool cols,
                     const float *a_pack, const float *b_pack, float *sums, int64_t ti,
                     int64_t tk, int64_t tj)
  {
    const LaneTileFn fn = KIND == 1 ? lane_tiles.split : KIND == 2 ? lane_tiles.mul_only
                                                                    : lane_tiles.fused;
    if (!fn)
      return LANES_NONE;
    for (int e = 0; e < count; ++e)
    {
      if constexpr (KIND == 1)
      {
        if (!rne_tile_format(macs[e]->mul) || !rne_tile_format(macs[e]->add))
          return LANES_NONE;
      }
      else if constexpr (KIND == 2)
      {
        if (!rne_tile_format(macs[e]->mul))
          return LANES_NONE;
      }
      else
      {
        if (!rne_tile_format(macs[e]->add))
          return LANES_NONE;
      }
    }
    // The split mac's two formats' constants, or the one format's, which
    // the tile is then passed twice (as run_rne_tile passes a format).
    alignas(64) LaneParams lp[KIND == 1 ? 2 : 1];
    const int64_t lanes = cols ? tj : ti * tj;
    for (int64_t l = 0; l < lanes; ++l)
    {
      const Mac &m = *slot_of[l];
      if constexpr (KIND == 1 || KIND == 2)
        set_lane(lp[0], l, m.mul.params);
      if constexpr (KIND == 1)
        set_lane(lp[1], l, m.add.params);
      if constexpr (KIND == 3)
        set_lane(lp[0], l, m.add.params);
    }
    const LaneParams &second = lp[KIND == 1 ? 1 : 0];
    return fn(a_pack, b_pack, sums, ti, tk, tj, cols, lp[0], second) ? LANES_RAN
                                                                     : LANES_HANDED_BACK;
  }

  template <int KIND, class Mac>
  bool run_palette_tile(const Mac *const *slot_of, const float *a_pack, const float *b_pack,
                        float *sums, int64_t ti, int64_t tk, int64_t tj)
  {
    static_assert(PALETTE_TILE * PALETTE_TILE <= RNE_TILE_MAX);
    const int64_t n = ti * tj;
    // The tile's distinct Macs (a palette has at most eight), and whether its
    // slots depend on the column alone or on the row alone.
    const Mac *macs[8];
    int count = 0;
    bool per_col = true, per_row = true;
    const Mac *last = nullptr;
    for (int64_t i = 0; i < ti; ++i)
      for (int64_t j = 0; j < tj; ++j)
      {
        const Mac *m = slot_of[i * tj + j];
        per_col &= m == slot_of[j];
        per_row &= m == slot_of[i * tj];
        if (m == last)
          continue;
        last = m;
        bool seen = false;
        for (int e = 0; e < count; ++e)
          seen |= macs[e] == m;
        if (!seen)
        {
          if (count == 8)
            return false;
          macs[count++] = m;
        }
      }
    if (count == 1)
      return run_rne_tile<KIND>(*macs[0], a_pack, b_pack, sums, ti, tk, tj);
    if constexpr (KIND >= 1 && KIND <= 3)
    {
      // A per-row map keeps its rows' groups below, which run full-width
      // tiles; every other map takes the lane tile.
      if (!per_row)
      {
        const LaneOutcome o =
            run_lane_tile<KIND>(slot_of, macs, count, per_col, a_pack, b_pack, sums, ti, tk, tj);
        if (o != LANES_NONE)
          return o == LANES_RAN;
      }
    }

    alignas(64) float saved[PALETTE_TILE * PALETTE_TILE];
    std::memcpy(saved, sums, static_cast<size_t>(n) * sizeof(float));
    auto hand_back = [&]
    {
      std::memcpy(sums, saved, static_cast<size_t>(n) * sizeof(float));
      return false;
    };
    alignas(64) float sub_a[PALETTE_TILE * PALETTE_TILE];
    alignas(64) float sub_b[PALETTE_TILE * PALETTE_TILE];
    alignas(64) float sub_s[PALETTE_TILE * PALETTE_TILE];
    int64_t member[PALETTE_TILE];

    if (per_col)
    {
      for (int e = 0; e < count; ++e)
      {
        int64_t w = 0;
        for (int64_t j = 0; j < tj; ++j)
          if (slot_of[j] == macs[e])
            member[w++] = j;
        for (int64_t k = 0; k < tk; ++k)
          for (int64_t c = 0; c < w; ++c)
            sub_b[k * w + c] = b_pack[k * tj + member[c]];
        for (int64_t i = 0; i < ti; ++i)
          for (int64_t c = 0; c < w; ++c)
            sub_s[i * w + c] = sums[i * tj + member[c]];
        if (!run_rne_tile<KIND>(*macs[e], a_pack, sub_b, sub_s, ti, tk, w))
          return hand_back();
        for (int64_t i = 0; i < ti; ++i)
          for (int64_t c = 0; c < w; ++c)
            sums[i * tj + member[c]] = sub_s[i * w + c];
      }
      return true;
    }

    if (per_row)
    {
      for (int e = 0; e < count; ++e)
      {
        int64_t h = 0;
        for (int64_t i = 0; i < ti; ++i)
          if (slot_of[i * tj] == macs[e])
            member[h++] = i;
        for (int64_t r = 0; r < h; ++r)
        {
          std::memcpy(sub_a + r * tk, a_pack + member[r] * tk,
                      static_cast<size_t>(tk) * sizeof(float));
          std::memcpy(sub_s + r * tj, sums + member[r] * tj,
                      static_cast<size_t>(tj) * sizeof(float));
        }
        if (!run_rne_tile<KIND>(*macs[e], sub_a, b_pack, sub_s, h, tk, tj))
          return hand_back();
        for (int64_t r = 0; r < h; ++r)
          std::memcpy(sums + member[r] * tj, sub_s + r * tj,
                      static_cast<size_t>(tj) * sizeof(float));
      }
      return true;
    }

    // Any other map: each entry on the whole tile, from the starting sums.
    for (int e = 0; e < count; ++e)
    {
      std::memcpy(sub_s, saved, static_cast<size_t>(n) * sizeof(float));
      if (!run_rne_tile<KIND>(*macs[e], a_pack, b_pack, sub_s, ti, tk, tj))
        return hand_back();
      for (int64_t l = 0; l < n; ++l)
        if (slot_of[l] == macs[e])
          sums[l] = sub_s[l];
    }
    return true;
  }
} // namespace mptorch::gemm_cpu
