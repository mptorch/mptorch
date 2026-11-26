// The lane tiles of the CPU palette GEMMs (cpu/gemm_palette_tile.h): a
// packed tile's K-block of a binaryK palette GEMM under RNE, every element on
// its own entry, in one pass over the tile.
//
// Each lane computes what the scalar kernel's step computes for its element,
// in the same order: the product (or the fused product), then
// cast_binaryK_rne_fast, here spelled with each lane's constants read from
// LaneParams instead of a BinaryKParams (cast_lane, the same operations on
// the same values). So the results are the scalar loop's word for word, and
// a K-block that leaves a NaN in its tile is handed back with its sums
// restored, for the reason cpu/gemm_rne_tile.h gives.
//
// The two settings of cpu/gemm_rne_tile.cpp hold here, for the same reasons:
// no-trapping-math through the pragma ahead of every include (it changes
// only the exception flags, and -ffp-contract=off still holds), and the
// instruction set through target attributes, chosen once when the extension
// loads. Its layout too: each K-step is a pass that writes the products row
// by row into a buffer, then one pass per cast, and a whole tile (32 x 32,
// a K-block of 32) runs a copy with those sizes fixed.

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-trapping-math")
#endif

#include "gemm_palette_tile.h"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace mptorch::gemm_cpu
{
#if defined(MPTORCH_FAST_CAST)
  namespace
  {
    // cast_binaryK_rne_fast (common/cast_binaryK.h) with lane l's constants.
    [[gnu::always_inline]] inline float cast_lane(float origin_float, const LaneParams &p,
                                                  int64_t l)
    {
      using F = FloatTraits<float>;
      float ax = fabsf(origin_float);
      float xc = copysignf(F::fmin_nonneg(ax, p.clamp_hi[l]), origin_float);
      float sub = F::rn_sub(F::rn_add(xc, p.magic[l]), p.magic[l]);
      float t = F::rn_mul(p.split_c[l], xc);
      float nrm = F::rn_sub(t, F::rn_sub(t, xc));
      float y = (ax < p.min_normal[l]) ? sub : nrm;
      if (fabsf(y) > p.max_finite[l])
        y = copysignf(p.ovf[l], origin_float);
      return (ax <= std::bit_cast<float>(F::INF_BITS)) ? y : origin_float;
    }

    // The products of K-step k for rows [i0, i1), into t (the fused mac's
    // fma also reads the sum).
    template <int KIND>
    [[gnu::always_inline]] inline void lane_products(const float *__restrict__ a_pack,
                                                     const float *__restrict__ b,
                                                     const float *__restrict__ sums,
                                                     float *__restrict__ t, int64_t k, int64_t i0,
                                                     int64_t i1, int64_t tk, int64_t tj)
    {
      for (int64_t i = i0; i < i1; ++i)
      {
        const float a = a_pack[i * tk + k];
        const float *__restrict__ s = sums + i * tj;
        float *__restrict__ r = t + i * tj;
#pragma omp simd
        for (int64_t j = 0; j < tj; ++j)
        {
          if constexpr (KIND == 3) // FusedMac::step's fma
            r[j] = std::fma(a, b[j], s[j]);
          else
            r[j] = a * b[j];
        }
      }
    }

    // The casts of one K-step over lanes [l0, l1) of the tile, lane l's
    // constants at index l (or at its column, l - row start, for COLS).
    template <int KIND>
    [[gnu::always_inline]] inline void lane_casts(float *__restrict__ sums, float *__restrict__ t,
                                                  int64_t l0, int64_t l1, int64_t shift,
                                                  const LaneParams &pm, const LaneParams &pa)
    {
      if constexpr (KIND == 1) // SplitMac::step: add(acc + mul(a, b))
      {
#pragma omp simd
        for (int64_t l = l0; l < l1; ++l)
          t[l] = cast_lane(t[l], pm, l - shift);
#pragma omp simd
        for (int64_t l = l0; l < l1; ++l)
          sums[l] = cast_lane(sums[l] + t[l], pa, l - shift);
      }
      else if constexpr (KIND == 2) // ... over an IdentityAdder
      {
#pragma omp simd
        for (int64_t l = l0; l < l1; ++l)
          sums[l] = sums[l] + cast_lane(t[l], pm, l - shift);
      }
      else // FusedMac::step: add(fma(a, b, acc))
      {
#pragma omp simd
        for (int64_t l = l0; l < l1; ++l)
          sums[l] = cast_lane(t[l], pa, l - shift);
      }
    }

    // Rows a band of the per-element lanes holds: its constants (12 KB for
    // the split mac's two casts) stay in the L1 cache through the K-block,
    // where the whole tile's (48 KB) did not, and its 8 x 32 lanes are enough
    // independent casts per pass for the out-of-order core.
    constexpr int64_t LANE_BAND = 8;

    // One K-step after another, each element's steps in K order. Per column
    // (COLS): the products of a step into t, then each cast as one pass over
    // each row, the constants at the column. Per element: the same, one band
    // of LANE_BAND rows at a time. KIND is gemm_rne_tile.h's: 1 split, 2
    // product only, 3 fused. FTI, FTJ and FTK fix the sizes at compile time;
    // 0 means the run-time size.
    template <int KIND, bool COLS, int FTI, int FTJ, int FTK>
    [[gnu::always_inline]] inline void lane_steps(const float *__restrict__ a_pack,
                                                  const float *__restrict__ b_pack,
                                                  float *__restrict__ sums, int64_t ti_,
                                                  int64_t tk_, int64_t tj_, const LaneParams &pm,
                                                  const LaneParams &pa)
    {
      const int64_t ti = FTI ? FTI : ti_, tk = FTK ? FTK : tk_, tj = FTJ ? FTJ : tj_;
      alignas(64) float t[RNE_TILE_MAX];
      if constexpr (COLS)
      {
        for (int64_t k = 0; k < tk; ++k)
        {
          lane_products<KIND>(a_pack, b_pack + k * tj, sums, t, k, 0, ti, tk, tj);
          for (int64_t i = 0; i < ti; ++i)
            lane_casts<KIND>(sums, t, i * tj, (i + 1) * tj, i * tj, pm, pa);
        }
      }
      else
      {
        for (int64_t i0 = 0; i0 < ti; i0 += LANE_BAND)
        {
          const int64_t i1 = i0 + LANE_BAND < ti ? i0 + LANE_BAND : ti;
          for (int64_t k = 0; k < tk; ++k)
          {
            lane_products<KIND>(a_pack, b_pack + k * tj, sums, t, k, i0, i1, tk, tj);
            lane_casts<KIND>(sums, t, i0 * tj, i1 * tj, 0, pm, pa);
          }
        }
      }
    }

    template <int KIND, bool COLS>
    [[gnu::always_inline]] inline void lane_tile(const float *a_pack, const float *b_pack,
                                                 float *sums, int64_t ti, int64_t tk, int64_t tj,
                                                 const LaneParams &pm, const LaneParams &pa)
    {
      if (ti == 32 && tj == 32 && tk == 32)
        lane_steps<KIND, COLS, 32, 32, 32>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else
        lane_steps<KIND, COLS, 0, 0, 0>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
    }

    template <int KIND>
    [[gnu::always_inline]] inline bool lane_body(const float *a_pack, const float *b_pack,
                                                 float *sums, int64_t ti, int64_t tk, int64_t tj,
                                                 bool cols, const LaneParams &pm,
                                                 const LaneParams &pa)
    {
      const int64_t n = ti * tj;
      alignas(64) float saved[RNE_TILE_MAX];
      std::memcpy(saved, sums, static_cast<size_t>(n) * sizeof(float));
      if (cols)
        lane_tile<KIND, true>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else
        lane_tile<KIND, false>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      // A NaN anywhere: a sum of two NaNs may have returned the other NaN
      // (cpu/gemm_rne_tile.h), so the K-block goes back to the scalar loop.
      uint32_t nan = 0;
#pragma omp simd reduction(| : nan)
      for (int64_t l = 0; l < n; ++l)
      {
        uint32_t w;
        std::memcpy(&w, &sums[l], sizeof w);
        nan |= static_cast<uint32_t>((w & 0x7FFFFFFFu) > 0x7F800000u);
      }
      if (nan)
      {
        std::memcpy(sums, saved, static_cast<size_t>(n) * sizeof(float));
        return false;
      }
      return true;
    }

#define MPTORCH_LANE_TILES(SUFFIX, TARGET)                                                         \
  TARGET bool split_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,     \
                             int64_t tj, bool cols, const LaneParams &pm, const LaneParams &pa)    \
  {                                                                                                \
    return lane_body<1>(a, b, s, ti, tk, tj, cols, pm, pa);                                        \
  }                                                                                                \
  TARGET bool mul_only_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,  \
                                int64_t tj, bool cols, const LaneParams &pm, const LaneParams &pa) \
  {                                                                                                \
    return lane_body<2>(a, b, s, ti, tk, tj, cols, pm, pa);                                        \
  }                                                                                                \
  TARGET bool fused_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,     \
                             int64_t tj, bool cols, const LaneParams &pm, const LaneParams &pa)    \
  {                                                                                                \
    return lane_body<3>(a, b, s, ti, tk, tj, cols, pm, pa);                                        \
  }

#if defined(__x86_64__)
    MPTORCH_LANE_TILES(avx512, __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,fma"))))
    MPTORCH_LANE_TILES(avx2, __attribute__((target("avx2,fma"))))
#elif defined(__aarch64__)
    MPTORCH_LANE_TILES(neon, )
#endif
#undef MPTORCH_LANE_TILES

    LaneTiles choose_lane_tiles()
    {
      LaneTiles t;
      // MPTORCH_CPU_SIMD as cpu/gemm_rne_tile.cpp reads it.
      const char *env = std::getenv("MPTORCH_CPU_SIMD");
      if (env && std::strcmp(env, "0") == 0)
        return t;
#if defined(__x86_64__)
      const bool avx2_only = env && std::strcmp(env, "avx2") == 0;
      __builtin_cpu_init();
      if (!avx2_only && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
          __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512bw"))
        t = {split_avx512, mul_only_avx512, fused_avx512, "avx512"};
      else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        t = {split_avx2, mul_only_avx2, fused_avx2, "avx2"};
#elif defined(__aarch64__)
      t = {split_neon, mul_only_neon, fused_neon, "neon"};
#endif
      return t;
    }
  } // namespace

  const LaneTiles lane_tiles = choose_lane_tiles();
#else
  // Without the fast cast there is no cast_binaryK_rne_fast to vectorize.
  const LaneTiles lane_tiles{};
#endif
} // namespace mptorch::gemm_cpu
