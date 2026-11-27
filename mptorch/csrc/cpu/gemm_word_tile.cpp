// The vector tiles of the CPU GEMM's binaryK steps under RNA, RU, RD, RZ and
// RO (cpu/gemm_word_tile.h).
//
// Each function runs one packed tile's K-block with loops the compiler
// vectorizes over common/cast_binaryK_word.h's branchless casts: a lane
// computes what the scalar kernel's step computes, in the same order, so the
// results are the scalar kernel's word for word, and a K-block that leaves a
// NaN in its tile is handed back to the scalar loop. The layout is
// cpu/gemm_rne_tile.cpp's: each K-step writes its products (or fused
// products) row by row into a buffer, then makes one pass over the whole
// tile per cast; a whole tile (32 x 32, a K-block of 32) runs a copy of the
// code with those sizes fixed. Two settings belong to this file alone, as to
// that one: no-trapping-math, through the pragma ahead of every include (the
// casts compare floats, and under -ftrapping-math the vectorizer will not
// select between arms that might raise different exceptions; no value
// changes), and the instruction set, by target attributes on x86-64 chosen
// once when the extension loads, NEON on arm64.

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-trapping-math")
#endif

#include "gemm_word_tile.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace mptorch::gemm_cpu
{
  namespace
  {
    // A tile of fewer than NARROW_TILE outputs goes back to the scalar loop
    // whole. Each K-step's passes would then run one lane or a few, and
    // their setup costs more than the lanes save: on one thread of a Ryzen
    // 7940HS the scalar loop was 1.2-2x faster for a single output (a sum
    // into one scalar), and these tiles first won at eight outputs
    // (dev/benchmarks/timing/cpu_narrow_gemm.py). NEON's tiles, of four
    // lanes, first win at four outputs (on an M3 Pro, 0.42-0.69x of the
    // scalar loop at four and six outputs under RNA, RU, RO and RZ).
#if defined(__aarch64__)
    constexpr int64_t NARROW_TILE = 4;
#else
    constexpr int64_t NARROW_TILE = 8;
#endif

    // One K-step after another, each element's steps in K order (see
    // cpu/gemm_rne_tile.cpp's tile_steps, whose structure this is). FTI, FTJ
    // and FTK fix the sizes at compile time; 0 means the run-time size.
    template <RoundMode RM, int KIND, int FTI, int FTJ, int FTK>
    [[gnu::always_inline]] inline void tile_steps(const float *__restrict__ a_pack,
                                                  const float *__restrict__ b_pack,
                                                  float *__restrict__ sums, int64_t ti_,
                                                  int64_t tk_, int64_t tj_, const BinaryKParams &pm,
                                                  const BinaryKParams &pa)
    {
      const int64_t ti = FTI ? FTI : ti_, tk = FTK ? FTK : tk_, tj = FTJ ? FTJ : tj_;
      const int64_t n = ti * tj;
      alignas(64) float t[RNE_TILE_MAX];
      for (int64_t k = 0; k < tk; ++k)
      {
        const float *__restrict__ b = b_pack + k * tj;
        for (int64_t i = 0; i < ti; ++i)
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
        if constexpr (KIND == 1) // SplitMac::step: add(acc + mul(a, b))
        {
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            t[l] = cast_binaryK_word<RM>(t[l], pm);
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = cast_binaryK_word<RM>(sums[l] + t[l], pa);
        }
        else if constexpr (KIND == 2) // ... over an IdentityAdder
        {
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = sums[l] + cast_binaryK_word<RM>(t[l], pm);
        }
        else // FusedMac::step: add(fma(a, b, acc))
        {
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = cast_binaryK_word<RM>(t[l], pa);
        }
      }
    }

    template <RoundMode RM, int KIND>
    [[gnu::always_inline]] inline bool tile_body(const float *__restrict__ a_pack,
                                                 const float *__restrict__ b_pack,
                                                 float *__restrict__ sums, int64_t ti, int64_t tk,
                                                 int64_t tj, const BinaryKParams &mul_p,
                                                 const BinaryKParams &add_p)
    {
      const BinaryKParams pm = mul_p, pa = add_p;
      const int64_t n = ti * tj;
      if (n < NARROW_TILE)
        return false;
      alignas(64) float saved[RNE_TILE_MAX];
      std::memcpy(saved, sums, static_cast<size_t>(n) * sizeof(float));
      if (ti == 32 && tj == 32 && tk == 32)
        tile_steps<RM, KIND, 32, 32, 32>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else
        tile_steps<RM, KIND, 0, 0, 0>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
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

// One mode's three tiles for one instruction set.
#define MPTORCH_WORD_TILES_MODE(SUFFIX, TARGET, RM, NAME)                                          \
  TARGET bool split_##NAME##_##SUFFIX(const float *a, const float *b, float *s, int64_t ti,        \
                                      int64_t tk, int64_t tj, const BinaryKParams &pm,             \
                                      const BinaryKParams &pa)                                     \
  {                                                                                                \
    return tile_body<RM, 1>(a, b, s, ti, tk, tj, pm, pa);                                          \
  }                                                                                                \
  TARGET bool mul_only_##NAME##_##SUFFIX(const float *a, const float *b, float *s, int64_t ti,     \
                                         int64_t tk, int64_t tj, const BinaryKParams &pm,          \
                                         const BinaryKParams &pa)                                  \
  {                                                                                                \
    return tile_body<RM, 2>(a, b, s, ti, tk, tj, pm, pa);                                          \
  }                                                                                                \
  TARGET bool fused_##NAME##_##SUFFIX(const float *a, const float *b, float *s, int64_t ti,        \
                                      int64_t tk, int64_t tj, const BinaryKParams &pm,             \
                                      const BinaryKParams &pa)                                     \
  {                                                                                                \
    return tile_body<RM, 3>(a, b, s, ti, tk, tj, pm, pa);                                          \
  }

#define MPTORCH_WORD_TILES(SUFFIX, TARGET)                                                         \
  MPTORCH_WORD_TILES_MODE(SUFFIX, TARGET, RoundMode::RNA, rna)                                     \
  MPTORCH_WORD_TILES_MODE(SUFFIX, TARGET, RoundMode::RU, ru)                                       \
  MPTORCH_WORD_TILES_MODE(SUFFIX, TARGET, RoundMode::RD, rd)                                       \
  MPTORCH_WORD_TILES_MODE(SUFFIX, TARGET, RoundMode::RZ, rz)                                       \
  MPTORCH_WORD_TILES_MODE(SUFFIX, TARGET, RoundMode::RO, ro)                                       \
  WordTiles table_##SUFFIX()                                                                       \
  {                                                                                                \
    WordTiles t;                                                                                   \
    t.split[0] = split_rna_##SUFFIX, t.mul_only[0] = mul_only_rna_##SUFFIX;                        \
    t.fused[0] = fused_rna_##SUFFIX;                                                               \
    t.split[1] = split_ru_##SUFFIX, t.mul_only[1] = mul_only_ru_##SUFFIX;                          \
    t.fused[1] = fused_ru_##SUFFIX;                                                                \
    t.split[2] = split_rd_##SUFFIX, t.mul_only[2] = mul_only_rd_##SUFFIX;                          \
    t.fused[2] = fused_rd_##SUFFIX;                                                                \
    t.split[3] = split_rz_##SUFFIX, t.mul_only[3] = mul_only_rz_##SUFFIX;                          \
    t.fused[3] = fused_rz_##SUFFIX;                                                                \
    t.split[4] = split_ro_##SUFFIX, t.mul_only[4] = mul_only_ro_##SUFFIX;                          \
    t.fused[4] = fused_ro_##SUFFIX;                                                                \
    t.isa = #SUFFIX;                                                                               \
    return t;                                                                                      \
  }

#if defined(__x86_64__)
    MPTORCH_WORD_TILES(avx512, __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,fma"))))
    MPTORCH_WORD_TILES(avx2, __attribute__((target("avx2,fma"))))
#elif defined(__aarch64__)
    MPTORCH_WORD_TILES(neon, )
#endif
#undef MPTORCH_WORD_TILES
#undef MPTORCH_WORD_TILES_MODE

    WordTiles choose_word_tiles()
    {
      // As choose_rne_tiles (cpu/gemm_rne_tile.cpp): MPTORCH_CPU_SIMD=0 keeps
      // the scalar loop, =avx2 stops below AVX-512, and anything else takes
      // the widest instruction set the CPU has.
      const char *env = std::getenv("MPTORCH_CPU_SIMD");
      if (env && std::strcmp(env, "0") == 0)
        return WordTiles{};
#if defined(__x86_64__)
      const bool avx2_only = env && std::strcmp(env, "avx2") == 0;
      __builtin_cpu_init();
      if (!avx2_only && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
          __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512bw"))
        return table_avx512();
      if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return table_avx2();
      return WordTiles{};
#elif defined(__aarch64__)
      return table_neon();
#else
      return WordTiles{};
#endif
    }
  } // namespace

  const WordTiles word_tiles = choose_word_tiles();
} // namespace mptorch::gemm_cpu
