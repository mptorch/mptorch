// The vector tiles of the CPU GEMM's binaryK steps under RNE
// (cpu/gemm_rne_tile.h).
//
// Each function runs one packed tile's K-block, with loops the compiler
// vectorizes over the shipped cast_binaryK_rne_fast: a lane computes what the
// scalar kernel's step computes, in the same order, so the results are the
// scalar kernel's word for word (a K-block that leaves a NaN in its tile is
// handed back to the scalar loop, for the reason cpu/gemm_rne_tile.h gives).
// Two settings belong to this file alone:
//
//   * no-trapping-math, through the pragma ahead of every include, so that
//     the cast is compiled under it too. Under GCC's default -ftrapping-math
//     the vectorizer may not compute both arms of the fast path (the
//     subnormal one and the Veltkamp one) and select between them, since
//     either arm may raise a floating-point exception that the scalar branch
//     would not, and an optimize attribute on the function alone stopped GCC
//     from inlining the cast. The option changes only the exception flags,
//     which nothing reads; no value changes. -ffp-contract=off still holds,
//     so no multiply-add is fused that the scalar code does not fuse. Clang
//     defaults to no-trapping-math. The option must stay out of setup.py's
//     host flags, where it would change every other object.
//   * the instruction set, through target attributes on x86-64 (the cast's
//     default-target definition is inlined into them), chosen once when the
//     extension loads; on arm64, NEON is the baseline.
//
// Each K-step is a pass that writes the products (or the fused products) row
// by row into a buffer, then one pass over the whole tile per cast; a whole
// tile (32 x 32, with a K-block of 32) runs a copy of the code with those
// sizes fixed. In one loop per row, the split tile's two casts would form one
// long dependency chain per lane (the product's cast feeding the sum's),
// which the CPU's out-of-order window cannot overlap across enough lanes; in
// its own pass each cast is a short chain, and its lanes are independent.
//
// NEON keeps the row loop (tile_rows) for the product-only and fused tiles:
// with a cast chain that short, the passes' store and reload per lane cost
// the M3 Pro's tiles 2-4%, while its split tile gained 1.08-1.17x.
//
// Measured on 2026-11-14 (dev/benchmarks/timing/cpu_gemm_ab.cpp, Ryzen 9
// 7940HS at 4.8 GHz, against the row loop; ns per mac at 256^3 on one
// thread): split 8p4/16p11 0.48 -> 0.39 with AVX-512 and 0.70 -> 0.53 with
// AVX2, and 1.37-1.51x faster at 100^3 and 77^3, whose edge tiles are
// ragged (smaller than 32 x 32); 1.12-1.23x at 1024^3 on sixteen threads.
// The fused and product-only tiles: 1.01-1.07x on whole tiles, 1.09-1.18x
// on ragged ones with AVX-512 (1.04-1.10x with AVX2). The scalar kernel's
// split step took 7.5 ns. On an M3 Pro (NEON, one thread, 256^3): split
// 0.70 -> 0.65; at 1024^3 on eleven threads, 1.15-1.17x.

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-trapping-math")
#endif

#include "gemm_rne_tile.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace mptorch::gemm_cpu
{
#if defined(MPTORCH_FAST_CAST)
  namespace
  {
    // One K-step after another, each element's steps in K order: the
    // products of a step, row by row, into t (the fused tile's fma also reads
    // the sum), then each cast as one pass over every lane of the tile. The
    // split tile rounds the product in t, then the sum. FTI, FTJ and FTK fix
    // the sizes at compile time; 0 means the run-time size.
    template <int KIND, int FTI, int FTJ, int FTK>
    [[gnu::always_inline]] inline void tile_steps(const float *__restrict__ a_pack,
                                                  const float *__restrict__ b_pack,
                                                  float *__restrict__ sums, int64_t ti_,
                                                  int64_t tk_, int64_t tj_,
                                                  const BinaryKParams &pm,
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
            t[l] = cast_binaryK_rne_fast(t[l], pm);
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = cast_binaryK_rne_fast(sums[l] + t[l], pa);
        }
        else if constexpr (KIND == 2) // ... over an IdentityAdder
        {
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = sums[l] + cast_binaryK_rne_fast(t[l], pm);
        }
        else // FusedMac::step: add(fma(a, b, acc))
        {
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = cast_binaryK_rne_fast(t[l], pa);
        }
      }
    }

    // The row loop, which NEON's product-only and fused tiles run (see the
    // top of the file): K outside the rows, and one loop per row that does
    // all of the step's operations.
    template <int KIND>
    [[gnu::always_inline]] inline void tile_rows(const float *__restrict__ a_pack,
                                                 const float *__restrict__ b_pack,
                                                 float *__restrict__ sums, int64_t ti, int64_t tk,
                                                 int64_t tj, const BinaryKParams &pm,
                                                 const BinaryKParams &pa)
    {
      for (int64_t k = 0; k < tk; ++k)
        for (int64_t i = 0; i < ti; ++i)
        {
          const float a = a_pack[i * tk + k];
          const float *__restrict__ b = b_pack + k * tj;
          float *__restrict__ s = sums + i * tj;
#pragma omp simd
          for (int64_t j = 0; j < tj; ++j)
          {
            if constexpr (KIND == 1)
              s[j] = cast_binaryK_rne_fast(s[j] + cast_binaryK_rne_fast(a * b[j], pm), pa);
            else if constexpr (KIND == 2)
              s[j] = s[j] + cast_binaryK_rne_fast(a * b[j], pm);
            else
              s[j] = cast_binaryK_rne_fast(std::fma(a, b[j], s[j]), pa);
          }
        }
    }

    // ROWS: the row loop in place of the passes.
    template <int KIND, bool ROWS = false>
    [[gnu::always_inline]] inline bool tile_body(const float *__restrict__ a_pack,
                                                 const float *__restrict__ b_pack,
                                                 float *__restrict__ sums, int64_t ti, int64_t tk,
                                                 int64_t tj, const BinaryKParams &mul_p,
                                                 const BinaryKParams &add_p)
    {
      const BinaryKParams pm = mul_p, pa = add_p;
      const int64_t n = ti * tj;
      alignas(64) float saved[RNE_TILE_MAX];
      std::memcpy(saved, sums, static_cast<size_t>(n) * sizeof(float));
      if constexpr (ROWS)
        tile_rows<KIND>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else if (ti == 32 && tj == 32 && tk == 32)
        tile_steps<KIND, 32, 32, 32>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else
        tile_steps<KIND, 0, 0, 0>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
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

// ROWS: whether the product-only and fused tiles run the row loop.
#define MPTORCH_RNE_TILES(SUFFIX, TARGET, ROWS)                                                    \
  TARGET bool split_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,     \
                             int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)         \
  {                                                                                                \
    return tile_body<1>(a, b, s, ti, tk, tj, pm, pa);                                              \
  }                                                                                                \
  TARGET bool mul_only_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,  \
                                int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)      \
  {                                                                                                \
    return tile_body<2, ROWS>(a, b, s, ti, tk, tj, pm, pa);                                        \
  }                                                                                                \
  TARGET bool fused_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,     \
                             int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)         \
  {                                                                                                \
    return tile_body<3, ROWS>(a, b, s, ti, tk, tj, pm, pa);                                        \
  }

#if defined(__x86_64__)
    MPTORCH_RNE_TILES(avx512, __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,fma"))),
                      false)
    MPTORCH_RNE_TILES(avx2, __attribute__((target("avx2,fma"))), false)
#elif defined(__aarch64__)
    MPTORCH_RNE_TILES(neon, , true)
#endif
#undef MPTORCH_RNE_TILES

    RneTiles choose_rne_tiles()
    {
      RneTiles t;
      // MPTORCH_CPU_SIMD=0 keeps the scalar loop, =avx2 stops below AVX-512
      // (so that a machine with AVX-512 can test the AVX2 tiles too), and any
      // other value, or none, takes the widest instruction set the CPU has.
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

  const RneTiles rne_tiles = choose_rne_tiles();
#else
  // Without the fast cast there is no cast_binaryK_rne_fast to vectorize.
  const RneTiles rne_tiles{};
#endif
} // namespace mptorch::gemm_cpu
