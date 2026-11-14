// The vectorized binaryK RNE tiles of the CPU GEMM with the carrier step
// rounded to odd (cpu/gemm_odd_tile.h; carrier_rounding=RoundMode.RO).
//
// gemm_rne_tile.cpp's tiles with the odd macs' steps: the same loop order,
// the same NaN hand-back, the same no-trapping-math pragma ahead of every
// include (so the cast and the odd primitives are compiled under it) and the
// same ISA choice, made once when the extension loads. Every lane computes
// what the scalar kernel's step computes, in the same order, so the results
// are the scalar kernel's word for word.
//
// Two bodies:
//
//   * the portable one, for AVX2 and NEON: the scalar step verbatim,
//     carrier_odd.h's host binary32 primitives (the exact product and a 2Sum
//     in binary64, then the odd fix on the word) and cast_binaryK_rne_fast,
//     vectorized by `omp simd`;
//   * with AVX-512, CUDA's spelling of the primitives: the step rounded down
//     and up by embedded rounding and the one whose word is odd kept, then the
//     cast, each a pass over the whole tile per K-step. For finite operands
//     that is the same value as the 2Sum's (round to odd is the odd one of the
//     two directed roundings, and both are exact when the step is), which the
//     CPU against CUDA tests of the odd ops already hold word for word; a NaN
//     is handed back either way, and infinite operands give IEEE's result in
//     both.
//
// Measured (Ryzen 9 7940HS, one thread, 256^3, through the ops; odd against
// the default op on its own tile): AVX-512 split 8p4/16p11 0.97-0.99x,
// product only 1.05-1.08x, fused 16p11 1.10-1.20x; the portable body there
// was 3.0-3.9x, and with AVX2 it is 3.9-4.8x. Before these tiles the odd op
// ran the scalar loop, 15-22x the default tile. The AVX-512 body's passes
// over the whole tile were what brought it from 1.24-2.21x, when each pass
// was a row's: the cast loops of a 32-lane row are mostly prologue and tail.

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-trapping-math")
#endif

#include "gemm_odd_tile.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace mptorch::gemm_cpu
{
#if defined(MPTORCH_FAST_CAST)
  namespace
  {
    namespace odd = mptorch::gemm::odd;

    // A NaN anywhere in the tile hands the block back (gemm_rne_tile.h).
    [[gnu::always_inline]] inline bool keep_or_restore(float *__restrict__ sums,
                                                       const float *__restrict__ saved, int64_t n)
    {
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

    // K outside the rows, as in gemm_rne_tile.cpp.
    template <int KIND>
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
      for (int64_t k = 0; k < tk; ++k)
        for (int64_t i = 0; i < ti; ++i)
        {
          const float a = a_pack[i * tk + k];
          const float *__restrict__ b = b_pack + k * tj;
          float *__restrict__ s = sums + i * tj;
#pragma omp simd
          for (int64_t j = 0; j < tj; ++j)
          {
            if constexpr (KIND == 4) // SplitMacOdd::step
              s[j] = cast_binaryK_rne_fast(
                  odd::add(s[j], cast_binaryK_rne_fast(odd::mul(a, b[j]), pm)), pa);
            else if constexpr (KIND == 5) // ... over an IdentityAdder
              s[j] = s[j] + cast_binaryK_rne_fast(odd::mul(a, b[j]), pm);
            else // FusedMacOdd::step
              s[j] = cast_binaryK_rne_fast(odd::fma(a, b[j], s[j]), pa);
          }
        }
      return keep_or_restore(sums, saved, n);
    }

#define MPTORCH_ODD_TILES(SUFFIX, TARGET, BODY)                                                    \
  TARGET bool split_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,     \
                             int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)         \
  {                                                                                                \
    return BODY<4>(a, b, s, ti, tk, tj, pm, pa);                                                   \
  }                                                                                                \
  TARGET bool mul_only_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,  \
                                int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)      \
  {                                                                                                \
    return BODY<5>(a, b, s, ti, tk, tj, pm, pa);                                                   \
  }                                                                                                \
  TARGET bool fused_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,     \
                             int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)         \
  {                                                                                                \
    return BODY<6>(a, b, s, ti, tk, tj, pm, pa);                                                   \
  }

#if defined(__x86_64__)
#define MPTORCH_AVX512 __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,fma")))

    // The directed pair and the select on RD's word parity, sixteen lanes at
    // a time, the row's tail masked (tj is at most 32).
    enum class OddOp
    {
      FMA,
      ADD,
      MUL
    };

    template <OddOp OP>
    [[gnu::always_inline]] MPTORCH_AVX512 inline __m512 odd16(__m512 x, __m512 y, __m512 z)
    {
      constexpr int DOWN = _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
      constexpr int UP = _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
      __m512 rd, ru;
      if constexpr (OP == OddOp::FMA)
      {
        rd = _mm512_fmadd_round_ps(x, y, z, DOWN);
        ru = _mm512_fmadd_round_ps(x, y, z, UP);
      }
      else if constexpr (OP == OddOp::ADD)
      {
        rd = _mm512_add_round_ps(x, y, DOWN);
        ru = _mm512_add_round_ps(x, y, UP);
      }
      else
      {
        rd = _mm512_mul_round_ps(x, y, DOWN);
        ru = _mm512_mul_round_ps(x, y, UP);
      }
      const __mmask16 odd_word =
          _mm512_test_epi32_mask(_mm512_castps_si512(rd), _mm512_set1_epi32(1));
      return _mm512_mask_blend_ps(odd_word, ru, rd);
    }

    // The tile's n = ti * tj lanes from l on, sixteen at a time, the tail masked.
    [[gnu::always_inline]] MPTORCH_AVX512 inline __mmask16 lanes(int64_t n, int64_t l)
    {
      return n - l >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << (n - l)) - 1);
    }

    // One K-step at a time over the whole tile: the directed passes, then the
    // casts, each a loop over every lane, so each element still takes its
    // steps in K order and a pass runs over up to RNE_TILE_MAX lanes rather
    // than a row's.
    template <int KIND>
    [[gnu::always_inline]] MPTORCH_AVX512 inline bool
    tile_body_directed(const float *__restrict__ a_pack, const float *__restrict__ b_pack,
                       float *__restrict__ sums, int64_t ti, int64_t tk, int64_t tj,
                       const BinaryKParams &mul_p, const BinaryKParams &add_p)
    {
      const BinaryKParams pm = mul_p, pa = add_p;
      const int64_t n = ti * tj;
      alignas(64) float saved[RNE_TILE_MAX];
      alignas(64) float t[RNE_TILE_MAX];
      std::memcpy(saved, sums, static_cast<size_t>(n) * sizeof(float));
      for (int64_t k = 0; k < tk; ++k)
      {
        const float *__restrict__ b = b_pack + k * tj;
        for (int64_t i = 0; i < ti; ++i)
        {
          const __m512 va = _mm512_set1_ps(a_pack[i * tk + k]);
          for (int64_t j = 0; j < tj; j += 16)
          {
            const __mmask16 m = lanes(tj, j);
            const __m512 vb = _mm512_maskz_loadu_ps(m, b + j);
            float *row = t + i * tj + j;
            if constexpr (KIND == 6)
              _mm512_mask_storeu_ps(
                  row, m, odd16<OddOp::FMA>(va, vb, _mm512_maskz_loadu_ps(m, sums + i * tj + j)));
            else
              _mm512_mask_storeu_ps(row, m, odd16<OddOp::MUL>(va, vb, vb));
          }
        }
        if constexpr (KIND == 6)
        {
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = cast_binaryK_rne_fast(t[l], pa);
        }
        else if constexpr (KIND == 5)
        {
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = sums[l] + cast_binaryK_rne_fast(t[l], pm);
        }
        else
        {
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            t[l] = cast_binaryK_rne_fast(t[l], pm);
          for (int64_t l = 0; l < n; l += 16)
          {
            const __mmask16 m = lanes(n, l);
            const __m512 vs = _mm512_maskz_loadu_ps(m, sums + l);
            const __m512 vt = _mm512_maskz_loadu_ps(m, t + l);
            _mm512_mask_storeu_ps(t + l, m, odd16<OddOp::ADD>(vs, vt, vt));
          }
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = cast_binaryK_rne_fast(t[l], pa);
        }
      }
      return keep_or_restore(sums, saved, n);
    }

    MPTORCH_ODD_TILES(avx512, MPTORCH_AVX512, tile_body_directed)
    MPTORCH_ODD_TILES(avx2, __attribute__((target("avx2,fma"))), tile_body)
#undef MPTORCH_AVX512
#elif defined(__aarch64__)
    MPTORCH_ODD_TILES(neon, , tile_body)
#endif
#undef MPTORCH_ODD_TILES

    // gemm_rne_tile.cpp's choice, restated rather than shared, so that file
    // and its object are what they were.
    OddTiles choose_odd_tiles()
    {
      OddTiles t;
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

  const OddTiles odd_tiles = choose_odd_tiles();
#else
  const OddTiles odd_tiles{};
#endif
} // namespace mptorch::gemm_cpu
