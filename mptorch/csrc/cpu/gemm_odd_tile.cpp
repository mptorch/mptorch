// The vectorized binaryK RNE tiles of the CPU GEMM with the carrier step
// rounded to odd (cpu/gemm_odd_tile.h; carrier_rounding=RoundMode.RO).
//
// gemm_rne_tile.cpp's tiles with the odd macs' steps: the same passes (a
// K-step's products row by row into a buffer, then each cast and each odd
// sum as a pass over the whole tile, and a copy with the sizes fixed for a
// whole tile), the same NaN hand-back, the same no-trapping-math pragma ahead
// of every include (so the cast and the odd primitives are compiled under it)
// and the same ISA choice, made once when the extension loads. Every lane
// computes what the scalar kernel's step computes, in the same order, so the
// results are the scalar kernel's word for word.
//
// The odd primitives (carrier_odd.h's) are spelled as CUDA spells them, the
// step rounded down and up and the one whose word is odd kept: round to odd
// is the odd one of the two directed roundings, and both are the exact value
// when the step is exact, which the CPU against CUDA tests of the odd ops hold
// word for word; a NaN is handed back either way, and infinite operands give
// IEEE's result. Each is the same value as the scalar primitive for every
// input (cpu_odd_tile_sweep.cpp). Two ways to round:
//
//   * AVX-512: embedded rounding, the pair in one pass.
//   * AVX2 and NEON: the FP unit's rounding mode (MXCSR, FPCR), set once per
//     pass: a pass rounded down into the buffer, a pass rounded up that keeps
//     the odd word, then round to nearest again for the casts. Each switch is
//     fenced by compiler barriers, and each pass reads its operands from
//     memory and writes its result to memory, so no operation can move into
//     another pass's mode (the compiler is not told about the mode: it folds
//     nothing here, since nothing in these passes is a constant).
//
// Measured (cpu_gemm_ab.cpp, ns per mac, one thread at 256^3; "->" is from
// the row loops these replaced: on AVX-512 the same directed passes without
// the fixed sizes, on AVX2 and NEON carrier_odd.h's binary64 primitives
// vectorized row by row, half binary32's width and a long chain per lane):
//
//   * Ryzen 9 7940HS at 4.8 GHz: AVX-512 split 8p4/16p11 0.46 -> 0.43, fused
//     16p11 0.26 -> 0.22, product only 0.24 -> 0.24, against the default
//     tile's 0.39, 0.21 and 0.22; AVX2 split 2.99 -> 0.64, fused 1.32 -> 0.31,
//     product only 1.05 -> 0.31, against 0.53, 0.27 and 0.28.
//   * M3 Pro: split 2.36 -> 0.90, fused 1.10 -> 0.45, product only
//     0.85 -> 0.46, against 0.65, 0.32 and 0.34; 1.8-3.0x faster at 1024^3 on
//     eleven threads.
//
// The binary32 error-free transforms (an fma for the product's error, 2Sum,
// Boldo and Muller's ErrFma to its error's sign, with a binary64 fallback)
// were measured as NEON's body and AVX2's: 1.3-2.6x slower than the mode on
// both, the fused tile no faster than the binary64 row loop on NEON (e6ffed9
// has them).

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

    // ---- the directed pair under the FP unit's rounding mode -------------

    [[gnu::always_inline]] inline float odd_of(float rd, float ru)
    {
      return (odd::word(rd) & 1u) ? rd : ru;
    }

    // Mode: word_t, get(), set(w) (fenced) and down(w) / up(w), the word
    // with its rounding field set toward -inf / +inf.
    template <class Mode>
    struct Directed
    {
      typename Mode::word_t nearest = Mode::get();

      template <int KIND>
      [[gnu::always_inline]] inline void products(const float *__restrict__ a, int64_t tk,
                                                  const float *__restrict__ b,
                                                  const float *__restrict__ sums,
                                                  float *__restrict__ r, int64_t ti,
                                                  int64_t tj) const
      {
        Mode::set(Mode::down(nearest));
        for (int64_t i = 0; i < ti; ++i)
        {
          const float ai = a[i * tk];
#pragma omp simd
          for (int64_t j = 0; j < tj; ++j)
          {
            if constexpr (KIND == 6)
              r[i * tj + j] = std::fma(ai, b[j], sums[i * tj + j]);
            else
              r[i * tj + j] = ai * b[j];
          }
        }
        Mode::set(Mode::up(nearest));
        for (int64_t i = 0; i < ti; ++i)
        {
          const float ai = a[i * tk];
#pragma omp simd
          for (int64_t j = 0; j < tj; ++j)
          {
            if constexpr (KIND == 6)
              r[i * tj + j] = odd_of(r[i * tj + j], std::fma(ai, b[j], sums[i * tj + j]));
            else
              r[i * tj + j] = odd_of(r[i * tj + j], ai * b[j]);
          }
        }
        Mode::set(nearest);
      }

      [[gnu::always_inline]] inline void odd_sums(const float *__restrict__ sums,
                                                  const float *__restrict__ t,
                                                  float *__restrict__ u, int64_t n) const
      {
        Mode::set(Mode::down(nearest));
#pragma omp simd
        for (int64_t l = 0; l < n; ++l)
          u[l] = sums[l] + t[l];
        Mode::set(Mode::up(nearest));
#pragma omp simd
        for (int64_t l = 0; l < n; ++l)
          u[l] = odd_of(u[l], sums[l] + t[l]);
        Mode::set(nearest);
      }
    };

#if defined(__x86_64__)
    struct Mxcsr // RC, bits 13-14: 00 nearest, 01 down, 10 up
    {
      using word_t = unsigned;
      [[gnu::always_inline]] static inline word_t get() { return _mm_getcsr(); }
      [[gnu::always_inline]] static inline void set(word_t w)
      {
        asm volatile("" ::: "memory");
        _mm_setcsr(w);
        asm volatile("" ::: "memory");
      }
      static constexpr word_t down(word_t w) { return (w & ~0x6000u) | 0x2000u; }
      static constexpr word_t up(word_t w) { return (w & ~0x6000u) | 0x4000u; }
    };
#elif defined(__aarch64__)
    struct Fpcr // RMode, bits 22-23: 00 nearest, 01 up (RP), 10 down (RM)
    {
      using word_t = uint64_t;
      [[gnu::always_inline]] static inline word_t get()
      {
        word_t w;
        asm volatile("mrs %0, fpcr" : "=r"(w)::"memory");
        return w;
      }
      [[gnu::always_inline]] static inline void set(word_t w)
      {
        asm volatile("msr fpcr, %0" ::"r"(w) : "memory");
      }
      static constexpr word_t down(word_t w) { return (w & ~(word_t(3) << 22)) | (word_t(2) << 22); }
      static constexpr word_t up(word_t w) { return (w & ~(word_t(3) << 22)) | (word_t(1) << 22); }
    };
#endif

    // ---- the tiles -------------------------------------------------------

    // A K-step after another, each element's steps in K order: the odd
    // products (or fused steps) into a buffer, then the casts and the split's
    // odd sums, each a pass over the whole tile. FTI, FTJ and FTK fix the
    // sizes (0: the run-time ones).
    template <int KIND, class Ops, int FTI, int FTJ, int FTK>
    [[gnu::always_inline]] inline void steps(const Ops &ops, const float *__restrict__ a_pack,
                                             const float *__restrict__ b_pack,
                                             float *__restrict__ sums, int64_t ti_, int64_t tk_,
                                             int64_t tj_, const BinaryKParams &pm,
                                             const BinaryKParams &pa)
    {
      const int64_t ti = FTI ? FTI : ti_, tk = FTK ? FTK : tk_, tj = FTJ ? FTJ : tj_;
      const int64_t n = ti * tj;
      alignas(64) float t[RNE_TILE_MAX];
      alignas(64) float u[RNE_TILE_MAX];
      for (int64_t k = 0; k < tk; ++k)
      {
        const float *__restrict__ b = b_pack + k * tj;
        if constexpr (KIND == 6) // FusedMacOdd::step
        {
          ops.template products<6>(a_pack + k, tk, b, sums, u, ti, tj);
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = cast_binaryK_rne_fast(u[l], pa);
        }
        else if constexpr (KIND == 5) // SplitMacOdd over an IdentityAdder
        {
          ops.template products<5>(a_pack + k, tk, b, sums, t, ti, tj);
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = sums[l] + cast_binaryK_rne_fast(t[l], pm);
        }
        else // SplitMacOdd::step
        {
          ops.template products<4>(a_pack + k, tk, b, sums, t, ti, tj);
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            t[l] = cast_binaryK_rne_fast(t[l], pm);
          ops.odd_sums(sums, t, u, n);
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = cast_binaryK_rne_fast(u[l], pa);
        }
      }
    }

    template <int KIND, class Ops>
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
      const Ops ops;
      if (ti == 32 && tj == 32 && tk == 32)
        steps<KIND, Ops, 32, 32, 32>(ops, a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else
        steps<KIND, Ops, 0, 0, 0>(ops, a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      return keep_or_restore(sums, saved, n);
    }

// BODY<KIND, OPS> with OPS given, BODY<KIND> without.
#define MPTORCH_ODD_TILES(SUFFIX, TARGET, BODY, ...)                                               \
  TARGET bool split_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,     \
                             int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)         \
  {                                                                                                \
    return BODY<4 __VA_OPT__(, ) __VA_ARGS__>(a, b, s, ti, tk, tj, pm, pa);                        \
  }                                                                                                \
  TARGET bool mul_only_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,  \
                                int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)      \
  {                                                                                                \
    return BODY<5 __VA_OPT__(, ) __VA_ARGS__>(a, b, s, ti, tk, tj, pm, pa);                        \
  }                                                                                                \
  TARGET bool fused_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,     \
                             int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)         \
  {                                                                                                \
    return BODY<6 __VA_OPT__(, ) __VA_ARGS__>(a, b, s, ti, tk, tj, pm, pa);                        \
  }

#if defined(__x86_64__)
#define MPTORCH_AVX512 __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,fma")))
#define MPTORCH_AVX2 __attribute__((target("avx2,fma")))

    // The directed pair and the select on RD's word parity, sixteen lanes at
    // a time by embedded rounding, the row's tail masked (tj is at most 32).
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

    // The tile's n lanes from l on, sixteen at a time, the tail masked.
    [[gnu::always_inline]] MPTORCH_AVX512 inline __mmask16 lanes(int64_t n, int64_t l)
    {
      return n - l >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << (n - l)) - 1);
    }

    // steps() with the odd passes in intrinsics: the directed pair needs no
    // mode, so the products and the odd sums are one pass each.
    template <int KIND, int FTI, int FTJ, int FTK>
    [[gnu::always_inline]] MPTORCH_AVX512 inline void
    steps_avx512(const float *__restrict__ a_pack, const float *__restrict__ b_pack,
                 float *__restrict__ sums, int64_t ti_, int64_t tk_, int64_t tj_,
                 const BinaryKParams &pm, const BinaryKParams &pa)
    {
      const int64_t ti = FTI ? FTI : ti_, tk = FTK ? FTK : tk_, tj = FTJ ? FTJ : tj_;
      const int64_t n = ti * tj;
      alignas(64) float t[RNE_TILE_MAX];
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
    }

    template <int KIND>
    [[gnu::always_inline]] MPTORCH_AVX512 inline bool
    tile_body_avx512(const float *__restrict__ a_pack, const float *__restrict__ b_pack,
                     float *__restrict__ sums, int64_t ti, int64_t tk, int64_t tj,
                     const BinaryKParams &mul_p, const BinaryKParams &add_p)
    {
      const BinaryKParams pm = mul_p, pa = add_p;
      const int64_t n = ti * tj;
      alignas(64) float saved[RNE_TILE_MAX];
      std::memcpy(saved, sums, static_cast<size_t>(n) * sizeof(float));
      if (ti == 32 && tj == 32 && tk == 32)
        steps_avx512<KIND, 32, 32, 32>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else
        steps_avx512<KIND, 0, 0, 0>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      return keep_or_restore(sums, saved, n);
    }

    MPTORCH_ODD_TILES(avx512, MPTORCH_AVX512, tile_body_avx512)
    MPTORCH_ODD_TILES(avx2, MPTORCH_AVX2, tile_body, Directed<Mxcsr>)
#undef MPTORCH_AVX512
#undef MPTORCH_AVX2
#elif defined(__aarch64__)
    MPTORCH_ODD_TILES(neon, , tile_body, Directed<Fpcr>)
#endif
#undef MPTORCH_ODD_TILES

    // gemm_rne_tile.cpp's choice, restated rather than shared, so that file
    // and its object do not depend on this one.
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
