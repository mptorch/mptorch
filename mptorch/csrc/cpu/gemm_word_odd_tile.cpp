// The vector tiles of the odd macs under RNA, RU, RD, RZ and RO
// (cpu/gemm_word_odd_tile.h).
//
// These are cpu/gemm_odd_tile.cpp's tiles with the word casts of
// common/cast_binaryK_word.h in place of cast_binaryK_rne_fast, and they share
// its structure: a K-step's odd products (or odd fused steps), row by row,
// into a buffer, then each cast and each odd sum as a pass over the whole
// tile, a copy of the code with the sizes fixed for a whole tile, the
// hand-back of a tile with a NaN, the no-trapping-math pragma ahead of every
// include, and the instruction set chosen once when the extension loads.
//
// The odd primitives come from the FP unit's rounding mode on every
// instruction set: a pass rounded down, a pass rounded up that keeps
// whichever result has an odd word, then round to nearest again for the
// casts, as gemm_odd_tile.cpp computes them under AVX2 and NEON (it says why
// that is the scalar step's result, and how the mode switches are fenced).
// Its AVX-512 path, embedded rounding, is not copied here: the casts of
// these modes cost more than the directed pair, so the simpler path serves
// both x86 instruction sets. The casts' only float operation is a compare,
// which no rounding mode changes, and they run in round to nearest anyway.

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-trapping-math")
#endif

#include "gemm_word_odd_tile.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace mptorch::gemm_cpu
{
  namespace
  {
    // A tile of fewer than eight outputs goes back to the scalar loop
    // whole. Each K-step's passes would then run one lane or a few, and
    // their setup costs more than the lanes save: on one thread the scalar
    // loop was 1.2-2x faster for a single output (a sum into one scalar),
    // and these tiles first won at eight outputs
    // (dev/benchmarks/timing/cpu_narrow_gemm.py).
    constexpr int64_t NARROW_TILE = 8;

    namespace odd = mptorch::gemm::odd;

    // A NaN anywhere in the tile hands the K-block back to the scalar loop,
    // its sums restored (cpu/gemm_rne_tile.h says why).
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

    [[gnu::always_inline]] inline float odd_of(float rd, float ru)
    {
      return (odd::word(rd) & 1u) ? rd : ru;
    }

    // gemm_odd_tile.cpp's Directed: Mode provides word_t, get(), set(w)
    // (fenced by compiler barriers), and down(w) and up(w).
    template <class Mode>
    struct Directed
    {
      typename Mode::word_t nearest = Mode::get();

      template <bool FUSED>
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
            if constexpr (FUSED)
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
            if constexpr (FUSED)
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
    using Ops = Directed<Mxcsr>;
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
      static constexpr word_t down(word_t w)
      {
        return (w & ~(word_t(3) << 22)) | (word_t(2) << 22);
      }
      static constexpr word_t up(word_t w) { return (w & ~(word_t(3) << 22)) | (word_t(1) << 22); }
    };
    using Ops = Directed<Fpcr>;
#endif

#if defined(__x86_64__) || defined(__aarch64__)
    // One K-step after another, each element's steps in K order, as
    // gemm_odd_tile.cpp's steps: KIND 4 SplitMacOdd, 5 SplitMacOdd over an
    // IdentityAdder, 6 FusedMacOdd. FTI, FTJ and FTK fix the sizes at compile
    // time; 0 means the run-time size.
    template <RoundMode RM, int KIND, int FTI, int FTJ, int FTK>
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
        if constexpr (KIND == 6) // FusedMacOdd::step: add(odd::fma(a, b, acc))
        {
          ops.template products<true>(a_pack + k, tk, b, sums, u, ti, tj);
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = cast_binaryK_word<RM>(u[l], pa);
        }
        else if constexpr (KIND == 5) // ... over an IdentityAdder: acc + mul(odd::mul(a, b))
        {
          ops.template products<false>(a_pack + k, tk, b, sums, t, ti, tj);
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = sums[l] + cast_binaryK_word<RM>(t[l], pm);
        }
        else // SplitMacOdd::step: add(odd::add(acc, mul(odd::mul(a, b))))
        {
          ops.template products<false>(a_pack + k, tk, b, sums, t, ti, tj);
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            t[l] = cast_binaryK_word<RM>(t[l], pm);
          ops.odd_sums(sums, t, u, n);
#pragma omp simd
          for (int64_t l = 0; l < n; ++l)
            sums[l] = cast_binaryK_word<RM>(u[l], pa);
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
      const Ops ops;
      if (ti == 32 && tj == 32 && tk == 32)
        steps<RM, KIND, 32, 32, 32>(ops, a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else
        steps<RM, KIND, 0, 0, 0>(ops, a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      return keep_or_restore(sums, saved, n);
    }

// One mode's three tiles for one instruction set.
#define MPTORCH_WORD_ODD_TILES_MODE(SUFFIX, TARGET, RM, NAME)                                      \
  TARGET bool split_##NAME##_##SUFFIX(const float *a, const float *b, float *s, int64_t ti,        \
                                      int64_t tk, int64_t tj, const BinaryKParams &pm,             \
                                      const BinaryKParams &pa)                                     \
  {                                                                                                \
    return tile_body<RM, 4>(a, b, s, ti, tk, tj, pm, pa);                                          \
  }                                                                                                \
  TARGET bool mul_only_##NAME##_##SUFFIX(const float *a, const float *b, float *s, int64_t ti,     \
                                         int64_t tk, int64_t tj, const BinaryKParams &pm,          \
                                         const BinaryKParams &pa)                                  \
  {                                                                                                \
    return tile_body<RM, 5>(a, b, s, ti, tk, tj, pm, pa);                                          \
  }                                                                                                \
  TARGET bool fused_##NAME##_##SUFFIX(const float *a, const float *b, float *s, int64_t ti,        \
                                      int64_t tk, int64_t tj, const BinaryKParams &pm,             \
                                      const BinaryKParams &pa)                                     \
  {                                                                                                \
    return tile_body<RM, 6>(a, b, s, ti, tk, tj, pm, pa);                                          \
  }

#define MPTORCH_WORD_ODD_TILES(SUFFIX, TARGET)                                                     \
  MPTORCH_WORD_ODD_TILES_MODE(SUFFIX, TARGET, RoundMode::RNA, rna)                                 \
  MPTORCH_WORD_ODD_TILES_MODE(SUFFIX, TARGET, RoundMode::RU, ru)                                   \
  MPTORCH_WORD_ODD_TILES_MODE(SUFFIX, TARGET, RoundMode::RD, rd)                                   \
  MPTORCH_WORD_ODD_TILES_MODE(SUFFIX, TARGET, RoundMode::RZ, rz)                                   \
  MPTORCH_WORD_ODD_TILES_MODE(SUFFIX, TARGET, RoundMode::RO, ro)                                   \
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
    MPTORCH_WORD_ODD_TILES(avx512,
                           __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,fma"))))
    MPTORCH_WORD_ODD_TILES(avx2, __attribute__((target("avx2,fma"))))
#else
    MPTORCH_WORD_ODD_TILES(neon, )
#endif
#undef MPTORCH_WORD_ODD_TILES
#undef MPTORCH_WORD_ODD_TILES_MODE
#endif

    WordTiles choose_word_odd_tiles()
    {
      // As choose_word_tiles (cpu/gemm_word_tile.cpp).
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

  const WordTiles word_odd_tiles = choose_word_odd_tiles();
} // namespace mptorch::gemm_cpu
