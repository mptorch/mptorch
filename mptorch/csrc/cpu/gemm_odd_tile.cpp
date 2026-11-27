// The vector tiles of the CPU GEMM's binaryK steps under RNE when the
// carrier step rounds to odd (cpu/gemm_odd_tile.h;
// carrier_rounding=RoundMode.RO).
//
// These are gemm_rne_tile.cpp's tiles with the odd macs' steps, and they
// share its structure: the same passes (a K-step's products, row by row,
// into a buffer, then each cast and each odd sum as a pass over the whole
// tile, and a copy of the code with the sizes fixed for a whole tile), the
// same hand-back of a tile with a NaN, the same no-trapping-math pragma
// ahead of every include (so that the cast and the odd primitives are
// compiled under it), and the same choice of instruction set, made once when
// the extension loads. Every lane computes what the scalar kernel's step
// computes, in the same order, so the results are the scalar kernel's word
// for word.
//
// The odd primitives are computed as the CUDA kernels compute them
// (common/carrier_odd.h): the operation is rounded down and rounded up, and
// whichever result has an odd word is kept. Round to odd is that one of the
// two directed roundings, and when the operation is exact both are the exact
// value; the CPU and CUDA odd ops agree on this word for word in the tests.
// A NaN is handed back to the scalar loop either way, and infinite operands
// give IEEE's result. Each primitive here gives the same value as the scalar
// one for every input (dev/benchmarks/sweeps/cpu_odd_tile_sweep.cpp). There
// are two ways to get the directed roundings:
//
//   * AVX-512: embedded rounding, both roundings in one pass.
//   * AVX2 and NEON: the FP unit's rounding mode (MXCSR, FPCR), set once per
//     pass: a pass rounded down into the buffer, a pass rounded up that keeps
//     the odd word, then round to nearest again for the casts. Each switch is
//     fenced by compiler barriers, and each pass reads its operands from
//     memory and writes its result to memory, so no operation can move into
//     another pass's mode. The compiler is not told about the mode change,
//     which is safe because it has nothing to fold here: nothing in these
//     passes is a constant.
//
// Measured on 2026-11-14 (dev/benchmarks/timing/cpu_gemm_ab.cpp, ns per mac,
// one thread at 256^3). "a -> b" compares an earlier row-loop version with
// this one: on AVX-512 the same directed passes without the fixed sizes, and
// on AVX2 and NEON carrier_odd.h's binary64 primitives vectorized row by row,
// at half binary32's width and with a long dependency chain per lane.
//
//   * Ryzen 9 7940HS at 4.8 GHz: AVX-512 split 8p4/16p11 0.46 -> 0.43, fused
//     16p11 0.26 -> 0.22, product only 0.24 -> 0.24, against the default
//     tile's 0.39, 0.21 and 0.22; AVX2 split 2.99 -> 0.64, fused 1.32 -> 0.31,
//     product only 1.05 -> 0.31, against 0.53, 0.27 and 0.28.
//   * M3 Pro: split 2.36 -> 0.90, fused 1.10 -> 0.45, product only
//     0.85 -> 0.46, against 0.65, 0.32 and 0.34; 1.8-3.0x faster at 1024^3 on
//     eleven threads.
//
// Binary32 error-free transforms (an fma for the product's error, 2Sum, and
// Boldo and Muller's ErrFma for the sign of its error, with a binary64
// fallback) were measured as the AVX2 and NEON bodies in place of the
// rounding mode: 1.3-2.6x slower on both, and on NEON the fused tile was no
// faster than the binary64 row loop (commit e6ffed9 has that code).

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
    // A tile of fewer than four outputs runs narrow_steps below in place of
    // the passes, for gemm_rne_tile.cpp's reasons (its NARROW_TILE): the
    // passes on one lane or a few cost up to 2x the scalar loop, and the
    // palette tiles call these functions on column and row groups, so the
    // tile is not handed back.
    constexpr int64_t NARROW_TILE = 4;

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

    // ---- the directed pair under the FP unit's rounding mode -------------

    [[gnu::always_inline]] inline float odd_of(float rd, float ru)
    {
      return (odd::word(rd) & 1u) ? rd : ru;
    }

    // Mode provides word_t (the control register's type), get(), set(w)
    // (fenced by compiler barriers), and down(w) and up(w), which return w
    // with its rounding field set toward -inf and toward +inf.
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

    // One K-step after another, each element's steps in K order: the odd
    // products (or the fused steps) into a buffer, then the casts and the
    // split tile's odd sums, each as a pass over the whole tile. FTI, FTJ and
    // FTK fix the sizes at compile time; 0 means the run-time size.
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

    // A narrow tile: each output's K steps as one chain, the chains of the
    // tile's outputs interleaved, with the scalar kernel's own step
    // (common/carrier_odd.h's SplitMacOdd and FusedMacOdd: the primitives
    // there, which the passes' primitives equal on every input, and the
    // cast), in the FP unit's default rounding.
    template <int KIND>
    [[gnu::always_inline]] inline float narrow_step(float a, float b, float s,
                                                    const BinaryKParams &pm,
                                                    const BinaryKParams &pa)
    {
      if constexpr (KIND == 4)
        return cast_binaryK_rne_fast(odd::add(s, cast_binaryK_rne_fast(odd::mul(a, b), pm)), pa);
      else if constexpr (KIND == 5)
        return s + cast_binaryK_rne_fast(odd::mul(a, b), pm);
      else
        return cast_binaryK_rne_fast(odd::fma(a, b, s), pa);
    }

    // N outputs, in registers across the K-block (through memory, each step
    // would wait on the last one's store), output l at row l / tj, column
    // l % tj.
    template <int KIND, int N>
    [[gnu::always_inline]] inline void narrow_lanes(const float *__restrict__ a_pack,
                                                    const float *__restrict__ b_pack,
                                                    float *__restrict__ sums, int64_t tk,
                                                    int64_t tj, const BinaryKParams &pm,
                                                    const BinaryKParams &pa)
    {
      float s[N];
      int64_t ai[N], bj[N];
      for (int l = 0; l < N; ++l)
        s[l] = sums[l], ai[l] = (l / tj) * tk, bj[l] = l % tj;
      for (int64_t k = 0; k < tk; ++k)
        for (int l = 0; l < N; ++l)
          s[l] = narrow_step<KIND>(a_pack[ai[l] + k], b_pack[k * tj + bj[l]], s[l], pm, pa);
      for (int l = 0; l < N; ++l)
        sums[l] = s[l];
    }

    template <int KIND>
    [[gnu::always_inline]] inline void narrow_steps(const float *__restrict__ a_pack,
                                                    const float *__restrict__ b_pack,
                                                    float *__restrict__ sums, int64_t ti,
                                                    int64_t tk, int64_t tj, const BinaryKParams &pm,
                                                    const BinaryKParams &pa)
    {
      static_assert(NARROW_TILE == 4, "a narrow_lanes per size below the cut-off");
      const int64_t n = ti * tj;
      if (n == 1)
        narrow_lanes<KIND, 1>(a_pack, b_pack, sums, tk, tj, pm, pa);
      else if (n == 2)
        narrow_lanes<KIND, 2>(a_pack, b_pack, sums, tk, tj, pm, pa);
      else
        narrow_lanes<KIND, 3>(a_pack, b_pack, sums, tk, tj, pm, pa);
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
      if (n < NARROW_TILE)
        narrow_steps<KIND>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else if (ti == 32 && tj == 32 && tk == 32)
        steps<KIND, Ops, 32, 32, 32>(ops, a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else
        steps<KIND, Ops, 0, 0, 0>(ops, a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      return keep_or_restore(sums, saved, n);
    }

// Each function calls BODY<KIND, OPS> when OPS is given, and BODY<KIND>
// otherwise.
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

    // The odd result of one operation on sixteen lanes: the operation rounded
    // down and up by AVX-512's embedded rounding, and a select on the parity
    // of the rounded-down word.
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

    // The mask of the lanes from l on, at most sixteen, among a tile's n
    // lanes.
    [[gnu::always_inline]] MPTORCH_AVX512 inline __mmask16 lanes(int64_t n, int64_t l)
    {
      return n - l >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << (n - l)) - 1);
    }

    // steps() with the odd passes written in intrinsics: embedded rounding
    // needs no mode switch, so the products and the odd sums are one pass
    // each.
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
      if (n < NARROW_TILE)
        narrow_steps<KIND>(a_pack, b_pack, sums, ti, tk, tj, pm, pa);
      else if (ti == 32 && tj == 32 && tk == 32)
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

    // The same choice as gemm_rne_tile.cpp's choose_rne_tiles, written out
    // again rather than shared, so that that file and its object do not
    // depend on this one.
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
