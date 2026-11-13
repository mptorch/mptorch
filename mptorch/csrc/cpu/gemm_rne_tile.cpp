// The vectorized binaryK RNE tiles of the CPU GEMM (cpu/gemm_rne_tile.h;
// added in eafa9aa).
//
// Each function is one packed tile's K-block, its column loop vectorized by
// the compiler over the shipped cast_binaryK_rne_fast: a lane computes what
// the scalar kernel's step computes, in the same order, so the results are
// the scalar kernel's word for word (a K-block that ends with a NaN in its
// tile is handed back to the scalar loop: gemm_rne_tile.h says why). Two
// things are this file's alone:
//
//   * no-trapping-math, by the pragma ahead of every include, so the cast is
//     compiled under it too. Under GCC's default -ftrapping-math the
//     vectorizer may not execute the fast path's two arms (the subnormal one
//     and the Veltkamp one) as selects, since either may raise an exception
//     the scalar branch would not, and an optimize attribute on the function
//     alone stopped GCC inlining the cast. Exception flags are all it
//     changes, and nothing reads them; no value moves. -ffp-contract=off
//     still holds, so no multiply-add is fused that the scalar code does
//     not fuse. Clang defaults to no-trapping-math.
//   * the ISA, by target attributes on x86-64 (the cast's default-target
//     definition inlines into them), chosen once when the extension loads;
//     arm64's NEON is its baseline.
//
// Measured (dev/benchmarks/timing/cpu_gemm_ab.cpp, Ryzen 9 7940HS, 256^3 on one
// thread, against the scalar kernel after eafa9aa's sign-test reorder): split
// 7.5 -> 1.44 ns/mac with AVX-512, 2.25 with AVX2; fused 5.3 -> 0.50 and 0.77.

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
    // K outside the rows: each output element still takes its steps in K
    // order, and one K-step's rows are independent chains, where rows
    // outside K leave a step's few vectors waiting on the last step's casts.
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
            if constexpr (KIND == 1) // SplitMac::step: add(acc + mul(a, b))
              s[j] = cast_binaryK_rne_fast(s[j] + cast_binaryK_rne_fast(a * b[j], pm), pa);
            else if constexpr (KIND == 2) // ... over an IdentityAdder
              s[j] = s[j] + cast_binaryK_rne_fast(a * b[j], pm);
            else // FusedMac::step: add(fma(a, b, acc))
              s[j] = cast_binaryK_rne_fast(std::fma(a, b[j], s[j]), pa);
          }
        }
      // A NaN anywhere: a sum of two NaNs may have returned the other one
      // (gemm_rne_tile.h), so the block goes back to the scalar loop.
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

#define MPTORCH_RNE_TILES(SUFFIX, TARGET)                                                          \
  TARGET bool split_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,     \
                             int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)         \
  {                                                                                                \
    return tile_body<1>(a, b, s, ti, tk, tj, pm, pa);                                              \
  }                                                                                                \
  TARGET bool mul_only_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,  \
                                int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)      \
  {                                                                                                \
    return tile_body<2>(a, b, s, ti, tk, tj, pm, pa);                                              \
  }                                                                                                \
  TARGET bool fused_##SUFFIX(const float *a, const float *b, float *s, int64_t ti, int64_t tk,     \
                             int64_t tj, const BinaryKParams &pm, const BinaryKParams &pa)         \
  {                                                                                                \
    return tile_body<3>(a, b, s, ti, tk, tj, pm, pa);                                              \
  }

#if defined(__x86_64__)
    MPTORCH_RNE_TILES(avx512, __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,fma"))))
    MPTORCH_RNE_TILES(avx2, __attribute__((target("avx2,fma"))))
#elif defined(__aarch64__)
    MPTORCH_RNE_TILES(neon, )
#endif
#undef MPTORCH_RNE_TILES

    RneTiles choose_rne_tiles()
    {
      RneTiles t;
      // MPTORCH_CPU_SIMD=0 keeps the scalar loop, =avx2 stops below AVX-512
      // (so a machine that has it can check the AVX2 tiles too), and anything
      // else, or nothing, takes the widest the CPU has.
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
