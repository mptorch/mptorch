// The vector loops of the elementwise binaryK quantizer
// (cpu/binaryK_quant_vector.h).
//
// Each loop casts one element per lane through the cast the scalar loop
// would call (cast_binaryK_rne_fast under RNE) or through its branchless
// copy (common/cast_binaryK_word.h, the other modes), so the result is the
// scalar loop's word for word. Two settings belong to this file alone, as
// they do to cpu/gemm_rne_tile.cpp and cpu/gemm_word_tile.cpp:
// no-trapping-math, through the pragma ahead of every include (the casts
// compare floats, and under -ftrapping-math the vectorizer will not select
// between arms that might raise different exceptions; no value changes, and
// -ffp-contract=off still holds), and the instruction set, by target
// attributes on x86-64 chosen once when the extension loads, NEON on arm64.

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-trapping-math")
#endif

#include "binaryK_quant_vector.h"

#include <cstdlib>
#include <cstring>

namespace mptorch_cpu
{
  namespace
  {
    template <RoundMode RM>
    [[gnu::always_inline]] inline float cast(float x, const BinaryKParams &p)
    {
#if defined(MPTORCH_FAST_CAST)
      if constexpr (RM == RoundMode::RNE)
        return cast_binaryK_rne_fast(x, p);
      else
#endif
        return cast_binaryK_word<RM>(x, p);
    }

    // No __restrict__: the in-place op passes a == o, which the loop allows,
    // since element i is read before it is written and nothing else is.
    template <RoundMode RM>
    [[gnu::always_inline]] inline void body(const float *a, float *o, int64_t n,
                                            const BinaryKParams &params)
    {
      const BinaryKParams p = params;
#pragma omp simd
      for (int64_t i = 0; i < n; ++i)
        o[i] = cast<RM>(a[i], p);
    }

// One mode's loop for one instruction set.
#define MPTORCH_VECTOR_QUANT(SUFFIX, TARGET, RM, NAME)                                             \
  TARGET void NAME##_##SUFFIX(const float *a, float *o, int64_t n, const BinaryKParams &p)         \
  {                                                                                                \
    body<RM>(a, o, n, p);                                                                          \
  }

// RNE's loop exists only where its fast path does.
#if defined(MPTORCH_FAST_CAST)
#define MPTORCH_VECTOR_QUANT_RNE(SUFFIX, TARGET)                                                   \
  MPTORCH_VECTOR_QUANT(SUFFIX, TARGET, RoundMode::RNE, rne)
#define MPTORCH_VECTOR_QUANT_RNE_ENTRY(SUFFIX)                                                     \
  t.run[static_cast<int>(RoundMode::RNE)] = rne_##SUFFIX;
#else
#define MPTORCH_VECTOR_QUANT_RNE(SUFFIX, TARGET)
#define MPTORCH_VECTOR_QUANT_RNE_ENTRY(SUFFIX)
#endif

// The six modes' loops for one instruction set, and their table.
#define MPTORCH_VECTOR_QUANTS(SUFFIX, TARGET)                                                      \
  MPTORCH_VECTOR_QUANT_RNE(SUFFIX, TARGET)                                                         \
  MPTORCH_VECTOR_QUANT(SUFFIX, TARGET, RoundMode::RNA, rna)                                        \
  MPTORCH_VECTOR_QUANT(SUFFIX, TARGET, RoundMode::RU, ru)                                          \
  MPTORCH_VECTOR_QUANT(SUFFIX, TARGET, RoundMode::RD, rd)                                          \
  MPTORCH_VECTOR_QUANT(SUFFIX, TARGET, RoundMode::RZ, rz)                                          \
  MPTORCH_VECTOR_QUANT(SUFFIX, TARGET, RoundMode::RO, ro)                                          \
  VectorQuants table_##SUFFIX()                                                                    \
  {                                                                                                \
    VectorQuants t;                                                                                \
    MPTORCH_VECTOR_QUANT_RNE_ENTRY(SUFFIX)                                                         \
    t.run[static_cast<int>(RoundMode::RNA)] = rna_##SUFFIX;                                        \
    t.run[static_cast<int>(RoundMode::RU)] = ru_##SUFFIX;                                          \
    t.run[static_cast<int>(RoundMode::RD)] = rd_##SUFFIX;                                          \
    t.run[static_cast<int>(RoundMode::RZ)] = rz_##SUFFIX;                                          \
    t.run[static_cast<int>(RoundMode::RO)] = ro_##SUFFIX;                                          \
    t.isa = #SUFFIX;                                                                               \
    return t;                                                                                      \
  }

#if defined(__x86_64__)
    MPTORCH_VECTOR_QUANTS(avx512, __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,fma"))))
    MPTORCH_VECTOR_QUANTS(avx2, __attribute__((target("avx2,fma"))))
#elif defined(__aarch64__)
    MPTORCH_VECTOR_QUANTS(neon, )
#endif
#undef MPTORCH_VECTOR_QUANTS
#undef MPTORCH_VECTOR_QUANT_RNE_ENTRY
#undef MPTORCH_VECTOR_QUANT_RNE
#undef MPTORCH_VECTOR_QUANT

    VectorQuants choose_vector_quants()
    {
      // As choose_word_tiles (cpu/gemm_word_tile.cpp): MPTORCH_CPU_SIMD=0
      // keeps the scalar loop, =avx2 stops below AVX-512, and anything else
      // takes the widest instruction set the CPU has.
      const char *env = std::getenv("MPTORCH_CPU_SIMD");
      if (env && std::strcmp(env, "0") == 0)
        return VectorQuants{};
#if defined(__x86_64__)
      const bool avx2_only = env && std::strcmp(env, "avx2") == 0;
      __builtin_cpu_init();
      if (!avx2_only && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
          __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512bw"))
        return table_avx512();
      if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return table_avx2();
      return VectorQuants{};
#elif defined(__aarch64__)
      return table_neon();
#else
      return VectorQuants{};
#endif
    }
  } // namespace

  const VectorQuants vector_quants = choose_vector_quants();
} // namespace mptorch_cpu
