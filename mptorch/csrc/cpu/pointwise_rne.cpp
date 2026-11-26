// The vector blocks of the fused binaryK_pointwise op under RNE
// (cpu/pointwise_rne.h).
//
// Each function runs one block with a loop the compiler vectorizes over the
// shipped cast_binaryK_rne_fast: a lane computes what the scalar loop computes
// for its element, the same carrier operation and the same cast. Two settings
// belong to this file alone, as they do to cpu/gemm_rne_tile.cpp:
//
//   * no-trapping-math and no-math-errno, through the pragma ahead of every
//     include, so that the cast and the step are compiled under them too.
//     Under -ftrapping-math the vectorizer may not compute both arms of the
//     cast's fast path and select between them, and under -fmath-errno
//     std::sqrt keeps a call to sqrtf for a negative operand, which sets
//     errno, so its loop does not vectorize. The options change only the
//     exception flags and errno, which nothing reads; no value changes, and
//     -ffp-contract=off still holds. They must stay out of setup.py's host
//     flags, where they would change every other object.
//   * the instruction set, through target attributes on x86-64, chosen once
//     when the extension loads; on arm64, NEON is the baseline.

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-trapping-math", "no-math-errno")
#endif

#include "pointwise_rne.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace mptorch_cpu::pointwise
{
#if defined(MPTORCH_FAST_CAST)
  namespace
  {
    // The step of op OP (common/pointwise.h's codes) to nearest, then the
    // cast, on each lane; HAS_Y says whether y is a tensor or the number yv.
    template <int OP, bool HAS_Y>
    [[gnu::always_inline]] inline void block(const float *__restrict__ x,
                                             const float *__restrict__ y, float yv,
                                             float *__restrict__ o, int64_t n,
                                             const BinaryKParams &p)
    {
#pragma omp simd
      for (int64_t i = 0; i < n; ++i)
      {
        const float b = HAS_Y ? y[i] : yv;
        float v;
        if constexpr (OP == 0)
          v = x[i] + b;
        else if constexpr (OP == 1)
          v = x[i] - b;
        else if constexpr (OP == 2)
          v = x[i] * b;
        else if constexpr (OP == 3)
          v = x[i] / b;
        else if constexpr (OP == 4)
          v = std::sqrt(x[i]);
        else if constexpr (OP == 5)
          v = b - x[i];
        else
          v = b / x[i];
        o[i] = cast_binaryK_rne_fast(v, p);
      }
    }

    template <int OP>
    [[gnu::always_inline]] inline void op_block(const float *x, const float *y, float yv, float *o,
                                                int64_t n, const BinaryKParams &p)
    {
      if (y)
        block<OP, true>(x, y, yv, o, n, p);
      else
        block<OP, false>(x, y, yv, o, n, p);
    }

    [[gnu::always_inline]] inline bool body(int32_t op, const float *x, const float *y, float yv,
                                            float *o, int64_t n, const BinaryKParams &params)
    {
      const BinaryKParams p = params;
      switch (op)
      {
      case 0:
        op_block<0>(x, y, yv, o, n, p);
        break;
      case 1:
        op_block<1>(x, y, yv, o, n, p);
        break;
      case 2:
        op_block<2>(x, y, yv, o, n, p);
        break;
      case 3:
        op_block<3>(x, y, yv, o, n, p);
        break;
      case 4:
        op_block<4>(x, y, yv, o, n, p);
        break;
      case 5:
        op_block<5>(x, y, yv, o, n, p);
        break;
      default:
        op_block<6>(x, y, yv, o, n, p);
        break;
      }
      uint32_t nan = 0;
#pragma omp simd reduction(| : nan)
      for (int64_t i = 0; i < n; ++i)
      {
        uint32_t w;
        std::memcpy(&w, &o[i], sizeof w);
        nan |= static_cast<uint32_t>((w & 0x7FFFFFFFu) > 0x7F800000u);
      }
      return nan == 0;
    }

#define MPTORCH_POINTWISE_RNE(SUFFIX, TARGET)                                                      \
  TARGET bool run_##SUFFIX(int32_t op, const float *x, const float *y, float yv, float *o,         \
                           int64_t n, const BinaryKParams &p)                                      \
  {                                                                                                \
    return body(op, x, y, yv, o, n, p);                                                            \
  }

#if defined(__x86_64__)
    MPTORCH_POINTWISE_RNE(avx512, __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,fma"))))
    MPTORCH_POINTWISE_RNE(avx2, __attribute__((target("avx2,fma"))))
#elif defined(__aarch64__)
    MPTORCH_POINTWISE_RNE(neon, )
#endif
#undef MPTORCH_POINTWISE_RNE

    RneBlocks choose()
    {
      RneBlocks t;
      // MPTORCH_CPU_SIMD as cpu/gemm_rne_tile.cpp reads it.
      const char *env = std::getenv("MPTORCH_CPU_SIMD");
      if (env && std::strcmp(env, "0") == 0)
        return t;
#if defined(__x86_64__)
      const bool avx2_only = env && std::strcmp(env, "avx2") == 0;
      __builtin_cpu_init();
      if (!avx2_only && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
          __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512bw"))
        t = {run_avx512, "avx512"};
      else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        t = {run_avx2, "avx2"};
#elif defined(__aarch64__)
      t = {run_neon, "neon"};
#endif
      return t;
    }
  } // namespace

  const RneBlocks pointwise_rne = choose();
#else
  // Without the fast cast there is no cast_binaryK_rne_fast to vectorize.
  const RneBlocks pointwise_rne{};
#endif
} // namespace mptorch_cpu::pointwise
