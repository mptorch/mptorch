#pragma once

// Vector code for the fused binaryK_pointwise op under RNE, which its CPU
// kernel (cpu/pointwise_binaryK.cpp) runs in place of the scalar loop when the
// two compute the same thing: the step rounds to nearest in binary32, the
// geometry is flat (every tensor operand laid out like the result), and the
// format is signed, under SUBNORMALS and admitted by make_binaryK_params'
// fast gate, exactly the case in which the scalar loop's
// cast_binaryK_nearest_even takes its fast path, cast_binaryK_rne_fast. The
// vector functions (cpu/pointwise_rne.cpp) compute the same step and call that
// same cast, one element per lane, so the result is the scalar loop's word
// for word. One case cannot be predicted: an operation on two NaNs returns
// one of them according to the operand order, which the vectorizer does not
// preserve, so a block whose result holds a NaN is computed again by the
// scalar loop. Which instruction set runs is chosen once, when the extension
// loads (pointwise_rne); a null entry, or MPTORCH_CPU_SIMD=0, keeps the
// scalar loop.

#include "../common/cast_binaryK.h"
#include <cstdint>

namespace mptorch_cpu::pointwise
{
  // o[i] = cast_binaryK_rne_fast(step(op, x[i], y[i]), p) for i < n, with
  // y[i] the number `yv` when y is null; false when an o[i] is a NaN, whose
  // block the caller then computes with the scalar loop. x, y and o do not
  // overlap.
  using RneBlockFn = bool (*)(int32_t op, const float *x, const float *y, float yv, float *o,
                              int64_t n, const BinaryKParams &p);

  struct RneBlocks
  {
    RneBlockFn run = nullptr;
    const char *isa = "none";
  };

  // AVX-512, or AVX2 with FMA, on x86-64, by what the CPU reports, and NEON
  // on arm64. Null without MPTORCH_FAST_CAST, on any other CPU, or under
  // MPTORCH_CPU_SIMD=0, and AVX2 at most under MPTORCH_CPU_SIMD=avx2 (read
  // once, when the extension loads).
  extern const RneBlocks pointwise_rne;
} // namespace mptorch_cpu::pointwise
