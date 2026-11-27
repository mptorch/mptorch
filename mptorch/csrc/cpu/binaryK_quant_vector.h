#pragma once

// Vector loops for the elementwise binaryK quantizer (cpu/binaryK_kernel.cpp)
// on float32 tensors in the deterministic modes. The kernel runs them in
// place of its scalar loop for a signed format whose scalar cast they
// compute, one element per lane:
//
//   * under RNE, cast_binaryK_rne_fast itself, which cast_binaryK_nearest_even
//     takes under SUBNORMALS for a format that passes the fast gate
//     (fast_rne);
//   * under RNA, RU, RD, RZ and RO, common/cast_binaryK_word.h's branchless
//     casts, whose words are the shipped casts' words for the formats
//     word_cast_format admits.
//
// Every other format, mode and dtype keeps the scalar loop. A cast reads one
// element, so unlike the GEMM's vector tiles nothing is handed back: the
// tiles' NaN rule is about the operand order of a sum. Which instruction set
// runs is chosen once, when the extension loads; a null entry, or
// MPTORCH_CPU_SIMD=0, keeps the scalar loop, and so does a build without
// MPTORCH_FAST_CAST under RNE, whose fast path it leaves out.

#include "../common/cast_binaryK_word.h"
#include <cstdint>

namespace mptorch_cpu
{
  // o[i] = the cast of a[i] for i < n. `o` may be `a` itself (the in-place
  // op); the two overlap in no other way.
  using VectorQuantFn = void (*)(const float *a, float *o, int64_t n, const BinaryKParams &p);

  struct VectorQuants
  {
    // Indexed by the RoundMode's value; null for the stochastic modes.
    VectorQuantFn run[static_cast<int>(RoundMode::SR)] = {};
    const char *isa = "none";
  };

  // Whether mode's loop computes the scalar loop's cast for a signed format
  // with params p under `subnormals`.
  inline bool vector_quant_format(RoundMode mode, SubnormalsMode subnormals, const BinaryKParams &p)
  {
    if (mode == RoundMode::RNE)
      return p.fast_rne && subnormals == SubnormalsMode::SUBNORMALS;
    return word_cast_format(true, subnormals, p);
  }

  // AVX-512, or AVX2 with FMA, on x86-64, by what the CPU reports, and NEON
  // on arm64; AVX2 at most under MPTORCH_CPU_SIMD=avx2, and none under
  // MPTORCH_CPU_SIMD=0 (read once, when the extension loads).
  extern const VectorQuants vector_quants;
} // namespace mptorch_cpu
