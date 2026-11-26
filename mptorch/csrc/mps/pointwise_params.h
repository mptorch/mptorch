#pragma once

// What a fused pointwise or table op's launch takes besides its tensors and
// its geometry (common/pointwise.h's PointwiseGeom, a buffer of its own):
// pointwise_entry.cpp fills it and pointwise_kernel.metal reads it, and both
// compile this one definition. It is not in launch_params.h, which the
// prelude includes, so that adding it changed no other kernel's source.

#if !defined(__METAL_VERSION__)
#include <cstdint>
#endif

namespace mptorch_mps
{
  // `n` elements of the result, the SR seed, and the step: an arithmetic
  // op's code and carrier rounding (common/pointwise.h's ArithStep) with its
  // number operand already in the carrier, or a table's segments, range,
  // scale, tails and reduction (PwlStep), as the CPU entries build them.
  struct PointwiseLaunch
  {
    uint64_t n;
    uint64_t seed;
    int32_t op;
    int32_t odd;
    float y;
    int32_t segments;
    float lo, hi, scale;
    int32_t tails;
    int32_t rsqrt;
    int32_t unused;
  };
  static_assert(sizeof(PointwiseLaunch) == 56,
                "PointwiseLaunch must have one layout on the host and the GPU");
} // namespace mptorch_mps
