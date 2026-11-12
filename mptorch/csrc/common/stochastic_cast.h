#pragma once

// _stochastic_cast: a format's stochastic cast with its random word given
// rather than drawn, elementwise, so that a test can run every draw R of N
// bits against P3109's definition and count exactly
// (tests/test_stochastic_rounding.py). Test-only, like _carrier_odd.
// ATen-free, for the CPU kernel and the .cu alike; the checks and the job's
// construction are stochastic_cast_host.h's.

#include "cast_binaryK.h"
#include "cast_fixedpoint.h"
#include "cast_superfp.h"

namespace mptorch::stochastic_cast
{
  // One format of one family: 0 binaryK, 1 superfp, 2 fixed point, with the
  // spec its prng_bits field carries (N and the variant, stochastic.h).
  template <class T>
  struct Job
  {
    int family;
    bool is_signed;
    SubnormalsMode subnormals;
    int spec;
    BinaryKParamsT<T> bk;
    SuperfpParamsT<T> sf;
    FixedPointParamsT<T> fx;
  };

  // The cast of x with the draw word `draw` (its MAN_BITS low bits).
  template <class T>
  CUDA_HOST_DEVICE_INLINE T cast(const MPTORCH_THREAD Job<T> &j, T x, typename FloatTraits<T>::word_t draw)
  {
    if (j.family == 0)
      return cast_binaryK_stochastic(x, draw, j.spec, j.is_signed, j.subnormals, j.bk);
    if (j.family == 1)
      return cast_superfp_stochastic(x, draw, j.spec, j.is_signed, j.sf);
    return cast_fixedpoint_stochastic(x, draw, j.spec, j.is_signed, j.fx);
  }
} // namespace mptorch::stochastic_cast
