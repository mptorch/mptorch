#pragma once

// _stochastic_cast: a format's stochastic cast, applied elementwise with each
// element's random word passed in rather than drawn, so that a test can run
// every draw R of N bits through the cast and count the outcomes exactly
// against P3109's definition (tests/test_stochastic_rounding.py). A test-only
// op, like _carrier_odd. This header includes no ATen, so the CPU kernel and
// the .cu share it; stochastic_cast_host.h has the checks and builds the Job.

#include "cast_binaryK.h"
#include "cast_fixedpoint.h"
#include "cast_superfp.h"

namespace mptorch::stochastic_cast
{
  // One format of one family (0 binaryK, 1 superfp, 2 fixed point), with the
  // spec its prng_bits field carries: N and the stochastic variant
  // (stochastic.h).
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

  // Casts x with the random word `draw`, of which the cast reads the
  // MAN_BITS low bits.
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
