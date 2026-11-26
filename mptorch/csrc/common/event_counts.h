#pragma once

// Numeric event counts: how a format's casts went. Each event is a property
// of one cast, read from its input x (a value of the carrier) and its result
// q (a value of the format, in the carrier):
//
//   CASTS        every cast: the denominator
//   OVERFLOWED   x finite, and its magnitude past the largest finite value
//                on x's side of zero, whatever it rounds to (an infinity, or
//                the largest value under a saturating mode)
//   UNDERFLOWED  x finite and nonzero, and q zero
//   SUBNORMAL    q nonzero and below the format's normal range: a binaryK
//                subnormal under SUBNORMALS, a superfp supernormal; never
//                under NORMALS or EXTENDED_NORMALS, nor for fixed point
//   INEXACT      x not a NaN, and q a different value from x (-0 and +0 are
//                one value)
//   NAN_RESULT   q a NaN, made by this cast or carried into it
//
// A format's largest finite magnitude is the same on both sides of zero,
// except for a signed fixed-point format that is not symmetric, whose
// bottom is one step further out than its top. An unsigned format sends a
// negative input to zero: that is an underflow, not an overflow, unless its
// magnitude is past the largest value. A NaN is counted where it is a
// result, so the NaN of a dot product's inf + (-inf) is counted at the cast
// that receives it, and the overflows show where the infinities came from.
//
// Everything is decided on the words, with unsigned compares of magnitudes,
// so the classification is exact on every backend, an Apple GPU's flush of
// binary32 subnormals included (bit_helper.h). mptorch/quant/events.py's
// recount states the same events on torch tensors, for the quantizers.

#include "bit_helper.h"
#include "cast_binaryK.h"
#include "cast_fixedpoint.h"
#include "cast_superfp.h"
#include "modes.h"

#if !defined(__METAL_VERSION__)
#include <cstdint>
#if !defined(__CUDA_ARCH__)
#include <bit>
#endif
#endif

namespace mptorch::events
{
  // Not OVERFLOW, UNDERFLOW and NAN: <math.h> defines NAN, and Darwin's
  // defines the other two as well (System V's matherr codes), which would
  // replace the enumerators.
  enum Event : int32_t
  {
    CASTS = 0,
    OVERFLOWED = 1,
    UNDERFLOWED = 2,
    SUBNORMAL = 3,
    INEXACT = 4,
    NAN_RESULT = 5,
    EVENTS = 6,
  };

  // The roles a GEMM's counts are kept by: a split mac's product, its sum
  // (or a fused mac's one rounding, and KAHAN's term, sum and compensation),
  // and BLOCK's and TREE's outer format.
  enum Role : int32_t
  {
    MUL = 0,
    ACC = 1,
    OUTER = 2,
    ROLES = 3,
  };

  CUDA_HOST_DEVICE_INLINE uint32_t word_of(float x)
  {
#if defined(__METAL_VERSION__)
    return as_type<uint32_t>(x);
#elif defined(__CUDA_ARCH__)
    return __float_as_uint(x);
#else
    return std::bit_cast<uint32_t>(x);
#endif
  }
#if !defined(__METAL_VERSION__)
  CUDA_HOST_DEVICE_INLINE uint64_t word_of(double x)
  {
#if defined(__CUDA_ARCH__)
    return static_cast<uint64_t>(__double_as_longlong(x));
#else
    return std::bit_cast<uint64_t>(x);
#endif
  }
#endif

  // The magnitude word of 2^e in carrier T: a normal's, a subnormal's below
  // the normal range, 0 below the subnormals, and the infinity's above the
  // finite range.
  template <class T>
  CUDA_HOST_DEVICE_INLINE typename FloatTraits<T>::word_t pow2_word(int e)
  {
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    if (e > F::MAX_EXP)
      return F::INF_BITS;
    if (e >= F::MIN_NORMAL_EXP)
      return word_t(e + F::BIAS) << F::MAN_BITS;
    const int s = e - (F::MIN_NORMAL_EXP - F::MAN_BITS);
    return s < 0 ? word_t(0) : word_t(1) << s;
  }

  // The magnitude word of a binaryK or superfp format's largest finite
  // value, counted in code points as make_normal_range_params (bit_helper.h)
  // and Python's max_finite count it: `reserved` codes at the top of the top
  // binade, which can move it a binade or two down. A largest value past the
  // carrier's gives the infinity's word, since then no finite input is past
  // it (the cast's own max_num is the carrier's largest value on the format's
  // grid there, which an input can exceed without the format overflowing).
  template <class T>
  CUDA_HOST_DEVICE_INLINE typename FloatTraits<T>::word_t largest_finite_word(int exp_bits, int man_bits,
                                                                             int bias, int reserved)
  {
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    const int64_t codes = int64_t(1) << man_bits;
    int64_t top_code = codes - 1 - reserved;
    int64_t top_field = (int64_t(1) << exp_bits) - 1;
    while (top_code < 0)
    {
      top_code += codes;
      --top_field;
    }
    const int64_t top_exp = top_field - bias;
    if (top_exp > F::MAX_EXP)
      return F::INF_BITS;
    if (top_field < 1 || top_exp < F::MIN_NORMAL_EXP)
      return word_t(0); // no finite normal value: Python refuses such formats
    return (word_t(top_exp + F::BIAS) << F::MAN_BITS) | (word_t(top_code) << (F::MAN_BITS - man_bits));
  }

  // A format's thresholds, as magnitude words of the carrier: the largest
  // finite magnitude for a positive input and for a negative one, and the
  // bottom of the normal range (0 for a format with no subnormal results).
  // They are built from the format's own fields, not from its cast's
  // constants, which are chosen for the cast (see largest_finite_word).
  template <class T>
  struct EventThresholdsT
  {
    using word_t = typename FloatTraits<T>::word_t;
    word_t pos_max;
    word_t neg_max;
    word_t normal_min;
  };

  // binaryK: P3109 reserves the infinity's code outside the finite domain
  // and, in an unsigned format, the NaN's; the subnormals lie below 2^(1 -
  // bias), under SUBNORMALS only (the other two modes have no subnormal
  // results).
  template <class T>
  CUDA_HOST_DEVICE_INLINE EventThresholdsT<T> binaryK_thresholds(int man_bits, int exp_bits, int bias,
                                                                 bool is_signed, SaturationMode saturation,
                                                                 SubnormalsMode subnormals)
  {
    EventThresholdsT<T> t;
    const int reserved = (saturation != SaturationMode::SAT_FINITE) + (is_signed ? 0 : 1);
    t.pos_max = largest_finite_word<T>(exp_bits, man_bits, bias, reserved);
    t.neg_max = t.pos_max;
    t.normal_min = subnormals == SubnormalsMode::SUBNORMALS ? pow2_word<T>(1 - bias)
                                                            : typename FloatTraits<T>::word_t(0);
    return t;
  }

  // superfp: no NaN code, the infinity's outside the finite domain; below
  // the top normal_binades binades are the supernormals
  // (superfp_region_cutoffs' normal_cutoff).
  template <class T>
  CUDA_HOST_DEVICE_INLINE EventThresholdsT<T> superfp_thresholds(int man_bits, int exp_bits,
                                                                 int normal_binades, int bias,
                                                                 SaturationMode saturation)
  {
    EventThresholdsT<T> t;
    const int reserved = saturation != SaturationMode::SAT_FINITE;
    t.pos_max = largest_finite_word<T>(exp_bits, man_bits, bias, reserved);
    t.neg_max = t.pos_max;
    t.normal_min = pow2_word<T>(superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias).normal_cutoff);
    return t;
  }

  // fixed point: its two ends, which the Python checks keep inside the
  // carrier; a signed format that is not symmetric reaches one step further
  // below zero than above. An unsigned format's negative inputs go to zero,
  // an underflow (see the top of the file), and overflow past its top.
  template <class T>
  CUDA_HOST_DEVICE_INLINE EventThresholdsT<T>
  fixedpoint_thresholds(const MPTORCH_THREAD FixedPointParamsT<T> &p, bool is_signed)
  {
    using F = FloatTraits<T>;
    EventThresholdsT<T> t;
    t.pos_max = word_of(p.max_val);
    t.neg_max = is_signed ? (word_of(p.min_val) & F::ABS_MASK) : t.pos_max;
    t.normal_min = typename F::word_t(0);
    return t;
  }

  // The events of one cast (the table at the top of the file), one bit each.
  template <class T>
  CUDA_HOST_DEVICE_INLINE uint32_t event_bits(T x, T q, const MPTORCH_THREAD EventThresholdsT<T> &t)
  {
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    const word_t wx = word_of(x), wq = word_of(q);
    const word_t ax = wx & F::ABS_MASK, aq = wq & F::ABS_MASK;
    const bool negative = (wx >> (F::WORD_BITS - 1)) != 0;
    const bool x_finite = ax < F::INF_BITS;
    const bool x_nan = ax > F::INF_BITS;
    uint32_t bits = 1u << CASTS;
    bits |= uint32_t(x_finite && ax > (negative ? t.neg_max : t.pos_max)) << OVERFLOWED;
    bits |= uint32_t(x_finite && ax != 0 && aq == 0) << UNDERFLOWED;
    bits |= uint32_t(aq != 0 && aq < t.normal_min) << SUBNORMAL;
    bits |= uint32_t(!x_nan && wq != wx && !(ax == 0 && aq == 0)) << INEXACT;
    bits |= uint32_t(aq > F::INF_BITS) << NAN_RESULT;
    return bits;
  }

  // A role's six counters, in 32 bits: a thread's or a task's counts, which
  // the kernels add into 64-bit totals.
  struct EventTally
  {
    uint32_t n[EVENTS];

    CUDA_HOST_DEVICE_INLINE void add(uint32_t bits)
    {
      for (int e = 0; e < EVENTS; ++e)
        n[e] += (bits >> e) & 1u;
    }
  };
} // namespace mptorch::events
