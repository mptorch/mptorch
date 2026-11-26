#pragma once

// The fused pointwise ops (binaryK_pointwise, superfp_pointwise,
// fixedpoint_pointwise): one carrier operation and one cast per element, in
// one pass,
//
//     o[j] = storage(cast(step(x[j], y[j]))),
//
// where x[j] and y[j] are element j's operands under broadcasting (a number
// operand gives every element the same y), step is the op in the carrier
// (binary32 for float32, float16 and bfloat16 tensors, binary64 for float64
// ones), cast is the elementwise quantizer's cast of the format in its round
// mode, from the same parameters, and storage is the one conversion back to
// the operands' dtype. That is mptorch.quant.pointwise's statement for qadd,
// qsub, qmul, qdiv and qsqrt with a Quant slot, which it otherwise computes
// as torch's operation in the carrier followed by the quantizer, and these
// ops give the same bits: the same operations, the result contiguous in the
// broadcast shape, and under stochastic rounding the same random word for
// element j, keyed on j as the quantizers key it (cpu/utils.h's
// quant_kernel_sr, cuda/utils.cuh's quant_kernel_all_sr). The exception is
// a float64 root on the CPU, which torch computes within an ulp and these
// ops correctly rounded.
//
// The step rounds to nearest, or, under carrier_rounding=RoundMode.RO, it is
// the exact result rounded to odd in the carrier (carrier_odd.h,
// carrier_odd_div_sqrt.h), so that the cast rounds the exact value for the
// formats carrier_odd.h's bounds admit. A number operand is converted to
// the carrier as torch converts one (to nearest), and under RO it is first
// rounded to the operands' dtype, as mptorch.quant.pointwise does, so that
// both operands have its precision.
//
// The layout. The host broadcasts the operands to the result's shape (stride
// 0 where an operand is broadcast), merges adjacent dimensions that every
// operand lays out alike, and drops dimensions of size 1 (PointwiseGeom).
// Element j of the result is its j-th element in row-major order, and the
// result is written densely. A geometry that is still wider than
// POINTWISE_MAX_DIMS is made one dimension by copying the operands, as is any
// geometry the device would index past 2^31 elements with strides.
//
// The header is shared by the host, nvcc and Metal: everything here is
// written for all three, with a Metal arm where an Apple GPU's flush of
// binary32 subnormals reaches a plain operation (bit_helper.h's section on
// it).

#include "bit_helper.h"
#include "carrier_odd_div_sqrt.h"
#include "cast_binaryK.h"
#include "cast_fixedpoint.h"
#include "cast_superfp.h"
#include "gemm_gather.h"
#include "modes.h"

#if !defined(__METAL_VERSION__)
#include <cmath>
#include <cstdint>
#if !defined(__CUDA_ARCH__)
#include <bit>
#endif
#endif

namespace mptorch::pointwise
{
  // The op codes of the schema's `op` argument. RSUB and RDIV take the
  // number first: y - x and y / x.
  enum PointwiseOp : int32_t
  {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    SQRT = 4,
    RSUB = 5,
    RDIV = 6,
  };

  enum : int32_t
  {
    POINTWISE_OPS = 7,
    POINTWISE_MAX_DIMS = 6,
  };

  // The broadcast geometry, outermost dimension first. size[] holds the
  // result's merged sizes, sa[] and sb[] the operands' strides in elements
  // (0 where broadcast). `flat` means one dimension in which every tensor
  // operand is laid out like the result, so element j is at j. `by[d]`
  // divides by size[d], for d >= 1, and is set whenever the geometry is not
  // flat (the host makes sure numel < 2^31 then).
  struct PointwiseGeom
  {
    int32_t ndim = 1;
    int32_t has_b = 0;
    int32_t flat = 1;
    int64_t numel = 0;
    int64_t size[POINTWISE_MAX_DIMS] = {};
    int64_t sa[POINTWISE_MAX_DIMS] = {};
    int64_t sb[POINTWISE_MAX_DIMS] = {};
    mptorch::gemm::FastDivmod by[POINTWISE_MAX_DIMS];
  };

  // The operands' offsets of element j < 2^31 of a geometry that is not
  // flat, innermost dimension first.
  CUDA_HOST_DEVICE_INLINE void offsets(const MPTORCH_THREAD PointwiseGeom &g, int32_t j,
                                       MPTORCH_THREAD int64_t &oa, MPTORCH_THREAD int64_t &ob)
  {
    oa = 0;
    ob = 0;
    for (int d = g.ndim - 1; d > 0; --d)
    {
      int32_t q, r;
      g.by[d].divmod(j, q, r);
      oa += r * g.sa[d];
      ob += r * g.sb[d];
      j = q;
    }
    oa += j * g.sa[0];
    ob += j * g.sb[0];
  }

  // ---- the step ---------------------------------------------------------
  //
  // The carrier's operations rounded to nearest, each spelled so that no
  // compiler fuses it with the cast's arithmetic: CUDA's _rn intrinsics,
  // which nvcc never contracts into a fused multiply-add; on the host,
  // setup.py's -ffp-contract=off; on an Apple GPU, bit_helper.h's exact
  // spellings, since the GPU flushes binary32 subnormals.
  namespace rn
  {
#if defined(__CUDA_ARCH__)
    __device__ __forceinline__ float add(float x, float y) { return __fadd_rn(x, y); }
    __device__ __forceinline__ float sub(float x, float y) { return __fsub_rn(x, y); }
    __device__ __forceinline__ float mul(float x, float y) { return __fmul_rn(x, y); }
    __device__ __forceinline__ float div(float x, float y) { return __fdiv_rn(x, y); }
    __device__ __forceinline__ float sqrt(float x) { return __fsqrt_rn(x); }
    __device__ __forceinline__ double add(double x, double y) { return __dadd_rn(x, y); }
    __device__ __forceinline__ double sub(double x, double y) { return __dsub_rn(x, y); }
    __device__ __forceinline__ double mul(double x, double y) { return __dmul_rn(x, y); }
    __device__ __forceinline__ double div(double x, double y) { return __ddiv_rn(x, y); }
    __device__ __forceinline__ double sqrt(double x) { return __dsqrt_rn(x); }
#elif defined(__METAL_VERSION__)
    // x + y is x - (-y), whose zero has the sum's sign in every case: +0 from
    // equal magnitudes of opposite signs, -0 only from -0 + -0.
    inline float add(float x, float y) { return metal_sub_rn(x, -y); }
    inline float sub(float x, float y) { return metal_sub_rn(x, y); }
    inline float mul(float x, float y) { return metal_mul_rn(x, y); }
    inline float div(float x, float y) { return metal_div_rn(x, y); }
    // The GPU flushes a subnormal radicand: m 2^-149 is (2m) 2^-150, and
    // 2m < 2^24 is exact, so its root, scaled by 2^-75 (exact: the result is
    // normal), is the root of x. A negative subnormal has a NaN root.
    inline float sqrt(float x)
    {
      const uint32_t w = as_type<uint32_t>(x);
      if (metal_is_subnormal(w))
        return (w >> 31) != 0 ? as_type<float>(0x7FC00000u)
                              : metal::precise::sqrt(float(2u * (w & 0x007FFFFFu))) *
                                    as_type<float>(0x1A000000u); // 2^-75
      return metal::precise::sqrt(x);
    }
#else
    inline float add(float x, float y) { return x + y; }
    inline float sub(float x, float y) { return x - y; }
    inline float mul(float x, float y) { return x * y; }
    inline float div(float x, float y) { return x / y; }
    inline float sqrt(float x) { return std::sqrt(x); }
    inline double add(double x, double y) { return x + y; }
    inline double sub(double x, double y) { return x - y; }
    inline double mul(double x, double y) { return x * y; }
    inline double div(double x, double y) { return x / y; }
    inline double sqrt(double x) { return std::sqrt(x); }
#endif
  } // namespace rn

  // The step of op `op` on x and y (SQRT ignores y): to nearest, or rounded
  // to odd. Under RO a difference is the sum with y negated, which flips the
  // sign of a NaN y as mptorch.quant.pointwise's own odd step does.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T step(int32_t op, bool odd, T x, T y)
  {
    if (odd)
    {
      switch (op)
      {
      case ADD:
        return mptorch::gemm::odd::add(x, y);
      case SUB:
        return mptorch::gemm::odd::add(x, -y);
      case MUL:
        return mptorch::gemm::odd::mul(x, y);
      case DIV:
        return mptorch::gemm::odd::div(x, y);
      case SQRT:
        return mptorch::gemm::odd::sqrt(x);
      case RSUB:
        return mptorch::gemm::odd::add(y, -x);
      default:
        return mptorch::gemm::odd::div(y, x);
      }
    }
    switch (op)
    {
    case ADD:
      return rn::add(x, y);
    case SUB:
      return rn::sub(x, y);
    case MUL:
      return rn::mul(x, y);
    case DIV:
      return rn::div(x, y);
    case SQRT:
      return rn::sqrt(x);
    case RSUB:
      return rn::sub(y, x);
    default:
      return rn::div(y, x);
    }
  }

  // The arithmetic ops' step as the loops take it (the tables' is PwlStep).
  struct ArithStep
  {
    int32_t op;
    bool odd;

    template <class T>
    CUDA_HOST_DEVICE_INLINE T operator()(T x, T y) const
    {
      return step(op, odd, x, y);
    }
  };

  // ---- tables ------------------------------------------------------------
  //
  // The fused table ops (binaryK_pwl, superfp_pwl, fixedpoint_pwl) evaluate a
  // mptorch.quant.PWLTable in the carrier, operation for operation as
  // mptorch/quant/pwl.py's _interpolate and _rsqrt_reduced compute it with
  // torch's operations, then cast as the arithmetic ops do:
  //
  //   xi = x, or x clamped to [lo, hi] under TAILS_CLAMP;
  //   u = (xi - lo) * scale, with scale = segments / (hi - lo) computed in
  //       binary64 and rounded to the carrier, as torch rounds a number;
  //   i = clamp(floor(u), 0, segments - 1), with a NaN u read as 0, and
  //   f = u - i;
  //   y = e[i] + f * (e[i + 1] - e[i]), three roundings;
  //   under TAILS_RELU, relu(x) for x outside [lo, hi].
  //
  // A table of 1/sqrt (`rsqrt`) first writes v = m 4^k from v's exponent
  // field, k = floor(exponent / 2), and returns T(m) 2^-k, the powers of two
  // built on the word as _power_of_two builds them.
  enum PwlTails : int32_t
  {
    TAILS_EXTEND = 0,
    TAILS_CLAMP = 1,
    TAILS_RELU = 2,
    PWL_TAILS = 3,
  };

  template <class T>
  CUDA_HOST_DEVICE_INLINE T floor_of(T x)
  {
#if defined(__METAL_VERSION__)
    return metal::floor(x);
#elif defined(__CUDA_ARCH__)
    return floor(x);
#else
    return std::floor(x);
#endif
  }

  // A carrier value's word and back, without type punning through a
  // reference (which GCC's aliasing rules let it read before the write).
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
  CUDA_HOST_DEVICE_INLINE float float_of(uint32_t w)
  {
#if defined(__METAL_VERSION__)
    return as_type<float>(w);
#elif defined(__CUDA_ARCH__)
    return __uint_as_float(w);
#else
    return std::bit_cast<float>(w);
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
  CUDA_HOST_DEVICE_INLINE double double_of(uint64_t w)
  {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(w));
#else
    return std::bit_cast<double>(w);
#endif
  }
#endif

  // 2^p for an integer p, from the word ((p + bias) shifted into the
  // exponent field), as pwl.py's _power_of_two: an infinity or a sign bit
  // where p is out of the normal range, as there.
  CUDA_HOST_DEVICE_INLINE float power_of_two(int32_t p, float)
  {
    return float_of(static_cast<uint32_t>(p + 127) << 23);
  }
#if !defined(__METAL_VERSION__)
  CUDA_HOST_DEVICE_INLINE double power_of_two(int32_t p, double)
  {
    return double_of(static_cast<uint64_t>(static_cast<int64_t>(p) + 1023) << 52);
  }
#endif

  // The unbiased exponent of v's exponent field, as pwl.py's _exponent.
  CUDA_HOST_DEVICE_INLINE int32_t exponent_of(float v)
  {
    return static_cast<int32_t>((word_of(v) >> 23) & 0xFFu) - 127;
  }
#if !defined(__METAL_VERSION__)
  CUDA_HOST_DEVICE_INLINE int32_t exponent_of(double v)
  {
    return static_cast<int32_t>((word_of(v) >> 52) & 0x7FFu) - 1023;
  }
#endif

  // `e` holds the segments + 1 entries in the carrier.
  template <class T>
  struct PwlStep
  {
    const T *e;
    int32_t segments;
    T lo, hi, scale;
    int32_t tails;
    bool rsqrt;

    CUDA_HOST_DEVICE_INLINE T interpolate(T x) const
    {
      T xi = x;
      if (tails == TAILS_CLAMP)
        xi = x < lo ? lo : (x > hi ? hi : x);
      const T u = rn::mul(rn::sub(xi, lo), scale);
      T fi = u != u ? T(0) : floor_of(u);
      const T top = T(segments - 1);
      fi = fi < T(0) ? T(0) : (fi > top ? top : fi);
      const T f = rn::sub(u, fi);
      const int32_t i = static_cast<int32_t>(fi);
      const T e0 = e[i], e1 = e[i + 1];
      T y = rn::add(e0, rn::mul(f, rn::sub(e1, e0)));
      if (tails == TAILS_RELU && (x < lo || x > hi))
        y = x > T(0) ? x : T(0);
      return y;
    }

    // The second operand is unused: a table has one.
    CUDA_HOST_DEVICE_INLINE T operator()(T x, T) const
    {
      if (!rsqrt)
        return interpolate(x);
      const int32_t k = exponent_of(x) >> 1;
      const T m = rn::mul(x, power_of_two(-2 * k, T(0)));
      return rn::mul(interpolate(m), power_of_two(-k, T(0)));
    }
  };

  // ---- the casts --------------------------------------------------------
  //
  // Each family's cast in round mode RM, as its elementwise quantizer calls
  // it (cpu/*_kernel.cpp, cuda/*_kernel.cu), with the constants built once
  // per call. operator()(x, sg) is a deterministic mode's cast and
  // operator()(x, rv, sg) stochastic rounding's, with the element's random
  // word; `sg` is the format's sign, which the CPU kernels pass as a
  // constant (a template parameter there, as in the quantizers) and the
  // device kernels from is_signed.

  template <class T, RoundMode RM>
  struct BinaryKRound
  {
    using word_t = typename FloatTraits<T>::word_t;
    BinaryKParamsT<T> p;
    StochasticT<T> st;
    SubnormalsMode sm;
    bool is_signed;

    CUDA_HOST_DEVICE_INLINE T operator()(T x, bool sg) const
    {
      if constexpr (RM == RoundMode::RNE)
        return cast_binaryK_nearest_even(x, sg, sm, p);
      else if constexpr (RM == RoundMode::RNA)
        return cast_binaryK_nearest_away(x, sg, sm, p);
      else if constexpr (RM == RoundMode::RU)
        return cast_binaryK_up(x, sg, sm, p);
      else if constexpr (RM == RoundMode::RD)
        return cast_binaryK_down(x, sg, sm, p);
      else if constexpr (RM == RoundMode::RZ)
        return cast_binaryK_zero(x, sg, sm, p);
      else
        return cast_binaryK_odd(x, sg, sm, p);
    }
    CUDA_HOST_DEVICE_INLINE T operator()(T x, word_t rv, bool sg) const
    {
      return cast_binaryK_stochastic(x, rv, st, sg, sm, p);
    }
  };

  // From binaryK's (K, P) to the cast's widths, as the quantizers do: P - 1
  // stored significand bits, and the exponent takes what the significand and,
  // in a signed format, the sign leave. `spec` is prng_bits with the SR
  // variant (stochastic_host.h); the stochastic constants are built under SR
  // only.
  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE BinaryKRound<T, RM> make_binaryK_round(int K, int P, int bias,
                                                                 bool is_signed, int spec,
                                                                 SaturationMode saturation_mode,
                                                                 SubnormalsMode subnormals_mode)
  {
    BinaryKRound<T, RM> r{};
    r.p = make_binaryK_params<T>(P - 1, is_signed ? K - P : K - P + 1, bias, is_signed,
                                 saturation_mode,
                                 subnormals_mode == SubnormalsMode::EXTENDED_NORMALS);
    if constexpr (RM == RoundMode::SR)
      r.st = make_stochastic_binaryK<T>(r.p, subnormals_mode, is_signed, spec);
    r.sm = subnormals_mode;
    r.is_signed = is_signed;
    return r;
  }

  template <class T, RoundMode RM>
  struct SuperfpRound
  {
    using word_t = typename FloatTraits<T>::word_t;
    SuperfpParamsT<T> p;
    StochasticT<T> st;
    bool is_signed;

    CUDA_HOST_DEVICE_INLINE T operator()(T x, bool sg) const
    {
      if constexpr (RM == RoundMode::RNE)
        return cast_superfp_nearest_even(x, sg, p);
      else if constexpr (RM == RoundMode::RNA)
        return cast_superfp_nearest_away(x, sg, p);
      else if constexpr (RM == RoundMode::RU)
        return cast_superfp_up(x, sg, p);
      else if constexpr (RM == RoundMode::RD)
        return cast_superfp_down(x, sg, p);
      else if constexpr (RM == RoundMode::RZ)
        return cast_superfp_zero(x, sg, p);
      else
        return cast_superfp_odd(x, sg, p);
    }
    CUDA_HOST_DEVICE_INLINE T operator()(T x, word_t rv, bool sg) const
    {
      return cast_superfp_stochastic(x, rv, st, sg, p);
    }
  };

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE SuperfpRound<T, RM> make_superfp_round(int man_bits, int exp_bits,
                                                                 int normal_binades, int bias,
                                                                 bool is_signed, int spec,
                                                                 SaturationMode saturation_mode)
  {
    SuperfpRound<T, RM> r{};
    r.p = make_superfp_params<T>(man_bits, exp_bits, normal_binades, bias, saturation_mode);
    if constexpr (RM == RoundMode::SR)
      r.st = make_stochastic_superfp<T>(r.p, is_signed, spec);
    r.is_signed = is_signed;
    return r;
  }

  template <class T, RoundMode RM>
  struct FixedPointRound
  {
    using word_t = typename FloatTraits<T>::word_t;
    FixedPointParamsT<T> p;
    int spec;
    bool is_signed;

    CUDA_HOST_DEVICE_INLINE T operator()(T x, bool sg) const
    {
      if constexpr (RM == RoundMode::RNE)
        return cast_fixedpoint_nearest_even(x, sg, p);
      else if constexpr (RM == RoundMode::RNA)
        return cast_fixedpoint_nearest_away(x, sg, p);
      else if constexpr (RM == RoundMode::RU)
        return cast_fixedpoint_up(x, sg, p);
      else if constexpr (RM == RoundMode::RD)
        return cast_fixedpoint_down(x, sg, p);
      else if constexpr (RM == RoundMode::RZ)
        return cast_fixedpoint_zero(x, sg, p);
      else
        return cast_fixedpoint_odd(x, sg, p);
    }
    CUDA_HOST_DEVICE_INLINE T operator()(T x, word_t rv, bool sg) const
    {
      return cast_fixedpoint_stochastic(x, rv, spec, sg, p);
    }
  };

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE FixedPointRound<T, RM> make_fixedpoint_round(int wl, int fl,
                                                                       bool is_signed,
                                                                       bool symmetric, int spec)
  {
    FixedPointRound<T, RM> r{};
    r.p = make_fixedpoint_params<T>(wl, fl, is_signed, symmetric);
    r.spec = spec;
    r.is_signed = is_signed;
    return r;
  }
} // namespace mptorch::pointwise
