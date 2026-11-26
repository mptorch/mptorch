#pragma once

// Division and the square root rounded to odd in the carrier: odd::div and
// odd::sqrt, beside carrier_odd.h's odd::fma, odd::add and odd::mul, for the
// fused pointwise ops (carrier_rounding=RoundMode.RO on a quotient or a
// root). They return what carrier_odd.h's table says its primitives return
// (an exact zero keeps IEEE's sign, which for a quotient or a root of
// nonzero operands never arises; a nonzero result below the smallest
// subnormal is the smallest subnormal; a finite result beyond Omega is
// Omega; specials are IEEE's), and on the host a NaN operand gives the plain
// operation's NaN, payload included. Division by zero and the root of a
// negative number are IEEE's too: an infinity, a NaN.
//
// A header of its own, so that no odd GEMM's translation unit, and no Metal
// kernel compiled from carrier_odd.h, sees new text. On an Apple GPU the two
// primitives are carrier_odd_metal.h's (binary32 only).
//
// The methods:
//   - CUDA: the operation rounded down and up (__fdiv_rd and __fdiv_ru,
//     __fsqrt_rd and __fsqrt_ru, and their binary64 twins), and the result
//     whose word is odd, as carrier_odd.h's device primitives do.
//   - The host's binary32: the binary64 quotient or root, rounded to odd at
//     24 bits (carrier_odd.h's narrow). A quotient of two 24-bit numbers is
//     either a number of at most 24 bits or not a dyadic rational at all,
//     and in the second case it differs from every number f of 24 bits by
//     more than 2^-48 |f| (|a - f b| is at least f's and b's last places
//     multiplied together); a root likewise, by more than 2^-49 |f|.
//     binary64 rounds to nearest within 2^-53 of the result, so its result
//     is exact exactly when the true one is, and otherwise lies strictly
//     between the same two binary32 neighbours. Binary32 operands put every quotient and root in
//     binary64's normal range, so there is no slow path.
//   - The host's binary64: q = RN(a / b) and its remainder fma(-q, b, a),
//     which is exact when the last places of q and b multiplied together
//     are at least 2^-1074 (then |a - q b| < 2^52 of that product, a
//     multiple of it); s = RN(sqrt(x)) and fma(-s, s, x), exact for
//     x >= 2^-970 likewise. The remainder's sign (times b's, for the
//     quotient) is the side of the exact result, and it is zero exactly when
//     q or s is exact: carrier_odd.h's odd fix does the rest. Operands
//     outside that range, and specials, take an out-of-line path that
//     normalizes the operands' significands with frexp, takes the fast path
//     on those, and scales the odd result back onto binary64's grid on the
//     word (scale_odd).

#include "carrier_odd.h"

#if !defined(__METAL_VERSION__)
// carrier_odd.h's macro, which it undefines at its end.
#if defined(__clang__) && !defined(__CUDA_ARCH__)
#define MPTORCH_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define MPTORCH_NO_CONTRACT
#endif

namespace mptorch::gemm
{
  namespace odd
  {
#if defined(__CUDA_ARCH__)
    __device__ __forceinline__ float div(float a, float b)
    {
      const float rd = __fdiv_rd(a, b), ru = __fdiv_ru(a, b);
      return (__float_as_uint(rd) & 1u) ? rd : ru;
    }
    __device__ __forceinline__ float sqrt(float x)
    {
      const float rd = __fsqrt_rd(x), ru = __fsqrt_ru(x);
      return (__float_as_uint(rd) & 1u) ? rd : ru;
    }
    __device__ __forceinline__ double div(double a, double b)
    {
      const double rd = __ddiv_rd(a, b), ru = __ddiv_ru(a, b);
      return (__double2loint(rd) & 1) ? rd : ru;
    }
    __device__ __forceinline__ double sqrt(double x)
    {
      const double rd = __dsqrt_rd(x), ru = __dsqrt_ru(x);
      return (__double2loint(rd) & 1) ? rd : ru;
    }
#else
    // ---- binary32: computed in binary64 ---------------------------------

    // r is a NaN exactly when an operand is a NaN, b is a zero or an
    // infinity, or a is an infinity: then the plain quotient is IEEE's.
    inline float div(float a, float b)
    {
      MPTORCH_NO_CONTRACT
      const double s = static_cast<double>(a) / static_cast<double>(b);
      const double r = s - static_cast<double>(static_cast<float>(s));
      if (r != r) [[unlikely]]
        return a / b;
      return narrow(s, 0.0);
    }

    // r is a NaN exactly when x is a NaN, negative or +inf (a -0 gives -0).
    inline float sqrt(float x)
    {
      MPTORCH_NO_CONTRACT
      const double s = std::sqrt(static_cast<double>(x));
      const double r = s - static_cast<double>(static_cast<float>(s));
      if (r != r) [[unlikely]]
        return std::sqrt(x);
      return narrow(s, 0.0);
    }

    // ---- binary64: the remainder by fma, with an exact slow path --------

    constexpr uint64_t SIGN64 = 0x8000000000000000ull;
    constexpr uint64_t MANT64 = 0x000FFFFFFFFFFFFFull;
    constexpr uint64_t ROOT_LO = static_cast<uint64_t>(-970 + 1023) << 52; // 2^-970
    constexpr uint64_t ROOT_HI = 0x7FF0000000000000ull;                     // +inf

    // RO(a / b) for operands that pass div's range test.
    inline double div_fast(double a, double b)
    {
      MPTORCH_NO_CONTRACT
      const double q = a / b;
      const double r = std::fma(-q, b, a);
      return fix(q, double_of(word(r) ^ (word(b) & SIGN64)));
    }

    // RO(sqrt(x)) for an x that passes sqrt's range test.
    inline double sqrt_fast(double x)
    {
      MPTORCH_NO_CONTRACT
      const double s = std::sqrt(x);
      return fix(s, std::fma(-s, s, x));
    }

    // RO of y * 2^k, for a normal y that is itself a binary64 rounding to odd
    // of the exact result divided by 2^k: the word with its exponent moved
    // where the result is normal; Omega beyond it (y * 2^k >= 2^1024 means
    // the exact result is above Omega); and below 2^-1022, the significand
    // shifted onto the 2^-1074 grid with a sticky bit, which rounds to odd
    // again (an integer shift, never a multiplication, which would round to
    // nearest there). Rounding to odd twice is rounding to odd once, since
    // y's own last bit is finer than that grid.
    inline double scale_odd(double y, int k)
    {
      const uint64_t w = word(y);
      const uint64_t sign = w & SIGN64;
      const int e = static_cast<int>((w >> 52) & 0x7FF) - 1023 + k;
      if (e > 1023)
        return double_of(sign | word(OMEGA));
      if (e >= -1022)
        return double_of(sign | (static_cast<uint64_t>(e + 1023) << 52) | (w & MANT64));
      const uint64_t m = (w & MANT64) | (1ull << 52);
      const int shift = -1022 - e; // at least 1
      if (shift >= 64)
        return double_of(sign | 1);
      const uint64_t sticky = (m & ((1ull << shift) - 1)) != 0;
      return double_of(sign | (m >> shift) | sticky);
    }

    // Everything div's range test sends here: specials, a zero, a subnormal
    // or huge divisor, a quotient that overflows or comes near the
    // subnormals. frexp's significands are in [1/2, 1), so their quotient is
    // in (1/2, 2) and passes the test.
    [[gnu::noinline]] inline double div_cold(double a, double b, double q)
    {
      if (!std::isfinite(a) || !std::isfinite(b) || (word(a) << 1) == 0 || (word(b) << 1) == 0)
        return q; // the plain a / b: exact, or IEEE's special value
      int ea, eb;
      const double ma = std::frexp(a, &ea), mb = std::frexp(b, &eb);
      return scale_odd(div_fast(ma, mb), ea - eb);
    }

    inline double div(double a, double b)
    {
      const double q = a / b;
      const uint64_t Q = (word(q) >> 52) & 0x7FF, B = (word(b) >> 52) & 0x7FF;
      // q and b normal, and their last places multiplied together at least
      // 2^-1074: e_q + e_b >= -970 for their unbiased exponents.
      const bool cold = (Q - 1 >= 2046) | (B - 1 >= 2046) | (Q + B < 1076);
      if (cold) [[unlikely]]
        return div_cold(a, b, q);
      return div_fast(a, b);
    }

    // Everything sqrt's range test sends here: a NaN, a zero, a negative x
    // or +inf (IEEE's root), and a positive x below 2^-970, whose root is
    // normal: x = m 2^e with e even and m in [1/2, 2), so sqrt(m) passes the
    // test and its scaling by 2^(e/2) >= 2^-537 is exact.
    [[gnu::noinline]] inline double sqrt_cold(double x, double s)
    {
      if (!(x > 0.0) || std::isinf(x))
        return s; // the plain root
      int e;
      double m = std::frexp(x, &e);
      if (e & 1)
      {
        m *= 2.0;
        e -= 1;
      }
      return scale_odd(sqrt_fast(m), e / 2);
    }

    inline double sqrt(double x)
    {
      if (word(x) - ROOT_LO >= ROOT_HI - ROOT_LO) [[unlikely]]
        return sqrt_cold(x, std::sqrt(x));
      return sqrt_fast(x);
    }
#endif
  } // namespace odd
} // namespace mptorch::gemm

#undef MPTORCH_NO_CONTRACT
#endif
