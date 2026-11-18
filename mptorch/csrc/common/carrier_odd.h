#pragma once

// The carrier's mac step rounded to odd (the ops for
// carrier_rounding=RoundMode.RO, custom_matmul_*_odd and custom_conv_*_odd).
//
// A NAIVE step rounds twice: first the carrier's add or fused multiply-add,
// to nearest, then the format's cast (gemm_policy.h's SplitMac and
// FusedMac). So the directed modes return the wrong neighbour whenever the
// exact sum lies within the carrier's last place of a format value. Rounding
// the carrier step to odd instead (round toward zero onto the carrier's
// grid, then set the last bit to 1 if the result is inexact) makes the cast
// round the exact value, for every format narrow enough (Boldo and
// Melquiond, inria-00080427). With P the format's precision and p the
// carrier's (24 or 53), RNE and RNA need P <= p - 2, the directed modes and
// RO need P <= p - 1, SR (StochasticA) needs P + prng_bits <= p - 1, and
// SR_B and SR_C need P + prng_bits <= p - 2, in the format's normal range.
// The Python spec builders refuse a call that breaks these bounds.
//
// This header holds what the odd ops add: the three primitives (odd::fma,
// odd::add and odd::mul) in both carriers, the two macs that use them, and
// the Args that builds those macs from an existing op's Args. On the CPU and
// CUDA, only the translation units of the odd ops and of the test-only
// _carrier_odd op include it, so the kernels of every other op compile
// exactly as they would without it. On MPS, the shared host code
// (mps/gemm_backend.cpp) includes it too, and the Metal compiler compiles it
// for the odd kernels only (mps/odd.metal), with binary32 primitives of its
// own (carrier_odd_metal.h), and with Metal versions of the statements where
// the GPU's flush of subnormals would reach a plain operation. Each Metal
// version is behind __METAL_VERSION__, with the host and CUDA statement,
// unchanged, in the other branch.
//
// What every primitive returns, the same on every backend, since the CPU and
// CUDA kernels must agree word for word:
//
//   exact zero from nonzero terms       +0
//   zero terms                          IEEE's sign: -0 only from -0 + -0
//   nonzero x below the smallest        +-the smallest subnormal, never 0
//     subnormal
//   finite |x| beyond the largest       +-the largest finite value, Omega:
//     finite value                        RZ's overflow, never infinity
//   infinite operands                   IEEE's result, which is exact
//   NaN                                 on CUDA, the hardware's NaN; on the
//                                         host, the plain operation's result
//                                         (std::fma, a + b, a * b), payload
//                                         included
//
// The "odd fix" that the host paths share: given a carrier value s and a
// residual tau that has the sign of x - s and is zero exactly when x = s, an
// inexact x lies strictly between s and its neighbour on tau's side. The two
// neighbours' words differ by one, and RO is the odd one: s | 1 when x is
// farther from zero than s, and (s - 1) | 1 when it is on s's side toward
// zero. This word arithmetic is continuous across binades and across the
// boundary between normals and subnormals, and s = Omega with x beyond it
// gives Omega | 1 = Omega.
//
// An independent review checked the algorithms against an exact Fraction
// reference on 2026-11-10, and tests/test_carrier_rounding.py checks the
// kernels against the same kind of reference through the _carrier_odd op.
//
// The host primitives assume what every host kernel here assumes: rounding
// to nearest, no contraction (setup.py's -ffp-contract=off; clang is also
// told so below, since by default it contracts a*b + c into an fma, which
// would break the error-free transforms), and no flushing of subnormals to
// zero (torch.set_flush_denormal(True) breaks them in subnormal ranges, as
// it breaks the carrier arithmetic everywhere else).

#include "gemm_policy.h"
#include "modes.h"

#if !defined(__METAL_VERSION__)
#include <cstdint>
#include <type_traits>
#if !defined(__CUDA_ARCH__)
#include <bit>
#include <cmath>
#endif

#if defined(__clang__) && !defined(__CUDA_ARCH__)
#define MPTORCH_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define MPTORCH_NO_CONTRACT
#endif
#else
#include "carrier_odd_metal.h"
#endif

namespace mptorch::gemm
{
  namespace odd
  {
#if defined(__METAL_VERSION__)
    // An Apple GPU's primitives are in carrier_odd_metal.h, included above.
#elif defined(__CUDA_ARCH__)
    // Two operations with directed rounding, and a select on the parity of
    // the round-down result's word. RD and RU give the same value when x is
    // exact, and adjacent words when it is not (RD = Omega and RU = inf on
    // overflow; RD = +0 and RU = the smallest subnormal below that): RO is
    // the odd one of the pair. An exact result with an even word takes RU's
    // value, which for a zero has round-toward-zero's sign (x + (-x) is +0
    // under RU and -0 under RD), and a NaN comes back as the hardware made
    // it. No sign test or comparison of values is needed, and the binary64
    // select stays off the FP64 units: the second DFMA is the only FP64
    // instruction it adds.
    __device__ __forceinline__ float fma(float a, float b, float c)
    {
      const float rd = __fmaf_rd(a, b, c), ru = __fmaf_ru(a, b, c);
      return (__float_as_uint(rd) & 1u) ? rd : ru;
    }
    __device__ __forceinline__ float add(float a, float b)
    {
      const float rd = __fadd_rd(a, b), ru = __fadd_ru(a, b);
      return (__float_as_uint(rd) & 1u) ? rd : ru;
    }
    __device__ __forceinline__ float mul(float a, float b)
    {
      const float rd = __fmul_rd(a, b), ru = __fmul_ru(a, b);
      return (__float_as_uint(rd) & 1u) ? rd : ru;
    }
    __device__ __forceinline__ double fma(double a, double b, double c)
    {
      const double rd = __fma_rd(a, b, c), ru = __fma_ru(a, b, c);
      return (__double2loint(rd) & 1) ? rd : ru;
    }
    __device__ __forceinline__ double add(double a, double b)
    {
      const double rd = __dadd_rd(a, b), ru = __dadd_ru(a, b);
      return (__double2loint(rd) & 1) ? rd : ru;
    }
    __device__ __forceinline__ double mul(double a, double b)
    {
      const double rd = __dmul_rd(a, b), ru = __dmul_ru(a, b);
      return (__double2loint(rd) & 1) ? rd : ru;
    }
#else
    // ---- host -----------------------------------------------------------

    inline uint32_t word(float x) { return std::bit_cast<uint32_t>(x); }
    inline uint64_t word(double x) { return std::bit_cast<uint64_t>(x); }
    inline float float_of(uint32_t w) { return std::bit_cast<float>(w); }
    inline double double_of(uint64_t w) { return std::bit_cast<double>(w); }

    // The odd fix (see the top of this file) on a binary64 word. It reads only
    // tau's sign and whether tau is zero; +0 and -0 both mean "exact".
    inline double fix(double s, double tau)
    {
      const uint64_t w = word(s), t = word(tau);
      const uint64_t inx = (t << 1) != 0;
      const uint64_t opp = ((w ^ t) >> 63) & inx;
      return double_of((w - opp) | inx);
    }

    // The odd fix of s toward a term of sign `negative` that is too small to
    // move it by more than a fraction of its last place: s is never the
    // exact result, so the fix always sets the last bit.
    inline double toward(double s, bool negative)
    {
      const uint64_t w = word(s);
      const uint64_t opp = (w >> 63) ^ static_cast<uint64_t>(negative);
      return double_of((w - opp) | 1);
    }

    // ---- binary32: computed in binary64, with no fma and no range branch --
    //
    // A product of two floats is exact in binary64 (48 bits, with |p| in
    // [2^-298, 2^256)), and a full 2Sum (s, t) then gives x = s + t exactly.
    // (A Fast2Sum in a fixed order is not enough: (2^-50, 2^-50, 1) and
    // (1, 1, 2^-100) lose t.) s is within half a binary64 ulp of x, so
    // f = (float)s is a binary32 neighbour of x (+-inf beyond Omega), s - f is
    // exact, and r = (s - f) + t has the sign of x - f and is zero exactly
    // when x = f. The odd fix of f's word by r is RO: an infinite f with a
    // finite x steps back to Omega, and a zero f with a nonzero x becomes the
    // smallest subnormal of x's sign. r is a NaN exactly when an operand is
    // not finite, and then the plain operation's result is returned.
    inline float narrow(double s, double t)
    {
      MPTORCH_NO_CONTRACT
      const float f = static_cast<float>(s);
      const double r = (s - static_cast<double>(f)) + t;
      const uint32_t w = word(f);
      const uint32_t inx = r != 0.0;
      const uint32_t opp = (static_cast<uint32_t>(r < 0.0) ^ (w >> 31)) & inx;
      return float_of((w - opp) | inx);
    }

    inline float fma(float a, float b, float c)
    {
      MPTORCH_NO_CONTRACT
      const double p = static_cast<double>(a) * static_cast<double>(b);
      const double d = c;
      const double s = p + d, pp = s - d, dd = s - pp;
      const double t = (p - pp) + (d - dd);
      const double r = (s - static_cast<double>(static_cast<float>(s))) + t;
      if (r != r) [[unlikely]]
        return std::fma(a, b, c);
      return narrow(s, t);
    }

    inline float add(float a, float b)
    {
      MPTORCH_NO_CONTRACT
      const double p = a, d = b;
      const double s = p + d, pp = s - d, dd = s - pp;
      const double t = (p - pp) + (d - dd);
      const double r = (s - static_cast<double>(static_cast<float>(s))) + t;
      if (r != r) [[unlikely]]
        return a + b;
      return narrow(s, t);
    }

    inline float mul(float a, float b)
    {
      MPTORCH_NO_CONTRACT
      const double s = static_cast<double>(a) * static_cast<double>(b);
      const double r = s - static_cast<double>(static_cast<float>(s));
      if (r != r) [[unlikely]]
        return a * b;
      return narrow(s, 0.0);
    }

    // ---- binary64: error-free transforms, with an exact slow path ---------

    // 2Sum: s + t = a + b exactly, with s = RN(a + b), for a and b in either
    // order (unless an intermediate overflows, which the callers rule out).
    inline void two_sum(double a, double b, double &s, double &t)
    {
      MPTORCH_NO_CONTRACT
      s = a + b;
      const double bb = s - a;
      t = (a - (s - bb)) + (b - bb);
    }

    // Fast2Sum, for |a| >= |b| (or a = 0): s + t = a + b exactly.
    inline void fast_two_sum(double a, double b, double &s, double &t)
    {
      MPTORCH_NO_CONTRACT
      s = a + b;
      t = b - (s - a);
    }

    // The word of 2^e, shifted left by one so that the sign bit drops out.
    constexpr uint64_t mag2(int e) { return static_cast<uint64_t>(e + 1023) << 53; }
    constexpr uint64_t TINY2 = mag2(-968); // below this, TwoProd's error underflows
    constexpr uint64_t HUGE2 = mag2(1021); // from this up, an intermediate may overflow
    constexpr uint64_t BIG2 = mag2(900);   // the bound for "far below a huge term"
    constexpr uint64_t SMALL2 = mag2(-800); // the bound for "far above a tiny product"
    constexpr uint64_t INF2 = 0x7FF0000000000000ull << 1;
    constexpr double OMEGA = 1.7976931348623157e308;

    // RO(a*b + c) by error-free transforms, for operands that pass the range
    // test: |a*b| and |c| below 2^1021, and a*b zero or with a representable
    // TwoProd error (|RN(a*b)| >= 2^-968). x = S + T + zt exactly. In both of
    // the cases the review distinguished (|uh| <= 2^48 |s1|, where zn is far
    // below s1, and cancellation, where s1 is exact by Sterbenz's lemma),
    // Fast2Sum's precondition holds, and tau = T + zt has the sign of x - S
    // and is zero exactly when x = S. An exact zero takes s1's IEEE sign
    // (fma(-0, 5, -0) is -0).
    inline double fma_fast(double a, double b, double c)
    {
      MPTORCH_NO_CONTRACT
      const double uh = a * b;
      const double ul = std::fma(a, b, -uh);
      double s1, e1, zn, zt, S, T;
      two_sum(uh, c, s1, e1);
      two_sum(e1, ul, zn, zt);
      fast_two_sum(s1, zn, S, T);
      const double r = fix(S, T + zt);
      return ((word(S) | word(s1)) << 1) == 0 ? s1 : r;
    }

    // RO of y * 2^-1248, where y = RO53(x * 2^1248) is normal: the way back
    // for the scaled-up tiny products below. If the result is normal, the
    // scaling is exact (two multiplications by 2^-624, each exact). In the
    // subnormal range it is not, so y is narrowed onto the 2^-1074 grid with
    // an integer shift and a sticky bit, never with a multiplication, which
    // would round to nearest there. Rounding to odd twice is the same as
    // rounding to odd once, because y's own odd bit is finer than that grid.
    inline double descale_1248(double y)
    {
      const double two_m624 = 0x1p-624;
      if (std::fabs(y) >= 0x1p226)
        return (y * two_m624) * two_m624;
      const uint64_t w = word(y);
      if ((w << 1) == 0)
        return y;
      const uint64_t sign = w & 0x8000000000000000ull;
      const int e = static_cast<int>((w >> 52) & 0x7FF) - 1023; // y is normal
      const uint64_t m = (w & 0x000FFFFFFFFFFFFFull) | (1ull << 52);
      const int shift = 226 - e; // the 2^-1074 grid is 2^174 at y's scale
      uint64_t k;
      bool sticky;
      if (shift >= 64)
      {
        k = 0;
        sticky = true;
      }
      else
      {
        k = m >> shift;
        sticky = (m & ((1ull << shift) - 1)) != 0;
      }
      return double_of(sign | k | static_cast<uint64_t>(sticky));
    }

    // Everything that the range test sends here: operands that are not
    // finite, a tiny nonzero product, a huge term. These are rare, so this
    // path is out of line and written for clarity rather than speed.
    [[gnu::noinline]] inline double fma_cold(double a, double b, double c)
    {
      if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
        return std::fma(a, b, c);
      const uint64_t A2 = word(a) << 1, B2 = word(b) << 1, C2 = word(c) << 1;
      if (A2 == 0 || B2 == 0)
        return std::fma(a, b, c); // a zero product, so the sum is exact
      const bool negative = ((word(a) ^ word(b)) >> 63) != 0;
      const double uh = a * b;
      const uint64_t U2 = word(uh) << 1;
      if (U2 < TINY2)
      {
        // |ab| < 2^-967, far below the last place of any c >= 2^-800.
        if (C2 >= SMALL2)
          return toward(c, negative);
        // Scale up exactly: |a|, |b| <= 2^106 here (each is at least
        // 2^-1074), so a * 2^624, b * 2^624 and c * 2^1248 are finite and
        // exact.
        const double up = 0x1p624;
        return descale_1248(fma_fast(a * up, b * up, (c * up) * up));
      }
      if (C2 >= HUGE2 && U2 < BIG2)
        return toward(c, negative); // |ab| < 2^901, below half an ulp of c
      if (U2 >= HUGE2 && C2 < BIG2)
      {
        if (U2 >= INF2)
          return negative ? -OMEGA : OMEGA; // |ab| beyond Omega
        // TwoProd's error is exact (it cannot underflow at this size) and a
        // multiple of at least 2^916, so when it is nonzero it outweighs
        // |c| < 2^900.
        const double ul = std::fma(a, b, -uh);
        return fix(uh, ((word(ul) << 1) != 0) ? ul : c);
      }
      // Both terms are at least 2^900: scale the larger factor and c by 1/16,
      // which is exact, and scale the result back.
      const double sixteenth = 0.0625;
      double a2 = a, b2 = b;
      if (A2 >= B2)
        a2 *= sixteenth;
      else
        b2 *= sixteenth;
      const double c2 = c * sixteenth;
      const double uh2 = a2 * b2;
      if ((word(uh2) << 1) >= HUGE2)
        return negative ? -OMEGA : OMEGA; // |ab| >= 2^1025 > |c| + Omega
      const double r = fma_fast(a2, b2, c2) * 16.0;
      if (std::isinf(r))
        return std::signbit(r) ? -OMEGA : OMEGA;
      return r;
    }

    inline double fma(double a, double b, double c)
    {
      const double uh = a * b;
      const uint64_t A2 = word(a) << 1, B2 = word(b) << 1, U2 = word(uh) << 1,
                     C2 = word(c) << 1;
      const bool cold = ((U2 < TINY2) & (A2 != 0) & (B2 != 0)) | (U2 >= HUGE2) | (C2 >= HUGE2);
      if (cold) [[unlikely]]
        return fma_cold(a, b, c);
      return fma_fast(a, b, c);
    }

    [[gnu::noinline]] inline double add_cold(double a, double b, double s)
    {
      if (!std::isfinite(a) || !std::isfinite(b))
        return s; // the plain a + b
      if (std::isinf(s))
        return std::signbit(s) ? -OMEGA : OMEGA;
      // A finite sum whose 2Sum overflowed internally (near Omega - 1.5
      // ulp(Omega)): Fast2Sum with the terms ordered by magnitude is exact,
      // and cannot overflow.
      double hi = a, lo = b;
      if ((word(a) << 1) < (word(b) << 1))
      {
        hi = b;
        lo = a;
      }
      double S, T;
      fast_two_sum(hi, lo, S, T);
      return fix(S, T);
    }

    inline double add(double a, double b)
    {
      double s, t;
      two_sum(a, b, s, t);
      if (t != t) [[unlikely]]
        return add_cold(a, b, s);
      return fix(s, t);
    }

    [[gnu::noinline]] inline double mul_cold(double a, double b, double uh)
    {
      if (!std::isfinite(a) || !std::isfinite(b) || (word(a) << 1) == 0 || (word(b) << 1) == 0)
        return uh; // the plain a * b: exact, or IEEE's special value
      if (std::isinf(uh))
        return std::signbit(uh) ? -OMEGA : OMEGA;
      return fma_cold(a, b, 0.0); // a tiny nonzero product, whose result is never zero
    }

    inline double mul(double a, double b)
    {
      const double uh = a * b;
      const uint64_t A2 = word(a) << 1, B2 = word(b) << 1, U2 = word(uh) << 1;
      const bool cold = ((U2 < TINY2) & (A2 != 0) & (B2 != 0)) | (U2 >= INF2);
      if (cold) [[unlikely]]
        return mul_cold(a, b, uh);
      return fix(uh, std::fma(a, b, -uh));
    }
#endif
  } // namespace odd

  // ----------------------------------------------------------------------
  // The macs.
  // ----------------------------------------------------------------------

#if defined(__METAL_VERSION__)
  // Whether an Adder is the IdentityAdder. Metal has no derived classes, so
  // its version cannot derive from std::true_type.
  template <class Adder>
  struct is_identity_adder
  {
    static constexpr MPTORCH_CONSTANT bool value = false;
  };
  template <class T>
  struct is_identity_adder<IdentityAdder<T>>
  {
    static constexpr MPTORCH_CONSTANT bool value = true;
  };
#else
  template <class Adder>
  struct is_identity_adder : std::false_type
  {
  };
  template <class T>
  struct is_identity_adder<IdentityAdder<T>> : std::true_type
  {
  };
#endif

  // FusedMac's step with the FMA rounded to odd, so that the format's cast
  // rounds a*b + acc itself. It exists only over a real Adder: a fused step
  // that no format rounds is the plain FusedMac<IdentityAdder>.
  template <class Adder>
  struct FusedMacOdd
  {
    using value_t = typename Adder::value_t;
    Adder add{};

    CUDA_HOST_DEVICE_INLINE value_t step(value_t a, value_t b, value_t acc, MPTORCH_THREAD PhiloxEngine &rng) const
    {
      return add(odd::fma(a, b, acc), rng);
    }
  };

  // SplitMac's step with the product rounded to odd before the multiply
  // format's cast, and the sum rounded to odd before the accumulate format's.
  // The multiplier policies multiply their operands themselves, so the odd
  // product is passed to one with 1 as the other factor, which is exact and
  // which the compiler removes. Over an IdentityAdder no format rounds the
  // sum, so it stays the carrier's add rounded to nearest, and only the
  // product is rounded to odd.
  template <class Multiplier, class Adder>
  struct SplitMacOdd
  {
    using value_t = typename Adder::value_t;
    Multiplier mul{};
    Adder add{};

    CUDA_HOST_DEVICE_INLINE value_t step(value_t a, value_t b, value_t acc, MPTORCH_THREAD PhiloxEngine &rng) const
    {
#if defined(__METAL_VERSION__)
      // The odd product goes straight to the multiply format's cast, and a
      // sum that no format rounds uses the nearest add that does not flush
      // subnormals (carrier_odd_metal.h).
      const value_t p = odd::cast_product(mul, odd::mul(a, b), rng);
      if constexpr (is_identity_adder<Adder>::value)
        return odd::add_rn(acc, p);
      else
        return add(odd::add(acc, p), rng);
#else
      const value_t p = mul(odd::mul(a, b), value_t(1), rng);
      if constexpr (is_identity_adder<Adder>::value)
        return acc + p;
      else
        return add(odd::add(acc, p), rng);
#endif
    }
  };

  // An existing single-format NAIVE Args with its mac's carrier steps
  // rounded to odd: the Args of the single-format and cross-family odd ops.
  // Like AccumulateArgs (gemm_accumulate.h), it reuses the base's factory,
  // which hands it the NaiveAccumulator to rebuild, so the choice between an
  // Adder and an IdentityAdder is written in one place. A FusedMac over an
  // IdentityAdder maps to nothing: the entry points refuse
  // fma_quant = false, so no kernel exists for it.
  template <class Base>
  struct OddArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = Base::draws_per_k_step;

    Base base{};

    template <class T, class F>
    struct Builder
    {
      MPTORCH_THREAD F &f;

      template <class Adder>
      void operator()(const MPTORCH_THREAD NaiveAccumulator<FusedMac<Adder>> &naive) const
      {
        if constexpr (!is_identity_adder<Adder>::value)
        {
          using Mac = FusedMacOdd<Adder>;
          f(NaiveAccumulator<Mac>{Mac{naive.mac.add}, T(0)});
        }
      }

      template <class Multiplier, class Adder>
      void operator()(const MPTORCH_THREAD NaiveAccumulator<SplitMac<Multiplier, Adder>> &naive) const
      {
        using Mac = SplitMacOdd<Multiplier, Adder>;
        f(NaiveAccumulator<Mac>{Mac{naive.mac.mul, naive.mac.add}, T(0)});
      }
    };

    template <class T, RoundMode RM, class F>
    void with_accumulator(MPTORCH_THREAD F &&f) const
    {
      base.template with_accumulator<T, RM>(Builder<T, F>{f});
    }
  };

} // namespace mptorch::gemm

#undef MPTORCH_NO_CONTRACT
