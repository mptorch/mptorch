#pragma once

// The carrier's mac step rounded to odd (the custom_matmul_*_odd ops).
//
// A NAIVE step rounds twice: the carrier's add or fused multiply-add to
// nearest, then the format's cast (gemm_policy.h's SplitMac and FusedMac).
// The directed modes then return the wrong neighbour whenever the exact sum
// lies within the carrier's last place of a format value. Rounding the
// carrier step to odd instead -- RZ onto the carrier's grid, with the last
// bit forced to 1 when the result is inexact -- makes the cast round the
// exact value, for every format narrow enough (Boldo and Melquiond,
// inria-00080427): RNE and RNA need P <= p - 2, the directed modes and RO
// P <= p - 1, and SR P + prng_bits <= p - 1 in the format's normal range,
// where p is 24 or 53. The Python spec builders hold a call to those bounds.
//
// This header is what the custom_matmul_*_odd ops add: the three primitives
// in both carriers, the two macs that use them, and the Args that builds the
// macs from an existing op's. No translation unit that existed before a13760a
// includes it, so the existing kernels are the bytes they were.
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
//   NaN                                 on CUDA the hardware's NaN; on the
//                                         host today's expression (std::fma,
//                                         a + b, a * b), payload and all
//
// The odd fix the host paths share: given a carrier value s and a residual
// tau that has the sign of x - s and is zero when x = s exactly, an inexact
// x lies strictly between s and its neighbour on tau's side. The two words
// differ by one and RO is the odd one: s | 1 when x is farther from zero
// than s, (s - 1) | 1 when it is on s's zero side. The word arithmetic is
// continuous across binades and the normal/subnormal boundary, and s = Omega
// with x beyond it gives Omega | 1 = Omega.
//
// The algorithms were designed and checked by an independent review on
// 2026-11-10 against an exact Fraction reference;
// tests/test_carrier_rounding.py holds the kernels to the same reference
// through the _carrier_odd op.
//
// The host primitives assume what every host kernel here assumes: round to
// nearest, no contraction (setup.py's -ffp-contract=off; clang is told so
// below too, since its default contracts a*b + c into an fma, which would
// break the error-free transforms), and no flush-to-zero
// (torch.set_flush_denormal(True) breaks them in subnormal ranges, as it
// breaks the carrier arithmetic everywhere else).

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

namespace mptorch::gemm
{
  namespace odd
  {
#if defined(__CUDA_ARCH__)
    // Two directed operations and a select on the parity of RD's word. RD
    // and RU are the same value when x is exact, and adjacent words when it
    // is not (RD = Omega with RU = inf on overflow, RD = +0 with RU = the
    // smallest subnormal below it): RO is the odd one of the pair. An exact
    // result with an even word takes RU's, which for a zero is RZ's sign
    // (x + (-x) is +0 under RU, -0 under RD), and a NaN comes back as the
    // hardware made it. The sign test and sticky bit of a value compare
    // are not needed, and the binary64 select stays off the FP64 pipe: the
    // second DFMA is the only FP64 instruction it adds.
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

    // The odd fix on a binary64 word (the header comment). tau's sign and
    // zeroness are all it reads; +0 and -0 both say "exact".
    inline double fix(double s, double tau)
    {
      const uint64_t w = word(s), t = word(tau);
      const uint64_t inx = (t << 1) != 0;
      const uint64_t opp = ((w ^ t) >> 63) & inx;
      return double_of((w - opp) | inx);
    }

    // The odd fix of s toward a value of sign `negative` that is too small
    // to move it by more than a fraction of its last place: s is never the
    // exact result, so the fix always sets the last bit.
    inline double toward(double s, bool negative)
    {
      const uint64_t w = word(s);
      const uint64_t opp = (w >> 63) ^ static_cast<uint64_t>(negative);
      return double_of((w - opp) | 1);
    }

    // ---- binary32: in binary64, with no fma and no range branch ----------
    //
    // A float product is exact in binary64 (48 bits, |p| in [2^-298, 2^256)),
    // and a full 2Sum (s, t) then holds x = s + t exactly (a fixed-order
    // Fast2Sum is not enough: (2^-50, 2^-50, 1) and (1, 1, 2^-100) lose t).
    // s is within half a binary64 ulp of x, so f = (float)s is a binary32
    // neighbour of x (+-inf past Omega), s - f is exact, and r = (s - f) + t
    // has the sign and zeroness of x - f. The odd fix on f's word with r is
    // RO: an infinite f with a finite x steps back to Omega, and a zero f
    // with a nonzero x becomes the smallest subnormal of x's sign. r is a
    // NaN exactly when an operand is not finite, which takes today's
    // expression.
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

    // ---- binary64: error-free transforms, with an exact cold path --------

    // 2Sum: s + t = a + b exactly, s = RN(a + b), for any order of a and b
    // (unless an intermediate overflows, which the callers keep out).
    inline void two_sum(double a, double b, double &s, double &t)
    {
      MPTORCH_NO_CONTRACT
      s = a + b;
      const double bb = s - a;
      t = (a - (s - bb)) + (b - bb);
    }

    // |a| >= |b| (or a = 0): s + t = a + b exactly.
    inline void fast_two_sum(double a, double b, double &s, double &t)
    {
      MPTORCH_NO_CONTRACT
      s = a + b;
      t = b - (s - a);
    }

    // The word, shifted left one so that the sign drops out, of 2^e.
    constexpr uint64_t mag2(int e) { return static_cast<uint64_t>(e + 1023) << 53; }
    constexpr uint64_t TINY2 = mag2(-968); // below it TwoProd's error underflows
    constexpr uint64_t HUGE2 = mag2(1021); // from it an intermediate may overflow
    constexpr uint64_t BIG2 = mag2(900);   // what "far below a huge term" means
    constexpr uint64_t SMALL2 = mag2(-800); // what "far above a tiny product" means
    constexpr uint64_t INF2 = 0x7FF0000000000000ull << 1;
    constexpr double OMEGA = 1.7976931348623157e308;

    // RO(a*b + c) by error-free transforms, for operands the range test let
    // through: |a*b| and |c| below 2^1021, and a*b zero or TwoProd's error
    // representable (|RN(a*b)| >= 2^-968). x = S + T + zt exactly; in both
    // of the review's regimes (|uh| <= 2^48 |s1|, where zn is far below s1,
    // and the cancellation one, where s1 is exact by Sterbenz) Fast2Sum's
    // precondition holds and tau = T + zt has the sign and zeroness of
    // x - S. An exact zero takes s1's IEEE sign (fma(-0, 5, -0) is -0).
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

    // RO of y * 2^-1248, where y = RO53(x * 2^1248) is normal: the scaled-up
    // tiny products below. Back in the normal range the scaling is exact
    // (two multiplies by 2^-624, each exact); in the subnormal range it is
    // not, and y is narrowed onto the 2^-1074 grid with an integer shift and
    // a sticky bit -- never a multiply, which would round to nearest there.
    // RO composes: y's own odd bit is finer than that grid's.
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
      const int shift = 226 - e; // the grid 2^-1074 is 2^174 at y's scale
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

    // Everything the range test sends here: non-finite operands, a tiny
    // nonzero product, a huge term. Rare, so out of line and spelled for
    // clarity.
    [[gnu::noinline]] inline double fma_cold(double a, double b, double c)
    {
      if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
        return std::fma(a, b, c);
      const uint64_t A2 = word(a) << 1, B2 = word(b) << 1, C2 = word(c) << 1;
      if (A2 == 0 || B2 == 0)
        return std::fma(a, b, c); // a zero product: the sum is exact
      const bool negative = ((word(a) ^ word(b)) >> 63) != 0;
      const double uh = a * b;
      const uint64_t U2 = word(uh) << 1;
      if (U2 < TINY2)
      {
        // |ab| < 2^-967, far below any c >= 2^-800's last place.
        if (C2 >= SMALL2)
          return toward(c, negative);
        // Scale up exactly: |a|, |b| <= 2^106 here (each is at least
        // 2^-1074), so a, b * 2^624 and c * 2^1248 are finite and exact.
        const double up = 0x1p624;
        return descale_1248(fma_fast(a * up, b * up, (c * up) * up));
      }
      if (C2 >= HUGE2 && U2 < BIG2)
        return toward(c, negative); // |ab| < 2^901, below half of c's ulp
      if (U2 >= HUGE2 && C2 < BIG2)
      {
        if (U2 >= INF2)
          return negative ? -OMEGA : OMEGA; // |ab| past Omega
        // TwoProd's error is exact (no underflow at this size), a multiple
        // of a quantum of at least 2^916, so it outweighs |c| < 2^900.
        const double ul = std::fma(a, b, -uh);
        return fix(uh, ((word(ul) << 1) != 0) ? ul : c);
      }
      // Both terms at least 2^900: scale the larger factor and c by 1/16,
      // exactly, and scale the result back.
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
        return s; // today's a + b
      if (std::isinf(s))
        return std::signbit(s) ? -OMEGA : OMEGA;
      // A finite sum whose 2Sum overflowed inside (Omega - 1.5 ulp(Omega)):
      // Fast2Sum ordered by magnitude is exact and cannot.
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
        return uh; // today's a * b: exact, or IEEE's special value
      if (std::isinf(uh))
        return std::signbit(uh) ? -OMEGA : OMEGA;
      return fma_cold(a, b, 0.0); // a tiny nonzero product: never a zero result
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

  template <class Adder>
  struct is_identity_adder : std::false_type
  {
  };
  template <class T>
  struct is_identity_adder<IdentityAdder<T>> : std::true_type
  {
  };

  // FusedMac's step with the FMA rounded to odd, so the format's cast rounds
  // a*b + acc itself. Only over a real Adder: a fused step no format rounds
  // after is the plain FusedMac<IdentityAdder>.
  template <class Adder>
  struct FusedMacOdd
  {
    using value_t = typename Adder::value_t;
    Adder add{};

    CUDA_HOST_DEVICE_INLINE value_t step(value_t a, value_t b, value_t acc, PhiloxEngine &rng) const
    {
      return add(odd::fma(a, b, acc), rng);
    }
  };

  // SplitMac's step with the product rounded to odd before the multiply
  // format's cast, and the sum before the accumulate format's. The
  // multiplier policies multiply their operands themselves, so the odd
  // product is handed to one with 1 as the other factor: exact, and folded
  // away. Over an IdentityAdder no format rounds the sum, which stays the
  // carrier's nearest add: only the product is odd.
  template <class Multiplier, class Adder>
  struct SplitMacOdd
  {
    using value_t = typename Adder::value_t;
    Multiplier mul{};
    Adder add{};

    CUDA_HOST_DEVICE_INLINE value_t step(value_t a, value_t b, value_t acc, PhiloxEngine &rng) const
    {
      const value_t p = mul(odd::mul(a, b), value_t(1), rng);
      if constexpr (is_identity_adder<Adder>::value)
        return acc + p;
      else
        return add(odd::add(acc, p), rng);
    }
  };

  // An existing NAIVE Args with its mac's carrier steps rounded to odd: the
  // custom_matmul_*_odd ops. Like AccumulateArgs (gemm_accumulate.h) it
  // reuses the base's factory, which hands it the NaiveAccumulator to
  // rebuild, so the choice between an Adder and an IdentityAdder stays
  // written once. A FusedMac over an IdentityAdder maps to nothing: the
  // entry points refuse fma_quant = false, so no kernel exists for it.
  template <class Base>
  struct OddArgs
  {
    static constexpr bool mixed = false;
    static constexpr uint64_t draws_per_k_step = Base::draws_per_k_step;

    Base base{};

    template <class T, class F>
    struct Builder
    {
      F &f;

      template <class Adder>
      void operator()(const NaiveAccumulator<FusedMac<Adder>> &naive) const
      {
        if constexpr (!is_identity_adder<Adder>::value)
        {
          using Mac = FusedMacOdd<Adder>;
          f(NaiveAccumulator<Mac>{Mac{naive.mac.add}, T(0)});
        }
      }

      template <class Multiplier, class Adder>
      void operator()(const NaiveAccumulator<SplitMac<Multiplier, Adder>> &naive) const
      {
        using Mac = SplitMacOdd<Multiplier, Adder>;
        f(NaiveAccumulator<Mac>{Mac{naive.mac.mul, naive.mac.add}, T(0)});
      }
    };

    template <class T, RoundMode RM, class F>
    void with_accumulator(F &&f) const
    {
      base.template with_accumulator<T, RM>(Builder<T, F>{f});
    }
  };

} // namespace mptorch::gemm

#undef MPTORCH_NO_CONTRACT
#endif // !__METAL_VERSION__
