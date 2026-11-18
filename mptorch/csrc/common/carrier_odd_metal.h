#pragma once

// The carrier step rounded to odd (RO) on an Apple GPU: carrier_odd.h's
// three primitives, odd::fma, odd::add and odd::mul, in binary32, for Metal
// only. carrier_odd.h includes this header in place of its host and CUDA
// primitives when __METAL_VERSION__ is defined, and nothing else includes
// it, so no host or CUDA object sees it.
//
// The host computes binary32 RO in binary64, and CUDA with a pair of
// operations in the two directed rounding modes. Metal has neither, and its
// float unit flushes binary32 subnormals to zero, operands and results, in
// every operation and compare. The flush is decided on the exact result,
// before rounding: metal::fma(2^-63 (1 - 2^-24), 2^-63, 0), whose nearest
// binary32 value is 2^-126, returns 0 (dev/benchmarks/mps_backend.py
// odd-probe). So each primitive is built from error-free transforms in
// binary32 (sequences of float operations whose rounding errors are
// recovered exactly), run only on operands that a test on their bits has
// admitted; every other input is first rescaled on its bits into that
// range. A primitive has three parts:
//
//   core   fma: TwoProd (p = a*b, e = fma(a, b, -p)), 2Sum(p, c), 2Sum(e1,
//          e), Fast2Sum(s1, zn), then the odd fix of S by tau = T + zt
//          (fix below). add: 2Sum, then the fix. mul: TwoProd, then the
//          fix. This is the host's binary64 fma_fast, written in binary32.
//   gate   a test on the bits, before the core, that sends an input to the
//          cold path when: the product of nonzero a and b has |p| < 2^-78
//          (which also catches a p the flush made zero, including the
//          product of a subnormal factor) or |p| >= 2^125 (an infinite or
//          NaN p included; for mul, only an infinite or NaN p); 0 < |c| <
//          2^-100 or |c| >= 2^125; and, for add, an operand is nonzero and
//          below 2^-103, or at least 2^125. Inside these bounds TwoProd's
//          error is a multiple of 2^(E_p - 47) >= 2^-125, and each 2Sum's a
//          multiple of the smaller operand's last place, >= 2^-123 (2^-126
//          for add, which leaves no slack). So every intermediate value,
//          including every exact one, is zero or at least 2^-126 in
//          magnitude, and none overflows: the flush cannot reach them.
//   cold   every other input. Zero products, zero terms and non-finite
//          operands are answered first, from the bits. The rest are
//          rescaled by a power of two on the bits so that the core's bounds
//          hold, run through a core of their own, and scaled back on the
//          bits. The rescaling is exact where the product and c are within
//          60 binades of each other; past that, the smaller term is replaced
//          by +-2^-100, a stand-in too small to reach the other term's last
//          place, which decides only the sign of the residual, the one
//          thing that term could change. Scaling back moves the result onto
//          the 2^-149 grid with the dropped bits ORed into the last one
//          (round to odd composes this way), and gives +-Omega (the largest
//          finite value) past the top of the range.
//
// Every primitive returns the results carrier_odd.h specifies, word for word
// with the host's binary32 primitives (NaN payloads aside), which
// tests/test_carrier_rounding.py and mps_backend.py's odd-sweep check.
//
// Also here, for the odd macs' operations that no format rounds next (the
// sums over an IdentityAdder, KAHAN's differences that no format rounds, a
// fold with no outer format): add_rn and sub_rn, round-to-nearest operations
// that the flush cannot reach (through bit_helper.h's metal_sub_rn); and
// cast_product, a multiplier's cast of a product already formed, which
// SplitMacOdd and OddProduct call instead of handing the multiplier the odd
// product times 1. These are in this header rather than in bit_helper.h,
// which every MPS kernel includes, so that no other kernel's source
// changes.

#include "gemm_policy.h"

#ifndef MPTORCH_ODD_COLD
#define MPTORCH_ODD_COLD __attribute__((noinline))
#endif

namespace mptorch::gemm
{
  namespace odd
  {
    constant uint32_t SIGN_BIT = 0x80000000u;
    constant uint32_t MANT_BITS = 0x007FFFFFu;
    constant uint32_t INF_WORD = 0x7F800000u;
    constant uint32_t OMEGA_WORD = 0x7F7FFFFFu;
    // The magnitude words of the gate's bounds (2^e is (e + 127) << 23).
    constant uint32_t TINY_PRODUCT = 49u << 23; // 2^-78
    constant uint32_t TINY_TERM = 27u << 23;    // 2^-100
    constant uint32_t TINY_ADD = 24u << 23;     // 2^-103
    constant uint32_t HUGE_TERM = 252u << 23;   // 2^125

    inline uint32_t word(float x) { return as_type<uint32_t>(x); }
    inline float float_of(uint32_t w) { return as_type<float>(w); }
    inline bool is_subnormal_magnitude(uint32_t m) { return m != 0 && m < 0x00800000u; }

    // The gate's test of one magnitude word m: whether it is outside
    // [lo, hi), in one unsigned compare. A zero term, which the cores take
    // as it is, is passed in as lo.
    inline bool in_gap(uint32_t m, uint32_t lo, uint32_t hi) { return m - lo >= hi - lo; }

    // The magnitude the gate tests for a product a * b, given |a|, |b| and
    // the GPU's |RN(a * b)| as words. It is |RN(a * b)|, except that a zero
    // factor makes the product an exact zero (or IEEE's NaN, with an
    // infinity), which the cores take as it is, so it passes. A subnormal
    // factor next to a nonzero one needs no test of its own: the GPU flushes
    // it, as an operand and in the product, even where the exact product is
    // normal (2^-130 * 2^100 is 0, measured with odd-probe's kernels on an M3
    // Pro, and checked by tests/test_mps.py's
    // test_the_odd_step_does_not_flush). So the product is a zero of nonzero
    // factors, which fails the gate.
    inline uint32_t product_magnitude(uint32_t A, uint32_t B, uint32_t P)
    {
      return metal::min(A, B) == 0 ? TINY_PRODUCT : P;
    }

    // The odd fix (carrier_odd.h's header describes it): s is made odd when
    // tau, the rounding error, is nonzero, stepping toward zero first if tau
    // has the opposite sign. tau's sign and zeroness are read from its word,
    // since a compare would see a flushed tau as zero.
    inline float fix(float s, float tau)
    {
      const uint32_t w = word(s), t = word(tau);
      const uint32_t inx = (t << 1) != 0 ? 1u : 0u;
      const uint32_t opp = ((w ^ t) >> 31) & inx;
      return float_of((w - opp) | inx);
    }

    // 2Sum: s + t = a + b exactly, whatever the order of a and b.
    inline void two_sum(float a, float b, thread float &s, thread float &t)
    {
      s = a + b;
      const float bb = s - a;
      t = (a - (s - bb)) + (b - bb);
    }

    // Fast2Sum, for |a| >= |b| (or a = 0): s + t = a + b exactly.
    inline void fast_two_sum(float a, float b, thread float &s, thread float &t)
    {
      s = a + b;
      t = b - (s - a);
    }

    // The cores, for operands inside the gate's bounds. An exact zero result
    // takes s1's IEEE sign, as the host's fma_fast does (fma(-0, 5, -0) is
    // -0).
    inline float fma_core(float a, float b, float c)
    {
      const float uh = a * b;
      const float ul = metal::fma(a, b, -uh);
      float s1, e1, zn, zt, S, T;
      two_sum(uh, c, s1, e1);
      two_sum(e1, ul, zn, zt);
      fast_two_sum(s1, zn, S, T);
      const float r = fix(S, T + zt);
      return ((word(S) | word(s1)) << 1) == 0 ? s1 : r;
    }

    inline float add_core(float a, float b)
    {
      float s, t;
      two_sum(a, b, s, t);
      return fix(s, t);
    }

    inline float mul_core(float a, float b)
    {
      const float uh = a * b;
      return fix(uh, metal::fma(a, b, -uh));
    }

    // A finite nonzero magnitude word, written as sig * 2^(e - 23) with sig in
    // [2^23, 2^24); a subnormal is normalized.
    inline void split(uint32_t m, thread uint32_t &sig, thread int &e)
    {
      const uint32_t field = m >> 23;
      if (field != 0)
      {
        sig = (m & MANT_BITS) | 0x00800000u;
        e = int(field) - 127;
      }
      else
      {
        const int l = 31 - int(metal::clz(m));
        sig = m << uint32_t(23 - l);
        e = l - 149;
      }
    }

    // The value sign * sig * 2^(e - 23), for e in binary32's normal range.
    inline float build(uint32_t sign, uint32_t sig, int e)
    {
      return float_of(sign | (uint32_t(e + 127) << 23) | (sig & MANT_BITS));
    }

    // A term d binades from the larger one: itself, scaled exactly, when it
    // is within 60 binades; past that, a stand-in of its sign, 2^-100, which
    // no last place of a term in [1, 4) can see.
    inline float scaled_term(uint32_t sign, uint32_t sig, int d)
    {
      return d < -60 ? build(sign, 0x00800000u, -100) : build(sign, sig, d);
    }

    // RO of y * 2^S, where y = RO(x * 2^-S) is what a core returned for the
    // rescaled operands: zero (an exact cancellation) or a normal value.
    inline float descale(float y, int S)
    {
      const uint32_t w = word(y), m = w & ~SIGN_BIT, sign = w & SIGN_BIT;
      if (m == 0)
        return y;
      const int field = int(m >> 23) + S;
      if (field >= 255)
        return float_of(sign | OMEGA_WORD); // RO's overflow: Omega, never infinity
      if (field >= 1)
        return float_of(sign | (uint32_t(field) << 23) | (m & MANT_BITS));
      // Onto the subnormal grid: y's 24 significand bits shifted right by
      // 1 - field, with the dropped bits (y's own odd bit among them) ORed
      // into the last one kept. A shift of 1 leaves at most 2^23 - 1, so it
      // never carries into 2^-126: an inexact value is never an even word.
      const uint32_t sig = (m & MANT_BITS) | 0x00800000u;
      const int shift = 1 - field;
      if (shift >= 25)
        return float_of(sign | 1u);
      const uint32_t dropped = sig & ((1u << uint32_t(shift)) - 1u);
      return float_of(sign | (sig >> uint32_t(shift)) | (dropped != 0 ? 1u : 0u));
    }

    // The cold paths, each a whole primitive, with a core of its own, for
    // the inputs its gate sends there. Cold paths that rescaled the operands
    // and then rejoined the fast path's core (one core per primitive) made
    // the split and KAHAN GEMMs 1.1-1.3x slower at 1024^3 on an M3 Pro
    // (2026-11-15; their K-loops are unrolled), and declaring the cold paths
    // noinline made the fused GEMM 1.07x faster than inlining them.
    MPTORCH_ODD_COLD float fma_cold(float a, float b, float c)
    {
      const uint32_t wa = word(a), wb = word(b), wc = word(c);
      const uint32_t A = wa & ~SIGN_BIT, B = wb & ~SIGN_BIT, C = wc & ~SIGN_BIT;
      if (A >= INF_WORD || B >= INF_WORD || C >= INF_WORD)
      {
        // IEEE's result. A subnormal factor of an infinity counts as +-1
        // (the product is that infinity), where the flush would make it
        // 0 * inf, a NaN.
        const float a2 = is_subnormal_magnitude(A) ? float_of((wa & SIGN_BIT) | 0x3F800000u) : a;
        const float b2 = is_subnormal_magnitude(B) ? float_of((wb & SIGN_BIT) | 0x3F800000u) : b;
        return metal::fma(a2, b2, c);
      }
      if (A == 0 || B == 0)
      {
        // A zero product: the sum is c, exactly, even a subnormal c; two
        // zero terms give -0 only from -0 + -0.
        if (C != 0)
          return c;
        return float_of((wa ^ wb) & wc & SIGN_BIT);
      }
      uint32_t sa, sb;
      int ea, eb;
      split(A, sa, ea);
      split(B, sb, eb);
      const uint32_t sab = (wa ^ wb) & SIGN_BIT;
      const int E = ea + eb; // a * b = a1 * b1 * 2^E, |a1 * b1| in [1, 4)
      float a1 = build(wa & SIGN_BIT, sa, 0), b1 = build(wb & SIGN_BIT, sb, 0), c1 = c;
      int S = E;
      if (C != 0)
      {
        uint32_t sc;
        int ec;
        split(C, sc, ec);
        const int d = ec - E;
        if (d > 60)
        {
          // The product is far below c's last place: c is scaled into
          // [1, 2), and the product replaced by a product of its sign that
          // is exact and just as negligible.
          c1 = build(wc & SIGN_BIT, sc, 0);
          a1 = build(sab, 0x00800000u, -50);
          b1 = build(0u, 0x00800000u, -50);
          S = ec;
        }
        else
          c1 = scaled_term(wc & SIGN_BIT, sc, d);
      }
      return descale(fma_core(a1, b1, c1), S);
    }

    MPTORCH_ODD_COLD float add_cold(float a, float b)
    {
      const uint32_t wa = word(a), wb = word(b);
      const uint32_t A = wa & ~SIGN_BIT, B = wb & ~SIGN_BIT;
      if (A >= INF_WORD || B >= INF_WORD)
        return a + b; // IEEE's: a flushed subnormal cannot change it
      if (A == 0)
        return B == 0 ? float_of(wa & wb) : b; // -0 only from -0 + -0
      if (B == 0)
        return a;
      uint32_t sa, sb;
      int ea, eb;
      split(A, sa, ea);
      split(B, sb, eb);
      const int E = ea > eb ? ea : eb;
      return descale(
          add_core(scaled_term(wa & SIGN_BIT, sa, ea - E), scaled_term(wb & SIGN_BIT, sb, eb - E)),
          E);
    }

    MPTORCH_ODD_COLD float mul_cold(float a, float b)
    {
      const uint32_t wa = word(a), wb = word(b);
      const uint32_t A = wa & ~SIGN_BIT, B = wb & ~SIGN_BIT;
      const uint32_t sign = (wa ^ wb) & SIGN_BIT;
      if (A > INF_WORD || B > INF_WORD)
        return a * b; // a NaN
      if (A == INF_WORD || B == INF_WORD)
        return (A == 0 || B == 0) ? a * b : float_of(sign | INF_WORD); // inf * 0 is NaN
      if (A == 0 || B == 0)
        return float_of(sign);
      uint32_t sa, sb;
      int ea, eb;
      split(A, sa, ea);
      split(B, sb, eb);
      return descale(mul_core(build(wa & SIGN_BIT, sa, 0), build(wb & SIGN_BIT, sb, 0)), ea + eb);
    }

    inline float fma(float a, float b, float c)
    {
      const uint32_t A = word(a) & ~SIGN_BIT, B = word(b) & ~SIGN_BIT, C = word(c) & ~SIGN_BIT;
      const uint32_t P = word(a * b) & ~SIGN_BIT;
      const bool cold = in_gap(product_magnitude(A, B, P), TINY_PRODUCT, HUGE_TERM) |
                        in_gap(C == 0 ? TINY_TERM : C, TINY_TERM, HUGE_TERM);
      if (cold)
        return fma_cold(a, b, c);
      return fma_core(a, b, c);
    }

    inline float add(float a, float b)
    {
      const uint32_t A = word(a) & ~SIGN_BIT, B = word(b) & ~SIGN_BIT;
      const bool cold = in_gap(A == 0 ? TINY_ADD : A, TINY_ADD, HUGE_TERM) |
                        in_gap(B == 0 ? TINY_ADD : B, TINY_ADD, HUGE_TERM);
      if (cold)
        return add_cold(a, b);
      return add_core(a, b);
    }

    inline float mul(float a, float b)
    {
      const uint32_t A = word(a) & ~SIGN_BIT, B = word(b) & ~SIGN_BIT;
      const uint32_t P = word(a * b) & ~SIGN_BIT;
      const bool cold = in_gap(product_magnitude(A, B, P), TINY_PRODUCT, INF_WORD);
      if (cold)
        return mul_cold(a, b);
      return mul_core(a, b);
    }

    // The round-to-nearest operations that no format rounds next, written so
    // that the flush cannot reach them: x + y and x - y as IEEE binary32
    // gives them, subnormals included. add_rn negates y on its word.
    inline float sub_rn(float x, float y) { return metal_sub_rn(x, y); }
    inline float add_rn(float x, float y) { return metal_sub_rn(x, float_of(word(y) ^ SIGN_BIT)); }

    // A multiplier's cast of a product already formed: the multiplier's
    // operator() body without its `a * b`. Without it the odd macs would
    // reach the cast as the odd product times 1, and a subnormal product
    // times 1 is a product the flush may reach (odd-probe found the
    // multiplication folded away, but nothing promises that).
    template <class T, RoundMode RM>
    inline T cast_product(const thread BinaryKMultiplierT<T, RM> &m, T x, thread PhiloxEngine &)
    {
      if constexpr (RM == RoundMode::RNA)
        return cast_binaryK_nearest_away(x, m.is_signed, m.subnormals_mode, m.params);
      else if constexpr (RM == RoundMode::RU)
        return cast_binaryK_up(x, m.is_signed, m.subnormals_mode, m.params);
      else if constexpr (RM == RoundMode::RD)
        return cast_binaryK_down(x, m.is_signed, m.subnormals_mode, m.params);
      else if constexpr (RM == RoundMode::RZ)
        return cast_binaryK_zero(x, m.is_signed, m.subnormals_mode, m.params);
      else if constexpr (RM == RoundMode::RO)
        return cast_binaryK_odd(x, m.is_signed, m.subnormals_mode, m.params);
      else // RoundMode::RNE
        return cast_binaryK_nearest_even(x, m.is_signed, m.subnormals_mode, m.params);
    }

    template <class T>
    inline T cast_product(const thread BinaryKMultiplierT<T, RoundMode::SR> &m, T x,
                          thread PhiloxEngine &rng)
    {
      return cast_binaryK_stochastic(x, rng(), m.sr, m.is_signed, m.subnormals_mode, m.params);
    }

    template <class T, RoundMode RM>
    inline T cast_product(const thread SuperfpMultiplierT<T, RM> &m, T x, thread PhiloxEngine &)
    {
      if constexpr (RM == RoundMode::RNA)
        return cast_superfp_nearest_away(x, m.is_signed, m.params);
      else if constexpr (RM == RoundMode::RU)
        return cast_superfp_up(x, m.is_signed, m.params);
      else if constexpr (RM == RoundMode::RD)
        return cast_superfp_down(x, m.is_signed, m.params);
      else if constexpr (RM == RoundMode::RZ)
        return cast_superfp_zero(x, m.is_signed, m.params);
      else if constexpr (RM == RoundMode::RO)
        return cast_superfp_odd(x, m.is_signed, m.params);
      else // RoundMode::RNE
        return cast_superfp_nearest_even(x, m.is_signed, m.params);
    }

    template <class T>
    inline T cast_product(const thread SuperfpMultiplierT<T, RoundMode::SR> &m, T x,
                          thread PhiloxEngine &rng)
    {
      return cast_superfp_stochastic(x, rng(), m.sr, m.is_signed, m.params);
    }
  } // namespace odd
} // namespace mptorch::gemm
