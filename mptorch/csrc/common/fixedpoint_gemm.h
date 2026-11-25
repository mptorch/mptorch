#pragma once

// The fixed-point GEMM (phase T of dev/continuation_plan.md): a mac whose
// multiply and accumulate formats are fixed point. For operands a and b in
// the carrier c (binary32 for float32, float16 and bfloat16 operands,
// binary64 for float64 ones), and Q_F the rounding to format F in the call's
// mode, each step of an output element's sum is
//
//     split:  p = Q_M(RN_c(a * b)),  s = Q_A(s + p)
//     fused:                         s = Q_F(RN_c(a * b) + s)
//
// from s = 0, in order, with every sum exact. The product is the carrier's,
// as every mac's is (gemm_policy.h). Each rounding saturates to the format's
// range, or, for the accumulate format of a call that asks for it, wraps
// around modulo 2^wl as two's complement or unsigned hardware does; an
// infinite product saturates either way.
//
// The definition is the integer accumulator's (FixedPointIntMac below): the
// running sum is an int64 code, each step is formed exactly on the finer of
// the two grids and rounded to the accumulate format's grid once, by integer
// shifts. Where the carrier computes every step exactly, the shipped float
// policies (FixedPointMultiplierT, FixedPointAdderT, in SplitMac) compute it
// instead, faster, with the same bits, stochastic rounding included: the
// integer accumulator reads each random word as cast_fixedpoint_stochastic
// does. Which one a call runs is the host's choice, made per call from the
// formats alone (mptorch/number/fixedpoint.py's _fixedpoint_gemm states the
// rule). tests/fixedpoint_gemm_reference.py writes the definition again,
// independently, and dev/benchmarks/sweeps/fixedpoint_int_sweep.cpp holds
// this header's integer rounding to the shipped casts.
//
// cast_fixedpoint.h is not changed, so that the elementwise quantizers'
// objects are what they were; the fast RNE cast below is this header's own.
// The header is valid Metal, like the other common/ headers.

#include "cast_fixedpoint.h"
#include "gemm_policy.h"
#include "modes.h"
#include "philox.h"
#include "stochastic.h"
#if !defined(__METAL_VERSION__)
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>
#endif

namespace mptorch::gemm
{

  struct FixedPointWidths
  {
    int wl = 0, fl = 0;
  };

  // A format's sign, symmetry and, under stochastic rounding, its spec: the
  // random bits and the variant (stochastic.h's sr_spec), which the entry
  // points encode from prng_bits.
  struct FixedPointCommon
  {
    bool is_signed = true;
    bool symmetric = false;
    int prng_bits = 0;
  };

  // --- the fast RNE cast --------------------------------------------------------

  // (x + C) - C with C = 1.5 * 2^(23 - fl) rounds x to the nearest multiple
  // of 2^-fl, ties to even, while |x| <= 2^(22 - fl): the sum then lies in
  // [2^(23 - fl), 2^(24 - fl)], where binary32's step is 2^-fl, and C is an
  // even number of those steps, so the even code is the even sum. The clamp
  // to the range then saturates. A larger |x| rounds to a coarser grid, but
  // stays above every value of a format of at most 22 magnitude bits, so the
  // clamp gives the end of the range; an infinity clamps too, and a NaN
  // passes (every compare with it is false). -0.0 and every value that
  // rounds to zero become +0.0, the difference of two equal positive values.
  // An unsigned format's range starts at 0, so a negative input becomes 0,
  // as the bitwise cast returns it.
  //
  // The gate keeps C and every nonzero result normal binary32 values: fl in
  // [-103, 125]. On the device the operations are __fadd_rn and __fsub_rn.
  // On the host they need binary32 evaluation, with no excess precision,
  // which MPTORCH_FAST_CAST's probe asserts (FLT_EVAL_METHOD == 0,
  // bit_helper.h), and no reassociation; with no multiplication in them,
  // contraction into an FMA cannot reach them. On Metal a subnormal input is
  // flushed to zero, which is its RNE result anyway, since the gate keeps the
  // step at or above 2^-125.
  // dev/benchmarks/sweeps/gemm_cast_fixedpoint_arith.cu holds it to the
  // bitwise cast and to reference_casts.h's ref_fixedpoint over every
  // binary32 input.
  struct FixedPointFast
  {
    bool on;
    float magic; // 1.5 * 2^(23 - fl)
    float lo, hi;
  };

  CUDA_HOST_DEVICE_INLINE FixedPointFast make_fixedpoint_fast(
      int wl, int fl, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<float> &p)
  {
    using F = FloatTraits<float>;
    FixedPointFast f{};
    const int mag_bits = wl - (is_signed ? 1 : 0);
    f.on = mag_bits >= 1 && mag_bits <= 22 && fl >= -103 && fl <= 125;
    f.magic = F::pow2(f.on ? 23 - fl : 0) * 1.5f; // exact: a normal power of two inside the gate
    f.lo = p.min_val;
    f.hi = p.max_val;
    return f;
  }

#if defined(MPTORCH_FAST_CAST)
  CUDA_HOST_DEVICE_INLINE float cast_fixedpoint_rne_fast(float x,
                                                         const MPTORCH_THREAD FixedPointFast &f)
  {
    using F = FloatTraits<float>;
    float q = F::rn_sub(F::rn_add(x, f.magic), f.magic);
    if (q > f.hi)
      q = f.hi;
    if (q < f.lo)
      q = f.lo;
    return q;
  }
#endif

  // A fixed-point format's cast in the carrier, in mode RM, as the float
  // policies round: the fast path under RNE where the gate admits it, the
  // shipped bitwise casts otherwise, and under SR one draw from `rng`.
  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE T fixedpoint_policy_cast(T x, bool is_signed, int spec,
                                                   const MPTORCH_THREAD FixedPointParamsT<T> &p,
                                                   const MPTORCH_THREAD FixedPointFast &fast,
                                                   MPTORCH_THREAD PhiloxEngine &rng)
  {
    if constexpr (RM == RoundMode::SR)
    {
      if constexpr (std::is_same_v<T, float>)
        return cast_fixedpoint_stochastic(x, rng(), spec, is_signed, p);
      else
        return cast_fixedpoint_stochastic(x, rng.next64(), spec, is_signed, p);
    }
    else if constexpr (RM == RoundMode::RNA)
      return cast_fixedpoint_nearest_away(x, is_signed, p);
    else if constexpr (RM == RoundMode::RU)
      return cast_fixedpoint_up(x, is_signed, p);
    else if constexpr (RM == RoundMode::RD)
      return cast_fixedpoint_down(x, is_signed, p);
    else if constexpr (RM == RoundMode::RZ)
      return cast_fixedpoint_zero(x, is_signed, p);
    else if constexpr (RM == RoundMode::RO)
      return cast_fixedpoint_odd(x, is_signed, p);
    else
    {
#if defined(MPTORCH_FAST_CAST)
      if constexpr (std::is_same_v<T, float>)
      {
        if (fast.on)
          return cast_fixedpoint_rne_fast(x, fast);
      }
#endif
      return cast_fixedpoint_nearest_even(x, is_signed, p);
    }
  }

  // --- the float policies: the carrier path ---------------------------------------

  // The product a*b, rounded by the carrier's multiplication and then to the
  // multiply format.
  template <class T, RoundMode RM>
  struct FixedPointMultiplierT
  {
    using value_t = T;
    bool is_signed;
    int spec;
    FixedPointParamsT<T> params;
    FixedPointFast fast;

    FixedPointMultiplierT() = default;
    CUDA_HOST_DEVICE_INLINE FixedPointMultiplierT(const MPTORCH_THREAD FixedPointWidths &w,
                                                  const MPTORCH_THREAD FixedPointCommon &c)
        : is_signed(c.is_signed), spec(c.prng_bits),
          params(make_fixedpoint_params<T>(w.wl, w.fl, c.is_signed, c.symmetric)), fast{}
    {
      if constexpr (std::is_same_v<T, float>)
        fast = make_fixedpoint_fast(w.wl, w.fl, c.is_signed, params);
    }

    CUDA_HOST_DEVICE_INLINE T operator()(T a, T b, MPTORCH_THREAD PhiloxEngine &rng) const
    {
      return fixedpoint_policy_cast<T, RM>(a * b, is_signed, spec, params, fast, rng);
    }
  };

  // A sum, rounded to the accumulate format. Only the carrier path uses it,
  // where the host has found that the carrier holds every sum exactly.
  template <class T, RoundMode RM>
  struct FixedPointAdderT
  {
    using value_t = T;
    bool is_signed;
    int spec;
    FixedPointParamsT<T> params;
    FixedPointFast fast;

    FixedPointAdderT() = default;
    CUDA_HOST_DEVICE_INLINE FixedPointAdderT(const MPTORCH_THREAD FixedPointWidths &w,
                                             const MPTORCH_THREAD FixedPointCommon &c)
        : is_signed(c.is_signed), spec(c.prng_bits),
          params(make_fixedpoint_params<T>(w.wl, w.fl, c.is_signed, c.symmetric)), fast{}
    {
      if constexpr (std::is_same_v<T, float>)
        fast = make_fixedpoint_fast(w.wl, w.fl, c.is_signed, params);
    }

    CUDA_HOST_DEVICE_INLINE T operator()(T x, MPTORCH_THREAD PhiloxEngine &rng) const
    {
      return fixedpoint_policy_cast<T, RM>(x, is_signed, spec, params, fast, rng);
    }
  };

  // --- the integer rounding: the definition -----------------------------------------

  // 2^e in the carrier, built on the word: infinity past the largest
  // exponent, a subnormal or zero below the normals. The integer mac's
  // constants take it, where a wide format's ends can be past binary32's
  // range (the result is then stored as float64, by its own scale).
  template <class T>
  CUDA_HOST_DEVICE_INLINE T fx_pow2(int e)
  {
    using F = FloatTraits<T>;
    using W = typename F::word_t;
    W w;
    if (e > F::MAX_EXP)
      w = F::INF_BITS;
    else if (e >= F::MIN_NORMAL_EXP)
      w = static_cast<W>(e + F::BIAS) << F::MAN_BITS;
    else if (e >= F::MIN_NORMAL_EXP - F::MAN_BITS)
      w = W(1) << (e - F::MIN_NORMAL_EXP + F::MAN_BITS);
    else
      w = 0;
    return reinterpret_cast<const MPTORCH_THREAD T &>(w);
  }

  // v << s and v >> s on int64 values through uint64, so that a negative v is
  // well defined under every language version the header is compiled as;
  // the right shift is arithmetic (a floor).
  CUDA_HOST_DEVICE_INLINE int64_t fx_shl(int64_t v, int s)
  {
    return static_cast<int64_t>(static_cast<uint64_t>(v) << s);
  }

  CUDA_HOST_DEVICE_INLINE int64_t fx_floor_shr(int64_t v, int s)
  {
    return v >= 0 ? static_cast<int64_t>(static_cast<uint64_t>(v) >> s)
                  : -static_cast<int64_t>((static_cast<uint64_t>(-(v + 1)) >> s) + 1);
  }

  // `frac * 2^k`, floored, for k of either sign: 2^k = 2^-s for a right shift.
  CUDA_HOST_DEVICE_INLINE uint64_t fx_scaled_floor(uint64_t frac, int k)
  {
    return k >= 0 ? frac << k : frac >> -k;
  }

  // The random bits of a draw, as cast_fixedpoint_stochastic reads them: the
  // top n bits of the word's mantissa field.
  template <class T>
  CUDA_HOST_DEVICE_INLINE uint64_t fixedpoint_draw_bits(typename FloatTraits<T>::word_t word, int n)
  {
    using F = FloatTraits<T>;
    return static_cast<uint64_t>((word & F::MAN_MASK) >> (F::MAN_BITS - n));
  }

  // Rounds v = x + eps, where x is an integer in units of 2^-(fl + s) and
  // 0 <= eps < 1 (eps > 0 exactly when `sticky`), to an integer in units of
  // 2^-fl, in mode RM. `neg` is the sign of v. Under SR, `r` holds the draw's
  // `n` random bits and `variant` the stochastic mode (0 for StochasticA, 1
  // for B, 2 for C), applied to |v| as P3109's Fig. 2 states them: with
  // X = frac(|v|) * 2^n, round away from zero iff floor(X) + r >= 2^n (A),
  // floor(2X) + 2r + 1 >= 2^(n+1) (B), or RNITE(X) + r >= 2^n (C). A sticky
  // value needs s >= n + 1 under SR, and s >= 1 otherwise, so that the bits
  // eps stands for are below every bit the decision reads; the callers keep
  // their guard bits so.
  template <RoundMode RM>
  CUDA_HOST_DEVICE_INLINE int64_t fixedpoint_round(int64_t x, bool sticky, int s, bool neg,
                                                   uint64_t r, int n, int variant)
  {
    if (s == 0)
      return x; // nothing below the step (callers never pass a sticky value here)
    const int64_t q = fx_floor_shr(x, s);
    const uint64_t rem = static_cast<uint64_t>(x - fx_shl(q, s)); // in [0, 2^s)
    if (rem == 0 && !sticky)
      return q;
    const uint64_t mid = uint64_t(1) << (s - 1); // not `half`, an MSL type
    if constexpr (RM == RoundMode::RNE)
      return q + ((rem > mid || (rem == mid && (sticky || (q & 1)))) ? 1 : 0);
    else if constexpr (RM == RoundMode::RNA)
      return q + ((rem > mid || (rem == mid && (sticky || !neg))) ? 1 : 0);
    else if constexpr (RM == RoundMode::RU)
      return q + 1;
    else if constexpr (RM == RoundMode::RD)
      return q;
    else if constexpr (RM == RoundMode::RZ)
      return q + (neg ? 1 : 0);
    else if constexpr (RM == RoundMode::RO)
      return q | 1;
    else
    {
      // |v|'s integer part and its fraction's bits in units of 2^-s, the
      // fraction's sticky part staying sticky
      int64_t mi;
      uint64_t frac;
      if (!neg)
      {
        mi = q;
        frac = rem;
      }
      else
      {
        // -(x + eps) = (-q - 1) * 2^s + (2^s - rem - eps)
        // (2^s - rem - eps lies in (0, 2^s), since v is inexact here)
        mi = -q - 1;
        frac = (uint64_t(1) << s) - rem - (sticky ? 1 : 0);
      }
      const uint64_t top = uint64_t(1) << n;
      bool up;
      if (variant == 1)
        up = fx_scaled_floor(frac, n + 1 - s) + 2 * r + 1 >= 2 * top;
      else if (variant == 2)
      {
        uint64_t rx;
        if (s <= n)
          rx = frac << (n - s);
        else
        {
          rx = frac >> (s - n);
          const uint64_t low = frac & ((uint64_t(1) << (s - n)) - 1);
          const uint64_t h = uint64_t(1) << (s - n - 1);
          rx += (low > h || (low == h && (sticky || (rx & 1)))) ? 1 : 0;
        }
        up = rx + r >= top;
      }
      else
        up = fx_scaled_floor(frac, n - s) + r >= top;
      const int64_t qm = mi + (up ? 1 : 0);
      return neg ? -qm : qm;
    }
  }

  // The accumulate format's code from a rounded integer: saturated to the
  // range, or wrapped around modulo 2^wl.
  CUDA_HOST_DEVICE_INLINE int64_t fixedpoint_saturate_code(int64_t q, int64_t lo, int64_t hi)
  {
    return q < lo ? lo : (q > hi ? hi : q);
  }

  CUDA_HOST_DEVICE_INLINE int64_t fixedpoint_wrap_code(int64_t q, int wl, bool is_signed)
  {
    const uint64_t mask = (uint64_t(1) << wl) - 1;
    const uint64_t u = static_cast<uint64_t>(q) & mask;
    if (!is_signed || (u >> (wl - 1)) == 0)
      return static_cast<int64_t>(u);
    return static_cast<int64_t>(u) - static_cast<int64_t>(uint64_t(1) << wl);
  }

  // A running sum that a NaN has reached: no code of a format of at most 53
  // magnitude bits is this value.
  constexpr MPTORCH_CONSTANT int64_t FIXEDPOINT_NAN_SUM = static_cast<int64_t>(uint64_t(1) << 63);

  // --- the integer mac -----------------------------------------------------------------

  // One step of an output element's sum, as integers. A split mac rounds the
  // product with the carrier path's own multiplier (the carrier's product,
  // then Q_M, a value on M's grid that becomes an integer code exactly), adds
  // it to the running sum on the finer of the two grids, where the sum is
  // exact, and rounds the sum to A's grid. A fused mac aligns the carrier's
  // product to F's grid with `guard` more bits and a sticky bit, adds it to
  // the running sum, and rounds once. Under SR a split mac draws for the
  // product and then for the sum, as SplitMac does, and a fused mac once.
  //
  // The host's checks bound every intermediate: a split step's exact sum
  // within 62 bits, and a fused step's, with its guard bits, as well. That
  // holds for a fused product below p_top, twice the format's largest
  // magnitude; one at or past it decides the sum's side, since no running
  // sum brings it back into the range, so it saturates the sum before it is
  // aligned, or, under wrap-around, is aligned modulo 2^64 on the word, which
  // keeps its residue modulo 2^wl.
  //
  // WRAP is a template parameter, not a flag: tested at run time in each
  // step it cost a fused SR step 6.7% and a fused RNE step 3% on the RTX 4060.
  template <class T, RoundMode RM, bool FUSED, bool WRAP>
  struct FixedPointIntMac
  {
    using value_t = T;
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    FixedPointMultiplierT<T, RM> mul; // the split mac's Q_M (unused when FUSED)
    int acc_wl, acc_fl;
    bool acc_signed;
    int spec; // the accumulate format's SR spec
    int64_t lo, hi;
    int sh_s, sh_p; // split: the running sum's and the product's shifts onto the finer grid
    T p_scale;      // split: 2^fl_M, a rounded product's code
    int guard;      // fused: the bits kept below F's grid
    T p_top;        // fused: 2^(mag_bits - fl + 1); a product at or past it decides the sum
    T out_scale;    // 2^-fl_A in the carrier
#if !defined(__METAL_VERSION__)
    double out_scale64;
#endif
    bool wide; // the result is stored as float64 (the kernels' wide store)

    CUDA_HOST_DEVICE_INLINE int64_t round_sum(int64_t x, bool sticky, int s, bool neg,
                                              word_t word) const
    {
      const int n = RM == RoundMode::SR ? sr_spec_bits(spec) : 0;
      const uint64_t r = RM == RoundMode::SR ? fixedpoint_draw_bits<T>(word, n) : 0;
      const int64_t q = fixedpoint_round<RM>(x, sticky, s, neg, r, n, sr_spec_variant(spec));
      if constexpr (WRAP)
        return fixedpoint_wrap_code(q, acc_wl, acc_signed);
      else
        return fixedpoint_saturate_code(q, lo, hi);
    }

    CUDA_HOST_DEVICE_INLINE word_t draw(MPTORCH_THREAD PhiloxEngine &rng) const
    {
      if constexpr (RM != RoundMode::SR)
        return 0;
      else if constexpr (std::is_same_v<T, float>)
        return rng();
      else
        return rng.next64();
    }

    CUDA_HOST_DEVICE_INLINE int64_t step(int64_t s, T a, T b,
                                         MPTORCH_THREAD PhiloxEngine &rng) const
    {
      if constexpr (!FUSED)
      {
        const T pq = mul(a, b, rng);
        const word_t word = draw(rng);
        if (s == FIXEDPOINT_NAN_SUM || pq != pq)
          return FIXEDPOINT_NAN_SUM;
        const int64_t p = static_cast<int64_t>(pq * p_scale); // an integer: pq is on M's grid
        const int64_t x = fx_shl(s, sh_s) + fx_shl(p, sh_p);
        return round_sum(x, false, sh_s, x < 0, word);
      }
      else
      {
        const T p = a * b;
        const word_t word = draw(rng);
        if (s == FIXEDPOINT_NAN_SUM || p != p)
          return FIXEDPOINT_NAN_SUM;
        const word_t w = reinterpret_cast<const MPTORCH_THREAD word_t &>(p);
        const bool p_neg = (w & F::SIGN_MASK) != 0;
        const word_t mag = w & F::ABS_MASK;
        const bool huge = !(p < p_top && p > -p_top); // an infinity too
        if (huge && (!WRAP || mag >= F::INF_BITS))
          return p_neg ? lo : hi; // the sum saturates on the product's side
        // p = m * 2^e on the word
        const int field = static_cast<int>(mag >> F::MAN_BITS);
        const int64_t m = static_cast<int64_t>(
            field == 0 ? (mag & F::MAN_MASK) : ((mag & F::MAN_MASK) | (word_t(1) << F::MAN_BITS)));
        const int e = (field == 0 ? 1 : field) - F::BIAS - F::MAN_BITS;
        const int sh = e + acc_fl + guard; // m's shift onto the grid 2^-(fl + guard)
        int64_t pa;
        bool sticky = false;
        if (sh >= 0)
          pa = sh >= 64 ? 0 : fx_shl(m, sh); // modulo 2^64: only wrap-around reaches a large one
        else
        {
          const int d = -sh;
          const int64_t kept = d >= 64 ? 0 : (m >> d);
          sticky = d >= 64 ? m != 0 : (m - (kept << d)) != 0;
          pa = kept;
        }
        // the magnitude's floor and sticky part, signed: -(pa + eps) is
        // floor -pa - 1 with the complementary sticky part. The sum is taken
        // modulo 2^64, on uint64, since under wrap-around pa can be 2^63.
        const uint64_t base = static_cast<uint64_t>(fx_shl(s, guard));
        const uint64_t ux = !p_neg ? base + static_cast<uint64_t>(pa)
                                   : base - static_cast<uint64_t>(pa) - (sticky ? 1u : 0u);
        const int64_t x = static_cast<int64_t>(ux);
        // the sum's sign: the product's when it dominates (wrap-around past
        // the range, where x is only known modulo 2^64), x's own otherwise
        const bool neg = huge ? p_neg : x < 0;
        return round_sum(x, sticky, guard, neg, word);
      }
    }

    // The element's value in the carrier, or NaN; rounded to the carrier when
    // the format is wider than it (a wide result is stored by value64).
    CUDA_HOST_DEVICE_INLINE T value(int64_t s) const
    {
      if (s == FIXEDPOINT_NAN_SUM)
      {
        const word_t nan_bits = F::INF_BITS | F::TIE;
        return reinterpret_cast<const MPTORCH_THREAD T &>(nan_bits);
      }
      return static_cast<T>(s) * out_scale;
    }

#if !defined(__METAL_VERSION__)
    CUDA_HOST_DEVICE_INLINE double value64(int64_t s) const
    {
      using D = FloatTraits<double>;
      if (s == FIXEDPOINT_NAN_SUM)
      {
        const typename D::word_t nan_bits = D::INF_BITS | D::TIE;
        return reinterpret_cast<const double &>(nan_bits);
      }
      return static_cast<double>(s) * out_scale64;
    }
#endif
  };

  // --- the integer accumulator ------------------------------------------------------------

#if !defined(__METAL_VERSION__)
  // The CPU kernel's per-element state for FixedPointIntAccumulator: int64
  // codes and the SR streams, as NaiveTile holds float sums. The kernel
  // reads results through the accumulator's static `result`, which takes the
  // Mac's scale, rather than through a finalize of its own.
  struct FixedPointIntTileView
  {
    int64_t *__restrict__ sums;
    PhiloxEngine *__restrict__ rng;
  };

  struct FixedPointIntTile
  {
    std::vector<int64_t> sums;
    std::vector<PhiloxEngine> streams;

    void resize(int64_t n)
    {
      sums.resize(static_cast<size_t>(n));
      streams.resize(static_cast<size_t>(n));
    }
    FixedPointIntTileView view() { return {sums.data(), streams.data()}; }
    void begin(int64_t n) { std::fill_n(sums.data(), static_cast<size_t>(n), int64_t(0)); }
    void seed_rng(int64_t idx, uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
      PhiloxEngine &e = streams[static_cast<size_t>(idx)];
      e.reset_state(seed, subsequence);
      e.set_offset(offset);
    }
  };
#endif

  // An output element's sum over a FixedPointIntMac: the object form, one
  // GPU thread's state, and the CPU tile form. Its result is read through
  // `finalize` (the carrier's value) or, for a wide format, `finalize_wide`
  // (float64): the kernels store which one `mac.wide` says (has_int_result_v).
  template <class Mac>
  struct FixedPointIntAccumulator
  {
    using mac_type = Mac;
    using value_t = typename Mac::value_t;
#if !defined(__METAL_VERSION__)
    using tile_type = FixedPointIntTile;
#endif
    // read by has_int_result_v, through the policies that derive from an
    // accumulator too (the conv ops' Gathered)
    static constexpr MPTORCH_CONSTANT bool int_result = true;

    Mac mac{};
    int64_t sum = 0;
    PhiloxEngine rng{};

    CUDA_HOST_DEVICE_INLINE void seed_rng(uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
      rng.reset_state(seed, subsequence);
      rng.set_offset(offset);
    }
    CUDA_HOST_DEVICE_INLINE void accumulate(value_t a, value_t b)
    {
      sum = mac.step(sum, a, b, rng);
    }
    CUDA_HOST_DEVICE_INLINE value_t finalize() const { return mac.value(sum); }

#if !defined(__METAL_VERSION__)
    CUDA_HOST_DEVICE_INLINE double finalize_wide() const { return mac.value64(sum); }

    static inline void accumulate(const FixedPointIntTileView &t, int64_t idx, const Mac &m,
                                  value_t a, value_t b)
    {
      t.sums[idx] = m.step(t.sums[idx], a, b, t.rng[idx]);
    }
    static inline value_t result(const FixedPointIntTile &t, int64_t idx, const Mac &m)
    {
      return m.value(t.sums[static_cast<size_t>(idx)]);
    }
    static inline double result_wide(const FixedPointIntTile &t, int64_t idx, const Mac &m)
    {
      return m.value64(t.sums[static_cast<size_t>(idx)]);
    }
#endif
  };

  // Whether an accumulator is a FixedPointIntAccumulator, or a policy that
  // derives from one (the conv ops' Gathered), whose results the kernels read
  // through the accumulator's own functions: the Mac's scale, and the wide
  // store. On the host and CUDA it reads int_result, which a derived policy
  // inherits, as is_positional_accumulator_v reads `positional`. Metal has no
  // derived classes and no `requires`, so there it is a specialization.
#if defined(__METAL_VERSION__)
  template <class A>
  struct has_int_result
  {
    static constexpr MPTORCH_CONSTANT bool value = false;
  };

  template <class Mac>
  struct has_int_result<FixedPointIntAccumulator<Mac>>
  {
    static constexpr MPTORCH_CONSTANT bool value = true;
  };

  template <class A>
  constexpr MPTORCH_CONSTANT bool has_int_result_v = has_int_result<A>::value;
#else
  template <class A>
  constexpr bool has_int_result_v = requires { A::int_result; };
#endif

  // --- the Args ---------------------------------------------------------------------

  // The carrier path: a split mac over the float policies, NAIVE.
  struct FixedPointSplitArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 2;

    FixedPointWidths mul{}, acc{};
    FixedPointCommon mul_c{}, acc_c{};

    template <class T, RoundMode RM, class Fn>
    void with_accumulator(MPTORCH_THREAD Fn &&f) const
    {
      using Mac = SplitMac<FixedPointMultiplierT<T, RM>, FixedPointAdderT<T, RM>>;
      f(NaiveAccumulator<Mac>{
          Mac{FixedPointMultiplierT<T, RM>(mul, mul_c), FixedPointAdderT<T, RM>(acc, acc_c)},
          T(0)});
    }
  };

  // The integer path, split or fused. WRAP asks for wrap-around on the
  // accumulate format's overflow, `wide` for the result as float64 (read
  // once per element, so a flag).
  template <bool FUSED, bool WRAP>
  struct FixedPointIntArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = FUSED ? 1 : 2;

    FixedPointWidths mul{}, acc{}; // mul unused when FUSED
    FixedPointCommon mul_c{}, acc_c{};
    bool wide = false;

    template <class T, RoundMode RM>
    CUDA_HOST_DEVICE_INLINE FixedPointIntMac<T, RM, FUSED, WRAP> make_mac() const
    {
      FixedPointIntMac<T, RM, FUSED, WRAP> m{};
      if constexpr (!FUSED)
        m.mul = FixedPointMultiplierT<T, RM>(mul, mul_c);
      const int mag = acc.wl - (acc_c.is_signed ? 1 : 0);
      m.acc_wl = acc.wl;
      m.acc_fl = acc.fl;
      m.acc_signed = acc_c.is_signed;
      m.spec = acc_c.prng_bits;
      if (!acc_c.is_signed)
      {
        m.lo = 0;
        m.hi = static_cast<int64_t>((uint64_t(1) << acc.wl) - 1);
      }
      else
      {
        m.hi = static_cast<int64_t>((uint64_t(1) << mag) - 1);
        m.lo = acc_c.symmetric ? -m.hi : -m.hi - 1;
      }
      const int g = mul.fl > acc.fl ? mul.fl : acc.fl; // the finer grid
      m.sh_s = g - acc.fl;
      m.sh_p = g - mul.fl;
      m.p_scale = fx_pow2<T>(mul.fl);
      const int n = sr_spec_bits(acc_c.prng_bits);
      m.guard = RM == RoundMode::SR ? n + 1 : 2;
      m.p_top = fx_pow2<T>(mag - acc.fl + 1);
      m.out_scale = fx_pow2<T>(-acc.fl);
#if !defined(__METAL_VERSION__)
      m.out_scale64 = fx_pow2<double>(-acc.fl);
#endif
      m.wide = wide;
      return m;
    }

    template <class T, RoundMode RM, class Fn>
    void with_accumulator(MPTORCH_THREAD Fn &&f) const
    {
      using Mac = FixedPointIntMac<T, RM, FUSED, WRAP>;
      f(FixedPointIntAccumulator<Mac>{make_mac<T, RM>(), 0});
    }
  };

  template <bool WRAP>
  using FixedPointIntSplitArgs = FixedPointIntArgs<false, WRAP>;
  template <bool WRAP>
  using FixedPointIntFusedArgs = FixedPointIntArgs<true, WRAP>;

} // namespace mptorch::gemm
