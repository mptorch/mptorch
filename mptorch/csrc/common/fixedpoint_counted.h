#pragma once

// The counted fixed-point GEMMs: gemm_counted.h's counting over the
// fixed-point GEMM (fixedpoint_gemm.h), on either of its paths.
//
// The carrier path is a SplitMac of the float policies, which Counted wraps
// as it wraps the other families'; its thresholds are the formats' ends.
//
// The integer accumulator's sum is not a cast of a carrier value: its step
// forms the exact sum on a finer grid and rounds it to the accumulate
// format with integer shifts (FixedPointIntMac::step). CountedFixedPointIntMac
// is that step again, statement for statement, counting both casts: the
// split mac's product, a carrier cast classified by event_counts.h's table,
// and the sum, classified on the integers by sum_bits with the same table:
//   overflow   the exact sum past the format's end on its side (for an
//              unsigned format, past its top in magnitude), or a finite
//              product so large that it decides the sum
//   underflow  a nonzero exact sum rounded to zero
//   inexact    a result other than the exact sum (rounded, saturated or
//              wrapped)
//   nan        a NaN sum
// No fixed-point result is a subnormal. The copy's results are held to
// FixedPointIntMac's, word for word, by tests/test_counted_gemm.py.
//
// Both paths accumulate through the positional contract, for gemm_counted.h's
// reason: CUDA's NAIVE path runs the zero steps that pad the last slab.

#include "fixedpoint_gemm.h"
#include "gemm_counted.h"

namespace mptorch::gemm
{
  template <class T>
  CUDA_HOST_DEVICE_INLINE events::EventThresholdsT<T> thresholds_of(const MPTORCH_THREAD FixedPointWidths &w,
                                                                    const MPTORCH_THREAD FixedPointCommon &c)
  {
    const FixedPointParamsT<T> p = make_fixedpoint_params<T>(w.wl, w.fl, c.is_signed, c.symmetric);
    return events::fixedpoint_thresholds<T>(p, c.is_signed);
  }

#if defined(__METAL_VERSION__)
  // CountedFixedPointIntMac on Metal, which has no derived classes: the
  // integer mac as a member, `m`, and the host's statements below with `m.`
  // for `this->` (and the fused step's mantissa named `mant`), except one:
  // the fused step's test of a huge product is on the words, as in
  // FixedPointIntMac::step's Metal arm, since p_top can be a binary32
  // subnormal, which the GPU's compare flushes.
  template <class T, RoundMode RM, bool FUSED, bool WRAP>
  struct CountedFixedPointIntMac
  {
    using value_t = T;
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    FixedPointIntMac<T, RM, FUSED, WRAP> m;
    events::EventThresholdsT<T> mul_th{};
    mutable events::EventTally mul_tally{};
    mutable events::EventTally acc_tally{};

    uint32_t sum_bits(int64_t x, bool sticky, int sh, int64_t q) const
    {
      const int64_t top = fx_shl(m.hi, sh);
      const int64_t bottom = m.acc_signed ? fx_shl(m.lo, sh) : -top;
      uint32_t bits = 1u << events::CASTS;
      bits |= uint32_t(x > top || (x == top && sticky) || x < bottom) << events::OVERFLOW;
      bits |= uint32_t((x != 0 || sticky) && q == 0) << events::UNDERFLOW;
      bits |= uint32_t(sticky || fx_shl(q, sh) != x) << events::INEXACT;
      return bits;
    }

    int64_t step(int64_t s, T a, T b, MPTORCH_THREAD PhiloxEngine &rng) const
    {
      constexpr uint32_t NAN_SUM_BITS = (1u << events::CASTS) | (1u << events::NAN_RESULT);
      if constexpr (!FUSED)
      {
        const T pq = m.mul(a, b, rng);
        count_cast(mul_tally, events::MUL, events::event_bits<T>(a * b, pq, mul_th));
        const word_t word = m.draw(rng);
        if (s == FIXEDPOINT_NAN_SUM || pq != pq)
        {
          count_cast(acc_tally, events::ACC, NAN_SUM_BITS);
          return FIXEDPOINT_NAN_SUM;
        }
        const int64_t p = static_cast<int64_t>(pq * m.p_scale); // an integer: pq is on M's grid
        const int64_t x = fx_shl(s, m.sh_s) + fx_shl(p, m.sh_p);
        const int64_t q = m.round_sum(x, false, m.sh_s, x < 0, word);
        count_cast(acc_tally, events::ACC, sum_bits(x, false, m.sh_s, q));
        return q;
      }
      else
      {
        const T p = a * b;
        const word_t word = m.draw(rng);
        if (s == FIXEDPOINT_NAN_SUM || p != p)
        {
          count_cast(acc_tally, events::ACC, NAN_SUM_BITS);
          return FIXEDPOINT_NAN_SUM;
        }
        const word_t w = reinterpret_cast<const MPTORCH_THREAD word_t &>(p);
        const bool p_neg = (w & F::SIGN_MASK) != 0;
        const word_t mag = w & F::ABS_MASK;
        const bool huge = mag >= reinterpret_cast<const MPTORCH_THREAD word_t &>(m.p_top);
        if (huge && (!WRAP || mag >= F::INF_BITS))
        {
          const int64_t q = p_neg ? m.lo : m.hi;
          const bool finite = mag < F::INF_BITS;
          count_cast(acc_tally, events::ACC,
                     (1u << events::CASTS) | (uint32_t(finite) << events::OVERFLOW) |
                         (uint32_t(finite && q == 0) << events::UNDERFLOW) | (1u << events::INEXACT));
          return q;
        }
        const int field = static_cast<int>(mag >> F::MAN_BITS);
        const int64_t mant = static_cast<int64_t>(
            field == 0 ? (mag & F::MAN_MASK) : ((mag & F::MAN_MASK) | (word_t(1) << F::MAN_BITS)));
        const int e = (field == 0 ? 1 : field) - F::BIAS - F::MAN_BITS;
        const int sh = e + m.acc_fl + m.guard;
        int64_t pa;
        bool sticky = false;
        if (sh >= 0)
          pa = sh >= 64 ? 0 : fx_shl(mant, sh);
        else
        {
          const int d = -sh;
          const int64_t kept = d >= 64 ? 0 : (mant >> d);
          sticky = d >= 64 ? mant != 0 : (mant - (kept << d)) != 0;
          pa = kept;
        }
        const uint64_t base = static_cast<uint64_t>(fx_shl(s, m.guard));
        const uint64_t ux = !p_neg ? base + static_cast<uint64_t>(pa)
                                   : base - static_cast<uint64_t>(pa) - (sticky ? 1u : 0u);
        const int64_t x = static_cast<int64_t>(ux);
        const bool neg = huge ? p_neg : x < 0;
        const int64_t q = m.round_sum(x, sticky, m.guard, neg, word);
        count_cast(acc_tally, events::ACC,
                   huge ? (1u << events::CASTS) | (1u << events::OVERFLOW) | (1u << events::INEXACT) |
                              (uint32_t(q == 0) << events::UNDERFLOW)
                        : sum_bits(x, sticky, m.guard, q));
        return q;
      }
    }

    T value(int64_t s) const { return m.value(s); }
  };
#else
  template <class T, RoundMode RM, bool FUSED, bool WRAP>
  struct CountedFixedPointIntMac : FixedPointIntMac<T, RM, FUSED, WRAP>
  {
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    events::EventThresholdsT<T> mul_th{}; // the split mac's multiply format
    // A GPU thread's counts: the product's (MUL) and the sum's (ACC).
    mutable events::EventTally mul_tally{};
    mutable events::EventTally acc_tally{};

    // The sum's events, from the exact sum (x, plus a part in (0, 1) when
    // sticky, on the grid 2^-(fl + sh)) and its rounded code q.
    CUDA_HOST_DEVICE_INLINE uint32_t sum_bits(int64_t x, bool sticky, int sh, int64_t q) const
    {
      const int64_t top = fx_shl(this->hi, sh);
      const int64_t bottom = this->acc_signed ? fx_shl(this->lo, sh) : -top;
      uint32_t bits = 1u << events::CASTS;
      bits |= uint32_t(x > top || (x == top && sticky) || x < bottom) << events::OVERFLOW;
      bits |= uint32_t((x != 0 || sticky) && q == 0) << events::UNDERFLOW;
      bits |= uint32_t(sticky || fx_shl(q, sh) != x) << events::INEXACT;
      return bits;
    }

    // FixedPointIntMac::step, with its casts counted.
    CUDA_HOST_DEVICE_INLINE int64_t step(int64_t s, T a, T b, MPTORCH_THREAD PhiloxEngine &rng) const
    {
      constexpr uint32_t NAN_SUM_BITS = (1u << events::CASTS) | (1u << events::NAN_RESULT);
      if constexpr (!FUSED)
      {
        const T pq = this->mul(a, b, rng);
        count_cast(mul_tally, events::MUL, events::event_bits<T>(a * b, pq, mul_th));
        const word_t word = this->draw(rng);
        if (s == FIXEDPOINT_NAN_SUM || pq != pq)
        {
          count_cast(acc_tally, events::ACC, NAN_SUM_BITS);
          return FIXEDPOINT_NAN_SUM;
        }
        const int64_t p = static_cast<int64_t>(pq * this->p_scale); // an integer: pq is on M's grid
        const int64_t x = fx_shl(s, this->sh_s) + fx_shl(p, this->sh_p);
        const int64_t q = this->round_sum(x, false, this->sh_s, x < 0, word);
        count_cast(acc_tally, events::ACC, sum_bits(x, false, this->sh_s, q));
        return q;
      }
      else
      {
        const T p = a * b;
        const word_t word = this->draw(rng);
        if (s == FIXEDPOINT_NAN_SUM || p != p)
        {
          count_cast(acc_tally, events::ACC, NAN_SUM_BITS);
          return FIXEDPOINT_NAN_SUM;
        }
        const word_t w = reinterpret_cast<const MPTORCH_THREAD word_t &>(p);
        const bool p_neg = (w & F::SIGN_MASK) != 0;
        const word_t mag = w & F::ABS_MASK;
        const bool huge = !(p < this->p_top && p > -this->p_top); // an infinity too
        if (huge && (!WRAP || mag >= F::INF_BITS))
        {
          // saturated on the product's side: past the range, unless the
          // product is infinite, which no format's range has an end for; and
          // an underflow when that end is an unsigned format's 0
          const int64_t q = p_neg ? this->lo : this->hi;
          const bool finite = mag < F::INF_BITS;
          count_cast(acc_tally, events::ACC,
                     (1u << events::CASTS) | (uint32_t(finite) << events::OVERFLOW) |
                         (uint32_t(finite && q == 0) << events::UNDERFLOW) | (1u << events::INEXACT));
          return q; // the sum saturates on the product's side
        }
        // p = m * 2^e on the word
        const int field = static_cast<int>(mag >> F::MAN_BITS);
        const int64_t m = static_cast<int64_t>(
            field == 0 ? (mag & F::MAN_MASK) : ((mag & F::MAN_MASK) | (word_t(1) << F::MAN_BITS)));
        const int e = (field == 0 ? 1 : field) - F::BIAS - F::MAN_BITS;
        const int sh = e + this->acc_fl + this->guard; // m's shift onto the grid 2^-(fl + guard)
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
        const uint64_t base = static_cast<uint64_t>(fx_shl(s, this->guard));
        const uint64_t ux = !p_neg ? base + static_cast<uint64_t>(pa)
                                   : base - static_cast<uint64_t>(pa) - (sticky ? 1u : 0u);
        const int64_t x = static_cast<int64_t>(ux);
        const bool neg = huge ? p_neg : x < 0;
        const int64_t q = this->round_sum(x, sticky, this->guard, neg, word);
        // a huge finite product under wrap-around: x is the sum modulo 2^64,
        // and the exact sum is past the range
        count_cast(acc_tally, events::ACC,
                   huge ? (1u << events::CASTS) | (1u << events::OVERFLOW) | (1u << events::INEXACT) |
                              (uint32_t(q == 0) << events::UNDERFLOW)
                        : sum_bits(x, sticky, this->guard, q));
        return q;
      }
    }
  };
#endif

  // The counted twin of a FixedPointIntMac type.
  template <class Mac>
  struct counted_int_mac;

  template <class T, RoundMode RM, bool FUSED, bool WRAP>
  struct counted_int_mac<FixedPointIntMac<T, RM, FUSED, WRAP>>
  {
    using type = CountedFixedPointIntMac<T, RM, FUSED, WRAP>;
    static constexpr MPTORCH_CONSTANT bool fused = FUSED;
  };

  // FixedPointIntAccumulator through the positional contract: the same sum,
  // k < K only.
  template <class Mac>
  struct PositionalFixedPointIntAccumulator
  {
    using mac_type = Mac;
    using value_t = typename Mac::value_t;
#if !defined(__METAL_VERSION__)
    using tile_type = FixedPointIntTile;
#endif
    static constexpr MPTORCH_CONSTANT bool positional = true;
    // the kernels read the result as the Mac scales it
    static constexpr MPTORCH_CONSTANT bool int_result = true;
    static constexpr MPTORCH_CONSTANT int unroll = 4;

    Mac mac{};
    int64_t sum = 0;
    PhiloxEngine rng{};

    CUDA_HOST_DEVICE_INLINE void seed_rng(uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
      rng.reset_state(seed, subsequence);
      rng.set_offset(offset);
    }
    CUDA_HOST_DEVICE_INLINE void accumulate(value_t a, value_t b, int) { sum = mac.step(sum, a, b, rng); }
    CUDA_HOST_DEVICE_INLINE void end_slab(int64_t, int64_t) {}
    CUDA_HOST_DEVICE_INLINE value_t finalize() const { return mac.value(sum); }
#if !defined(__METAL_VERSION__)
    CUDA_HOST_DEVICE_INLINE double finalize_wide() const { return mac.value64(sum); }

    static inline void accumulate(const FixedPointIntTileView &t, int64_t idx,
                                  const PositionalFixedPointIntAccumulator &cfg, value_t a, value_t b,
                                  int)
    {
      t.sums[idx] = cfg.mac.step(t.sums[idx], a, b, t.rng[idx]);
    }
    static inline void end_slab(const FixedPointIntTileView &, int64_t,
                                const PositionalFixedPointIntAccumulator &, int64_t, int64_t)
    {
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

  // The Args of a counted fixed-point call: the Args its plan runs (the
  // carrier path's FixedPointSplitArgs, or a FixedPointIntArgs), and where the
  // counts go.
  template <class Args>
  struct FixedPointCountingArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = Args::draws_per_k_step;

    Args base{};
#if defined(__METAL_VERSION__)
    uint32_t counts = 0; // unused: see gemm_counted.h's Counting
#else
    unsigned long long *counts = nullptr;
#endif

    template <class T, RoundMode RM, class F>
    void with_accumulator(MPTORCH_THREAD F &&f) const
    {
      if constexpr (std::is_same_v<Args, FixedPointSplitArgs>)
      {
        using M = FixedPointMultiplierT<T, RM>;
        using A = FixedPointAdderT<T, RM>;
        using Mac = SplitMac<Counted<M, events::MUL>, Counted<A, events::ACC>>;
        const Mac mac{{M(base.mul, base.mul_c), thresholds_of<T>(base.mul, base.mul_c)},
                      {A(base.acc, base.acc_c), thresholds_of<T>(base.acc, base.acc_c)}};
        f(Counting<PositionalNaiveAccumulator<Mac>>{{mac}, counts});
      }
      else
      {
        const auto plain = base.template make_mac<T, RM>();
        using Twin = counted_int_mac<std::remove_const_t<decltype(plain)>>;
        using Mac = typename Twin::type;
        Mac mac{plain};
        if constexpr (!Twin::fused) // a fused mac's multiply widths are unused zeros
          mac.mul_th = thresholds_of<T>(base.mul, base.mul_c);
        f(Counting<PositionalFixedPointIntAccumulator<Mac>>{{mac}, counts});
      }
    }
  };

#if defined(__CUDACC__)
  template <class T, RoundMode RM, bool FUSED, bool WRAP>
  __device__ inline void counting_flush_mac(const CountedFixedPointIntMac<T, RM, FUSED, WRAP> &m,
                                            bool in_range, unsigned long long *counts)
  {
    counting_flush_tally(m.mul_tally, events::MUL, in_range, counts);
    counting_flush_tally(m.acc_tally, events::ACC, in_range, counts);
  }
#endif

#if defined(__METAL_VERSION__)
  // The integer mac's two tallies, for gemm_counted.h's counting_tallies.
  template <class T, RoundMode RM, bool FUSED, bool WRAP>
  inline void counting_gather_mac(
      const MPTORCH_THREAD CountedFixedPointIntMac<T, RM, FUSED, WRAP> &m,
      MPTORCH_THREAD events::EventTally *t)
  {
    for (int e = 0; e < events::EVENTS; ++e)
    {
      t[events::MUL].n[e] += m.mul_tally.n[e];
      t[events::ACC].n[e] += m.acc_tally.n[e];
    }
  }
#endif
} // namespace mptorch::gemm
