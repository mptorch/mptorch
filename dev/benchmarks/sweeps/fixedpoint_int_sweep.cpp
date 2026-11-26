// The fixed-point GEMM's integer path (common/fixedpoint_gemm.h), against the
// shipped casts, on the host. Four modes:
//
//   round  fixedpoint_round, saturated or wrapped, against cast_fixedpoint.h's
//          casts in binary32, over every exact sum of small spans: every
//          integer x at grid 2^-(fl + s) up to twice past the range, for
//          shifts s of 1, 2, 3, 6 and 9 bits (those binary32 holds the sums
//          of), in the six deterministic modes and under
//          StochasticA, B and C with 0, 1, 3 and 6 random bits over every
//          draw, and, under SR, the split step's rounding from per-launch
//          constants (fixedpoint_round_sr) too. Wrap-around is held to
//          the cast into a format wide enough never to saturate, then
//          reduced modulo 2^wl.
//   fused  FixedPointIntMac's fused step, with its guard and sticky bits,
//          against the shipped cast in binary64 on the exact sum
//          s * 2^-fl + p, for random float32 products p from 29 binades
//          below the format's top to 12 past it (sums binary64 holds),
//          infinities and zeros, every mode, both overflow behaviours. Two
//          kinds of product binary64 cannot add exactly have references of
//          their own: a tiny one, down to 100 binades under the step, rounds
//          as a stand-in close to the grid, and a far one, a multiple of the
//          step up to 90 binades past the range, saturates or wraps by its
//          residue modulo 2^wl. Then far products alone into accumulators of
//          44 to 53 bits, where the residue reaches the aligned product's
//          shifts of 40 to 64 bits. Under SR the step's draw is read from a
//          copy of its engine and given to the cast.
//   chain  the integer path against the carrier path (SplitMac over
//          FixedPointMultiplierT and FixedPointAdderT) over random K-chains,
//          for format pairs whose exact sums binary32 holds: every mode, the
//          same Philox streams, the same words. Then the same for pairs whose
//          sums binary32 does not hold, on operands of one sign near the
//          multiply format's top, so that the sums outgrow binary32's
//          significand: there they must sometimes differ (the negative
//          control).
//   wide   the integer path with a 32- and a 40-bit accumulator in binary32,
//          stored as float64, against the carrier path in binary64, on
//          operands whose products and sums both formats hold exactly: every
//          mode returns the exact sum, so what this checks is the wide
//          accumulator and its store, not the rounding (the two paths draw
//          words of different widths).
//
// Build & run (from the repo root), with the extension's host flags:
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 -I mptorch/csrc
//       dev/benchmarks/sweeps/fixedpoint_int_sweep.cpp -o /tmp/fx_int && /tmp/fx_int
// Pass a mode, or nothing for all four (~5 s on 16 threads). Add
// -fsanitize=undefined -fno-sanitize-recover for the sanitizer's run.
// Exit status 0 when every comparison that must agree agrees.

#include "common/cast_fixedpoint.h"
#include "common/fixedpoint_gemm.h"
#include "common/modes.h"
#include "common/stochastic.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace mptorch::gemm;

namespace
{
  struct Fmt
  {
    int wl, fl;
    bool is_signed, symmetric;
    int mag() const { return wl - (is_signed ? 1 : 0); }
  };

  // The shipped cast in mode RM (and under SR variant `variant` with n
  // random bits, the word laid out as the cast reads it), on the carrier T.
  template <class T, RoundMode RM>
  T shipped(T x, const Fmt &f, const FixedPointParamsT<T> &p, uint64_t r, int n, int variant)
  {
    if constexpr (RM == RoundMode::SR)
    {
      using W = typename FloatTraits<T>::word_t;
      const W word = static_cast<W>(r << (FloatTraits<T>::MAN_BITS - n));
      return cast_fixedpoint_stochastic(x, word, sr_spec(n, variant), f.is_signed, p);
    }
    else if constexpr (RM == RoundMode::RNA)
      return cast_fixedpoint_nearest_away(x, f.is_signed, p);
    else if constexpr (RM == RoundMode::RU)
      return cast_fixedpoint_up(x, f.is_signed, p);
    else if constexpr (RM == RoundMode::RD)
      return cast_fixedpoint_down(x, f.is_signed, p);
    else if constexpr (RM == RoundMode::RZ)
      return cast_fixedpoint_zero(x, f.is_signed, p);
    else if constexpr (RM == RoundMode::RO)
      return cast_fixedpoint_odd(x, f.is_signed, p);
    else
      return cast_fixedpoint_nearest_even(x, f.is_signed, p);
  }

  int64_t lo_of(const Fmt &f)
  {
    return !f.is_signed ? 0
                        : (f.symmetric ? -((int64_t(1) << f.mag()) - 1) : -(int64_t(1) << f.mag()));
  }
  int64_t hi_of(const Fmt &f)
  {
    return f.is_signed ? (int64_t(1) << f.mag()) - 1 : (int64_t(1) << f.wl) - 1;
  }

  const char *name(RoundMode rm, int variant)
  {
    if (rm == RoundMode::SR)
      return variant == 0 ? "SR_A" : variant == 1 ? "SR_B" : "SR_C";
    static const char *n[] = {"RNE", "RNA", "RU", "RD", "RZ", "RO"}; // modes.h's order
    return n[static_cast<int>(rm)];
  }

  int failures = 0;

  void report(const char *mode, const char *what, unsigned long long bad, unsigned long long total)
  {
    std::printf("%-6s %-44s %12llu checked  %llu mismatched\n", mode, what, total, bad);
    failures += bad != 0;
  }

  // --- round -------------------------------------------------------------------------

  template <RoundMode RM>
  void round_one(const Fmt &f, int s, int n, int variant, bool wrap)
  {
    const auto p = make_fixedpoint_params<float>(f.wl, f.fl, f.is_signed, f.symmetric);
    // a format with f's step and room above: the cast's unsaturated rounding
    const Fmt wide{f.mag() + s + 4, f.fl, true, false};
    const auto pw = make_fixedpoint_params<float>(wide.wl, wide.fl, true, false);
    const int64_t lim = int64_t(1) << (f.mag() + s + 1);
    const int64_t lo = lo_of(f), hi = hi_of(f);
    const uint64_t draws = RM == RoundMode::SR ? (uint64_t(1) << n) : 1;
    unsigned long long bad = 0, total = 0;
#pragma omp parallel for reduction(+ : bad, total) schedule(static)
    for (int64_t x = -lim; x <= lim; ++x)
    {
      const float v = std::ldexp(static_cast<float>(x), -(f.fl + s)); // exact: at most 24 bits
      for (uint64_t r = 0; r < draws; ++r)
      {
        int64_t want;
        if (!wrap)
          want = static_cast<int64_t>(std::ldexp(shipped<float, RM>(v, f, p, r, n, variant), f.fl));
        else
          want = fixedpoint_wrap_code(static_cast<int64_t>(std::ldexp(
                                          shipped<float, RM>(v, wide, pw, r, n, variant), f.fl)),
                                      f.wl, f.is_signed);
        const int64_t q = fixedpoint_round<RM>(x, false, s, x < 0, r, n, variant);
        const int64_t got =
            wrap ? fixedpoint_wrap_code(q, f.wl, f.is_signed) : fixedpoint_saturate_code(q, lo, hi);
        bad += got != want;
        ++total;
        if constexpr (RM == RoundMode::SR)
        {
          // the split step's rounding from per-launch constants
          const auto c = make_fixedpoint_int_sr(s, sr_spec(n, variant));
          const int64_t q2 = fixedpoint_round_sr(x, false, s, x < 0, r, c);
          const int64_t got2 = wrap ? fixedpoint_wrap_code(q2, f.wl, f.is_signed)
                                    : fixedpoint_saturate_code(q2, lo, hi);
          bad += got2 != want;
          ++total;
        }
      }
    }
    char what[96];
    std::snprintf(what, sizeof what, "%s wl=%d fl=%d %s s=%d n=%d%s", name(RM, variant), f.wl, f.fl,
                  f.is_signed ? (f.symmetric ? "sym" : "signed") : "unsigned", s, n,
                  wrap ? " wrap" : "");
    if (bad)
      report("round", what, bad, total);
  }

  void round_mode()
  {
    const Fmt fmts[] = {{4, 0, true, false}, {6, 2, true, false},  {8, 4, true, false},
                        {8, 4, true, true},  {8, 4, false, false}, {10, -3, true, false},
                        {12, 6, true, false}};
    unsigned long long cases = 0;
    const int before = failures;
    for (const Fmt &f : fmts)
      for (int s : {1, 2, 3, 6, 9})
      {
        if (f.mag() + s + 6 > 24)
          continue;
        for (bool wrap : {false, true})
        {
          if (wrap && f.symmetric)
            continue;
          round_one<RoundMode::RNE>(f, s, 0, 0, wrap);
          round_one<RoundMode::RNA>(f, s, 0, 0, wrap);
          round_one<RoundMode::RU>(f, s, 0, 0, wrap);
          round_one<RoundMode::RD>(f, s, 0, 0, wrap);
          round_one<RoundMode::RZ>(f, s, 0, 0, wrap);
          round_one<RoundMode::RO>(f, s, 0, 0, wrap);
          for (int variant = 0; variant < 3; ++variant)
            for (int n : {0, 1, 3, 6})
              round_one<RoundMode::SR>(f, s, n, variant, wrap);
          cases += 18;
        }
      }
    std::printf("round  %llu (format, shift, mode, overflow) cases over every sum and draw: %s\n",
                cases, failures == before ? "all agree" : "MISMATCHES above");
  }

  // --- fused -------------------------------------------------------------------------

  template <RoundMode RM>
  void fused_one(const Fmt &f, int n, int variant, bool wrap, std::mt19937_64 &gen,
                 bool far_only = false)
  {
    // the step saturating and the step wrapping around (a template parameter)
    const FixedPointCommon c{f.is_signed, f.symmetric,
                             RM == RoundMode::SR ? sr_spec(n, variant) : 0};
    FixedPointIntFusedArgs<false> a_sat;
    FixedPointIntFusedArgs<true> a_wrap;
    a_sat.acc = a_wrap.acc = {f.wl, f.fl};
    a_sat.acc_c = a_wrap.acc_c = c;
    const auto m_sat = a_sat.make_mac<float, RM>();
    const auto m_wrap = a_wrap.make_mac<float, RM>();
    // the binary64 references' formats: f, and f with 16 more bits for the
    // sums that wrap (the far-only pass, whose formats are too wide for
    // binary64's casts, has integer references only)
    const Fmt wide{far_only ? f.wl : f.mag() + 16, f.fl, true, false};
    const auto p64 =
        make_fixedpoint_params<double>(far_only ? 2 : f.wl, f.fl, f.is_signed, f.symmetric);
    const auto pw64 = make_fixedpoint_params<double>(far_only ? 2 : wide.wl, wide.fl, true, false);
    const int64_t lo = lo_of(f), hi = hi_of(f);
    std::uniform_int_distribution<int64_t> code(lo, hi);
    // the product's exponent: low enough that binary64 still holds the sum
    // (its lowest bit 53 bits below the sum's top), high enough to saturate
    std::uniform_int_distribution<int> expo(f.mag() - f.fl - 29, f.mag() - f.fl + 12);
    // a tiny product, past the stand-in's bound below, down to 64 bits and
    // more under the grid (all sticky)
    const int nn = RM == RoundMode::SR ? n : 0;
    std::uniform_int_distribution<int> tiny_expo(std::max(-f.fl - 100, -140), -f.fl - nn - 5);
    // a far product, a multiple of the step at least 2^12 past the range,
    // up to 90 binades past it (a shift of 64 bits and more)
    std::uniform_int_distribution<int> far_expo(std::max(f.mag() - f.fl + 13, 23 - f.fl),
                                                std::min(f.mag() - f.fl + 90, 127));
    std::uniform_int_distribution<uint32_t> mant(0, (1u << 23) - 1);
    unsigned long long bad = 0, total = 0;
    PhiloxEngine eng{};
    eng.reset_state(77, 5);
    for (int t = 0; t < 400000; ++t)
    {
      const int64_t s = code(gen);
      float pf;
      const int kind = t % 50;
      if (kind == 0)
        pf = INFINITY;
      else if (kind == 1)
        pf = -INFINITY;
      else if (kind == 2)
        pf = 0.0f;
      else
        pf = std::ldexp(1.0f + std::ldexp(static_cast<float>(mant(gen)), -23), expo(gen)) *
             ((gen() & 1) ? -1.f : 1.f);
      if (kind == 3)
        pf = std::ldexp(std::round(std::ldexp(pf, f.fl + 1)),
                        -(f.fl + 1)); // a tie on the grid, or a code
      const bool tiny = !far_only && kind >= 4 && kind < 9,
                 far = far_only ? kind >= 2 : kind >= 9 && kind < 14;
      const float sgn = (gen() & 1) ? -1.f : 1.f;
      if (tiny)
        pf =
            sgn * std::ldexp(1.0f + std::ldexp(static_cast<float>(mant(gen)), -23), tiny_expo(gen));
      if (far)
        pf = sgn * std::ldexp(1.0f + std::ldexp(static_cast<float>(mant(gen)), -23), far_expo(gen));
      PhiloxEngine copy = eng;
      const uint32_t w = RM == RoundMode::SR ? copy() : 0u;
      const uint64_t r = RM == RoundMode::SR ? fixedpoint_draw_bits<float>(w, n) : 0;
      const int64_t got = wrap ? m_wrap.step(s, pf, 1.0f, eng) : m_sat.step(s, pf, 1.0f, eng);
      int64_t want;
      // A tiny product lies within 2^-(fl + N + 4) of a step's multiple, so
      // the sum is in the same open interval of every mode's decision as the
      // sum with the stand-in +-2^-(fl + N + 4) (frac(X), frac(2X) and
      // RNITE(X) near 0 or near the top), which binary64 holds exactly.
      const double p_ref =
          tiny ? std::copysign(std::ldexp(1.0, -f.fl - nn - 4), pf) : static_cast<double>(pf);
      const double v = std::ldexp(static_cast<double>(s), -f.fl) + p_ref; // exact
      if (std::isinf(pf))
        want = pf > 0 ? hi : lo; // an infinity saturates, under wrap-around too
      else if (far && !wrap)
        want = pf > 0 ? hi : lo;
      else if (far)
      {
        // a multiple of the step: the sum is on the grid, and wraps exactly,
        // its residue computed in integers
        int e2;
        const float fr = std::frexp(std::fabs(pf), &e2); // |p| = fr * 2^e2, fr in [0.5, 1)
        const uint64_t sig = static_cast<uint64_t>(std::ldexp(fr, 24));
        const int sh = e2 - 24 + f.fl; // >= 0 by far_expo's bounds
        const uint64_t res = sh >= 64 ? 0 : (sig << sh) & ((uint64_t(1) << f.wl) - 1);
        want = fixedpoint_wrap_code(
            s + (pf < 0 ? -static_cast<int64_t>(res) : static_cast<int64_t>(res)), f.wl,
            f.is_signed);
      }
      else if (!wrap)
        want =
            static_cast<int64_t>(std::ldexp(shipped<double, RM>(v, f, p64, r, n, variant), f.fl));
      else
        want = fixedpoint_wrap_code(static_cast<int64_t>(std::ldexp(
                                        shipped<double, RM>(v, wide, pw64, r, n, variant), f.fl)),
                                    f.wl, f.is_signed);
      bad += got != want;
      ++total;
      if (got != want && bad <= 3)
        std::printf("  fused %s wl=%d fl=%d s=%lld p=%a: got %lld want %lld\n", name(RM, variant),
                    f.wl, f.fl, (long long)s, pf, (long long)got, (long long)want);
    }
    char what[96];
    std::snprintf(what, sizeof what, "%s wl=%d fl=%d %s n=%d%s%s", name(RM, variant), f.wl, f.fl,
                  f.is_signed ? (f.symmetric ? "sym" : "signed") : "unsigned", n,
                  wrap ? " wrap" : "", far_only ? " far" : "");
    report("fused", what, bad, total);
  }

  void fused_mode()
  {
    std::mt19937_64 gen(2026);
    const Fmt fmts[] = {{8, 4, true, false},
                        {8, 4, false, false},
                        {12, 6, true, true},
                        {16, 4, true, false}};
    for (const Fmt &f : fmts)
      for (bool wrap : {false, true})
      {
        if (wrap && f.symmetric)
          continue;
        fused_one<RoundMode::RNE>(f, 0, 0, wrap, gen);
        fused_one<RoundMode::RNA>(f, 0, 0, wrap, gen);
        fused_one<RoundMode::RU>(f, 0, 0, wrap, gen);
        fused_one<RoundMode::RD>(f, 0, 0, wrap, gen);
        fused_one<RoundMode::RZ>(f, 0, 0, wrap, gen);
        fused_one<RoundMode::RO>(f, 0, 0, wrap, gen);
        for (int variant = 0; variant < 3; ++variant)
          fused_one<RoundMode::SR>(f, 7, variant, wrap, gen);
      }
    // accumulators past binary64's reach, under wrap-around, on far products
    // only, whose references are integer: there the residue's bits reach the
    // aligned product's shifts of 40 to 64 bits (N = 5 keeps the fused
    // step's 62-bit bound with 53 magnitude bits)
    const Fmt wide_fmts[] = {{53, 0, false, false}, {52, 4, true, false}, {44, 12, true, false}};
    for (const Fmt &f : wide_fmts)
    {
      fused_one<RoundMode::RNE>(f, 0, 0, true, gen, true);
      fused_one<RoundMode::RU>(f, 0, 0, true, gen, true);
      fused_one<RoundMode::RZ>(f, 0, 0, true, gen, true);
      fused_one<RoundMode::RO>(f, 0, 0, true, gen, true);
      for (int variant = 0; variant < 3; ++variant)
        fused_one<RoundMode::SR>(f, 5, variant, true, gen, true);
    }
  }

  // --- chain -------------------------------------------------------------------------

  template <RoundMode RM, class T = float>
  unsigned long long chain_one(const Fmt &mf, const Fmt &af, int variant, int prng, uint64_t seed,
                               int trials, bool must_agree, bool wide = false, bool large = false)
  {
    const int spec = RM == RoundMode::SR ? sr_spec(prng, variant) : 0;
    FixedPointCommon mc{mf.is_signed, mf.symmetric, spec}, ac{af.is_signed, af.symmetric, spec};
    FixedPointSplitArgs aa;
    aa.mul = {mf.wl, mf.fl};
    aa.acc = {af.wl, af.fl};
    aa.mul_c = mc;
    aa.acc_c = ac;
    FixedPointIntSplitArgs<false> ba;
    ba.mul = aa.mul;
    ba.acc = aa.acc;
    ba.mul_c = mc;
    ba.acc_c = ac;
    ba.wide = wide;
    unsigned long long bad = 0;
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int t = 0; t < trials; ++t)
    {
      const int K = 96;
      std::vector<T> av(K), bv(K);
      const double scale = std::ldexp(1.0, mf.mag() / 2 - mf.fl / 2);
      for (int k = 0; k < K; ++k)
      {
        av[k] = static_cast<T>(g(gen) * scale);
        bv[k] = static_cast<T>(g(gen) * scale);
        if (large) // one sign, each product near the multiply format's top
        {
          const double top = std::sqrt(std::ldexp(1.0, mf.mag() - mf.fl));
          av[k] = static_cast<T>(top * (0.5 + std::fabs(g(gen)) / 8));
          bv[k] = static_cast<T>(top * (0.5 + std::fabs(g(gen)) / 8));
        }
        if (wide) // operands on an 8-bit grid of step 2^-4: products binary32 holds exactly
        {
          av[k] = static_cast<T>(std::fmax(-128, std::fmin(127, std::round(g(gen) * 16))) / 16);
          bv[k] = static_cast<T>(std::fmax(-128, std::fmin(127, std::round(g(gen) * 16))) / 16);
        }
      }
      if (t % 97 == 5)
        av[K / 2] = NAN;
      if (t % 89 == 7)
        av[K / 3] = INFINITY;
      // the carrier path
      T a_res;
      aa.template with_accumulator<T, RM>(
          [&](auto acc)
          {
            acc.seed_rng(seed, static_cast<uint64_t>(t));
            for (int k = 0; k < K; ++k)
              acc.accumulate(av[k], bv[k]);
            a_res = acc.finalize();
          });
      // the integer path
      T b_res;
      ba.template with_accumulator<float, RM>(
          [&](auto acc)
          {
            acc.seed_rng(seed, static_cast<uint64_t>(t));
            for (int k = 0; k < K; ++k)
              acc.accumulate(static_cast<float>(av[k]), static_cast<float>(bv[k]));
            b_res = wide ? static_cast<T>(acc.finalize_wide()) : static_cast<T>(acc.finalize());
          });
      const bool both_nan = a_res != a_res && b_res != b_res;
      if (!both_nan && a_res != b_res)
      {
        ++bad;
        if (must_agree && bad <= 3)
          std::printf("  chain %s mul (%d,%d) acc (%d,%d) trial %d: carrier %a integer %a\n",
                      name(RM, variant), mf.wl, mf.fl, af.wl, af.fl, t, (double)a_res,
                      (double)b_res);
      }
    }
    return bad;
  }

  template <class T = float>
  void chain_pair(const Fmt &mf, const Fmt &af, bool must_agree, bool wide = false)
  {
    const int trials = 3000;
    const bool large = !must_agree;
    // the random bits, inside both casts' domain (mag_bits + N at most the
    // carrier's precision, which the Python checks enforce): the multiply
    // format is rounded in binary32 on both paths, the accumulate format in
    // the carrier path's carrier
    const int prng = std::min({5, 24 - mf.mag(), (wide ? 53 : 24) - af.mag()});
    unsigned long long bad = 0, total = 0;
    bad += chain_one<RoundMode::RNE, T>(mf, af, 0, 0, 11, trials, must_agree, wide, large);
    bad += chain_one<RoundMode::RNA, T>(mf, af, 0, 0, 12, trials, must_agree, wide, large);
    bad += chain_one<RoundMode::RU, T>(mf, af, 0, 0, 13, trials, must_agree, wide, large);
    bad += chain_one<RoundMode::RD, T>(mf, af, 0, 0, 14, trials, must_agree, wide, large);
    bad += chain_one<RoundMode::RZ, T>(mf, af, 0, 0, 15, trials, must_agree, wide, large);
    bad += chain_one<RoundMode::RO, T>(mf, af, 0, 0, 16, trials, must_agree, wide, large);
    for (int variant = 0; variant < 3; ++variant)
      bad += chain_one<RoundMode::SR, T>(mf, af, variant, prng, 20 + variant, trials, must_agree,
                                         wide, large);
    total = 9ull * trials;
    char what[96];
    std::snprintf(what, sizeof what, "mul (%d,%d%s) acc (%d,%d%s) span %d N=%d%s", mf.wl, mf.fl,
                  mf.is_signed ? "" : " u", af.wl, af.fl,
                  af.is_signed ? (af.symmetric ? " sym" : "") : " u",
                  std::max(mf.mag() - mf.fl, af.mag() - af.fl) + 1 + std::max(mf.fl, af.fl), prng,
                  wide ? " (binary64 carrier, wide)" : "");
    if (must_agree)
      report(wide ? "wide" : "chain", what, bad, total);
    else
      std::printf("%-6s %-44s %12llu chains  %llu differ (the negative control: some must)\n",
                  "chain", what, total, bad);
    if (!must_agree && bad == 0)
      ++failures;
  }

  void chain_mode()
  {
    // pairs whose exact sums binary32 holds (span <= 24): the carrier path is the integer path
    chain_pair({16, 8, true, false}, {16, 4, true, false}, true);
    chain_pair({12, 6, true, false}, {20, 6, true, false}, true);
    chain_pair({10, 4, false, false}, {16, 2, false, false}, true);
    chain_pair({16, 8, true, true}, {14, 6, true, true}, true);
    chain_pair({8, 0, true, false}, {24, 0, true, false}, true);
    // pairs past it, each format one binary32 rounds correctly, their exact
    // sums not: binary32 rounds some of them
    chain_pair({22, 8, true, false}, {24, 0, true, false}, false);
    chain_pair({20, 12, true, false}, {24, 4, true, false}, false);
  }

  void wide_mode()
  {
    // a 32-bit accumulator, held exactly as integers, against binary64's sums
    chain_pair<double>({16, 8, true, false}, {32, 8, true, false}, true, true);
    chain_pair<double>({16, 8, true, false}, {40, 12, true, false}, true, true);
  }
} // namespace

int main(int argc, char **argv)
{
  const char *mode = argc > 1 ? argv[1] : "all";
  const bool all = !std::strcmp(mode, "all");
  if (all || !std::strcmp(mode, "round"))
    round_mode();
  if (all || !std::strcmp(mode, "fused"))
    fused_mode();
  if (all || !std::strcmp(mode, "chain"))
    chain_mode();
  if (all || !std::strcmp(mode, "wide"))
    wide_mode();
  std::printf("%s\n", failures ? "FAILED" : "every comparison agrees");
  return failures ? 1 : 0;
}
