#pragma once

// Branchless forms of the binaryK casts in binary32 for the deterministic
// modes that have no float-arithmetic fast path: RNA, RU, RD, RZ and RO.
// Host only, for the CPU's vector tiles (cpu/gemm_word_tile.cpp).
//
// The shipped casts (cast_binaryK.h) branch on the input: on its sign (the
// directed modes hand a negative input's magnitude to the other helper), on
// whether it is below the format's smallest normal, a NaN or an infinity,
// and inside the clips. On a GEMM's sums the sign is a coin flip, so the
// scalar loop mispredicts a branch on most steps (bit_helper.h's
// MPTORCH_NEGATIVE_TO_UNSIGNED says what one such branch cost the RNE step),
// and a loop that branches cannot be vectorized at all.
// Each function below computes every arm of its cast on every input, as
// integer arithmetic on the word, and selects one: the same expressions,
// in the same order, as the shipped arms, so the result is the shipped
// cast's word for word. dev/benchmarks/sweeps/cpu_word_cast_sweep.cpp holds
// them to the shipped casts over all 2^32 binary32 inputs.
//
// They read the same BinaryKParams, and nothing new: adding a field would
// move every object that holds one. They serve the formats `word_cast_format`
// admits: signed, under SUBNORMALS, with 1 to 22 stored mantissa bits. An
// unsigned format's negative inputs, the other subnormal modes, and the
// man_bits == 0 arm of the odd cast keep the shipped casts.
//
// The shifts by a per-input amount (the subnormal arm's exp_diff) are
// clamped first, so that the arm an input does not take never shifts by
// the word's width or more: those values are computed and then not chosen.

#include "cast_binaryK.h"

#include <bit>
#include <cstdint>

#if !defined(__CUDA_ARCH__) && !defined(__METAL_VERSION__)

namespace binaryk_word
{
  constexpr uint32_t SIGN = 0x80000000u;
  constexpr uint32_t ABS = 0x7fffffffu;
  constexpr uint32_t MAN = 0x007fffffu;
  constexpr uint32_t INF = 0x7f800000u;

  inline uint32_t word(float x)
  {
    return std::bit_cast<uint32_t>(x);
  }
  inline float value(uint32_t w)
  {
    return std::bit_cast<float>(w);
  }

  // round_bitwise_*(w, k)'s mask for k bits kept below the leading bit: the
  // 23 - k low bits, none at all for k >= 23 (the round is then the
  // identity). k >= -1 here.
  inline uint32_t mask_for(int k)
  {
    const int sh = 23 - (k < 23 ? k : 23);
    return (sh >= 32) ? ~0u : (uint32_t(1) << sh) - 1u;
  }

  // the input's exponent field, unbiased, as the shipped casts compute it
  inline int target_exp(uint32_t w)
  {
    return int((w >> 23) & 0xFFu) - 127;
  }

  // saturate_nonfinite (bit_helper.h)
  inline uint32_t nonfinite(uint32_t w, const BinaryKParams &p)
  {
    const bool is_inf = (w & MAN) == 0;
    const bool clamps = p.saturation_mode == SaturationMode::SAT_FINITE ||
                        p.saturation_mode == SaturationMode::SAT_FINITE_EXTENDED;
    return (is_inf && clamps) ? ((w & SIGN) | p.max_num) : w;
  }

  // clip_normal_range_magnitude<U> (bit_helper.h), U one of the three the
  // casts below use
  template <UnderflowMode U>
  inline uint32_t clip_magnitude(uint32_t old, uint32_t q, const BinaryKParams &p)
  {
    const uint32_t ax = old & ABS;
    q &= ABS;
    const uint32_t ovf = (p.saturation_mode == SaturationMode::SAT_NONE) ? INF : p.max_num;
    uint32_t under;
    if constexpr (U == UnderflowMode::NEAREST_AWAY)
      under = (ax >= p.half_num) ? p.min_num : 0u;
    else if constexpr (U == UnderflowMode::AWAY)
      under = p.min_num;
    else
      under = 0u;
    const uint32_t r = (q > p.max_num) ? ovf : ((q < p.min_num) ? under : q);
    return (ax == 0u) ? 0u : r;
  }

  // clip_normal_range_exponent<U>: the magnitude with the input's sign, unless
  // there is no magnitude to sign
  template <UnderflowMode U>
  inline uint32_t clip_signed(uint32_t old, uint32_t q, const BinaryKParams &p)
  {
    const uint32_t m = clip_magnitude<U>(old, q, p);
    return m | ((old & SIGN) & -uint32_t(m != 0u));
  }

  // clip_subnormal_range_exponent (bit_helper.h)
  inline uint32_t clip_sub(uint32_t old, uint32_t q, int mes)
  {
    const int eq = int((q >> 23) & 0xFFu);
    const uint32_t offset = uint32_t(eq == mes - 1);
    const uint32_t lifted = ((q + (offset << 23)) | (old & SIGN)) * offset;
    const uint32_t r = (eq < mes) ? lifted : unsigned_zero_bits(q);
    return (q == 0u) ? q : r;
  }

  // clip_subnormal_range_exponent_up (bit_helper.h)
  inline uint32_t clip_sub_up(uint32_t old, uint32_t q, int mes)
  {
    const int eq = int((q >> 23) & 0xFFu);
    const uint32_t smallest = uint32_t(int32_t(uint32_t(mes) << 23)) | (old & SIGN);
    const uint32_t r = (eq < mes) ? smallest : unsigned_zero_bits(q);
    return ((old << 1) == 0u) ? 0u : r;
  }

  // negate_magnitude (bit_helper.h), on the word
  inline uint32_t negate(uint32_t m)
  {
    return m ^ (SIGN & -uint32_t((m << 1) != 0u));
  }

  // cast_absolute_up (cast_binaryK.h) where `away` is set, and
  // cast_absolute_down where it is not: the magnitude word m (an input that
  // compares >= 0, -0.0's word included, or a negative input with its sign
  // flipped) rounded away from zero or toward it. One function with a lane's
  // direction as a select, rather than both helpers on every lane: RU and RD
  // need either on any lane, and the two together were more than GCC would
  // vectorize. The two helpers share their mask and differ in which word
  // gets it, the subnormal clip and the underflow rule.
  inline uint32_t absolute_directed(uint32_t m, bool away, const BinaryKParams &p)
  {
    const int te = target_exp(m);
    // round_bitwise_up adds the mask only to a positive word, and
    // round_bitwise_down only to a negative one
    const uint32_t add_to = ((m >> 31) != 0u) != away ? ~0u : 0u;
    const int exp_diff = p.man_bits - (p.min_exp - te);
    const uint32_t mk = mask_for(exp_diff < 0 ? 0 : exp_diff);
    const uint32_t qs_r = (m + (mk & add_to)) & ~mk;
    const int mes = p.subnormal_min_exponent_store;
    const uint32_t qs =
        away ? clip_sub_up(m, qs_r, mes) : clip_sub(m, (exp_diff > -1) ? qs_r : 0u, mes);
    const uint32_t qn_r = p.round_bypass ? m : ((m + (p.round_mask & add_to)) & ~p.round_mask);
    // clip_magnitude<AWAY> or <ZERO>, which differ only in what an underflow
    // becomes
    const uint32_t ax = m & ABS;
    const uint32_t q = qn_r & ABS;
    const uint32_t ovf = (p.saturation_mode == SaturationMode::SAT_NONE) ? INF : p.max_num;
    const uint32_t under = away ? p.min_num : 0u;
    const uint32_t clipped = (q > p.max_num) ? ovf : ((q < p.min_num) ? under : q);
    const uint32_t qn = (ax == 0u) ? 0u : clipped;
    return (te < p.min_exp) ? qs : ((te == 128) ? nonfinite(m, p) : qn);
  }

  // cast_absolute_down (cast_binaryK.h): the magnitude word rounded toward zero
  inline uint32_t absolute_down(uint32_t m, const BinaryKParams &p)
  {
    const int te = target_exp(m);
    const uint32_t down = (m >> 31) ? ~0u : 0u; // round_bitwise_down adds it only to a negative one
    const int exp_diff = p.man_bits - (p.min_exp - te);
    const uint32_t mk = mask_for(exp_diff < 0 ? 0 : exp_diff);
    const uint32_t qs_r = (exp_diff > -1) ? ((m + (mk & down)) & ~mk) : 0u;
    const uint32_t qs = clip_sub(m, qs_r, p.subnormal_min_exponent_store);
    const uint32_t qn_r = p.round_bypass ? m : ((m + (p.round_mask & down)) & ~p.round_mask);
    const uint32_t qn = clip_magnitude<UnderflowMode::ZERO>(m, qn_r, p);
    return (te < p.min_exp) ? qs : ((te == 128) ? nonfinite(m, p) : qn);
  }

  // cast_binaryK_nearest_away, on the signed word as the shipped cast is
  inline uint32_t nearest_away(uint32_t w, const BinaryKParams &p)
  {
    const int te = target_exp(w);
    const int exp_diff = p.man_bits - (p.min_exp - te);
    const int k = exp_diff < -1 ? -1 : exp_diff;
    const uint32_t mk = mask_for(k);
    const uint32_t tie = (k >= 23) ? 0u : (uint32_t(1) << (22 - k));
    const uint32_t qs_r = (exp_diff >= -1) ? ((w + tie) & ~mk) : 0u;
    const uint32_t qs = clip_sub(w, qs_r, p.subnormal_min_exponent_store);
    const uint32_t qn_r = p.round_bypass ? w : ((w + p.round_tie) & ~p.round_mask);
    const uint32_t qn = clip_signed<UnderflowMode::NEAREST_AWAY>(w, qn_r, p);
    return (te < p.min_exp) ? qs : ((te == 128) ? nonfinite(w, p) : qn);
  }

  // cast_binaryK_odd for man_bits >= 1, on the signed word
  inline uint32_t odd(uint32_t w, const BinaryKParams &p)
  {
    const int te = target_exp(w);
    const int exp_diff = p.man_bits - (p.min_exp - te);
    const uint32_t mk = mask_for(exp_diff < 1 ? 1 : exp_diff);
    const uint32_t rounded = (w & ~mk) | (uint32_t((w & mk) != 0u) * (mk + 1u));
    const uint32_t qs_r = (exp_diff > 0) ? rounded : (w & ~MAN);
    const uint32_t qs = clip_sub_up(w, qs_r, p.subnormal_min_exponent_store);
    const uint32_t rm = p.round_mask;
    const uint32_t qn_r = p.round_bypass ? w : ((w & ~rm) | (uint32_t((w & rm) != 0u) * (rm + 1u)));
    const uint32_t qn = clip_signed<UnderflowMode::AWAY>(w, qn_r, p);
    return (te < p.min_exp) ? qs : ((te == 128) ? nonfinite(w, p) : qn);
  }
} // namespace binaryk_word

// Whether a format's casts in these modes may take the forms below: a signed
// format under SUBNORMALS with 1 to 22 stored mantissa bits.
inline bool word_cast_format(bool is_signed, SubnormalsMode subnormals, const BinaryKParams &p)
{
  return is_signed && subnormals == SubnormalsMode::SUBNORMALS && p.man_bits >= 1 &&
         p.man_bits <= 22;
}

// The cast of x to the format p describes, rounded as RM says, word for word
// the shipped cast for a format word_cast_format admits.
template <RoundMode RM>
inline float cast_binaryK_word(float x, const BinaryKParams &p)
{
  using namespace binaryk_word;
  const uint32_t w = word(x);
  if constexpr (RM == RoundMode::RNA)
    return value(nearest_away(w, p));
  else if constexpr (RM == RoundMode::RO)
    return value(odd(w, p));
  else
  {
    // The directed modes, as cast_binaryK_up, _down and _zero build them: an
    // input that compares >= 0 (-0.0 does) goes to one magnitude helper as
    // it is, any other (a negative value, or a NaN) to its counterpart with
    // its sign flipped, and comes back negated.
    const bool nonneg = x >= 0.0f;
    const uint32_t m = nonneg ? w : (w ^ SIGN);
    if constexpr (RM == RoundMode::RZ)
    {
      const uint32_t d = absolute_down(m, p);
      return value(nonneg ? d : negate(d));
    }
    else
    {
      // RU rounds a nonnegative input's magnitude up and a negative one's
      // down, RD the other way round
      static_assert(RM == RoundMode::RU || RM == RoundMode::RD, "RNA, RU, RD, RZ or RO");
      const uint32_t r = absolute_directed(m, (RM == RoundMode::RU) == nonneg, p);
      return value(nonneg ? r : negate(r));
    }
  }
}

#endif
