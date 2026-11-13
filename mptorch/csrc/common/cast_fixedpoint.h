#pragma once

#include "bit_helper.h"
#include "modes.h"
#include "stochastic.h"

// The fixed-point format cast, one function per rounding mode, in the same
// shape as cast_binaryK.h and cast_superfp.h: each reads a FixedPointParamsT
// built once per tensor, and is a template on the carrier (bit_helper.h's
// FloatTraits).
//
// A fixed-point format holds the multiples of its step 2^-fl that a wl-bit
// integer holds, so it is a float that is subnormal everywhere: one fixed step
// from zero to the largest value. Each cast is therefore binaryK's subnormal
// arm (cast_binaryK.h) applied to the whole range. A value whose leading bit
// is 2^target_exp keeps exp_diff = target_exp + fl bits below it, which is
// binaryK's man_bits - (min_exp - target_exp) with the step in place of its
// smallest subnormal; a value under the step (exp_diff < 0) is zero or the
// step; and the result saturates to the ends of the range, which have no
// infinity beyond them. Stochastic rounding is integer arithmetic on the same
// two cases, the draw added directly below the step, so it is P3109's
// StochasticA exactly (see cast_fixedpoint_stochastic).

// The constants the casts read, flat for the reason BinaryKParamsT is (a
// nested aggregate in the GEMM instantiations costs nvcc minutes).
template <class T>
struct FixedPointParamsT
{
    int fl;                  // exp_diff = target_exp + fl, the bits a value keeps
    int step_exponent_store; // the carrier's exponent field of the step 2^-fl
    int mag_bits;            // wl - is_signed, the bits of a magnitude code
    int shift_exp;           // mag_bits - fl: one binade above the largest value, where SR saturates
    T max_val;               // the largest value, which +inf and overflow saturate to
    T min_val;               // the smallest: 0 unsigned, -max_val symmetric, else -2^(mag_bits - fl)
};

using FixedPointParams = FixedPointParamsT<float>;

// The carrier value whose word is `w`.
template <class T>
CUDA_HOST_DEVICE_INLINE T fixedpoint_word_value(typename FloatTraits<T>::word_t w)
{
    return reinterpret_cast<const MPTORCH_THREAD T &>(w);
}

// Builds the constants once per tensor. The two bounds are built on the word,
// not by float arithmetic, so they are exact on every backend: the largest
// value is mag_bits ones, a leading 1 at 2^top_exp and mag_bits - 1 ones
// below it, and the two's complement bottom is the power of two 2^min_exp.
// The number.py checks hold both inside the carrier's normals from below, and
// raise for an end of the range past the carrier's largest value, so neither
// bound is past it from Python; the infinity branches below are kept as a
// guard, since the carrier cannot hold such a bound.
template <class T = float>
CUDA_HOST_DEVICE_INLINE FixedPointParamsT<T> make_fixedpoint_params(int wl, int fl, bool is_signed, bool symmetric)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    const int mag_bits = wl - (is_signed ? 1 : 0);
    const int top_exp = mag_bits - 1 - fl; // the largest value's leading bit
    const int min_exp = mag_bits - fl;     // the two's complement bottom is -2^min_exp

    FixedPointParamsT<T> p;
    p.fl = fl;
    p.step_exponent_store = F::BIAS - fl;
    p.mag_bits = mag_bits;
    p.shift_exp = min_exp;

    if (top_exp > F::MAX_EXP)
        p.max_val = fixedpoint_word_value<T>(F::INF_BITS);
    else
    {
        const word_t exponent = (word_t)(top_exp + F::BIAS) << F::MAN_BITS;
        const word_t ones = (((word_t)1 << (mag_bits - 1)) - 1) << (F::MAN_BITS - (mag_bits - 1));
        p.max_val = fixedpoint_word_value<T>(exponent | ones);
    }

    if (!is_signed)
        p.min_val = T(0);
    else if (symmetric)
        p.min_val = -p.max_val;
    else if (min_exp > F::MAX_EXP)
        p.min_val = fixedpoint_word_value<T>(F::SIGN_MASK | F::INF_BITS);
    else
        p.min_val = fixedpoint_word_value<T>(F::SIGN_MASK | ((word_t)(min_exp + F::BIAS) << F::MAN_BITS));

    return p;
}

// A rounded value past either end becomes that end. Rounding can carry a value
// one step past the top (1.875 to 2.0 in wl=4, fl=2), and a value far above
// the range is left unrounded, since its exp_diff reaches the carrier's
// mantissa, and lands here.
template <class T>
CUDA_HOST_DEVICE_INLINE T fixedpoint_saturate(T quantized, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    if (quantized > p.max_val)
        return p.max_val;
    if (quantized < p.min_val)
        return p.min_val;
    return quantized;
}

// A NaN or an infinity, the words with an all-ones exponent field: a NaN (a
// nonzero mantissa) passes through, and an infinity saturates to the end of
// the range on its side, since a fixed-point word has no code for one.
template <class T>
CUDA_HOST_DEVICE_INLINE T fixedpoint_nonfinite(T x, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    const word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    if ((target & F::MAN_MASK) != 0)
        return x;
    return (target & F::SIGN_MASK) ? p.min_val : p.max_val;
}

// Rounds to nearest, ties to even: the NearestTiesToEven projection of x onto
// the fixed-point format p describes, where "even" is the parity of the
// integer code. A negative input to an unsigned format returns 0, a NaN passes
// through, and every other value is binaryK's subnormal arm with exp_diff =
// target_exp + fl, saturated to the range.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_nearest_even(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    if (MPTORCH_NEGATIVE_TO_UNSIGNED(x, is_signed))
        return T(0);

    word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    if (target_exp == F::INF_EXP)
        return fixedpoint_nonfinite(x, p);

    // the bits kept below the leading one, down to the step: negative for a
    // value under the step, which is then zero or the step. Half a step is
    // the tie, and zero the even code, so only a value above half leaves zero.
    int exp_diff = target_exp + p.fl;
    int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << (F::EXP_BITS + 1)) > 0));

    // Rounded only where the result is nonzero: binaryK's subnormal arm never
    // sees an exp_diff below -2, but a tiny input's here would shift the word
    // past its width, which C++ leaves undefined.
    word_t quantize_bits = not_uflow ? round_bitwise_nearest_even(target, exp_diff) : word_t(0);
    quantize_bits = clip_subnormal_range_exponent(target, quantize_bits, p.step_exponent_store);
    T quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);

    return fixedpoint_saturate(quantized, p);
}

// Rounds to nearest, ties away from zero. Under the step, half a step is a tie
// the step wins, so every value from exp_diff == -1 up is nonzero.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_nearest_away(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    if (MPTORCH_NEGATIVE_TO_UNSIGNED(x, is_signed))
        return T(0);

    word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    if (target_exp == F::INF_EXP)
        return fixedpoint_nonfinite(x, p);

    int exp_diff = target_exp + p.fl;
    int not_uflow = exp_diff >= -1;

    word_t quantize_bits = not_uflow ? round_bitwise_nearest_away(target, exp_diff) : word_t(0);
    quantize_bits = clip_subnormal_range_exponent(target, quantize_bits, p.step_exponent_store);
    T quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);

    return fixedpoint_saturate(quantized, p);
}

// Rounds to odd. Under the step the candidates are zero (code 0) and the step
// (code 1), so every nonzero value there rounds to the step: the power of two
// under it is taken, and the _up clip lifts it onto the step. At exp_diff ==
// 0 that power of two is the step itself, code 1, already odd.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_odd(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    if (MPTORCH_NEGATIVE_TO_UNSIGNED(x, is_signed))
        return T(0);

    word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    if (target_exp == F::INF_EXP)
        return fixedpoint_nonfinite(x, p);

    int exp_diff = target_exp + p.fl;

    word_t quantize_bits = (exp_diff > 0) ? round_bitwise_odd(target, exp_diff) : (target & ~F::MAN_MASK);
    quantize_bits = clip_subnormal_range_exponent_up(target, quantize_bits, p.step_exponent_store);
    T quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);

    return fixedpoint_saturate(quantized, p);
}

// Rounds a magnitude up (away from zero), for cast_fixedpoint_up and
// cast_fixedpoint_down: assumes x >= 0 and finite. Under the step the value
// rounds up to a whole power of two, which the _up clip lifts onto the step.
template <class T>
CUDA_HOST_DEVICE_INLINE T fixedpoint_absolute_up(T x, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;
    int exp_diff = target_exp + p.fl;

    word_t quantize_bits = round_bitwise_up(target, exp_diff < 0 ? 0 : exp_diff);
    quantize_bits = clip_subnormal_range_exponent_up(target, quantize_bits, p.step_exponent_store);
    return reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
}

// Rounds a magnitude down (toward zero); the other half of
// fixedpoint_absolute_up.
template <class T>
CUDA_HOST_DEVICE_INLINE T fixedpoint_absolute_down(T x, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;
    int exp_diff = target_exp + p.fl;

    word_t quantize_bits = (exp_diff >= 0) ? round_bitwise_down(target, exp_diff) : word_t(0);
    quantize_bits = clip_subnormal_range_exponent(target, quantize_bits, p.step_exponent_store);
    return reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
}

// The three directed modes (toward +inf, toward -inf, toward zero) in terms
// of the two magnitude helpers above, as in cast_binaryK_up: rounding a
// negative value up is rounding its magnitude down, and so on. Unlike
// binaryK's, they saturate after the sign is back on, since the range is not
// symmetric: -2 is a value of wl=4, fl=2, and +2 is not.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_up(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    if (MPTORCH_NEGATIVE_TO_UNSIGNED(x, is_signed))
        return T(0);

    // A NaN or an infinity is settled before the sign is taken off, so the
    // helpers only ever see a finite magnitude. The sign goes off and back on
    // on the word (flip_sign, negate_magnitude), so a magnitude that rounded
    // to zero comes back +0.0 and no payload is canonicalized.
    word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    if ((int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS == F::INF_EXP)
        return fixedpoint_nonfinite(x, p);

    T rounded = MPTORCH_IS_NONNEGATIVE(x) ? fixedpoint_absolute_up(x, p)
                                          : negate_magnitude(fixedpoint_absolute_down(flip_sign(x), p));
    return fixedpoint_saturate(rounded, p);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_down(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    if (MPTORCH_NEGATIVE_TO_UNSIGNED(x, is_signed))
        return T(0);

    // see cast_fixedpoint_up
    word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    if ((int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS == F::INF_EXP)
        return fixedpoint_nonfinite(x, p);

    T rounded = MPTORCH_IS_NONNEGATIVE(x) ? fixedpoint_absolute_down(x, p)
                                          : negate_magnitude(fixedpoint_absolute_up(flip_sign(x), p));
    return fixedpoint_saturate(rounded, p);
}

// Rounds toward zero: down for a non-negative value, up for a negative one.
// A NaN compares false and takes the up arm, which passes it through.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_zero(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    if (MPTORCH_IS_NONNEGATIVE(x))
        return cast_fixedpoint_down(x, is_signed, p);
    return cast_fixedpoint_up(x, is_signed, p);
}

// Stochastic rounding with N random bits: P3109's StochasticA, B or C, as the
// spec in prng_bits says (common/stochastic.h; R is the top N of the draw's
// MAN_BITS bits, so N = 0 makes A RZ). With eta the fraction of |x| between
// its two neighbours on the grid of step 2^-fl, the magnitude rounds away
// from zero iff floor(eta * 2^N) + R >= 2^N (A), floor(eta * 2^(N+1)) + 2R +
// 1 >= 2^(N+1) (B) or RNITE(eta * 2^N) + R >= 2^N (C). rand_prob (a word of
// the carrier whose MAN_BITS low bits are the draw) and the spec stay
// per-call arguments, as in cast_binaryK_stochastic.
//
// The arithmetic is integer, on the word. A value at or above the step keeps
// exp_diff fraction bits above it: the draw goes in directly below the step
// with the variant's constants below it, the bits below the step are cleared,
// and a carry crosses a binade by itself. A value under the step is zero or
// the step, by stochastic.h's bit test on its significand. Nothing is rounded
// in the carrier first -- adding a power of two above the range, as binaryK's
// subnormal SR arm once did, rounds |x| to nearest at MAN_BITS - mag_bits bits
// below the step before the draw sees it, which made N = 0 RNE at mag_bits =
// MAN_BITS -- and no float operation touches the input, so a subnormal is not
// flushed on Metal. The carrier check's bound, mag_bits + N <= MAN_BITS + 1
// (number.py), keeps every shift in range: in the range exp_diff <= mag_bits
// - 1, so a value at or above the step has at least N fraction bits, and
// exactly N where the bound is met (then B and C are A).
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_stochastic(T x, typename FloatTraits<T>::word_t rand_prob, int spec,
                                                     bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    if (MPTORCH_NEGATIVE_TO_UNSIGNED(x, is_signed))
        return T(0);

    word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    word_t mag = target & F::ABS_MASK;
    word_t sign = target & F::SIGN_MASK;
    int field = (int)(mag >> F::MAN_BITS);
    int target_exp = field - F::BIAS;

    if (target_exp == F::INF_EXP)
        return fixedpoint_nonfinite(x, p);
    if (mag == 0)
        return T(0); // either zero comes back +0.0

    // A value at or past 2^shift_exp is past the range: it saturates.
    if (target_exp >= p.shift_exp)
        return fixedpoint_saturate(x, p);

    const int prng_bits = sr_spec_bits(spec);
    const int variant = sr_spec_variant(spec);
    word_t r = (rand_prob & F::MAN_MASK) >> (F::MAN_BITS - prng_bits);

    // The fraction bits of |x| / step sit below bit `below` of the word. A
    // subnormal input's field reads as the exponent 1 - BIAS.
    int exp_diff = (field == 0 ? 1 - F::BIAS : target_exp) + p.fl;
    if (exp_diff >= F::MAN_BITS)
        return fixedpoint_saturate(x, p); // a multiple of the step already

    if (exp_diff >= 0)
    {
        int below = F::MAN_BITS - exp_diff;
        word_t or_v, tie;
        sr_word_constants<word_t>(below - prng_bits, prng_bits, variant, or_v, tie);
        word_t q = sr_round_word<word_t>(mag, r, below - prng_bits, prng_bits, or_v, tie);
        // zero only for a subnormal input under a step the carrier holds as a
        // subnormal (fl >= BIAS): the sign goes back on a nonzero result only
        if (q == 0)
            return T(0);
        word_t signed_q = q | sign;
        return fixedpoint_saturate(reinterpret_cast<const MPTORCH_THREAD T &>(signed_q), p);
    }

    // Under the step: |x| / step = s * 2^(exp_diff - MAN_BITS), so s has
    // MAN_BITS - exp_diff bits below the step.
    word_t s = (mag & F::MAN_MASK) | (field == 0 ? word_t(0) : (word_t(1) << F::MAN_BITS));
    bool sticky;
    const word_t top2 = sr_top2<word_t>(s, F::MAN_BITS - exp_diff, prng_bits, F::WORD_BITS, sticky);
    if (!sr_away<word_t>(top2, sticky, r, prng_bits, variant))
        return T(0);
    // the step on the word, a subnormal of the carrier past fl = BIAS - 1
    word_t step = p.step_exponent_store >= 1 ? (word_t)p.step_exponent_store << F::MAN_BITS
                                             : word_t(1) << (F::MAN_BITS - 1 + p.step_exponent_store);
    word_t signed_step = step | sign;
    return fixedpoint_saturate(reinterpret_cast<const MPTORCH_THREAD T &>(signed_step), p);
}
