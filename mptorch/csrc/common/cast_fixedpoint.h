#pragma once

#include "bit_helper.h"
#include "modes.h"

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
// infinity beyond them. Stochastic rounding follows binaryK's subnormal SR arm
// instead: one shift puts every value under a single leading bit, where one
// round at a fixed position lands on the grid.

// The constants the casts read, flat for the reason BinaryKParamsT is (a
// nested aggregate in the GEMM instantiations costs nvcc minutes).
template <class T>
struct FixedPointParamsT
{
    int fl;                  // exp_diff = target_exp + fl, the bits a value keeps
    int step_exponent_store; // the carrier's exponent field of the step 2^-fl
    int mag_bits;            // the bits stochastic rounding keeps below its shift
    int shift_exp;           // mag_bits - fl: SR's shift, one binade above the largest value
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
// raise for a range reaching the carrier's top binade, so neither bound is
// past the carrier's largest value from Python; the infinity branches below
// are kept as a guard, since the carrier cannot hold such a bound.
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

    if (MPTORCH_IS_NEGATIVE(x) && !is_signed)
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

    if (MPTORCH_IS_NEGATIVE(x) && !is_signed)
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

    if (MPTORCH_IS_NEGATIVE(x) && !is_signed)
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

    if (MPTORCH_IS_NEGATIVE(x) && !is_signed)
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

    if (MPTORCH_IS_NEGATIVE(x) && !is_signed)
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

// Stochastic rounding: adds prng_bits random bits below the step and truncates
// (P3109's StochasticA). Like cast_binaryK_stochastic, rand_prob (a word of
// the carrier whose MAN_BITS low bits are the draw) and prng_bits stay per-call
// arguments. The cast is binaryK's subnormal SR arm over the whole range:
// adding 2^shift_exp, with the input's sign, puts every value in the binade
// whose grid at mag_bits is the step, so one stochastic round at mag_bits
// lands on it; the exact subtraction afterwards moves it back, and a value
// that rounded to zero comes back +0.0.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_stochastic(T x, typename FloatTraits<T>::word_t rand_prob, int prng_bits,
                                                     bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    if (MPTORCH_IS_NEGATIVE(x) && !is_signed)
        return T(0);

    word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    if (target_exp == F::INF_EXP)
        return fixedpoint_nonfinite(x, p);

    // keep only the prng_bits draw bits directly below the mag_bits kept ones
    rand_prob = rand_prob & F::MAN_MASK;
    rand_prob = rand_prob & ~((word_t(1) << (F::MAN_BITS - p.mag_bits - prng_bits)) - 1u);

    // A value at or past the shift is past the range: it saturates without
    // rounding, so the add below never leaves the shift's binade.
    if (target_exp >= p.shift_exp)
        return fixedpoint_saturate(x, p);

    word_t shift_bits = ((word_t)(p.shift_exp + F::BIAS) << F::MAN_BITS) | (target & F::SIGN_MASK);
    T shift = reinterpret_cast<const MPTORCH_THREAD T &>(shift_bits);
    T val = MPTORCH_ADD_TO_POWER_OF_TWO(x, shift);
    word_t val_bits = reinterpret_cast<const MPTORCH_THREAD word_t &>(val);
    word_t q = round_bitwise_stochastic(val_bits, rand_prob, p.mag_bits);
    T quantized = reinterpret_cast<const MPTORCH_THREAD T &>(q) - shift;

    return fixedpoint_saturate(quantized, p);
}
