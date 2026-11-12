#pragma once

// The (man_bits, exp_bits, bias, ...) spelling of every format cast, and the
// clip and rounding helpers only it calls.
//
// This is the arithmetic mptorch shipped before c53cde8 and 9f286f5 (the CPU
// and CUDA elementwise quantizers) replaced the per-call derivation with a
// BinaryKParams/SuperfpParams computed once, as the GEMMs already did at
// Multiplier/Adder construction. Every kernel in mptorch/csrc now calls the
// precomputed spelling; the only thing that still calls this one is the
// "original" arm of the exhaustive sweeps in this directory, which is exactly
// what it is for -- an independently written reference the fast paths are
// checked against over all 2^32 float inputs, not a second implementation the
// library dispatches to. Keeping it here rather than in the shipped headers
// says so, and takes ~800 lines out of three headers that every translation
// unit in the extension parses.
//
// Two consequences worth stating, because they are what make the arrangement
// safe:
//
//   * A reference that shares code with the thing it checks is not a
//     reference. These bodies derive their constants inline, from the format
//     arguments, the way the code before c53cde8 did; the shipped ones read
//     them out of a struct built by make_binaryK_params/make_superfp_params.
//     The sweeps compare the two, so the derivation is under test on both
//     sides rather than assumed on one.
//   * Nothing here may drift silently. The sweeps are the check: any change
//     to a shipped cast that this file does not match shows up as a mismatch
//     count, not as a passing test. If a format's semantics genuinely change,
//     both spellings change together.
//
// Like the shipped casts, every cast here is a template on its carrier, binary32
// or binary64, and for the same reason that the rest of this file is written
// apart from them, the carrier is too. Nothing below reads bit_helper.h's
// FloatTraits: the carrier is its two field widths, written once in RefTraits,
// and every constant a cast needs is derived from them where it is used --
// where FloatTraits writes each one out as a literal -- so a wrong constant on
// either side is a mismatch rather than a shared mistake. The pun is memcpy
// where the shipped casts' is a reinterpret_cast, and the bitwise rounds are
// this file's own rather than bit_helper.h's, which the float-only version of
// this file used to share.
// What is still shared is the UnderflowMode enum, which is a name, and
// superfp_region_cutoffs (see the superfp section).
//
// Include path: these files build with `-I mptorch/csrc` (see each one's
// header comment), so "common/..." resolves; temp/'s Makefile passes it too.

#include "common/bit_helper.h"
#include "common/cast_superfp.h"
#include "common/modes.h"
#include <cmath>
#include <cstring>

// The binary32 pun, spelled as the code before c53cde8 spelled it: on an
// lvalue for the macros, on a value for bits_to_float. The shipped headers
// spell the pun at each use (bit_helper.h) and no longer define these; the
// sweeps that include this file still use them.
#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

CUDA_HOST_DEVICE_INLINE float bits_to_float(uint32_t bits)
{
    return BITS_TO_FLOAT(&bits);
}

// ---------------------------------------------------------------------------
// The carrier, as the reference writes it down: two widths, and nothing else.
// ---------------------------------------------------------------------------
template <class T>
struct RefTraits; // binary32 and binary64 only

template <>
struct RefTraits<float>
{
    using word_t = uint32_t;
    static constexpr int MAN = 23;
    static constexpr int EXP = 8;
};

template <>
struct RefTraits<double>
{
    using word_t = uint64_t;
    static constexpr int MAN = 52;
    static constexpr int EXP = 11;
};

// Everything else, derived.
template <class T>
struct Ref
{
    using word_t = typename RefTraits<T>::word_t;
    static constexpr int MAN = RefTraits<T>::MAN;
    static constexpr int EXP = RefTraits<T>::EXP;
    static constexpr int BIAS = (1 << (EXP - 1)) - 1;
    static constexpr int ALL_ONES = (1 << EXP) - 1; // the exponent field of an infinity or a NaN
    static constexpr int SPECIAL = ALL_ONES - BIAS; // ... unbiased
    static constexpr word_t ONE = 1;
    static constexpr word_t MAN_MASK = (ONE << MAN) - 1;
    static constexpr word_t SIGN = ONE << (MAN + EXP);
    static constexpr word_t ABS = SIGN - 1;
    static constexpr word_t INF = (word_t)ALL_ONES << MAN;

    CUDA_HOST_DEVICE_INLINE static word_t bits(T value)
    {
        word_t w;
        memcpy(&w, &value, sizeof w);
        return w;
    }
    CUDA_HOST_DEVICE_INLINE static T value(word_t w)
    {
        T v;
        memcpy(&v, &w, sizeof v);
        return v;
    }
    CUDA_HOST_DEVICE_INLINE static int field(word_t w) { return (int)((w >> MAN) & (word_t)ALL_ONES); }
};

// ---------------------------------------------------------------------------
// The zero.
//
// P3109's zero is code point 0 and there is no other: the standard has no
// negative zero, and superfp spends no code on one. Whatever sign the rounding
// below leaves on a zero, the value it stands for is +0.0, so every cast here
// returns through this. It tests the value where the shipped casts'
// unsigned_zero tests the carrier's word, so the two have to agree on what a
// zero is from different ends (a reference has to be independent of the
// shipped casts in its rules, not only in its code).
// ---------------------------------------------------------------------------
template <class T>
CUDA_HOST_DEVICE_INLINE T reference_zero(T value)
{
    return (value == T(0)) ? T(0) : value;
}

// ---------------------------------------------------------------------------
// Bitwise rounding to man_bits significand bits, on the carrier's word.
//
// A negative man_bits reaches these from a subnormal arm, where it counts
// binades below the smallest normal; every caller tests for the underflow
// before it rounds rather than after, so none of these shifts ever runs past
// the word.
// ---------------------------------------------------------------------------
template <class T>
CUDA_HOST_DEVICE_INLINE typename Ref<T>::word_t ref_round_nearest_even(typename Ref<T>::word_t target, int man_bits)
{
    using R = Ref<T>;
    if (man_bits >= R::MAN)
        return target;
    const int dropped = R::MAN - man_bits;
    typename R::word_t mask = (R::ONE << dropped) - 1;
    typename R::word_t half = R::ONE << (dropped - 1);
    typename R::word_t rounded = (target + half) & ~mask;
    // at a tie the kept bit goes back to zero -- except at man_bits == 0,
    // where the kept bit is the implicit one, which is odd, and the carry
    // stands (a negative man_bits, from a subnormal arm, clears like any other)
    if ((target & mask) == half && man_bits != 0)
        rounded &= ~(R::ONE << dropped);
    return rounded;
}

// To the nearest power of two, ties to the power whose carrier exponent field
// is odd.
template <class T>
CUDA_HOST_DEVICE_INLINE typename Ref<T>::word_t ref_round_nearest_even_pow2(typename Ref<T>::word_t target)
{
    using R = Ref<T>;
    typename R::word_t half = R::ONE << (R::MAN - 1);
    typename R::word_t rounded = (target + half) & ~R::MAN_MASK;
    if ((target & R::MAN_MASK) == half && !((rounded >> R::MAN) & R::ONE))
        rounded -= R::ONE << R::MAN;
    return rounded;
}

template <class T>
CUDA_HOST_DEVICE_INLINE typename Ref<T>::word_t ref_round_nearest_away(typename Ref<T>::word_t target, int man_bits)
{
    using R = Ref<T>;
    if (man_bits >= R::MAN)
        return target;
    const int dropped = R::MAN - man_bits;
    return (target + (R::ONE << (dropped - 1))) & ~((R::ONE << dropped) - 1);
}

template <class T>
CUDA_HOST_DEVICE_INLINE typename Ref<T>::word_t ref_round_up(typename Ref<T>::word_t target, int man_bits)
{
    using R = Ref<T>;
    if (man_bits >= R::MAN)
        return target;
    typename R::word_t mask = (R::ONE << (R::MAN - man_bits)) - 1;
    bool negative = (target & R::SIGN) != 0;
    return (target + (negative ? 0 : mask)) & ~mask;
}

template <class T>
CUDA_HOST_DEVICE_INLINE typename Ref<T>::word_t ref_round_down(typename Ref<T>::word_t target, int man_bits)
{
    using R = Ref<T>;
    if (man_bits >= R::MAN)
        return target;
    typename R::word_t mask = (R::ONE << (R::MAN - man_bits)) - 1;
    bool negative = (target & R::SIGN) != 0;
    return (target + (negative ? mask : 0)) & ~mask;
}

template <class T>
CUDA_HOST_DEVICE_INLINE typename Ref<T>::word_t ref_round_odd(typename Ref<T>::word_t target, int man_bits)
{
    using R = Ref<T>;
    if (man_bits >= R::MAN)
        return target;
    typename R::word_t mask = (R::ONE << (R::MAN - man_bits)) - 1;
    typename R::word_t kept = target & ~mask;
    return ((target & mask) != 0) ? (kept | (mask + 1)) : kept;
}

// ---------------------------------------------------------------------------
// Stochastic rounding, from P3109's definition (arXiv:2606.04028, Fig. 2).
//
// With eta the fraction of |x| between its two neighbours lo < hi, X =
// eta * 2^N and R the draw's N-bit field, the magnitude goes to hi iff
//
//   StochasticA   floor(X) + R >= 2^N
//   StochasticB   floor(2X) + 2R + 1 >= 2^(N+1)
//   StochasticC   RNITE(X) + R >= 2^N      (RNITE: to nearest, ties to even)
//
// Written as the figure is: floor(X), floor(2X) and RNITE(X) are computed as
// numbers -- by shifts where |x| - lo is an integer multiple of a power of
// two of hi - lo, by exact integer compares where hi - lo is not a power of
// two (an EXTENDED_NORMALS floor) -- and compared with R as the figure
// compares them. The shipped casts (common/stochastic.h) instead fold the
// three into one word add and one bit test, so a mistake in either shows up
// as a mismatch. A variant is the spec's (0 A, 1 B, 2 C); N and R are the
// cast's to pass in.
// ---------------------------------------------------------------------------

// floor(X), floor(2X) and RNITE(X) for X = rem * 2^(N - d), rem < 2^d an
// integer (the fraction of a split on a power-of-two grid, d of its bits
// below the step).
struct RefSrX
{
    unsigned long long floor_x, floor_2x, rnite_x;
};

CUDA_HOST_DEVICE_INLINE RefSrX ref_sr_x_bits(unsigned long long rem, int d, int n)
{
    using u64 = unsigned long long;
    auto shift_right = [](u64 v, int s) -> u64 { return s >= 64 ? 0ull : (s <= 0 ? v << (-s) : v >> s); };
    RefSrX x;
    x.floor_x = shift_right(rem, d - n);
    x.floor_2x = shift_right(rem, d - n - 1);
    // the part of X below its integer, against one half
    x.rnite_x = x.floor_x;
    if (d - n >= 1)
    {
        const u64 below = d - n >= 64 ? rem : rem & ((1ull << (d - n)) - 1ull);
        if (d - n - 1 < 64)
        {
            const u64 half = 1ull << (d - n - 1);
            if (below > half || (below == half && (x.floor_x & 1ull)))
                x.rnite_x += 1;
        }
    }
    return x;
}

// sign(a * 2^ea - b * 2^eb) for a, b < 2^63, exactly.
CUDA_HOST_DEVICE_INLINE int ref_cmp_scaled(unsigned long long a, int ea, unsigned long long b, int eb)
{
    if (a == 0 || b == 0)
        return (a != 0) - (b != 0);
    int la = 0, lb = 0;
    for (unsigned long long v = a; v; v >>= 1)
        ++la;
    for (unsigned long long v = b; v; v >>= 1)
        ++lb;
    if (la + ea != lb + eb)
        return la + ea > lb + eb ? 1 : -1;
    // the same top bit: shift the one with the larger exponent onto the other
    if (ea > eb)
        a <<= (ea - eb);
    else
        b <<= (eb - ea);
    return a > b ? 1 : (a < b ? -1 : 0);
}

// floor(X), floor(2X) and RNITE(X) for X = (m * 2^em) * 2^N / (qm * 2^eq),
// X < 2^N, by search over the integers: what an EXTENDED_NORMALS floor needs,
// whose smallest value qm * 2^eq is not a power of two.
CUDA_HOST_DEVICE_INLINE RefSrX ref_sr_x_scaled(unsigned long long m, int em, unsigned long long qm, int eq, int n)
{
    using u64 = unsigned long long;
    // the largest c in [0, cap] with c * (q * 2^-N / 2^j) <= m * 2^em, that
    // is floor(2^j X)
    auto floor_of = [&](int j, u64 cap) -> u64
    {
        u64 lo = 0, hi = cap;
        while (lo < hi)
        {
            const u64 mid = lo + (hi - lo + 1) / 2;
            if (ref_cmp_scaled(m, em + n + j, mid * qm, eq) >= 0)
                lo = mid;
            else
                hi = mid - 1;
        }
        return lo;
    };
    RefSrX x;
    x.floor_x = floor_of(0, 1ull << n);
    x.floor_2x = floor_of(1, 2ull << n);
    // RNITE: X against floor(X) + 1/2, that is 2X against 2 floor(X) + 1
    const int c = ref_cmp_scaled(m, em + n + 1, (2 * x.floor_x + 1) * qm, eq);
    x.rnite_x = x.floor_x + ((c > 0 || (c == 0 && (x.floor_x & 1ull))) ? 1 : 0);
    return x;
}

// The figure's three tests.
CUDA_HOST_DEVICE_INLINE bool ref_sr_up(const RefSrX &x, unsigned long long r, int n, int variant)
{
    if (variant == 1)
        return x.floor_2x + 2 * r + 1 >= (2ull << n);
    if (variant == 2)
        return x.rnite_x + r >= (1ull << n);
    return x.floor_x + r >= (1ull << n);
}

// The draw's N-bit field directly below a format's man_bits significand bits.
template <class T>
CUDA_HOST_DEVICE_INLINE unsigned long long ref_sr_field(typename Ref<T>::word_t rand_prob, int man_bits, int n)
{
    using R = Ref<T>;
    return (unsigned long long)(((rand_prob & R::MAN_MASK) >> (R::MAN - man_bits - n)) & ((R::ONE << n) - 1));
}

// |x| split on a grid of step 2^q: |x| / 2^q = whole + rem / 2^d. The
// significand s and exponent of the input's word, with a subnormal's field
// read as 1 - BIAS.
struct RefSplit
{
    unsigned long long whole, rem;
    int d;
};

template <class T>
CUDA_HOST_DEVICE_INLINE RefSplit ref_split(typename Ref<T>::word_t w, int q)
{
    using R = Ref<T>;
    using u64 = unsigned long long;
    const int field = R::field(w);
    const u64 s = (u64)(w & R::MAN_MASK) | (field ? (1ull << R::MAN) : 0ull);
    const int t = (field ? field : 1) - R::BIAS - R::MAN - q; // |x| / 2^q = s * 2^t
    RefSplit sp;
    if (t >= 0)
    {
        sp.whole = t < 64 ? s << t : 0ull; // an input past the grid's reach is on it
        sp.rem = 0;
        sp.d = 0;
    }
    else
    {
        sp.d = -t;
        sp.whole = sp.d < 64 ? s >> sp.d : 0ull;
        sp.rem = sp.d < 64 ? s - (sp.whole << sp.d) : s;
    }
    return sp;
}

// whole * 2^q as a value of the carrier.
template <class T>
CUDA_HOST_DEVICE_INLINE T ref_grid_value(unsigned long long whole, int q)
{
    return (T)ldexp((double)whole, q);
}

// ---------------------------------------------------------------------------
// Exponent clipping, integer-argument forms.
//
// bit_helper.h keeps the precomputed siblings (and the
// SubnormalRangeParams/NormalRangeParams that feed them); these three derive
// min_exponent_store / max_num from the format on every call, which is the
// difference the sweeps are measuring.
// ---------------------------------------------------------------------------

// clips the exponent of a floating point format with subnormal values
template <class T>
CUDA_HOST_DEVICE_INLINE typename Ref<T>::word_t ref_clip_subnormal(typename Ref<T>::word_t old_num,
                                                                   typename Ref<T>::word_t quantized_num,
                                                                   int exp_bits, int man_bits, int bias)
{
    using R = Ref<T>;
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = R::field(quantized_num);
    int min_exponent_store = -(bias - 1) - man_bits + R::BIAS;

    typename R::word_t old_sign = old_num & R::SIGN;
    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
    {
        if (quantized_exponent_store == min_exponent_store - 1)
            quantized_num = (quantized_num + (R::ONE << R::MAN)) | old_sign;
        else
            quantized_num = 0;
    }

    return quantized_num;
}

template <class T>
CUDA_HOST_DEVICE_INLINE typename Ref<T>::word_t ref_clip_subnormal_up(typename Ref<T>::word_t old_num,
                                                                      typename Ref<T>::word_t quantized_num,
                                                                      int exp_bits, int man_bits, int bias)
{
    using R = Ref<T>;
    // a zero input stays zero; a nonzero one never rounds into it here
    if ((old_num & R::ABS) == 0)
        return quantized_num;

    int quantized_exponent_store = R::field(quantized_num);
    int min_exponent_store = -(bias - 1) - man_bits + R::BIAS;

    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
        quantized_num = ((typename R::word_t)min_exponent_store << R::MAN) | (old_num & R::SIGN);

    return quantized_num;
}

// P3109's SatFinite, in either domain: every result is finite, an infinite
// input included. SAT_FINITE is it in the finite domain, SAT_FINITE_EXTENDED in
// the extended one, which keeps the +infinity code (and so the lower largest
// finite value) that SAT_NONE and SAT_PROPAGATE have.
CUDA_HOST_DEVICE_INLINE bool ref_sat_finite(SaturationMode saturation_mode)
{
    return saturation_mode == SaturationMode::SAT_FINITE || saturation_mode == SaturationMode::SAT_FINITE_EXTENDED;
}

// clips the exponent of a floating point format without subnormal values (binaryK version)
//
// The top of the range is IEEE P3109's (arXiv:2606.04028), derived from its
// code points rather than from an exponent and a mask: the non-negative codes
// are 0 .. 2^(exp_bits + man_bits) - 1, code c decodes to exponent field
// c >> man_bits and trailing significand c mod 2^man_bits, and the top one or
// two are not finite -- +infinity outside SAT_FINITE (the extended domain), and
// NaN above it when `nan_code` is set (an unsigned binaryK format; superfp
// passes false). The largest finite value is the highest code left, decoded in
// value space, and a finite result saturates against its magnitude. Counted in
// uint64_t, so exp_bits + man_bits stops at 63 -- binary64 itself -- which every
// sweep grid respects.
template <UnderflowMode U, class T>
CUDA_HOST_DEVICE_INLINE typename Ref<T>::word_t ref_clip_normal(typename Ref<T>::word_t old_num,
                                                                typename Ref<T>::word_t quantized_num,
                                                                int exp_bits, int man_bits, int bias,
                                                                SaturationMode saturation_mode, bool nan_code,
                                                                bool extended_normals = false,
                                                                typename Ref<T>::word_t rand_prob = 0)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    if ((old_num & R::ABS) == 0)
        return 0; // a zero input rounds to zero, and P3109's zero is unsigned

    word_t sign = old_num & R::SIGN;
    quantized_num &= R::ABS;
    bool nonfinite_input = R::field(old_num) == R::ALL_ONES;
    if (nonfinite_input && quantized_num > R::INF)
        return sign | quantized_num; // NaN
    if (nonfinite_input && !ref_sat_finite(saturation_mode))
        return sign | quantized_num; // an infinite input stays infinite outside SatFinite

    uint64_t reserved = (saturation_mode != SaturationMode::SAT_FINITE ? 1 : 0) + (nan_code ? 1 : 0);
    uint64_t code = ((uint64_t)1 << (exp_bits + man_bits)) - 1 - reserved;
    int64_t e_field = (int64_t)(code >> man_bits);
    uint64_t trailing = code - ((uint64_t)e_field << man_bits);
    int64_t e = e_field - bias;
    word_t max_num = 0; // no finite normal datum: a format P3109 does not define
    if (e_field >= 1 && e > R::BIAS)
    {
        // beyond the carrier: its largest value on the format's grid
        word_t largest = R::INF - 1;
        max_num = largest & ~((man_bits >= R::MAN) ? word_t(0) : ((R::ONE << (R::MAN - man_bits)) - 1));
    }
    else if (e_field >= 1 && e >= 1 - R::BIAS)
    {
        // exact in binary64 for man_bits <= 52, and so for either carrier
        double significand = 1.0 + (double)trailing / (double)((uint64_t)1 << man_bits);
        max_num = R::bits((T)ldexp(significand, (int)e));
    }

    // under SAT_FINITE an infinity is an overflow like any other
    if (nonfinite_input)
        return sign | max_num;

    // The bottom of the range, as a value rather than as an exponent field.
    // `extended_normals` lowers the floor by a binade, but not onto that
    // binade's mantissa-zero code: that code is the format's zero, and with
    // the sign bit its NaN, as in every binaryK format, so the values start
    // one step above it -- and at man_bits == 0 the step carries into the next
    // binade, which is the same floor NORMALS has.
    //
    // The floor takes part only if it is one of the carrier's normals, and
    // that is asked of the value, after the step is added: a floor below the
    // carrier's normals, or past its largest finite value, leaves nothing to
    // compare against.
    const int foot = 1 - bias - (extended_normals ? 1 : 0); // the exponent of the floor's binade
    double lo = ldexp(1.0, foot);
    if (extended_normals)
        lo += ldexp(1.0, foot - (man_bits < R::MAN ? man_bits : R::MAN));
    word_t min_num = 0, half_num = 0;
    if (lo >= ldexp(1.0, 1 - R::BIAS) && lo < ldexp(1.0, R::BIAS + 1))
    {
        min_num = R::bits((T)lo);
        half_num = R::bits((T)(lo * 0.5));
    }

    // handle overflow
    if (quantized_num > max_num)
        quantized_num = (saturation_mode == SaturationMode::SAT_NONE) ? word_t(R::INF) : max_num;
    // handle underflow: the two candidates are zero and the format's smallest
    // value, and the mode picks between them as it does one region lower,
    // between zero and the smallest subnormal. UnderflowMode::NONE has no
    // floor: the stochastic casts decide theirs before the clip, and a
    // superfp's normal binades can start below this floor.
    else if (U != UnderflowMode::NONE && quantized_num < min_num)
    {
        word_t ax = old_num & R::ABS;
        if constexpr (U == UnderflowMode::NEAREST_EVEN)
            quantized_num = (ax > half_num) ? min_num : 0;
        else if constexpr (U == UnderflowMode::NEAREST_AWAY)
            quantized_num = (ax >= half_num) ? min_num : 0;
        else if constexpr (U == UnderflowMode::AWAY)
            quantized_num = min_num;
        else // ZERO
            quantized_num = 0;
    }

    // P3109's zero is unsigned, and the underflow arm above can make one
    if (quantized_num != 0)
        quantized_num |= sign;

    return quantized_num;
}

// ---------------------------------------------------------------------------
// binaryK
// ---------------------------------------------------------------------------

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_nearest_even(T origin_float, int man_bits, int exp_bits, int bias, bool is_signed,
                                                    SaturationMode saturation_mode, SubnormalsMode subnormals)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    word_t target = R::bits(origin_float), quantize_bits;
    T quantized;

    int target_exp = R::field(target) - R::BIAS;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        bool not_uflow = exp_diff > -1 || (exp_diff == -1 && (target & R::MAN_MASK) != 0);
        if (!not_uflow)
            quantize_bits = 0;
        else if (man_bits > 0)
            quantize_bits = ref_round_nearest_even<T>(target, exp_diff);
        else
            quantize_bits = ref_round_nearest_even_pow2<T>(target);
        quantize_bits = ref_clip_subnormal<T>(target, quantize_bits, exp_bits, man_bits, bias);
        quantized = R::value(quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
        {
            quantize_bits = ref_clip_normal<UnderflowMode::NEAREST_EVEN, T>(target, target, exp_bits, man_bits, bias,
                                                                           saturation_mode, !is_signed);
            quantized = R::value(quantize_bits);
        }
        else
        {
            quantized = origin_float;
        }
    }
    // normal value range or overflow
    else
    {
        quantize_bits = (man_bits > 0) ? ref_round_nearest_even<T>(target, man_bits)
                                       : ref_round_nearest_even_pow2<T>(target);
        quantize_bits = ref_clip_normal<UnderflowMode::NEAREST_EVEN, T>(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode, !is_signed,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = R::value(quantize_bits);
    }

    return reference_zero(quantized);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_nearest_away(T origin_float, int man_bits, int exp_bits, int bias, bool is_signed,
                                                    SaturationMode saturation_mode, SubnormalsMode subnormals)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    word_t target = R::bits(origin_float), quantize_bits;
    T quantized;

    int target_exp = R::field(target) - R::BIAS;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        quantize_bits = (exp_diff >= -1) ? ref_round_nearest_away<T>(target, exp_diff) : 0;
        quantize_bits = ref_clip_subnormal<T>(target, quantize_bits, exp_bits, man_bits, bias);
        quantized = R::value(quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
        {
            quantize_bits = ref_clip_normal<UnderflowMode::NEAREST_AWAY, T>(target, target, exp_bits, man_bits, bias,
                                                                           saturation_mode, !is_signed);
            quantized = R::value(quantize_bits);
        }
        else
        {
            quantized = origin_float;
        }
    }
    // normal value range or overflow
    else
    {
        quantize_bits = ref_round_nearest_away<T>(target, man_bits);
        quantize_bits = ref_clip_normal<UnderflowMode::NEAREST_AWAY, T>(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode, !is_signed,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = R::value(quantize_bits);
    }

    return reference_zero(quantized);
}

// rounds to odd
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_odd(T origin_float, int man_bits, int exp_bits, int bias, bool is_signed,
                                           SaturationMode saturation_mode, SubnormalsMode subnormals)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    word_t target = R::bits(origin_float), quantize_bits;
    T quantized;

    int target_exp = R::field(target) - R::BIAS;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        // round to odd never flushes a nonzero input to zero, so the "_up"
        // subnormal clip is used to clamp underflowing nonzero values to
        // the smallest subnormal rather than to zero
        int exp_diff = man_bits - (min_exp - target_exp);
        if (exp_diff > 0)
        {
            quantize_bits = ref_round_odd<T>(target, exp_diff);
        }
        else
        {
            // precision has collapsed to zero: every value in this range
            // rounds to the smallest subnormal, whose significand is
            // always 1 (odd), so no sticky-driven carry is needed
            quantize_bits = target & ~R::MAN_MASK;
        }
        quantize_bits = ref_clip_subnormal_up<T>(target, quantize_bits, exp_bits, man_bits, bias);
        quantized = R::value(quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
        {
            quantize_bits = ref_clip_normal<UnderflowMode::AWAY, T>(target, target, exp_bits, man_bits, bias,
                                                                   saturation_mode, !is_signed);
            quantized = R::value(quantize_bits);
        }
        else
        {
            quantized = origin_float;
        }
    }
    // normal value range or overflow
    else
    {
        if (man_bits > 0)
        {
            quantize_bits = ref_round_odd<T>(target, man_bits);
        }
        else
        {
            // no explicit significand bits: "odd" refers to the parity of
            // the target format's own biased exponent, not the carrier's
            // exponent field (their biases generally differ in parity)
            bool sticky = (target & R::MAN_MASK) != 0;
            bool already_odd = ((target_exp + bias) & 1) != 0;
            quantize_bits = target & ~R::MAN_MASK;
            if (sticky && !already_odd)
                quantize_bits += R::ONE << R::MAN;
        }
        quantize_bits = ref_clip_normal<UnderflowMode::AWAY, T>(target, quantize_bits, exp_bits, man_bits, bias,
                                                               saturation_mode, !is_signed,
                                                               subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = R::value(quantize_bits);
    }

    return reference_zero(quantized);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_absolute_up(T origin_float, int man_bits, int exp_bits, int bias, bool is_signed,
                                           SaturationMode saturation_mode, SubnormalsMode subnormals)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    word_t target = R::bits(origin_float), quantize_bits;
    T quantized;

    int target_exp = R::field(target) - R::BIAS;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        quantize_bits = ref_round_up<T>(target, exp_diff < 0 ? 0 : exp_diff);
        quantize_bits = ref_clip_subnormal_up<T>(target, quantize_bits, exp_bits, man_bits, bias);
        quantized = R::value(quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
        {
            quantize_bits = ref_clip_normal<UnderflowMode::AWAY, T>(target, target, exp_bits, man_bits, bias,
                                                                   saturation_mode, !is_signed);
            quantized = R::value(quantize_bits);
        }
        else
        {
            quantized = origin_float;
        }
    }
    // normal value range or overflow
    else
    {
        quantize_bits = ref_round_up<T>(target, man_bits);
        quantize_bits = ref_clip_normal<UnderflowMode::AWAY, T>(target, quantize_bits, exp_bits, man_bits, bias,
                                                               saturation_mode, !is_signed,
                                                               subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = R::value(quantize_bits);
    }

    return quantized;
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_absolute_down(T origin_float, int man_bits, int exp_bits, int bias, bool is_signed,
                                             SaturationMode saturation_mode, SubnormalsMode subnormals)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    word_t target = R::bits(origin_float), quantize_bits;
    T quantized;

    int target_exp = R::field(target) - R::BIAS;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && subnormals == SubnormalsMode::SUBNORMALS)
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        quantize_bits = (exp_diff > -1) ? ref_round_down<T>(target, exp_diff) : 0;
        quantize_bits = ref_clip_subnormal<T>(target, quantize_bits, exp_bits, man_bits, bias);
        quantized = R::value(quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
        {
            quantize_bits = ref_clip_normal<UnderflowMode::ZERO, T>(target, target, exp_bits, man_bits, bias,
                                                                   saturation_mode, !is_signed);
            quantized = R::value(quantize_bits);
        }
        else
        {
            quantized = origin_float;
        }
    }
    // normal value range or overflow
    else
    {
        quantize_bits = ref_round_down<T>(target, man_bits);
        quantize_bits = ref_clip_normal<UnderflowMode::ZERO, T>(target, quantize_bits, exp_bits, man_bits, bias,
                                                               saturation_mode, !is_signed,
                                                               subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = R::value(quantize_bits);
    }

    return quantized;
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_up(T origin_float, int man_bits, int exp_bits, int bias, bool is_signed,
                                          SaturationMode saturation_mode, SubnormalsMode subnormals)
{
    using R = Ref<T>;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    // A NaN compares false against zero, so it would take the negating arm
    // below, and -(-NaN) is not required to hand back the same NaN: on CUDA
    // it canonicalizes the payload and the sign, which is a real difference
    // between the backends and not a spelling of one. cast_absolute_up's own
    // NaN/inf arm answers both signs correctly, so the non-finite inputs are
    // sent straight to it. (The shipped cast gets there by negating on the
    // word instead -- see bit_helper.h's flip_sign; cast_superfp_up below has
    // carried this guard from the start.)
    if (R::field(R::bits(origin_float)) == R::ALL_ONES)
        return cast_absolute_up(origin_float, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals);

    if (origin_float >= T(0))
        return reference_zero(
            cast_absolute_up(origin_float, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals));
    else
        return reference_zero(
            -cast_absolute_down(-origin_float, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals));
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_down(T origin_float, int man_bits, int exp_bits, int bias, bool is_signed,
                                            SaturationMode saturation_mode, SubnormalsMode subnormals)
{
    using R = Ref<T>;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    // see cast_binaryK_up
    if (R::field(R::bits(origin_float)) == R::ALL_ONES)
        return cast_absolute_down(origin_float, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals);

    if (origin_float >= T(0))
        return reference_zero(
            cast_absolute_down(origin_float, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals));
    else
        return reference_zero(
            -cast_absolute_up(-origin_float, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals));
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_zero(T origin_float, int man_bits, int exp_bits, int bias, bool is_signed,
                                            SaturationMode saturation_mode, SubnormalsMode subnormals)
{
    if (origin_float >= T(0))
        return cast_binaryK_down(origin_float, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals);
    else
        return cast_binaryK_up(origin_float, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals);
}

// Stochastic rounding, StochasticA, B or C with N random bits, as the spec
// in `spec` says (common/stochastic.h: N in the low byte, the variant above
// it): each region's two neighbours lo < hi, and the definition above on
// them. The regions are the format's: its normal binades (the grid of the
// input's binade), under SUBNORMALS the subnormal grid below 2^min_exp, and
// under NORMALS or EXTENDED_NORMALS zero and the floor below the floor.
// The one place the shipped stochastic casts are not P3109's definition, by
// the user's choice (2026-11-11; stochastic.h's sr_away_below_or_above): a
// carrier subnormal under a floor 2^f whose half draw quantum, 2^(f - N - 1),
// is finer than the carrier's subnormal spacing rounds away. Only formats that
// warn that the carrier cannot place their bottom have such a floor; at or
// above 2^f the definition goes up too, so this changes only inputs under it.
template <class T>
CUDA_HOST_DEVICE_INLINE bool ref_sr_coarse_floor(int field, int f, int n)
{
    using R = Ref<T>;
    return field == 0 && f - n - 1 < 1 - R::BIAS - R::MAN;
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_stochastic(T origin_float, typename Ref<T>::word_t rand_prob, int spec,
                                                  int man_bits, int exp_bits, int bias, bool is_signed,
                                                  SaturationMode saturation_mode, SubnormalsMode subnormals)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    using u64 = unsigned long long;
    const int n = spec & 0xFF, variant = (spec >> 8) & 3;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    const word_t target = R::bits(origin_float);
    const word_t sign = target & R::SIGN;
    const int field = R::field(target);
    const int target_exp = field - R::BIAS;
    const int min_exp = -bias + 1;
    const bool extended = subnormals == SubnormalsMode::EXTENDED_NORMALS;

    if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
            return R::value(ref_clip_normal<UnderflowMode::NONE, T>(target, target, exp_bits, man_bits, bias,
                                                                    saturation_mode, !is_signed));
        return origin_float;
    }

    const T ax = R::value(target & R::ABS);
    const u64 r = ref_sr_field<T>(rand_prob, man_bits, n);
    T out;

    // the floor of NORMALS and EXTENDED_NORMALS, as ref_clip_normal derives it
    const int foot = 1 - bias - (extended ? 1 : 0);
    const int step = man_bits < R::MAN ? man_bits : R::MAN;
    double floor_v = ldexp(1.0, foot) + (extended ? ldexp(1.0, foot - step) : 0.0);
    const bool floor_alive = floor_v >= ldexp(1.0, 1 - R::BIAS) && floor_v < ldexp(1.0, R::BIAS + 1);

    if (subnormals == SubnormalsMode::SUBNORMALS && target_exp < min_exp)
    {
        // the subnormal grid, of step 2^(min_exp - man_bits)
        const int q = min_exp - man_bits;
        const RefSplit sp = ref_split<T>(target & R::ABS, q);
        const bool up = ref_sr_up(ref_sr_x_bits(sp.rem, sp.d, n), r, n, variant);
        out = ref_grid_value<T>(sp.whole + (up ? 1 : 0), q);
    }
    else if (subnormals != SubnormalsMode::SUBNORMALS && floor_alive && (double)ax < floor_v)
    {
        // zero and the floor
        bool up;
        if (!extended || man_bits == 0)
        {
            const int f = extended ? foot + 1 : foot; // EXTENDED_NORMALS at man 0 carries into 2^(foot+1)
            const RefSplit sp = ref_split<T>(target & R::ABS, f);
            up = ref_sr_coarse_floor<T>(field, f, n) || ref_sr_up(ref_sr_x_bits(sp.rem, sp.d, n), r, n, variant);
        }
        else
        {
            // floor_v = (2^step + 1) * 2^(foot - step), and |x| = s * 2^(e - MAN)
            const u64 s = (u64)(target & R::MAN_MASK) | (field ? (1ull << R::MAN) : 0ull);
            const int e = field ? target_exp : 1 - R::BIAS;
            up = ref_sr_up(ref_sr_x_scaled(s, e - R::MAN, (1ull << step) + 1ull, foot - step, n), r, n, variant);
        }
        out = up ? (T)floor_v : T(0);
    }
    else
    {
        // the grid of the input's binade
        const int q = (field ? target_exp : 1 - R::BIAS) - man_bits;
        const RefSplit sp = ref_split<T>(target & R::ABS, q);
        const bool up = ref_sr_up(ref_sr_x_bits(sp.rem, sp.d, n), r, n, variant);
        const T v = ref_grid_value<T>(sp.whole + (up ? 1 : 0), q);
        const word_t quantize_bits = ref_clip_normal<UnderflowMode::NONE, T>(
            target, R::bits(v) | sign, exp_bits, man_bits, bias, saturation_mode, !is_signed, extended);
        return reference_zero(R::value(quantize_bits));
    }

    return reference_zero(sign ? -out : out);
}

// ---------------------------------------------------------------------------
// superfp
//
// superfp_region_cutoffs stays in cast_superfp.h: make_superfp_params calls
// it, so it is live in the extension and not a reference-only helper. It is
// the one piece of arithmetic this file shares, and it reads no carrier.
// ---------------------------------------------------------------------------

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_nearest_even(T origin_float, int man_bits, int exp_bits, int normal_binades,
                                                    int bias, bool is_signed, SaturationMode saturation_mode)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    word_t target = R::bits(origin_float), quantize_bits;
    T quantized{T(0)};

    int target_exp = R::field(target) - R::BIAS;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;
    if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
        {
            quantize_bits = ref_clip_normal<UnderflowMode::NEAREST_EVEN, T>(target, target, exp_bits, man_bits, bias,
                                                                           saturation_mode, false);
            quantized = R::value(quantize_bits);
        }
        else
        {
            quantized = origin_float;
        }
    }
    else if (supernormal)
    {
        quantize_bits = ref_round_nearest_even_pow2<T>(target);
        quantized = R::value(quantize_bits);
    }
    else if (underflow)
    {
        // ties-to-even between 0 and the smallest supernormal magnitude
        // (2^supernormal_cutoff): only the topmost underflow exponent
        // (supernormal_cutoff - 1) can be at or above the halfway point;
        // anything strictly below always flushes to zero.
        word_t sign_bit = target & R::SIGN;
        quantize_bits = sign_bit;
        if (target_exp == co.supernormal_cutoff - 1 && (target & R::MAN_MASK) != 0)
        {
            // strictly above half -> round up to the smallest supernormal;
            // exactly at half (mantissa all zero) ties to even = zero.
            quantize_bits = ((word_t)(co.supernormal_cutoff + R::BIAS) << R::MAN) | sign_bit;
        }
        quantized = R::value(quantize_bits);
    }
    else
    {
        quantize_bits = (man_bits > 0) ? ref_round_nearest_even<T>(target, man_bits)
                                       : ref_round_nearest_even_pow2<T>(target);
        quantize_bits = ref_clip_normal<UnderflowMode::NEAREST_EVEN, T>(target, quantize_bits, exp_bits, man_bits, bias,
                                                                       saturation_mode, false);
        quantized = R::value(quantize_bits);
    }

    return reference_zero(quantized);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_nearest_away(T origin_float, int man_bits, int exp_bits, int normal_binades,
                                                    int bias, bool is_signed, SaturationMode saturation_mode)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    word_t target = R::bits(origin_float), quantize_bits;
    T quantized;

    int target_exp = R::field(target) - R::BIAS;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
        {
            quantize_bits = ref_clip_normal<UnderflowMode::NEAREST_AWAY, T>(target, target, exp_bits, man_bits, bias,
                                                                           saturation_mode, false);
            quantized = R::value(quantize_bits);
        }
        else
        {
            quantized = origin_float;
        }
    }
    else if (supernormal)
    {
        quantize_bits = ref_round_nearest_away<T>(target, 0);
        quantized = R::value(quantize_bits);
    }
    else if (underflow)
    {
        // ties-to-away: bump to the smallest supernormal magnitude as soon
        // as the halfway point is reached (unlike nearest-even, the exact
        // tie also rounds away from zero).
        word_t sign_bit = target & R::SIGN;
        if (target_exp >= co.supernormal_cutoff - 1)
            quantize_bits = ((word_t)(co.supernormal_cutoff + R::BIAS) << R::MAN) | sign_bit;
        else
            quantize_bits = sign_bit;
        quantized = R::value(quantize_bits);
    }
    else
    {
        quantize_bits = ref_round_nearest_away<T>(target, man_bits);
        quantize_bits = ref_clip_normal<UnderflowMode::NEAREST_AWAY, T>(target, quantize_bits, exp_bits, man_bits, bias,
                                                                       saturation_mode, false);
        quantized = R::value(quantize_bits);
    }

    return reference_zero(quantized);
}

// rounds to odd
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_odd(T origin_float, int man_bits, int exp_bits, int normal_binades, int bias,
                                           bool is_signed, SaturationMode saturation_mode)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    word_t target = R::bits(origin_float), quantize_bits;
    T quantized;

    int target_exp = R::field(target) - R::BIAS;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
        {
            quantize_bits = ref_clip_normal<UnderflowMode::AWAY, T>(target, target, exp_bits, man_bits, bias,
                                                                   saturation_mode, false);
            quantized = R::value(quantize_bits);
        }
        else
        {
            quantized = origin_float;
        }
    }
    else if (supernormal)
    {
        // supernormal codepoints have no explicit mantissa bits, so
        // "oddness" refers to the parity of the supernormal grid index
        // itself (1 for the smallest magnitude, 2 for the next, ...), a
        // different reference frame than the carrier's exponent field
        // parity used by nearest-even, and different again from the
        // *target format's own* biased exponent parity used in the normal-
        // region man_bits==0 case below.
        int stored_index = target_exp - co.supernormal_cutoff + 1;
        bool sticky = (target & R::MAN_MASK) != 0;
        bool already_odd = (stored_index & 1) != 0;
        quantize_bits = target & ~R::MAN_MASK;
        if (sticky && !already_odd)
            quantize_bits += R::ONE << R::MAN;
        quantized = R::value(quantize_bits);
    }
    else if (underflow)
    {
        // round to odd never flushes a nonzero input to zero; the smallest
        // supernormal codepoint's grid index is always 1 (odd), so a bare
        // bump (no sticky-driven adjustment) always yields an odd result.
        if ((target & R::ABS) == 0)
        {
            quantized = origin_float;
        }
        else
        {
            quantize_bits = ((word_t)(co.supernormal_cutoff + R::BIAS) << R::MAN) | (target & R::SIGN);
            quantized = R::value(quantize_bits);
        }
    }
    else
    {
        if (man_bits > 0)
        {
            quantize_bits = ref_round_odd<T>(target, man_bits);
        }
        else
        {
            // no explicit significand bits: oddness refers to the parity of
            // the target format's own biased exponent (target_exp + bias),
            // mirroring cast_binaryK_odd's man_bits==0 handling.
            bool sticky = (target & R::MAN_MASK) != 0;
            bool already_odd = ((target_exp + bias) & 1) != 0;
            quantize_bits = target & ~R::MAN_MASK;
            if (sticky && !already_odd)
                quantize_bits += R::ONE << R::MAN;
        }
        quantize_bits = ref_clip_normal<UnderflowMode::AWAY, T>(target, quantize_bits, exp_bits, man_bits, bias,
                                                               saturation_mode, false);
        quantized = R::value(quantize_bits);
    }

    return reference_zero(quantized);
}

// unsigned helper for cast_superfp_up/cast_superfp_down: assumes
// origin_float >= 0.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_absolute_up(T origin_float, int man_bits, int exp_bits, int normal_binades,
                                                   int bias, SaturationMode saturation_mode)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    word_t target = R::bits(origin_float), quantize_bits;
    T quantized;

    int target_exp = R::field(target) - R::BIAS;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
        {
            quantize_bits = ref_clip_normal<UnderflowMode::AWAY, T>(target, target, exp_bits, man_bits, bias,
                                                                   saturation_mode, false);
            quantized = R::value(quantize_bits);
        }
        else
        {
            quantized = origin_float;
        }
    }
    else if (supernormal)
    {
        quantize_bits = ref_round_up<T>(target, 0);
        quantized = R::value(quantize_bits);
    }
    else if (underflow)
    {
        // rounding up (toward +inf) never flushes a nonzero magnitude to
        // zero: any nonzero underflowing input bumps up to the smallest
        // supernormal magnitude.
        if ((target & R::ABS) == 0)
        {
            quantized = origin_float;
        }
        else
        {
            quantize_bits = (word_t)(co.supernormal_cutoff + R::BIAS) << R::MAN;
            quantized = R::value(quantize_bits);
        }
    }
    else
    {
        quantize_bits = ref_round_up<T>(target, man_bits);
        quantize_bits = ref_clip_normal<UnderflowMode::AWAY, T>(target, quantize_bits, exp_bits, man_bits, bias,
                                                               saturation_mode, false);
        quantized = R::value(quantize_bits);
    }

    return quantized;
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_absolute_down(T origin_float, int man_bits, int exp_bits, int normal_binades,
                                                     int bias, SaturationMode saturation_mode)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    word_t target = R::bits(origin_float), quantize_bits;
    T quantized;

    int target_exp = R::field(target) - R::BIAS;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
        {
            quantize_bits = ref_clip_normal<UnderflowMode::ZERO, T>(target, target, exp_bits, man_bits, bias,
                                                                   saturation_mode, false);
            quantized = R::value(quantize_bits);
        }
        else
        {
            quantized = origin_float;
        }
    }
    else if (supernormal)
    {
        quantize_bits = ref_round_down<T>(target, 0);
        quantized = R::value(quantize_bits);
    }
    else if (underflow)
    {
        // rounding down (toward -inf, on a nonnegative magnitude) always
        // flushes to zero: there is no partial precision in this region to
        // create a boundary case worth preserving.
        quantize_bits = target & R::SIGN;
        quantized = R::value(quantize_bits);
    }
    else
    {
        quantize_bits = ref_round_down<T>(target, man_bits);
        quantize_bits = ref_clip_normal<UnderflowMode::ZERO, T>(target, quantize_bits, exp_bits, man_bits, bias,
                                                               saturation_mode, false);
        quantized = R::value(quantize_bits);
    }

    return quantized;
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_up(T origin_float, int man_bits, int exp_bits, int normal_binades, int bias,
                                          bool is_signed, SaturationMode saturation_mode)
{
    using R = Ref<T>;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    // NaN comparisons are always false, so `origin_float >= 0` alone would
    // route every NaN through the sign-flip dispatch below; unary negation
    // of a NaN is not guaranteed to preserve its exact bit pattern on every
    // backend (observed on CUDA, where -(-NaN) can canonicalize the
    // payload), so NaN/Inf must be passed through directly instead.
    typename R::word_t target = R::bits(origin_float);
    if (R::field(target) == R::ALL_ONES)
    {
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
            return R::value(ref_clip_normal<UnderflowMode::ZERO, T>(target, target, exp_bits, man_bits, bias,
                                                                   saturation_mode, false));
        return origin_float;
    }

    if (origin_float >= T(0))
        return reference_zero(
            cast_superfp_absolute_up(origin_float, man_bits, exp_bits, normal_binades, bias, saturation_mode));
    else
        return reference_zero(
            -cast_superfp_absolute_down(-origin_float, man_bits, exp_bits, normal_binades, bias, saturation_mode));
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_down(T origin_float, int man_bits, int exp_bits, int normal_binades, int bias,
                                            bool is_signed, SaturationMode saturation_mode)
{
    using R = Ref<T>;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    // see cast_superfp_up for why NaN/Inf must be special-cased before the
    // sign-flip dispatch rather than relying on it to preserve the payload.
    typename R::word_t target = R::bits(origin_float);
    if (R::field(target) == R::ALL_ONES)
    {
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
            return R::value(ref_clip_normal<UnderflowMode::ZERO, T>(target, target, exp_bits, man_bits, bias,
                                                                   saturation_mode, false));
        return origin_float;
    }

    if (origin_float >= T(0))
        return reference_zero(
            cast_superfp_absolute_down(origin_float, man_bits, exp_bits, normal_binades, bias, saturation_mode));
    else
        return reference_zero(
            -cast_superfp_absolute_up(-origin_float, man_bits, exp_bits, normal_binades, bias, saturation_mode));
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_zero(T origin_float, int man_bits, int exp_bits, int normal_binades, int bias,
                                            bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float >= T(0))
        return cast_superfp_down(origin_float, man_bits, exp_bits, normal_binades, bias, is_signed, saturation_mode);
    else
        return cast_superfp_up(origin_float, man_bits, exp_bits, normal_binades, bias, is_signed, saturation_mode);
}

// Stochastic rounding, StochasticA, B or C as `spec` says, on superfp's
// regions: its normal binades (the grid of the input's binade), a
// supernormal binade (the power of two 2^e and the next), and zero and the
// smallest value 2^c below it. R is the draw's field below the normal
// significand, as for binaryK. A carrier subnormal is read with the
// exponent its field encodes, 1 - BIAS, and one in a supernormal binade (a
// cutoff past the carrier's normals, which no carrier holds) is taken as
// below the smallest value, as the shipped cast takes it.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_stochastic(T origin_float, typename Ref<T>::word_t rand_prob, int spec,
                                                  int man_bits, int exp_bits, int normal_binades, int bias,
                                                  bool is_signed, SaturationMode saturation_mode)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    using u64 = unsigned long long;
    const int n = spec & 0xFF, variant = (spec >> 8) & 3;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    const word_t target = R::bits(origin_float);
    const word_t sign = target & R::SIGN;
    const int field = R::field(target);
    const int target_exp = field - R::BIAS;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    if (target_exp == R::SPECIAL)
    {
        // an infinity is an overflow under SAT_FINITE: the clip's overflow arm
        // saturates it to max_num like any other out-of-range exponent
        if (ref_sat_finite(saturation_mode) && (target & R::MAN_MASK) == 0)
            return R::value(ref_clip_normal<UnderflowMode::NONE, T>(target, target, exp_bits, man_bits, bias,
                                                                    saturation_mode, false));
        return origin_float;
    }

    const u64 r = ref_sr_field<T>(rand_prob, man_bits, n);
    T out;
    if (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff && field != 0)
    {
        // 2^e and 2^(e+1): eta is the mantissa's fraction, X = m * 2^(N - MAN)
        const u64 m = (u64)(target & R::MAN_MASK);
        const bool up = ref_sr_up(ref_sr_x_bits(m, R::MAN, n), r, n, variant);
        out = (T)ldexp(1.0, target_exp + (up ? 1 : 0));
    }
    else if (target_exp < co.normal_cutoff)
    {
        // zero and 2^c
        const RefSplit sp = ref_split<T>(target & R::ABS, co.supernormal_cutoff);
        const RefSrX x = ref_sr_x_bits(sp.rem, sp.d, n);
        // an input at or past 2^c reaches here only as a carrier subnormal
        // in a format no carrier holds: past 2^N, every variant goes up
        const bool up = sp.whole != 0 || ref_sr_coarse_floor<T>(field, co.supernormal_cutoff, n) ||
                        ref_sr_up(x, r, n, variant);
        out = up ? (T)ldexp(1.0, co.supernormal_cutoff) : T(0);
    }
    else
    {
        const int q = target_exp - man_bits;
        const RefSplit sp = ref_split<T>(target & R::ABS, q);
        const bool up = ref_sr_up(ref_sr_x_bits(sp.rem, sp.d, n), r, n, variant);
        const T v = ref_grid_value<T>(sp.whole + (up ? 1 : 0), q);
        const word_t quantize_bits = ref_clip_normal<UnderflowMode::NONE, T>(target, R::bits(v) | sign, exp_bits,
                                                                             man_bits, bias, saturation_mode, false);
        return reference_zero(R::value(quantize_bits));
    }
    return reference_zero(sign ? -out : out);
}

// ---------------------------------------------------------------------------
// Fixed point (cast_fixedpoint.h), from its definition rather than from the
// shipped cast's shape.
//
// FixedPoint(wl, fl, is_signed, symmetric) holds n * 2^-fl for the integers n
// a wl-bit word holds: |n| <= 2^mag_bits - 1, mag_bits = wl - is_signed, plus
// n = -2^mag_bits in two's complement (signed and not symmetric), and n >= 0
// unsigned. The shipped cast is binaryK's subnormal arm on the carrier's word,
// with clips; this one never looks at a binade. It writes |x| exactly as
// s * 2^t in integers (s the significand, t its exponent plus fl), so the
// integer part of |x| / step is s shifted by t and the fraction is what the
// shift drops; each mode decides from that fraction alone whether the
// magnitude code goes up by one; the code is clamped to the end on its side;
// and the value is built back with ldexp. A NaN passes through, an infinity
// saturates (there is no code for one), a negative input to an unsigned
// format is 0, and a zero is +0.0 (P3109's one zero).
//
// SR is P3109's StochasticA with rand_bits = N random bits, the top N of the
// draw's MAN bits (the convention cast_fixedpoint_stochastic reads its draw
// in): the magnitude goes up iff floor(eta * 2^N) + R >= 2^N, eta the
// fraction. That is the definition, so a shipped cast that rounds |x| in the
// carrier before the draw (as the one this replaced did) mismatches here.
// ---------------------------------------------------------------------------
template <class T>
CUDA_HOST_DEVICE_INLINE T ref_fixedpoint(T x, int wl, int fl, bool is_signed, bool symmetric, RoundMode mode,
                                         typename Ref<T>::word_t rand, int rand_bits)
{
    using R = Ref<T>;
    using word_t = typename R::word_t;
    using u64 = unsigned long long;

    if (!is_signed && x < T(0))
        return T(0);

    const word_t w = R::bits(x);
    const bool neg = (w & R::SIGN) != 0;
    const int field = R::field(w);
    const int mag_bits = wl - (is_signed ? 1 : 0);
    // the largest magnitude code on this side of zero
    const u64 cap = (neg && is_signed && !symmetric) ? (1ull << mag_bits) : (1ull << mag_bits) - 1;

    u64 n = 0;
    if (field == R::ALL_ONES)
    {
        if ((w & R::MAN_MASK) != 0)
            return x; // a NaN, whole
        n = cap;      // an infinity is past either end
    }
    else
    {
        const u64 s = (u64)(w & R::MAN_MASK) | (field ? (1ull << R::MAN) : 0ull);
        if (s == 0)
            return T(0);
        // |x| / step = s * 2^t
        const int t = (field ? field : 1) - R::BIAS - R::MAN + fl;
        int width = 0; // s's bit length
        for (u64 v = s; v; v >>= 1)
            ++width;
        if (t >= 0)
            n = (width - 1 + t > mag_bits) ? cap + 1 : (s << t); // past the top is past the cap
        else
        {
            const int d = -t; // the fraction is rem / 2^d
            const u64 whole = d < 64 ? (s >> d) : 0ull;
            const u64 rem = d < 64 ? (s - (whole << d)) : s;
            // the fraction against one half: -1 below, 0 at, +1 above
            int half = -1;
            if (d - 1 < 64)
            {
                const u64 h = 1ull << (d - 1);
                half = rem < h ? -1 : (rem == h ? 0 : 1);
            }
            const bool inexact = rem != 0;
            bool up = false;
            switch (mode)
            {
            case RoundMode::RNE:
                up = half > 0 || (half == 0 && (whole & 1));
                break;
            case RoundMode::RNA:
                up = half >= 0;
                break;
            case RoundMode::RU:
                up = inexact && !neg;
                break;
            case RoundMode::RD:
                up = inexact && neg;
                break;
            case RoundMode::RZ:
                up = false;
                break;
            case RoundMode::RO:
                up = inexact && !(whole & 1);
                break;
            case RoundMode::SR:
            case RoundMode::SR_B:
            case RoundMode::SR_C:
            {
                // the figure's test, R the draw's top N bits
                const u64 r = (u64)((rand & R::MAN_MASK) >> (R::MAN - rand_bits));
                up = ref_sr_up(ref_sr_x_bits(rem, d, rand_bits), r, rand_bits,
                               (int)mode - (int)RoundMode::SR);
                break;
            }
            }
            n = whole + (up ? 1 : 0);
        }
    }
    if (n > cap)
        n = cap;
    if (n == 0)
        return T(0);
    const T v = ldexp((T)n, -fl);
    return neg ? -v : v;
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_nearest_even(T x, int wl, int fl, bool is_signed, bool symmetric)
{
    return ref_fixedpoint<T>(x, wl, fl, is_signed, symmetric, RoundMode::RNE, 0, 0);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_nearest_away(T x, int wl, int fl, bool is_signed, bool symmetric)
{
    return ref_fixedpoint<T>(x, wl, fl, is_signed, symmetric, RoundMode::RNA, 0, 0);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_up(T x, int wl, int fl, bool is_signed, bool symmetric)
{
    return ref_fixedpoint<T>(x, wl, fl, is_signed, symmetric, RoundMode::RU, 0, 0);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_down(T x, int wl, int fl, bool is_signed, bool symmetric)
{
    return ref_fixedpoint<T>(x, wl, fl, is_signed, symmetric, RoundMode::RD, 0, 0);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_zero(T x, int wl, int fl, bool is_signed, bool symmetric)
{
    return ref_fixedpoint<T>(x, wl, fl, is_signed, symmetric, RoundMode::RZ, 0, 0);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_odd(T x, int wl, int fl, bool is_signed, bool symmetric)
{
    return ref_fixedpoint<T>(x, wl, fl, is_signed, symmetric, RoundMode::RO, 0, 0);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_stochastic(T x, typename Ref<T>::word_t rand, int rand_bits, int wl, int fl,
                                                     bool is_signed, bool symmetric)
{
    return ref_fixedpoint<T>(x, wl, fl, is_signed, symmetric, RoundMode::SR, rand, rand_bits);
}
