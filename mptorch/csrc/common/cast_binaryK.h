#pragma once

#include "bit_helper.h"
#include "modes.h"

// The binaryK format cast, one function per rounding mode.
//
// binaryK is IEEE P3109's family of binary floating-point formats: K bits, P
// of them precision, a default bias of 2^(K-P-1) (2^(K-P) unsigned). Each cast
// is P3109's projection (round to the format's precision, then saturate),
// computed on the carrier's bit pattern (bit_helper.h's FloatTraits). modes.h
// names the P3109 mode behind each RoundMode and SaturationMode, and
// bit_helper.h's make_normal_range_params counts the top of the range in
// P3109's code points.
//
// Each cast reads a BinaryKParamsT built once, at Multiplier/Adder
// construction (gemm_policy.h) for the GEMM and once per tensor for the
// elementwise quantizers, rather than re-deriving its constants from
// man_bits/exp_bits/bias on every call. A second spelling of each cast, which
// takes the format parameters directly and derives everything per call, lives
// in dev/benchmarks/reference_casts.h; the exhaustive sweeps there check the
// bodies below against it over all 2^32 float inputs. The two are independent
// derivations of the same format on purpose: change one and change the other,
// or the sweeps will say so.

// The constants the casts read. Round-mode-agnostic: every field uses the
// same formula whichever cast reads it, so one struct serves all of them and
// a mode simply leaves the fields it does not need alone. See bit_helper.h's
// "Precomputed-parameter forms" note for what is worth precomputing.
//
// Flat by design: a nested aggregate replicated across the GEMM's
// instantiations took nvcc's cicc from 18 s to ten minutes. A base class is
// a nested aggregate too, which is why the fast path's fields are not one:
// the primary template below is every carrier's, and the binary32
// specialization after it repeats its fields, in the same order, before
// adding the fast path's. A binary64 struct therefore carries no fields it
// cannot use.
template <class T>
struct BinaryKParamsT
{
    using word_t = typename FloatTraits<T>::word_t;
    int man_bits; // kept raw: drives the subnormal branch's data-dependent
                  // exp_diff and the man_bits > 0 round-formula selection,
                  // neither of which can be precomputed away
    int min_exp;  // = -bias + 1
    // shared round-to-nearest/up/down/odd bitwise constants
    bool round_bypass;
    word_t round_mask;
    word_t round_tie;
    int round_shift;
    // for clip_normal_range_exponent
    SaturationMode saturation_mode;
    word_t min_num;  // the smallest magnitude the format represents
    word_t half_num; // half of it, the round-to-nearest boundary below it
    word_t max_num;
    // for clip_subnormal_range_exponent
    int subnormal_min_exponent_store;
    // no fast path for this carrier (FloatTraits::HAS_FAST_CAST); the casts
    // test the flag all the same, see cast_binaryK_nearest_even for why
    static constexpr MPTORCH_CONSTANT bool fast_rne = false;
};

template <>
struct BinaryKParamsT<float>
{
    using word_t = uint32_t;
    // the primary template's fields, in its order
    int man_bits;
    int min_exp;
    bool round_bypass;
    word_t round_mask;
    word_t round_tie;
    int round_shift;
    SaturationMode saturation_mode;
    word_t min_num;
    word_t half_num;
    word_t max_num;
    int subnormal_min_exponent_store;
    // Float-arithmetic RNE fast path, see cast_binaryK_rne_fast below.
    // fast_rne is the format half of the gate; the caller ANDs it with
    // subnormals == SubnormalsMode::SUBNORMALS.
    bool fast_rne;
    float fast_split_c;    // 2^(23 - man_bits) + 1, the Veltkamp split constant
    float fast_magic;      // 1.5 * 2^(23 + min_exp - man_bits): subnormal magic
    float fast_min_normal; // 2^min_exp: below this the subnormal grid applies
    float fast_max_finite; // the largest finite magnitude (max_num as a float)
    float fast_clamp_hi;   // 2 * max_finite: the clamp before the split
    float fast_ovf;        // inf under OVF_INF, else max_finite
};

using BinaryKParams = BinaryKParamsT<float>;

// Builds the constants for a binaryK format with man_bits, exp_bits and bias,
// for the carrier T, which defaults to binary32 (what every caller that names
// no carrier builds). is_signed and saturation_mode decide the reserved codes
// at the top of the range; extended_normals selects the EXTENDED_NORMALS
// floor (modes.h).
template <class T = float>
CUDA_HOST_DEVICE_INLINE BinaryKParamsT<T> make_binaryK_params(int man_bits, int exp_bits, int bias, bool is_signed,
                                                              SaturationMode saturation_mode,
                                                              bool extended_normals)
{
    using F = FloatTraits<T>;
    BinaryKParamsT<T> p;
    p.man_bits = man_bits;
    p.min_exp = -bias + 1;

    RoundParamsT<T> round_p = make_round_params<T>(man_bits);
    p.round_bypass = round_p.bypass;
    p.round_mask = round_p.mask;
    p.round_tie = round_p.tie;
    p.round_shift = round_p.shift;

    // P3109's reserved codes at the top: +infinity outside the finite domain
    // (SAT_FINITE), and NaN above it in an unsigned format.
    const int reserved_codes = (saturation_mode != SaturationMode::SAT_FINITE) + (is_signed ? 0 : 1);
    NormalRangeParamsT<T> normal_p =
        make_normal_range_params<T>(exp_bits, man_bits, bias, saturation_mode, reserved_codes, extended_normals);
    p.saturation_mode = normal_p.saturation_mode;
    p.min_num = normal_p.min_num;
    p.half_num = normal_p.half_num;
    p.max_num = normal_p.max_num;

    p.subnormal_min_exponent_store = make_subnormal_range_params<T>(man_bits, bias).min_exponent_store;

    if constexpr (F::HAS_FAST_CAST)
    {
        // ---- gate and constants for the float-arithmetic RNE fast path.
        // Every condition below is a range check on the derived exponents: each
        // guarantees that one of the fast path's four float operations stays
        // inside binary32 and therefore exact. See cast_binaryK_rne_fast.
        int e_split = 23 - man_bits;                     // split_c's exponent
        int e_magic = 23 + p.min_exp - man_bits;         // the magic's exponent
        int max_exp = (int)(p.max_num >> 23) - 127;      // max_num's exponent
        p.fast_rne =
            // man_bits == 0 takes round_bitwise_nearest_even's structurally
            // different zero-argument overload (ties on the exponent's parity,
            // not the significand's), which the Veltkamp split does not model;
            // man_bits > 22 leaves the magic add without the half-binade of
            // headroom it needs to stay inside its own binade.
            man_bits >= 1 && man_bits <= 22 &&
            // SAT_PROPAGATE clamps a finite value that overflows but keeps an
            // infinite input infinite, and the clamp below turns both into the
            // same magnitude before the single saturating compare sees them.
            saturation_mode != SaturationMode::SAT_PROPAGATE &&
            // max_num == 0 is a format with no finite normal value at all (see
            // make_normal_range_params): nothing for the compare to keep.
            p.max_num != 0 &&
            // 2^min_exp, 1.5 * 2^e_magic and the subnormal grid step
            // 2^(min_exp - man_bits) must all be normal binary32 values.
            // The step must also be at least 2^-125, so that half of it is
            // a normal value too: every binary32 subnormal input then lies
            // below that half and rounds to zero, which is what the magic
            // add gives it on an Apple GPU, where the add flushes the
            // subnormal operand to zero first (bit_helper.h, "The Apple
            // GPU's flush of binary32 subnormals"). At a step
            // of 2^-126 the inputs in (2^-127, 2^-126) round up on every other
            // backend. That step is below every format's binary32 floor
            // (number.py's _Carrier), so only a format the range check warns
            // about loses the fast path to it.
            p.min_exp >= -126 && p.min_exp <= 126 &&
            p.min_exp - man_bits >= -125 &&
            e_magic >= -126 && e_magic <= 126 &&
            // 2 * max_finite (the clamp) and fast_split_c * that product must
            // both stay finite: |clamp_hi| < 2^(max_exp + 2) and
            // fast_split_c < 2^(e_split + 1).
            max_exp <= 126 && max_exp + e_split + 3 <= 128 &&
            // The subnormal range has to sit below the format's largest finite
            // value: the integer path's subnormal branch runs
            // clip_subnormal_range_exponent, which only handles underflow, so
            // it would return values above max_num rather than saturating
            // them, which the fast path's single saturating compare cannot
            // reproduce. min_exp <= max_exp implies 2^min_exp <= max_finite. A
            // nonzero max_num has an exponent field of at least 1 and so
            // implies it in turn; the test guards make_normal_range_params,
            // not the format.
            p.min_exp <= max_exp;

        p.fast_split_c = F::pow2(e_split) + 1.0f;
        p.fast_magic = 1.5f * F::pow2(e_magic);
        p.fast_min_normal = F::pow2(p.min_exp);
        p.fast_max_finite = reinterpret_cast<const MPTORCH_THREAD T &>(p.max_num);
        p.fast_clamp_hi = 2.0f * p.fast_max_finite;
        const uint32_t inf_bits = F::INF_BITS;
        p.fast_ovf = (saturation_mode == SaturationMode::OVF_INF) ? reinterpret_cast<const MPTORCH_THREAD T &>(inf_bits) : p.fast_max_finite;
        if (!p.fast_rne)
        {
            // keep the unused constants finite so a disabled gate can never
            // produce a signalling value if the path is ever entered by mistake
            p.fast_split_c = 1.0f;
            p.fast_magic = 1.0f;
            p.fast_min_normal = 0.0f;
            p.fast_max_finite = 0.0f;
            p.fast_clamp_hi = 0.0f;
            p.fast_ovf = 0.0f;
        }
    }
    return p;
}

// Float-arithmetic replacement for cast_binaryK_nearest_even's bit-twiddling
// body, for the formats make_binaryK_params' gate admits. Round-to-nearest-
// even is what binary32 hardware already does, so the whole cast reduces to
// putting the value on the target format's grid and letting the FPU round:
//
//   * a Veltkamp split rounds to man_bits + 1 significand bits at any
//     exponent. With c = 2^s + 1 and t = c * x, Dekker's theorem says that
//     t - (t - x) is x rounded to nearest at 24 - s bits, exactly, provided
//     nothing overflows and each operation is rounded on its own;
//   * a magic-constant add rounds onto the subnormal range's fixed absolute
//     spacing of 2^(min_exp - man_bits): adding 1.5 * 2^e_magic puts the sum
//     in a binade whose ulp is that spacing, so the FPU rounds to it, and
//     subtracting the constant again is exact;
//   * saturation is one compare and one select on the *rounded* magnitude.
//
// Roughly 14 instructions and no branches, against ~125 instructions with
// ~17 data-dependent branches for the integer path. Verified against that
// path over all 2^32 float inputs for every admitted format by
// dev/benchmarks/gemm_cast_float_arith.cu.
//
// Correct only where the arithmetic is not contracted: the identity depends
// on `t` being a separately rounded binary32 value, and letting the compiler
// fuse the split's multiply and subtract into an FMA breaks it (248 million
// mismatches in the exhaustive sweep when nvcc was allowed to). That is why
// the operations are named rather than written as `*` and `-`, and why the
// path is behind MPTORCH_FAST_CAST: on the device the _rn intrinsics forbid
// contraction by themselves, on the host it takes -ffp-contract=off, which
// only the build can promise. See bit_helper.h.
//
// binary32's only (FloatTraits' HAS_FAST_CAST), so it is not a template: the
// casts below take it only in their binary32 instantiation.
#if defined(MPTORCH_FAST_CAST)
CUDA_HOST_DEVICE_INLINE float cast_binaryK_rne_fast(float origin_float, const MPTORCH_THREAD BinaryKParams &p)
{
    using F = FloatTraits<float>;
    float ax = fabsf(origin_float);
    // clamp before rounding so the Veltkamp product cannot overflow; anything
    // at or above the clamp is out of the format's range either way.
    float xc = copysignf(F::fmin_nonneg(ax, p.fast_clamp_hi), origin_float);
    float sub = F::rn_sub(F::rn_add(xc, p.fast_magic), p.fast_magic);
    float t = F::rn_mul(p.fast_split_c, xc);
    float nrm = F::rn_sub(t, F::rn_sub(t, xc));
    float y = (ax < p.fast_min_normal) ? sub : nrm;
    if (fabsf(y) > p.fast_max_finite)
        y = copysignf(p.fast_ovf, origin_float);
    // NaN passes through unchanged; inf takes `y`, which is what the integer
    // path's target_exp == 128 arm (saturate_nonfinite) gives it: the clamp
    // put it at fast_clamp_hi, which the split keeps and the compare above
    // turns into fast_ovf (inf under OVF_INF, max_finite under SAT_FINITE).
    // One ordered compare separates the two: `<= inf` is false only for NaN.
    //
    // Nothing to unsign here, because `y` is never -0.0: saturation gives
    // +-fast_ovf, which the gate keeps nonzero; a split of |x| >= 2^min_exp
    // keeps its exponent; and `sub` is a round-to-nearest difference of two
    // positive values, +0.0 when they are equal, for a -0.0 input too.
    const uint32_t inf_bits = F::INF_BITS;
    return (ax <= reinterpret_cast<const MPTORCH_THREAD float &>(inf_bits)) ? y : origin_float;
}
#endif

// Rounds to nearest, ties to even: P3109's NearestTiesToEven projection of
// origin_float onto the format p describes, with the subnormal region
// handled per `subnormals` (modes.h). A negative input to an unsigned format
// returns 0. Every arm returns a word the clips have already made P3109's
// unsigned zero, or a nonzero one, so nothing is unsigned on the way out.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_nearest_even(T origin_float, bool is_signed, SubnormalsMode subnormals,
                                                    const MPTORCH_THREAD BinaryKParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    if (MPTORCH_IS_NEGATIVE(origin_float) && !is_signed)
        return T(0);

#if defined(MPTORCH_FAST_CAST)
    // The gate is warp-uniform in the single-format kernels (every thread
    // reads the same params) and per-slot uniform in the mixed ones, so the
    // test costs a predicated compare and the integer body below is jumped
    // over, not fetched. On the host it is loop-invariant over a whole GEMM
    // and the branch predictor sees one outcome forever. Hoisting it out of
    // the elementwise quantizer's loop instead, with the loop instantiated on
    // the cast it chose, measured 0.98x-1.11x of this and slowed the rows
    // that take the integer path, so it is not worth a second spelling.
    //
    // The test is a plain `if` for every carrier (fast_rne is a constant
    // false where FloatTraits has no fast path) and only the call is behind
    // `if constexpr`. Putting the test inside it too makes nvcc lower the
    // `&&` with its operands the other way round in the SR arm below: the
    // same instructions, but no longer the same kernel.
    if (p.fast_rne && subnormals == SubnormalsMode::SUBNORMALS)
    {
        if constexpr (F::HAS_FAST_CAST)
            return cast_binaryK_rne_fast(origin_float, p);
    }
#endif

    word_t target, quantize_bits;
    target = reinterpret_cast<const MPTORCH_THREAD word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;
    bool subnormal = (target_exp < p.min_exp);

    // subnormal inputs, when the format has subnormals: round to the fixed
    // step 2^(min_exp - man_bits), which leaves exp_diff significand bits
    // above the step (0 or -1 for the smallest values), then floor
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = p.man_bits - (p.min_exp - target_exp);
        int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << (F::EXP_BITS + 1)) > 0));
        word_t rounded_val = (p.man_bits > 0)
                                 ? round_bitwise_nearest_even(target, exp_diff)
                                 : round_bitwise_nearest_even(target);
        quantize_bits = not_uflow * rounded_val;
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits, p.subnormal_min_exponent_store);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    // NaN/inf inputs: NaN passes through, inf saturates under SAT_FINITE
    else if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    // normal value range or overflow
    else
    {
        quantize_bits = (p.man_bits > 0)
                            ? round_bitwise_nearest_even(target, p.round_bypass, p.round_mask, p.round_tie, p.round_shift)
                            : round_bitwise_nearest_even(target);
        quantize_bits = clip_normal_range_exponent<UnderflowMode::NEAREST_EVEN>(
            target, quantize_bits, p.saturation_mode, p.min_num, p.half_num, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }

    // Nothing to unsign on the way out: every arm above returns a word the
    // clips have already made P3109's unsigned zero, or a nonzero one.
    return quantized;
}

// Rounds to nearest, ties away from zero. Shares BinaryKParamsT with RNE, as
// every mode here does; round_shift is the one field it has no use for.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_nearest_away(T origin_float, bool is_signed, SubnormalsMode subnormals,
                                                    const MPTORCH_THREAD BinaryKParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    if (MPTORCH_IS_NEGATIVE(origin_float) && !is_signed)
        return T(0);

    word_t target, quantize_bits;
    target = reinterpret_cast<const MPTORCH_THREAD word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;
    bool subnormal = (target_exp < p.min_exp);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = p.man_bits - (p.min_exp - target_exp);
        int not_uflow = exp_diff >= -1;
        quantize_bits = not_uflow * round_bitwise_nearest_away(target, exp_diff);
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits, p.subnormal_min_exponent_store);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    else if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_nearest_away(target, p.round_bypass, p.round_mask, p.round_tie);
        quantize_bits = clip_normal_range_exponent<UnderflowMode::NEAREST_AWAY>(
            target, quantize_bits, p.saturation_mode, p.min_num, p.half_num, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }

    return quantized; // already unsigned where it is zero, see the RNE cast
}

// Rounds to odd. The man_bits == 0 branch needs the raw `bias`, recovered as
// `1 - p.min_exp` rather than stored separately; there the kept "bit" is the
// exponent's LSB, so an inexact value moves to the next power of two when its
// exponent is even.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_odd(T origin_float, bool is_signed, SubnormalsMode subnormals,
                                           const MPTORCH_THREAD BinaryKParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    if (MPTORCH_IS_NEGATIVE(origin_float) && !is_signed)
        return T(0);

    word_t target, quantize_bits;
    target = reinterpret_cast<const MPTORCH_THREAD word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;
    bool subnormal = (target_exp < p.min_exp);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = p.man_bits - (p.min_exp - target_exp);
        if (exp_diff > 0)
        {
            quantize_bits = round_bitwise_odd(target, exp_diff);
        }
        else
        {
            quantize_bits = target & ~F::MAN_MASK;
        }
        quantize_bits = clip_subnormal_range_exponent_up(target, quantize_bits, p.subnormal_min_exponent_store);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    else if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    else
    {
        if (p.man_bits > 0)
        {
            quantize_bits = round_bitwise_odd(target, p.round_bypass, p.round_mask);
        }
        else
        {
            int bias = 1 - p.min_exp;
            word_t mask = F::MAN_MASK;
            bool sticky = (target & mask) != 0;
            bool already_odd = ((target_exp + bias) & 1) != 0;
            quantize_bits = target & ~mask;
            if (sticky && !already_odd)
                quantize_bits += (word_t(1) << F::MAN_BITS);
        }
        quantize_bits = clip_normal_range_exponent<UnderflowMode::AWAY>(
            target, quantize_bits, p.saturation_mode, p.min_num, p.half_num, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }

    return quantized; // already unsigned where it is zero, see the RNE cast
}

// Rounds a magnitude up (away from zero), for cast_binaryK_up and
// cast_binaryK_down: assumes origin_float >= 0. The subnormal branch's
// exp_diff is data-dependent (it varies per call), so round_bitwise_up(target,
// exp_diff) there derives its mask itself; only the normal branch's
// man_bits-derived round reads the precomputed constants.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_absolute_up(T origin_float, SubnormalsMode subnormals, const MPTORCH_THREAD BinaryKParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    word_t target, quantize_bits;
    target = reinterpret_cast<const MPTORCH_THREAD word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;
    bool subnormal = (target_exp < p.min_exp);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = p.man_bits - (p.min_exp - target_exp);
        quantize_bits = round_bitwise_up(target, exp_diff < 0 ? 0 : exp_diff);
        quantize_bits = clip_subnormal_range_exponent_up(target, quantize_bits, p.subnormal_min_exponent_store);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    else if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_up(target, p.round_bypass, p.round_mask);
        // the magnitude form: this helper is handed |x|, so the sign the
        // signed form puts back is always the one it took off (bit_helper.h)
        quantize_bits = clip_normal_range_magnitude<UnderflowMode::AWAY>(
            target, quantize_bits, p.saturation_mode, p.min_num, p.half_num, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }

    return quantized;
}

// Rounds a magnitude down (toward zero); the other half of cast_absolute_up.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_absolute_down(T origin_float, SubnormalsMode subnormals, const MPTORCH_THREAD BinaryKParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    word_t target, quantize_bits;
    target = reinterpret_cast<const MPTORCH_THREAD word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;
    bool subnormal = (target_exp < p.min_exp);

    if (subnormal && subnormals == SubnormalsMode::SUBNORMALS)
    {
        int exp_diff = p.man_bits - (p.min_exp - target_exp);
        int not_uflow = exp_diff > -1;
        quantize_bits = not_uflow * round_bitwise_down(target, exp_diff);
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits, p.subnormal_min_exponent_store);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    else if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_down(target, p.round_bypass, p.round_mask);
        // the magnitude form: this helper is handed |x|, so the sign the
        // signed form puts back is always the one it took off (bit_helper.h)
        quantize_bits = clip_normal_range_magnitude<UnderflowMode::ZERO>(
            target, quantize_bits, p.saturation_mode, p.min_num, p.half_num, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }

    return quantized;
}

// The three directed modes (toward +inf, toward -inf, toward zero) in terms
// of the two magnitude helpers above: rounding a negative value toward +inf
// is rounding its magnitude toward zero, and so on.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_up(T origin_float, bool is_signed, SubnormalsMode subnormals,
                                          const MPTORCH_THREAD BinaryKParamsT<T> &p)
{
    if (MPTORCH_IS_NEGATIVE(origin_float) && !is_signed)
        return T(0);

    // The helper returns a magnitude (a -0.0 input compares >= 0, and the
    // clips give it back as +0.0), so only the negation can make a -0.0, by
    // negating a magnitude that rounded to zero. Both negations are on the
    // word (bit_helper.h): negate_magnitude leaves a zero unsigned, and an
    // XOR on the sign bit cannot canonicalize a NaN's payload the way the
    // device's `neg.f32` may. A NaN takes the second arm, since every compare
    // against it is false, and comes back out of it unchanged.
    if (MPTORCH_IS_NONNEGATIVE(origin_float))
        return cast_absolute_up(origin_float, subnormals, p);
    else
        return negate_magnitude(cast_absolute_down(flip_sign(origin_float), subnormals, p));
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_down(T origin_float, bool is_signed, SubnormalsMode subnormals,
                                            const MPTORCH_THREAD BinaryKParamsT<T> &p)
{
    if (MPTORCH_IS_NEGATIVE(origin_float) && !is_signed)
        return T(0);

    // See cast_binaryK_up: the negation is the one way left to a -0.0, and
    // the one place a NaN could lose its payload.
    if (MPTORCH_IS_NONNEGATIVE(origin_float))
        return cast_absolute_down(origin_float, subnormals, p);
    else
        return negate_magnitude(cast_absolute_up(flip_sign(origin_float), subnormals, p));
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_zero(T origin_float, bool is_signed, SubnormalsMode subnormals,
                                            const MPTORCH_THREAD BinaryKParamsT<T> &p)
{
    if (MPTORCH_IS_NONNEGATIVE(origin_float))
        return cast_binaryK_down(origin_float, is_signed, subnormals, p);
    else
        return cast_binaryK_up(origin_float, is_signed, subnormals, p);
}

// Stochastic rounding: adds rand_bits random bits below the retained
// significand and truncates (P3109's StochasticA). Unlike the deterministic
// modes this takes a per-call random value (rand_prob, a word of the carrier
// whose MAN_BITS low bits are the draw) and rand_bits (fixed per
// Multiplier/Adder but specific to its role, so not folded into
// BinaryKParamsT); only the bitwise constants precompute away.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_binaryK_stochastic(T origin_float, typename FloatTraits<T>::word_t rand_prob,
                                                  int rand_bits, bool is_signed, SubnormalsMode subnormals,
                                                  const MPTORCH_THREAD BinaryKParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    if (MPTORCH_IS_NEGATIVE(origin_float) && !is_signed)
        return T(0);

    word_t target, quantize_bits;
    target = reinterpret_cast<const MPTORCH_THREAD word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;
    bool subnormal = (target_exp < p.min_exp);

    // keep only the rand_bits draw bits directly below the retained significand
    rand_prob = rand_prob & F::MAN_MASK;
    rand_prob = rand_prob & ~((word_t(1) << (F::MAN_BITS - p.man_bits - rand_bits)) - 1u);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        // Adding 2^min_exp (with the input's sign) moves the value into the
        // binade whose grid at man_bits is the subnormal step, so one
        // stochastic round at man_bits lands on that grid; the exact
        // subtraction afterwards moves it back.
        T shift_float, val;
        word_t shift_bits = (word_t)((typename F::sword_t)(F::BIAS + p.min_exp) << F::MAN_BITS) | (target & F::SIGN_MASK);
        shift_float = reinterpret_cast<const MPTORCH_THREAD T &>(shift_bits);
        val = MPTORCH_ADD_TO_POWER_OF_TWO(origin_float, shift_float);
        target = reinterpret_cast<const MPTORCH_THREAD word_t &>(val);
        quantize_bits = round_bitwise_stochastic(target, rand_prob, p.man_bits);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits) - shift_float;
    }
    else if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_stochastic(target, rand_prob, p.man_bits);
#if defined(MPTORCH_FAST_CAST)
        // Stochastic rounding itself has no hardware analogue (binary32 rounds
        // to nearest, so nothing in the FPU reproduces "add random bits below
        // the retained significand, then truncate"), and the line above stays
        // bitwise. What SR shares with RNE is everything after the rounding:
        // clip_normal_range_exponent, whose overflow test is one magnitude
        // compare, followed by an underflow test and a sign re-attach.
        //
        // Two facts make the compare sufficient here, and both are already in
        // the gate. Saturation is a magnitude test on the rounded value, and
        // fast_rne's `saturation_mode != SAT_PROPAGATE` leaves the two modes
        // whose saturated magnitude fast_ovf holds. And the clip's *underflow*
        // arm is unreachable: `add_r & ~mask` clears low significand bits but
        // never lowers an exponent, so a value that entered this branch at or
        // above min_exp leaves it there too, which is what `subnormals ==
        // SUBNORMALS` guarantees (anything below took the subnormal arm above).
        //
        // The flag is fast_rne rather than one of its own: its remaining
        // conditions are the Veltkamp split's, which SR does not use, so
        // reusing it only ever gates SR off where it could have run (man_bits
        // == 0, and the split-product bound). One flag, one sweep: verified in
        // dev/benchmarks/gemm_cast_sr_arith.cu. The test is outside the `if
        // constexpr` for the reason the RNE cast gives.
        if (p.fast_rne && subnormals == SubnormalsMode::SUBNORMALS)
        {
            if constexpr (F::HAS_FAST_CAST)
            {
                T y = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
                return (fabsf(y) > p.fast_max_finite) ? copysignf(p.fast_ovf, origin_float) : y;
            }
        }
#endif
        quantize_bits = clip_normal_range_exponent<UnderflowMode::STOCHASTIC>(
            target, quantize_bits, p.saturation_mode, p.min_num, p.half_num, p.max_num, rand_prob);
        quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);
    }

    return quantized; // already unsigned where it is zero, see the RNE cast
}
