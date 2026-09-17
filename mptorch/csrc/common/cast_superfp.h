#pragma once

#include "bit_helper.h"
#include "modes.h"

// The superfp format cast, one function per rounding mode, in the same shape
// as cast_binaryK.h: each reads a SuperfpParamsT built once instead of
// re-deriving its constants from man_bits/exp_bits/normal_binades/bias on
// every call, and is a template on the carrier (bit_helper.h's FloatTraits).
// The format-taking spelling that the exhaustive sweeps check these against
// lives in dev/benchmarks/reference_casts.h, written independently on
// purpose.

// superfp has no subnormals. Its top `normal_binades` binades hold ordinary
// normal values with man_bits of significand; the rest of the exponent
// range, which would otherwise encode subnormals, is the "supernormal"
// region, where every code is a power of two (implicit significand 1.0, no
// mantissa bits) and so the range is extended downward by 2^man_bits binades
// per encoding binade. Below the supernormal region everything flushes to
// zero. These two cutoffs classify a carrier's unbiased exponent into one of
// the three regions: normal (>= normal_cutoff), supernormal
// ([supernormal_cutoff, normal_cutoff)), or underflow (< supernormal_cutoff).
// Neither depends on the carrier, so this is not a template.
struct SuperfpCutoffs
{
    int normal_cutoff;
    int supernormal_cutoff;
};

// Computes the two region cutoffs of a superfp format. The supernormal region
// spans (2^exp_bits - normal_binades) * 2^man_bits binades, which overflows
// `int` once exp_bits + man_bits reaches 31 (exp_bits 9 with binary32's 23
// mantissa bits already does it, and binary64's 52 always would). So the span
// is taken in 64 bits, and a cutoff below `int`'s range is clamped to one
// above INT_MIN, so that `supernormal_cutoff - 1` stays representable. The
// clamp moves no answer: every exponent a carrier's word can hold is above
// either value, so the underflow region is empty both ways. Every cutoff
// `int` could hold before is unchanged; the ones it could not were wrapping.
CUDA_HOST_DEVICE_INLINE SuperfpCutoffs superfp_region_cutoffs(int man_bits, int exp_bits,
                                                               int normal_binades, int bias)
{
    SuperfpCutoffs c;
    c.normal_cutoff = ((1 << exp_bits) - 1 - bias) - normal_binades + 1;
    const int64_t span = ((int64_t)1 << exp_bits) - normal_binades;
    const int64_t cutoff = (int64_t)c.normal_cutoff - span * ((int64_t)1 << man_bits) + 1;
    const int64_t floor = (int64_t)INT32_MIN + 1;
    c.supernormal_cutoff = (int)((cutoff < floor) ? floor : cutoff);
    return c;
}

// The constants the casts below read: the superfp twin of BinaryKParamsT in
// cast_binaryK.h, round-mode-agnostic for the same reason, and flat for the
// same reason (a nested aggregate replicated across the GEMM instantiations
// blew up nvcc's compile time): the primary template is every carrier's, and
// the binary32 specialization repeats its fields before adding the fast
// path's.
//
// The seven fast_* floats each derive from the format fields above them.
// Carrying them costs registers, and rebuilding them at the point of use was
// measured instead: with the mixed superfp split-mac GEMM at 60-64 registers
// either spelling fits the same occupancy, and the rebuilt one is 1-4%
// slower, so they are carried.
template <class T>
struct SuperfpParamsT
{
    using word_t = typename FloatTraits<T>::word_t;
    int man_bits; // kept raw: drives the man_bits > 0 round-formula selection
    int bias;     // kept raw: cast_superfp_odd's man_bits == 0 branch reads it
    // region classification (normal/supernormal/underflow)
    int normal_cutoff;
    int supernormal_cutoff;
    // shared round-to-nearest/up/down/odd bitwise constants
    bool round_bypass;
    word_t round_mask;
    word_t round_tie;
    int round_shift;
    // for clip_normal_range_exponent. No floor among them: superfp's normal
    // arm cannot underflow (see UnderflowMode::NONE in bit_helper.h), so the
    // clip is given the overflow bound and zeros, and the two words a binaryK
    // format carries for its floor stay out of this struct, which keeps them
    // out of the register-bound superfp split-mac kernel.
    SaturationMode saturation_mode;
    word_t max_num;
    // no fast path for this carrier; see BinaryKParamsT's
    static constexpr bool fast_rne = false;
};

template <>
struct SuperfpParamsT<float>
{
    using word_t = uint32_t;
    // the primary template's fields, in its order
    int man_bits;
    int bias;
    int normal_cutoff;
    int supernormal_cutoff;
    bool round_bypass;
    word_t round_mask;
    word_t round_tie;
    int round_shift;
    SaturationMode saturation_mode;
    word_t max_num;
    // Float-arithmetic RNE fast path, see cast_superfp_rne_fast below.
    // Unlike BinaryKParams::fast_rne there is no second half to the gate:
    // superfp takes no SubnormalsMode, so the flag alone decides.
    bool fast_rne;
    float fast_split_c;    // 2^(23 - man_bits) + 1, the Veltkamp split constant
    float fast_normal_min; // 2^normal_cutoff: at or above this, normal region
    float fast_super_min;  // 2^supernormal_cutoff, the smallest supernormal
    float fast_super_half; // 2^(supernormal_cutoff - 1), the flush midpoint
    float fast_max_finite; // the largest finite magnitude (max_num as a float)
    float fast_clamp_hi;   // 2 * max_finite: the clamp before the split
    float fast_ovf;        // inf under OVF_INF, else max_finite
};

using SuperfpParams = SuperfpParamsT<float>;

// Builds the constants for a superfp format with man_bits, exp_bits,
// normal_binades and bias, for the carrier T, which defaults to binary32 as
// make_binaryK_params' does.
template <class T = float>
CUDA_HOST_DEVICE_INLINE SuperfpParamsT<T> make_superfp_params(int man_bits, int exp_bits, int normal_binades,
                                                              int bias, SaturationMode saturation_mode)
{
    using F = FloatTraits<T>;
    SuperfpParamsT<T> p;
    p.man_bits = man_bits;
    p.bias = bias;

    SuperfpCutoffs cutoffs = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);
    p.normal_cutoff = cutoffs.normal_cutoff;
    p.supernormal_cutoff = cutoffs.supernormal_cutoff;

    RoundParamsT<T> round_p = make_round_params<T>(man_bits);
    p.round_bypass = round_p.bypass;
    p.round_mask = round_p.mask;
    p.round_tie = round_p.tie;
    p.round_shift = round_p.shift;

    // The top binade's last code is +infinity outside SAT_FINITE, as in
    // binaryK; superfp spends no code on NaN, signed or not.
    const int reserved_codes = (saturation_mode != SaturationMode::SAT_FINITE);
    NormalRangeParamsT<T> normal_p =
        make_normal_range_params<T>(exp_bits, man_bits, bias, saturation_mode, reserved_codes);
    p.saturation_mode = normal_p.saturation_mode;
    p.max_num = normal_p.max_num;

    if constexpr (F::HAS_FAST_CAST)
    {
        // ---- gate and constants for the float-arithmetic RNE fast path.
        // The normal region's rounding is character-for-character binaryK's,
        // so the first three conditions are make_binaryK_params', unchanged.
        // The rest replace binaryK's subnormal-grid conditions with the ones
        // the supernormal and underflow regions need. See
        // cast_superfp_rne_fast.
        int e_split = 23 - man_bits;                // split_c's exponent
        int max_exp = (int)(p.max_num >> 23) - 127; // max_num's exponent
        int min_exp = -bias + 1;                    // the normal clip's floor
        p.fast_rne =
            // man_bits == 0 takes round_bitwise_nearest_even's structurally
            // different zero-argument overload (ties on the exponent's parity,
            // not the significand's), which the Veltkamp split does not model.
            // The upper bound is 23 rather than binaryK's 22: that one comes
            // from the magic add's headroom, which superfp has no use for, so
            // the only limit left is the split constant's own. At man_bits ==
            // 24 its exponent 23 - man_bits goes negative and the identity
            // breaks (the sweep in dev/benchmarks/gemm_cast_superfp_arith.cu
            // shows 23 clean and 24 mismatching on ~1.7e7 inputs per
            // configuration).
            man_bits >= 1 && man_bits <= 23 &&
            // SAT_PROPAGATE clamps a finite value that overflows but keeps an
            // infinite input infinite, which one saturating compare cannot
            // tell apart once the clamp has run.
            saturation_mode != SaturationMode::SAT_PROPAGATE &&
            // no finite normal value at all (see make_normal_range_params)
            p.max_num != 0 &&
            // 2 * max_finite (the clamp) and fast_split_c * that product must
            // both stay finite.
            max_exp <= 126 && max_exp + e_split + 3 <= 128 &&
            // 2^normal_cutoff is a region boundary the selects compare
            // against, so it has to be a normal binary32 value.
            p.normal_cutoff >= -125 && p.normal_cutoff <= 127 &&
            // So are 2^supernormal_cutoff and its midpoint, unless the whole
            // underflow region sits below binary32, which is the common case
            // for man_bits >= 8 (the cutoff is normal_cutoff minus a multiple
            // of 2^man_bits, so it falls away fast). supernormal_cutoff <=
            // -127 puts every representable input, float32 subnormals and
            // zero included, at or above it, so nothing ever underflows and
            // the two constants are set to zero below rather than to
            // unrepresentable powers of two. -126 is the one value excluded
            // by both arms: there target_exp == -127 is exactly
            // supernormal_cutoff - 1, so the integer path rounds *every*
            // nonzero float32 subnormal up to 2^-126 while a compare against
            // the midpoint 2^-127 would keep only half of them.
            ((p.supernormal_cutoff >= -125 && p.supernormal_cutoff <= 127) ||
             p.supernormal_cutoff <= -127) &&
            // The regions have to be ordered as the integer path's if/else
            // chain reads them. normal_binades == 1 << exp_bits puts
            // supernormal_cutoff one above normal_cutoff, and then an exponent
            // equal to normal_cutoff satisfies the *underflow* test and
            // flushes to zero, while a single `ax >= 2^normal_cutoff` compare
            // would send it to the normal region.
            p.supernormal_cutoff <= p.normal_cutoff &&
            // In the normal region rounding can only raise the exponent, so
            // clip_normal_range_exponent's underflow branch is unreachable and
            // saturation is the compare's only job, provided the region starts
            // at or above that branch's floor.
            p.normal_cutoff >= min_exp;

        bool no_underflow = p.supernormal_cutoff <= -127;
        p.fast_split_c = F::pow2(e_split) + 1.0f;
        p.fast_normal_min = F::pow2(p.normal_cutoff);
        // With the underflow region out of binary32's reach there is no floor
        // to lift a rounded magnitude to, so fast_super_min is zero and the
        // fmax leaves the rounding alone, which is what "nothing underflows"
        // means.
        //
        // fast_super_half is still a real threshold there, because the
        // *rounding* has one binary32 does not: the nearest power of two to a
        // float32 subnormal below 2^-127 is zero, so the supernormal arm
        // returns a magnitude of zero for those and for nothing else (the
        // word test is ab < 0x00400000, and ab == 0x00400000 ties up to
        // 2^-126). Naming that boundary here, the largest float32 below
        // 2^-127, rather than 0 is what keeps a zero out of the fast path's
        // copysign, and with it the sign-clear that would otherwise have to
        // run on every value.
        p.fast_super_min = no_underflow ? 0.0f : F::pow2(p.supernormal_cutoff);
        const uint32_t below_tie = F::BELOW_TIE;
        p.fast_super_half = no_underflow ? reinterpret_cast<const T &>(below_tie) : F::pow2(p.supernormal_cutoff - 1);
        p.fast_max_finite = reinterpret_cast<const T &>(p.max_num);
        p.fast_clamp_hi = 2.0f * p.fast_max_finite;
        const uint32_t inf_bits = F::INF_BITS;
        p.fast_ovf = (saturation_mode == SaturationMode::OVF_INF) ? reinterpret_cast<const T &>(inf_bits) : p.fast_max_finite;
        if (!p.fast_rne)
        {
            // keep the unused constants finite so a disabled gate can never
            // produce a signalling value if the path is ever entered by mistake
            p.fast_split_c = 1.0f;
            p.fast_normal_min = 0.0f;
            p.fast_super_min = 0.0f;
            p.fast_super_half = 0.0f;
            p.fast_max_finite = 0.0f;
            p.fast_clamp_hi = 0.0f;
            p.fast_ovf = 0.0f;
        }
    }
    return p;
}

// Float-arithmetic replacement for cast_superfp_nearest_even's bit-twiddling
// body, for the formats make_superfp_params' gate admits. The superfp twin of
// cast_binaryK_rne_fast (see the note there for the Veltkamp split and the
// contraction contract), and it borrows that function's normal-region half
// unchanged: superfp's normal branch is character-for-character binaryK's,
// so a Veltkamp split rounds it to man_bits + 1 significand bits and one
// compare saturates it.
//
// What differs is everything below normal_cutoff. binaryK has a subnormal
// grid of fixed absolute spacing, which a magic-constant add lands on in two
// float operations. superfp instead has the supernormal region, which rounds
// to the nearest *power of two* with ties broken to the even exponent, a
// rule about the exponent field rather than the significand, so no sequence
// of float operations reproduces it (a 1-bit Veltkamp split ties the other
// way: it sends 3.0 to 4.0 where superfp sends it to 2.0). That region
// therefore stays bitwise, but it is the cheap half of the integer path: six
// branchless integer operations, against the ~17 data-dependent branches the
// fast path replaces overall.
//
// One float compare places `ax` in a region, and unlike the binaryK twin the
// two arms stay behind a real branch rather than being computed and selected
// between. That is a measurement, not a preference: the region below
// 2^normal_cutoff, where ordinary data lives (normal_binades is usually 1 or
// 2), is already five operations in the integer path, so computing the
// normal arm as well and selecting it away costs 7% in the GEMM
// (dev/benchmarks/gemm_cast_superfp_arith.cu measures both forms). Warps are
// close to region-uniform for real operand distributions, so the branch is
// nearly free where the extra arm is not.
//
//   ax >= 2^normal_cutoff            normal      -> Veltkamp split, saturate
//   2^(supernormal_cutoff - 1) < ax  supernormal -> nearest power of two, at
//                                                   least 2^supernormal_cutoff
//   otherwise                        underflow   -> +0.0
//
// The second is stated as a strict compare against the *midpoint* rather than
// against 2^supernormal_cutoff because that is what the integer path's
// underflow branch does: at the topmost underflow exponent any nonzero
// significand rounds up, and exactly the midpoint ties to even, which is zero.
//
// NaN passes through unchanged and inf is handled as the integer path's
// target_exp == 128 arm does (saturate_nonfinite: inf under OVF_INF, the
// largest finite value under SAT_FINITE). Neither needs a branch of its own:
// `inf >= fast_normal_min` is true so inf takes the normal arm, where the
// clamp and the saturating compare produce exactly fast_ovf, and every
// compare against NaN is false so NaN takes the supernormal one.
//
// binary32's only, like the binaryK twin, and behind MPTORCH_FAST_CAST for
// the same reason: the split's `t` must be a separately rounded value.
#if defined(MPTORCH_FAST_CAST)
CUDA_HOST_DEVICE_INLINE float cast_superfp_rne_fast(float origin_float, const SuperfpParams &p)
{
    using F = FloatTraits<float>;
    float ax = fabsf(origin_float);
    const uint32_t inf_bits = F::INF_BITS;
    const float inf = reinterpret_cast<const float &>(inf_bits);

    if (ax >= p.fast_normal_min)
    {
        // normal region: clamp before rounding so the Veltkamp product cannot
        // overflow; anything at or above the clamp is out of range either way.
        float xc = F::fmin_nonneg(ax, p.fast_clamp_hi);
        float t = F::rn_mul(p.fast_split_c, xc);
        float nrm = F::rn_sub(t, F::rn_sub(t, xc));
        if (nrm > p.fast_max_finite)
            nrm = p.fast_ovf;
        // `<= inf` rather than `< inf`: inf itself is out of range and takes
        // nrm, which the compare above already made fast_ovf (inf under
        // OVF_INF, the largest finite value under SAT_FINITE), and only a
        // NaN (which never reaches this arm) would fail the compare. `nrm` is
        // never zero (|x| >= 2^normal_cutoff, and the split keeps its
        // exponent), so this return has nothing to unsign.
        return copysignf((ax <= inf) ? nrm : ax, origin_float);
    }

    // supernormal region: round |x| to the nearest power of two, ties to the
    // even exponent (round_bitwise_nearest_even's zero-argument overload,
    // inlined here on a sign-cleared word).
    uint32_t ab = reinterpret_cast<const uint32_t &>(ax);
    uint32_t q = (ab + 0x00400000u) & 0xFF800000u;
    uint32_t is_tie = (ab & 0x007FFFFFu) == 0x00400000u;
    q -= ((is_tie << 23) & ~q);
    float spn = reinterpret_cast<const float &>(q);
    // The sign goes on the supernormal magnitude, which fast_super_half
    // guarantees is nonzero, and the flush arm then selects a bare +0.0 over
    // it: P3109's zero is unsigned, and so is superfp's. Signing first costs
    // nothing (it is the same copysign, one select earlier) and is what lets
    // this path return without a sign-clear on every value. The max covers
    // the topmost underflow binade, whose nearest power of two is below the
    // region but which rounds up into it.
    float signed_spn = copysignf(F::fmax_nonneg(spn, p.fast_super_min), origin_float);
    float out = (ax > p.fast_super_half) ? signed_spn : 0.0f;
    // Every compare against a NaN is false, so a NaN falls through to the
    // input itself: exact bit pattern, payload and all.
    return (ax < inf) ? out : origin_float;
}
#endif

// Rounds to nearest, ties to even: the NearestTiesToEven projection of
// origin_float onto the superfp format p describes. A negative input to an
// unsigned format returns 0. Every arm returns a word that is already
// P3109's unsigned zero where it is zero, so nothing is unsigned on the way
// out.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_nearest_even(T origin_float, bool is_signed, const SuperfpParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    if (origin_float < T(0) && !is_signed)
        return T(0);

#if defined(MPTORCH_FAST_CAST_SUPERFP_RNE)
    // Warp-uniform in the single-format kernels and per-slot uniform in the
    // mixed ones, so the test costs a predicated compare and the integer body
    // below is jumped over, not fetched. See cast_binaryK_nearest_even for
    // why the test stands outside the `if constexpr`, and bit_helper.h for
    // why this is the one fast path the host does not take.
    if (p.fast_rne)
    {
        if constexpr (F::HAS_FAST_CAST)
            return cast_superfp_rne_fast(origin_float, p);
    }
#endif

    word_t target, quantize_bits;
    target = reinterpret_cast<const word_t &>(origin_float);
    T quantized{T(0)};

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;
    if (target_exp == F::INF_EXP)
    {
        // NaN/inf inputs: NaN passes through, inf saturates under SAT_FINITE
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (supernormal)
    {
        // unsigned_zero_bits because a subnormal of the carrier reaches this
        // arm when the underflow region is out of the carrier's reach
        // (supernormal_cutoff below its smallest exponent field), and rounding
        // it to the nearest power of two can clear the magnitude while keeping
        // the sign. Two instructions on a word already in a register.
        quantize_bits = unsigned_zero_bits(round_bitwise_nearest_even(target));
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (underflow)
    {
        // ties-to-even between 0 and the smallest supernormal magnitude
        // (2^supernormal_cutoff): only the topmost underflow exponent
        // (supernormal_cutoff - 1) can be at or above the halfway point;
        // anything strictly below always flushes to zero, +0.0, since superfp
        // has no -0.0 encoding any more than P3109 does.
        word_t sign_bit = target & F::SIGN_MASK;
        quantize_bits = 0u;
        if (target_exp == p.supernormal_cutoff - 1 && (target & F::MAN_MASK) != 0)
        {
            // strictly above half -> round up to the smallest supernormal;
            // exactly at half (mantissa all zero) ties to even = zero.
            quantize_bits = ((word_t)(p.supernormal_cutoff + F::BIAS) << F::MAN_BITS) | sign_bit;
        }
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else
    {
        quantize_bits = (p.man_bits > 0)
                            ? round_bitwise_nearest_even(target, p.round_bypass, p.round_mask, p.round_tie, p.round_shift)
                            : round_bitwise_nearest_even(target);
        quantize_bits = clip_normal_range_exponent<UnderflowMode::NONE>(
            target, quantize_bits, p.saturation_mode, word_t(0), word_t(0), p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }

    return quantized; // already unsigned where it is zero, see the arms above
}

// Rounds to nearest, ties away from zero.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_nearest_away(T origin_float, bool is_signed, const SuperfpParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    word_t target, quantize_bits;
    target = reinterpret_cast<const word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (supernormal)
    {
        // see cast_superfp_nearest_even's supernormal arm for the mask
        quantize_bits = unsigned_zero_bits(round_bitwise_nearest_away(target, 0));
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (underflow)
    {
        word_t sign_bit = target & F::SIGN_MASK;
        if (target_exp >= p.supernormal_cutoff - 1)
        {
            quantize_bits = ((word_t)(p.supernormal_cutoff + F::BIAS) << F::MAN_BITS) | sign_bit;
        }
        else
        {
            quantize_bits = 0u; // flushed: the zero is unsigned
        }
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_nearest_away(target, p.round_bypass, p.round_mask, p.round_tie);
        quantize_bits = clip_normal_range_exponent<UnderflowMode::NONE>(
            target, quantize_bits, p.saturation_mode, word_t(0), word_t(0), p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }

    return quantized; // already unsigned where it is zero, see the arms above
}

// Rounds to odd. In the supernormal region the codes are powers of two, so
// "odd" is the parity of the code's index from the bottom of the region and
// an inexact value moves up one power of two when its index is even.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_odd(T origin_float, bool is_signed, const SuperfpParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    word_t target, quantize_bits;
    target = reinterpret_cast<const word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (supernormal)
    {
        int stored_index = target_exp - p.supernormal_cutoff + 1;
        word_t mask = F::MAN_MASK;
        bool sticky = (target & mask) != 0;
        bool already_odd = (stored_index & 1) != 0;
        quantize_bits = target & ~mask;
        if (sticky && !already_odd)
            quantize_bits += (word_t(1) << F::MAN_BITS);
        // see cast_superfp_nearest_even's supernormal arm
        quantize_bits = unsigned_zero_bits(quantize_bits);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (underflow)
    {
        if ((target & F::ABS_MASK) == 0)
        {
            quantized = T(0); // a zero input rounds to the unsigned zero
        }
        else
        {
            word_t sign_bit = target & F::SIGN_MASK;
            quantize_bits = ((word_t)(p.supernormal_cutoff + F::BIAS) << F::MAN_BITS) | sign_bit;
            quantized = reinterpret_cast<const T &>(quantize_bits);
        }
    }
    else
    {
        if (p.man_bits > 0)
        {
            quantize_bits = round_bitwise_odd(target, p.round_bypass, p.round_mask);
        }
        else
        {
            word_t mask = F::MAN_MASK;
            bool sticky = (target & mask) != 0;
            bool already_odd = ((target_exp + p.bias) & 1) != 0;
            quantize_bits = target & ~mask;
            if (sticky && !already_odd)
                quantize_bits += (word_t(1) << F::MAN_BITS);
        }
        quantize_bits = clip_normal_range_exponent<UnderflowMode::NONE>(
            target, quantize_bits, p.saturation_mode, word_t(0), word_t(0), p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }

    return quantized; // already unsigned where it is zero, see the arms above
}

// Rounds a magnitude up (away from zero), for cast_superfp_up and
// cast_superfp_down: assumes origin_float >= 0.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_absolute_up(T origin_float, const SuperfpParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    word_t target, quantize_bits;
    target = reinterpret_cast<const word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (supernormal)
    {
        // the mask is for the -0.0 input, which compares >= 0 and so arrives
        // here as a magnitude with a sign bit; see the RNE cast's arm
        quantize_bits = unsigned_zero_bits(round_bitwise_up(target, 0));
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (underflow)
    {
        if ((target & F::ABS_MASK) == 0)
        {
            quantized = T(0); // -0.0 included: the zero is unsigned
        }
        else
        {
            quantize_bits = (word_t)(p.supernormal_cutoff + F::BIAS) << F::MAN_BITS;
            quantized = reinterpret_cast<const T &>(quantize_bits);
        }
    }
    else
    {
        quantize_bits = round_bitwise_up(target, p.round_bypass, p.round_mask);
        // the magnitude form: this helper is handed |x|, so the sign the
        // signed form puts back is always the one it took off (bit_helper.h)
        quantize_bits = clip_normal_range_magnitude<UnderflowMode::NONE>(
            target, quantize_bits, p.saturation_mode, word_t(0), word_t(0), p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }

    return quantized;
}

// Rounds a magnitude down (toward zero); the other half of
// cast_superfp_absolute_up.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_absolute_down(T origin_float, const SuperfpParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    word_t target, quantize_bits;
    target = reinterpret_cast<const word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (supernormal)
    {
        // see cast_superfp_absolute_up's arm
        quantize_bits = unsigned_zero_bits(round_bitwise_down(target, 0));
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (underflow)
    {
        quantize_bits = 0u; // flushed: the zero is unsigned
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_down(target, p.round_bypass, p.round_mask);
        // the magnitude form: this helper is handed |x|, so the sign the
        // signed form puts back is always the one it took off (bit_helper.h)
        quantize_bits = clip_normal_range_magnitude<UnderflowMode::NONE>(
            target, quantize_bits, p.saturation_mode, word_t(0), word_t(0), p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }

    return quantized;
}

// The three directed modes (toward +inf, toward -inf, toward zero) in terms
// of the two magnitude helpers above.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_up(T origin_float, bool is_signed, const SuperfpParamsT<T> &p)
{
    if (origin_float < T(0) && !is_signed)
        return T(0);

    // NaN comparisons are always false, so a NaN takes the negating arm
    // below. Negating on the word (bit_helper.h's flip_sign and
    // negate_magnitude) carries it through the helper's own NaN/inf arm
    // untouched, where `-x` on the device may canonicalize a NaN's payload
    // and sign. As in cast_binaryK_up, the helper gives back a magnitude, so
    // that same negation is the one way left to a -0.0, and negate_magnitude
    // leaves a zero unsigned.
    if (origin_float >= 0)
        return cast_superfp_absolute_up(origin_float, p);
    else
        return negate_magnitude(cast_superfp_absolute_down(flip_sign(origin_float), p));
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_down(T origin_float, bool is_signed, const SuperfpParamsT<T> &p)
{
    if (origin_float < T(0) && !is_signed)
        return T(0);

    // see cast_superfp_up
    if (origin_float >= 0)
        return cast_superfp_absolute_down(origin_float, p);
    else
        return negate_magnitude(cast_superfp_absolute_up(flip_sign(origin_float), p));
}

template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_zero(T origin_float, bool is_signed, const SuperfpParamsT<T> &p)
{
    if (origin_float >= T(0))
        return cast_superfp_down(origin_float, is_signed, p);
    else
        return cast_superfp_up(origin_float, is_signed, p);
}

// Stochastic rounding. Like cast_binaryK_stochastic, rand_prob (a word of
// the carrier whose MAN_BITS low bits are the draw) and rand_bits stay
// per-call arguments; only the region-cutoff and bitwise constants
// precompute away. In the supernormal region the draw is applied to the
// whole mantissa field, so the value moves to the next power of two with
// probability equal to its fractional position between the two.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_superfp_stochastic(T origin_float, typename FloatTraits<T>::word_t rand_prob,
                                                  int rand_bits, bool is_signed, const SuperfpParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    if (origin_float < T(0) && !is_signed)
        return T(0);

    word_t target, quantize_bits;
    target = reinterpret_cast<const word_t &>(origin_float);
    T quantized;

    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    word_t rand_bits_raw = rand_prob & F::MAN_MASK;

    if (target_exp == F::INF_EXP)
    {
        quantize_bits = saturate_nonfinite(target, p.saturation_mode, p.max_num);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (supernormal)
    {
        // see cast_superfp_nearest_even's supernormal arm
        quantize_bits = unsigned_zero_bits(round_bitwise_stochastic(target, rand_bits_raw, 0));
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }
    else if (underflow)
    {
        // Adding the smallest supernormal (with the input's sign) moves the
        // value into that binade, where a stochastic round to zero mantissa
        // bits picks between it and the next power of two; the exact
        // subtraction afterwards turns that into a pick between 0 and the
        // smallest supernormal, with probability proportional to |x|.
        word_t sign_bit = target & F::SIGN_MASK;
        word_t shift_bits = ((word_t)(p.supernormal_cutoff + F::BIAS) << F::MAN_BITS) | sign_bit;
        T shift_float = reinterpret_cast<const T &>(shift_bits);
        T val = origin_float + shift_float;
        word_t target2 = reinterpret_cast<const word_t &>(val);
        quantize_bits = round_bitwise_stochastic(target2, rand_bits_raw, 0);
        quantized = reinterpret_cast<const T &>(quantize_bits) - shift_float;
    }
    else
    {
        word_t rand_prob_man = rand_bits_raw & ~((word_t(1) << (F::MAN_BITS - p.man_bits - rand_bits)) - 1u);
        quantize_bits = round_bitwise_stochastic(target, rand_prob_man, p.man_bits);
#if defined(MPTORCH_FAST_CAST)
        // The same compare-instead-of-clip as cast_binaryK_stochastic's normal
        // arm; see the note there for why one magnitude test reproduces
        // clip_normal_range_exponent. The clip's underflow arm is unreachable
        // for the same reason in different words: this arm is entered only at
        // or above normal_cutoff, the gate's `normal_cutoff >= min_exp` puts
        // that at or above the arm's floor, and `y` is nonzero.
        if (p.fast_rne)
        {
            if constexpr (F::HAS_FAST_CAST)
            {
                T y = reinterpret_cast<const T &>(quantize_bits);
                return (fabsf(y) > p.fast_max_finite) ? copysignf(p.fast_ovf, origin_float) : y;
            }
        }
#endif
        quantize_bits = clip_normal_range_exponent<UnderflowMode::NONE>(
            target, quantize_bits, p.saturation_mode, word_t(0), word_t(0), p.max_num,
            rand_prob_man);
        quantized = reinterpret_cast<const T &>(quantize_bits);
    }

    return quantized; // already unsigned where it is zero, see the arms above
}
