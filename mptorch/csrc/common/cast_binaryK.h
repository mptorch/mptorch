#pragma once

#include "bit_helper.h"
#include "modes.h"

CUDA_HOST_DEVICE_INLINE float cast_binaryK_nearest_even(float origin_float, int man_bits, int exp_bits,
                                                        int bias, bool is_signed,
                                                        SaturationMode saturation_mode,
                                                        SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
        uint32_t rounded_val = (man_bits > 0)
                                   ? round_bitwise_nearest_even(target, exp_diff)
                                   : round_bitwise_nearest_even(target);
        quantize_bits = not_uflow * rounded_val;
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits,
                                                      exp_bits, man_bits, bias);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        quantize_bits = (man_bits > 0)
                            ? round_bitwise_nearest_even(target, man_bits)
                            : round_bitwise_nearest_even(target);
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_binaryK_nearest_even: reads a
// BinaryKParams computed once at Multiplier/Adder construction
// (gemm_policy.h) instead of re-deriving man_bits/exp_bits/bias-derived
// constants on every call. See bit_helper.h's "Precomputed-parameter
// overloads" note. Only covers RNE, matching the GEMM core's current scope.
// Flat by design -- see dev/gemm_core_roadmap.md item 6.
struct BinaryKParams
{
    int man_bits; // kept raw: drives the subnormal branch's data-dependent
                  // exp_diff and the man_bits > 0 round-formula selection,
                  // neither of which can be precomputed away
    int min_exp;  // = -bias + 1
    // shared round-to-nearest/up/down/odd bitwise constants
    bool round_bypass;
    uint32_t round_mask;
    uint32_t round_tie;
    int round_shift;
    // for clip_normal_range_exponent
    SaturationMode saturation_mode;
    int max_exponent_store;
    int min_exponent_store;
    uint32_t max_num;
    // for clip_subnormal_range_exponent
    int subnormal_min_exponent_store;
    // Float-arithmetic RNE fast path -- see cast_binaryK_rne_fast below.
    // fast_rne is the format half of the gate; the caller ANDs it with
    // subnormals == SubnormalsMode::SUBNORMALS.
    bool fast_rne;
    float fast_split_c;    // 2^(23 - man_bits) + 1, the Veltkamp splitting constant
    float fast_magic;      // 1.5 * 2^(23 + min_exp - man_bits), the subnormal grid's magic constant
    float fast_min_normal; // 2^min_exp: below this the subnormal grid applies
    float fast_max_finite; // largest finite magnitude the format stores (= max_num)
    float fast_clamp_hi;   // 2 * max_finite: keeps fast_split_c * x from overflowing
    float fast_ovf;        // what an out-of-range magnitude becomes (inf, or max_finite)
};

// 2^e as a float, for e in [-126, 127]. Used to build the fast path's
// constants without pulling <cmath> into a header that CUDA device code
// includes; the values are all exact powers of two by construction.
CUDA_HOST_DEVICE_INLINE float binaryK_pow2f(int e)
{
    uint32_t bits = (uint32_t)(e + 127) << 23;
    return BITS_TO_FLOAT(&bits);
}

CUDA_HOST_DEVICE_INLINE BinaryKParams make_binaryK_params(int man_bits, int exp_bits, int bias,
                                                          SaturationMode saturation_mode,
                                                          bool extended_normals)
{
    BinaryKParams p;
    p.man_bits = man_bits;
    p.min_exp = -bias + 1;

    RoundParams round_p = make_round_params(man_bits);
    p.round_bypass = round_p.bypass;
    p.round_mask = round_p.mask;
    p.round_tie = round_p.tie;
    p.round_shift = round_p.shift;

    NormalRangeParams normal_p = make_normal_range_params(exp_bits, man_bits, bias, saturation_mode, extended_normals);
    p.saturation_mode = normal_p.saturation_mode;
    p.max_exponent_store = normal_p.max_exponent_store;
    p.min_exponent_store = normal_p.min_exponent_store;
    p.max_num = normal_p.max_num;

    p.subnormal_min_exponent_store = make_subnormal_range_params(man_bits, bias).min_exponent_store;

    // ---- gate and constants for the float-arithmetic RNE fast path.
    // Every condition below is a range check on the derived exponents: each
    // guarantees that one of the fast path's four float operations stays
    // inside binary32 and therefore exact. See cast_binaryK_rne_fast.
    int e_split = 23 - man_bits;                     // fast_split_c = 2^e_split + 1
    int e_magic = 23 + p.min_exp - man_bits;         // fast_magic  = 1.5 * 2^e_magic
    int max_exp = p.max_exponent_store - 127;        // unbiased exponent of max_num
    p.fast_rne =
        // man_bits == 0 takes round_bitwise_nearest_even's structurally
        // different zero-argument overload (ties on the exponent's parity,
        // not the significand's), which the Veltkamp split does not model;
        // man_bits > 22 leaves the magic add without the half-binade of
        // headroom it needs to stay inside its own binade.
        man_bits >= 1 && man_bits <= 22 &&
        // SAT_PROPAGATE leaves a value whose exponent is exactly
        // max_exponent_store unclamped even when its significand exceeds
        // max_num, so max_num is not that mode's largest finite magnitude
        // and the single fast_max_finite compare cannot express it.
        saturation_mode != SaturationMode::SAT_PROPAGATE &&
        // 2^min_exp, 1.5 * 2^e_magic and the subnormal grid step
        // 2^(min_exp - man_bits) must all be normal binary32 values.
        p.min_exp >= -126 && p.min_exp <= 126 &&
        p.min_exp - man_bits >= -126 &&
        e_magic >= -126 && e_magic <= 126 &&
        // 2 * max_finite (the clamp) and fast_split_c * that product must
        // both stay finite: |clamp_hi| < 2^(max_exp + 2) and
        // fast_split_c < 2^(e_split + 1).
        max_exp <= 126 && max_exp + e_split + 3 <= 128 &&
        // The subnormal range has to sit below the format's largest finite
        // value. It need not: exp_bits == 1 with man_bits == 1 puts
        // max_exponent_store *below* min_exp, and the integer path's
        // subnormal branch runs clip_subnormal_range_exponent, which only
        // handles underflow -- so it returns values above max_num rather
        // than saturating them, which the fast path's single saturating
        // compare cannot reproduce. min_exp <= max_exp implies
        // 2^min_exp <= max_finite, which is the condition that matters.
        p.min_exp <= max_exp;

    p.fast_split_c = binaryK_pow2f(e_split) + 1.0f;
    p.fast_magic = 1.5f * binaryK_pow2f(e_magic);
    p.fast_min_normal = binaryK_pow2f(p.min_exp);
    p.fast_max_finite = BITS_TO_FLOAT(&p.max_num);
    p.fast_clamp_hi = 2.0f * p.fast_max_finite;
    uint32_t inf_bits = 0x7F800000u;
    p.fast_ovf = (saturation_mode == SaturationMode::OVF_INF) ? BITS_TO_FLOAT(&inf_bits) : p.fast_max_finite;
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
    return p;
}

// Float-arithmetic replacement for cast_binaryK_nearest_even's bit-twiddling
// body, for the formats make_binaryK_params' gate admits. Round-to-nearest-
// even is what binary32 hardware already does, so the whole cast reduces to
// putting the value on the target format's grid and letting the FPU round:
//
//   * a Veltkamp split rounds to man_bits + 1 significand bits at any
//     exponent (Dekker's theorem: with c = 2^s + 1, t - (t - x) is x rounded
//     to 24 - s bits, exactly, provided nothing overflows);
//   * a magic-constant add rounds onto the subnormal range's fixed absolute
//     spacing of 2^(min_exp - man_bits);
//   * saturation is one compare and one select on the *rounded* magnitude.
//
// Roughly 14 instructions and no branches, against ~125 instructions with
// ~17 data-dependent branches for the integer path. Verified against that
// path over all 2^32 float inputs for every admitted format --
// dev/benchmarks/gemm_cast_float_arith.cu.
//
// CUDA only, and deliberately so: the identity depends on `t` being a
// separately rounded binary32 value, which the _rn intrinsics guarantee.
// Plain `*`/`-` let nvcc contract the split into an FMA and the identity
// breaks (248 M mismatches in the exhaustive sweep); on the host the same
// hazard exists via -ffp-contract, with no equally cheap way to forbid it,
// so host builds keep the integer path.
#if defined(__CUDA_ARCH__)
CUDA_HOST_DEVICE_INLINE float cast_binaryK_rne_fast(float origin_float, const BinaryKParams &p)
{
    float ax = fabsf(origin_float);
    // clamp before rounding so the Veltkamp product cannot overflow; anything
    // at or above the clamp is out of the format's range either way.
    float xc = copysignf(fminf(ax, p.fast_clamp_hi), origin_float);
    float sub = __fsub_rn(__fadd_rn(xc, p.fast_magic), p.fast_magic);
    float t = __fmul_rn(p.fast_split_c, xc);
    float nrm = __fsub_rn(t, __fsub_rn(t, xc));
    float y = (ax < p.fast_min_normal) ? sub : nrm;
    if (fabsf(y) > p.fast_max_finite)
        y = copysignf(p.fast_ovf, origin_float);
    // inf and NaN pass through unchanged (the integer path's target_exp == 128
    // branch); a single ordered compare covers both.
    return (ax < __int_as_float(0x7F800000)) ? y : origin_float;
}
#endif

CUDA_HOST_DEVICE_INLINE float cast_binaryK_nearest_even(float origin_float, bool is_signed,
                                                        SubnormalsMode subnormals, const BinaryKParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

#if defined(__CUDA_ARCH__)
    // Warp-uniform in the single-format kernels (every thread reads the same
    // acc_proto) and per-slot uniform in the mixed ones, so the test itself
    // costs a predicated compare; the integer body below is jumped over, not
    // fetched.
    if (p.fast_rne && subnormals == SubnormalsMode::SUBNORMALS)
        return cast_binaryK_rne_fast(origin_float, p);
#endif

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    bool subnormal = (target_exp < p.min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = p.man_bits - (p.min_exp - target_exp);
        int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
        uint32_t rounded_val = (p.man_bits > 0)
                                   ? round_bitwise_nearest_even(target, exp_diff)
                                   : round_bitwise_nearest_even(target);
        quantize_bits = not_uflow * rounded_val;
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits, p.subnormal_min_exponent_store);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        quantize_bits = (p.man_bits > 0)
                            ? round_bitwise_nearest_even(target, p.round_bypass, p.round_mask, p.round_tie, p.round_shift)
                            : round_bitwise_nearest_even(target);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_nearest_away(float origin_float, int man_bits, int exp_bits,
                                                        int bias, bool is_signed,
                                                        SaturationMode saturation_mode,
                                                        SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        int not_uflow = exp_diff >= -1;
        quantize_bits = not_uflow * round_bitwise_nearest_away(target, exp_diff);
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits,
                                                      exp_bits, man_bits, bias);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        quantize_bits = round_bitwise_nearest_away(target, man_bits);
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_binaryK_nearest_away. BinaryKParams
// is round-mode-agnostic (its fields use the same formulas regardless of
// which mode reads them), so this reuses the same struct/constructor as
// RNE; round_shift goes unused here.
CUDA_HOST_DEVICE_INLINE float cast_binaryK_nearest_away(float origin_float, bool is_signed,
                                                        SubnormalsMode subnormals, const BinaryKParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    bool subnormal = (target_exp < p.min_exp);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = p.man_bits - (p.min_exp - target_exp);
        int not_uflow = exp_diff >= -1;
        quantize_bits = not_uflow * round_bitwise_nearest_away(target, exp_diff);
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits, p.subnormal_min_exponent_store);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else
    {
        quantize_bits = round_bitwise_nearest_away(target, p.round_bypass, p.round_mask, p.round_tie);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// rounds to odd
CUDA_HOST_DEVICE_INLINE float cast_binaryK_odd(float origin_float, int man_bits, int exp_bits,
                                               int bias, bool is_signed,
                                               SaturationMode saturation_mode,
                                               SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
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
            quantize_bits = round_bitwise_odd(target, exp_diff);
        }
        else
        {
            // precision has collapsed to zero: every value in this range
            // rounds to the smallest subnormal, whose significand is
            // always 1 (odd), so no sticky-driven carry is needed
            quantize_bits = target & ~0x007FFFFFu;
        }
        quantize_bits = clip_subnormal_range_exponent_up(target, quantize_bits,
                                                          exp_bits, man_bits, bias);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        if (man_bits > 0)
        {
            quantize_bits = round_bitwise_odd(target, man_bits);
        }
        else
        {
            // no explicit significand bits: "odd" refers to the parity of
            // the target format's own biased exponent, not the float32
            // exponent field's (their biases generally differ in parity)
            uint32_t mask = 0x007FFFFFu;
            bool sticky = (target & mask) != 0;
            bool already_odd = ((target_exp + bias) & 1) != 0;
            quantize_bits = target & ~mask;
            if (sticky && !already_odd)
                quantize_bits += (1u << 23);
        }
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_binaryK_odd. The man_bits == 0
// branch needs raw `bias`, recovered as `1 - p.min_exp` rather than storing
// it separately.
CUDA_HOST_DEVICE_INLINE float cast_binaryK_odd(float origin_float, bool is_signed,
                                               SubnormalsMode subnormals, const BinaryKParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
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
            quantize_bits = target & ~0x007FFFFFu;
        }
        quantize_bits = clip_subnormal_range_exponent_up(target, quantize_bits, p.subnormal_min_exponent_store);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (target_exp == 128)
    {
        quantized = origin_float;
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
            uint32_t mask = 0x007FFFFFu;
            bool sticky = (target & mask) != 0;
            bool already_odd = ((target_exp + bias) & 1) != 0;
            quantize_bits = target & ~mask;
            if (sticky && !already_odd)
                quantize_bits += (1u << 23);
        }
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_absolute_up(float origin_float, int man_bits, int exp_bits, int bias,
                                               SaturationMode saturation_mode,
                                               SubnormalsMode subnormals)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        quantize_bits = round_bitwise_up(target, exp_diff < 0 ? 0 : exp_diff);
        quantize_bits = clip_subnormal_range_exponent_up(target, quantize_bits,
                                                         exp_bits, man_bits, bias);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        quantize_bits = round_bitwise_up(target, man_bits);
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_absolute_up. The subnormal
// branch's exp_diff is data-dependent (varies per call), so
// round_bitwise_up(target, exp_diff) there is left as-is -- only the normal
// branch's man_bits-derived round precomputes.
CUDA_HOST_DEVICE_INLINE float cast_absolute_up(float origin_float, SubnormalsMode subnormals,
                                               const BinaryKParams &p)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    bool subnormal = (target_exp < p.min_exp);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = p.man_bits - (p.min_exp - target_exp);
        quantize_bits = round_bitwise_up(target, exp_diff < 0 ? 0 : exp_diff);
        quantize_bits = clip_subnormal_range_exponent_up(target, quantize_bits, p.subnormal_min_exponent_store);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else
    {
        quantize_bits = round_bitwise_up(target, p.round_bypass, p.round_mask);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_absolute_down(float origin_float, int man_bits, int exp_bits,
                                                 int bias, SaturationMode saturation_mode,
                                                 SubnormalsMode subnormals)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && subnormals == SubnormalsMode::SUBNORMALS)
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        int not_uflow = exp_diff > -1;
        quantize_bits = not_uflow * round_bitwise_down(target, exp_diff);
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits,
                                                      exp_bits, man_bits, bias);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        quantize_bits = round_bitwise_down(target, man_bits);
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_absolute_down -- see
// cast_absolute_up's precomputed overload above.
CUDA_HOST_DEVICE_INLINE float cast_absolute_down(float origin_float, SubnormalsMode subnormals,
                                                 const BinaryKParams &p)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    bool subnormal = (target_exp < p.min_exp);

    if (subnormal && subnormals == SubnormalsMode::SUBNORMALS)
    {
        int exp_diff = p.man_bits - (p.min_exp - target_exp);
        int not_uflow = exp_diff > -1;
        quantize_bits = not_uflow * round_bitwise_down(target, exp_diff);
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits, p.subnormal_min_exponent_store);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else
    {
        quantize_bits = round_bitwise_down(target, p.round_bypass, p.round_mask);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_up(float origin_float, int man_bits, int exp_bits, int bias,
                                              bool is_signed, SaturationMode saturation_mode,
                                              SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    if (origin_float >= 0)
        return cast_absolute_up(origin_float, man_bits, exp_bits, bias,
                                saturation_mode, subnormals);
    else
        return -cast_absolute_down(-origin_float, man_bits, exp_bits, bias,
                                   saturation_mode, subnormals);
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_down(float origin_float, int man_bits, int exp_bits,
                                                int bias, bool is_signed,
                                                SaturationMode saturation_mode,
                                                SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    if (origin_float >= 0)
        return cast_absolute_down(origin_float, man_bits, exp_bits, bias,
                                  saturation_mode, subnormals);
    else
        return -cast_absolute_up(-origin_float, man_bits, exp_bits, bias,
                                 saturation_mode, subnormals);
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_zero(float origin_float, int man_bits, int exp_bits,
                                                int bias, bool is_signed,
                                                SaturationMode saturation_mode,
                                                SubnormalsMode subnormals)
{
    if (origin_float >= 0.0f)
        return cast_binaryK_down(origin_float, man_bits, exp_bits, bias, is_signed,
                                 saturation_mode, subnormals);
    else
        return cast_binaryK_up(origin_float, man_bits, exp_bits, bias, is_signed,
                               saturation_mode, subnormals);
}

// Precomputed-parameter overloads of cast_binaryK_up/_down/_zero above.
CUDA_HOST_DEVICE_INLINE float cast_binaryK_up(float origin_float, bool is_signed,
                                              SubnormalsMode subnormals, const BinaryKParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    if (origin_float >= 0)
        return cast_absolute_up(origin_float, subnormals, p);
    else
        return -cast_absolute_down(-origin_float, subnormals, p);
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_down(float origin_float, bool is_signed,
                                                SubnormalsMode subnormals, const BinaryKParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    if (origin_float >= 0)
        return cast_absolute_down(origin_float, subnormals, p);
    else
        return -cast_absolute_up(-origin_float, subnormals, p);
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_zero(float origin_float, bool is_signed,
                                                SubnormalsMode subnormals, const BinaryKParams &p)
{
    if (origin_float >= 0.0f)
        return cast_binaryK_down(origin_float, is_signed, subnormals, p);
    else
        return cast_binaryK_up(origin_float, is_signed, subnormals, p);
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_stochastic(float origin_float, uint32_t rand_prob,
                                                      int rand_bits, int man_bits, int exp_bits, int bias,
                                                      bool is_signed, SaturationMode saturation_mode,
                                                      SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    rand_prob = rand_prob & 0x007FFFFFu;
    rand_prob = rand_prob & ~((1u << (23 - man_bits - rand_bits)) - 1u);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        float shift_float, val;
        int shift_bits = ((127 + min_exp) << 23) | (target & 0x80000000u);
        shift_float = BITS_TO_FLOAT(&shift_bits);
        val = origin_float + shift_float;
        target = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else
    {
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_binaryK_stochastic. Unlike the
// deterministic modes, this still needs a per-call random value (rand_prob)
// and rand_bits (a Multiplier/Adder-fixed but role-specific width, not
// folded into BinaryKParams) -- only the round-to-nearest-even bit-
// twiddling constants precompute away.
CUDA_HOST_DEVICE_INLINE float cast_binaryK_stochastic(float origin_float, uint32_t rand_prob, int rand_bits,
                                                       bool is_signed, SubnormalsMode subnormals,
                                                       const BinaryKParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    bool subnormal = (target_exp < p.min_exp);

    rand_prob = rand_prob & 0x007FFFFFu;
    rand_prob = rand_prob & ~((1u << (23 - p.man_bits - rand_bits)) - 1u);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        float shift_float, val;
        int shift_bits = ((127 + p.min_exp) << 23) | (target & 0x80000000u);
        shift_float = BITS_TO_FLOAT(&shift_bits);
        val = origin_float + shift_float;
        target = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target, rand_prob, p.man_bits);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else
    {
        quantize_bits = round_bitwise_stochastic(target, rand_prob, p.man_bits);
#if defined(__CUDA_ARCH__)
        // The SR half of item 7. Stochastic rounding itself has no hardware
        // analogue -- binary32 rounds to nearest, so nothing in the FPU
        // reproduces "add random bits below the retained significand, then
        // truncate", and the three lines above stay exactly as they are. What
        // SR *does* share with RNE is everything after the rounding, and
        // clip_normal_range_exponent is the expensive part: five branches to
        // decide a value the same single compare G3 uses can decide.
        //
        // Two facts make the compare sufficient here, and both are already in
        // the gate. Saturation is a magnitude test on the rounded value --
        // that is fast_rne's `saturation_mode != SAT_PROPAGATE`. And the clip's
        // *underflow* arm is unreachable: `add_r & ~mask` clears low
        // significand bits but never lowers an exponent, so a value that
        // entered this branch at or above min_exp leaves it there too, which
        // is what `subnormals == SUBNORMALS` guarantees (anything below took
        // the subnormal arm above).
        //
        // The flag is fast_rne rather than one of its own: its remaining
        // conditions are the Veltkamp split's, which SR does not use, so
        // reusing it only ever gates SR off where it could have run
        // (man_bits == 0, and the split-product bound). One flag, one sweep.
        // Verified in dev/benchmarks/gemm_cast_sr_arith.cu.
        if (p.fast_rne && subnormals == SubnormalsMode::SUBNORMALS)
        {
            float y = BITS_TO_FLOAT(&quantize_bits);
            return (fabsf(y) > p.fast_max_finite) ? copysignf(p.fast_ovf, origin_float) : y;
        }
#endif
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}
