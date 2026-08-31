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
};

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
    return p;
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_nearest_even(float origin_float, bool is_signed,
                                                        SubnormalsMode subnormals, const BinaryKParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

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
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}
