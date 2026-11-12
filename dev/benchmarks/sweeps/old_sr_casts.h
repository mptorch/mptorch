#pragma once

// binaryK's and superfp's stochastic casts as they stood at 4164cee, before
// cf69807 rounded every region exactly in integers on the word
// (common/stochastic.h), frozen here as the negative control of
// cast_all_modes_sweep's `sr-mean-old`: binary32, on the host, with the
// helpers they called (round_bitwise_stochastic, the clip's STOCHASTIC floor)
// copied in, since the shipped headers no longer have them. What it should
// show: the carrier add's error near draw levels in binaryK's SUBNORMALS arm
// and superfp's arm below its smallest value, and, wherever prng_bits leaves
// draw bits unused, the full draw of superfp's supernormal binades and the
// man_bits + N bits of binaryK's floors.

#include "common/cast_binaryK.h"
#include "common/cast_superfp.h"
#include <cstring>

namespace old_sr
{
    inline uint32_t w_of(float f)
    {
        uint32_t w;
        memcpy(&w, &f, sizeof w);
        return w;
    }
    inline float f_of(uint32_t w)
    {
        float f;
        memcpy(&f, &w, sizeof f);
        return f;
    }

    inline uint32_t round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits)
    {
        const uint32_t mask = (uint32_t)((1 << (23 - man_bits)) - 1);
        return (target + (rand_prob & mask)) & ~mask;
    }

    // clip_normal_range_exponent<UnderflowMode::STOCHASTIC> (the floor asked
    // of the input, one draw u * min_val) and <NONE>
    inline uint32_t clip(uint32_t old_num, uint32_t quantized, SaturationMode sat, uint32_t min_num,
                         uint32_t max_num, uint32_t rand_prob, bool floor)
    {
        const uint32_t ax = old_num & 0x7FFFFFFFu;
        if (ax == 0)
            return 0;
        quantized &= 0x7FFFFFFFu;
        if (quantized > max_num)
            quantized = sat == SaturationMode::SAT_NONE ? 0x7F800000u : max_num;
        else if (floor && ax < min_num)
        {
            const float u = (float)(rand_prob & 0x007FFFFFu) * (1.0f / 8388608.0f);
            const float bound = u * f_of(min_num);
            quantized = ax > w_of(bound) ? min_num : 0u;
        }
        return quantized | ((old_num & 0x80000000u) & (0u - (uint32_t)(quantized != 0)));
    }

    inline float cast_binaryK(float x, uint32_t rand_prob, int rand_bits, bool is_signed, SubnormalsMode sub,
                              const BinaryKParams &p)
    {
        if (x < 0 && !is_signed)
            return 0.0f;
        uint32_t target = w_of(x);
        const int target_exp = (int)((target >> 23) & 0xFFu) - 127;
        rand_prob &= 0x007FFFFFu;
        rand_prob &= ~((1u << (23 - p.man_bits - rand_bits)) - 1u);
        if (target_exp < p.min_exp && sub == SubnormalsMode::SUBNORMALS)
        {
            const float shift = f_of(((uint32_t)(127 + p.min_exp) << 23) | (target & 0x80000000u));
            const float val = x + shift;
            return f_of(round_bitwise_stochastic(w_of(val), rand_prob, p.man_bits)) - shift;
        }
        if (target_exp == 128)
            return f_of(saturate_nonfinite(target, p.saturation_mode, p.max_num));
        const uint32_t q = round_bitwise_stochastic(target, rand_prob, p.man_bits);
        return f_of(clip(target, q, p.saturation_mode, p.min_num, p.max_num, rand_prob, true));
    }

    inline float cast_superfp(float x, uint32_t rand_prob, int rand_bits, bool is_signed, const SuperfpParams &p)
    {
        if (x < 0 && !is_signed)
            return 0.0f;
        const uint32_t target = w_of(x);
        const int target_exp = (int)((target >> 23) & 0xFFu) - 127;
        const uint32_t raw = rand_prob & 0x007FFFFFu;
        if (target_exp == 128)
            return f_of(saturate_nonfinite(target, p.saturation_mode, p.max_num));
        if (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff)
            return f_of(round_bitwise_stochastic(target, raw, 0));
        if (target_exp < p.supernormal_cutoff)
        {
            const float shift = f_of(((uint32_t)(p.supernormal_cutoff + 127) << 23) | (target & 0x80000000u));
            const float val = x + shift;
            return f_of(round_bitwise_stochastic(w_of(val), raw, 0)) - shift;
        }
        const uint32_t man = raw & ~((1u << (23 - p.man_bits - rand_bits)) - 1u);
        const uint32_t q = round_bitwise_stochastic(target, man, p.man_bits);
        return f_of(clip(target, q, p.saturation_mode, 0u, p.max_num, man, false));
    }
} // namespace old_sr
