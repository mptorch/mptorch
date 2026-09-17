#pragma once

// Rounding a binary64 value onto a narrower IEEE binary format, once.
//
// A tensor narrower than its carrier stores what the carrier computed, and
// that store is a rounding of its own. Under `carrier=torch.float64` the
// carrier's result is float64 and the store is float64 -> float32, float16
// or bfloat16, and torch's own conversion to the last two goes through
// float32 on both backends, which rounds twice: a value a hair past a
// midpoint of the narrow grid is first rounded onto the midpoint and then to
// even, so 65519.999999 comes back as float16 infinity instead of 65504. This
// is the conversion done once, on the word, for the narrow_float64 op that
// mptorch/quant/ops.py's `_narrowed` calls.
//
// `ExpBits` and `ManBits` are the target's fields and `Word` its storage: (5,
// 10) is float16, (8, 7) bfloat16 and (8, 23) float32. The result is IEEE 754's
// convertFormat under roundTiesToEven: a finite value rounds to the nearest
// value of the target, a tie to the even one; a value that rounds past the
// largest finite value is an infinity of its sign, and one at or below half
// the smallest subnormal a zero of its sign; an infinity stays one, and
// a NaN stays a NaN, quieted and keeping the top of its payload, as the
// hardware conversions do. For (8, 23) that is exactly
// `static_cast<float>(double)`, which tests/test_float64_carrier.py holds it
// to, along with a Fraction reference at every tie.
//
// Integer arithmetic on the words rather than a float division and a
// nearbyint: nothing here depends on the floating-point environment, a
// contraction flag or the device's rounding of a conversion, so the host and
// the device objects compute the same bits by construction.

#include "bit_helper.h"
#include <cstdint>

template <class Word, int ExpBits, int ManBits>
CUDA_HOST_DEVICE_INLINE Word narrow_binary64(uint64_t w)
{
    constexpr int bias = (1 << (ExpBits - 1)) - 1;
    constexpr int min_normal_exp = 1 - bias;
    constexpr uint64_t man_mask = (uint64_t{1} << ManBits) - 1;
    constexpr uint64_t inf_word = ((uint64_t{1} << ExpBits) - 1) << ManBits;

    const uint64_t sign = (w >> 63) << (ExpBits + ManBits);
    const int field = static_cast<int>((w >> 52) & 0x7FF);
    const uint64_t frac = w & ((uint64_t{1} << 52) - 1);

    if (field == 0x7FF)
    {
        // an infinity, or a NaN: the quiet bit set, and the payload's top bits
        const uint64_t nan = (frac >> (52 - ManBits)) | (uint64_t{1} << (ManBits - 1));
        return static_cast<Word>(sign | inf_word | (frac ? nan : 0));
    }

    // the exponent of x's leading bit, for a normal x
    const int e = field - 1023;
    if (e > bias)
        return static_cast<Word>(sign | inf_word);

    // How many of the significand's 53 bits fall below the target's last one:
    // all but ManBits under the leading bit, and more below the target's
    // smallest normal, where its grid stops shrinking. Past 63 nothing reaches
    // even half the smallest subnormal. A binary64 subnormal (field 0) always
    // lands there, so the implicit bit below is only ever a normal's.
    const int shift = 52 - ManBits + (e < min_normal_exp ? min_normal_exp - e : 0);
    if (shift > 63)
        return static_cast<Word>(sign);

    // Round to nearest, ties to even, on the integer significand: the dropped
    // bits above half round up, below half truncate, and exactly half rounds
    // up only when the kept part is odd.
    const uint64_t m = frac | (uint64_t{1} << 52);
    uint64_t q = m >> shift;
    const uint64_t rest = m & ((uint64_t{1} << shift) - 1);
    const uint64_t half = uint64_t{1} << (shift - 1);
    q += (rest > half) | ((rest == half) & (q & 1));

    // q is the target's significand at its exponent: at most 2^(ManBits + 1),
    // which is the carry into the next binade
    int target_exp = e < min_normal_exp ? min_normal_exp : e;
    if (q >> (ManBits + 1))
    {
        q >>= 1;
        ++target_exp;
    }
    if (target_exp > bias)
        return static_cast<Word>(sign | inf_word);
    if (q >> ManBits) // a normal, the smallest one included
        return static_cast<Word>(sign | (static_cast<uint64_t>(target_exp + bias) << ManBits) |
                                 (q & man_mask));
    return static_cast<Word>(sign | q); // a subnormal, or zero
}
