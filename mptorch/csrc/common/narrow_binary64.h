#pragma once

// Rounds a binary64 value to a narrower IEEE binary format, once.
//
// A result tensor narrower than its carrier stores what the carrier
// computed, and that store is a rounding of its own. Under
// `carrier=torch.float64` the carrier's result is float64, and the store is
// a conversion from float64 to float32, float16 or bfloat16. torch's own
// conversion to bfloat16 goes through float32 and so rounds twice: a value
// just past a midpoint of the bfloat16 grid is first rounded onto the
// midpoint and then to even, so 1 + 2^-8 + 2^-30 becomes bfloat16 1.0 instead
// of 1 + 2^-7. Its conversion to float16 does the same on some platforms,
// where 65519.999999 becomes float16 infinity instead of 65504; an arm64 CPU
// converts float64 to float16 in one step and gets it right. This function is
// the conversion done once, on the word, on every platform, for the
// narrow_float64 op that `_narrowed` (mptorch/quant/ops/_checks.py) calls.
//
// `ExpBits` and `ManBits` are the target format's field widths and `Word`
// its storage type: (5, 10) is float16, (8, 7) bfloat16 and (8, 23) float32.
// The result is IEEE 754's convertFormat under roundTiesToEven: a finite
// value rounds to the nearest value of the target, a tie to the one with an
// even significand; a value that rounds past the largest finite value becomes
// an infinity of its sign, and one at or below half the smallest subnormal a
// zero of its sign; an infinity stays an infinity; and a NaN stays a NaN,
// quieted and keeping the top bits of its payload, as the hardware
// conversions do. For (8, 23) that is exactly `static_cast<float>(double)`,
// which tests/test_float64_carrier.py checks, along with a Fraction
// reference at every tie.
//
// It is integer arithmetic on the words rather than a float division and a
// nearbyint, so nothing depends on the floating-point environment, on a
// contraction flag or on how the device rounds a conversion, and the host
// and device objects compute the same bits by construction.

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
        // an infinity, or a NaN, which keeps the top bits of its payload and
        // gets the quiet bit set
        const uint64_t nan = (frac >> (52 - ManBits)) | (uint64_t{1} << (ManBits - 1));
        return static_cast<Word>(sign | inf_word | (frac ? nan : 0));
    }

    // the exponent of x's leading bit, for a normal x
    const int e = field - 1023;
    if (e > bias)
        return static_cast<Word>(sign | inf_word);

    // How many of the 53 significand bits fall below the target's last bit:
    // all but the ManBits after the leading bit, and more below the target's
    // smallest normal, where the target's grid spacing stops shrinking. A
    // shift past 63 leaves nothing that reaches half the smallest subnormal,
    // and the result is a zero. A binary64 subnormal (field 0) always takes
    // that exit, so the implicit leading bit added below always belongs to a
    // normal value.
    const int shift = 52 - ManBits + (e < min_normal_exp ? min_normal_exp - e : 0);
    if (shift > 63)
        return static_cast<Word>(sign);

    // Round to nearest, ties to even, on the integer significand: dropped
    // bits above half round up, below half truncate, and exactly half rounds
    // up only when the kept part is odd.
    const uint64_t m = frac | (uint64_t{1} << 52);
    uint64_t q = m >> shift;
    const uint64_t rest = m & ((uint64_t{1} << shift) - 1);
    const uint64_t half = uint64_t{1} << (shift - 1);
    q += (rest > half) | ((rest == half) & (q & 1));

    // q is the target's significand, implicit bit included, at exponent
    // target_exp: at most 2^(ManBits + 1), when the rounding carried into the
    // next binade
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
