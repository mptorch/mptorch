#pragma once

#include "bit_helper.h"
#include "modes.h"
#include <type_traits>

// superfp has no subnormals: below the "supernormal" region (which repurposes
// what would otherwise be subnormal encoding space to extend the exponent
// range with implicit-mantissa-1.0 powers of two) everything flushes to
// zero. These two cutoffs classify a float32 unbiased exponent into one of
// three regions: normal (>= normal_cutoff), supernormal ([supernormal_cutoff,
// normal_cutoff)), or underflow (< supernormal_cutoff).
struct SuperfpCutoffs
{
    int normal_cutoff;
    int supernormal_cutoff;
};

CUDA_HOST_DEVICE_INLINE SuperfpCutoffs superfp_region_cutoffs(int man_bits, int exp_bits,
                                                               int normal_binades, int bias)
{
    SuperfpCutoffs c;
    c.normal_cutoff = ((1 << exp_bits) - 1 - bias) - normal_binades + 1;
    c.supernormal_cutoff = c.normal_cutoff - ((1 << exp_bits) - normal_binades) * (1 << man_bits) + 1;
    return c;
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_nearest_even(float origin_float,
                                                        int man_bits, int exp_bits, int normal_binades,
                                                        int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized{0.0f};

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;
    if (target_exp == 128)
    {
        // handle NaN/inf inputs
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_nearest_even(target);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // ties-to-even between 0 and the smallest supernormal magnitude
        // (2^supernormal_cutoff): only the topmost underflow exponent
        // (supernormal_cutoff - 1) can be at or above the halfway point;
        // anything strictly below always flushes to (signed) zero.
        uint32_t sign_bit = target & 0x80000000u;
        quantize_bits = sign_bit;
        if (target_exp == co.supernormal_cutoff - 1 && (target & 0x007FFFFFu) != 0)
        {
            // strictly above half -> round up to the smallest supernormal;
            // exactly at half (mantissa all zero) ties to even = zero.
            quantize_bits = ((uint32_t)(co.supernormal_cutoff + 127) << 23) | sign_bit;
        }
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else
    {
        quantize_bits = (man_bits > 0)
                            ? round_bitwise_nearest_even(target, man_bits)
                            : round_bitwise_nearest_even(target);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// 2^e as a float, for e in [-126, 127]. The binaryK twin of this lives in
// cast_binaryK.h; duplicated rather than shared so neither header has to
// include the other for one four-line helper.
CUDA_HOST_DEVICE_INLINE float superfp_pow2f(int e)
{
    uint32_t bits = (uint32_t)(e + 127) << 23;
    return BITS_TO_FLOAT(&bits);
}

// The same, with the exponent pinned to what binary32 can encode. Used only
// by the LEAN params below, for two separate reasons:
//   * e == -127 encodes as +0.0f exactly, which is what makes the low clamp
//     reproduce make_superfp_params' `no_underflow ? 0.0f` select rather than
//     merely approximate it -- no_underflow *is* supernormal_cutoff <= -127.
//   * outside the fast-RNE gate the cutoffs are unconstrained, and the stored
//     form answers that by zeroing its constants. Pinning both ends is the
//     derived form's version of the same promise: every constant it hands back
//     is a finite non-negative float whatever the format fields say.
CUDA_HOST_DEVICE_INLINE float superfp_pow2f_clamped(int e)
{
    e = (e < -127) ? -127 : e;
    e = (e > 127) ? 127 : e;
    return superfp_pow2f(e);
}

// The seven constants cast_superfp_rne_fast (and the RNE-gated tail of
// cast_superfp_stochastic) work from. Each is a pure function of the format
// fields above it, so a params struct can either carry them or rebuild them at
// the point of use -- see SuperfpParamsT.
struct SuperfpFastConstants
{
    float split_c;    // 2^(23 - man_bits) + 1, the Veltkamp splitting constant
    float normal_min; // 2^normal_cutoff: at or above this the normal region applies
    float super_min;  // 2^supernormal_cutoff, the smallest supernormal magnitude
    float super_half; // 2^(supernormal_cutoff - 1), the flush-to-zero midpoint
    float max_finite; // largest finite magnitude the format stores (= max_num)
    float clamp_hi;   // 2 * max_finite: keeps split_c * x from overflowing
    float ovf;        // what an out-of-range magnitude becomes (inf, or max_finite)
};

struct SuperfpNoFastConstants
{
};

// Parameter pack for the precomputed-parameter cast_superfp_* overloads -- see
// the analogous BinaryKParams/cast_binaryK_nearest_even overload in
// cast_binaryK.h. Only covers RNE, matching the GEMM core's current scope.
// Flat by design -- see dev/gemm_core_roadmap.md item 6.
//
// LEAN = false stores the seven floats; LEAN = true derives them on each read.
// Both spellings return bit-identical constants wherever fast_rne is set, which
// is the only place anything reads them -- the accessors below are the whole of
// the difference, and every caller goes through them.
//
// Which to pick is a register question, not a correctness one. A kernel whose
// params ride in the constant bank gets the stored floats for free and should
// take them; a kernel that copies a Mac out of a FormatPalette into registers
// pays one register per float per format, twice over for a SplitMac, and would
// rather spend a few ALU ops. On sm_89 that is the difference between 2 and 3
// resident blocks for the mixed superfp split-mac. See dev/gemm_perf_audit.md
// (finding G10, and R4 for the measurement that says not to do it everywhere).
template <bool LEAN>
struct SuperfpParamsT : std::conditional_t<LEAN, SuperfpNoFastConstants, SuperfpFastConstants>
{
    int man_bits; // kept raw: drives the man_bits > 0 round-formula selection
    int bias;     // kept raw: cast_superfp_odd's man_bits == 0 branch needs it directly
    // region classification (normal/supernormal/underflow)
    int normal_cutoff;
    int supernormal_cutoff;
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
    // Float-arithmetic RNE fast path -- see cast_superfp_rne_fast below.
    // Unlike BinaryKParams::fast_rne there is no second half to the gate:
    // superfp takes no SubnormalsMode, so the flag alone decides. Stored in
    // both spellings: it is a long predicate over the format fields, not a
    // constant worth rebuilding, and it gates every read below.
    bool fast_rne;

    CUDA_HOST_DEVICE_INLINE float fast_split_c() const
    {
        if constexpr (LEAN)
            return superfp_pow2f_clamped(23 - man_bits) + 1.0f;
        else
            return this->split_c;
    }
    CUDA_HOST_DEVICE_INLINE float fast_normal_min() const
    {
        if constexpr (LEAN)
            return superfp_pow2f_clamped(normal_cutoff);
        else
            return this->normal_min;
    }
    // the low clamp is the no_underflow select: see superfp_pow2f_clamped
    CUDA_HOST_DEVICE_INLINE float fast_super_min() const
    {
        if constexpr (LEAN)
            return superfp_pow2f_clamped(supernormal_cutoff);
        else
            return this->super_min;
    }
    CUDA_HOST_DEVICE_INLINE float fast_super_half() const
    {
        if constexpr (LEAN)
            return superfp_pow2f_clamped(supernormal_cutoff - 1);
        else
            return this->super_half;
    }
    CUDA_HOST_DEVICE_INLINE float fast_max_finite() const
    {
        if constexpr (LEAN)
        {
            // a local copy: BITS_TO_FLOAT casts away nothing, and max_num is
            // const through a const member function
            uint32_t bits = max_num;
            return BITS_TO_FLOAT(&bits);
        }
        else
            return this->max_finite;
    }
    CUDA_HOST_DEVICE_INLINE float fast_clamp_hi() const
    {
        if constexpr (LEAN)
            return 2.0f * fast_max_finite();
        else
            return this->clamp_hi;
    }
    CUDA_HOST_DEVICE_INLINE float fast_ovf() const
    {
        if constexpr (LEAN)
        {
            uint32_t inf_bits = 0x7F800000u;
            return (saturation_mode == SaturationMode::OVF_INF) ? BITS_TO_FLOAT(&inf_bits) : fast_max_finite();
        }
        else
            return this->ovf;
    }
};

using SuperfpParams = SuperfpParamsT<false>;
using SuperfpParamsLean = SuperfpParamsT<true>;

// LEAN defaults to false so every existing caller keeps the stored form; only
// the mixed GEMM entry points that are register-bound ask for the other.
template <bool LEAN = false>
CUDA_HOST_DEVICE_INLINE SuperfpParamsT<LEAN> make_superfp_params(int man_bits, int exp_bits, int normal_binades,
                                                                 int bias, SaturationMode saturation_mode)
{
    SuperfpParamsT<LEAN> p;
    p.man_bits = man_bits;
    p.bias = bias;

    SuperfpCutoffs cutoffs = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);
    p.normal_cutoff = cutoffs.normal_cutoff;
    p.supernormal_cutoff = cutoffs.supernormal_cutoff;

    RoundParams round_p = make_round_params(man_bits);
    p.round_bypass = round_p.bypass;
    p.round_mask = round_p.mask;
    p.round_tie = round_p.tie;
    p.round_shift = round_p.shift;

    NormalRangeParams normal_p = make_normal_range_params(exp_bits, man_bits, bias, saturation_mode);
    p.saturation_mode = normal_p.saturation_mode;
    p.max_exponent_store = normal_p.max_exponent_store;
    p.min_exponent_store = normal_p.min_exponent_store;
    p.max_num = normal_p.max_num;

    // ---- gate and constants for the float-arithmetic RNE fast path.
    // The normal region's rounding is character-for-character binaryK's, so
    // the first three conditions are G3's, unchanged. The rest replace
    // binaryK's subnormal-grid conditions with the ones the supernormal and
    // underflow regions need. See cast_superfp_rne_fast.
    int e_split = 23 - man_bits;              // fast_split_c = 2^e_split + 1
    int max_exp = p.max_exponent_store - 127; // unbiased exponent of max_num
    int min_exp = -bias + 1;                  // clip_normal_range_exponent's underflow floor
    p.fast_rne =
        // man_bits == 0 takes round_bitwise_nearest_even's structurally
        // different zero-argument overload (ties on the exponent's parity,
        // not the significand's), which the Veltkamp split does not model.
        // The upper bound is 23 rather than binaryK's 22: that one comes from
        // the magic add's headroom, which superfp has no use for, so the only
        // limit left is the split constant's own -- at man_bits == 24 its
        // exponent 23 - man_bits goes negative and the identity breaks (the
        // sweep in dev/benchmarks/gemm_cast_superfp_arith.cu shows 23 clean
        // and 24 mismatching on ~1.7e7 inputs per configuration).
        man_bits >= 1 && man_bits <= 23 &&
        // SAT_PROPAGATE leaves a value whose exponent is exactly
        // max_exponent_store unclamped even when its significand exceeds
        // max_num, which a single saturating compare cannot express.
        saturation_mode != SaturationMode::SAT_PROPAGATE &&
        // 2 * max_finite (the clamp) and fast_split_c * that product must both
        // stay finite.
        max_exp <= 126 && max_exp + e_split + 3 <= 128 &&
        // 2^normal_cutoff is a region boundary the selects compare against, so
        // it has to be a normal binary32 value.
        p.normal_cutoff >= -125 && p.normal_cutoff <= 127 &&
        // So are 2^supernormal_cutoff and its midpoint -- unless the whole
        // underflow region sits below binary32, which is the common case for
        // man_bits >= 8 (the cutoff is normal_cutoff minus a multiple of
        // 2^man_bits, so it falls away fast). supernormal_cutoff <= -127 puts
        // every representable input, float32 subnormals and zero included, at
        // or above it, so nothing ever underflows and the two constants are
        // set to zero below rather than to unrepresentable powers of two.
        // -126 is the one value excluded by both arms: there target_exp ==
        // -127 is exactly supernormal_cutoff - 1, so the integer path rounds
        // *every* nonzero float32 subnormal up to 2^-126 while a compare
        // against the midpoint 2^-127 would keep only half of them.
        ((p.supernormal_cutoff >= -125 && p.supernormal_cutoff <= 127) ||
         p.supernormal_cutoff <= -127) &&
        // The regions have to be ordered as the integer path's if/else chain
        // reads them. normal_binades == 1 << exp_bits puts supernormal_cutoff
        // one above normal_cutoff, and then an exponent equal to normal_cutoff
        // satisfies the *underflow* test and flushes to zero, while a single
        // `ax >= 2^normal_cutoff` compare would send it to the normal region.
        p.supernormal_cutoff <= p.normal_cutoff &&
        // In the normal region rounding can only raise the exponent, so
        // clip_normal_range_exponent's underflow branch is unreachable and
        // saturation is the compare's only job -- provided the region starts
        // at or above that branch's floor.
        p.normal_cutoff >= min_exp;

    // The LEAN form has nowhere to put these: its accessors rebuild each one
    // from the fields already set above, which is the whole of the difference
    // between the two spellings. See SuperfpParamsT.
    if constexpr (!LEAN)
    {
        bool no_underflow = p.supernormal_cutoff <= -127;
        p.split_c = superfp_pow2f(e_split) + 1.0f;
        p.normal_min = superfp_pow2f(p.normal_cutoff);
        // zeros when the underflow region is out of binary32's reach: the guard
        // `ax > 0` then admits every nonzero input to the supernormal result, and
        // fmaxf against 0 leaves it alone -- which is what "nothing underflows"
        // means. Zero itself still takes the `else` arm and stays (signed) zero.
        p.super_min = no_underflow ? 0.0f : superfp_pow2f(p.supernormal_cutoff);
        p.super_half = no_underflow ? 0.0f : superfp_pow2f(p.supernormal_cutoff - 1);
        p.max_finite = BITS_TO_FLOAT(&p.max_num);
        p.clamp_hi = 2.0f * p.max_finite;
        uint32_t inf_bits = 0x7F800000u;
        p.ovf = (saturation_mode == SaturationMode::OVF_INF) ? BITS_TO_FLOAT(&inf_bits) : p.max_finite;
        if (!p.fast_rne)
        {
            // keep the unused constants finite so a disabled gate can never
            // produce a signalling value if the path is ever entered by mistake
            p.split_c = 1.0f;
            p.normal_min = 0.0f;
            p.super_min = 0.0f;
            p.super_half = 0.0f;
            p.max_finite = 0.0f;
            p.clamp_hi = 0.0f;
            p.ovf = 0.0f;
        }
    }
    return p;
}

// Float-arithmetic replacement for cast_superfp_nearest_even's bit-twiddling
// body, for the formats make_superfp_params' gate admits. The superfp twin of
// cast_binaryK_rne_fast (see the long note there), and it borrows that
// function's normal-region half unchanged: superfp's normal branch is
// character-for-character binaryK's, so a Veltkamp split rounds it to
// man_bits + 1 significand bits and one compare saturates it.
//
// What differs is everything below normal_cutoff. binaryK has a subnormal
// grid of fixed absolute spacing, which a magic-constant add lands on in two
// float operations. superfp instead has the supernormal region, which rounds
// to the nearest *power of two* with ties broken to the even exponent -- a
// rule about the exponent field, not the significand, so no sequence of float
// operations reproduces it (a 1-bit Veltkamp split ties the other way: it
// sends 3.0 to 4.0 where superfp sends it to 2.0). That region therefore
// stays bitwise, but it is the cheap half of the original: six branchless
// integer operations, against the ~17 data-dependent branches this replaces.
//
// One float compare places `ax` in a region, and unlike the binaryK twin the
// two arms stay behind a real branch rather than being computed and selected
// between. That is a measurement, not a preference: the region below
// 2^normal_cutoff -- where ordinary data lives, since normal_binades is
// usually 1 or 2 -- is *already* five operations in the integer path, so
// computing the normal arm as well to select it away costs 7% in the GEMM
// (dev/benchmarks/gemm_cast_superfp_arith.cu measures both forms). Warps are
// close to region-uniform for real operand distributions, so the branch is
// nearly free where the extra arm is not.
//
//   ax >= 2^normal_cutoff             normal      -> Veltkamp split + saturate
//   2^(supernormal_cutoff - 1) < ax   supernormal -> nearest power of two,
//                                                    lifted to 2^supernormal_cutoff
//   otherwise                         underflow   -> signed zero
//
// The second is stated as a strict compare against the *midpoint* rather than
// against 2^supernormal_cutoff because that is what the integer path's
// underflow branch does: at the topmost underflow exponent any nonzero
// significand rounds up, and exactly the midpoint ties to even, which is zero.
//
// inf and NaN pass through unchanged (the integer path's target_exp == 128
// branch). Rather than a branch of their own they ride the arms' final
// select: `inf >= fast_normal_min` is true so inf takes the normal arm, and
// every compare against NaN is false so NaN takes the supernormal one.
//
// CUDA only, for the same reason as the binaryK twin: the Veltkamp identity
// needs `t` separately rounded, which only the _rn intrinsics guarantee.
#if defined(__CUDA_ARCH__)
template <bool LEAN>
CUDA_HOST_DEVICE_INLINE float cast_superfp_rne_fast(float origin_float, const SuperfpParamsT<LEAN> &p)
{
    float ax = fabsf(origin_float);
    const float inf = __int_as_float(0x7F800000);

    if (ax >= p.fast_normal_min())
    {
        // normal region: clamp before rounding so the Veltkamp product cannot
        // overflow; anything at or above the clamp is out of range either way.
        float xc = fminf(ax, p.fast_clamp_hi());
        float t = __fmul_rn(p.fast_split_c(), xc);
        float nrm = __fsub_rn(t, __fsub_rn(t, xc));
        if (nrm > p.fast_max_finite())
            nrm = p.fast_ovf();
        return copysignf((ax < inf) ? nrm : ax, origin_float);
    }

    // supernormal region: round |x| to the nearest power of two, ties to the
    // even exponent -- round_bitwise_nearest_even's zero-argument overload,
    // inlined here on a sign-cleared word.
    uint32_t ab = FLOAT_TO_BITS(&ax);
    uint32_t q = (ab + 0x00400000u) & 0xFF800000u;
    uint32_t is_tie = (ab & 0x007FFFFFu) == 0x00400000u;
    q -= ((is_tie << 23) & ~q);
    float spn = BITS_TO_FLOAT(&q);
    // fmaxf covers the topmost underflow binade, whose nearest power of two is
    // below the region but which rounds up into it.
    spn = (ax > p.fast_super_half()) ? fmaxf(spn, p.fast_super_min()) : 0.0f;
    // copysignf carries the sign -- and a NaN's payload -- back onto the
    // magnitude, so the NaN case returns the input's exact bit pattern.
    return copysignf((ax < inf) ? spn : ax, origin_float);
}
#endif

template <bool LEAN>
CUDA_HOST_DEVICE_INLINE float cast_superfp_nearest_even(float origin_float, bool is_signed,
                                                         const SuperfpParamsT<LEAN> &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

#if defined(__CUDA_ARCH__)
    // Warp-uniform in the single-format kernels and per-slot uniform in the
    // mixed ones, so the test costs a predicated compare and the integer body
    // below is jumped over, not fetched. See cast_binaryK_nearest_even.
    if (p.fast_rne)
        return cast_superfp_rne_fast(origin_float, p);
#endif

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized{0.0f};

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;
    if (target_exp == 128)
    {
        // handle NaN/inf inputs
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_nearest_even(target);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // ties-to-even between 0 and the smallest supernormal magnitude
        // (2^supernormal_cutoff): only the topmost underflow exponent
        // (supernormal_cutoff - 1) can be at or above the halfway point;
        // anything strictly below always flushes to (signed) zero.
        uint32_t sign_bit = target & 0x80000000u;
        quantize_bits = sign_bit;
        if (target_exp == p.supernormal_cutoff - 1 && (target & 0x007FFFFFu) != 0)
        {
            // strictly above half -> round up to the smallest supernormal;
            // exactly at half (mantissa all zero) ties to even = zero.
            quantize_bits = ((uint32_t)(p.supernormal_cutoff + 127) << 23) | sign_bit;
        }
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
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

CUDA_HOST_DEVICE_INLINE float cast_superfp_nearest_away(float origin_float,
                                                         int man_bits, int exp_bits, int normal_binades,
                                                         int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_nearest_away(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // ties-to-away: bump to the smallest supernormal magnitude as soon
        // as the halfway point is reached (unlike nearest-even, the exact
        // tie also rounds away from zero).
        uint32_t sign_bit = target & 0x80000000u;
        if (target_exp >= co.supernormal_cutoff - 1)
        {
            quantize_bits = ((uint32_t)(co.supernormal_cutoff + 127) << 23) | sign_bit;
        }
        else
        {
            quantize_bits = sign_bit;
        }
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_nearest_away(target, man_bits);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_nearest_away above.
template <bool LEAN>
CUDA_HOST_DEVICE_INLINE float cast_superfp_nearest_away(float origin_float, bool is_signed,
                                                         const SuperfpParamsT<LEAN> &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_nearest_away(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        uint32_t sign_bit = target & 0x80000000u;
        if (target_exp >= p.supernormal_cutoff - 1)
        {
            quantize_bits = ((uint32_t)(p.supernormal_cutoff + 127) << 23) | sign_bit;
        }
        else
        {
            quantize_bits = sign_bit;
        }
        quantized = BITS_TO_FLOAT(&quantize_bits);
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
CUDA_HOST_DEVICE_INLINE float cast_superfp_odd(float origin_float,
                                               int man_bits, int exp_bits, int normal_binades,
                                               int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        // supernormal codepoints have no explicit mantissa bits, so
        // "oddness" refers to the parity of the supernormal grid index
        // itself (1 for the smallest magnitude, 2 for the next, ...), a
        // different reference frame than the float32 exponent field's
        // parity used by nearest-even, and different again from the
        // *target format's own* biased exponent parity used in the normal-
        // region man_bits==0 case below.
        int stored_index = target_exp - co.supernormal_cutoff + 1;
        uint32_t mask = 0x007FFFFFu;
        bool sticky = (target & mask) != 0;
        bool already_odd = (stored_index & 1) != 0;
        quantize_bits = target & ~mask;
        if (sticky && !already_odd)
            quantize_bits += (1u << 23);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // round to odd never flushes a nonzero input to zero; the smallest
        // supernormal codepoint's grid index is always 1 (odd), so a bare
        // bump (no sticky-driven adjustment) always yields an odd result.
        if ((target & 0x7FFFFFFFu) == 0)
        {
            quantized = origin_float;
        }
        else
        {
            uint32_t sign_bit = target & 0x80000000u;
            quantize_bits = ((uint32_t)(co.supernormal_cutoff + 127) << 23) | sign_bit;
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }
    else
    {
        if (man_bits > 0)
        {
            quantize_bits = round_bitwise_odd(target, man_bits);
        }
        else
        {
            // no explicit significand bits: oddness refers to the parity of
            // the target format's own biased exponent (target_exp + bias),
            // mirroring cast_binaryK_odd's man_bits==0 handling.
            uint32_t mask = 0x007FFFFFu;
            bool sticky = (target & mask) != 0;
            bool already_odd = ((target_exp + bias) & 1) != 0;
            quantize_bits = target & ~mask;
            if (sticky && !already_odd)
                quantize_bits += (1u << 23);
        }
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_odd above.
template <bool LEAN>
CUDA_HOST_DEVICE_INLINE float cast_superfp_odd(float origin_float, bool is_signed, const SuperfpParamsT<LEAN> &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        int stored_index = target_exp - p.supernormal_cutoff + 1;
        uint32_t mask = 0x007FFFFFu;
        bool sticky = (target & mask) != 0;
        bool already_odd = (stored_index & 1) != 0;
        quantize_bits = target & ~mask;
        if (sticky && !already_odd)
            quantize_bits += (1u << 23);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        if ((target & 0x7FFFFFFFu) == 0)
        {
            quantized = origin_float;
        }
        else
        {
            uint32_t sign_bit = target & 0x80000000u;
            quantize_bits = ((uint32_t)(p.supernormal_cutoff + 127) << 23) | sign_bit;
            quantized = BITS_TO_FLOAT(&quantize_bits);
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
            uint32_t mask = 0x007FFFFFu;
            bool sticky = (target & mask) != 0;
            bool already_odd = ((target_exp + p.bias) & 1) != 0;
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

// unsigned helper for cast_superfp_up/cast_superfp_down: assumes
// origin_float >= 0.
CUDA_HOST_DEVICE_INLINE float cast_superfp_absolute_up(float origin_float,
                                                        int man_bits, int exp_bits, int normal_binades,
                                                        int bias, SaturationMode saturation_mode)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_up(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // rounding up (toward +inf) never flushes a nonzero magnitude to
        // zero: any nonzero underflowing input bumps up to the smallest
        // supernormal magnitude.
        if ((target & 0x7FFFFFFFu) == 0)
        {
            quantized = origin_float;
        }
        else
        {
            quantize_bits = (uint32_t)(co.supernormal_cutoff + 127) << 23;
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }
    else
    {
        quantize_bits = round_bitwise_up(target, man_bits);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_absolute_up above.
template <bool LEAN>
CUDA_HOST_DEVICE_INLINE float cast_superfp_absolute_up(float origin_float, const SuperfpParamsT<LEAN> &p)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_up(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        if ((target & 0x7FFFFFFFu) == 0)
        {
            quantized = origin_float;
        }
        else
        {
            quantize_bits = (uint32_t)(p.supernormal_cutoff + 127) << 23;
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
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

CUDA_HOST_DEVICE_INLINE float cast_superfp_absolute_down(float origin_float,
                                                          int man_bits, int exp_bits, int normal_binades,
                                                          int bias, SaturationMode saturation_mode)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_down(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // rounding down (toward -inf, on a nonnegative magnitude) always
        // flushes to zero: there is no partial precision in this region to
        // create a boundary case worth preserving.
        quantize_bits = target & 0x80000000u;
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_down(target, man_bits);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_absolute_down above.
template <bool LEAN>
CUDA_HOST_DEVICE_INLINE float cast_superfp_absolute_down(float origin_float, const SuperfpParamsT<LEAN> &p)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_down(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        quantize_bits = target & 0x80000000u;
        quantized = BITS_TO_FLOAT(&quantize_bits);
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

CUDA_HOST_DEVICE_INLINE float cast_superfp_up(float origin_float,
                                              int man_bits, int exp_bits, int normal_binades,
                                              int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    // NaN comparisons are always false, so `origin_float >= 0` alone would
    // route every NaN through the sign-flip dispatch below; unary negation
    // of a NaN is not guaranteed to preserve its exact bit pattern on every
    // backend (observed on CUDA, where -(-NaN) can canonicalize the
    // payload), so NaN/Inf must be passed through directly instead.
    uint32_t target = FLOAT_TO_BITS(&origin_float);
    if ((int)((target >> 23) & 0xFF) - 127 == 128)
        return origin_float;

    if (origin_float >= 0)
        return cast_superfp_absolute_up(origin_float, man_bits, exp_bits, normal_binades, bias,
                                        saturation_mode);
    else
        return -cast_superfp_absolute_down(-origin_float, man_bits, exp_bits, normal_binades, bias,
                                           saturation_mode);
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_down(float origin_float,
                                                int man_bits, int exp_bits, int normal_binades,
                                                int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    // see cast_superfp_up for why NaN/Inf must be special-cased before the
    // sign-flip dispatch rather than relying on it to preserve the payload.
    uint32_t target = FLOAT_TO_BITS(&origin_float);
    if ((int)((target >> 23) & 0xFF) - 127 == 128)
        return origin_float;

    if (origin_float >= 0)
        return cast_superfp_absolute_down(origin_float, man_bits, exp_bits, normal_binades, bias,
                                          saturation_mode);
    else
        return -cast_superfp_absolute_up(-origin_float, man_bits, exp_bits, normal_binades, bias,
                                         saturation_mode);
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_zero(float origin_float,
                                                int man_bits, int exp_bits, int normal_binades,
                                                int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float >= 0.0f)
        return cast_superfp_down(origin_float, man_bits, exp_bits, normal_binades, bias, is_signed,
                                 saturation_mode);
    else
        return cast_superfp_up(origin_float, man_bits, exp_bits, normal_binades, bias, is_signed,
                               saturation_mode);
}

// Precomputed-parameter overloads of cast_superfp_up/_down/_zero above.
template <bool LEAN>
CUDA_HOST_DEVICE_INLINE float cast_superfp_up(float origin_float, bool is_signed, const SuperfpParamsT<LEAN> &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    // see the *_bits-taking cast_superfp_up above for why NaN/Inf must be
    // special-cased before the sign-flip dispatch.
    uint32_t target = FLOAT_TO_BITS(&origin_float);
    if ((int)((target >> 23) & 0xFF) - 127 == 128)
        return origin_float;

    if (origin_float >= 0)
        return cast_superfp_absolute_up(origin_float, p);
    else
        return -cast_superfp_absolute_down(-origin_float, p);
}

template <bool LEAN>
CUDA_HOST_DEVICE_INLINE float cast_superfp_down(float origin_float, bool is_signed, const SuperfpParamsT<LEAN> &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target = FLOAT_TO_BITS(&origin_float);
    if ((int)((target >> 23) & 0xFF) - 127 == 128)
        return origin_float;

    if (origin_float >= 0)
        return cast_superfp_absolute_down(origin_float, p);
    else
        return -cast_superfp_absolute_up(-origin_float, p);
}

template <bool LEAN>
CUDA_HOST_DEVICE_INLINE float cast_superfp_zero(float origin_float, bool is_signed, const SuperfpParamsT<LEAN> &p)
{
    if (origin_float >= 0.0f)
        return cast_superfp_down(origin_float, is_signed, p);
    else
        return cast_superfp_up(origin_float, is_signed, p);
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_stochastic(float origin_float, uint32_t rand_prob,
                                                       int rand_bits, int man_bits, int exp_bits,
                                                       int normal_binades, int bias, bool is_signed,
                                                       SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    uint32_t rand_bits_raw = rand_prob & 0x007FFFFFu;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        // the supernormal grid has zero explicit precision (man_bits acts
        // as 0 here) regardless of the format's actual man_bits, so the
        // full 23 bits of randomness are used rather than the narrower
        // rand_bits-relative mask used in the normal region below.
        quantize_bits = round_bitwise_stochastic(target, rand_bits_raw, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // shift the value up by the smallest supernormal magnitude so it
        // lands in [smallest, 2*smallest), stochastically round it there
        // at 0-bit (whole power-of-two) precision, then shift back --
        // mirrors cast_binaryK_stochastic's subnormal shift trick.
        uint32_t sign_bit = target & 0x80000000u;
        uint32_t shift_bits = ((uint32_t)(co.supernormal_cutoff + 127) << 23) | sign_bit;
        float shift_float = BITS_TO_FLOAT(&shift_bits);
        float val = origin_float + shift_float;
        uint32_t target2 = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target2, rand_bits_raw, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else
    {
        uint32_t rand_prob_man = rand_bits_raw & ~((1u << (23 - man_bits - rand_bits)) - 1u);
        quantize_bits = round_bitwise_stochastic(target, rand_prob_man, man_bits);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_stochastic. Like
// cast_binaryK_stochastic's precomputed overload, rand_prob/rand_bits stay
// per-call arguments -- only the region-cutoff and round-to-nearest-even
// constants precompute away.
template <bool LEAN>
CUDA_HOST_DEVICE_INLINE float cast_superfp_stochastic(float origin_float, uint32_t rand_prob, int rand_bits,
                                                       bool is_signed, const SuperfpParamsT<LEAN> &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    uint32_t rand_bits_raw = rand_prob & 0x007FFFFFu;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_stochastic(target, rand_bits_raw, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        uint32_t sign_bit = target & 0x80000000u;
        uint32_t shift_bits = ((uint32_t)(p.supernormal_cutoff + 127) << 23) | sign_bit;
        float shift_float = BITS_TO_FLOAT(&shift_bits);
        float val = origin_float + shift_float;
        uint32_t target2 = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target2, rand_bits_raw, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else
    {
        uint32_t rand_prob_man = rand_bits_raw & ~((1u << (23 - p.man_bits - rand_bits)) - 1u);
        quantize_bits = round_bitwise_stochastic(target, rand_prob_man, p.man_bits);
#if defined(__CUDA_ARCH__)
        // The same compare-instead-of-clip as cast_binaryK_stochastic's normal
        // arm -- see the note there for why one magnitude test reproduces
        // clip_normal_range_exponent. The clip's underflow arm is unreachable
        // for the same reason in different words: this arm is entered only at
        // or above normal_cutoff, and the gate's `normal_cutoff >= min_exp`
        // puts that at or above the arm's floor.
        if (p.fast_rne)
        {
            float y = BITS_TO_FLOAT(&quantize_bits);
            return (fabsf(y) > p.fast_max_finite()) ? copysignf(p.fast_ovf(), origin_float) : y;
        }
#endif
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}
