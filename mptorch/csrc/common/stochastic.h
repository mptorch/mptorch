#pragma once

// IEEE P3109's three stochastic rounding modes, in integer arithmetic on the
// carrier's word. Let |x| lie between two neighbouring values of the format,
// with eta the fraction of the way from the lower one to the upper one, let N
// be the format's prng_bits, and let R be a random integer in [0, 2^N). Each
// mode rounds the magnitude away from zero, to the upper neighbour, if and
// only if (arXiv:2606.04028, Fig. 2):
//
//   StochasticA   floor(eta * 2^N) + R >= 2^N                 RoundMode::SR
//   StochasticB   floor(eta * 2^(N+1)) + 2R + 1 >= 2^(N+1)    RoundMode::SR_B
//   StochasticC   RNITE(eta * 2^N) + R >= 2^N                 RoundMode::SR_C
//
// RNITE rounds to nearest, ties to even. With X = eta * 2^N and k = 2^N - R,
// the conditions read: A rounds away iff X >= k; B iff X >= k - 1/2; C iff
// X > k - 1/2, or X = k - 1/2 and k is even (for N >= 1 that is when R is
// even; at N = 0, k = 1 and the tie never rounds away). So with N = 0, A is
// round toward zero, B is round to nearest with ties away from zero, and C
// is round to nearest with ties toward zero. Where |x| has no bits below the
// N-bit field, X is an integer and B and C agree with A.
//
// Every cast finds the two neighbours on the word, so nothing here rounds in
// the carrier first, and no float operation touches the input (which on an
// Apple GPU would flush a binary32 subnormal to zero). One of two forms
// makes the decision:
//
//   * The word form, for a value on a grid the word can represent (a
//     format's normal binades, binaryK's subnormals, superfp's supernormal
//     binades): add the draw just below the kept bits, and a constant below
//     the draw, then clear the bits below the kept ones. It is StochasticA's
//     add-and-truncate plus two integer operations, with the constants built
//     once per format (StochasticT, make_stochastic).
//   * The bit test, sr_away, which decides from floor(2X) and whether 2X has
//     a fractional part: for the regions under a format's smallest step,
//     where the value is shifted down rather than added to. sr_away_scaled
//     is its cross-multiplied form, for a smallest value that is not a power
//     of two (the EXTENDED_NORMALS floor).
//
// Under stochastic rounding a format's prng_bits field carries a spec: N in
// the low byte, and the variant (0 for A, 1 for B, 2 for C) above it. Only
// the entry points and the drivers write a spec (they turn RoundMode::SR_B
// and SR_C into RoundMode::SR plus the variant), so every kernel has one
// stochastic instantiation and the deterministic ones never see a spec.

#include "bit_helper.h"
#include "modes.h"

// The one arm no common format reaches (binaryK's cross-multiplied
// EXTENDED_NORMALS floor) is compiled out of line on the host and CUDA, so
// that its code stays out of the GEMMs' unrolled K-loop. It takes its
// operands by value, since a reference into a policy would force the policy
// onto the device's stack. Every other arm is inline: narrow formats reach
// them in most warps, and a call costs the loop more than inline code does
// (cast_binaryK_stochastic's note has the measurements). Metal folds the
// format's literals and has no such attribute, so there it is inline too.
#if defined(__METAL_VERSION__)
#define MPTORCH_SR_COLD inline
#elif defined(__CUDACC__)
#define MPTORCH_SR_COLD __host__ __device__ __noinline__
#else
#define MPTORCH_SR_COLD __attribute__((noinline, cold))
#endif

// The spec a format's prng_bits field carries under stochastic rounding, and
// its two parts.
CUDA_HOST_DEVICE_INLINE int sr_spec(int bits, int variant) { return bits | (variant << 8); }
CUDA_HOST_DEVICE_INLINE int sr_spec_bits(int spec) { return spec & 0xFF; }
CUDA_HOST_DEVICE_INLINE int sr_spec_variant(int spec) { return (spec >> 8) & 3; }

// The variant a stochastic RoundMode names: 0 for SR (StochasticA), 1 for
// SR_B, 2 for SR_C, and -1 for a deterministic mode.
CUDA_HOST_DEVICE_INLINE int sr_variant_of(RoundMode rm)
{
    return rm == RoundMode::SR ? 0 : rm == RoundMode::SR_B ? 1 : rm == RoundMode::SR_C ? 2 : -1;
}

// The two constants the word form adds and subtracts below a draw whose
// lowest bit is bit s of the word. B adds half a unit of that bit. C adds
// half a unit and takes one off when R is odd (on a tie, k = 2^N - R is then
// odd and the value must stay down), and at N = 0 it adds half a unit less
// one (a tie goes toward zero). A needs neither, and neither applies when
// s = 0.
template <class W>
CUDA_HOST_DEVICE_INLINE void sr_word_constants(int s, int n, int variant, MPTORCH_THREAD W &or_v,
                                               MPTORCH_THREAD W &tie)
{
    or_v = W(0);
    tie = W(0);
    if (s >= 1 && variant != 0)
    {
        // Not named `half`, which is a type in MSL, and MSL compiles this
        // header too.
        const W half_unit = W(1) << (s - 1);
        if (variant == 1)
            or_v = half_unit;
        else if (n >= 1)
        {
            or_v = half_unit;
            tie = W(1);
        }
        else
            or_v = half_unit - W(1);
    }
}

// A stochastic rounding's constants for one format, built once: where the
// draw's field is (the N bits of the draw word directly below the format's
// significand, bits [MAN_BITS - man_bits - N, MAN_BITS - man_bits)), and the
// word form's constants for that field and for a superfp supernormal binade
// (whose grid keeps no mantissa bit), each with the mask of the bits it
// keeps, so that the hot path shifts nothing.
template <class T>
struct StochasticT
{
    using word_t = typename FloatTraits<T>::word_t;
    int bits;    // N
    int variant; // 0 A, 1 B, 2 C
    int shift;   // the field's lowest bit: MAN_BITS - man_bits - N
    word_t field; // 2^N - 1
    word_t draw;  // the field at its place in the word: field << shift
    word_t or_v;
    word_t tie;   // C's tie, 0 or 1: R's lowest bit, bit `shift` of the draw
    word_t keep;  // the bits the word form keeps: from shift + N up
    word_t or_sup; // the constants for a draw at bit MAN_BITS - N
    word_t tie_sup;
    word_t keep_sup;
    // The family's builder fills in the rest (make_stochastic_binaryK,
    // make_stochastic_superfp): the word bounds the hot path tests, so that
    // a value inside the format's normal range costs one compare. `wmask` is
    // ABS_MASK for a signed format and all ones for an unsigned one, so that
    // an unsigned format's negative inputs fall outside every range; each
    // range is [lo, lo + span) of the masked word.
    word_t wmask;
    word_t lo;
    word_t span;
    word_t lo_sup; // superfp's supernormal binades
    word_t span_sup;
};

// The word of the smallest carrier value whose exponent is at least e (2^e
// itself when e is a normal exponent). Comparing a magnitude's word against
// it answers the exponent test `target_exp >= e`. It is clamped to the
// exponent field's range, so that a bound below the range admits every word.
template <class T>
CUDA_HOST_DEVICE_INLINE typename FloatTraits<T>::word_t sr_field_word(int e)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    const int f = e + F::BIAS;
    return f <= 0 ? word_t(0) : f >= (int)F::FIELD_MASK ? (word_t)F::FIELD_MASK << F::MAN_BITS
                                                          : (word_t)f << F::MAN_BITS;
}

// The constants every family shares, from the format's man_bits and the
// spec. The range bounds are left empty for the family's builder.
template <class T>
CUDA_HOST_DEVICE_INLINE StochasticT<T> make_stochastic(int man_bits, int spec)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    StochasticT<T> st;
    st.bits = sr_spec_bits(spec);
    st.variant = sr_spec_variant(spec);
    st.shift = F::MAN_BITS - man_bits - st.bits;
    st.field = (word_t(1) << st.bits) - word_t(1);
    st.draw = st.field << st.shift;
    sr_word_constants<word_t>(st.shift, st.bits, st.variant, st.or_v, st.tie);
    st.keep = ~((word_t(1) << (st.shift + st.bits)) - word_t(1));
    sr_word_constants<word_t>(F::MAN_BITS - st.bits, st.bits, st.variant, st.or_sup, st.tie_sup);
    st.keep_sup = ~((word_t(1) << F::MAN_BITS) - word_t(1));
    st.wmask = F::ABS_MASK;
    st.lo = word_t(0);
    st.span = word_t(0);
    st.lo_sup = word_t(0);
    st.span_sup = word_t(0);
    return st;
}

// 2^e as a carrier word: a normal, or a subnormal below the normals, or zero
// below the subnormals (a step no carrier holds, which only a format that
// warns about its range can have).
template <class T>
CUDA_HOST_DEVICE_INLINE typename FloatTraits<T>::word_t sr_pow2_word(int e)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    if (e >= F::MIN_NORMAL_EXP)
        return (word_t)(e + F::BIAS) << F::MAN_BITS;
    if (e >= F::MIN_NORMAL_EXP - F::MAN_BITS)
        return word_t(1) << (e - (F::MIN_NORMAL_EXP - F::MAN_BITS));
    return word_t(0);
}

// R: the draw's field, as an integer in [0, 2^N).
template <class T>
CUDA_HOST_DEVICE_INLINE typename FloatTraits<T>::word_t
sr_draw(typename FloatTraits<T>::word_t rand_prob, const MPTORCH_THREAD StochasticT<T> &st)
{
    return ((rand_prob & FloatTraits<T>::MAN_MASK) >> st.shift) & st.field;
}

// The word form: w rounded to the grid that keeps its bits from bit s + n
// up, with R added at bit s and the variant's constants below it. Requires
// s >= 0 and s + n less than the word's width.
template <class W>
CUDA_HOST_DEVICE_INLINE W sr_round_word(W w, W r, int s, int n, W or_v, W tie)
{
    const W t = w + (r << s) + or_v - (r & tie);
    return t & ~((W(1) << (s + n)) - W(1));
}

// The word form at the format's own field, from the draw word as it is: R
// stays where the draw holds it, C's tie bit is the draw's bit `shift`, and
// the mask is the precomputed one. It gives sr_round_word's value with
// nothing derived per call.
template <class T>
CUDA_HOST_DEVICE_INLINE typename FloatTraits<T>::word_t
sr_round_in_place(typename FloatTraits<T>::word_t w, typename FloatTraits<T>::word_t rand_prob,
                  const MPTORCH_THREAD StochasticT<T> &st)
{
    using word_t = typename FloatTraits<T>::word_t;
    const word_t t = w + (rand_prob & st.draw) + st.or_v - ((rand_prob >> st.shift) & st.tie);
    return t & st.keep;
}

// The bit test: given top2 = floor(2X), and `sticky`, whether 2X has a
// fractional part, decides whether the magnitude rounds away. A needs
// top2 >= 2k; B needs top2 >= 2k - 1; C needs top2 >= 2k - 1, except when
// 2X = 2k - 1 exactly and k is odd.
template <class W>
CUDA_HOST_DEVICE_INLINE bool sr_away(W top2, bool sticky, W r, int n, int variant)
{
    const bool k_even = n >= 1 && (r & W(1)) == W(0);
    const W d = variant == 0 ? W(0) : variant == 1 ? W(1) : ((sticky || k_even) ? W(1) : W(0));
    return top2 + (r << 1) + d >= (W(1) << (n + 1));
}

// The bit test for a region under a format's smallest step, inline and
// without branches. There X = m * 2^(N - below), and sh = below - N - 1 is
// at least 0, because a magnitude under the step has more bits below the
// step than the N-bit field holds; so floor(2X) = m >> sh. The shift is
// clamped to the word's width, where it leaves nothing, since m <
// 2^(MAN_BITS + 1); `sticky` is whether the shift dropped a set bit. It
// gives sr_top2 and sr_away's answer, for the arms where sr_top2's other
// cases cannot occur.
template <class W>
CUDA_HOST_DEVICE_INLINE bool sr_away_below(W m, int sh, W r, int n, int variant, int word_bits)
{
    const int c = sh < word_bits ? sh : word_bits - 1;
    const W top2 = m >> c;
    const bool sticky = (top2 << c) != m;
    const bool k_even = n >= 1 && (r & W(1)) == W(0);
    const W d = variant == 0 ? W(0) : (variant == 1 || sticky || k_even) ? W(1) : W(0);
    return top2 + (r << 1) + d >= (W(1) << (n + 1));
}

// The same, where sh can be negative; a negative sh rounds away. Only a
// carrier subnormal, read at the exponent 1 - BIAS, can have a negative sh,
// and only under a floor 2^f whose draw step's half, 2^(f - N - 1), is finer
// than the carrier's subnormal spacing: binaryK's NORMALS floor at 2^-126
// with man + N = MAN_BITS, and a superfp whose smallest value 2^c is at the
// bottom of the carrier's range (2^c from 2^-149 to 2^-126 in binary32).
// Both are formats that warn that the carrier cannot place their bottom, and
// their deterministic casts misplace these inputs too. Rounding away is the
// definition's answer for every input at or above 2^f (X >= 2^N there); in
// such a format an input below 2^f rounds away too, where the definition
// would round it away with probability |x| / 2^f. The exact answer,
// floor(2X) = m << -sh, was measured with dev/benchmarks/timing/gemm_sr_ab.cu:
// its few instructions, in arms most kernels never run, cost the CUDA SR
// split GEMMs 6-8% through the unrolled K-loop's instruction footprint, for
// every format, so the user chose this rule (2026-11-11).
// dev/benchmarks/sweeps/reference_casts.h states it in the format's terms,
// and tests/test_stochastic_rounding.py checks both cases against it.
template <class W>
CUDA_HOST_DEVICE_INLINE bool sr_away_below_or_above(W m, int sh, W r, int n, int variant, int word_bits)
{
    return sh < 0 || sr_away_below<W>(m, sh, r, n, variant, word_bits);
}

// floor(2X), and in `sticky` whether 2X has a fractional part, for
// X = m * 2^(N - below): a value m on a grid with `below` bits below its
// step. `below` may exceed the word's width, when m is a significand far
// under the step, or be less than N + 1, when floor(2X) is m shifted up.
template <class W>
CUDA_HOST_DEVICE_INLINE W sr_top2(W m, int below, int n, int word_bits, MPTORCH_THREAD bool &sticky)
{
    const int sh = below - n - 1; // floor(2X) = m >> sh
    if (sh <= 0)
    {
        sticky = false;
        return m << (-sh);
    }
    if (sh >= word_bits)
    {
        sticky = m != W(0);
        return W(0);
    }
    sticky = (m & ((W(1) << sh) - W(1))) != W(0);
    return m >> sh;
}

// The bit test in cross-multiplied form, for a smallest value v = d * 2^q
// that is not a power of two (an EXTENDED_NORMALS floor, d = 2^man + 1).
// `lhs_shift` is the shift that expresses m, the input's significand, in
// units of v / (d * 2^(N+1)); the magnitude then rounds away, to v, iff
// m * 2^lhs_shift >= j * d, with j = 2k for A and 2k - 1 for B and C, and on
// equality C also needs k even. lhs_shift <= 1 and j * d < 2^(N + man + 2),
// so nothing overflows the word.
template <class W>
CUDA_HOST_DEVICE_INLINE bool sr_away_scaled(W m, int lhs_shift, W d, W r, int n, int variant, int word_bits)
{
    const W k2 = (W(1) << (n + 1)) - (r << 1);
    const W rhs = (variant == 0 ? k2 : k2 - W(1)) * d;
    W lhs;
    bool rem = false;
    if (lhs_shift >= 0)
        lhs = m << lhs_shift;
    else if (-lhs_shift >= word_bits)
    {
        lhs = W(0);
        rem = m != W(0);
    }
    else
    {
        lhs = m >> (-lhs_shift);
        rem = (m & ((W(1) << (-lhs_shift)) - W(1))) != W(0);
    }
    if (lhs != rhs)
        return lhs > rhs;
    if (rem)
        return true;
    return variant != 2 || (n >= 1 && (r & W(1)) == W(0));
}
