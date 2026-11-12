#pragma once

// P3109's three stochastic rounding modes, in integers on the carrier's word.
// With eta the fraction of |x| between its two neighbouring values, R a
// random integer in [0, 2^N) and N the format's prng_bits, each rounds the
// magnitude away from zero iff
// (arXiv:2606.04028, Fig. 2):
//
//   StochasticA   floor(eta * 2^N) + R >= 2^N                 RoundMode::SR
//   StochasticB   floor(eta * 2^(N+1)) + 2R + 1 >= 2^(N+1)    RoundMode::SR_B
//   StochasticC   RNITE(eta * 2^N) + R >= 2^N                 RoundMode::SR_C
//
// RNITE rounds to nearest, ties to even. With X = eta * 2^N and k = 2^N - R,
// that is: A away iff X >= k; B iff X >= k - 1/2; C iff X > k - 1/2, or
// X = k - 1/2 and k is even (for N >= 1, iff R is even; at N = 0, k = 1 and
// the tie never rounds away). So N = 0 makes A RZ, B RNA, and C a round to
// nearest with ties toward zero; and where no bits of |x| lie below the
// N-bit field, X is an integer and B and C are A.
//
// Every cast states its two neighbours on the word, so nothing here rounds
// in the carrier first and no float operation touches the input (which on an
// Apple GPU would flush a binary32 subnormal). Two forms decide:
//
//   * The word form, for a value on a grid the word holds (a format's normal
//     binades, binaryK's subnormals, superfp's supernormal binades): add the
//     draw just below the kept bits, a constant below it, and truncate. It is
//     A's add-and-truncate plus two integer operations, with the constants
//     built once per format (StochasticT, make_stochastic).
//   * The bit test, sr_away, from floor(2X) and whether 2X has a fraction
//     left: the regions under a format's smallest step, where the value is
//     shifted rather than added to. sr_away_scaled is its cross-multiplied
//     form for a floor that is not a power of two (EXTENDED_NORMALS).
//
// Under SR a format's prng_bits field carries a spec: N in the low byte and
// the variant (0 A, 1 B, 2 C) above it. Only the entry points and the
// drivers write one (they map RoundMode::SR_B and SR_C to RoundMode::SR and
// the variant), so every kernel has one stochastic instantiation and the
// deterministic ones never see a spec.

#include "bit_helper.h"
#include "modes.h"

// An arm no common format reaches (binaryK's cross-multiplied
// EXTENDED_NORMALS floor) is out of line on the host and CUDA, so its code
// stays out of the GEMMs' unrolled K-loop. It takes its operands by value,
// since a reference into a policy would put the policy on the device's
// stack. Every other arm is inline: a call costs the loop more than an arm
// that narrow formats reach in most warps (cast_binaryK_stochastic's note
// has the measurements). Metal folds the format's literals and has no
// attribute for it, so there it is inline too.
#if defined(__METAL_VERSION__)
#define MPTORCH_SR_COLD inline
#elif defined(__CUDACC__)
#define MPTORCH_SR_COLD __host__ __device__ __noinline__
#else
#define MPTORCH_SR_COLD __attribute__((noinline, cold))
#endif

// The spec a format's prng_bits field carries under stochastic rounding.
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
// lowest bit sits at bit s of the word: B adds half a unit of that bit, C
// adds half a unit, less one when R is odd (the tie toward even k), and at
// N = 0 half a unit less one (the tie toward zero). Nothing when s = 0.
template <class W>
CUDA_HOST_DEVICE_INLINE void sr_word_constants(int s, int n, int variant, MPTORCH_THREAD W &or_v,
                                               MPTORCH_THREAD W &tie)
{
    or_v = W(0);
    tie = W(0);
    if (s >= 1 && variant != 0)
    {
        // Not `half`: that is a type in MSL, which compiles this header too.
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

// A stochastic round's constants for one format: the draw's field (its N
// bits directly below the format's significand, bits [MAN - man - N, MAN -
// man) of the draw word), and the word form's constants at that field and at
// a supernormal binade's (superfp's, whose grid keeps no mantissa bit), each
// with the mask of the bits it keeps, so the hot path shifts nothing.
template <class T>
struct StochasticT
{
    using word_t = typename FloatTraits<T>::word_t;
    int bits;    // N
    int variant; // 0 A, 1 B, 2 C
    int shift;   // the field's lowest bit: MAN_BITS - man_bits - N
    word_t field; // 2^N - 1
    word_t draw;  // the field in place: field << shift
    word_t or_v;
    word_t tie;   // C's tie, 0 or 1: R's lowest bit, bit `shift` of the draw
    word_t keep;  // the bits the word form keeps: from shift + N up
    word_t or_sup; // the constants for a draw at bit MAN_BITS - N
    word_t tie_sup;
    word_t keep_sup;
    // The family's builder fills the rest (make_stochastic_binaryK,
    // make_stochastic_superfp): the words the hot path tests, so that a
    // value inside the format's normal range takes one compare. `wmask` is
    // ABS_MASK for a signed format and every bit for an unsigned one, whose
    // negative inputs then fail every range; a range is [lo, lo + span) of
    // the masked word.
    word_t wmask;
    word_t lo;
    word_t span;
    word_t lo_sup; // superfp's supernormal binades
    word_t span_sup;
};

// The word of the first value whose exponent field reads e + BIAS: the
// bound a field compare `target_exp >= e` draws on the word, clamped to the
// field's range so a bound past either end keeps the compare's verdict.
template <class T>
CUDA_HOST_DEVICE_INLINE typename FloatTraits<T>::word_t sr_field_word(int e)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    const int f = e + F::BIAS;
    return f <= 0 ? word_t(0) : f >= (int)F::FIELD_MASK ? (word_t)F::FIELD_MASK << F::MAN_BITS
                                                          : (word_t)f << F::MAN_BITS;
}

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

// 2^e on the carrier's word: a normal, a subnormal below the normals, and
// zero below the subnormals (a step no carrier holds, in a format only
// warned about).
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
// up, R added at bit s with the variant's constants below it. Requires
// s >= 0 and s + n below the word's width.
template <class W>
CUDA_HOST_DEVICE_INLINE W sr_round_word(W w, W r, int s, int n, W or_v, W tie)
{
    const W t = w + (r << s) + or_v - (r & tie);
    return t & ~((W(1) << (s + n)) - W(1));
}

// The word form at the format's own field, from the draw word itself: R
// stays where the draw holds it, C's tie is the draw's bit `shift`, and the
// mask is the policy's. sr_round_word's value, with nothing derived per call.
template <class T>
CUDA_HOST_DEVICE_INLINE typename FloatTraits<T>::word_t
sr_round_in_place(typename FloatTraits<T>::word_t w, typename FloatTraits<T>::word_t rand_prob,
                  const MPTORCH_THREAD StochasticT<T> &st)
{
    using word_t = typename FloatTraits<T>::word_t;
    const word_t t = w + (rand_prob & st.draw) + st.or_v - ((rand_prob >> st.shift) & st.tie);
    return t & st.keep;
}

// The bit test: with top2 = floor(2X) and `sticky` whether 2X has a fraction
// left, whether the magnitude rounds away. A needs top2 >= 2k; B top2 >=
// 2k - 1; C top2 >= 2k - 1 unless 2X = 2k - 1 exactly with k odd.
template <class W>
CUDA_HOST_DEVICE_INLINE bool sr_away(W top2, bool sticky, W r, int n, int variant)
{
    const bool k_even = n >= 1 && (r & W(1)) == W(0);
    const W d = variant == 0 ? W(0) : variant == 1 ? W(1) : ((sticky || k_even) ? W(1) : W(0));
    return top2 + (r << 1) + d >= (W(1) << (n + 1));
}

// The bit test for a region under a format's smallest step, inline and
// without branches: X = m * 2^(N - below) with sh = below - N - 1 >= 0 (a
// magnitude under the step has more bits below it than the field holds), so
// floor(2X) = m >> sh, clamped to the word's width, where m < 2^(MAN_BITS +
// 1) leaves nothing; `sticky` is whether the shift dropped a bit. sr_top2 and
// sr_away's value, for the arms that cannot take sr_top2's other cases.
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

// The same where sh can be negative, which rounds away. Only a carrier
// subnormal, read at the exponent 1 - BIAS, gets a negative sh, and only
// under a floor 2^f whose draw quantum's half, 2^(f - N - 1), is finer than
// the carrier's subnormal spacing: binaryK's NORMALS floor at 2^-126 with
// man + N = MAN_BITS, and a superfp whose smallest value 2^c is itself a
// carrier subnormal (c between -149 and -126 in binary32). Both are formats
// that warn that the carrier cannot place their bottom, and their
// deterministic casts misplace these inputs too. Rounding away is the
// definition's answer for every input at or above 2^f (X >= 2^N there); one
// below it in such a format goes up too, where the definition would go up
// with probability |x| / 2^f. The exact answer, floor(2X) = m << -sh, was
// measured with gemm_sr_ab.cu: its few instructions, in arms most kernels never
// run, cost the CUDA SR split GEMMs 6-8% through the unrolled K-loop's
// instruction footprint, for every format, so the user chose this rule
// (2026-11-11). dev/benchmarks/sweeps/reference_casts.h states it in the
// format's terms, and tests/test_stochastic_rounding.py holds both cases to it.
template <class W>
CUDA_HOST_DEVICE_INLINE bool sr_away_below_or_above(W m, int sh, W r, int n, int variant, int word_bits)
{
    return sh < 0 || sr_away_below<W>(m, sh, r, n, variant, word_bits);
}

// floor(2X) and the fraction flag for X = m * 2^(N - below): a value m on a
// grid that has `below` bits below its step (below may exceed the word's
// width, or be less than N + 1, when m is a significand far under the step).
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

// The bit test for X = m * 2^sh * 2^(N+1) / ... in its cross-multiplied
// form, for a smallest value v = d * 2^q that is not a power of two (an
// EXTENDED_NORMALS floor, d = 2^man + 1): `lhs_shift` is the shift that puts
// m (the input's significand) in units of v / (d * 2^(N+1)), so the
// magnitude rounds away to v iff m * 2^lhs_shift >= j * d, j = 2k (A) or
// 2k - 1 (B, C), and at equality C needs k even. lhs_shift <= 1, and j * d
// < 2^(N + man + 2), so nothing overflows the word.
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
