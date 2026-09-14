#pragma once

#include "modes.h"
#include <cmath>
#include <cstdint>
#include <type_traits>

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define CUDA_HOST_DEVICE_INLINE inline
#endif

// MPTORCH_FAST_CAST admits the float-arithmetic cast fast paths in
// cast_binaryK.h and cast_superfp.h (findings G3, G7, G8). They replace ~125
// branchy integer instructions with ~14 branchless float ones, and they are
// exact only if every arithmetic operation they name is a separately rounded
// binary32 operation: the Veltkamp split's `t - (t - x)` is Dekker's identity,
// which a fused multiply-add silently breaks (248 M mismatches over the
// exhaustive sweep when nvcc was allowed to contract it).
//
// Device code always qualifies -- the paths are written with the _rn
// intrinsics, which name the rounding and cannot be contracted -- so the
// default below turns them on for the device pass of any .cu, whether it is
// built by setup.py or by hand (dev/benchmarks, temp/). Host code qualifies
// only when the build can also promise no contraction, which is a compiler
// flag rather than a property of the source, so there the macro comes from the
// command line: setup.py defines it together with -ffp-contract=off, both or
// neither. It is deliberately not passed to nvcc, whose host pass compiles
// these same headers under a host compiler we do not hand flags to.
//
// See dev/gemm_roadmap.md (finding C5).
#if !defined(MPTORCH_FAST_CAST) && defined(__CUDA_ARCH__)
#define MPTORCH_FAST_CAST 1
#endif

// Of the four paths the contract admits, three pay on both backends and one
// does not. cast_superfp_rne_fast trades the integer path's ~17 branches for a
// Veltkamp split plus a region branch, and superfp's integer path is not the
// expensive one binaryK's is: its supernormal arm -- where ordinary data lives,
// since normal_binades is usually 1 or 2 -- is already five operations. On a
// GPU that still wins, because warps are close to region-uniform and the
// branch is nearly free. On a host it pays only where ordinary data does not
// go: 1.11x with the operands in the normal region, 0.74x with them in the
// supernormal one -- and normal_binades is usually 1 or 2, which is what puts
// them there (dev/gemm_roadmap.md, finding C5). So the host keeps the integer
// path for that one cast. The function itself stays defined under the contract
// above, so the host sweeps still verify it -- it is rejected on speed, not on
// correctness, and the two are worth being able to tell apart.
#if defined(MPTORCH_FAST_CAST) && defined(__CUDA_ARCH__)
#define MPTORCH_FAST_CAST_SUPERFP_RNE 1
#endif

// The half of that contract a header can check. setup.py probes the compiler it
// is about to invoke, but the flags torch actually builds with are Python's own
// CFLAGS plus ours, which the probe never sees -- so an x87 target reaching this
// far should stop the build rather than quietly round every intermediate to 80
// bits and take the identity with it. (Contraction has no such tell: neither GCC
// nor Clang defines a macro for -ffp-contract, which is why that half is a
// compiler flag paired with an exhaustive sweep instead.)
#if defined(MPTORCH_FAST_CAST) && !defined(__CUDACC__)
#include <cfloat>
static_assert(FLT_EVAL_METHOD == 0,
              "MPTORCH_FAST_CAST needs float expressions evaluated in float; "
              "this target has excess precision (x87). Drop -DMPTORCH_FAST_CAST.");
#endif

// ---------------------------------------------------------------------------
// The carrier.
//
// Every cast rounds on the bit pattern of the IEEE binary format the value is
// computed in -- its *carrier*. What depends on which one -- the word, the
// field widths and masks, the bias, and the spellings of the operations the
// fast paths name -- is read from FloatTraits<T>, so each helper below, and
// each params struct and cast in cast_binaryK.h and cast_superfp.h, is written
// once, as a template on the carrier. `BinaryKParams` and `SuperfpParams` are
// the binary32 instantiations, and every kernel in the extension still
// computes in binary32; binary64 is the second carrier
// (dev/binary64_carrier_plan.md). The one thing that differs about it beyond
// the constants is HAS_FAST_CAST: the float-arithmetic fast paths are
// binary32's only, so binary64 always takes the integer path.
//
// The helpers that see only a word -- the bitwise rounds and the clips --
// deduce the carrier from it (WordTraits below), so a caller hands them the
// word it has and cannot pair a binary64 word with binary32's masks. Those
// that see a value deduce it from the value.
//
// The templates compile to the instructions the binary32-only code did, on
// both compilers: .text and .nv_fatbin came out byte for byte the same, which
// is how "no change" was checked. Four spellings are what that took. Each is
// invisible in the arithmetic, and each moved codegen when it was done the
// obvious way:
//
//   * The pun between a value and its word is `reinterpret_cast<const T &>`
//     on the variable, written where it is used, never inside a to_bits()/
//     from_bits() function. A call, however inlined, hides the variable's
//     address from the passes that promote it before inlining: by reference
//     it turned a select on the float into a select on the word in superfp's
//     RNA underflow arm, which moved nine device kernels, and by value it
//     moved the RZ quantizer by 24 instructions. A constant is copied into a
//     variable first, since device code cannot take a reference to a static
//     member.
//
//     The pun reads a word through a float glvalue, which the standard does
//     not allow and -Wstrict-aliasing says so; it is what these casts have
//     always done, and the exhaustive sweeps are what vouch for it. memcpy
//     is the spelling the standard does allow, and it was measured too,
//     written at the use site like the pun: it changes 45 of the 128 device
//     kernels (four of them by a register) and 371 host functions, 84 of
//     them CPU GEMM kernels, and it takes the host .text from 2,175,586 to
//     2,112,562 B. Not a change of values, but not the same kernels either,
//     so it fails the gate this template was held to; whether the smaller
//     host code is also faster is a question for a benchmark, not for this
//     note (dev/binary64_carrier_plan.md, phase 1b).
//   * fabsf and copysignf are called by name, not through a trait. GCC folds
//     the builtins before inlining and a wrapper only after, which reordered
//     the host superfp SR quantizers. (min/max and the rn_ operations were
//     wrappers before this template and are unchanged by being members.)
//   * A shift that was written on `int` is written on sword_t, the carrier's
//     signed word, not on the unsigned one: `1 << s` is undefined past the
//     sign bit, which GCC reads as `s < 31` and the unsigned spelling does
//     not let it, and round_bitwise_stochastic's mask moved the host SR
//     quantizers until it was signed again. For binary32 sword_t is the `int`
//     it always was; for binary64 it is wide enough for a 52-bit shift.
//   * A fast-path gate is a plain `if` on the params' fast_rne, which is a
//     constant false for a carrier without one, and only the body it guards
//     is behind `if constexpr` -- see cast_binaryK_nearest_even.
// ---------------------------------------------------------------------------
template <class T>
struct FloatTraits; // defined for the two carriers below, and nothing else

template <>
struct FloatTraits<float>
{
    using value_t = float;
    using word_t = uint32_t;
    using sword_t = int32_t;
    static constexpr int MAN_BITS = 23;   // significand field
    static constexpr int EXP_BITS = 8;    // exponent field
    static constexpr int BIAS = 127;
    static constexpr int WORD_BITS = 32;
    static constexpr int MAX_EXP = 127;   // the largest finite value's exponent
    static constexpr int INF_EXP = 128;   // an all-ones field, unbiased: infinity and NaN
    static constexpr int TOP_FIELD = 254; // the largest finite value's field
    static constexpr int MIN_NORMAL_EXP = -126;
    // The lowest floor a format can put its smallest value at when its grid
    // there is read off the input's exponent field, which every subnormal of
    // the carrier shares (dev/gemm_roadmap.md, T4).
    static constexpr int MIN_SIMULABLE_EXP = -125;
    static constexpr word_t FIELD_MASK = 0xFFu;
    static constexpr word_t SIGN_MASK = 0x80000000u;
    static constexpr word_t ABS_MASK = 0x7FFFFFFFu;
    static constexpr word_t MAN_MASK = 0x007FFFFFu;
    static constexpr word_t INF_BITS = 0x7F800000u;
    static constexpr word_t TIE = 0x00400000u;       // half the exponent field's low bit: the tie to a power of two
    static constexpr word_t BELOW_TIE = 0x003FFFFFu; // the largest subnormal below 2^(MIN_NORMAL_EXP - 1)
    static constexpr bool HAS_FAST_CAST = true;

    // 2^e for e in [MIN_NORMAL_EXP, MAX_EXP], exact by construction, and
    // without pulling ldexp into a header that CUDA device code includes.
    CUDA_HOST_DEVICE_INLINE static value_t pow2(int e)
    {
        word_t bits = (word_t)(e + BIAS) << MAN_BITS;
        return reinterpret_cast<const value_t &>(bits);
    }

    // 2^-MAN_BITS: what turns the MAN_BITS low bits of a random word into a
    // uniform draw in [0, 1). Spelled as the quotient so it folds to the same
    // constant it always has.
    CUDA_HOST_DEVICE_INLINE static value_t ulp_scale() { return 1.0f / 8388608.0f; }

    // rn_add/rn_sub/rn_mul are the operations the Veltkamp split needs rounded
    // on their own. On the device that is what __fadd_rn and friends mean. On
    // the host the plain operators already are IEEE binary32 operations -- the
    // only thing that can merge two of them is contraction into an FMA, which
    // MPTORCH_FAST_CAST's contract forbids -- so the shim is the operator
    // itself, named so the requirement stays legible at the call site. (fma_f32
    // in gemm_policy.h is the reverse case: the one place an FMA is intended,
    // spelled explicitly so no flag can take it away.)
    //
    // min/max where the fast paths use them: on a magnitude clamped against a
    // positive bound, never on a negative operand. Under that precondition
    // the ternary and fminf/fmaxf agree everywhere, NaN included -- every
    // compare against a NaN is false, so both spellings return the bound,
    // which is the arm the fast paths' final select overrides anyway.
    //
    // Two spellings because each backend folds exactly one of them into its
    // single instruction and neither folds the other: GCC will not turn fminf
    // into minss (their NaN results differ in general) and emits a libm call
    // in the middle of the GEMM's inner loop, worth 1.12x of the whole CPU
    // GEMM; nvcc will not turn the ternary into min.f32 and emits setp + selp.
#if defined(__CUDA_ARCH__)
    CUDA_HOST_DEVICE_INLINE static value_t rn_add(value_t a, value_t b) { return __fadd_rn(a, b); }
    CUDA_HOST_DEVICE_INLINE static value_t rn_sub(value_t a, value_t b) { return __fsub_rn(a, b); }
    CUDA_HOST_DEVICE_INLINE static value_t rn_mul(value_t a, value_t b) { return __fmul_rn(a, b); }
    CUDA_HOST_DEVICE_INLINE static value_t fmin_nonneg(value_t a, value_t b) { return fminf(a, b); }
    CUDA_HOST_DEVICE_INLINE static value_t fmax_nonneg(value_t a, value_t b) { return fmaxf(a, b); }
#else
    CUDA_HOST_DEVICE_INLINE static value_t rn_add(value_t a, value_t b) { return a + b; }
    CUDA_HOST_DEVICE_INLINE static value_t rn_sub(value_t a, value_t b) { return a - b; }
    CUDA_HOST_DEVICE_INLINE static value_t rn_mul(value_t a, value_t b) { return a * b; }
    CUDA_HOST_DEVICE_INLINE static value_t fmin_nonneg(value_t a, value_t b) { return (a < b) ? a : b; }
    CUDA_HOST_DEVICE_INLINE static value_t fmax_nonneg(value_t a, value_t b) { return (a > b) ? a : b; }
#endif
};

template <>
struct FloatTraits<double>
{
    using value_t = double;
    using word_t = uint64_t;
    using sword_t = int64_t;
    static constexpr int MAN_BITS = 52;
    static constexpr int EXP_BITS = 11;
    static constexpr int BIAS = 1023;
    static constexpr int WORD_BITS = 64;
    static constexpr int MAX_EXP = 1023;
    static constexpr int INF_EXP = 1024;
    static constexpr int TOP_FIELD = 2046;
    static constexpr int MIN_NORMAL_EXP = -1022;
    static constexpr int MIN_SIMULABLE_EXP = -1021;
    static constexpr word_t FIELD_MASK = 0x7FFu;
    static constexpr word_t SIGN_MASK = 0x8000000000000000u;
    static constexpr word_t ABS_MASK = 0x7FFFFFFFFFFFFFFFu;
    static constexpr word_t MAN_MASK = 0x000FFFFFFFFFFFFFu;
    static constexpr word_t INF_BITS = 0x7FF0000000000000u;
    static constexpr word_t TIE = 0x0008000000000000u;
    static constexpr word_t BELOW_TIE = 0x0007FFFFFFFFFFFFu;
    // The integer path only, for now: a binary64 fast path is a measured item
    // of its own, since FP64 is a fraction of the float rate on consumer GPUs.
    static constexpr bool HAS_FAST_CAST = false;

    CUDA_HOST_DEVICE_INLINE static value_t pow2(int e)
    {
        word_t bits = (word_t)(e + BIAS) << MAN_BITS;
        return reinterpret_cast<const value_t &>(bits);
    }
    CUDA_HOST_DEVICE_INLINE static value_t ulp_scale() { return 1.0 / 4503599627370496.0; }

    // see the binary32 specialization for what each of these promises
#if defined(__CUDA_ARCH__)
    CUDA_HOST_DEVICE_INLINE static value_t rn_add(value_t a, value_t b) { return __dadd_rn(a, b); }
    CUDA_HOST_DEVICE_INLINE static value_t rn_sub(value_t a, value_t b) { return __dsub_rn(a, b); }
    CUDA_HOST_DEVICE_INLINE static value_t rn_mul(value_t a, value_t b) { return __dmul_rn(a, b); }
    CUDA_HOST_DEVICE_INLINE static value_t fmin_nonneg(value_t a, value_t b) { return fmin(a, b); }
    CUDA_HOST_DEVICE_INLINE static value_t fmax_nonneg(value_t a, value_t b) { return fmax(a, b); }
#else
    CUDA_HOST_DEVICE_INLINE static value_t rn_add(value_t a, value_t b) { return a + b; }
    CUDA_HOST_DEVICE_INLINE static value_t rn_sub(value_t a, value_t b) { return a - b; }
    CUDA_HOST_DEVICE_INLINE static value_t rn_mul(value_t a, value_t b) { return a * b; }
    CUDA_HOST_DEVICE_INLINE static value_t fmin_nonneg(value_t a, value_t b) { return (a < b) ? a : b; }
    CUDA_HOST_DEVICE_INLINE static value_t fmax_nonneg(value_t a, value_t b) { return (a > b) ? a : b; }
#endif
};

// The constants above are written out, as they appear in IEEE 754, rather than
// derived from one another; this checks each against the ones it follows from.
template <class T>
constexpr bool float_traits_consistent()
{
    using F = FloatTraits<T>;
    using W = typename F::word_t;
    return sizeof(typename F::value_t) == sizeof(W) && F::WORD_BITS == 8 * (int)sizeof(W) &&
           F::WORD_BITS == 1 + F::EXP_BITS + F::MAN_BITS && F::BIAS == (1 << (F::EXP_BITS - 1)) - 1 &&
           F::MAX_EXP == F::BIAS && F::INF_EXP == F::BIAS + 1 && F::TOP_FIELD == 2 * F::BIAS &&
           F::MIN_NORMAL_EXP == 1 - F::BIAS && F::MIN_SIMULABLE_EXP == F::MIN_NORMAL_EXP + 1 &&
           F::FIELD_MASK == (W(1) << F::EXP_BITS) - 1 && F::SIGN_MASK == W(1) << (F::WORD_BITS - 1) &&
           F::ABS_MASK == ~F::SIGN_MASK && F::MAN_MASK == (W(1) << F::MAN_BITS) - 1 &&
           F::INF_BITS == F::FIELD_MASK << F::MAN_BITS && F::TIE == W(1) << (F::MAN_BITS - 1) &&
           F::BELOW_TIE == F::TIE - 1 && std::is_same_v<typename F::sword_t, std::make_signed_t<W>>;
}
static_assert(float_traits_consistent<float>(), "FloatTraits<float> disagrees with itself");
static_assert(float_traits_consistent<double>(), "FloatTraits<double> disagrees with itself");

// The carrier a tensor of scalar type S rounds in: binary64 for a double, and
// binary32 for everything else -- float, and the half-width types, whose
// values binary32 holds and whose casts are binary32's.
template <class S>
using carrier_t = std::conditional_t<std::is_same_v<S, double>, double, float>;

// The carrier whose word W is. Declared for the two carriers' words only, so a
// word of any other type -- an `int`, a word of the wrong width -- fails to
// compile instead of reaching binary32's masks.
template <class W>
struct CarrierOfWord;
template <>
struct CarrierOfWord<uint32_t>
{
    using type = float;
};
template <>
struct CarrierOfWord<uint64_t>
{
    using type = double;
};
template <class W>
using WordTraits = FloatTraits<typename CarrierOfWord<W>::type>;

// What a cast returns instead of -0.0, on the word. IEEE P3109 has one zero,
// code point 0, and it is unsigned -- in a signed binaryK the encoding that
// would hold -0.0 is the format's NaN -- and superfp spends no code on a
// negative zero either. So a result that rounds to zero is +0.0 whatever the
// sign of the value that got there, -0.0 itself included: the value simulated
// is the format's, not the carrier's (dev/gemm_roadmap.md, T2).
//
// A mask on the word, not a select: g++ compiles `((w << 1) == 0) ? 0u : w`
// to a branch and this to a cmov, and vectorizes this form in a loop. A NaN's
// word is nonzero after the shift, so the mask keeps it whole -- payload,
// signalling bit and all.
//
// The casts apply this where a zero is *made*, not on the way out: the clip
// functions below drop the sign with the magnitude, superfp's region arms
// return a bare 0, and the one zero that can still be signed afterwards --
// the directed modes negating a magnitude that rounded away -- is handled by
// negate_magnitude at the end of this file. T2 had every integer path return
// through the float spelling of this instead, which cost the elementwise
// casts 1.05-1.23x; T3 is what moved it to the arms (dev/gemm_roadmap.md).
template <class W>
CUDA_HOST_DEVICE_INLINE W unsigned_zero_bits(W w)
{
    return w & -(W)((w << 1) != 0);
}

// rounds to nearest, ties to even
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_nearest_even(W target, int man_bits)
{
    using F = WordTraits<W>;
    if (man_bits >= F::MAN_BITS)
        return target;
    W mask = (W)(((typename F::sword_t)1 << (F::MAN_BITS - man_bits)) - 1);
    W tie = (W)((typename F::sword_t)1 << (F::MAN_BITS - 1 - man_bits));
    W add_r = target + tie;
    W quantized = add_r & ~mask;
    W is_tie = (target & mask) == tie;
    W odd = (man_bits == 0) ? 0 : 1; // if man_bits == 0, implicit bit is 1 (odd) so we always round up (carry to exponent)
    return quantized & ~((is_tie & odd) << (F::MAN_BITS - man_bits));
}

template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_nearest_even(W target)
{
    using F = WordTraits<W>;
    W tie = F::TIE;
    W quantized = (target + tie) & ~F::MAN_MASK;
    W is_tie = (target & F::MAN_MASK) == tie;
    return quantized - ((is_tie << F::MAN_BITS) & ~quantized);
}

// Precomputed-parameter forms below: same bit-twiddling as the
// man_bits/exp_bits/bias-taking functions, but reading values derived from
// those format parameters once instead of recomputing them on every call. A
// GEMM Multiplier/Adder (gemm_policy.h) precomputes them at construction and
// reuses them for a whole kernel launch (up to M*N*K calls); the elementwise
// quantizers do the same once per tensor. Deliberately flat (no nested
// structs) -- see dev/gemm_roadmap.md item 6 for the compile-time blowup that
// motivated this. RoundMode::SR is out of scope here (it needs a per-call
// random value, not a format constant).
//
// Which spelling each helper still keeps differs, and the question is whether
// its integer argument is always a format constant:
//
//   round_bitwise_*   both forms. The casts call the man_bits-taking one with
//                     a per-value exp_diff in their subnormal branches, so it
//                     is live in the extension, not only in the sweeps.
//   clip_*            precomputed only. The format-taking forms went to
//                     dev/benchmarks/reference_casts.h along with the casts
//                     that were their last callers (finding H2).

// Precomputed for round_bitwise_nearest_even(target, man_bits) and reused by
// nearest-away/up/down/odd's precomputed overloads below (they share the
// same bypass/mask/tie formulas). Not valid for the man_bits == 0 case,
// which uses the structurally different zero-arg overload above.
template <class T>
struct RoundParamsT
{
    using word_t = typename FloatTraits<T>::word_t;
    bool bypass; // man_bits >= MAN_BITS: round_bitwise_nearest_even returns target unchanged
    word_t mask;
    word_t tie;
    int shift; // MAN_BITS - man_bits
};

template <class T>
CUDA_HOST_DEVICE_INLINE RoundParamsT<T> make_round_params(int man_bits)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    RoundParamsT<T> p;
    p.bypass = man_bits >= F::MAN_BITS;
    if (p.bypass)
    {
        p.mask = 0u;
        p.tie = 0u;
        p.shift = 0;
        return p;
    }
    p.shift = F::MAN_BITS - man_bits;
    p.mask = (word_t(1) << p.shift) - 1u;
    p.tie = word_t(1) << (p.shift - 1);
    return p;
}

template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_nearest_even(W target, bool bypass, W mask, W tie, int shift)
{
    if (bypass)
        return target;
    W add_r = target + tie;
    W quantized = add_r & ~mask;
    W is_tie = (target & mask) == tie;
    return quantized & ~(is_tie << shift);
}

// rounds to nearest, ties to away
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_nearest_away(W target, int man_bits)
{
    using F = WordTraits<W>;
    if (man_bits >= F::MAN_BITS)
        return target;
    W mask = (W)(((typename F::sword_t)1 << (F::MAN_BITS - man_bits)) - 1);
    W tie = (W)((typename F::sword_t)1 << (F::MAN_BITS - 1 - man_bits));
    W add_r = target + tie;
    return add_r & ~mask;
}

// Precomputed sibling of round_bitwise_nearest_away above, via RoundParamsT
// (never needs the `shift` field).
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_nearest_away(W target, bool bypass, W mask, W tie)
{
    if (bypass)
        return target;
    W add_r = target + tie;
    return add_r & ~mask;
}

// rounds up, towards positive infinity
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_up(W target, int man_bits)
{
    using F = WordTraits<W>;
    if (man_bits >= F::MAN_BITS)
        return target;
    W mask = (W)(((typename F::sword_t)1 << (F::MAN_BITS - man_bits)) - 1);
    W sign = target >> (F::WORD_BITS - 1);
    W add_r = target + (sign ? 0 : mask);
    return add_r & ~mask;
}

// Precomputed sibling of round_bitwise_up above: `sign` is per-value data,
// not a format constant, so only bypass/mask (man_bits-derived) precompute.
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_up(W target, bool bypass, W mask)
{
    using F = WordTraits<W>;
    if (bypass)
        return target;
    W sign = target >> (F::WORD_BITS - 1);
    W add_r = target + (sign ? 0 : mask);
    return add_r & ~mask;
}

// rounds down, towards negative infinity
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_down(W target, int man_bits)
{
    using F = WordTraits<W>;
    if (man_bits >= F::MAN_BITS)
        return target;
    W mask = (W)(((typename F::sword_t)1 << (F::MAN_BITS - man_bits)) - 1);
    W sign = target >> (F::WORD_BITS - 1);
    W add_r = target + (sign ? mask : 0);
    return add_r & ~mask;
}

// Precomputed sibling of round_bitwise_down above -- see round_bitwise_up's.
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_down(W target, bool bypass, W mask)
{
    using F = WordTraits<W>;
    if (bypass)
        return target;
    W sign = target >> (F::WORD_BITS - 1);
    W add_r = target + (sign ? mask : 0);
    return add_r & ~mask;
}

// rounds to odd: truncates towards zero to man_bits, then ORs the "sticky
// bit" of the discarded bits into the kept LSB. When man_bits == 0 the kept
// "bit" is the exponent's LSB (no explicit significand), so a nonzero
// sticky bit carries into the exponent.
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_odd(W target, int man_bits)
{
    using F = WordTraits<W>;
    if (man_bits >= F::MAN_BITS)
        return target;
    W mask = (W)(((typename F::sword_t)1 << (F::MAN_BITS - man_bits)) - 1);
    W lsb = (W)((typename F::sword_t)1 << (F::MAN_BITS - man_bits));
    W sticky = (target & mask) != 0;
    return (target & ~mask) | (sticky * lsb);
}

// Precomputed sibling of round_bitwise_odd above. lsb = mask + 1, so no
// separate field is needed for it.
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_odd(W target, bool bypass, W mask)
{
    if (bypass)
        return target;
    W lsb = mask + 1u;
    W sticky = (target & mask) != 0;
    return (target & ~mask) | (sticky * lsb);
}

// stochastic rounding
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_stochastic(W target, W rand_prob, int man_bits)
{ // passing number of random bits as second parameter
    // (all the bits after the least significant bit which is based on prng);
    // target is the original number
    using F = WordTraits<W>;
    W mask = (W)(((typename F::sword_t)1 << (F::MAN_BITS - man_bits)) - 1);
    // adding random bits to target (which is not masked)
    W add_r = target + (rand_prob & mask);
    // masking out bits on the right hand side of the significant bits (truncating)
    W quantized = add_r & ~mask;
    return quantized;
}

// Precomputed for clip_subnormal_range_exponent. exp_bits is accepted by the
// format-taking form (dev/benchmarks/reference_casts.h) for signature symmetry
// with its sibling clip functions but unused by its body, so only
// min_exponent_store -- derived from man_bits and bias -- needs precomputing.
struct SubnormalRangeParams
{
    int min_exponent_store;
};

template <class T>
CUDA_HOST_DEVICE_INLINE SubnormalRangeParams make_subnormal_range_params(int man_bits, int bias)
{
    SubnormalRangeParams p;
    p.min_exponent_store = -(bias - 1) - man_bits + FloatTraits<T>::BIAS;
    return p;
}

template <class W>
CUDA_HOST_DEVICE_INLINE W clip_subnormal_range_exponent(W old_num, W quantized_num, int min_exponent_store)
{
    using F = WordTraits<W>;
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = (int)((quantized_num >> F::MAN_BITS) & F::FIELD_MASK);

    W old_sign = old_num & F::SIGN_MASK;
    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
    {
        int offset = (quantized_exponent_store == (min_exponent_store - 1));
        quantized_num += offset * (W(1) << F::MAN_BITS);
        quantized_num |= old_sign;
        quantized_num *= offset; // offset == 0 is the underflow, and it is +0.0
    }
    else
        // A word with no magnitude but a sign bit -- a negative value the
        // rounding above took to zero -- reaches this arm only in a format
        // whose min_exponent_store is at or below zero, where the underflow
        // test is dead for it. P3109's zero is unsigned either way (T2/T3).
        quantized_num = unsigned_zero_bits(quantized_num);

    return quantized_num;
}

// The _up variant: clamps an underflowing nonzero value to the smallest
// subnormal rather than to zero, for the directed and round-to-odd modes.
// Whether the value is nonzero is a question about the input, not about the
// rounded word: a -0.0 input truncates to a lone sign bit, which is not 0 and
// was pushed out to minus the smallest subnormal, and a subnormal input of the
// carrier can truncate to exactly 0 and stayed there. P3109 rounds zero to
// zero and never rounds a nonzero value away from zero into it, so a zero
// input is answered with an unsigned zero here rather than with its rounded
// word. Shares SubnormalRangeParams/make_subnormal_range_params with the plain
// clip_subnormal_range_exponent above.
template <class W>
CUDA_HOST_DEVICE_INLINE W clip_subnormal_range_exponent_up(W old_num, W quantized_num, int min_exponent_store)
{
    using F = WordTraits<W>;
    if ((old_num << 1) == 0)
        return 0u; // a zero input rounds to zero, and that zero is unsigned

    int quantized_exponent_store = (int)((quantized_num >> F::MAN_BITS) & F::FIELD_MASK);

    W old_sign = old_num & F::SIGN_MASK;
    if (quantized_exponent_store < min_exponent_store)
    {
        quantized_num = (W)((typename F::sword_t)min_exponent_store << F::MAN_BITS);
        quantized_num |= old_sign;
    }
    else
        // as in clip_subnormal_range_exponent above: a signed word with no
        // magnitude, in a format whose underflow test cannot reach it
        quantized_num = unsigned_zero_bits(quantized_num);

    return quantized_num;
}

// Precomputed for the normal-range clip below -- the form every cast in
// cast_binaryK.h and cast_superfp.h calls.
template <class T>
struct NormalRangeParamsT
{
    using word_t = typename FloatTraits<T>::word_t;
    SaturationMode saturation_mode; // still needed raw: selects the overflow branch's outcome
    word_t min_num;                 // the smallest magnitude the format represents, as a word
    word_t half_num;                // half of it: the round-to-nearest boundary below it
    word_t max_num;                 // the largest finite magnitude, as a word
};

// The top of the range is counted in code points, the way IEEE P3109 assigns
// them (arXiv:2606.04028, SVII): of the top binade's 2^man_bits codes, the last
// `reserved_codes` hold no finite value, and max_num is the largest code left.
// binaryK reserves P3109's: +infinity outside SAT_FINITE (the extended domain)
// and, in an unsigned format, NaN above it. superfp reserves the infinity only.
//
// With man_bits <= 1 the reserved codes can outnumber the top binade's, and the
// largest finite code then sits one or two binades lower. A format whose
// largest finite code would have exponent field 0 -- no finite normal value at
// all, which P3109 rules out by requiring three bits -- gets max_num = 0, and
// every nonzero result of the normal arm overflows. So does one whose largest
// finite value is below the carrier's normal range.
//
// The bottom is a code point too, and the subnormals mode is what says which
// one. `SUBNORMALS` and `NORMALS` put the floor at the smallest normal, whose
// code is the first of exponent field 1. `EXTENDED_NORMALS` lowers that by a
// binade -- exponent code 0 becomes one more binade of normals -- but *not*
// its mantissa-zero code: that one is the format's zero, and with the sign bit
// its NaN, as in every other binaryK format, so the extended binade's values
// start one step above it. With man_bits == 0 that step carries into the next
// binade, which is the right answer as well: a binade of a single code, spent
// on the zero, holds no value, and the format is `NORMALS` with extra steps.
template <class T>
CUDA_HOST_DEVICE_INLINE NormalRangeParamsT<T> make_normal_range_params(int exp_bits, int man_bits, int bias,
                                                                        SaturationMode saturation_mode,
                                                                        int reserved_codes,
                                                                        bool extended_normals = false)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    NormalRangeParamsT<T> p;
    p.saturation_mode = saturation_mode;

    const int man = (man_bits < F::MAN_BITS) ? man_bits : F::MAN_BITS; // the carrier holds no finer grid

    const int min_exponent_store = -(bias - 1) + F::BIAS - extended_normals;
    // The floor's own exponent field, which is the store's except where the
    // man_bits == 0 carry below moves it up a binade. The range test reads
    // this and not the store: at a store of 0 that carry lands on the
    // carrier's smallest normal, a floor like any other -- NORMALS' own at
    // the same bias -- where reading the store left the format with none; and
    // at TOP_FIELD it lands past the carrier's largest value, where NORMALS'
    // has none either and reading the store made the infinity a floor.
    const int floor_field = min_exponent_store + (extended_normals && man == 0);
    if (floor_field <= 0 || floor_field > F::TOP_FIELD)
    {
        // The floor lies outside the carrier's normal range, so no rounded
        // word can be below it: leave the underflow test dead rather than
        // fabricate a word for a value the carrier does not have. (Such a
        // format is one `mptorch.number`'s range check refuses or warns about.)
        p.min_num = 0u;
        p.half_num = 0u;
    }
    else
    {
        // `+`, not `|`: with man_bits == 0 the step is 1 << MAN_BITS, which
        // is the exponent field's own low bit, and the sum is the carry into
        // the next binade that the comment above describes. Above man_bits ==
        // 0 the mantissa field is zero and the two spellings agree.
        p.min_num = ((word_t)min_exponent_store << F::MAN_BITS) +
                    (extended_normals ? (word_t(1) << (F::MAN_BITS - man)) : word_t(0));
        // halved as a value, not as a word: decrementing the exponent field
        // is the same thing only while the field stays at 1 or above, and a
        // format whose floor is the carrier's smallest normal -- which the
        // range check warns about but does not refuse -- puts it one below,
        // where the word means something else entirely.
        T min_val = reinterpret_cast<const T &>(p.min_num);
        T half = min_val * T(0.5);
        p.half_num = reinterpret_cast<const word_t &>(half);
    }

    // On sword_t: a binade holds 2^52 codes in binary64, and `int` holds none
    // of them; in binary32 it is the `int` this always was.
    using sword_t = typename F::sword_t;
    const sword_t codes = (sword_t)1 << man;
    sword_t top_code = codes - 1 - reserved_codes; // the largest finite code, within its binade
    int top_field = (1 << exp_bits) - 1;           // and that binade's exponent field
    while (top_code < 0)
    {
        top_code += codes;
        --top_field;
    }
    const int top_exp = top_field - bias;
    if (top_field < 1 || top_exp < F::MIN_NORMAL_EXP)
        p.max_num = 0u;
    else if (top_exp > F::MAX_EXP)
        // A format whose largest finite value lies beyond the carrier's has
        // no finite input that overflows, so max_num is only ever read for a
        // non-finite one -- an infinity, or a rounding that carried into the
        // all-ones field -- and there SAT_FINITE and SAT_PROPAGATE still owe a
        // finite answer: the carrier's largest value on the format's grid.
        p.max_num = ((word_t)F::TOP_FIELD << F::MAN_BITS) | ((word_t)(codes - 1) << (F::MAN_BITS - man));
    else
        p.max_num = ((word_t)(top_exp + F::BIAS) << F::MAN_BITS) | ((word_t)top_code << (F::MAN_BITS - man));
    return p;
}

// What a magnitude below the format's smallest value becomes -- the only thing
// the rounding modes differ about down there.
//
// The two candidates are always the same pair, zero and that smallest value,
// so each mode picks between them exactly as it picks between zero and the
// smallest subnormal one region lower: nearest with the tie going to the even
// code (zero), nearest-away taking the value, the directed modes taking their
// own direction, round-to-odd taking the nonzero one -- it is the odd code
// where there is one, and never discarding the fact that the input was nonzero
// where there is not -- and stochastic taking it with probability |x| / it.
// The same four spellings as `clip_subnormal_range_exponent` and its `_up`
// twin, which are that region's.
//
// Only `NORMALS` and `EXTENDED_NORMALS` reach this. Under `SUBNORMALS`
// anything below the smallest normal took the subnormal branch, and rounding a
// normal can only raise its exponent, so the arm is dead there -- which is why
// the mode is a template parameter and each policy decides *inside* the
// branch: the paths that never underflow pay for none of it.
enum class UnderflowMode
{
    NEAREST_EVEN, // above half the smallest value, take it; at half, zero
    NEAREST_AWAY, // at or above half, take it
    AWAY,         // any nonzero magnitude takes it: toward-away, and to-odd
    ZERO,         // toward zero
    STOCHASTIC,   // take it with probability |x| / it
    // Nothing below the floor can reach the clip, so it carries no arm at all.
    // superfp is the case: its normal arm is entered only at or above
    // `normal_cutoff`, which a well-formed format puts at or above `min_exp`
    // (`normal_binades <= 2^exp_bits - 1`, which `mptorch.number` enforces),
    // and rounding a normal only ever raises its exponent. Spelling that here
    // rather than picking a policy that would never run is worth 1.12x of the
    // superfp split-mac SR GEMM: the arm is cold, but its constants and the
    // float compare it wants still cost that kernel registers (T5, and finding
    // G10 for why that kernel in particular).
    NONE,
};

// Saturates a value the normal arm has rounded, and floors it. P3109 rounds
// first and saturates second, which makes overflow a magnitude test: anything
// above max_num -- a rounding that landed on a reserved code, carried into the
// next binade, or carried a finite input all the way into the all-ones field
// -- is out of range, and becomes infinity under OVF_INF (SatNone) and max_num
// under the other two (SatFinite, SatPropagate: the value was finite, so
// SatPropagate has nothing to propagate). Non-finite inputs never get here:
// every cast sends them to saturate_nonfinite before it rounds.
//
// This form returns a magnitude; clip_normal_range_exponent below is this plus
// the sign, and the note there says which callers want which.
template <UnderflowMode U, class W>
CUDA_HOST_DEVICE_INLINE W clip_normal_range_magnitude(W old_num, W quantized_num, SaturationMode saturation_mode,
                                                      W min_num, W half_num, W max_num, W rand_prob = 0)
{
    using F = WordTraits<W>;
    W ax = old_num & F::ABS_MASK;
    if (ax == 0u)
        return 0u; // a zero input rounds to zero, and P3109's zero is unsigned

    quantized_num &= F::ABS_MASK;

    // handle overflow
    if (quantized_num > max_num)
        quantized_num = (saturation_mode == SaturationMode::OVF_INF) ? W(F::INF_BITS) : max_num;
    // handle underflow
    else if constexpr (U == UnderflowMode::NONE)
    {
        // nothing to handle: see the enum
    }
    else if constexpr (U == UnderflowMode::STOCHASTIC)
    {
        // SR asks whether the *input* is below the floor, where the other
        // policies ask the rounded word. For them the two agree: a value that
        // rounds up onto the floor lands on the answer the arm would give it.
        // SR's rounding is a draw, though, and the one that carried a value
        // onto the floor is the same word this arm would draw with, so asking
        // the rounded word took the floor with probability P(carry) +
        // P(no carry) * P(|x| > u * min) -- 1.0 for 0.75 of the floor at
        // man_bits 0, where P3109 says 0.75. Asking the input discards the
        // round below the floor and draws once.
        if (ax < min_num)
        {
            // |x| > u * min_val with u uniform in [0, 1), which is the same
            // draw as round_bitwise_stochastic's one step higher, written as a
            // multiply because the step down here is min_val rather than a
            // power of two (EXTENDED_NORMALS' floor is not one). The product
            // is one rounding wide, which is below the resolution `rand_bits`
            // asks for whenever that is under MAN_BITS + 1.
            using T = typename F::value_t;
            T min_val = reinterpret_cast<const T &>(min_num);
            T xf = reinterpret_cast<const T &>(ax);
            T u = (T)(rand_prob & F::MAN_MASK) * F::ulp_scale();
            quantized_num = (xf > u * min_val) ? min_num : W(0);
        }
    }
    else if (quantized_num < min_num)
    {
        if constexpr (U == UnderflowMode::NEAREST_EVEN)
            quantized_num = (ax > half_num) ? min_num : W(0);
        else if constexpr (U == UnderflowMode::NEAREST_AWAY)
            quantized_num = (ax >= half_num) ? min_num : W(0);
        else if constexpr (U == UnderflowMode::AWAY)
            quantized_num = min_num;
        else // ZERO
            quantized_num = 0u;
    }

    return quantized_num;
}

// The same, for a signed input: the magnitude above with the input's sign put
// back on it -- unless there is no magnitude to sign. A value that flushed to
// zero in the underflow arm, and a format whose max_num is zero, both arrive
// here with nothing to sign, and P3109's zero is unsigned. This is where the
// casts that round through the integer path get that; the mask costs a cmov
// and an or, where dropping the sign at the cast's return instead cost
// 1.05-1.23x (T2/T3).
//
// Splitting the two is worth more than it looks. cast_absolute_up and its
// three siblings pass |x|, so their sign bit is always clear and everything
// this function adds is dead -- but it is dead at runtime, not at compile
// time, and the compiler cannot see it. Handing them the magnitude form took
// the binaryK `RZ` GEMM from 1.13x of pre-T2 to 0.92x on a GPU, and the fma
// one to 0.87x: below where it started, because the sign work the directed
// modes had been doing for nothing goes with it (T3).
template <UnderflowMode U, class W>
CUDA_HOST_DEVICE_INLINE W clip_normal_range_exponent(W old_num, W quantized_num, SaturationMode saturation_mode,
                                                     W min_num, W half_num, W max_num, W rand_prob = 0)
{
    using F = WordTraits<W>;
    W magnitude = clip_normal_range_magnitude<U>(old_num, quantized_num, saturation_mode,
                                                 min_num, half_num, max_num, rand_prob);
    W sign = old_num & F::SIGN_MASK;
    return magnitude | (sign & -(W)(magnitude != 0));
}

// What an input whose exponent field is all ones -- an infinity or a NaN --
// becomes. A NaN passes through unchanged under every mode, payload included.
// So does an infinity, except under SAT_FINITE, whose contract is that every
// return value is finite: there it becomes the format's largest finite
// magnitude with the input's sign -- max_num, the same value an overflowing
// finite input saturates to under that mode. Every integer-path cast's
// NaN/inf arm reads this; the float-arithmetic fast paths reproduce it with
// their final select instead (see cast_binaryK_rne_fast). Uniform per format,
// and on the arm only non-finite inputs take, so it costs the finite ones
// nothing.
template <class W>
CUDA_HOST_DEVICE_INLINE W saturate_nonfinite(W target, SaturationMode saturation_mode, W max_num)
{
    using F = WordTraits<W>;
    bool is_inf = (target & F::MAN_MASK) == 0;
    return (is_inf && saturation_mode == SaturationMode::SAT_FINITE) ? ((target & F::SIGN_MASK) | max_num)
                                                                      : target;
}

// The two negations the directed modes need, both on the word rather than as
// `-x`, and neither of them for speed.
//
// `-x` is a float operation, and on the device that is `neg.f32`, which PTX
// does not require to hand back a NaN's payload: -(-NaN) canonicalizes, and
// the sign with it, so a negative NaN came out of cast_binaryK_up as a
// positive one -- on some builds. Which ones is a matter of how nvcc inlined
// the cast that day, which is the worst way for a documented contract to hold
// ("P3109's single NaN is whichever NaN came in", docs/source/concepts.rst).
// cast_superfp_up carried an explicit NaN arm for exactly this; an XOR flips
// the sign bit and touches nothing else, so neither twin needs one now.
// (`neg.f64` makes the same non-promise, so binary64 negates on the word too.)
//
// The second is also where a -0.0 can still be made after the clips have run:
// negating a magnitude that rounded to zero. Masking the flip with
// unsigned_zero_bits' test leaves that zero alone -- and leaves a NaN whole,
// since its word is nonzero after the shift.
template <class T>
CUDA_HOST_DEVICE_INLINE T flip_sign(T x)
{
    using word_t = typename FloatTraits<T>::word_t;
    word_t w = reinterpret_cast<const word_t &>(x) ^ FloatTraits<T>::SIGN_MASK;
    return reinterpret_cast<const T &>(w);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T negate_magnitude(T m)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    word_t w = reinterpret_cast<const word_t &>(m);
    w ^= F::SIGN_MASK & -(word_t)((w << 1) != 0);
    return reinterpret_cast<const T &>(w);
}
