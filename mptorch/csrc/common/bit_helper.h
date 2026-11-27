#pragma once

#include "modes.h"
#if !defined(__METAL_VERSION__)
#include <cmath>
#include <cstdint>
#include <type_traits>
#endif

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define CUDA_HOST_DEVICE_INLINE inline
#endif

// The third compiler. The MPS backend (csrc/mps/) compiles this header, and
// the other common/ headers its kernels include, as Metal Shading Language at
// run time (mps/prelude.metal supplies what MSL lacks from the C++ standard
// library). MSL is C++ with address spaces: every reference and pointer
// names the address space it points into, and a static data member must be
// `constant`. The two macros below supply those keywords on Metal and are
// empty everywhere else, so the host and CUDA compilers see no difference.
// Only code the device runs needs them. Host-only parts (the CPU GEMM's tile
// state in gemm_policy.h) and binary64 (Metal has no double) are behind
// `!defined(__METAL_VERSION__)` instead.
#if defined(__METAL_VERSION__)
#define MPTORCH_THREAD thread
#define MPTORCH_CONSTANT constant
#else
#define MPTORCH_THREAD
#define MPTORCH_CONSTANT
#endif

// MPTORCH_FAST_CAST enables the float-arithmetic fast paths of the casts in
// cast_binaryK.h and cast_superfp.h. They replace about 125 branchy integer
// instructions with about 14 branchless float ones, and they are exact only
// if every arithmetic operation they spell out is a separately rounded
// binary32 operation. The Veltkamp split at their core computes t = c * x
// and then t - (t - x), which is x rounded to fewer significand bits
// (Dekker's identity); a compiler that fuses the multiply and the
// subtraction into one FMA keeps t unrounded, and the identity fails (about
// 248 million of the 2^32 binary32 inputs came out wrong when nvcc was
// allowed to contract it).
//
// Device code always meets that condition: the paths use the _rn intrinsics
// (__fmul_rn and the like), which state their rounding and cannot be
// contracted, so the default below turns the paths on for the device pass of
// any .cu, whether built by setup.py or by hand (dev/benchmarks, temp/). Host
// code meets it only when the build promises no contraction, which is a
// compiler flag rather than a property of the source, so there the macro
// must come from the command line: setup.py defines it together with
// -ffp-contract=off, both or neither. It is deliberately not passed to nvcc,
// whose host pass compiles these same headers with a host compiler that gets
// no such flag. Metal has no _rn intrinsics either, and gets the macro the
// way the host does: mps/prelude.metal defines it next to
// `#pragma METAL fp contract(off)`, Metal's spelling of the flag.
#if !defined(MPTORCH_FAST_CAST) && defined(__CUDA_ARCH__)
#define MPTORCH_FAST_CAST 1
#endif

// The contract covers four fast paths (cast_binaryK_rne_fast,
// cast_superfp_rne_fast, and the overflow compares of the binaryK and
// superfp stochastic casts), and the CUDA device and the host take all four.
// One of them, cast_superfp_rne_fast, is spelled differently on each,
// because superfp's integer path is not as expensive as binaryK's: its
// supernormal arm, where ordinary data lies (normal_binades is usually 1 or
// 2), is a branch on the exponent field and five operations. The device's
// spelling still wins there, because the values of a warp nearly always lie
// in one region and its selects are nearly free; on a host the same
// spelling ran at 0.85-0.98x of the integer path on supernormal operands,
// and 0.77-0.84x under KAHAN. The host's own spelling (the integer path's
// shape with the fast path's arithmetic, measured at f9a6aa9) runs at
// 1.05-1.32x on supernormal operands in the superfp GEMMs (split, FMA,
// KAHAN), at parity in the cross-family one, whose products are binaryK's,
// and at 1.04-1.40x elementwise; on normal-region operands at 0.97x or
// better, except bfloat16 elementwise on one thread, at 0.90-0.95x, a path
// that is slow in every rounding mode for reasons of its own (see
// cast_superfp_rne_fast). The Apple GPU does not take this path, for
// correctness: Metal compiles the device's spelling, whose underflow test
// compares against fast_super_half, which for a format with no underflow
// region is a binary32 subnormal, and the GPU flushes a subnormal operand of
// a compare to zero (see "The Apple GPU's flush of binary32 subnormals"
// below).
#if defined(MPTORCH_FAST_CAST) && !defined(__METAL_VERSION__)
#define MPTORCH_FAST_CAST_SUPERFP_RNE 1
#endif

// The part of the contract a header can check. setup.py probes the compiler
// it is about to invoke, but torch builds with Python's own CFLAGS plus
// mptorch's, which the probe never sees, so an x87 target that gets this far
// should stop the build rather than quietly evaluate every intermediate in
// 80 bits and break the identity. Contraction leaves no such trace: neither
// GCC nor Clang defines a macro for -ffp-contract, which is why that part of
// the contract is a compiler flag backed by an exhaustive sweep instead.
// Metal has no <cfloat> and no excess precision to rule out.
#if defined(MPTORCH_FAST_CAST) && !defined(__CUDACC__) && !defined(__METAL_VERSION__)
#include <cfloat>
static_assert(FLT_EVAL_METHOD == 0,
              "MPTORCH_FAST_CAST needs float expressions evaluated in float; "
              "this target has excess precision (x87). Drop -DMPTORCH_FAST_CAST.");
#endif

// ---------------------------------------------------------------------------
// The carrier.
//
// Every cast rounds on the bit pattern of the IEEE binary format the value
// is computed in, its *carrier*: binary32 for float32, float16 and bfloat16
// tensors, and binary64 for float64 ones (a call that names
// carrier=torch.float64 hands the kernels float64 tensors). Everything that
// depends on the carrier (the word type, the field widths and masks, the
// bias, and the spellings of the operations the fast paths need) is read
// from FloatTraits<T>, so each helper below, and each params struct and cast
// in cast_binaryK.h, cast_superfp.h and cast_fixedpoint.h, is written once,
// as a template on the carrier. `BinaryKParams` and `SuperfpParams` are the
// binary32 instantiations. The one difference between the carriers beyond
// the constants is HAS_FAST_CAST: the float-arithmetic fast paths exist for
// binary32 only, so binary64 always takes the integer path.
//
// Helpers that see only a word (the bitwise rounds and the clips) deduce the
// carrier from the word's type (WordTraits below), so a caller passes the
// word it has and cannot pair a binary64 word with binary32's masks. Helpers
// that see a value deduce the carrier from the value's type.
//
// The templates compiled to byte-identical .text and .nv_fatbin with the
// hand-written binary32 code they replaced, on both compilers, and the four
// spellings below keep it that way. None of them changes the arithmetic, and
// each obvious alternative changed the generated code:
//
//   * The pun between a value and its word is `reinterpret_cast<const T &>`
//     on the variable, written where it is used, never inside a to_bits() or
//     from_bits() function. A call, however inlined, hides the variable's
//     address from the passes that promote it to a register before inlining,
//     and both a by-reference and a by-value wrapper changed the generated
//     kernels (one turned a select on the float into a select on the word,
//     the other grew a quantizer by 24 instructions). A constant is copied
//     into a local first, since device code cannot take a reference to a
//     static member.
//
//     The pun reads a word through a float glvalue, which the standard does
//     not allow (-Wstrict-aliasing warns about it); it is what these casts
//     have always done, and the exhaustive sweeps vouch for the results.
//     memcpy is the spelling the standard allows, and it was measured: it
//     changes 45 of the 128 device kernels and 371 host functions, not in
//     their results but in their instructions, so it fails the byte-identical
//     requirement.
//   * fabsf and copysignf are called by name, not through a trait. GCC folds
//     the builtins before inlining and a wrapper only after, which reordered
//     the host superfp SR quantizers.
//   * A shift on the word is written on sword_t, the carrier's signed word
//     type, not on the unsigned one: the binary32 code always shifted an
//     `int`, and the unsigned spelling changed what GCC generated. For
//     binary64 sword_t is wide enough for a 52-bit shift.
//   * A fast-path gate is a plain `if` on the params' fast_rne, which is a
//     constant false for a carrier without a fast path, and only the body it
//     guards is behind `if constexpr` (see cast_binaryK_nearest_even).
// ---------------------------------------------------------------------------
template <class T>
struct FloatTraits; // defined for the two carriers below, and nothing else

template <>
struct FloatTraits<float>
{
    using value_t = float;
    using word_t = uint32_t;
    using sword_t = int32_t;
    static constexpr MPTORCH_CONSTANT int MAN_BITS = 23;   // significand field
    static constexpr MPTORCH_CONSTANT int EXP_BITS = 8;    // exponent field
    static constexpr MPTORCH_CONSTANT int BIAS = 127;
    static constexpr MPTORCH_CONSTANT int WORD_BITS = 32;
    static constexpr MPTORCH_CONSTANT int MAX_EXP = 127;   // the largest finite value's exponent
    static constexpr MPTORCH_CONSTANT int INF_EXP = 128;   // the all-ones field, unbiased: infinities and NaNs
    static constexpr MPTORCH_CONSTANT int TOP_FIELD = 254; // the largest finite value's field
    static constexpr MPTORCH_CONSTANT int MIN_NORMAL_EXP = -126;
    // The lowest exponent a format's smallest value can have when the cast
    // reads the grid there from the input's exponent field: half of that
    // value must be a carrier normal, because every carrier subnormal has
    // exponent field 0, and the field alone cannot tell on which side of
    // half the smallest value such an input lies.
    static constexpr MPTORCH_CONSTANT int MIN_SIMULABLE_EXP = -125;
    static constexpr MPTORCH_CONSTANT word_t FIELD_MASK = 0xFFu;
    static constexpr MPTORCH_CONSTANT word_t SIGN_MASK = 0x80000000u;
    static constexpr MPTORCH_CONSTANT word_t ABS_MASK = 0x7FFFFFFFu;
    static constexpr MPTORCH_CONSTANT word_t MAN_MASK = 0x007FFFFFu;
    static constexpr MPTORCH_CONSTANT word_t INF_BITS = 0x7F800000u;
    static constexpr MPTORCH_CONSTANT word_t TIE = 0x00400000u;       // the tie when rounding to a power of two
    static constexpr MPTORCH_CONSTANT word_t BELOW_TIE = 0x003FFFFFu; // TIE - 1, just below it
    static constexpr MPTORCH_CONSTANT bool HAS_FAST_CAST = true;

    // 2^e for e in [MIN_NORMAL_EXP, MAX_EXP], exact by construction, without
    // pulling ldexp into a header that CUDA device code includes.
    CUDA_HOST_DEVICE_INLINE static value_t pow2(int e)
    {
        word_t bits = (word_t)(e + BIAS) << MAN_BITS;
        return reinterpret_cast<const MPTORCH_THREAD value_t &>(bits);
    }

    // rn_add, rn_sub and rn_mul are the operations the Veltkamp split needs
    // rounded separately. On the device that is what __fadd_rn and its
    // siblings guarantee. On the host the plain operators already are IEEE
    // binary32 operations, and only contraction into an FMA could merge two
    // of them, which MPTORCH_FAST_CAST's contract forbids, so the shim is the
    // operator itself, named so that the requirement stays visible at the
    // call site. (fma_f32 in gemm_policy.h is the opposite case: the one
    // place an FMA is intended, spelled out so that no flag can take it
    // away.)
    //
    // fmin_nonneg and fmax_nonneg are min and max as the fast paths use
    // them: on a magnitude clamped against a positive bound, never on a
    // negative operand. Under that precondition the ternary and fminf/fmaxf
    // agree everywhere, NaN included: every compare against a NaN is false,
    // so both spellings return the bound, and the fast paths' final select
    // overrides that result anyway.
    //
    // There are two spellings because each compiler turns exactly one of
    // them into a single instruction: GCC will not turn fminf into minss
    // (their NaN results differ in general) and emits a libm call in the
    // middle of the GEMM's inner loop, which cost the whole CPU GEMM 1.12x;
    // nvcc will not turn the ternary into min.f32 and emits a compare and a
    // select (setp and selp).
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

// Metal has no double, so the binary64 carrier, and everything that names
// it, exists on the host and CUDA only.
#if !defined(__METAL_VERSION__)
template <>
struct FloatTraits<double>
{
    using value_t = double;
    using word_t = uint64_t;
    using sword_t = int64_t;
    static constexpr MPTORCH_CONSTANT int MAN_BITS = 52;
    static constexpr MPTORCH_CONSTANT int EXP_BITS = 11;
    static constexpr MPTORCH_CONSTANT int BIAS = 1023;
    static constexpr MPTORCH_CONSTANT int WORD_BITS = 64;
    static constexpr MPTORCH_CONSTANT int MAX_EXP = 1023;
    static constexpr MPTORCH_CONSTANT int INF_EXP = 1024;
    static constexpr MPTORCH_CONSTANT int TOP_FIELD = 2046;
    static constexpr MPTORCH_CONSTANT int MIN_NORMAL_EXP = -1022;
    static constexpr MPTORCH_CONSTANT int MIN_SIMULABLE_EXP = -1021;
    static constexpr MPTORCH_CONSTANT word_t FIELD_MASK = 0x7FFu;
    static constexpr MPTORCH_CONSTANT word_t SIGN_MASK = 0x8000000000000000u;
    static constexpr MPTORCH_CONSTANT word_t ABS_MASK = 0x7FFFFFFFFFFFFFFFu;
    static constexpr MPTORCH_CONSTANT word_t MAN_MASK = 0x000FFFFFFFFFFFFFu;
    static constexpr MPTORCH_CONSTANT word_t INF_BITS = 0x7FF0000000000000u;
    static constexpr MPTORCH_CONSTANT word_t TIE = 0x0008000000000000u;
    static constexpr MPTORCH_CONSTANT word_t BELOW_TIE = 0x0007FFFFFFFFFFFFu;
    // The integer path only. A binary64 fast path would need measuring of
    // its own, since FP64 throughput is a fraction of the float rate on
    // consumer GPUs, and nothing has needed one.
    static constexpr MPTORCH_CONSTANT bool HAS_FAST_CAST = false;

    CUDA_HOST_DEVICE_INLINE static value_t pow2(int e)
    {
        word_t bits = (word_t)(e + BIAS) << MAN_BITS;
        return reinterpret_cast<const MPTORCH_THREAD value_t &>(bits);
    }

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
#endif // !__METAL_VERSION__

// The constants above are written out as they appear in IEEE 754 rather than
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
#if !defined(__METAL_VERSION__)
static_assert(float_traits_consistent<double>(), "FloatTraits<double> disagrees with itself");

// The carrier a tensor of scalar type S is rounded in: binary64 for double,
// and binary32 for everything else (float, and the 16-bit types, whose values
// binary32 holds exactly and whose casts are binary32's).
template <class S>
using carrier_t = std::conditional_t<std::is_same_v<S, double>, double, float>;
#endif

// The carrier whose word type is W. Defined for the two carriers' words only,
// so that a word of any other type (an `int`, or a word of the wrong width)
// fails to compile instead of meeting binary32's masks.
template <class W>
struct CarrierOfWord;
template <>
struct CarrierOfWord<uint32_t>
{
    using type = float;
};
#if !defined(__METAL_VERSION__)
template <>
struct CarrierOfWord<uint64_t>
{
    using type = double;
};
#endif
template <class W>
using WordTraits = FloatTraits<typename CarrierOfWord<W>::type>;

// Turns a -0.0 word into +0.0 and leaves every other word alone. IEEE P3109
// has one zero, code point 0, and it is unsigned: in a signed format, binaryK
// or superfp, the code that would hold -0.0 is the format's NaN. So a result
// that rounds to zero is +0.0 whatever the sign of the value that got there,
// a -0.0 input included: the value simulated is the format's, not the
// carrier's.
//
// A mask on the word, not a select: g++ compiles `((w << 1) == 0) ? 0u : w`
// to a branch and this to a cmov, and vectorizes this form in a loop. A
// NaN's word is nonzero after the shift, so the mask keeps it whole, payload,
// signalling bit and all.
//
// The casts apply this where a zero is made, not on the way out: the clip
// functions below drop the sign along with the magnitude, superfp's region
// arms return a bare 0, and the one zero that can still be signed afterwards
// (a directed mode negating a magnitude that rounded to zero) is handled by
// negate_magnitude at the end of this file. Clearing the sign once at every
// cast's return instead cost the elementwise casts 1.05-1.23x, since that is
// a float operation on every value rather than a word operation on the few
// that reach a zero.
template <class W>
CUDA_HOST_DEVICE_INLINE W unsigned_zero_bits(W w)
{
    return w & -(W)((w << 1) != 0);
}

// Rounds the carrier word `target` to man_bits mantissa bits, to nearest,
// ties to even, in integer arithmetic on the word. Adding the tie (half the
// dropped range) and clearing the dropped bits rounds to nearest, and a
// carry out of the mantissa runs into the exponent field, which gives the
// correct next value. On an exact tie the add rounded up; clearing the kept
// lowest bit then makes the result even either way: it undoes the round-up
// when it landed on an odd mantissa, and changes nothing when it landed on
// an even one. With man_bits == 0 the kept significand is the implicit 1,
// which is odd, so a tie always rounds up (the carry lands in the exponent).
// A man_bits at or above the carrier's makes this the identity. The casts
// call this with a per-value man_bits (exp_diff) in their subnormal arms,
// which is why this spelling is used beside the precomputed one below.
// man_bits must be at least MAN_BITS - WORD_BITS + 2 (-7 in binary32, -10 in
// binary64), here and in the nearest-away, up, down and odd helpers below
// that take it: below that the mask's shift reaches the word's width, or its
// `- 1` overflows, which is undefined behaviour even when the result is then
// discarded. The casts never pass less than -1: they skip the call for a
// value that underflows, or, rounding up and to odd, pass at least 0.
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
    W odd = (man_bits == 0) ? 0 : 1; // man_bits 0: implicit 1 is odd, so tie up
    return quantized & ~((is_tie & odd) << (F::MAN_BITS - man_bits));
}

// Rounds to zero mantissa bits, that is, to the nearest power of two, with
// ties to the even exponent. This is the round for a man_bits == 0 format and
// for superfp's supernormal region, whose codes are powers of two, so "even"
// is a property of the exponent field rather than of a mantissa. On a tie
// the add carries into the exponent; the last line steps the field back down
// when that put it on an odd exponent.
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_nearest_even(W target)
{
    using F = WordTraits<W>;
    W tie = F::TIE;
    W quantized = (target + tie) & ~F::MAN_MASK;
    W is_tie = (target & F::MAN_MASK) == tie;
    return quantized - ((is_tie << F::MAN_BITS) & ~quantized);
}

// Precomputed-parameter forms. The bitwise helpers come in two spellings:
// one derives its mask and tie from man_bits on every call, and one reads
// them from a RoundParamsT built once per format. A GEMM Multiplier or Adder
// (gemm_policy.h) builds the params when it is constructed and reuses them
// for a whole launch (up to M*N*K calls); the elementwise quantizers build
// them once per tensor. The params structs are deliberately flat, with no
// nested aggregates: a nested struct copied into each of the GEMM kernel's
// instantiations took nvcc's cicc from 18 s to ten minutes. Stochastic
// rounding has precomputed constants of its own (StochasticT, stochastic.h).
//
// Which helpers keep the spelling that takes the format depends on whether
// their integer argument is always a format constant:
//
//   round_bitwise_*   both forms. The casts call the one taking man_bits with
//                     a per-value exp_diff in their subnormal branches, so it
//                     is used in the extension, not only in the sweeps.
//   clip_*            precomputed only. The forms that take the format are
//                     in dev/benchmarks/sweeps/reference_casts.h, as part of
//                     the reference the exhaustive sweeps check these
//                     against.

// The constants of round_bitwise_nearest_even(target, man_bits), which the
// precomputed nearest-away, up, down and odd overloads below reuse (they
// share the bypass, mask and tie formulas). Not valid for man_bits == 0,
// which uses the zero-argument overload above, a different computation.
template <class T>
struct RoundParamsT
{
    using word_t = typename FloatTraits<T>::word_t;
    bool bypass; // man_bits >= MAN_BITS: the round is an identity
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

// The precomputed form of round_bitwise_nearest_even(target, man_bits), for
// man_bits >= 1 (where the `odd` term is always 1 and drops out).
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

// Rounds to nearest, ties away from zero: adding the tie and truncating is
// exactly that, since a tie then carries into the kept lowest bit.
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

// The precomputed form of round_bitwise_nearest_away above, from
// RoundParamsT (it does not need the `shift` field).
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_nearest_away(W target, bool bypass, W mask, W tie)
{
    if (bypass)
        return target;
    W add_r = target + tie;
    return add_r & ~mask;
}

// Rounds up, toward positive infinity: a positive word with any dropped bit
// set moves to the next grid point (adding the full mask and truncating),
// and a negative one truncates toward zero, which is upward for it.
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

// The precomputed form of round_bitwise_up above: `sign` depends on the
// value, not on the format, so only bypass and mask, which depend on
// man_bits, are precomputed.
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

// Rounds down, toward negative infinity: the mirror image of
// round_bitwise_up.
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

// The precomputed form of round_bitwise_down above; see round_bitwise_up's.
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

// Rounds to odd: truncates toward zero to man_bits, then ORs the sticky bit
// (whether any dropped bit was set) into the kept lowest bit, so an inexact
// result always has an odd last bit, and a later round to nearest at fewer
// bits cannot suffer from double rounding. When man_bits == 0 the kept "bit"
// is the exponent's lowest bit (there is no stored mantissa), so a set
// sticky bit lands in the exponent.
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

// The precomputed form of round_bitwise_odd above. lsb = mask + 1, so it
// needs no field of its own.
template <class W>
CUDA_HOST_DEVICE_INLINE W round_bitwise_odd(W target, bool bypass, W mask)
{
    if (bypass)
        return target;
    W lsb = mask + 1u;
    W sticky = (target & mask) != 0;
    return (target & ~mask) | (sticky * lsb);
}

// The constant of clip_subnormal_range_exponent: min_exponent_store is the
// carrier's exponent field for the format's smallest subnormal,
// 2^(min_exp - man_bits) with min_exp = 1 - bias. The reference spelling in
// dev/benchmarks/sweeps/reference_casts.h also takes exp_bits, for symmetry
// with its other clip functions, but does not use it.
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

// Applies the floor to a word the subnormal arm has rounded, for the nearest
// modes and round-down. old_num is the input word and quantized_num its
// rounding to the format's subnormal grid. A rounded word whose exponent
// field is below the smallest subnormal's has underflowed. One exactly one
// field below is the arm's exp_diff == -1 case (a value above half the
// smallest subnormal, which rounds up to it, where the round at -1 bits can
// leave the exponent field one short), and is lifted onto the smallest
// subnormal with the input's sign; anything lower becomes +0.0.
template <class W>
CUDA_HOST_DEVICE_INLINE W clip_subnormal_range_exponent(W old_num, W quantized_num, int min_exponent_store)
{
    using F = WordTraits<W>;
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = (int)((quantized_num >> F::MAN_BITS) & F::FIELD_MASK);

    W old_sign = old_num & F::SIGN_MASK;
    // underflow, or a round up to the smallest subnormal
    if (quantized_exponent_store < min_exponent_store)
    {
        int offset = (quantized_exponent_store == (min_exponent_store - 1));
        quantized_num += offset * (W(1) << F::MAN_BITS);
        quantized_num |= old_sign;
        quantized_num *= offset; // offset == 0 is the underflow, and it is +0.0
    }
    else
        // A word with no magnitude but a sign bit (a negative value the
        // rounding took to zero) reaches this arm only in a format whose
        // min_exponent_store is at or below zero, where the underflow test
        // above cannot catch it. P3109's zero is unsigned either way.
        quantized_num = unsigned_zero_bits(quantized_num);

    return quantized_num;
}

// The _up variant, for round-up and round-to-odd: it clamps an underflowing
// nonzero value to the smallest subnormal rather than to zero. Whether the
// value is nonzero is a question about the input, not about the rounded
// word: a -0.0 input truncates to a lone sign bit, which is not 0 and would
// be pushed out to minus the smallest subnormal, and a carrier subnormal
// input can truncate to exactly 0 although it was nonzero. Under these modes
// only a zero input rounds to zero, so a zero input gets an unsigned zero
// here, whatever its rounded word. It uses the same SubnormalRangeParams
// (make_subnormal_range_params) as clip_subnormal_range_exponent above.
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
        // magnitude, in a format whose underflow test cannot catch it
        quantized_num = unsigned_zero_bits(quantized_num);

    return quantized_num;
}

// The constants of the normal-range clip below, the form every cast in
// cast_binaryK.h and cast_superfp.h calls.
template <class T>
struct NormalRangeParamsT
{
    using word_t = typename FloatTraits<T>::word_t;
    SaturationMode saturation_mode; // kept as is: selects what an overflow becomes
    word_t min_num;                 // the bottom of the normal range (the floor)
    word_t half_num;                // half of it: the round-to-nearest boundary
    word_t max_num;                 // the largest finite magnitude, as a word
};

// Builds the normal-range constants of a format with exp_bits, man_bits and
// bias, in the carrier T. The top of the range is counted in code points,
// the way IEEE P3109 assigns them (arXiv:2606.04028, Section VII): of the
// top binade's 2^man_bits codes, the last `reserved_codes` hold no finite
// value, and max_num is the largest code left. binaryK and superfp both
// reserve P3109's codes: +infinity except under SAT_FINITE (that is, in the
// extended domain, SAT_FINITE_EXTENDED included) and, in an unsigned format,
// the NaN above it.
//
// With man_bits <= 1 the reserved codes can outnumber the top binade's, and
// the largest finite code is then one or two binades lower. A format whose
// largest finite code would have exponent field 0 (no finite normal value at
// all, which P3109 rules out by requiring at least three bits) gets
// max_num = 0, and every nonzero result of the normal arm overflows. So does
// a format whose largest finite value is below the carrier's normal range.
//
// The bottom is a code point too, and the subnormals mode says which one.
// SUBNORMALS and NORMALS put the floor at the smallest normal, whose code is
// the first one with exponent field 1. EXTENDED_NORMALS lowers it by a
// binade (exponent field 0 becomes one more binade of normals), but not to
// that binade's code with a zero mantissa: that code is the format's zero,
// and with the sign bit set its NaN, as in every other binaryK format, so the
// extended binade's values start one step above it. With man_bits == 0 that
// step carries into the next binade, which is also the right answer: a
// binade of a single code, spent on the zero, holds no value, and the format
// is NORMALS.
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

    const int man = (man_bits < F::MAN_BITS) ? man_bits : F::MAN_BITS; // clamp

    // man is clamped to the carrier's mantissa width: the carrier has no
    // finer grid.
    const int min_exponent_store = -(bias - 1) + F::BIAS - extended_normals;
    // The floor's own exponent field, which is min_exponent_store except
    // where the man_bits == 0 carry below moves it up a binade. The range
    // test reads this field, not the store: at a store of 0 the carry lands
    // on the carrier's smallest normal, a floor like any other (NORMALS' own
    // floor at the same bias), where reading the store would leave the
    // format with none; and at TOP_FIELD it lands past the carrier's largest
    // value, where NORMALS has no floor either and reading the store would
    // make the infinity a floor.
    const int floor_field = min_exponent_store + (extended_normals && man == 0);
    if (floor_field <= 0 || floor_field > F::TOP_FIELD)
    {
        // The floor lies outside the carrier's normal range, so no rounded
        // word can be below it: leave the underflow test unable to fire
        // rather than make up a word for a value the carrier does not have.
        // (`mptorch.number`'s range check refuses or warns about such a
        // format.)
        p.min_num = 0u;
        p.half_num = 0u;
    }
    else
    {
        // `+`, not `|`: with man_bits == 0 the step is 1 << MAN_BITS, the
        // exponent field's own lowest bit, and the sum is the carry into the
        // next binade described above. For man_bits > 0 the mantissa field
        // is zero and the two spellings agree.
        p.min_num = ((word_t)min_exponent_store << F::MAN_BITS) +
                    (extended_normals ? (word_t(1) << (F::MAN_BITS - man)) : word_t(0));
        // Halved on the word, in two cases. Above the carrier's smallest
        // normal, halving decrements the exponent field. A format whose floor
        // is that smallest normal (which the range check warns about but does
        // not refuse) has a subnormal half: the floor's significand, implicit
        // bit included, shifted down one place into field 0, with the bit
        // shifted out rounded to nearest even. That bit, when set, is an
        // exact tie, so the result rounds up only when the kept part is odd,
        // which makes it even, and a carry out of the significand lands on
        // the smallest normal, which is also the right answer. Both cases
        // equal what `min_val * 0.5` rounds to. The halving is done on the
        // word because an Apple GPU flushes a subnormal product to zero, and
        // the MPS backend builds these constants on the GPU (see
        // MPTORCH_THREAD above).
        const word_t floor_bits = p.min_num >> F::MAN_BITS;
        if (floor_bits >= 2)
            p.half_num = p.min_num - (word_t(1) << F::MAN_BITS);
        else
        {
            const word_t sig = (p.min_num & F::MAN_MASK) | (word_t(1) << F::MAN_BITS);
            p.half_num = (sig >> 1) + (sig & (sig >> 1) & 1u);
        }
    }

    // On sword_t: a binade of binary64 holds 2^52 codes, more than an `int`
    // can count; in binary32 sword_t is the `int` this always was.
    using sword_t = typename F::sword_t;
    const sword_t codes = (sword_t)1 << man;
    sword_t top_code = codes - 1 - reserved_codes; // the top finite code, within its binade
    int top_field = (1 << exp_bits) - 1;           // that binade's field
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
        // no finite input that overflows, so max_num is only read for a
        // non-finite one (an infinity, or a rounding that carried into the
        // all-ones field), and there SAT_FINITE and SAT_PROPAGATE still need
        // a finite answer: the carrier's largest value on the format's grid.
        p.max_num = ((word_t)F::TOP_FIELD << F::MAN_BITS) | ((word_t)(codes - 1) << (F::MAN_BITS - man));
    else
        p.max_num = ((word_t)(top_exp + F::BIAS) << F::MAN_BITS) | ((word_t)top_code << (F::MAN_BITS - man));
    return p;
}

// ---------------------------------------------------------------------------
// The Apple GPU's flush of binary32 subnormals.
//
// The MPS backend's GPU flushes binary32 subnormals to zero, operands and
// results, in every arithmetic operation and every compare, under every
// Metal math mode (safe included). The casts round on the word, so the
// flush matters only in the few places where they do float arithmetic on,
// or compare, a value that can be a carrier subnormal. Each of those is
// written through one of the macros or helpers below, which on the host and
// CUDA are the plain expression, and on Metal a version the flush cannot
// reach.
// ---------------------------------------------------------------------------

// The sign tests the casts branch on: `x < 0` (a negative input to an
// unsigned format returns 0) and `x >= 0` (which magnitude helper a directed
// mode hands |x| to). On the host and CUDA they are the float compares they
// look like.
//
// On an Apple GPU they cannot be: there `-2^-140 < 0` is false and
// `-2^-140 >= 0` is true. A negative subnormal would then take the
// nonnegative arm of cast_binaryK_up: cast_absolute_up would round a word
// with the sign bit set and return minus the smallest subnormal, a result
// below the input under round-up. The Metal spelling therefore reads the
// word, which the flush cannot reach, and keeps IEEE's answer for every
// other input: a NaN is neither negative nor nonnegative, and -0.0 is
// nonnegative. Everything else these casts do to a subnormal input,
// stochastic rounding included (stochastic.h), is integer arithmetic on its
// word; mps/gemm.metal states what the flush can still change, which is
// only the GEMM's arithmetic between its roundings.
#if defined(__METAL_VERSION__)
inline bool metal_is_negative(float x)
{
    const uint32_t w = as_type<uint32_t>(x);
    return w - 0x80000001u < 0x7F800000u; // -denorm_min .. -inf
}
inline bool metal_is_nonnegative(float x)
{
    const uint32_t w = as_type<uint32_t>(x);
    return w <= 0x7F800000u || w == 0x80000000u; // +0.0 .. +inf, and -0.0
}
#define MPTORCH_IS_NEGATIVE(x) metal_is_negative(x)
#define MPTORCH_IS_NONNEGATIVE(x) metal_is_nonnegative(x)
#else
#define MPTORCH_IS_NEGATIVE(x) ((x) < 0)
#define MPTORCH_IS_NONNEGATIVE(x) ((x) >= 0)
#endif

// Whether a cast returns its unsigned format's zero for a negative input:
// the first test of every binaryK, superfp and fixed-point cast. On the host
// the format's signedness is tested first. GCC branches on the operands in
// the order they are written, so with the sign first a cast branched on the
// sign of its input before reading the policy's is_signed, which does not
// change over the loop, and the sign of a product or sum of N(0, 1) values
// is a coin flip: about one branch misprediction per CPU split-mac step,
// which made the RNE split GEMM 3.4x slower than its two casts account for
// (fixed in eafa9aa). The elementwise quantizers take is_signed as a
// template constant, which folds away in either order. The device and Metal
// keep the sign first: nvcc predicates the test, and the CUDA RNE GEMMs
// timed the same either way, so their code was left unchanged.
#if defined(__CUDA_ARCH__) || defined(__METAL_VERSION__)
#define MPTORCH_NEGATIVE_TO_UNSIGNED(x, is_signed) (MPTORCH_IS_NEGATIVE(x) && !(is_signed))
#else
#define MPTORCH_NEGATIVE_TO_UNSIGNED(x, is_signed) (!(is_signed) && MPTORCH_IS_NEGATIVE(x))
#endif

// IEEE binary32 a * b, a / b and a - b, rounded to nearest even, with
// subnormal operands and results, for the block formats' decode and pack
// (block_decode.h's MPTORCH_BLOCK_MUL, _DIV and _SUB). These multiply by
// scales as small as 2^-127, divide by tensor scales and subtract a decoded
// value from its input, where the float unit's flush would give a result
// different from the CPU's. The casts need none of these: they round on the
// word.
//
// Each returns the float unit's result wherever no subnormal is involved,
// as an operand or as the result, since the unit is IEEE there (its division
// too, in safe math: no mismatch against the CPU in 2^22 random pairs of
// normal values), and otherwise rounds the exact result on the word.
// metal_round_magnitude is that rounding: it rounds sig * 2^e2, plus a
// nonzero fraction of sig's last unit when `sticky` is set, to nearest even
// on binary32's grid (2^-149 at the bottom, infinity past the top), and
// returns the word of the magnitude. A caller that sets sticky leaves at
// least two bits of sig below the grid's unit, which is what makes the
// sticky fraction matter only for ties.
#if defined(__METAL_VERSION__)
inline int metal_leading_bit64(uint64_t x)
{
    const uint32_t hi = uint32_t(x >> 32);
    return hi != 0 ? 63 - int(metal::clz(hi)) : 31 - int(metal::clz(uint32_t(x)));
}

inline uint32_t metal_round_magnitude(uint64_t sig, int e2, bool sticky)
{
    const int X = metal_leading_bit64(sig) + e2; // the value's binade
    const int lsb = X - 23 < -149 ? -149 : X - 23;
    const int shift = lsb - e2;
    uint64_t q;
    if (shift <= 0)
        q = sig << uint32_t(-shift);
    else if (shift > 64)
        q = 0; // sig < 2^64 is then below half the grid's unit
    else if (shift == 64)
        q = sig > (uint64_t(1) << 63) || (sig == (uint64_t(1) << 63) && sticky) ? 1u : 0u;
    else
    {
        q = sig >> uint32_t(shift);
        const uint64_t rest = sig & ((uint64_t(1) << uint32_t(shift)) - 1u);
        const uint64_t tie = uint64_t(1) << uint32_t(shift - 1);
        if (rest > tie || (rest == tie && (sticky || (q & 1u) != 0)))
            ++q;
    }
    if (lsb == -149)
        return uint32_t(q); // a subnormal, or 2^-126 for q = 2^23
    int e = X;
    if (q == (uint64_t(1) << 24))
    {
        q >>= 1;
        ++e;
    }
    if (e > 127)
        return 0x7F800000u;
    return (uint32_t(e + 127) << 23) | (uint32_t(q) & 0x007FFFFFu);
}

// Splits a finite nonzero binary32 word into sig * 2^e, with sig an integer
// of at most 24 bits.
inline void metal_split_finite(uint32_t w, thread uint64_t &sig, thread int &e)
{
    const uint32_t field = (w >> 23) & 0xFFu;
    sig = field == 0 ? uint64_t(w & 0x007FFFFFu) : uint64_t((w & 0x007FFFFFu) | 0x00800000u);
    e = field == 0 ? -149 : int(field) - 150;
}

inline bool metal_is_subnormal(uint32_t w)
{
    return (w & 0x7F800000u) == 0 && (w & 0x007FFFFFu) != 0;
}

inline float metal_mul_rn(float a, float b)
{
    const float r = a * b;
    const uint32_t aw = as_type<uint32_t>(a), bw = as_type<uint32_t>(b);
    const uint32_t rf = (as_type<uint32_t>(r) >> 23) & 0xFFu;
    const bool zero = (aw & 0x7FFFFFFFu) == 0 || (bw & 0x7FFFFFFFu) == 0;
    if (!metal_is_subnormal(aw) && !metal_is_subnormal(bw) && (rf >= 2 || zero))
        return r;
    const uint32_t sign = (aw ^ bw) & 0x80000000u;
    if ((aw & 0x7FFFFFFFu) > 0x7F800000u || (bw & 0x7FFFFFFFu) > 0x7F800000u)
        return r; // a NaN
    if ((aw & 0x7FFFFFFFu) == 0x7F800000u || (bw & 0x7FFFFFFFu) == 0x7F800000u)
        return as_type<float>(sign | 0x7F800000u); // an infinity times a subnormal
    if (zero)
        return as_type<float>(sign);
    uint64_t sa, sb;
    int ea, eb;
    metal_split_finite(aw, sa, ea);
    metal_split_finite(bw, sb, eb);
    return as_type<float>(sign | metal_round_magnitude(sa * sb, ea + eb, false));
}

inline float metal_div_rn(float a, float b)
{
    const float r = a / b;
    const uint32_t aw = as_type<uint32_t>(a), bw = as_type<uint32_t>(b);
    const uint32_t rf = (as_type<uint32_t>(r) >> 23) & 0xFFu;
    const uint32_t am = aw & 0x7FFFFFFFu, bm = bw & 0x7FFFFFFFu;
    if (!metal_is_subnormal(aw) && !metal_is_subnormal(bw) && (rf >= 2 || am == 0 || bm == 0x7F800000u))
        return r;
    const uint32_t sign = (aw ^ bw) & 0x80000000u;
    if (am > 0x7F800000u || bm > 0x7F800000u)
        return r; // a NaN
    if (am == 0x7F800000u || bm == 0)
        return as_type<float>(sign | 0x7F800000u); // with the other subnormal
    if (bm == 0x7F800000u || am == 0)
        return as_type<float>(sign);
    uint64_t sa, sb;
    int ea, eb;
    metal_split_finite(aw, sa, ea);
    metal_split_finite(bw, sb, eb);
    // Both significands have at most 24 bits, so the numerator, shifted to
    // the top of 64 bits, leaves a quotient of at least 39 bits: the 24 kept
    // and the guard bits the rounding needs.
    const int la = metal_leading_bit64(sa), lb = metal_leading_bit64(sb);
    sa <<= uint32_t(63 - la);
    sb <<= uint32_t(23 - lb);
    const uint64_t q = sa / sb;
    const bool sticky = sa - q * sb != 0;
    return as_type<float>(sign | metal_round_magnitude(q, (ea - (63 - la)) - (eb - (23 - lb)), sticky));
}

inline float metal_sub_rn(float a, float b)
{
    const float r = a - b;
    const uint32_t aw = as_type<uint32_t>(a), bw = as_type<uint32_t>(b);
    const uint32_t rf = (as_type<uint32_t>(r) >> 23) & 0xFFu;
    // Normal operands whose difference is below 2^-126 give it exactly, as a
    // subnormal, which the unit flushes to zero; a zero from equal operands
    // is the unit's to sign.
    if (!metal_is_subnormal(aw) && !metal_is_subnormal(bw) && (rf >= 1 || a == b))
        return r;
    if ((aw & 0x7FFFFFFFu) >= 0x7F800000u || (bw & 0x7FFFFFFFu) >= 0x7F800000u)
        return r; // a NaN, or an infinity less a subnormal or the reverse
    uint64_t sa = 0, sb = 0;
    int ea = -149, eb = -149;
    if ((aw & 0x7FFFFFFFu) != 0)
        metal_split_finite(aw, sa, ea);
    if ((bw & 0x7FFFFFFFu) != 0)
        metal_split_finite(bw, sb, eb);
    const int e = ea < eb ? ea : eb;
    // More than 39 binades apart, the smaller operand is far below half the
    // larger's unit, and the unit's result, the larger operand with the
    // flushed one dropped, is the correctly rounded one.
    if (ea - e > 39 || eb - e > 39)
        return r;
    const int64_t va = int64_t(sa << uint32_t(ea - e)), vb = int64_t(sb << uint32_t(eb - e));
    const int64_t d = ((aw >> 31) != 0 ? -va : va) - ((bw >> 31) != 0 ? -vb : vb);
    if (d == 0)
        return 0.0f;
    const uint32_t sign = d < 0 ? 0x80000000u : 0u;
    return as_type<float>(sign | metal_round_magnitude(uint64_t(d < 0 ? -d : d), e, false));
}
#endif

// What a magnitude below the format's smallest value becomes, which is the
// only question the rounding modes answer differently down there.
//
// The two candidates are always zero and that smallest value, so each mode
// chooses between them exactly as it chooses between zero and the smallest
// subnormal one region lower: nearest-even gives the tie to the even code
// (zero), nearest-away takes the value, the directed modes take their own
// direction, round-to-odd takes the nonzero candidate (the odd code where
// there is one; where there is not, it still keeps the fact that the input
// was nonzero), and stochastic rounding takes the value with probability
// |x| divided by it. These are the rules that clip_subnormal_range_exponent
// and its _up variant implement for the subnormal region, all but the last:
// stochastic.h's casts make stochastic rounding's choice themselves, so it
// has no member here.
//
// Only NORMALS and EXTENDED_NORMALS reach this. Under SUBNORMALS anything
// below the smallest normal took the subnormal branch, and rounding a normal
// can only raise its exponent, so the clip's underflow arm cannot run there.
// That is why the mode is a template parameter and each rule is chosen
// inside the branch: the paths that never underflow pay nothing for it.
enum class UnderflowMode
{
    NEAREST_EVEN, // above half the smallest value, take it; at half, zero
    NEAREST_AWAY, // at or above half, take it
    AWAY,         // any nonzero magnitude takes it: toward-away, and to-odd
    ZERO,         // toward zero
    // Nothing below the floor can reach the clip, so it has no underflow arm
    // at all. superfp is this case: its normal arm is entered only at or
    // above `normal_cutoff`, which a well-formed format puts at or above
    // `min_exp` (`normal_binades <= 2^exp_bits - 1`, which `mptorch.number`
    // enforces), and rounding a normal only ever raises its exponent. Saying
    // so here, rather than choosing a rule that would never run, is worth
    // 1.12x on the superfp split-mac SR GEMM: the arm never runs, but its
    // constants and the float compare it needs still cost that
    // register-bound kernel registers.
    NONE,
};

// Saturates a value the normal arm has rounded, applies the floor, and
// returns the magnitude. old_num is the input word, quantized_num its
// rounding to the format's grid. P3109 rounds first and saturates second,
// which makes overflow a magnitude test: anything above max_num (a rounding
// that landed on a reserved code, carried into the next binade, or carried a
// finite input all the way into the all-ones field) is out of range, and
// becomes infinity under SAT_NONE (SatNone) and max_num under the other
// three modes (SatFinite in either domain, and SatPropagate, since the value
// was finite and there is nothing to propagate). Non-finite inputs never get
// here: every cast sends them to saturate_nonfinite before it rounds.
//
// clip_normal_range_exponent below is this function plus the sign, and its
// note says which callers want which.
template <UnderflowMode U, class W>
CUDA_HOST_DEVICE_INLINE W clip_normal_range_magnitude(W old_num, W quantized_num, SaturationMode saturation_mode,
                                                      W min_num, W half_num, W max_num)
{
    using F = WordTraits<W>;
    W ax = old_num & F::ABS_MASK;
    if (ax == 0u)
        return 0u; // a zero input rounds to zero, and P3109's zero is unsigned

    quantized_num &= F::ABS_MASK;

    // handle overflow
    if (quantized_num > max_num)
        quantized_num = (saturation_mode == SaturationMode::SAT_NONE) ? W(F::INF_BITS) : max_num;
    // handle underflow
    else if constexpr (U == UnderflowMode::NONE)
    {
        // nothing to do: see UnderflowMode::NONE
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

// The same, for a signed input: the magnitude above with the input's sign
// put back on it, unless there is no magnitude to sign. A value that flushed
// to zero in the underflow arm, and any result of a format whose max_num is
// zero, arrive here with nothing to sign, and P3109's zero is unsigned. This
// is where the casts that round through the integer path get their unsigned
// zero; the mask costs a cmov and an or, where clearing the sign at the
// cast's return instead cost 1.05-1.23x.
//
// Keeping the two forms separate is worth more than it looks.
// cast_absolute_up and its three counterparts pass |x|, so their sign bit is
// always clear and everything this function adds does nothing, but the
// compiler cannot see that. Giving them the magnitude form took the binaryK
// RZ GEMM on a GPU from 1.13x to 0.92x of its time before the unsigned zero
// was introduced, and the FMA one to 0.87x, since the sign work the directed
// modes had been doing for nothing went with it.
template <UnderflowMode U, class W>
CUDA_HOST_DEVICE_INLINE W clip_normal_range_exponent(W old_num, W quantized_num, SaturationMode saturation_mode,
                                                     W min_num, W half_num, W max_num)
{
    using F = WordTraits<W>;
    W magnitude = clip_normal_range_magnitude<U>(old_num, quantized_num, saturation_mode,
                                                 min_num, half_num, max_num);
    W sign = old_num & F::SIGN_MASK;
    return magnitude | (sign & -(W)(magnitude != 0));
}

// What an input whose exponent field is all ones (an infinity or a NaN)
// becomes. A NaN passes through unchanged under every mode, payload
// included. So does an infinity, except under P3109's SatFinite (SAT_FINITE
// and SAT_FINITE_EXTENDED), which promises that every result is finite:
// there it becomes the format's largest finite magnitude, max_num, with the
// input's sign, the same value an overflowing finite input saturates to
// under those modes. Every integer-path cast's NaN/inf arm calls this; the
// float-arithmetic fast paths reproduce it with their final select instead
// (see cast_binaryK_rne_fast). It is on the arm only non-finite inputs take,
// so it costs the finite ones nothing.
template <class W>
CUDA_HOST_DEVICE_INLINE W saturate_nonfinite(W target, SaturationMode saturation_mode, W max_num)
{
    using F = WordTraits<W>;
    bool is_inf = (target & F::MAN_MASK) == 0;
    bool clamps_inf =
        saturation_mode == SaturationMode::SAT_FINITE || saturation_mode == SaturationMode::SAT_FINITE_EXTENDED;
    return (is_inf && clamps_inf) ? ((target & F::SIGN_MASK) | max_num) : target;
}

// The two negations the directed modes need, both done on the word rather
// than as `-x`, and neither for speed.
//
// `-x` is a float operation, and on the device that is `neg.f32`, which PTX
// does not require to keep a NaN's payload: -(-NaN) may come back as the
// canonical NaN, and lose its sign with it, so a negative NaN could leave
// cast_binaryK_up as a positive one in some builds and not in others,
// depending on how nvcc inlined the cast. The promise is that a NaN passes
// through whole ("P3109's single NaN is whichever NaN came in",
// docs/source/concepts.rst), and an XOR on the sign bit touches nothing
// else, so neither function needs a NaN arm of its own. `neg.f64` makes the
// same lack of promise, so binary64 negates on the word too.
//
// flip_sign is the unconditional flip, used to hand a negative input to the
// magnitude helpers as |x|. negate_magnitude is the flip on the way back,
// and also the one place where a -0.0 could still be made after the clips
// have run: by negating a magnitude that rounded to zero. Masking the flip
// with unsigned_zero_bits' test leaves that zero alone, and leaves a NaN
// whole, since its word is nonzero after the shift.
template <class T>
CUDA_HOST_DEVICE_INLINE T flip_sign(T x)
{
    using word_t = typename FloatTraits<T>::word_t;
    word_t w = reinterpret_cast<const MPTORCH_THREAD word_t &>(x) ^ FloatTraits<T>::SIGN_MASK;
    return reinterpret_cast<const MPTORCH_THREAD T &>(w);
}

template <class T>
CUDA_HOST_DEVICE_INLINE T negate_magnitude(T m)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    word_t w = reinterpret_cast<const MPTORCH_THREAD word_t &>(m);
    w ^= F::SIGN_MASK & -(word_t)((w << 1) != 0);
    return reinterpret_cast<const MPTORCH_THREAD T &>(w);
}
