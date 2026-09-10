#pragma once

#include "modes.h"
#include <cmath>
#include <cstdint>

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

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

// The two spellings the fast paths need that differ between host and device.
//
// `bits_to_float` is BITS_TO_FLOAT for a value rather than an lvalue, standing
// in for the device-only __int_as_float; on both sides it is a reinterpret and
// compiles to nothing.
//
// rn_add/rn_sub/rn_mul are the binary32 operations the Veltkamp split needs
// rounded on their own. On the device that is what __fadd_rn and friends mean.
// On the host the plain operators already are IEEE binary32 operations -- the
// only thing that can merge two of them is contraction into an FMA, which
// MPTORCH_FAST_CAST's contract forbids -- so the shim is the operator itself,
// named so the requirement stays legible at the call site. (fma_f32 in
// gemm_policy.h is the reverse case: the one place an FMA is intended, spelled
// explicitly so no flag can take it away.)
CUDA_HOST_DEVICE_INLINE float bits_to_float(uint32_t bits)
{
    return BITS_TO_FLOAT(&bits);
}

#if defined(__CUDA_ARCH__)
CUDA_HOST_DEVICE_INLINE float rn_add(float a, float b) { return __fadd_rn(a, b); }
CUDA_HOST_DEVICE_INLINE float rn_sub(float a, float b) { return __fsub_rn(a, b); }
CUDA_HOST_DEVICE_INLINE float rn_mul(float a, float b) { return __fmul_rn(a, b); }
#else
CUDA_HOST_DEVICE_INLINE float rn_add(float a, float b) { return a + b; }
CUDA_HOST_DEVICE_INLINE float rn_sub(float a, float b) { return a - b; }
CUDA_HOST_DEVICE_INLINE float rn_mul(float a, float b) { return a * b; }
#endif

// min/max where the fast paths use them: on a magnitude clamped against a
// positive bound, never on a negative operand. Under that precondition the
// ternary and fminf/fmaxf agree everywhere, NaN included -- every compare
// against a NaN is false, so both spellings return the bound, which is the
// arm the fast paths' final select overrides anyway.
//
// Two spellings because each backend folds exactly one of them into its
// single instruction and neither folds the other: GCC will not turn fminf
// into minss (their NaN results differ in general) and emits a libm call in
// the middle of the GEMM's inner loop, worth 1.12x of the whole CPU GEMM;
// nvcc will not turn the ternary into min.f32 and emits setp + selp.
#if defined(__CUDA_ARCH__)
CUDA_HOST_DEVICE_INLINE float fmin_nonneg(float a, float b) { return fminf(a, b); }
CUDA_HOST_DEVICE_INLINE float fmax_nonneg(float a, float b) { return fmaxf(a, b); }
#else
CUDA_HOST_DEVICE_INLINE float fmin_nonneg(float a, float b) { return (a < b) ? a : b; }
CUDA_HOST_DEVICE_INLINE float fmax_nonneg(float a, float b) { return (a > b) ? a : b; }
#endif

// rounds to nearest, ties to even
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_nearest_even(uint32_t target, int man_bits)
{
    if (man_bits >= 23)
        return target;
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t tie = 1 << (22 - man_bits);
    uint32_t add_r = target + tie;
    uint32_t quantized = add_r & ~mask;
    uint32_t is_tie = (target & mask) == tie;
    uint32_t odd = (man_bits == 0) ? 0 : 1; // if man_bits == 0, implicit bit is 1 (odd) so we always round up (carry to exponent)
    return quantized & ~((is_tie & odd) << (23 - man_bits));
}

CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_nearest_even(uint32_t target)
{
    uint32_t tie = 0x00400000;
    uint32_t quantized = (target + tie) & ~0x007FFFFF;
    uint32_t is_tie = (target & 0x007FFFFF) == tie;
    return quantized - ((is_tie << 23) & ~quantized);
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
struct RoundParams
{
    bool bypass; // man_bits >= 23: round_bitwise_nearest_even returns target unchanged
    uint32_t mask;
    uint32_t tie;
    int shift; // 23 - man_bits
};

CUDA_HOST_DEVICE_INLINE RoundParams make_round_params(int man_bits)
{
    RoundParams p;
    p.bypass = man_bits >= 23;
    if (p.bypass)
    {
        p.mask = 0u;
        p.tie = 0u;
        p.shift = 0;
        return p;
    }
    p.shift = 23 - man_bits;
    p.mask = (1u << p.shift) - 1u;
    p.tie = 1u << (p.shift - 1);
    return p;
}

CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_nearest_even(uint32_t target, bool bypass, uint32_t mask,
                                                            uint32_t tie, int shift)
{
    if (bypass)
        return target;
    uint32_t add_r = target + tie;
    uint32_t quantized = add_r & ~mask;
    uint32_t is_tie = (target & mask) == tie;
    return quantized & ~(is_tie << shift);
}

// rounds to nearest, ties to away
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_nearest_away(uint32_t target, int man_bits)
{
    if (man_bits >= 23)
        return target;
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t tie = 1 << (22 - man_bits);
    uint32_t add_r = target + tie;
    return add_r & ~mask;
}

// Precomputed sibling of round_bitwise_nearest_away above, via RoundParams
// (never needs the `shift` field).
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_nearest_away(uint32_t target, bool bypass, uint32_t mask, uint32_t tie)
{
    if (bypass)
        return target;
    uint32_t add_r = target + tie;
    return add_r & ~mask;
}

// rounds up, towards positive infinity
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_up(uint32_t target, int man_bits)
{
    if (man_bits >= 23)
        return target;
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t sign = target >> 31;
    uint32_t add_r = target + (sign ? 0 : mask);
    return add_r & ~mask;
}

// Precomputed sibling of round_bitwise_up above: `sign` is per-value data,
// not a format constant, so only bypass/mask (man_bits-derived) precompute.
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_up(uint32_t target, bool bypass, uint32_t mask)
{
    if (bypass)
        return target;
    uint32_t sign = target >> 31;
    uint32_t add_r = target + (sign ? 0 : mask);
    return add_r & ~mask;
}

// rounds down, towards negative infinity
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_down(uint32_t target, int man_bits)
{
    if (man_bits >= 23)
        return target;
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t sign = target >> 31;
    uint32_t add_r = target + (sign ? mask : 0);
    return add_r & ~mask;
}

// Precomputed sibling of round_bitwise_down above -- see round_bitwise_up's.
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_down(uint32_t target, bool bypass, uint32_t mask)
{
    if (bypass)
        return target;
    uint32_t sign = target >> 31;
    uint32_t add_r = target + (sign ? mask : 0);
    return add_r & ~mask;
}

// rounds to odd: truncates towards zero to man_bits, then ORs the "sticky
// bit" of the discarded bits into the kept LSB. When man_bits == 0 the kept
// "bit" is the exponent's LSB (no explicit significand), so a nonzero
// sticky bit carries into the exponent.
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_odd(uint32_t target, int man_bits)
{
    if (man_bits >= 23)
        return target;
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t lsb = 1 << (23 - man_bits);
    uint32_t sticky = (target & mask) != 0;
    return (target & ~mask) | (sticky * lsb);
}

// Precomputed sibling of round_bitwise_odd above. lsb = mask + 1, so no
// separate field is needed for it.
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_odd(uint32_t target, bool bypass, uint32_t mask)
{
    if (bypass)
        return target;
    uint32_t lsb = mask + 1u;
    uint32_t sticky = (target & mask) != 0;
    return (target & ~mask) | (sticky * lsb);
}

// stochastic rounding
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits)
{ // passing number of random bits as second parameter
    // (all the bits after the least significant bit which is based on prng);
    // target is the original number
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    // adding random bits to target (which is not masked)
    uint32_t add_r = target + (rand_prob & mask);
    // masking out bits on the right hand side of the significant bits (truncating)
    uint32_t quantized = add_r & ~mask;
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

CUDA_HOST_DEVICE_INLINE SubnormalRangeParams make_subnormal_range_params(int man_bits, int bias)
{
    SubnormalRangeParams p;
    p.min_exponent_store = -(bias - 1) - man_bits + 127;
    return p;
}

CUDA_HOST_DEVICE_INLINE uint32_t clip_subnormal_range_exponent(uint32_t old_num, uint32_t quantized_num,
                                                                int min_exponent_store)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = (int)((quantized_num >> 23) & 0xFF);

    uint32_t old_sign = old_num & 0x80000000u;
    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
    {
        int offset = (quantized_exponent_store == (min_exponent_store - 1));
        quantized_num += offset * (1u << 23);
        quantized_num |= old_sign;
        quantized_num *= offset;
    }

    return quantized_num;
}

// The _up variant: clamps an underflowing nonzero value to the smallest
// subnormal rather than to zero, for the directed and round-to-odd modes.
// Whether the value is nonzero is a question about the input, not about the
// rounded word: a -0.0 input truncates to 0x80000000, which is not 0 and was
// pushed out to minus the smallest subnormal, and a float32 subnormal input
// can truncate to exactly 0 and stayed there. P3109 rounds zero to zero and
// never rounds a nonzero value away from zero into it.
// Shares SubnormalRangeParams/make_subnormal_range_params with the plain
// clip_subnormal_range_exponent above.
CUDA_HOST_DEVICE_INLINE uint32_t clip_subnormal_range_exponent_up(uint32_t old_num, uint32_t quantized_num,
                                                                  int min_exponent_store)
{
    if ((old_num << 1) == 0)
        return quantized_num;

    int quantized_exponent_store = (int)((quantized_num >> 23) & 0xFF);

    uint32_t old_sign = old_num & 0x80000000u;
    if (quantized_exponent_store < min_exponent_store)
    {
        quantized_num = min_exponent_store << 23;
        quantized_num |= old_sign;
    }

    return quantized_num;
}

// Precomputed for clip_normal_range_exponent -- the form every cast in
// cast_binaryK.h and cast_superfp.h calls.
struct NormalRangeParams
{
    SaturationMode saturation_mode; // still needed raw: selects the overflow branch's outcome
    int min_exponent_store;
    uint32_t max_num; // the largest finite magnitude, as a binary32 word
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
// finite value is below binary32's normal range.
CUDA_HOST_DEVICE_INLINE NormalRangeParams make_normal_range_params(int exp_bits, int man_bits, int bias,
                                                                    SaturationMode saturation_mode,
                                                                    int reserved_codes,
                                                                    bool extended_normals = false)
{
    NormalRangeParams p;
    p.saturation_mode = saturation_mode;
    p.min_exponent_store = -(bias - 1) + 127 - extended_normals;

    const int man = (man_bits < 23) ? man_bits : 23; // binary32 holds no finer grid
    const int codes = 1 << man;
    int top_code = codes - 1 - reserved_codes; // the largest finite code, within its binade
    int top_field = (1 << exp_bits) - 1;       // and that binade's exponent field
    while (top_code < 0)
    {
        top_code += codes;
        --top_field;
    }
    const int top_exp = top_field - bias;
    if (top_field < 1 || top_exp < -126)
        p.max_num = 0u;
    else if (top_exp > 127)
        // A format whose largest finite value lies beyond binary32's has no
        // finite input that overflows, so max_num is only ever read for a
        // non-finite one -- an infinity, or a rounding that carried into
        // exponent 255 -- and there SAT_FINITE and SAT_PROPAGATE still owe a
        // finite answer: the largest binary32 value on the format's grid.
        p.max_num = (254u << 23) | ((uint32_t)(codes - 1) << (23 - man));
    else
        p.max_num = ((uint32_t)(top_exp + 127) << 23) | ((uint32_t)top_code << (23 - man));
    return p;
}

// Saturates a value the normal arm has rounded. P3109 rounds first and
// saturates second, which makes overflow a magnitude test: anything above
// max_num -- a rounding that landed on a reserved code, carried into the next
// binade, or carried a finite input all the way into exponent 255 -- is out of
// range, and becomes infinity under OVF_INF (SatNone) and max_num under the
// other two (SatFinite, SatPropagate: the value was finite, so SatPropagate
// has nothing to propagate). Non-finite inputs never get here: every cast
// sends them to saturate_nonfinite before it rounds.
CUDA_HOST_DEVICE_INLINE uint32_t clip_normal_range_exponent(uint32_t old_num, uint32_t quantized_num,
                                                             SaturationMode saturation_mode,
                                                             int min_exponent_store, uint32_t max_num)
{
    if (quantized_num == 0)
        return quantized_num;

    uint32_t sign = old_num & 0x80000000u;
    quantized_num &= 0x7FFFFFFFu;

    int quantized_exponent_store = (int)((quantized_num >> 23) & 0xFF);

    // handle overflow
    if (quantized_num > max_num)
        quantized_num = (saturation_mode == SaturationMode::OVF_INF) ? 0x7F800000u : max_num;
    // handle underflow
    else if (quantized_exponent_store < min_exponent_store)
    {
        uint32_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > (1 << 22));
        quantized_num = offset * (min_exponent_store << 23);
    }

    quantized_num |= sign;

    return quantized_num;
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
CUDA_HOST_DEVICE_INLINE uint32_t saturate_nonfinite(uint32_t target, SaturationMode saturation_mode,
                                                    uint32_t max_num)
{
    bool is_inf = (target & 0x007FFFFFu) == 0;
    return (is_inf && saturation_mode == SaturationMode::SAT_FINITE) ? ((target & 0x80000000u) | max_num)
                                                                      : target;
}
