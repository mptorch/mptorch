#pragma once

#include "cast_binaryK.h"
#include "cast_superfp.h"
#include "modes.h"
#include "philox.h"
#if !defined(__METAL_VERSION__)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>
#endif

// ---------------------------------------------------------------------------
// Multiplier policies: each rounds one product a*b of a dot product to a
// format.
//
// Each computes its cast's constants, which depend only on the format
// (BinaryKParamsT and SuperfpParamsT; see cast_binaryK.h and cast_superfp.h),
// once when it is constructed rather than on every call of operator(), which
// runs up to M*N*K times per kernel launch. The default constructors exist
// only so that SplitMac's and FusedMac's default member initializers compile;
// a policy is always built with the explicit constructor.
//
// RM, the rounding mode, is a template parameter rather than a field read at
// run time. A `switch (round_mode)` over the seven cast bodies inside the
// K-loop (fourteen for a split mac, which rounds twice per step) is more code
// than the loop can fetch efficiently, and stops the loop from being fully
// unrolled; measured, it cost 1.2-1.8x the kernel time. With the mode fixed
// at compile time, the `if constexpr` chain leaves one body, and the loop
// unrolls again. prng_bits stays a run-time field, because it is one integer
// read rather than a code path.
//
// Under RoundMode::SR, each call draws one random value from the PhiloxEngine
// (philox.h) that operator() and Mac::step take next to the operands. The
// engine belongs to the accumulator and is seeded once per output element,
// through seed_rng(), before that element's K-loop starts. The Multiplier
// and Adder hold no random state, only prng_bits, the number of random bits
// the cast uses.
//
// T is the carrier, the floating-point type in which the products, the sums
// and the casts are computed (bit_helper.h's FloatTraits): float for a
// float32, float16 or bfloat16 GEMM, and double for a float64 one. A binary64
// draw is 64 bits, two of the stream's 32-bit words taken low word first
// (PhiloxEngine::next64), so an SR K-step consumes twice as many words as in
// binary32. The binary32 case is its own `if constexpr` branch, so that the
// float instantiations compile exactly as they would without the double
// ones.

template <class T, RoundMode RM>
struct BinaryKMultiplierT
{
    using value_t = T;
    bool is_signed;
    SubnormalsMode subnormals_mode;
    int prng_bits;
    BinaryKParamsT<T> params;

    BinaryKMultiplierT() = default;
    CUDA_HOST_DEVICE_INLINE BinaryKMultiplierT(int man_bits, int exp_bits, int bias, bool is_signed,
                                               SaturationMode saturation_mode,
                                               SubnormalsMode subnormals_mode, int prng_bits = 0)
        : is_signed(is_signed), subnormals_mode(subnormals_mode), prng_bits(prng_bits),
          params(make_binaryK_params<T>(man_bits, exp_bits, bias, is_signed, saturation_mode,
                                        subnormals_mode == SubnormalsMode::EXTENDED_NORMALS))
    {
    }

    // The product a*b, rounded once by the carrier's multiplication and then
    // cast to the format under RM.
    CUDA_HOST_DEVICE_INLINE T operator()(T a, T b, MPTORCH_THREAD PhiloxEngine &rng) const
    {
        T x = a * b;
        if constexpr (RM == RoundMode::RNA)
            return cast_binaryK_nearest_away(x, is_signed, subnormals_mode, params);
        else if constexpr (RM == RoundMode::RU)
            return cast_binaryK_up(x, is_signed, subnormals_mode, params);
        else if constexpr (RM == RoundMode::RD)
            return cast_binaryK_down(x, is_signed, subnormals_mode, params);
        else if constexpr (RM == RoundMode::RZ)
            return cast_binaryK_zero(x, is_signed, subnormals_mode, params);
        else if constexpr (RM == RoundMode::RO)
            return cast_binaryK_odd(x, is_signed, subnormals_mode, params);
        else // RoundMode::RNE
            return cast_binaryK_nearest_even(x, is_signed, subnormals_mode, params);
    }
};

// The stochastic version: the primary template's members, followed by the
// constants of the stochastic cast (StochasticT, stochastic.h), which are
// built once with the policy, so a step pays only the cast's integer
// operations on the word. It is a specialization rather than an extra member
// of the primary template, so that the deterministic policies, and every
// kernel built from them, keep their layout.
template <class T>
struct BinaryKMultiplierT<T, RoundMode::SR>
{
    using value_t = T;
    bool is_signed;
    SubnormalsMode subnormals_mode;
    int prng_bits;
    BinaryKParamsT<T> params;
    StochasticT<T> sr;

    BinaryKMultiplierT() = default;
    CUDA_HOST_DEVICE_INLINE BinaryKMultiplierT(int man_bits, int exp_bits, int bias, bool is_signed,
                                               SaturationMode saturation_mode,
                                               SubnormalsMode subnormals_mode, int prng_bits = 0)
        : is_signed(is_signed), subnormals_mode(subnormals_mode), prng_bits(prng_bits),
          params(make_binaryK_params<T>(man_bits, exp_bits, bias, is_signed, saturation_mode,
                                        subnormals_mode == SubnormalsMode::EXTENDED_NORMALS)),
          sr(make_stochastic_binaryK<T>(params, subnormals_mode, is_signed, prng_bits))
    {
    }

    CUDA_HOST_DEVICE_INLINE T operator()(T a, T b, MPTORCH_THREAD PhiloxEngine &rng) const
    {
        T x = a * b;
        if constexpr (std::is_same_v<T, float>)
            return cast_binaryK_stochastic(x, rng(), sr, is_signed, subnormals_mode, params);
        else
            return cast_binaryK_stochastic(x, rng.next64(), sr, is_signed, subnormals_mode, params);
    }
};

template <class T, RoundMode RM>
struct SuperfpMultiplierT
{
    using value_t = T;
    bool is_signed;
    int prng_bits;
    SuperfpParamsT<T> params;

    SuperfpMultiplierT() = default;
    CUDA_HOST_DEVICE_INLINE SuperfpMultiplierT(int man_bits, int exp_bits, int normal_binades, int bias,
                                               bool is_signed, SaturationMode saturation_mode,
                                               int prng_bits = 0)
        : is_signed(is_signed), prng_bits(prng_bits),
          params(make_superfp_params<T>(man_bits, exp_bits, normal_binades, bias, saturation_mode))
    {
    }

    CUDA_HOST_DEVICE_INLINE T operator()(T a, T b, MPTORCH_THREAD PhiloxEngine &rng) const
    {
        T x = a * b;
        if constexpr (RM == RoundMode::RNA)
            return cast_superfp_nearest_away(x, is_signed, params);
        else if constexpr (RM == RoundMode::RU)
            return cast_superfp_up(x, is_signed, params);
        else if constexpr (RM == RoundMode::RD)
            return cast_superfp_down(x, is_signed, params);
        else if constexpr (RM == RoundMode::RZ)
            return cast_superfp_zero(x, is_signed, params);
        else if constexpr (RM == RoundMode::RO)
            return cast_superfp_odd(x, is_signed, params);
        else // RoundMode::RNE
            return cast_superfp_nearest_even(x, is_signed, params);
    }
};

// The stochastic version (see BinaryKMultiplierT's).
template <class T>
struct SuperfpMultiplierT<T, RoundMode::SR>
{
    using value_t = T;
    bool is_signed;
    int prng_bits;
    SuperfpParamsT<T> params;
    StochasticT<T> sr;

    SuperfpMultiplierT() = default;
    CUDA_HOST_DEVICE_INLINE SuperfpMultiplierT(int man_bits, int exp_bits, int normal_binades, int bias,
                                               bool is_signed, SaturationMode saturation_mode,
                                               int prng_bits = 0)
        : is_signed(is_signed), prng_bits(prng_bits),
          params(make_superfp_params<T>(man_bits, exp_bits, normal_binades, bias, saturation_mode)),
          sr(make_stochastic_superfp<T>(params, is_signed, prng_bits))
    {
    }

    CUDA_HOST_DEVICE_INLINE T operator()(T a, T b, MPTORCH_THREAD PhiloxEngine &rng) const
    {
        T x = a * b;
        if constexpr (std::is_same_v<T, float>)
            return cast_superfp_stochastic(x, rng(), sr, is_signed, params);
        else
            return cast_superfp_stochastic(x, rng.next64(), sr, is_signed, params);
    }
};

// ---------------------------------------------------------------------------
// Adder policies: each rounds one update of the running sum to a format. A
// Mac policy (below) uses one, either as the add half of a split mac (a
// multiply and an add, each rounded) or as the single rounding of a fused
// mac's multiply-add. They compute their constants at construction and take
// RM as a template parameter, like the multipliers.

template <class T, RoundMode RM>
struct BinaryKAdderT
{
    using value_t = T;
    bool is_signed;
    SubnormalsMode subnormals_mode;
    int prng_bits;
    BinaryKParamsT<T> params;

    BinaryKAdderT() = default;
    CUDA_HOST_DEVICE_INLINE BinaryKAdderT(int man_bits, int exp_bits, int bias, bool is_signed,
                                          SaturationMode saturation_mode,
                                          SubnormalsMode subnormals_mode, int prng_bits = 0)
        : is_signed(is_signed), subnormals_mode(subnormals_mode), prng_bits(prng_bits),
          params(make_binaryK_params<T>(man_bits, exp_bits, bias, is_signed, saturation_mode,
                                        subnormals_mode == SubnormalsMode::EXTENDED_NORMALS))
    {
    }

    CUDA_HOST_DEVICE_INLINE T operator()(T x, MPTORCH_THREAD PhiloxEngine &rng) const
    {
        if constexpr (RM == RoundMode::RNA)
            return cast_binaryK_nearest_away(x, is_signed, subnormals_mode, params);
        else if constexpr (RM == RoundMode::RU)
            return cast_binaryK_up(x, is_signed, subnormals_mode, params);
        else if constexpr (RM == RoundMode::RD)
            return cast_binaryK_down(x, is_signed, subnormals_mode, params);
        else if constexpr (RM == RoundMode::RZ)
            return cast_binaryK_zero(x, is_signed, subnormals_mode, params);
        else if constexpr (RM == RoundMode::RO)
            return cast_binaryK_odd(x, is_signed, subnormals_mode, params);
        else // RoundMode::RNE
            return cast_binaryK_nearest_even(x, is_signed, subnormals_mode, params);
    }
};

// The stochastic version (see BinaryKMultiplierT's).
template <class T>
struct BinaryKAdderT<T, RoundMode::SR>
{
    using value_t = T;
    bool is_signed;
    SubnormalsMode subnormals_mode;
    int prng_bits;
    BinaryKParamsT<T> params;
    StochasticT<T> sr;

    BinaryKAdderT() = default;
    CUDA_HOST_DEVICE_INLINE BinaryKAdderT(int man_bits, int exp_bits, int bias, bool is_signed,
                                          SaturationMode saturation_mode,
                                          SubnormalsMode subnormals_mode, int prng_bits = 0)
        : is_signed(is_signed), subnormals_mode(subnormals_mode), prng_bits(prng_bits),
          params(make_binaryK_params<T>(man_bits, exp_bits, bias, is_signed, saturation_mode,
                                        subnormals_mode == SubnormalsMode::EXTENDED_NORMALS)),
          sr(make_stochastic_binaryK<T>(params, subnormals_mode, is_signed, prng_bits))
    {
    }

    CUDA_HOST_DEVICE_INLINE T operator()(T x, MPTORCH_THREAD PhiloxEngine &rng) const
    {
        if constexpr (std::is_same_v<T, float>)
            return cast_binaryK_stochastic(x, rng(), sr, is_signed, subnormals_mode, params);
        else
            return cast_binaryK_stochastic(x, rng.next64(), sr, is_signed, subnormals_mode, params);
    }
};

template <class T, RoundMode RM>
struct SuperfpAdderT
{
    using value_t = T;
    bool is_signed;
    int prng_bits;
    SuperfpParamsT<T> params;

    SuperfpAdderT() = default;
    CUDA_HOST_DEVICE_INLINE SuperfpAdderT(int man_bits, int exp_bits, int normal_binades, int bias,
                                          bool is_signed, SaturationMode saturation_mode,
                                          int prng_bits = 0)
        : is_signed(is_signed), prng_bits(prng_bits),
          params(make_superfp_params<T>(man_bits, exp_bits, normal_binades, bias, saturation_mode))
    {
    }

    CUDA_HOST_DEVICE_INLINE T operator()(T x, MPTORCH_THREAD PhiloxEngine &rng) const
    {
        if constexpr (RM == RoundMode::RNA)
            return cast_superfp_nearest_away(x, is_signed, params);
        else if constexpr (RM == RoundMode::RU)
            return cast_superfp_up(x, is_signed, params);
        else if constexpr (RM == RoundMode::RD)
            return cast_superfp_down(x, is_signed, params);
        else if constexpr (RM == RoundMode::RZ)
            return cast_superfp_zero(x, is_signed, params);
        else if constexpr (RM == RoundMode::RO)
            return cast_superfp_odd(x, is_signed, params);
        else // RoundMode::RNE
            return cast_superfp_nearest_even(x, is_signed, params);
    }
};

// The stochastic version (see BinaryKMultiplierT's).
template <class T>
struct SuperfpAdderT<T, RoundMode::SR>
{
    using value_t = T;
    bool is_signed;
    int prng_bits;
    SuperfpParamsT<T> params;
    StochasticT<T> sr;

    SuperfpAdderT() = default;
    CUDA_HOST_DEVICE_INLINE SuperfpAdderT(int man_bits, int exp_bits, int normal_binades, int bias,
                                          bool is_signed, SaturationMode saturation_mode,
                                          int prng_bits = 0)
        : is_signed(is_signed), prng_bits(prng_bits),
          params(make_superfp_params<T>(man_bits, exp_bits, normal_binades, bias, saturation_mode)),
          sr(make_stochastic_superfp<T>(params, is_signed, prng_bits))
    {
    }

    CUDA_HOST_DEVICE_INLINE T operator()(T x, MPTORCH_THREAD PhiloxEngine &rng) const
    {
        if constexpr (std::is_same_v<T, float>)
            return cast_superfp_stochastic(x, rng(), sr, is_signed, params);
        else
            return cast_superfp_stochastic(x, rng.next64(), sr, is_signed, params);
    }
};

// The adder that rounds nothing, for a running sum that stays in the
// carrier's precision: a SplitMac with accumulate_quant=false, which rounds
// only the product, or a FusedMac with fma_quant=false, which rounds
// nothing.
template <class T>
struct IdentityAdder
{
    using value_t = T;
    CUDA_HOST_DEVICE_INLINE T operator()(T x, MPTORCH_THREAD PhiloxEngine & /*rng*/) const { return x; }
};

// ---------------------------------------------------------------------------
// Mac policies (mac: multiply-accumulate): each computes one step of a dot
// product from the two operands and the running sum, step(a, b, acc) -> the
// new sum. A SplitMac rounds the product and the sum separately, two
// roundings per step. A FusedMac rounds the result of one fused multiply-add,
// one rounding per step as in a hardware FMA unit, using an Adder for that
// rounding. NaiveAccumulator below only ever calls Mac::step, so it drives
// either kind. TREE accumulation (gemm_accumulate.h) is the exception: it
// sums the products pairwise, so it needs a SplitMac, since a fused
// multiply-add has no separate product.

// The carrier's fused multiply-add: a*b + c with one rounding. The device
// functions and std::fma are all correctly rounded, so the backends agree
// bit for bit (on an Apple GPU, except where it flushes binary32 subnormals
// to zero; see mps/gemm.metal).
CUDA_HOST_DEVICE_INLINE float fma_f32(float a, float b, float c)
{
#if defined(__CUDA_ARCH__)
    return fmaf(a, b, c);
#elif defined(__METAL_VERSION__)
    return metal::fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

#if !defined(__METAL_VERSION__)
CUDA_HOST_DEVICE_INLINE double fma_f64(double a, double b, double c)
{
#if defined(__CUDA_ARCH__)
    return fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
#endif

// Both halves of a mac compute in one carrier, the Adder's. The product and
// the sum are rounded to the carrier's precision (24 bits in binary32, 53 in
// binary64) before the format's cast sees them. So, unlike the elementwise
// quantizers, a float64 GEMM of float32 values can differ from the float32
// GEMM of the same values: the product of two 24-bit significands has 48
// bits, which binary64 holds exactly and binary32 has already rounded.
template <class Multiplier, class Adder>
struct SplitMac
{
    using value_t = typename Adder::value_t;

    Multiplier mul{};
    Adder add{};

    CUDA_HOST_DEVICE_INLINE value_t step(value_t a, value_t b, value_t acc, MPTORCH_THREAD PhiloxEngine &rng) const
    {
        return add(acc + mul(a, b, rng), rng);
    }
};

template <class Adder>
struct FusedMac
{
    using value_t = typename Adder::value_t;

    Adder add{};

    CUDA_HOST_DEVICE_INLINE value_t step(value_t a, value_t b, value_t acc, MPTORCH_THREAD PhiloxEngine &rng) const
    {
        if constexpr (std::is_same_v<value_t, float>)
            return add(fma_f32(a, b, acc), rng);
        else
            return add(fma_f64(a, b, acc), rng);
    }
};

// ---------------------------------------------------------------------------
// Accumulator policies: each holds the state of a dot product's running sum.
// A GEMM kernel calls seed_rng(), accumulate() and finalize() on one (and,
// on the accumulators of gemm_accumulate.h, end_slab() as well).
// accumulate() takes the two operands rather than their product, so that an
// accumulator can drive either a SplitMac or a FusedMac. NaiveAccumulator,
// a plain sum in order, is AccumulateAlgorithm::NAIVE; KAHAN, BLOCK and TREE
// are in gemm_accumulate.h.
//
// seed_rng() prepares the random stream for RoundMode::SR. The kernel calls
// it once per output element, just before that element's K-loop starts, with
// the element's linear index in the whole [batch, M, N] output as the
// subsequence. Keying the stream on that index, rather than on a thread or
// tile index, makes an SR result independent of the block size, the grid's
// shape and the number of CPU threads, and makes batch element 0 of a
// batched call draw exactly what the 2D call draws. The stream cannot be
// seeded at construction, because a CPU tile's per-element state is created
// all at once, before it is known which element each slot will hold.
//
// The state has two forms. A GPU thread owns one whole accumulator and keeps
// it in registers, which is what the object form below is for. A CPU tile
// needs one per output element, and a whole accumulator for each would copy
// the Mac policy per element; see NaiveTile.

// The raw pointers through which a CPU kernel reads a NaiveTile, taken once
// per tile. The K-loop has to keep the base pointers in registers, which the
// compiler cannot do through a std::vector member that the loop's stores
// might alias; `__restrict__` on plain pointers says that they do not.
//
// This and NaiveTile are used only by the CPU kernel, so Metal never sees
// them.
#if !defined(__METAL_VERSION__)
template <class T>
struct NaiveTileView
{
    T *__restrict__ sums;
    PhiloxEngine *__restrict__ rng;
};

// The per-element state of a CPU tile's running sums (host only).
//
// The Mac policy is the same for every element of a single-format call and
// comes from the palette on a palette call, so it should not be copied per
// element. A split-mac binaryK accumulator is 216 B under RNE and 344 B
// under SR (two policies with their cast constants, a 44 B random engine and
// a 4 B sum; measured with g++, 2026-11-18), so a 32x32 tile of them would be
// a heap allocation of 216 KB or more, and as many bytes copied, to carry one
// float of running sum each. NaiveTile holds only the state, so what the
// K-loop touches on every step is a dense array of sums.
//
// The sums are in the carrier T, so a binary64 tile takes 8 B per sum.
template <class T>
struct NaiveTile
{
    std::vector<T> sums;
    std::vector<PhiloxEngine> streams;

    // Called once per worker task rather than once per tile: the buffers are
    // reused for every tile that task computes.
    //
    // The streams are allocated whether or not the mode is RoundMode::SR,
    // which wastes 44 B per element under the six deterministic modes (45 KB
    // per worker task for a 32x32 tile, never read). Both ways of avoiding it
    // cost more than it does: a stride chosen at run time leaves a multiply
    // in the innermost loop, and one fixed at compile time duplicates the
    // whole K-loop, which measured 7-12% slower on the split macs, whose cast
    // bodies are the largest.
    void resize(int64_t n)
    {
        sums.resize(static_cast<size_t>(n));
        streams.resize(static_cast<size_t>(n));
    }

    NaiveTileView<T> view() { return {sums.data(), streams.data()}; }

    // Zeroes the first `n` sums for a new tile.
    void begin(int64_t n) { std::fill_n(sums.data(), static_cast<size_t>(n), T(0)); }

    // Seeds element `idx`'s stream. The kernel passes the element's linear
    // index in the whole output as the subsequence.
    void seed_rng(int64_t idx, uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
        PhiloxEngine &e = streams[static_cast<size_t>(idx)];
        e.reset_state(seed, subsequence);
        e.set_offset(offset);
    }

    T finalize(int64_t idx) const { return sums[static_cast<size_t>(idx)]; }
};
#endif // !__METAL_VERSION__

// NAIVE accumulation, a plain sum in order: sum <- mac.step(a, b, sum) for
// each K-step. The object form (mac, sum, rng) is one GPU thread's whole
// state; the static tile form is the CPU kernel's.
template <class Mac>
struct NaiveAccumulator
{
    using mac_type = Mac;
    using value_t = typename Mac::value_t;
#if !defined(__METAL_VERSION__)
    using tile_type = NaiveTile<value_t>;
#endif

    Mac mac{};
    value_t sum = value_t(0);
    PhiloxEngine rng{};

    CUDA_HOST_DEVICE_INLINE void seed_rng(uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
        rng.reset_state(seed, subsequence);
        rng.set_offset(offset);
    }
    CUDA_HOST_DEVICE_INLINE void accumulate(value_t a, value_t b) { sum = mac.step(a, b, sum, rng); }
    CUDA_HOST_DEVICE_INLINE value_t finalize() const { return sum; }

    // The tile form of accumulate(), host only. The running sum and the SR
    // stream come from the tile, and the Mac from `m`: one Mac shared by the
    // whole call on the single-format path, and this element's palette slot
    // on the palette path. It evaluates the same expression as the object
    // form above, in the same order, so the two give identical values.
#if !defined(__METAL_VERSION__)
    static inline void accumulate(const NaiveTileView<value_t> &t, int64_t idx, const Mac &m,
                                  value_t a, value_t b)
    {
        t.sums[idx] = m.step(a, b, t.sums[idx], t.rng[idx]);
    }
#endif
};

// ---------------------------------------------------------------------------
// Palettes: a format chosen per output element.
//
// A palette call chooses, for each output element C[row, col], which of up
// to MAX_GEMM_FORMATS Mac policies computes that element's whole sum over K.
// Every slot is a fully built Mac, its cast constants computed at
// construction like the single-format path's acc_proto.mac, so the K-loop
// derives nothing. On the CPU and CUDA, the kernel's per-element prologue,
// where RoundMode::SR seeds the element's stream, takes the chosen slot
// before the K-loop starts, and the loop is the same code as on the
// single-format path. (The MPS backend builds only the chosen slot instead;
// see the with_slot factories in gemm_args.h.)
//
// FormatPalette is passed to the kernel by value, next to acc_proto and in
// the same way. It is a plain array and nothing else, because a kernel
// argument with pointers or a size known only at run time would need a
// device allocation with its own lifetime. So a Mac must stay trivially
// copyable.
//
// The kernel reads prec_idx[row * idx_row_stride + col * idx_col_stride]
// (plus a batch term), so one code path serves a dense [M, N] map
// (row_stride = N, col_stride = 1), a per-row [M, 1] map (1, 0) and a
// per-column [1, N] map (0, 1), with no branches. Whether that path runs is
// the kernel's MIXED template parameter, not a property of the palette: the
// single-format instantiations take no palette at all (see PaletteArg).
constexpr MPTORCH_CONSTANT int MAX_GEMM_FORMATS = 8;
static_assert((MAX_GEMM_FORMATS & (MAX_GEMM_FORMATS - 1)) == 0,
              "MAX_GEMM_FORMATS must be a power of two: slot() masks with it");

// The number of slots in use is deliberately not a member, because nothing
// in the kernel would read it. The host checks the palette's length before
// packing it (gemm_host.h's check_palette_lengths, which also guarantees
// that a MIXED kernel gets a non-empty palette), and slot() masks the index
// rather than checking it against a bound.
template <class Mac>
struct FormatPalette
{
    Mac slots[MAX_GEMM_FORMATS] = {};

    // Returns the slot for a precision index. The index is masked rather
    // than trusted: resolve_prec_idx checks the whole map's bounds on the
    // host, but it remembers maps it has checked (gemm_host.h), so the mask
    // is what guarantees that an index outside [0, n) can at worst pick the
    // wrong slot, and never reads past the array. It costs one AND, on a path
    // that is compiled out when MIXED is false.
    CUDA_HOST_DEVICE_INLINE const MPTORCH_THREAD Mac &slot(int32_t idx) const
    {
        return slots[idx & (MAX_GEMM_FORMATS - 1)];
    }
};

// The type of the kernels' palette parameter: empty on the single-format
// path, and a FormatPalette on the palette path. A FormatPalette of eight
// split macs is 1.3 KB under RNE and 2.3 KB under SR (measured with g++,
// 2026-11-18), and a kernel argument is copied to the device on every launch
// whether or not the kernel reads it, so a single-format kernel that took
// one would copy that much zeroed policy per call for code it compiles
// out.
struct NoPalette
{
};

template <bool MIXED, class Mac>
using PaletteArg = std::conditional_t<MIXED, FormatPalette<Mac>, NoPalette>;
