#pragma once

#include "cast_binaryK.h"
#include "cast_superfp.h"
#include "modes.h"
#include "philox.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------------
// Multiplier policies: quantize a single dot-product term a*b.
//
// Each precomputes its cast function's format-derived constants
// (BinaryKParamsT / SuperfpParamsT, see cast_binaryK.h / cast_superfp.h)
// once at construction rather than on every operator() call, which runs up
// to M*N*K times per kernel launch. The default constructors exist only so
// that SplitMac/FusedMac's default member initializers are well-formed; a
// policy is always built through the explicit constructor.
//
// RM, the rounding mode, is a template parameter rather than a runtime
// field. A `switch (round_mode)` over the seven cast bodies inside the
// K-loop (fourteen for a split mac, which rounds twice per step) is more
// instruction memory than the loop can fetch from and stops it from being
// fully unrolled; measured, it cost 1.2-1.8x of kernel time. With the mode
// fixed at compile time, the `if constexpr` chain leaves one live body and
// the loop unrolls again. prng_bits stays a runtime field because it is one
// integer read, not a code path.
//
// RoundMode::SR draws one random value per call from the PhiloxEngine
// (philox.h) threaded through operator() and Mac::step alongside the
// operands. The engine lives on NaiveAccumulator below and is seeded once
// per output element, through seed_rng(), before that element's K-loop
// starts; the Multiplier and Adder stay stateless and hold only prng_bits,
// the number of random bits the cast compares against.
//
// T is the carrier the products, the sums and the casts are computed in
// (bit_helper.h's FloatTraits): float for a float32, float16 or bfloat16
// GEMM, double for a float64 one. A binary64 draw is 64 bits, two of the
// stream's words taken low word first (PhiloxEngine::next64), so an SR
// K-step consumes twice the words it does in binary32. The binary32 arm is
// its own `if constexpr` branch so the float instantiations are unchanged
// by the existence of the double ones.

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

    // The product a*b, rounded once by the carrier's multiply and then cast
    // to the format under RM.
    CUDA_HOST_DEVICE_INLINE T operator()(T a, T b, PhiloxEngine &rng) const
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
        else if constexpr (RM == RoundMode::SR && std::is_same_v<T, float>)
            return cast_binaryK_stochastic(x, rng(), prng_bits, is_signed, subnormals_mode, params);
        else if constexpr (RM == RoundMode::SR)
            return cast_binaryK_stochastic(x, rng.next64(), prng_bits, is_signed, subnormals_mode, params);
        else // RoundMode::RNE
            return cast_binaryK_nearest_even(x, is_signed, subnormals_mode, params);
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

    CUDA_HOST_DEVICE_INLINE T operator()(T a, T b, PhiloxEngine &rng) const
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
        else if constexpr (RM == RoundMode::SR && std::is_same_v<T, float>)
            return cast_superfp_stochastic(x, rng(), prng_bits, is_signed, params);
        else if constexpr (RM == RoundMode::SR)
            return cast_superfp_stochastic(x, rng.next64(), prng_bits, is_signed, params);
        else // RoundMode::RNE
            return cast_superfp_nearest_even(x, is_signed, params);
    }
};

// ---------------------------------------------------------------------------
// Adder policies: quantize a single running-sum update. Used inside a Mac
// policy (below), either as the "add" half of a split multiply-then-add, or
// as the single quantizer applied to a fused multiply-add's result. Same
// constants-at-construction and RM-as-template design as the multipliers.

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

    CUDA_HOST_DEVICE_INLINE T operator()(T x, PhiloxEngine &rng) const
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
        else if constexpr (RM == RoundMode::SR && std::is_same_v<T, float>)
            return cast_binaryK_stochastic(x, rng(), prng_bits, is_signed, subnormals_mode, params);
        else if constexpr (RM == RoundMode::SR)
            return cast_binaryK_stochastic(x, rng.next64(), prng_bits, is_signed, subnormals_mode, params);
        else // RoundMode::RNE
            return cast_binaryK_nearest_even(x, is_signed, subnormals_mode, params);
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

    CUDA_HOST_DEVICE_INLINE T operator()(T x, PhiloxEngine &rng) const
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
        else if constexpr (RM == RoundMode::SR && std::is_same_v<T, float>)
            return cast_superfp_stochastic(x, rng(), prng_bits, is_signed, params);
        else if constexpr (RM == RoundMode::SR)
            return cast_superfp_stochastic(x, rng.next64(), prng_bits, is_signed, params);
        else // RoundMode::RNE
            return cast_superfp_nearest_even(x, is_signed, params);
    }
};

// No-op adder: used when only the multiply (SplitMac) or nothing at all
// (FusedMac) should be quantized and the running sum stays in the carrier's
// precision (accumulate_quant=false / fma_quant=false).
template <class T>
struct IdentityAdder
{
    using value_t = T;
    CUDA_HOST_DEVICE_INLINE T operator()(T x, PhiloxEngine & /*rng*/) const { return x; }
};

// ---------------------------------------------------------------------------
// Mac (multiply-accumulate step) policies: compute one dot-product step from
// the raw operands and the running sum, step(a, b, acc) -> acc'. SplitMac
// quantizes the product and the sum separately (two roundings per step).
// FusedMac quantizes a single fused multiply-add's result (one rounding,
// matching a real FMA unit), reusing an Adder as its result quantizer. The
// Accumulator policies below are Mac-generic (they only ever call
// Mac::step), so the same Accumulator drives either. A pairwise or tree
// reduction, if one is ever added, would be SplitMac-only: a fused
// multiply-add has no standalone product term to combine.

// The carrier's fused multiply-add: a*b + c with one rounding. The device
// intrinsic and std::fma are both correctly rounded, so the two backends
// agree bit for bit.
CUDA_HOST_DEVICE_INLINE float fma_f32(float a, float b, float c)
{
#if defined(__CUDA_ARCH__)
    return fmaf(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

CUDA_HOST_DEVICE_INLINE double fma_f64(double a, double b, double c)
{
#if defined(__CUDA_ARCH__)
    return fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

// Both halves of a mac compute in one carrier, the Adder's. The product and
// the sum are rounded to the carrier's precision before the format's cast
// sees them, 24 bits in binary32 and 53 in binary64, which is why a float64
// GEMM of float32 values is not the float32 GEMM's image, unlike the
// elementwise quantizers: two 24-bit operands have a 48-bit product, exact
// in binary64 and already rounded in binary32.
template <class Multiplier, class Adder>
struct SplitMac
{
    using value_t = typename Adder::value_t;

    Multiplier mul{};
    Adder add{};

    CUDA_HOST_DEVICE_INLINE value_t step(value_t a, value_t b, value_t acc, PhiloxEngine &rng) const
    {
        return add(acc + mul(a, b, rng), rng);
    }
};

template <class Adder>
struct FusedMac
{
    using value_t = typename Adder::value_t;

    Adder add{};

    CUDA_HOST_DEVICE_INLINE value_t step(value_t a, value_t b, value_t acc, PhiloxEngine &rng) const
    {
        if constexpr (std::is_same_v<value_t, float>)
            return add(fma_f32(a, b, acc), rng);
        else
            return add(fma_f64(a, b, acc), rng);
    }
};

// ---------------------------------------------------------------------------
// Accumulator policies: own a dot product's running reduction state. A GEMM
// kernel only ever calls seed_rng()/accumulate()/finalize() on one.
// accumulate() takes the raw operands rather than a pre-multiplied term so
// that a Mac-generic Accumulator can drive either SplitMac or FusedMac.
// NaiveAccumulator, a plain sequential sum, is the only one
// (AccumulateAlgorithm::NAIVE).
//
// seed_rng() is the RoundMode::SR lifecycle hook: the kernel calls it once
// per output element, right before that element's K-loop starts, with a
// subsequence equal to the element's global linear index into the whole
// [batch, M, N] output. Keying on that index, rather than on a thread or
// tile id, is what makes an SR result independent of the block size, the
// grid geometry and the CPU thread count, and what makes batch element 0 of
// a batched call draw exactly what the 2D call draws. Seeding cannot happen
// at construction because a CPU tile's per-element state is created in
// bulk, before it is known which element each slot will hold.
//
// There are two shapes of this. A GPU thread owns one whole Accumulator and
// keeps it in registers, which is what the object form below is for. A CPU
// tile needs one per output element, and giving each of them a whole
// Accumulator would copy the Mac policy per element; see NaiveTile.

// The raw pointers a CPU kernel reads a NaiveTile through, taken once per
// tile. The K-loop must keep the base pointers in registers, which it
// cannot do through a std::vector member that the stores in the loop body
// might alias; `__restrict__` on plain pointers says they do not.
template <class T>
struct NaiveTileView
{
    T *__restrict__ sums;
    PhiloxEngine *__restrict__ rng;
};

// Per-output-element reduction state for a CPU tile (host-only).
//
// The Mac policy is uniform across a single-format call and comes from the
// palette on a mixed one, so it should not be copied per element: a
// split-mac binaryK Accumulator is 208 B (two policies with their cast
// constants, a 44 B engine and a 4 B sum), which would make a 32x32 tile a
// 208 KB heap allocation and as many bytes of copy-construction, to carry
// one float of running sum each. NaiveTile holds the state and nothing
// else, so what the K-loop touches on every step is a dense array of sums.
//
// The sums are the carrier's (T): a binary64 tile is 8 B per sum.
template <class T>
struct NaiveTile
{
    std::vector<T> sums;
    std::vector<PhiloxEngine> streams;

    // Called once per worker task rather than per tile: the buffers are
    // reused across every tile that task takes.
    //
    // The streams are allocated whether or not RoundMode::SR is on, which
    // wastes 44 B per element under the six deterministic modes (45 KB per
    // worker task, never read). Both ways of avoiding that cost more than it
    // does: a runtime stride leaves a multiply in the innermost loop, and a
    // compile-time one duplicates the whole K-loop, which measured 7-12%
    // slower on the split macs, whose cast bodies are the largest.
    void resize(int64_t n)
    {
        sums.resize(static_cast<size_t>(n));
        streams.resize(static_cast<size_t>(n));
    }

    NaiveTileView<T> view() { return {sums.data(), streams.data()}; }

    // Zero the first `n` sums for a new tile.
    void begin(int64_t n) { std::fill_n(sums.data(), static_cast<size_t>(n), T(0)); }

    // Seed element `idx`'s stream; the kernel passes the element's global
    // linear index as the subsequence.
    void seed_rng(int64_t idx, uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
        PhiloxEngine &e = streams[static_cast<size_t>(idx)];
        e.reset_state(seed, subsequence);
        e.set_offset(offset);
    }

    T finalize(int64_t idx) const { return sums[static_cast<size_t>(idx)]; }
};

// Sequential summation: sum <- mac.step(a, b, sum) for each K-step, in
// order. The object form (mac, sum, rng) is one GPU thread's whole state;
// the static tile form is the CPU kernel's.
template <class Mac>
struct NaiveAccumulator
{
    using mac_type = Mac;
    using value_t = typename Mac::value_t;
    using tile_type = NaiveTile<value_t>;

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

    // Tile form of accumulate(), host-only: the running sum and the SR stream
    // come from the tile, the format policy from `m`, which is one shared Mac
    // for the whole call on the single-format path and this element's
    // palette slot on the mixed one. Same expression as the object form
    // above, in the same order, so the two produce identical values.
    static inline void accumulate(const NaiveTileView<value_t> &t, int64_t idx, const Mac &m,
                                  value_t a, value_t b)
    {
        t.sums[idx] = m.step(a, b, t.sums[idx], t.rng[idx]);
    }
};

// ---------------------------------------------------------------------------
// Spatially-varying (per-output-element) mixed-format support.
//
// A GEMM call may select, per output element C[row, col], which of up to
// MAX_GEMM_FORMATS precomputed Mac policies drives that element's entire
// K-reduction, without re-deriving format constants in the hot loop: every
// slot is a fully built Mac, its BinaryKParams/SuperfpParams computed at
// construction exactly like the single-format ops' acc_proto.mac. The
// kernel's per-element prologue, right where RoundMode::SR seeds its
// stream, copies the chosen slot into the Accumulator's Mac before the
// K-loop starts; the loop itself is the same code as the single-format path.
//
// FormatPalette is passed by value into the kernel alongside acc_proto, with
// the same marshalling. It is a plain array plus nothing else, because a
// kernel-argument struct with pointers or a dynamic size would need its own
// device allocation and lifetime; Mac must stay trivially copyable to ride
// in it.
//
// prec_idx is read as prec_idx[row * idx_row_stride + col * idx_col_stride]
// (plus a batch term), so one kernel path covers a dense [M, N] index
// (row_stride = N, col_stride = 1), a per-row [M, 1] index (1, 0), and a
// per-column [1, N] index (0, 1) with no branching. Which path runs is the
// kernel's MIXED template parameter, not a property of the palette: the
// single-format instantiations do not take one at all (see PaletteArg).
constexpr int MAX_GEMM_FORMATS = 8;
static_assert((MAX_GEMM_FORMATS & (MAX_GEMM_FORMATS - 1)) == 0,
              "MAX_GEMM_FORMATS must be a power of two: slot() masks with it");

// The slot count is deliberately not a member: nothing downstream reads it.
// The host validates the palette length before packing (gemm_host.h's
// check_palette_lengths, which is also what makes MIXED imply a non-empty
// palette), and slot() masks rather than bounds-checks, so a count riding
// into the kernel would be state no one can act on.
template <class Mac>
struct FormatPalette
{
    Mac slots[MAX_GEMM_FORMATS] = {};

    // Read a slot by precision index. The index is masked rather than
    // trusted: resolve_prec_idx bounds-checks the whole map host-side, but
    // that check is memoized per map (gemm_host.h), so the mask is what
    // guarantees an index outside [0, n) can only ever pick the wrong *slot*
    // and never read past the array. One AND, on a path that is compiled
    // out entirely when MIXED is false.
    CUDA_HOST_DEVICE_INLINE const Mac &slot(int32_t idx) const
    {
        return slots[idx & (MAX_GEMM_FORMATS - 1)];
    }
};

// What the kernels declare their palette parameter as: empty on the
// single-format path, the real thing on the mixed one. A FormatPalette is
// 1.4-1.5 KB (eight fully built Macs), and a kernel argument is copied into
// the parameter bank on every launch whether or not the body reads it, so a
// single-format instantiation that took one would pay that copy of zeroed
// policy per call for a prologue it compiles out.
struct NoPalette
{
};

template <bool MIXED, class Mac>
using PaletteArg = std::conditional_t<MIXED, FormatPalette<Mac>, NoPalette>;
