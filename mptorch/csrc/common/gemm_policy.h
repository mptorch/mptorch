#pragma once

#include "cast_binaryK.h"
#include "cast_superfp.h"
#include "modes.h"
#include "philox.h"
#include <cmath>
#include <cstdint>
#include <type_traits>

// ------------------------------------------------------------------------------------
// Multiplier policies: quantize a single dot-product term a*b.
//
// Each precomputes its cast function's format-derived constants
// (BinaryKParams/SuperfpParams, see cast_binaryK.h/cast_superfp.h) once at
// construction rather than on every operator() call (up to M*N*K times per
// kernel launch). round_mode is a runtime field switched on inside
// operator(), not a template parameter -- see dev/gemm_core_roadmap.md
// (GEMM kernel section, item 6) for why templating on it would blow up the
// kernel's instantiation count. Default constructors exist only so
// SplitMac/FusedMac's default member initializers stay well-formed; they're
// never actually invoked.
//
// RoundMode::SR draws one random value per call from a PhiloxEngine
// (philox.h) threaded through operator()/Mac::step alongside the raw
// operands. The engine itself lives on NaiveAccumulator below, seeded once
// per output element via seed_rng() before its K-loop starts --
// Multiplier/Adder stay stateless; only prng_bits (the width of randomness
// used) is stored here. See dev/gemm_core_roadmap.md's GEMM stochastic
// rounding section for the full design.

struct BinaryKMultiplier
{
    bool is_signed;
    SubnormalsMode subnormals_mode;
    RoundMode round_mode;
    int prng_bits;
    BinaryKParams params;

    BinaryKMultiplier() = default;
    CUDA_HOST_DEVICE_INLINE BinaryKMultiplier(int man_bits, int exp_bits, int bias, bool is_signed,
                                              SaturationMode saturation_mode, RoundMode round_mode,
                                              SubnormalsMode subnormals_mode, int prng_bits = 0)
        : is_signed(is_signed), subnormals_mode(subnormals_mode), round_mode(round_mode), prng_bits(prng_bits),
          params(make_binaryK_params(man_bits, exp_bits, bias, saturation_mode,
                                     subnormals_mode == SubnormalsMode::EXTENDED_NORMALS))
    {
    }

    CUDA_HOST_DEVICE_INLINE float operator()(float a, float b, PhiloxEngine &rng) const
    {
        float x = a * b;
        switch (round_mode)
        {
        case RoundMode::RNA:
            return cast_binaryK_nearest_away(x, is_signed, subnormals_mode, params);
        case RoundMode::RU:
            return cast_binaryK_up(x, is_signed, subnormals_mode, params);
        case RoundMode::RD:
            return cast_binaryK_down(x, is_signed, subnormals_mode, params);
        case RoundMode::RZ:
            return cast_binaryK_zero(x, is_signed, subnormals_mode, params);
        case RoundMode::RO:
            return cast_binaryK_odd(x, is_signed, subnormals_mode, params);
        case RoundMode::SR:
            return cast_binaryK_stochastic(x, rng(), prng_bits, is_signed, subnormals_mode, params);
        default: // RoundMode::RNE
            return cast_binaryK_nearest_even(x, is_signed, subnormals_mode, params);
        }
    }
};

// LEAN selects between the two SuperfpParams spellings (cast_superfp.h): the
// stored one everywhere by default, the derived one where a FormatPalette
// copies the whole policy into registers and the seven floats are what it
// cannot afford. Nothing else about the policy changes -- LEAN is not visible
// past `params`, and the two produce identical values.
template <bool LEAN = false>
struct SuperfpMultiplierT
{
    bool is_signed;
    RoundMode round_mode;
    int prng_bits;
    SuperfpParamsT<LEAN> params;

    SuperfpMultiplierT() = default;
    CUDA_HOST_DEVICE_INLINE SuperfpMultiplierT(int man_bits, int exp_bits, int normal_binades, int bias,
                                               bool is_signed, SaturationMode saturation_mode, RoundMode round_mode,
                                               int prng_bits = 0)
        : is_signed(is_signed), round_mode(round_mode), prng_bits(prng_bits),
          params(make_superfp_params<LEAN>(man_bits, exp_bits, normal_binades, bias, saturation_mode))
    {
    }

    CUDA_HOST_DEVICE_INLINE float operator()(float a, float b, PhiloxEngine &rng) const
    {
        float x = a * b;
        switch (round_mode)
        {
        case RoundMode::RNA:
            return cast_superfp_nearest_away(x, is_signed, params);
        case RoundMode::RU:
            return cast_superfp_up(x, is_signed, params);
        case RoundMode::RD:
            return cast_superfp_down(x, is_signed, params);
        case RoundMode::RZ:
            return cast_superfp_zero(x, is_signed, params);
        case RoundMode::RO:
            return cast_superfp_odd(x, is_signed, params);
        case RoundMode::SR:
            return cast_superfp_stochastic(x, rng(), prng_bits, is_signed, params);
        default: // RoundMode::RNE
            return cast_superfp_nearest_even(x, is_signed, params);
        }
    }
};

using SuperfpMultiplier = SuperfpMultiplierT<>;
using SuperfpMultiplierLean = SuperfpMultiplierT<true>;

// ------------------------------------------------------------------------------------
// Adder policies: quantize a single running-sum update. Used inside a Mac
// policy (below), either as the "add" half of a split multiply-then-add, or
// as the single quantizer applied to a fused multiply-add's result.

struct BinaryKAdder
{
    bool is_signed;
    SubnormalsMode subnormals_mode;
    RoundMode round_mode;
    int prng_bits;
    BinaryKParams params;

    BinaryKAdder() = default;
    CUDA_HOST_DEVICE_INLINE BinaryKAdder(int man_bits, int exp_bits, int bias, bool is_signed,
                                         SaturationMode saturation_mode, RoundMode round_mode,
                                         SubnormalsMode subnormals_mode, int prng_bits = 0)
        : is_signed(is_signed), subnormals_mode(subnormals_mode), round_mode(round_mode), prng_bits(prng_bits),
          params(make_binaryK_params(man_bits, exp_bits, bias, saturation_mode,
                                     subnormals_mode == SubnormalsMode::EXTENDED_NORMALS))
    {
    }

    CUDA_HOST_DEVICE_INLINE float operator()(float x, PhiloxEngine &rng) const
    {
        switch (round_mode)
        {
        case RoundMode::RNA:
            return cast_binaryK_nearest_away(x, is_signed, subnormals_mode, params);
        case RoundMode::RU:
            return cast_binaryK_up(x, is_signed, subnormals_mode, params);
        case RoundMode::RD:
            return cast_binaryK_down(x, is_signed, subnormals_mode, params);
        case RoundMode::RZ:
            return cast_binaryK_zero(x, is_signed, subnormals_mode, params);
        case RoundMode::RO:
            return cast_binaryK_odd(x, is_signed, subnormals_mode, params);
        case RoundMode::SR:
            return cast_binaryK_stochastic(x, rng(), prng_bits, is_signed, subnormals_mode, params);
        default: // RoundMode::RNE
            return cast_binaryK_nearest_even(x, is_signed, subnormals_mode, params);
        }
    }
};

// LEAN as on SuperfpMultiplierT above.
template <bool LEAN = false>
struct SuperfpAdderT
{
    bool is_signed;
    RoundMode round_mode;
    int prng_bits;
    SuperfpParamsT<LEAN> params;

    SuperfpAdderT() = default;
    CUDA_HOST_DEVICE_INLINE SuperfpAdderT(int man_bits, int exp_bits, int normal_binades, int bias,
                                          bool is_signed, SaturationMode saturation_mode, RoundMode round_mode,
                                          int prng_bits = 0)
        : is_signed(is_signed), round_mode(round_mode), prng_bits(prng_bits),
          params(make_superfp_params<LEAN>(man_bits, exp_bits, normal_binades, bias, saturation_mode))
    {
    }

    CUDA_HOST_DEVICE_INLINE float operator()(float x, PhiloxEngine &rng) const
    {
        switch (round_mode)
        {
        case RoundMode::RNA:
            return cast_superfp_nearest_away(x, is_signed, params);
        case RoundMode::RU:
            return cast_superfp_up(x, is_signed, params);
        case RoundMode::RD:
            return cast_superfp_down(x, is_signed, params);
        case RoundMode::RZ:
            return cast_superfp_zero(x, is_signed, params);
        case RoundMode::RO:
            return cast_superfp_odd(x, is_signed, params);
        case RoundMode::SR:
            return cast_superfp_stochastic(x, rng(), prng_bits, is_signed, params);
        default: // RoundMode::RNE
            return cast_superfp_nearest_even(x, is_signed, params);
        }
    }
};

using SuperfpAdder = SuperfpAdderT<>;
using SuperfpAdderLean = SuperfpAdderT<true>;

// No-op adder: used when only the multiply (Split) or the fused step
// (Fused) should be quantized and the running sum is meant to otherwise
// stay in full precision (accumulate_quant=false / fma_quant=false).
struct IdentityAdder
{
    CUDA_HOST_DEVICE_INLINE float operator()(float x, PhiloxEngine & /*rng*/) const { return x; }
};

// ------------------------------------------------------------------------------------
// Mac (multiply-accumulate step) policies: compute one dot-product step
// from the raw operands and the running sum -- step(a, b, acc) -> acc'.
// SplitMac quantizes the product and the sum separately (two roundings).
// FusedMac quantizes a single hardware-style fused multiply-add's result
// (one rounding, matching a real FMA unit), reusing Adder as its result
// quantizer. Accumulator policies below are Mac-generic (only ever call
// Mac::step), so the same Accumulator drives either.
//
// Tree-based summation (dev/gemm_core_roadmap.md) is SplitMac-only: a fused
// multiply-add has no standalone product term for a pairwise tree combiner.

CUDA_HOST_DEVICE_INLINE float fma_f32(float a, float b, float c)
{
#if defined(__CUDA_ARCH__)
    return fmaf(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

template <class Multiplier, class Adder>
struct SplitMac
{
    Multiplier mul{};
    Adder add{};

    CUDA_HOST_DEVICE_INLINE float step(float a, float b, float acc, PhiloxEngine &rng) const
    {
        return add(acc + mul(a, b, rng), rng);
    }
};

template <class Adder>
struct FusedMac
{
    Adder add{};

    CUDA_HOST_DEVICE_INLINE float step(float a, float b, float acc, PhiloxEngine &rng) const
    {
        return add(fma_f32(a, b, acc), rng);
    }
};

// ------------------------------------------------------------------------------------
// Accumulator policies: own a dot product's running reduction state. A GEMM
// kernel only ever calls seed_rng()/accumulate()/finalize() on one --
// accumulate() takes raw operands (not a pre-multiplied term) so a Mac-
// generic Accumulator can drive either SplitMac or FusedMac. Only
// NaiveAccumulator (AccumulateAlgorithm::NAIVE) is implemented so far; see
// dev/gemm_core_roadmap.md for Kahan/block/tree variants.
//
// seed_rng() is the RoundMode::SR lifecycle hook: the kernel calls it once
// per output element, right after the Accumulator is placed at its final
// coordinates and before its K-loop starts -- necessary because a CPU
// tile's Accumulator vector is copy-constructed from one prototype, so
// per-element identity isn't known until after construction.
template <class Mac>
struct NaiveAccumulator
{
    using mac_type = Mac;

    Mac mac{};
    float sum = 0.f;
    PhiloxEngine rng{};

    CUDA_HOST_DEVICE_INLINE void seed_rng(uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
        rng.reset_state(seed, subsequence);
        rng.set_offset(offset);
    }
    CUDA_HOST_DEVICE_INLINE void accumulate(float a, float b) { sum = mac.step(a, b, sum, rng); }
    CUDA_HOST_DEVICE_INLINE float finalize() const { return sum; }
};

// ------------------------------------------------------------------------------------
// Spatially-varying (per-output-element) mixed-format support.
//
// A GEMM call may select, per output element C[row, col], which of up to
// MAX_GEMM_FORMATS precomputed Mac policies drives that element's entire
// K-reduction -- the analogue of the mm_impl prototype's per-element
// precision index (temp/mm_kernel_new.h) but *without* re-deriving format
// constants in the hot loop: every slot is a fully-built Mac (its
// BinaryKParams/SuperfpParams already computed at construction, exactly
// like the single-format ops' acc_proto.mac). The kernel's existing
// per-element prologue -- right where RoundMode::SR seeds its stream --
// copies the chosen slot into the Accumulator's Mac before the K-loop
// starts; the loop itself stays byte-identical to the single-format path.
//
// FormatPalette is passed by value into the kernel alongside acc_proto
// (same marshalling). It is kept flat -- a plain array + count -- per
// dev/gemm_core_roadmap.md item 6's "keep kernel-argument policy structs
// flat" lesson; Mac must stay trivially copyable to ride in it.
//
// prec_idx is read as prec_idx[row * idx_row_stride + col * idx_col_stride]
// so one kernel path covers a dense [M, N] index (row_stride = N,
// col_stride = 1), a per-row [M, 1] index (1, 0), and a per-column [1, N]
// index (0, 1) with no branching. n == 0 selects the single-format path
// and the hook is skipped entirely.
constexpr int MAX_GEMM_FORMATS = 8;
static_assert((MAX_GEMM_FORMATS & (MAX_GEMM_FORMATS - 1)) == 0,
              "MAX_GEMM_FORMATS must be a power of two: slot() masks with it");

template <class Mac>
struct FormatPalette
{
    Mac slots[MAX_GEMM_FORMATS] = {};
    int n = 0;

    // Read a slot by precision index. The index is masked rather than
    // trusted: resolve_prec_idx bounds-checks the whole map host-side, but
    // that check is memoized per map (see custom_matmul_kernel.cu), so the
    // mask is what guarantees an index outside [0, n) can only ever pick the
    // wrong *slot* and never read past the array. One AND, on a path that is
    // compiled out entirely when MIXED is false.
    CUDA_HOST_DEVICE_INLINE const Mac &slot(int32_t idx) const
    {
        return slots[idx & (MAX_GEMM_FORMATS - 1)];
    }
};
