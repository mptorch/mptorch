#pragma once

#include "cast_binaryK.h"
#include "cast_superfp.h"
#include "modes.h"
#include <cmath>

// ------------------------------------------------------------------------------------
// Multiplier policies: quantize a single dot-product term a*b.

struct BinaryKMultiplier
{
    int man_bits, exp_bits, bias;
    bool is_signed;
    SaturationMode saturation_mode;
    SubnormalsMode subnormals_mode;

    CUDA_HOST_DEVICE_INLINE float operator()(float a, float b) const
    {
        return cast_binaryK_nearest_even(a * b, man_bits, exp_bits, bias, is_signed,
                                          saturation_mode, subnormals_mode);
    }
};

struct SuperfpMultiplier
{
    int man_bits, exp_bits, normal_binades, bias;
    bool is_signed;
    SaturationMode saturation_mode;

    CUDA_HOST_DEVICE_INLINE float operator()(float a, float b) const
    {
        return cast_superfp_nearest_even(a * b, man_bits, exp_bits, normal_binades, bias,
                                         is_signed, saturation_mode);
    }
};

// ------------------------------------------------------------------------------------
// Adder policies: quantize a single running-sum update. Used inside a Mac
// policy (below), either as the "add" half of a split multiply-then-add, or
// as the single quantizer applied to a fused multiply-add's result.

struct BinaryKAdder
{
    int man_bits, exp_bits, bias;
    bool is_signed;
    SaturationMode saturation_mode;
    SubnormalsMode subnormals_mode;

    CUDA_HOST_DEVICE_INLINE float operator()(float x) const
    {
        return cast_binaryK_nearest_even(x, man_bits, exp_bits, bias, is_signed,
                                         saturation_mode, subnormals_mode);
    }
};

struct SuperfpAdder
{
    int man_bits, exp_bits, normal_binades, bias;
    bool is_signed;
    SaturationMode saturation_mode;

    CUDA_HOST_DEVICE_INLINE float operator()(float x) const
    {
        return cast_superfp_nearest_even(x, man_bits, exp_bits, normal_binades, bias,
                                         is_signed, saturation_mode);
    }
};

// No-op adder: used when only the multiply (Split) or the fused step
// (Fused) should be quantized and the running sum is meant to otherwise
// stay in full precision (accumulate_quant=false / fma_quant=false).
struct IdentityAdder
{
    CUDA_HOST_DEVICE_INLINE float operator()(float x) const { return x; }
};

// ------------------------------------------------------------------------------------
// Mac (multiply-accumulate step) policies: compute one dot-product step
// from the raw operands and the running sum -- step(a, b, acc) -> acc'.
// Accumulator policies (below) are Mac-generic: they never multiply
// operands themselves, only ever call Mac::step, so the same Accumulator
// works for either arithmetic style.
//
// SplitMac quantizes the product and the sum separately (two roundings) --
// this is the original design. FusedMac quantizes a single hardware-style
// fused multiply-add's result (one rounding, matching a real FMA unit and
// dev/cuda/custom_matmul_fma.cu); it reuses Adder as its result quantizer,
// since quantizing an FMA's result is the same operation as quantizing a
// running sum, just applied to a different raw value.
//
// Tree-based summation (see dev/gemm_core_roadmap.md) is SplitMac-only: a
// fused multiply-add has no standalone product term to hand to a pairwise
// tree combiner.

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

    CUDA_HOST_DEVICE_INLINE float step(float a, float b, float acc) const
    {
        return add(acc + mul(a, b));
    }
};

template <class Adder>
struct FusedMac
{
    Adder add{};

    CUDA_HOST_DEVICE_INLINE float step(float a, float b, float acc) const
    {
        return add(fma_f32(a, b, acc));
    }
};

// ------------------------------------------------------------------------------------
// Accumulator policies: own a dot product's running reduction state.
//
// A GEMM kernel only ever calls accumulate()/finalize() on an Accumulator --
// new accumulation algorithms (Kahan-compensated summation, block
// summation, tree summation, ...) are added by writing a new policy with
// this same two-method interface, without touching any kernel code.
// accumulate() takes the raw dot-product operands (not a pre-multiplied
// term), precisely so a Mac-generic Accumulator can drive either SplitMac
// or FusedMac. Only NaiveAccumulator (mirroring AccumulateAlgorithm::NAIVE)
// is implemented so far; see dev/gemm_core_roadmap.md for the rest.

template <class Mac>
struct NaiveAccumulator
{
    Mac mac{};
    float sum = 0.f;

    CUDA_HOST_DEVICE_INLINE void accumulate(float a, float b) { sum = mac.step(a, b, sum); }
    CUDA_HOST_DEVICE_INLINE float finalize() const { return sum; }
};
