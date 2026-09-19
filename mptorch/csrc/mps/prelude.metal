// The head of every Metal library the MPS backend compiles: what the shared
// headers under common/ take from the C++ standard library, spelled with
// Metal's, then those headers themselves, then the loads and stores the
// kernels share. setup.py resolves the quoted includes below into one string
// (mps/metal_source.h), and the backend appends a few lines naming one
// instantiation (mps/metal_runtime.mm) and then gemm.metal or quantize.metal.

#include <metal_stdlib>

// The build's two promises to the fast casts, made the way setup.py makes
// them to the host compiler (bit_helper.h, MPTORCH_FAST_CAST): no
// contraction of separately written float operations, and no fast math, and
// only then the macro that admits the Veltkamp split. The library's compile
// options ask for safe math as well (metal_runtime.mm); these pragmas keep
// the promise in the source, where a library compiled some other way still
// carries it.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#define MPTORCH_FAST_CAST 1

// The library is compiled as MSL 3.1 (metal_runtime.mm says why), which is
// C++14. The headers use two C++17 features, `if constexpr` and nested
// namespace definitions, which the compiler takes in C++14 as extensions
// with the C++17 meaning, and warns about at every use; the warnings would
// bury a real error in a failed compile's message.
#pragma clang diagnostic ignored "-Wc++17-extensions"

#ifndef INT32_MIN
#define INT32_MIN (-2147483647 - 1)
#endif

namespace std
{
using metal::conditional_t;
using metal::is_same_v;
using metal::make_signed_t;
} // namespace std

// Called by name in the casts (bit_helper.h says why). Both are bit
// operations on the word, as they are on every other backend, so a NaN keeps
// its payload and a subnormal is not flushed.
inline float fabsf(float x) { return metal::fabs(x); }
inline float copysignf(float x, float y) { return metal::copysign(x, y); }

#include "../common/gemm_args.h"
#include "launch_params.h"

namespace mptorch_mps
{

// Storage dtype <-> the binary32 carrier, bit for bit what the CPU backend's
// static_casts do, so that a result stored to a narrow tensor is the one the
// CPU stores.
//
// float is the carrier itself. bfloat16 is done on the word, as c10 does it:
// widening is a shift, and narrowing is round-to-nearest-even on the word
// with every NaN becoming 0x7FC0 (c10's round_to_nearest_even). The GPU's
// own conversion would flush a binary32 subnormal before rounding it.
//
// half is c10's too, which on arm64 is the CPU's `fcvt` instruction: IEEE
// conversion, round to nearest even, with half's subnormals. The GPU's
// conversion is the same arithmetic, and differs from `fcvt` only on a NaN:
// `fcvt` quiets a signalling one and keeps its payload (the top bits of it,
// narrowing), where the GPU hands a signalling NaN back signalling. Both
// directions therefore take NaNs on the word, as `fcvt` does, and every
// other value through the GPU. tests/test_mps.py runs every half word
// through both.
inline float to_carrier(float x) { return x; }
inline float to_carrier(bfloat x) { return as_type<float>(uint32_t(as_type<ushort>(x)) << 16); }
inline float to_carrier(half x)
{
    const uint32_t h = as_type<ushort>(x);
    if ((h & 0x7FFFu) > 0x7C00u)
        return as_type<float>(((h & 0x8000u) << 16) | 0x7FC00000u | ((h & 0x1FFu) << 13));
    return float(x);
}

template <class S>
inline S from_carrier(float x);

template <>
inline float from_carrier<float>(float x) { return x; }

template <>
inline half from_carrier<half>(float x)
{
    const uint32_t w = as_type<uint32_t>(x);
    if ((w & 0x7FFFFFFFu) > 0x7F800000u)
        return as_type<half>(ushort(((w >> 16) & 0x8000u) | 0x7E00u | ((w >> 13) & 0x1FFu)));
    return half(x);
}

template <>
inline bfloat from_carrier<bfloat>(float x)
{
    const uint32_t w = as_type<uint32_t>(x);
    if ((w & 0x7FFFFFFFu) > 0x7F800000u)
        return as_type<bfloat>(ushort(0x7FC0));
    return as_type<bfloat>(ushort((w + (((w >> 16) & 1u) + 0x7FFFu)) >> 16));
}

} // namespace mptorch_mps
