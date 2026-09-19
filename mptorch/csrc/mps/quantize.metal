// The elementwise quantizers, binaryK_quant and superfp_quant: every element
// rounded to the format by the cast the CPU backend calls, in its carrier
// (binary32, since MPS has no float64), and stored back in the tensor's
// dtype as the CPU stores it (prelude.metal's from_carrier).
//
// Compiled once per instantiation, after prelude.metal and a few lines the
// backend writes (quantize.cpp) that define
//
//   mpt_storage_t     float, half or bfloat
//   MPT_ROUND_MODE    the RoundMode
//   MPT_SUPERFP       1 for superfp_quant, 0 for binaryK_quant
//   MPT_IS_SIGNED     the format's sign, as the CPU's IsSigned template
//                     parameter is, so an unsigned format's early return
//                     folds away on a signed one
//   MPT_SUBNORMALS    binaryK's SubnormalsMode
//   MPT_PRNG_BITS     SR's random bits
//   mpt_params()      make_binaryK_params / make_superfp_params of the
//                     format, with its widths as literals, so its constants
//                     fold (gemm.metal says why that matters)
//
// Each thread takes four consecutive elements, the four that share a Philox
// block under RoundMode::SR: element i takes word i & 3 of block i >> 2 of
// the stream keyed on the CPU generator's seed, which is the CPU kernel's
// layout (cpu/utils.h's quant_kernel_sr), so an SR result is the CPU's.
//
// The casts round on the word, and the few places where they compare or
// compute on the value itself (their sign tests, SR's add into the floor's
// binade, and SR's bound on a NORMALS floor) are spelled so that the GPU's
// flush of binary32 subnormals cannot reach them (bit_helper.h, "The Apple
// GPU's flush of binary32 subnormals"). So every input rounds as on the
// CPU, subnormal ones included, in every mode: dev/benchmarks/mps_backend.py
// sweeps all 2^32 words to that effect.

namespace mptorch_mps
{

inline float quantize_one(float x, uint32_t draw)
{
    const auto p = mpt_params();
    constexpr RoundMode RM = MPT_ROUND_MODE;
    constexpr bool S = MPT_IS_SIGNED;
#if MPT_SUPERFP
    if constexpr (RM == RoundMode::SR)
        return cast_superfp_stochastic(x, draw, MPT_PRNG_BITS, S, p);
    else if constexpr (RM == RoundMode::RNA)
        return cast_superfp_nearest_away(x, S, p);
    else if constexpr (RM == RoundMode::RU)
        return cast_superfp_up(x, S, p);
    else if constexpr (RM == RoundMode::RD)
        return cast_superfp_down(x, S, p);
    else if constexpr (RM == RoundMode::RZ)
        return cast_superfp_zero(x, S, p);
    else if constexpr (RM == RoundMode::RO)
        return cast_superfp_odd(x, S, p);
    else
        return cast_superfp_nearest_even(x, S, p);
#else
    constexpr SubnormalsMode SM = MPT_SUBNORMALS;
    if constexpr (RM == RoundMode::SR)
        return cast_binaryK_stochastic(x, draw, MPT_PRNG_BITS, S, SM, p);
    else if constexpr (RM == RoundMode::RNA)
        return cast_binaryK_nearest_away(x, S, SM, p);
    else if constexpr (RM == RoundMode::RU)
        return cast_binaryK_up(x, S, SM, p);
    else if constexpr (RM == RoundMode::RD)
        return cast_binaryK_down(x, S, SM, p);
    else if constexpr (RM == RoundMode::RZ)
        return cast_binaryK_zero(x, S, SM, p);
    else if constexpr (RM == RoundMode::RO)
        return cast_binaryK_odd(x, S, SM, p);
    else
        return cast_binaryK_nearest_even(x, S, SM, p);
#endif
}

} // namespace mptorch_mps

kernel void mpt_quantize(device const mpt_storage_t *x [[buffer(0)]],
                         device mpt_storage_t *y [[buffer(1)]],
                         constant mptorch_mps::QuantLaunch &q [[buffer(2)]],
                         uint t [[thread_position_in_grid]])
{
    const uint64_t first = (uint64_t)t * 4;
    PhiloxBlock block;
    if constexpr (MPT_ROUND_MODE == RoundMode::SR)
        block = philox_block(q.seed, t, 0);
    for (int l = 0; l < 4; ++l)
    {
        const uint64_t i = first + l;
        if (i >= q.n)
            return;
        y[i] = mptorch_mps::from_carrier<mpt_storage_t>(
            mptorch_mps::quantize_one(mptorch_mps::to_carrier(x[i]), block.word(l)));
    }
}
