#pragma once

#include "bit_helper.h"
#include <cstdint>

// ------------------------------------------------------------------------------------
// Philox4x32-10, bit-identical to at::philox_engine (ATen/core/PhiloxRNGEngine.h)
// over the reset_state()/set_offset()/operator() sequence the GEMM accumulators
// use: same round function, same constants, same counter and cache bookkeeping,
// so a stream seeded the same way yields the same words in the same order.
//
// Why not just use at::philox_engine, which this replaces verbatim: it caches
// its four output words in a std::array and reads one back as
// output_[static_cast<int>(STATE)] -- a *dynamically indexed array member*.
// ptxas cannot promote such an array to registers, so it places the whole
// enclosing object in local memory, and NaiveAccumulator embeds the engine
// alongside the running sum and the Mac policy. The running sum was therefore
// loaded and stored through local memory on every one of the K accumulate
// steps -- for all seven rounding modes, including the six deterministic ones
// that never draw a random number. Every shipped GEMM instantiation with a
// real Adder reported a 112-168 byte stack frame; the one where the SR branch
// is unreachable (FusedMac<IdentityAdder>) reported 0.
//
// Naming the four cached words and selecting between them with a switch is the
// whole fix: no array member, so the accumulator stays in registers. See
// dev/gemm_perf_audit.md (finding G1) for the measurements.
struct PhiloxEngine
{
    static constexpr uint32_t kPhilox10A = 0x9E3779B9;
    static constexpr uint32_t kPhilox10B = 0xBB67AE85;
    static constexpr uint32_t kPhiloxSA = 0xD2511F53;
    static constexpr uint32_t kPhiloxSB = 0xCD9E8D57;
    // at::philox_engine's default seed, kept so a default-constructed engine
    // (the acc_proto prototype's) starts from the same state it used to.
    static constexpr uint64_t kDefaultSeed = 67280421310721;

    // Counter, key and output cache as named scalars rather than arrays.
    uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    uint32_t k0 = static_cast<uint32_t>(kDefaultSeed);
    uint32_t k1 = static_cast<uint32_t>(kDefaultSeed >> 32);
    uint32_t o0 = 0, o1 = 0, o2 = 0, o3 = 0;
    uint32_t state = 0;

    CUDA_HOST_DEVICE_INLINE void reset_state(uint64_t seed = kDefaultSeed, uint64_t subsequence = 0)
    {
        k0 = static_cast<uint32_t>(seed);
        k1 = static_cast<uint32_t>(seed >> 32);
        c0 = 0;
        c1 = 0;
        c2 = static_cast<uint32_t>(subsequence);
        c3 = static_cast<uint32_t>(subsequence >> 32);
        state = 0;
    }

    // Skip to the given 128-bit block of this subsequence.
    CUDA_HOST_DEVICE_INLINE void set_offset(uint64_t offset)
    {
        c0 = static_cast<uint32_t>(offset);
        c1 = static_cast<uint32_t>(offset >> 32);
    }

    // One unique 32-bit value per call, regenerating the 4-word cache every
    // fourth call (matching at::philox_engine's own bookkeeping).
    CUDA_HOST_DEVICE_INLINE uint32_t operator()()
    {
        if (state == 0)
        {
            generate();
            incr();
        }
        uint32_t ret;
        switch (state)
        {
        case 0:
            ret = o0;
            break;
        case 1:
            ret = o1;
            break;
        case 2:
            ret = o2;
            break;
        default:
            ret = o3;
            break;
        }
        state = (state + 1) & 3u;
        return ret;
    }

private:
    CUDA_HOST_DEVICE_INLINE static uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t *result_high)
    {
#if defined(__CUDA_ARCH__)
        *result_high = __umulhi(a, b);
        return a * b;
#else
        const uint64_t product = static_cast<uint64_t>(a) * b;
        *result_high = static_cast<uint32_t>(product >> 32);
        return static_cast<uint32_t>(product);
#endif
    }

    CUDA_HOST_DEVICE_INLINE static void single_round(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3,
                                                     uint32_t key0, uint32_t key1)
    {
        uint32_t hi0 = 0, hi1 = 0;
        uint32_t lo0 = mulhilo32(kPhiloxSA, x0, &hi0);
        uint32_t lo1 = mulhilo32(kPhiloxSB, x2, &hi1);
        uint32_t r0 = hi1 ^ x1 ^ key0;
        uint32_t r2 = hi0 ^ x3 ^ key1;
        x0 = r0;
        x1 = lo1;
        x2 = r2;
        x3 = lo0;
    }

    // 10 rounds, the key bumped between all but the last -- at::philox_engine's
    // rand() with its back-compat default n_rounds = 10.
    CUDA_HOST_DEVICE_INLINE void generate()
    {
        uint32_t x0 = c0, x1 = c1, x2 = c2, x3 = c3;
        uint32_t key0 = k0, key1 = k1;
        for (int round = 0; round < 9; ++round)
        {
            single_round(x0, x1, x2, x3, key0, key1);
            key0 += kPhilox10A;
            key1 += kPhilox10B;
        }
        single_round(x0, x1, x2, x3, key0, key1);
        o0 = x0;
        o1 = x1;
        o2 = x2;
        o3 = x3;
    }

    // Advance the counter by one 128-bit block.
    CUDA_HOST_DEVICE_INLINE void incr()
    {
        if (++c0)
            return;
        if (++c1)
            return;
        if (++c2)
            return;
        ++c3;
    }
};

// The four words of one Philox 128-bit block, addressed by position rather
// than consumed as a stream.
//
// This is what the *elementwise* SR quantizers want, and it is a different
// access pattern from the GEMM's. There, one thread owns one output element
// and draws K (or 2K) values in sequence, so PhiloxEngine's stream interface
// fits. Here every element consumes exactly one random value, so a stream per
// element would throw away three words of every block it generates. Keying
// the block on `index >> 2` instead and picking word `index & 3` amortizes
// one 10-round generate over four consecutive elements -- which is also
// exactly one vectorized float lane group, so the vector path costs one
// generate per iteration for float and two for half/bfloat16.
//
// Named scalars and a switch rather than an array, for the reason at the top
// of this file: an array member read at a runtime index cannot be promoted
// to registers, and would put the caller's whole frame in local memory.
struct PhiloxBlock
{
    uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;

    CUDA_HOST_DEVICE_INLINE uint32_t word(int k) const
    {
        switch (k & 3)
        {
        case 0:
            return w0;
        case 1:
            return w1;
        case 2:
            return w2;
        default:
            return w3;
        }
    }
};

// Generate block `subsequence` of the (seed, offset) stream. Same counter
// layout as PhiloxEngine's own -- (offset, subsequence) as the 128-bit
// counter, seed as the key -- so blocks drawn here and streams drawn through
// the engine never collide as long as their subsequences differ, which is
// the standard ATen convention (one subsequence per element or per thread,
// `counter_offset` blocks of `offset` reserved per call).
CUDA_HOST_DEVICE_INLINE PhiloxBlock philox_block(uint64_t seed, uint64_t subsequence,
                                                 uint64_t offset)
{
    PhiloxEngine e;
    e.reset_state(seed, subsequence);
    e.set_offset(offset);
    PhiloxBlock b;
    b.w0 = e();
    b.w1 = e();
    b.w2 = e();
    b.w3 = e();
    return b;
}
