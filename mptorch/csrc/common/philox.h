#pragma once

#include "bit_helper.h"
#include <cstdint>

// ---------------------------------------------------------------------------
// Philox4x32-10, bit-identical to at::philox_engine
// (ATen/core/PhiloxRNGEngine.h) over the reset_state()/set_offset()/
// operator() sequence the GEMM accumulators use: same round function, same
// constants, same counter and cache bookkeeping, so a stream seeded the same
// way yields the same words in the same order.
//
// Philox is a counter-based generator: the output block is a fixed function
// of a 128-bit counter and a 64-bit key, so any block of any stream can be
// produced directly without stepping through the ones before it. That is
// what lets every output element of a GEMM own an independent stream keyed
// on its own index, at no per-element setup cost beyond writing the counter.
//
// Why not use at::philox_engine directly: it caches its four output words
// in a std::array and reads one back at a runtime index, `output_[STATE]`.
// ptxas cannot promote an array indexed at runtime to registers, so it
// places the whole enclosing object in local memory, and NaiveAccumulator
// embeds the engine alongside the running sum and the Mac policy. The
// running sum was therefore loaded and stored through local memory on every
// one of the K accumulate steps, for all seven rounding modes, including the
// six deterministic ones that never draw. Every GEMM instantiation with a
// real Adder showed a 112-168 byte stack frame; the one where the SR branch
// is unreachable (FusedMac<IdentityAdder>) showed 0. Naming the four cached
// words and selecting between them with a switch is the whole fix: with no
// array member, the accumulator stays in registers.
struct PhiloxEngine
{
    static constexpr uint32_t kPhilox10A = 0x9E3779B9;
    static constexpr uint32_t kPhilox10B = 0xBB67AE85;
    static constexpr uint32_t kPhiloxSA = 0xD2511F53;
    static constexpr uint32_t kPhiloxSB = 0xCD9E8D57;
    // at::philox_engine's default seed, so a default-constructed engine (the
    // kernels' acc_proto prototype) starts from the same state ATen's would.
    static constexpr uint64_t kDefaultSeed = 67280421310721;

    // Counter (c0..c3), key (k0, k1) and output cache (o0..o3) as named
    // scalars rather than arrays, for the reason above. `state` is the index
    // of the next cached word to hand out, 0..3.
    uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    uint32_t k0 = static_cast<uint32_t>(kDefaultSeed);
    uint32_t k1 = static_cast<uint32_t>(kDefaultSeed >> 32);
    uint32_t o0 = 0, o1 = 0, o2 = 0, o3 = 0;
    uint32_t state = 0;

    // Start the stream (seed, subsequence) from its first block: the seed is
    // the key, the subsequence the high 64 bits of the counter, and the low
    // 64 bits (the block offset) are zero until set_offset().
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

    // One 32-bit value per call, regenerating the 4-word cache every fourth
    // call, with the same bookkeeping as at::philox_engine.
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

    // A binary64 GEMM's draw: the next two values of the stream, low word
    // first. Two statements, because the order in which two calls in one
    // expression are evaluated is unspecified. A draw may straddle two
    // blocks, which the stream's own bookkeeping handles; the words are the
    // stream's, in the order a binary32 GEMM would have read them.
    CUDA_HOST_DEVICE_INLINE uint64_t next64()
    {
        const uint64_t lo = (*this)();
        const uint64_t hi = (*this)();
        return (hi << 32) | lo;
    }

private:
    // The low 32 bits of a*b, with the high 32 in *result_high.
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

    // One Philox round: two 32x32 multiplies and the key mixed in, with the
    // word permutation at::philox_engine uses.
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

    // Fill the cache from the current counter: 10 rounds, the key bumped
    // between all but the last, as at::philox_engine's rand() does with its
    // default n_rounds = 10.
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

    // Advance the 128-bit counter by one block, carrying across the words.
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
// one 10-round generate over four consecutive elements, which is also
// exactly one vectorized float lane group, so the vector path costs one
// generate per iteration for float and two for half/bfloat16.
//
// Named scalars and a switch rather than an array, for the reason at the top
// of this file: an array member read at a runtime index cannot be promoted
// to registers, and would put the caller's whole frame in local memory.
//
// A binary64 element draws a 64-bit word, two of the block's, so a block
// covers two elements rather than four. The layout is the same rule at
// either width: element `j` of a carrier whose draw is WPE 32-bit words
// long takes block `j / (4 / WPE)`, words from `(j % (4 / WPE)) * WPE` on,
// low word first. For binary32 that is exactly the `j >> 2`, `j & 3` above,
// so no float stream moves; for binary64 it is block `j >> 1`, words
// `2 * (j & 1)` and the one after, one block per double2 vector, as one
// block is one float4 vector's.
struct PhiloxBlock
{
    uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;

    // Word `k & 3` of the block.
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

    // Words `k` and `k + 1` as one 64-bit draw, low word first; `k` is 0 or 2.
    CUDA_HOST_DEVICE_INLINE uint64_t word64(int k) const
    {
        return (k & 2) ? ((uint64_t)w3 << 32) | w2 : ((uint64_t)w1 << 32) | w0;
    }
};

// Generate block `offset` of the (seed, subsequence) stream. Same counter
// layout as PhiloxEngine's own, (offset, subsequence) as the 128-bit counter
// and seed as the key, so blocks drawn here and streams drawn through the
// engine never collide as long as their subsequences differ. That is the
// standard ATen convention: one subsequence per element or per thread, and
// `counter_offset` blocks of `offset` reserved per call.
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
