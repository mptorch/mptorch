#pragma once

#include "bit_helper.h"
#if !defined(__METAL_VERSION__)
#include <cstdint>
#endif

// ---------------------------------------------------------------------------
// Philox4x32-10, bit-identical to at::philox_engine
// (ATen/core/PhiloxRNGEngine.h) for the sequence of reset_state(),
// set_offset() and operator() calls the GEMM accumulators make: the same
// round function, constants, counter and output-cache bookkeeping, so a
// stream seeded the same way yields the same words in the same order.
//
// Philox is a counter-based generator: each 128-bit output block is a fixed
// function of a 128-bit counter and a 64-bit key, so any block of any stream
// can be computed directly, without generating the blocks before it. That is
// what lets every output element of a GEMM have an independent stream keyed
// on its own index, at no setup cost beyond writing the counter.
//
// Why not use at::philox_engine itself: it caches its four output words in a
// std::array and reads them back at a run-time index, `output_[STATE]`.
// ptxas cannot keep an array indexed at run time in registers, so it puts
// the whole enclosing object in local memory, and NaiveAccumulator holds the
// engine next to the running sum and the Mac policy. The running sum was
// then loaded from and stored to local memory on every one of the K
// accumulate steps, in all seven rounding modes, the six that never draw
// included: every GEMM instantiation with a real Adder had a 112-168 byte
// stack frame, and the one where the SR branch cannot be reached
// (FusedMac<IdentityAdder>) had none. Naming the four cached words and
// choosing between them with a switch is the whole fix: with no array member,
// the accumulator stays in registers.
struct PhiloxEngine
{
    static constexpr MPTORCH_CONSTANT uint32_t kPhilox10A = 0x9E3779B9;
    static constexpr MPTORCH_CONSTANT uint32_t kPhilox10B = 0xBB67AE85;
    static constexpr MPTORCH_CONSTANT uint32_t kPhiloxSA = 0xD2511F53;
    static constexpr MPTORCH_CONSTANT uint32_t kPhiloxSB = 0xCD9E8D57;
    // at::philox_engine's default seed, so a default-constructed engine (the
    // kernels' acc_proto prototype) starts from the same state ATen's would.
    static constexpr MPTORCH_CONSTANT uint64_t kDefaultSeed = 67280421310721;

    // The counter (c0..c3), key (k0, k1) and output cache (o0..o3), as named
    // scalars rather than arrays, for the reason above. `state` is the index
    // of the next cached word to return, 0 to 3.
    uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    uint32_t k0 = static_cast<uint32_t>(kDefaultSeed);
    uint32_t k1 = static_cast<uint32_t>(kDefaultSeed >> 32);
    uint32_t o0 = 0, o1 = 0, o2 = 0, o3 = 0;
    uint32_t state = 0;

    // Starts the stream (seed, subsequence) at its first block: the seed is
    // the key, the subsequence the counter's high 64 bits, and its low 64
    // bits (the block offset) are zero until set_offset().
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

    // Moves to the given 128-bit block of this subsequence.
    CUDA_HOST_DEVICE_INLINE void set_offset(uint64_t offset)
    {
        c0 = static_cast<uint32_t>(offset);
        c1 = static_cast<uint32_t>(offset >> 32);
    }

    // Returns one 32-bit word per call, refilling the four-word cache every
    // fourth call, with at::philox_engine's bookkeeping.
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

    // A binary64 GEMM's draw: the next two words of the stream, low word
    // first. They are read in two statements because the order in which two
    // calls in one expression are evaluated is unspecified. A draw may span
    // two blocks, which the stream's bookkeeping handles; the words are the
    // ones a binary32 GEMM would have read, in the same order.
    CUDA_HOST_DEVICE_INLINE uint64_t next64()
    {
        const uint64_t lo = (*this)();
        const uint64_t hi = (*this)();
        return (hi << 32) | lo;
    }

private:
    // The low 32 bits of a * b; the high 32 go to *result_high.
    CUDA_HOST_DEVICE_INLINE static uint32_t mulhilo32(uint32_t a, uint32_t b, MPTORCH_THREAD uint32_t *result_high)
    {
#if defined(__CUDA_ARCH__)
        *result_high = __umulhi(a, b);
        return a * b;
#elif defined(__METAL_VERSION__)
        *result_high = metal::mulhi(a, b);
        return a * b;
#else
        const uint64_t product = static_cast<uint64_t>(a) * b;
        *result_high = static_cast<uint32_t>(product >> 32);
        return static_cast<uint32_t>(product);
#endif
    }

    // One Philox round: two 32 x 32-bit multiplies, the key mixed in, and
    // at::philox_engine's permutation of the words.
    CUDA_HOST_DEVICE_INLINE static void single_round(MPTORCH_THREAD uint32_t &x0, MPTORCH_THREAD uint32_t &x1,
                                                     MPTORCH_THREAD uint32_t &x2, MPTORCH_THREAD uint32_t &x3,
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

    // Fills the cache from the current counter: 10 rounds, with the key
    // incremented between consecutive rounds, as at::philox_engine's rand()
    // does with its default n_rounds = 10.
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

    // Advances the 128-bit counter by one block, carrying across the words.
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

// The four words of one Philox block, read by position rather than as a
// stream.
//
// This is what the elementwise SR quantizers need, and it is a different
// access pattern from the GEMM's. In a GEMM one thread owns one output
// element and draws K (or 2K) words in sequence, which PhiloxEngine's stream
// interface fits. In an elementwise quantizer every element takes exactly
// one random word, so a stream per element would discard three words of
// every block it generates. Keying the block on `index >> 2` instead and
// taking word `index & 3` spreads one 10-round generation over four
// consecutive elements, which is also exactly one float4 vector, so the
// vectorized loop runs one generation per iteration for float and two for
// half and bfloat16.
//
// Named scalars and a switch rather than an array, for the reason given at
// the top of this file: an array member read at a run-time index cannot be
// kept in registers, and would put the caller's whole frame in local memory.
//
// A binary64 element draws a 64-bit word, two of the block's, so a block
// covers two elements rather than four. One rule gives the layout at either
// width: element `j` of a carrier whose draw is WPE 32-bit words long takes
// block `j / (4 / WPE)`, words `(j % (4 / WPE)) * WPE` onward, low word
// first. For binary32 that is the `j >> 2` and `j & 3` above; for binary64 it
// is block `j >> 1`, words `2 * (j & 1)` and the next, one block per double2
// vector, as one block serves one float4 vector.
struct PhiloxBlock
{
    uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;

    // Returns word `k & 3` of the block.
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

    // Returns words `k` and `k + 1` as one 64-bit draw, low word first; `k`
    // is 0 or 2.
    CUDA_HOST_DEVICE_INLINE uint64_t word64(int k) const
    {
        return (k & 2) ? ((uint64_t)w3 << 32) | w2 : ((uint64_t)w1 << 32) | w0;
    }
};

// Generates block `offset` of the (seed, subsequence) stream. The counter
// layout is PhiloxEngine's: (offset, subsequence) as the 128-bit counter and
// the seed as the key, so blocks drawn here and streams drawn through the
// engine never collide as long as their subsequences differ. That is ATen's
// usual convention: one subsequence per element or per thread, and
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
