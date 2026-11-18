#pragma once

// The three accumulate algorithms besides NAIVE (modes.h's
// AccumulateAlgorithm): KAHAN, BLOCK and TREE. They are accumulator policies
// that drive the same Mac policies as NaiveAccumulator (gemm_policy.h). This
// header also holds AccumulateArgs, which builds them from the arguments of
// an *_accumulated op, the version of a single-format op that takes these
// algorithms.
//
// gemm_policy.h and gemm_args.h deliberately do not include this header.
// The NAIVE kernels are compiled from those two, by three compilers, and
// nothing here may change what they compile to. The CUDA and CPU kernel
// headers include this one, and recognize its accumulators by their
// `positional` member; a NaiveAccumulator, which has none, takes exactly the
// code path it would take if this header did not exist. The Metal prelude
// includes it too: mps/gemm.metal drives these accumulators through the same
// contract, and recognizes them with a trait of its own, since
// is_positional_accumulator_v is a C++20 requires-expression, which Metal's
// C++14 lacks. So this header is written like the other common/ headers, with
// MPTORCH_THREAD on references, and function objects instead of lambdas.
//
// What they compute. `mul` and `add` are the Mac's two roundings (`add` is
// the identity when the accumulate format is off), and `outer` is a third
// (the identity when no outer format is given). k runs over the K steps of
// one output element's dot product, in order. Under RoundMode::SR, every
// rounding that is not the identity draws once from the element's random
// stream, in the order in which the expressions below are written.
//
//   KAHAN   y = add(mul(a*b) - c)         split;  add(fma(a, b, -c)) fused
//           t = add(s + y)
//           c = add(add(t - s) - y)
//           s = t                          result: s
//
//   BLOCK   blk = Mac::step(a, b, blk)     NAIVE's step, on the block's sum
//           after step k, if (k+1) % block_size == 0 or k+1 == K:
//             tot = outer(tot + blk); blk = 0          result: tot
//
//   TREE    (split macs only) p_i = mul(a*b) is product i of its block, which
//           has block_size = 2^L products. It is merged upward through the
//           levels that the trailing one bits of i select,
//           v = add(level[l] + v), and stored at the first level whose bit
//           of i is zero. This is a binary counter, which sums the block
//           pairwise, (p0+p1)+(p2+p3), in the order the products arrive.
//           Product 2^L - 1 merges through all L levels, and the result is
//           the block's root: tot = outer(tot + root). A last, partial block
//           is flushed at K: its occupied levels are merged lowest first,
//           v = add(level[l] + v), and the result is added to tot in the same
//           way.                                      result: tot
//
// Nothing is padded: step k exists only for k < K, on every backend, so a
// result does not depend on the slabs of 16 steps in which the CUDA kernel
// stages its operands. (Padding the end with zero products, which costs NAIVE
// nothing, would change KAHAN's result: a zero product still applies the
// pending compensation.) No result is -0.0: every sum above starts from +0.0,
// a cast never returns -0.0, and x + y is -0.0 only when both are.
//
// The contract with the kernel is NaiveAccumulator's plus a position (the
// "positional contract"):
//
//   accumulate(a, b, pos)   pos = k & 15, the step's position in its slab.
//                           BLOCK adds a finished block to the total here
//                           when block_size divides 16, and TREE walks its
//                           first four levels.
//   end_slab(k_end, K)      after step k_end - 1, when k_end % 16 == 0 or
//                           k_end == K, so at most once per 16 steps. BLOCK
//                           adds a finished block here when block_size is a
//                           multiple of 16, TREE walks its levels from 4 up,
//                           and both handle the last, partial block.
//
// This is why BLOCK's block_size must divide 16 or be a multiple of 16: the
// test for the end of a block is then a mask per step or a remainder per
// slab, never a division per step. Every array below is indexed by
// constants only (the walks over the levels are written out), so a GPU
// thread keeps the whole state in registers; an array indexed at run time
// would be placed in local memory (see the note on PhiloxEngine in
// philox.h).
//
// The state per output element is two carrier values for KAHAN, two for
// BLOCK and ten for TREE.

#include "gemm_args.h"

// ---------------------------------------------------------------------------
// The parts of a Mac that the algorithms use separately. They are free
// functions rather than members of SplitMac and FusedMac, so that
// gemm_policy.h, and the NAIVE kernels built from it, do not change for
// them.

// KAHAN's term: the product minus the running compensation, rounded to the
// accumulate format. A fused mac computes it with one FMA, as its step
// does.
template <class Multiplier, class Adder>
CUDA_HOST_DEVICE_INLINE typename Adder::value_t
mac_term(const MPTORCH_THREAD SplitMac<Multiplier, Adder> &m, typename Adder::value_t a,
         typename Adder::value_t b, typename Adder::value_t c, MPTORCH_THREAD PhiloxEngine &rng)
{
    return m.add(m.mul(a, b, rng) - c, rng);
}

template <class Adder>
CUDA_HOST_DEVICE_INLINE typename Adder::value_t
mac_term(const MPTORCH_THREAD FusedMac<Adder> &m, typename Adder::value_t a,
         typename Adder::value_t b, typename Adder::value_t c, MPTORCH_THREAD PhiloxEngine &rng)
{
    if constexpr (std::is_same_v<typename Adder::value_t, float>)
        return m.add(fma_f32(a, b, -c), rng);
#if !defined(__METAL_VERSION__)
    else
        return m.add(fma_f64(a, b, -c), rng);
#endif
}

// Whether a Mac is a fused mac, whose step rounds once where a split mac's
// rounds twice. BLOCK's unroll factor depends on it.
template <class Mac>
struct is_fused_mac
{
    static constexpr MPTORCH_CONSTANT bool value = false;
};

template <class Adder>
struct is_fused_mac<FusedMac<Adder>>
{
    static constexpr MPTORCH_CONSTANT bool value = true;
};

// Whether a Mac's carrier operations round to odd (carrier_rounding=
// RoundMode.RO). It is false here, and carrier_odd_accumulate.h sets it to
// true for the odd macs. For those, the algorithms below replace each
// carrier operation whose result a format then rounds by its round-to-odd
// version (odd_sum, odd_diff, odd_fold, defined in that header and found by
// argument-dependent lookup). The choice is an `if constexpr` placed in front
// of the plain statement, which every other Mac compiles exactly as it
// would without it, rather than a function that every Mac calls. Such
// functions, even inlined, changed 162 device functions and 22 host objects
// of the existing kernels when they were tried: their arguments are
// evaluated in a different order from the expression's operands, and the
// operand order and the register allocation changed with it. A discarded
// branch compiles to nothing. So a change here must keep the plain
// statements as they are and test the trait, not wrap them.
template <class Mac>
struct carrier_rounds_to_odd
{
    static constexpr MPTORCH_CONSTANT bool value = false;
};

// The Mac that an accumulator drives, given the one its base Args built:
// that same Mac, except that carrier_odd_accumulate.h maps an odd split mac
// to one whose multiply rounds the product to odd, which is the product that
// TREE and KAHAN's term need. It is a type mapping, so that no statement
// below changes for it.
template <class Mac>
struct accumulate_mac
{
    using type = Mac;
};

// ---------------------------------------------------------------------------
// The state of one output element's sum. Each is a plain aggregate,
// initialized to zero, whose result() returns the field that holds the
// finished sum.

template <class T>
struct KahanState
{
    T sum = T(0);
    T comp = T(0);
    CUDA_HOST_DEVICE_INLINE T result() const { return sum; }
};

template <class T>
struct BlockState
{
    T blk = T(0);
    T tot = T(0);
    CUDA_HOST_DEVICE_INLINE T result() const { return tot; }
};

// lo[l] holds a finished subtree of 2^l products and hi[l] one of 2^(4+l)
// products. `slab` is the subtree of 16 products that accumulate() passes to
// end_slab().
template <class T>
struct TreeState
{
    T lo[4] = {T(0), T(0), T(0), T(0)};
    T hi[4] = {T(0), T(0), T(0), T(0)};
    T slab = T(0);
    T tot = T(0);
    CUDA_HOST_DEVICE_INLINE T result() const { return tot; }
};

// The CPU tile for these accumulators: the same interface as NaiveTile
// (gemm_policy.h), over a dense array of states instead of a dense array of
// sums. Host only.
#if !defined(__METAL_VERSION__)
template <class State>
struct StateTileView
{
    State *__restrict__ st;
    PhiloxEngine *__restrict__ rng;
};

template <class T, class State>
struct StateTile
{
    std::vector<State> states;
    std::vector<PhiloxEngine> streams;

    void resize(int64_t n)
    {
        states.resize(static_cast<size_t>(n));
        streams.resize(static_cast<size_t>(n));
    }

    StateTileView<State> view() { return {states.data(), streams.data()}; }

    // Resets the whole state of the first `n` elements to zero for a new
    // tile. (A TreeState's levels would not need it, since they are written
    // before they are read, but its total does, and one fill is simpler than
    // two rules.)
    void begin(int64_t n) { std::fill_n(states.data(), static_cast<size_t>(n), State{}); }

    void seed_rng(int64_t idx, uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
        PhiloxEngine &e = streams[static_cast<size_t>(idx)];
        e.reset_state(seed, subsequence);
        e.set_offset(offset);
    }

    T finalize(int64_t idx) const { return states[static_cast<size_t>(idx)].result(); }
};
#endif // !__METAL_VERSION__

// ---------------------------------------------------------------------------
// The accumulators. Like NaiveAccumulator, each has an object form, which is
// one GPU thread's whole state, and static functions over a CPU tile. Both
// forms call the same `step` and `slab` functions, which read the
// accumulator object as the configuration, so the two cannot disagree.

template <class Mac>
struct KahanAccumulator
{
    using mac_type = Mac;
    using value_t = typename Mac::value_t;
    using state_type = KahanState<value_t>;
#if !defined(__METAL_VERSION__)
    using tile_type = StateTile<value_t, state_type>;
#endif
    static constexpr MPTORCH_CONSTANT bool positional = true;
    static constexpr MPTORCH_CONSTANT int unroll = 4;

    Mac mac{};
    state_type st{};
    PhiloxEngine rng{};

    CUDA_HOST_DEVICE_INLINE static void step(const MPTORCH_THREAD KahanAccumulator &cfg,
                                             MPTORCH_THREAD state_type &st,
                                             MPTORCH_THREAD PhiloxEngine &rng, value_t a, value_t b,
                                             int /*pos*/)
    {
        const value_t y = mac_term(cfg.mac, a, b, st.comp, rng);
        if constexpr (carrier_rounds_to_odd<Mac>::value)
        {
            const value_t t = cfg.mac.add(odd_sum(cfg.mac, st.sum, y), rng);
            st.comp = cfg.mac.add(
                odd_diff(cfg.mac, cfg.mac.add(odd_diff(cfg.mac, t, st.sum), rng), y), rng);
            st.sum = t;
            return;
        }
        const value_t t = cfg.mac.add(st.sum + y, rng);
        st.comp = cfg.mac.add(cfg.mac.add(t - st.sum, rng) - y, rng);
        st.sum = t;
    }

    CUDA_HOST_DEVICE_INLINE static void slab(const MPTORCH_THREAD KahanAccumulator &,
                                             MPTORCH_THREAD state_type &,
                                             MPTORCH_THREAD PhiloxEngine &, int64_t, int64_t)
    {
    }

    CUDA_HOST_DEVICE_INLINE void seed_rng(uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
        rng.reset_state(seed, subsequence);
        rng.set_offset(offset);
    }
    CUDA_HOST_DEVICE_INLINE void accumulate(value_t a, value_t b, int pos) { step(*this, st, rng, a, b, pos); }
    CUDA_HOST_DEVICE_INLINE void end_slab(int64_t k_end, int64_t K) { slab(*this, st, rng, k_end, K); }
    CUDA_HOST_DEVICE_INLINE value_t finalize() const { return st.result(); }

#if !defined(__METAL_VERSION__)
    static inline void accumulate(const StateTileView<state_type> &t, int64_t idx,
                                  const KahanAccumulator &cfg, value_t a, value_t b, int pos)
    {
        step(cfg, t.st[idx], t.rng[idx], a, b, pos);
    }
    static inline void end_slab(const StateTileView<state_type> &t, int64_t idx,
                                const KahanAccumulator &cfg, int64_t k_end, int64_t K)
    {
        slab(cfg, t.st[idx], t.rng[idx], k_end, K);
    }
#endif
};

// `outer_quant` is a flag read at run time rather than a choice of Outer type
// (an IdentityAdder when it is off). Adding a block's sum to the total
// happens once per block, not once per step, so the flag costs little,
// whereas a second Outer type would double the number of BLOCK and TREE
// kernels.
template <class Mac, class Outer>
struct BlockAccumulator
{
    using mac_type = Mac;
    using value_t = typename Mac::value_t;
    using state_type = BlockState<value_t>;
#if !defined(__METAL_VERSION__)
    using tile_type = StateTile<value_t, state_type>;
#endif
    static constexpr MPTORCH_CONSTANT bool positional = true;
    static constexpr MPTORCH_CONSTANT int unroll = is_fused_mac<Mac>::value ? 16 : 4;

    Mac mac{};
    Outer outer{};
    bool outer_quant = false;
    // A block_size that divides 16 (other than 16 itself) ends its blocks
    // within the slab, where (pos + 1) & lo_mask is zero; any other ends
    // them at the end of a slab, where k_end % block is zero.
    bool in_slab = false;
    int lo_mask = 0;
    int64_t block = 16;
    state_type st{};
    PhiloxEngine rng{};

    BlockAccumulator() = default;
    CUDA_HOST_DEVICE_INLINE BlockAccumulator(const MPTORCH_THREAD Mac &mac,
                                             const MPTORCH_THREAD Outer &outer, bool outer_quant,
                                             int block_size)
        : mac(mac), outer(outer), outer_quant(outer_quant), in_slab(block_size < 16),
          lo_mask(block_size - 1), block(block_size)
    {
    }

    CUDA_HOST_DEVICE_INLINE static void fold(const MPTORCH_THREAD BlockAccumulator &cfg,
                                             MPTORCH_THREAD state_type &st,
                                             MPTORCH_THREAD PhiloxEngine &rng)
    {
        if constexpr (carrier_rounds_to_odd<Mac>::value)
        {
            const value_t x = odd_fold(cfg.mac, cfg.outer_quant, st.tot, st.blk);
            st.tot = cfg.outer_quant ? cfg.outer(x, rng) : x;
            st.blk = value_t(0);
            return;
        }
        const value_t x = st.tot + st.blk;
        st.tot = cfg.outer_quant ? cfg.outer(x, rng) : x;
        st.blk = value_t(0);
    }

    CUDA_HOST_DEVICE_INLINE static void step(const MPTORCH_THREAD BlockAccumulator &cfg,
                                             MPTORCH_THREAD state_type &st,
                                             MPTORCH_THREAD PhiloxEngine &rng, value_t a, value_t b,
                                             int pos)
    {
        st.blk = cfg.mac.step(a, b, st.blk, rng);
        if (cfg.in_slab && ((pos + 1) & cfg.lo_mask) == 0)
            fold(cfg, st, rng);
    }

    CUDA_HOST_DEVICE_INLINE static void slab(const MPTORCH_THREAD BlockAccumulator &cfg,
                                             MPTORCH_THREAD state_type &st,
                                             MPTORCH_THREAD PhiloxEngine &rng, int64_t k_end,
                                             int64_t K)
    {
        // When blocks end within the slab, every whole block has already been
        // added, and what is left at K is a partial one. Otherwise blocks are
        // added here or not at all.
        const bool due = cfg.in_slab ? (k_end == K && (K & cfg.lo_mask) != 0)
                                     : (k_end == K || k_end % cfg.block == 0);
        if (due)
            fold(cfg, st, rng);
    }

    CUDA_HOST_DEVICE_INLINE void seed_rng(uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
        rng.reset_state(seed, subsequence);
        rng.set_offset(offset);
    }
    CUDA_HOST_DEVICE_INLINE void accumulate(value_t a, value_t b, int pos) { step(*this, st, rng, a, b, pos); }
    CUDA_HOST_DEVICE_INLINE void end_slab(int64_t k_end, int64_t K) { slab(*this, st, rng, k_end, K); }
    CUDA_HOST_DEVICE_INLINE value_t finalize() const { return st.result(); }

#if !defined(__METAL_VERSION__)
    static inline void accumulate(const StateTileView<state_type> &t, int64_t idx,
                                  const BlockAccumulator &cfg, value_t a, value_t b, int pos)
    {
        step(cfg, t.st[idx], t.rng[idx], a, b, pos);
    }
    static inline void end_slab(const StateTileView<state_type> &t, int64_t idx,
                                const BlockAccumulator &cfg, int64_t k_end, int64_t K)
    {
        slab(cfg, t.st[idx], t.rng[idx], k_end, K);
    }
#endif
};

template <class Mac, class Outer>
struct TreeAccumulator
{
    using mac_type = Mac;
    using value_t = typename Mac::value_t;
    using state_type = TreeState<value_t>;
#if !defined(__METAL_VERSION__)
    using tile_type = StateTile<value_t, state_type>;
#endif
    static constexpr MPTORCH_CONSTANT bool positional = true;
    static constexpr MPTORCH_CONSTANT int unroll = 4;

    Mac mac{};
    Outer outer{};
    bool outer_quant = false;
    // block_size = 2^levels. The first min(levels, 4) levels are within a
    // slab of 16 steps (lo), and the others span slabs (hi).
    int lo_levels = 4;
    int hi_levels = 0;
    int lo_mask = 15;
    int hi_mask = 0;
    int block_mask = 15;
    state_type st{};
    PhiloxEngine rng{};

    TreeAccumulator() = default;
    CUDA_HOST_DEVICE_INLINE TreeAccumulator(const MPTORCH_THREAD Mac &mac,
                                            const MPTORCH_THREAD Outer &outer, bool outer_quant,
                                            int levels)
        : mac(mac), outer(outer), outer_quant(outer_quant), lo_levels(levels < 4 ? levels : 4),
          hi_levels(levels < 4 ? 0 : levels - 4), lo_mask((1 << (levels < 4 ? levels : 4)) - 1),
          hi_mask((1 << (levels < 4 ? 0 : levels - 4)) - 1), block_mask((1 << levels) - 1)
    {
    }

    CUDA_HOST_DEVICE_INLINE static void fold(const MPTORCH_THREAD TreeAccumulator &cfg,
                                             MPTORCH_THREAD state_type &st,
                                             MPTORCH_THREAD PhiloxEngine &rng, value_t root)
    {
        if constexpr (carrier_rounds_to_odd<Mac>::value)
        {
            const value_t x = odd_fold(cfg.mac, cfg.outer_quant, st.tot, root);
            st.tot = cfg.outer_quant ? cfg.outer(x, rng) : x;
            return;
        }
        const value_t x = st.tot + root;
        st.tot = cfg.outer_quant ? cfg.outer(x, rng) : x;
    }

    // One level of the binary counter. `i` is the subtree's index among its
    // siblings, and `l` the level that `slot` holds. Returns whether `v` has
    // yet to be stored at a level. A level at or above `levels` is not part
    // of the walk, so a `v` that has merged through all of them is still
    // unstored, and is the finished subtree.
    CUDA_HOST_DEVICE_INLINE static bool walk(const MPTORCH_THREAD TreeAccumulator &cfg,
                                             MPTORCH_THREAD value_t &slot, int l, int levels, int i,
                                             MPTORCH_THREAD value_t &v,
                                             MPTORCH_THREAD PhiloxEngine &rng)
    {
        if (l >= levels)
            return true;
        if ((i >> l) & 1)
        {
            if constexpr (carrier_rounds_to_odd<Mac>::value)
            {
                v = cfg.mac.add(odd_sum(cfg.mac, slot, v), rng);
                return true;
            }
            v = cfg.mac.add(slot + v, rng);
            return true;
        }
        slot = v;
        return false;
    }

    // Merges one level of a partial block, if it is occupied, into the value
    // being flushed.
    CUDA_HOST_DEVICE_INLINE static void drain(const MPTORCH_THREAD TreeAccumulator &cfg, value_t slot,
                                              bool occupied, MPTORCH_THREAD value_t &v,
                                              MPTORCH_THREAD bool &any,
                                              MPTORCH_THREAD PhiloxEngine &rng)
    {
        if (!occupied)
            return;
        if constexpr (carrier_rounds_to_odd<Mac>::value)
        {
            v = any ? cfg.mac.add(odd_sum(cfg.mac, slot, v), rng) : slot;
            any = true;
            return;
        }
        v = any ? cfg.mac.add(slot + v, rng) : slot;
        any = true;
    }

    CUDA_HOST_DEVICE_INLINE static void step(const MPTORCH_THREAD TreeAccumulator &cfg,
                                             MPTORCH_THREAD state_type &st,
                                             MPTORCH_THREAD PhiloxEngine &rng, value_t a, value_t b,
                                             int pos)
    {
        value_t v = cfg.mac.mul(a, b, rng);
        const int i = pos & cfg.lo_mask;
        const int n = cfg.lo_levels;
        bool open = walk(cfg, st.lo[0], 0, n, i, v, rng);
        open = open && walk(cfg, st.lo[1], 1, n, i, v, rng);
        open = open && walk(cfg, st.lo[2], 2, n, i, v, rng);
        open = open && walk(cfg, st.lo[3], 3, n, i, v, rng);
        if (open)
        {
            // Every level within the slab has merged: this is a whole block
            // of at most 16 products, or a subtree of 16 products of a
            // larger block.
            if (cfg.hi_levels == 0)
                fold(cfg, st, rng, v);
            else
                st.slab = v;
        }
    }

    CUDA_HOST_DEVICE_INLINE static void slab(const MPTORCH_THREAD TreeAccumulator &cfg,
                                             MPTORCH_THREAD state_type &st,
                                             MPTORCH_THREAD PhiloxEngine &rng, int64_t k_end,
                                             int64_t K)
    {
        if (cfg.hi_levels != 0 && (k_end & 15) == 0)
        {
            // A whole slab has just ended, and st.slab is its subtree: run
            // the same counter over slabs.
            value_t v = st.slab;
            const int i = static_cast<int>((k_end >> 4) - 1) & cfg.hi_mask;
            const int n = cfg.hi_levels;
            bool open = walk(cfg, st.hi[0], 0, n, i, v, rng);
            open = open && walk(cfg, st.hi[1], 1, n, i, v, rng);
            open = open && walk(cfg, st.hi[2], 2, n, i, v, rng);
            open = open && walk(cfg, st.hi[3], 3, n, i, v, rng);
            if (open)
                fold(cfg, st, rng, v);
        }
        if (k_end != K)
            return;
        // The last block, if K left it partial. Its size, in binary, says
        // which levels hold a subtree. They are merged lowest level first,
        // so each merge keeps the earlier products on the left.
        const int r = static_cast<int>(K & cfg.block_mask);
        if (r == 0)
            return;
        value_t v = value_t(0);
        bool any = false;
        drain(cfg, st.lo[0], (r & 1) != 0, v, any, rng);
        drain(cfg, st.lo[1], (r & 2) != 0, v, any, rng);
        drain(cfg, st.lo[2], (r & 4) != 0, v, any, rng);
        drain(cfg, st.lo[3], (r & 8) != 0, v, any, rng);
        drain(cfg, st.hi[0], (r & 16) != 0, v, any, rng);
        drain(cfg, st.hi[1], (r & 32) != 0, v, any, rng);
        drain(cfg, st.hi[2], (r & 64) != 0, v, any, rng);
        drain(cfg, st.hi[3], (r & 128) != 0, v, any, rng);
        fold(cfg, st, rng, v);
    }

    CUDA_HOST_DEVICE_INLINE void seed_rng(uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
        rng.reset_state(seed, subsequence);
        rng.set_offset(offset);
    }
    CUDA_HOST_DEVICE_INLINE void accumulate(value_t a, value_t b, int pos) { step(*this, st, rng, a, b, pos); }
    CUDA_HOST_DEVICE_INLINE void end_slab(int64_t k_end, int64_t K) { slab(*this, st, rng, k_end, K); }
    CUDA_HOST_DEVICE_INLINE value_t finalize() const { return st.result(); }

#if !defined(__METAL_VERSION__)
    static inline void accumulate(const StateTileView<state_type> &t, int64_t idx,
                                  const TreeAccumulator &cfg, value_t a, value_t b, int pos)
    {
        step(cfg, t.st[idx], t.rng[idx], a, b, pos);
    }
    static inline void end_slab(const StateTileView<state_type> &t, int64_t idx,
                                const TreeAccumulator &cfg, int64_t k_end, int64_t K)
    {
        slab(cfg, t.st[idx], t.rng[idx], k_end, K);
    }
#endif
};

// Whether a kernel drives this accumulator through the positional contract
// (accumulate with a position, then end_slab) rather than through
// NaiveAccumulator's.
#if !defined(__METAL_VERSION__)
template <class Accumulator>
inline constexpr bool is_positional_accumulator_v = requires { Accumulator::positional; };
#endif

namespace mptorch::gemm
{

  // For each single-format Args that an AccumulateArgs can wrap: the family
  // of the outer format (its Widths and Common types), and whether the mac
  // has a separate product, which TREE needs.
  template <class Base>
  struct AccumulateFamily;

  template <>
  struct AccumulateFamily<BinaryKSplitArgs>
  {
    using Widths = BinaryKWidths;
    using Common = BinaryKCommon;
    static constexpr MPTORCH_CONSTANT bool has_product = true;
  };

  template <>
  struct AccumulateFamily<BinaryKFusedArgs>
  {
    using Widths = BinaryKWidths;
    using Common = BinaryKCommon;
    static constexpr MPTORCH_CONSTANT bool has_product = false;
  };

  template <>
  struct AccumulateFamily<SuperfpSplitArgs>
  {
    using Widths = SuperfpWidths;
    using Common = SuperfpCommon;
    static constexpr MPTORCH_CONSTANT bool has_product = true;
  };

  template <>
  struct AccumulateFamily<SuperfpFusedArgs>
  {
    using Widths = SuperfpWidths;
    using Common = SuperfpCommon;
    static constexpr MPTORCH_CONSTANT bool has_product = false;
  };

  // The largest number of random draws one K-step of an algorithm may take
  // under RoundMode::SR, given what the Mac's own step draws (two for a
  // split mac, one for a fused one). The result sizes a reservation of
  // random numbers, so an upper bound is enough. KAHAN rounds y, t, t - s and
  // c besides the product: the mac's count plus three. BLOCK adds at most
  // one rounding of the total per step. TREE rounds one product per step,
  // fewer merges than products over a block, and at most one addition to
  // the total per block, and the flush of a partial block merges fewer
  // levels than the block has products.
  inline uint64_t accumulate_draws_per_k_step(AccumulateAlgorithm alg, uint64_t mac_draws)
  {
    switch (alg)
    {
    case AccumulateAlgorithm::KAHAN:
      return mac_draws + 3;
    case AccumulateAlgorithm::BLOCK:
      return mac_draws + 1;
    case AccumulateAlgorithm::TREE:
      return mac_draws + 2;
    default:
      return mac_draws;
    }
  }

  // The base-2 logarithm of TREE's block_size, a power of two in [2, 256]
  // that the host has already checked.
  CUDA_HOST_DEVICE_INLINE int tree_levels(int block_size)
  {
    int levels = 0;
    while ((1 << (levels + 1)) <= block_size)
      ++levels;
    return levels;
  }

  // The Args of a KAHAN, BLOCK or TREE call: a single-format op's Args plus
  // what its *_accumulated twin's schema says about the accumulation. A NAIVE
  // call never builds one; it runs on `Base` itself.
  //
  // with_accumulator<T, RM, ALG> reuses Base's factory to build the Mac: the
  // factory hands it a NaiveAccumulator, whose Mac is the one to drive, so
  // the choice between an Adder and an IdentityAdder is written in one
  // place.
  template <class Base>
  struct AccumulateArgs
  {
    using Widths = typename AccumulateFamily<Base>::Widths;
    using Common = typename AccumulateFamily<Base>::Common;
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    static constexpr MPTORCH_CONSTANT bool has_product = AccumulateFamily<Base>::has_product;

    Base base{};
    AccumulateAlgorithm alg = AccumulateAlgorithm::KAHAN;
    int block_size = 0;
    bool outer_quant = false;
    Widths outer{};
    Common outer_c{};

    // The largest number of random draws one K-step may take under
    // RoundMode::SR. It depends on the algorithm, so unlike Base's it is not
    // a constant of the type.
    uint64_t draws_per_k_step() const
    {
      return accumulate_draws_per_k_step(alg, Base::draws_per_k_step);
    }

    template <class T, RoundMode RM, AccumulateAlgorithm ALG, class F>
    struct Builder
    {
      const MPTORCH_THREAD AccumulateArgs &args;
      MPTORCH_THREAD F &f;

      template <class Naive>
      void operator()(const MPTORCH_THREAD Naive &naive) const
      {
        using Mac = typename accumulate_mac<typename Naive::mac_type>::type;
        if constexpr (ALG == AccumulateAlgorithm::KAHAN)
        {
          f(KahanAccumulator<Mac>{naive.mac});
        }
        else
        {
          // An outer format that is off arrives as zero widths, from which
          // no cast's constants should be derived. The adder is then never
          // called, so a value-initialized one is enough.
          using Outer = decltype(make_add<T, RM>(args.outer, args.outer_c));
          const Outer outer = args.outer_quant ? make_add<T, RM>(args.outer, args.outer_c) : Outer{};
          if constexpr (ALG == AccumulateAlgorithm::BLOCK)
            f(BlockAccumulator<Mac, Outer>{naive.mac, outer, args.outer_quant, args.block_size});
          else
            f(TreeAccumulator<Mac, Outer>{naive.mac, outer, args.outer_quant,
                                          tree_levels(args.block_size)});
        }
      }
    };

    template <class T, RoundMode RM, AccumulateAlgorithm ALG, class F>
    void with_accumulator(MPTORCH_THREAD F &&f) const
    {
      base.template with_accumulator<T, RM>(Builder<T, RM, ALG, F>{*this, f});
    }
  };

} // namespace mptorch::gemm
