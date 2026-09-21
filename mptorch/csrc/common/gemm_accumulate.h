#pragma once

// The three accumulate algorithms past NAIVE (modes.h's AccumulateAlgorithm):
// KAHAN, BLOCK and TREE, as Accumulator policies over the same Mac policies
// NaiveAccumulator drives (gemm_policy.h), and the Args wrapper that builds
// them from the schema of an *_accumulated op, the twin each single-format op
// has for them.
//
// A header of its own, which gemm_policy.h and gemm_args.h do not include, on
// purpose: those two are what the NAIVE kernels are compiled from, on three
// compilers, and nothing here may move them. The CUDA and CPU kernel headers
// include this one and recognize its accumulators by `positional` below; a
// NaiveAccumulator, which has no such member, takes exactly the code path it
// took before this file existed (dev/gemm_roadmap.md, R-2, has the binary
// comparison). The Metal prelude does not include it, and the MPS backend
// binds the *_accumulated ops to functions that raise until it does
// (dev/continuation_plan.md, phase H). The code is nonetheless written the
// way the other common/ headers are, with MPTORCH_THREAD on references and
// function objects rather than lambdas, so that phase H ports it rather than
// rewrites it.
//
// Semantics. `mul` and `add` are the Mac's two roundings (`add` is the
// identity when the accumulate format is off), `outer` a third (the identity
// when no outer format is given), and k runs over the K steps of one output
// element's dot product, in order. Under RoundMode::SR every rounding that
// is not an identity draws once from the element's stream, in the order the
// expressions below are written.
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
//   TREE    (split macs only) p_i = mul(a*b) is product i of its block of
//           block_size = 2^L. It is merged upward through the levels the
//           trailing ones of i select, v = add(level[l] + v), and parked at
//           the first level whose bit is clear: a binary counter, which sums
//           the block pairwise, (p0+p1)+(p2+p3), in the order the products
//           arrive. Product 2^L - 1 merges through all L levels and leaves
//           the block's root: tot = outer(tot + root). A last partial block
//           is flushed at K: the occupied levels are merged lowest first,
//           v = add(level[l] + v), and the result folded the same way.
//                                                     result: tot
//
// Nothing is padded: step k exists only for k < K, on every backend, so a
// result does not depend on the 16-deep slabs the CUDA kernel stages its
// operands in. (Padding the tail with zero products, which is free for NAIVE,
// would not be for KAHAN: a zero product still applies the pending
// compensation.) No result is -0.0: every sum above starts from +0.0, a cast
// never returns -0.0, and x + y is -0.0 only when both are.
//
// The kernel contract is NaiveAccumulator's plus a position:
//
//   accumulate(a, b, pos)   pos = k & 15. BLOCK folds a block that divides
//                           16 here, TREE walks its first four levels.
//   end_slab(k_end, K)      after step k_end - 1, when k_end % 16 == 0 or
//                           k_end == K: what happens at most once per 16
//                           steps. BLOCK's fold for a block_size that is a
//                           multiple of 16, TREE's levels 4 and up, and both
//                           algorithms' handling of the last partial block.
//
// which is why BLOCK's block_size must divide 16 or be a multiple of it: the
// fold test is then a mask per step or a modulo per slab, never a division
// per step. Every array below is indexed by constants only (the level walks
// are written out), so a GPU thread keeps the whole state in registers; a
// dynamically indexed array would be placed in local memory (philox.h).
//
// State per output element: KAHAN two carrier values, BLOCK two, TREE ten.

#include "gemm_args.h"

// ---------------------------------------------------------------------------
// The pieces of a Mac the algorithms need separately. Free functions rather
// than members of SplitMac/FusedMac so that gemm_policy.h stays as it was.

// KAHAN's term: the product less the running compensation, rounded to the
// accumulate format. A fused mac forms it with one FMA, as its step does.
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

// Whether a Mac is a fused multiply-add, whose step is one rounding where a
// split mac's is two. BLOCK's unroll factor depends on it.
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

// ---------------------------------------------------------------------------
// Per-output-element state. Plain aggregates, zero-initialized, each naming
// the field a finished reduction is read from.

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

// lo[l] and hi[l] hold a finished subtree of 2^l and 2^(4+l) products; `slab`
// is the 16-product subtree accumulate() hands to end_slab().
template <class T>
struct TreeState
{
    T lo[4] = {T(0), T(0), T(0), T(0)};
    T hi[4] = {T(0), T(0), T(0), T(0)};
    T slab = T(0);
    T tot = T(0);
    CUDA_HOST_DEVICE_INLINE T result() const { return tot; }
};

// The CPU tile for these accumulators: NaiveTile's contract (gemm_policy.h)
// over a dense array of State instead of a dense array of sums. Host-only.
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

    // A new tile: the first `n` elements' state back to zero, every field of
    // it. (A TreeState's levels would not need it, being written before they
    // are read, but its total does, and one fill is simpler than two rules.)
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
// The accumulators. Each is, like NaiveAccumulator, one GPU thread's whole
// state in its object form and a set of static functions over a tile on the
// CPU. Both forms call the same `step` / `slab`, with the accumulator itself
// as the read-only configuration, so the two cannot disagree.

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

// `outer_quant` is a runtime flag rather than a second Outer type: the fold
// runs once per block, not once per step, and an IdentityAdder instantiation
// of every BLOCK and TREE kernel would double their number for it.
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
    // A block that divides 16 (and is not 16) folds inside the slab, on
    // (pos + 1) & lo_mask; any other folds at a slab's end, on k_end % block.
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
        // In-slab blocks have folded every whole block already; what is left
        // at K is a partial one. Slab-end blocks fold here or not at all.
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
    // block_size = 2^levels. The first min(levels, 4) levels live inside a
    // 16-step slab (lo), the rest across slabs (hi).
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
        const value_t x = st.tot + root;
        st.tot = cfg.outer_quant ? cfg.outer(x, rng) : x;
    }

    // One level of the binary counter: `i` is the subtree's index among its
    // siblings and `l` the level `slot` holds. Returns whether `v` is still
    // looking for its level. A level at or past `levels` is not part of the
    // walk, so a `v` that has merged through all of them stays open and is
    // the finished subtree.
    CUDA_HOST_DEVICE_INLINE static bool walk(const MPTORCH_THREAD TreeAccumulator &cfg,
                                             MPTORCH_THREAD value_t &slot, int l, int levels, int i,
                                             MPTORCH_THREAD value_t &v,
                                             MPTORCH_THREAD PhiloxEngine &rng)
    {
        if (l >= levels)
            return true;
        if ((i >> l) & 1)
        {
            v = cfg.mac.add(slot + v, rng);
            return true;
        }
        slot = v;
        return false;
    }

    // One occupied level of a partial block, merged into the flush.
    CUDA_HOST_DEVICE_INLINE static void drain(const MPTORCH_THREAD TreeAccumulator &cfg, value_t slot,
                                              bool occupied, MPTORCH_THREAD value_t &v,
                                              MPTORCH_THREAD bool &any,
                                              MPTORCH_THREAD PhiloxEngine &rng)
    {
        if (!occupied)
            return;
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
            // Every in-slab level merged: a whole block of at most 16, or
            // one 16-product subtree of a larger block.
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
            // A whole slab just ended, and st.slab is its subtree: the same
            // counter, over slabs.
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
        // The last block, if K left it partial: its size in binary says which
        // levels hold a subtree. Lowest first, so each merge keeps the
        // earlier products on the left.
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

// Whether a kernel drives this Accumulator through the positional contract
// (accumulate with a position, end_slab) rather than NaiveAccumulator's.
#if !defined(__METAL_VERSION__)
template <class Accumulator>
inline constexpr bool is_positional_accumulator_v = requires { Accumulator::positional; };
#endif

namespace mptorch::gemm
{

  // The family an accumulated op's outer format belongs to, and whether its
  // mac has a product term to build a tree of, by the single-format Args it
  // wraps.
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

  // The most random values one K-step of an algorithm may draw under
  // RoundMode::SR, given what the Mac's own step draws (2 split, 1 fused). An
  // upper bound is all the reservation it sizes needs. KAHAN rounds y, t,
  // t - s and c on top of the multiply: the mac's count plus three. BLOCK
  // adds at most one fold per step. TREE draws one multiply per product,
  // fewer than one merge per product over a block and at most one fold, and
  // its flush of a partial block merges fewer levels than the block has
  // products.
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

  // log2 of a power of two in [2, 256], TREE's block_size; the host has
  // validated it.
  CUDA_HOST_DEVICE_INLINE int tree_levels(int block_size)
  {
    int levels = 0;
    while ((1 << (levels + 1)) <= block_size)
      ++levels;
    return levels;
  }

  // A single-format op's Args plus what its *_accumulated twin's schema says
  // about the accumulation: the Args of a KAHAN, BLOCK or TREE call. A NAIVE
  // call never builds one; it runs on `Base` itself, as it always has.
  //
  // with_accumulator<T, RM, ALG> reuses Base's factory for the Mac, which
  // hands it a NaiveAccumulator whose Mac is the one to drive, so the choice
  // between an Adder and an IdentityAdder stays written once.
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

    // The most random values one K-step may draw under RoundMode::SR, which
    // depends on the algorithm and so is not the constant it is on Base.
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
        using Mac = typename Naive::mac_type;
        if constexpr (ALG == AccumulateAlgorithm::KAHAN)
        {
          f(KahanAccumulator<Mac>{naive.mac});
        }
        else
        {
          // An outer format that is off arrives as zero widths, which no
          // cast's constants should be derived from; the adder is then
          // never called and a value-initialized one will do.
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
