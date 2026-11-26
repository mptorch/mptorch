#pragma once

// The counted GEMMs: the single-format GEMM ops under every accumulate
// algorithm, NAIVE included, counting their casts' numeric events
// (event_counts.h) by role: the products a split mac rounds (MUL), the sums
// (ACC: a split mac's sum, a fused mac's one rounding, KAHAN's term, sum and
// compensation, TREE's merges), and BLOCK's and TREE's outer format (OUTER).
//
// Counted<Policy, ROLE> wraps a multiplier or an adder: it calls the policy,
// then classifies the cast's input and result. The algorithms round only
// through their Mac's policies and the outer one (gemm_policy.h's SplitMac
// and FusedMac, gemm_accumulate.h), so a Mac built from Counted policies
// counts every cast of every algorithm, and no algorithm changes for it. The
// wrapper only reads the cast, and draws nothing, so a counted GEMM's result
// is its twin's word for word, under SR too. An IdentityAdder is never
// wrapped: it rounds to no format.
//
// Where the counts go:
//   - On a GPU, each thread's accumulator is its own copy of the prototype,
//     and the counts are mutable fields of the wrappers in it. After the
//     thread's store, each warp sums its threads' counts and adds them to the
//     call's with one 64-bit atomic per count (counting_thread_end).
//   - On the CPU, the tasks share the prototype through a const reference, so
//     a wrapper adds to its role's row of a thread_local tally, and the kernel
//     adds a task's tally to the call's counts when the task ends
//     (counting_task_end).
//   - On an Apple GPU, as on CUDA, each thread's counts are in its own
//     accumulator; mps/gemm.metal, compiled under MPT_COUNTING, sums them
//     over the threadgroup (counting_tallies gathers them) and adds the sums
//     to the call's counts. Metal has no derived classes, so there Counting
//     holds the accumulator as a member instead of deriving from it.
// All are integer sums, so the counts depend on neither the order nor the
// number of threads.
//
// NAIVE is counted through the positional contract
// (PositionalNaiveAccumulator), which steps k < K only. The CUDA kernel's
// NAIVE path also runs the zero steps that pad the last slab of 16; they
// round an exact sum to itself, so the results agree either way, but each
// would count a cast the CPU kernel never makes.
//
// The kernels recognize a counted accumulator by `requires {
// Accumulator::counting; }`, spelled in place so that they include nothing
// for it, and find counting_task_end and counting_thread_end by
// argument-dependent lookup. The units that instantiate a counted kernel
// include this header; no other unit sees it.

#include "event_counts.h"
#include "gemm_accumulate.h"

#if !defined(__METAL_VERSION__)
#include <cstdint>
#include <type_traits>
#endif

namespace mptorch::gemm
{
#if !defined(__CUDA_ARCH__) && !defined(__METAL_VERSION__)
  // The CPU's per-task counts: a counted cast on a CPU thread adds to its
  // role's row here, and counting_task_end moves the rows into the call's
  // counts. 64 bits, since one task can make more than 2^32 casts.
  inline thread_local uint64_t counting_task_counts[events::ROLES][events::EVENTS] = {};
#endif

  // Counts one cast of a role: into the thread's own tally on a GPU, into
  // the task's row of counting_task_counts on the CPU.
  CUDA_HOST_DEVICE_INLINE void count_cast(MPTORCH_THREAD events::EventTally &tally, int role,
                                          uint32_t bits)
  {
#if defined(__CUDA_ARCH__) || defined(__METAL_VERSION__)
    (void)role;
    tally.add(bits);
#else
    (void)tally;
    uint64_t *row = counting_task_counts[role];
    for (int e = 0; e < events::EVENTS; ++e)
      row[e] += (bits >> e) & 1u;
#endif
  }

  template <class Policy, int ROLE>
  struct Counted
  {
    using value_t = typename Policy::value_t;

    Policy policy{};
    events::EventThresholdsT<value_t> th{};
    // A GPU thread's counts. The CPU keeps its counts in
    // counting_task_counts instead, and leaves these at zero.
    mutable events::EventTally tally{};

    CUDA_HOST_DEVICE_INLINE void count(value_t x, value_t q) const
    {
      count_cast(tally, ROLE, events::event_bits<value_t>(x, q, th));
    }

    // A multiplier's cast: its input is the carrier's product, which the
    // policy computes as this does, one rounded multiplication.
    CUDA_HOST_DEVICE_INLINE value_t operator()(value_t a, value_t b, MPTORCH_THREAD PhiloxEngine &rng) const
    {
      const value_t q = policy(a, b, rng);
      count(a * b, q);
      return q;
    }

    // An adder's cast.
    CUDA_HOST_DEVICE_INLINE value_t operator()(value_t x, MPTORCH_THREAD PhiloxEngine &rng) const
    {
      const value_t q = policy(x, rng);
      count(x, q);
      return q;
    }
  };

  // A role's thresholds, from the format's own fields (event_counts.h).
  template <class T>
  CUDA_HOST_DEVICE_INLINE events::EventThresholdsT<T> thresholds_of(const MPTORCH_THREAD BinaryKWidths &w,
                                                                    const MPTORCH_THREAD BinaryKCommon &c)
  {
    return events::binaryK_thresholds<T>(w.man_bits, w.exp_bits, w.bias, c.is_signed, c.sat, c.sub);
  }

  template <class T>
  CUDA_HOST_DEVICE_INLINE events::EventThresholdsT<T> thresholds_of(const MPTORCH_THREAD SuperfpWidths &w,
                                                                    const MPTORCH_THREAD SuperfpCommon &c)
  {
    return events::superfp_thresholds<T>(w.man_bits, w.exp_bits, w.normal_binades, w.bias, c.sat);
  }

  // A base Args' Mac, its policies wrapped: a split mac's product and sum
  // (base.mul, base.acc), a fused mac's one rounding (base.fma).
  template <class T, class M, class A, class Base>
  CUDA_HOST_DEVICE_INLINE auto counted_mac(const MPTORCH_THREAD SplitMac<M, A> &m, const MPTORCH_THREAD Base &b)
  {
    using CM = Counted<M, events::MUL>;
    const CM mul{m.mul, thresholds_of<T>(b.mul, b.mul_c)};
    if constexpr (std::is_same_v<A, IdentityAdder<T>>)
      return SplitMac<CM, A>{mul, m.add};
    else
      return SplitMac<CM, Counted<A, events::ACC>>{mul, {m.add, thresholds_of<T>(b.acc, b.acc_c)}};
  }

  template <class T, class A, class Base>
  CUDA_HOST_DEVICE_INLINE auto counted_mac(const MPTORCH_THREAD FusedMac<A> &m, const MPTORCH_THREAD Base &b)
  {
    if constexpr (std::is_same_v<A, IdentityAdder<T>>)
      return m;
    else
      return FusedMac<Counted<A, events::ACC>>{{m.add, thresholds_of<T>(b.fma, b.fma_c)}};
  }

  // NAIVE through the positional contract (gemm_accumulate.h): the sum of
  // NaiveAccumulator, step by step, k < K only (see the top of the file).
  template <class T>
  struct NaiveState
  {
    T sum = T(0);
    CUDA_HOST_DEVICE_INLINE T result() const { return sum; }
  };

  template <class Mac>
  struct PositionalNaiveAccumulator
  {
    using mac_type = Mac;
    using value_t = typename Mac::value_t;
    using state_type = NaiveState<value_t>;
#if !defined(__METAL_VERSION__)
    using tile_type = StateTile<value_t, state_type>;
#endif
    static constexpr MPTORCH_CONSTANT bool positional = true;
    static constexpr MPTORCH_CONSTANT int unroll = 4;

    Mac mac{};
    state_type st{};
    PhiloxEngine rng{};

    CUDA_HOST_DEVICE_INLINE static void step(const MPTORCH_THREAD PositionalNaiveAccumulator &cfg,
                                             MPTORCH_THREAD state_type &st,
                                             MPTORCH_THREAD PhiloxEngine &rng, value_t a, value_t b,
                                             int /*pos*/)
    {
      st.sum = cfg.mac.step(a, b, st.sum, rng);
    }

    CUDA_HOST_DEVICE_INLINE static void slab(const MPTORCH_THREAD PositionalNaiveAccumulator &,
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
                                  const PositionalNaiveAccumulator &cfg, value_t a, value_t b, int pos)
    {
      step(cfg, t.st[idx], t.rng[idx], a, b, pos);
    }
    static inline void end_slab(const StateTileView<state_type> &t, int64_t idx,
                                const PositionalNaiveAccumulator &cfg, int64_t k_end, int64_t K)
    {
      slab(cfg, t.st[idx], t.rng[idx], k_end, K);
    }
#endif
  };

#if defined(__METAL_VERSION__)
  // A counted accumulator on Metal: the accumulator as a member, and the
  // calls gemm.metal makes passed on to it. The kernel binds the call's
  // counts itself (MPT_COUNTING), so `counts` holds nothing: it is there for
  // CountingArgs' initializers, which are the host's. Every counted
  // accumulator takes the positional contract.
  template <class Acc>
  struct Counting
  {
    using value_t = typename Acc::value_t;
    static constexpr MPTORCH_CONSTANT bool counting = true;
    static constexpr MPTORCH_CONSTANT bool positional = true;

    Acc acc;
    uint32_t counts;

    void seed_rng(uint64_t seed, uint64_t subsequence, uint64_t offset = 0)
    {
      acc.seed_rng(seed, subsequence, offset);
    }
    void accumulate(value_t a, value_t b, int pos) { acc.accumulate(a, b, pos); }
    void end_slab(int64_t k_end, int64_t K) { acc.end_slab(k_end, K); }
    value_t finalize() const { return acc.finalize(); }
  };
#else
  // A counted accumulator: the accumulator, and where the call's counts go,
  // an int64 [ROLES, EVENTS] tensor's data. A Gathered (the conv ops) wraps
  // it in turn.
  template <class Acc>
  struct Counting : Acc
  {
    static constexpr bool counting = true;
    unsigned long long *counts = nullptr;
  };
#endif

  // The Args of a counted op: its single-format twin's Args (`base`), the
  // algorithm, and the *_accumulated schema's tail. The algorithm is a
  // template parameter, so each has an instantiation (and a unit) of its
  // own, and launch_as builds it as it builds any single-format Args. It is
  // not named `alg`: the conv launch takes an Args with an `alg` for an
  // AccumulateArgs.
  template <class Base, AccumulateAlgorithm ALG>
  struct CountingArgs
  {
    using Widths = typename AccumulateFamily<Base>::Widths;
    using Common = typename AccumulateFamily<Base>::Common;
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    static constexpr MPTORCH_CONSTANT bool has_product = AccumulateFamily<Base>::has_product;
    static constexpr MPTORCH_CONSTANT AccumulateAlgorithm counted_alg = ALG;

    Base base{};
    int block_size = 0;
    bool outer_quant = false;
    Widths outer{};
    Common outer_c{};
#if defined(__METAL_VERSION__)
    uint32_t counts = 0; // unused: see Counting
#else
    unsigned long long *counts = nullptr;
#endif

    template <class T, RoundMode RM, class F>
    struct Builder
    {
      const MPTORCH_THREAD CountingArgs &args;
      MPTORCH_THREAD F &f;

      template <class Naive>
      void operator()(const MPTORCH_THREAD Naive &naive) const
      {
        const auto mac = counted_mac<T>(naive.mac, args.base);
        using Mac = std::remove_const_t<decltype(mac)>;
        if constexpr (ALG == AccumulateAlgorithm::NAIVE)
        {
          f(Counting<PositionalNaiveAccumulator<Mac>>{{mac}, args.counts});
        }
        else if constexpr (ALG == AccumulateAlgorithm::KAHAN)
        {
          f(Counting<KahanAccumulator<Mac>>{{mac}, args.counts});
        }
        else
        {
          // As AccumulateArgs builds it: an outer format that is off is never
          // called, and a value-initialized one is enough.
          using Plain = decltype(make_add<T, RM>(args.outer, args.outer_c));
          using Outer = Counted<Plain, events::OUTER>;
          const Outer outer = args.outer_quant
                                  ? Outer{make_add<T, RM>(args.outer, args.outer_c),
                                          thresholds_of<T>(args.outer, args.outer_c)}
                                  : Outer{};
          if constexpr (ALG == AccumulateAlgorithm::BLOCK)
            f(Counting<BlockAccumulator<Mac, Outer>>{
                BlockAccumulator<Mac, Outer>{mac, outer, args.outer_quant, args.block_size}, args.counts});
          else if constexpr (has_product) // TREE: the driver has refused a fused mac
            f(Counting<TreeAccumulator<Mac, Outer>>{
                TreeAccumulator<Mac, Outer>{mac, outer, args.outer_quant, tree_levels(args.block_size)},
                args.counts});
        }
      }
    };

    template <class T, RoundMode RM, class F>
    void with_accumulator(MPTORCH_THREAD F &&f) const
    {
      base.template with_accumulator<T, RM>(Builder<T, RM, F>{*this, f});
    }
  };

#if !defined(__METAL_VERSION__)
#if !defined(__CUDA_ARCH__)
  // The end of a CPU task: its counts into the call's, and the tally back
  // to zero for the thread's next task. The add is the compilers' builtin
  // rather than std::atomic_ref, which older libc++ releases lack.
  template <class Acc>
  inline void counting_task_end(const Counting<Acc> &acc)
  {
    for (int r = 0; r < events::ROLES; ++r)
      for (int e = 0; e < events::EVENTS; ++e)
      {
        uint64_t &v = counting_task_counts[r][e];
        if (v != 0)
          __atomic_fetch_add(acc.counts + r * events::EVENTS + e, static_cast<unsigned long long>(v),
                             __ATOMIC_RELAXED);
        v = 0;
      }
  }
#endif

#if defined(__CUDACC__)
  // The end of a GPU thread, after its store: each warp sums its threads'
  // counts (a thread outside the result contributes none) and lane 0 adds
  // the sums to the call's. Every thread of the block reaches it, as the
  // full-mask shuffles need.
  __device__ inline void counting_flush_tally(const events::EventTally &t, int role, bool in_range,
                                              unsigned long long *counts)
  {
    const unsigned lane = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31u;
    for (int e = 0; e < events::EVENTS; ++e)
    {
      unsigned long long v = in_range ? t.n[e] : 0ull;
      for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
      if (lane == 0 && v != 0)
        atomicAdd(counts + role * events::EVENTS + e, v);
    }
  }

  template <class P, int ROLE>
  __device__ inline void counting_flush(const Counted<P, ROLE> &c, bool in_range,
                                        unsigned long long *counts)
  {
    counting_flush_tally(c.tally, ROLE, in_range, counts);
  }

  template <class P>
  __device__ inline void counting_flush(const P &, bool, unsigned long long *)
  {
  }

  template <class M, class A>
  __device__ inline void counting_flush_mac(const SplitMac<M, A> &m, bool in_range,
                                            unsigned long long *counts)
  {
    counting_flush(m.mul, in_range, counts);
    counting_flush(m.add, in_range, counts);
  }

  template <class A>
  __device__ inline void counting_flush_mac(const FusedMac<A> &m, bool in_range,
                                            unsigned long long *counts)
  {
    counting_flush(m.add, in_range, counts);
  }

  template <class Acc>
  __device__ inline void counting_thread_end(const Counting<Acc> &acc, bool in_range)
  {
    counting_flush_mac(acc.mac, in_range, acc.counts);
    if constexpr (requires { acc.outer; })
      counting_flush(acc.outer, in_range, acc.counts);
  }
#endif
#endif // !__METAL_VERSION__

#if defined(__METAL_VERSION__)
  // A GPU thread's counts on Metal, what counting_thread_end adds on CUDA:
  // each wrapper's tally added into `t`, which has a row per role, for
  // gemm.metal to sum over the threadgroup. The overloads are chosen as
  // counting_flush's are, and the outer format's by the accumulator's type,
  // since Metal has no requires-expression.
  template <class P, int ROLE>
  inline void counting_gather(const MPTORCH_THREAD Counted<P, ROLE> &c,
                              MPTORCH_THREAD events::EventTally *t)
  {
    for (int e = 0; e < events::EVENTS; ++e)
      t[ROLE].n[e] += c.tally.n[e];
  }

  template <class P>
  inline void counting_gather(const MPTORCH_THREAD P &, MPTORCH_THREAD events::EventTally *)
  {
  }

  template <class M, class A>
  inline void counting_gather_mac(const MPTORCH_THREAD SplitMac<M, A> &m,
                                  MPTORCH_THREAD events::EventTally *t)
  {
    counting_gather(m.mul, t);
    counting_gather(m.add, t);
  }

  template <class A>
  inline void counting_gather_mac(const MPTORCH_THREAD FusedMac<A> &m,
                                  MPTORCH_THREAD events::EventTally *t)
  {
    counting_gather(m.add, t);
  }

  template <class Acc>
  inline void counting_gather_outer(const MPTORCH_THREAD Acc &, MPTORCH_THREAD events::EventTally *)
  {
  }

  template <class Mac, class Outer>
  inline void counting_gather_outer(const MPTORCH_THREAD BlockAccumulator<Mac, Outer> &acc,
                                    MPTORCH_THREAD events::EventTally *t)
  {
    counting_gather(acc.outer, t);
  }

  template <class Mac, class Outer>
  inline void counting_gather_outer(const MPTORCH_THREAD TreeAccumulator<Mac, Outer> &acc,
                                    MPTORCH_THREAD events::EventTally *t)
  {
    counting_gather(acc.outer, t);
  }

  template <class Acc>
  inline void counting_tallies(const MPTORCH_THREAD Counting<Acc> &c,
                               MPTORCH_THREAD events::EventTally *t)
  {
    counting_gather_mac(c.acc.mac, t);
    counting_gather_outer(c.acc, t);
  }
#endif
} // namespace mptorch::gemm
