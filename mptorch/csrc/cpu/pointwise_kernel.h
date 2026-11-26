#pragma once

// The CPU kernels of the fused pointwise and table ops (common/pointwise.h
// states them): the walk over the broadcast geometry, the deterministic and
// the stochastic loops over a step (pointwise::ArithStep or PwlStep), and the
// bodies every family's entry points share. The family files
// (cpu/pointwise_binaryK.cpp, pointwise_superfp.cpp, pointwise_fixedpoint.cpp)
// supply the casts. They are files of their own, so that the elementwise
// quantizers' objects are what they were.
//
// A chunk runs in blocks: the steps into a buffer (the operands read through
// the geometry), then the casts over it in the loop of cpu/utils.h's
// quant_kernel or quant_kernel_sr: the sign a template parameter, the cast's
// constants built once per call, and under SR element j's random word word
// j & 3 of Philox block j >> 2 (in binary64 the 64-bit word 2 * (j & 1) of
// block j >> 1), generated again at the start of every chunk. One loop doing
// both, with the op chosen per element, measured 1.23x the two passes' time
// on one thread for a binaryK RU add (7.83 ms against 6.38, 2048 x 2048).

#include "../common/dispatch.h"
#include "../common/modes.h"
#include "../common/pointwise_host.h"
#include "../common/stochastic_host.h"
#include "utils.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace mptorch_cpu::pointwise
{
  using mptorch::pointwise::PointwiseGeom;

  // One call's operands and its step. `y` is the number operand in the
  // carrier, and `b` null when there is none.
  template <typename scalar_t, class Step>
  struct Call
  {
    using scalar_type = scalar_t;
    using step_type = Step;
    const scalar_t *a;
    const scalar_t *b;
    scalar_t *o;
    PointwiseGeom g;
    carrier_t<scalar_t> y;
    Step step;
  };

  // body(j, oa, ob) for every element j in [begin, end), in order, with its
  // operands' offsets: j itself in a flat geometry; otherwise the offsets of
  // element `begin` from its multi-index, then runs along the innermost
  // dimension, carried into the outer ones at each run's end.
  template <class Body>
  inline void walk(const PointwiseGeom &g, int64_t begin, int64_t end, Body &&body)
  {
    if (g.flat)
    {
      for (int64_t j = begin; j < end; ++j)
        body(j, j, j);
      return;
    }
    int64_t idx[mptorch::pointwise::POINTWISE_MAX_DIMS];
    int64_t oa = 0, ob = 0, rest = begin;
    for (int d = g.ndim - 1; d >= 0; --d)
    {
      idx[d] = rest % g.size[d];
      rest /= g.size[d];
      oa += idx[d] * g.sa[d];
      ob += idx[d] * g.sb[d];
    }
    const int last = g.ndim - 1;
    const int64_t sa = g.sa[last], sb = g.sb[last];
    for (int64_t j = begin; j < end;)
    {
      const int64_t run = std::min(end - j, g.size[last] - idx[last]);
      for (int64_t k = 0; k < run; ++k)
        body(j + k, oa + k * sa, ob + k * sb);
      j += run;
      oa += run * sa;
      ob += run * sb;
      idx[last] += run;
      for (int d = last; d > 0 && idx[d] == g.size[d]; --d)
      {
        oa += g.sa[d - 1] - g.size[d] * g.sa[d];
        ob += g.sb[d - 1] - g.size[d] * g.sb[d];
        idx[d] = 0;
        ++idx[d - 1];
      }
    }
  }

  // The steps of elements [j0, j0 + n) into t. In a flat geometry the
  // arithmetic step runs one loop per op, the op chosen outside it;
  // otherwise each element's step is taken through the walk. A function of
  // the call alone, so it is instantiated per dtype and step, not per cast.
  template <int OP, bool HAS_B, typename scalar_t, typename T>
  inline void flat_steps(const scalar_t *a, const scalar_t *b, T y, T *t, int64_t n)
  {
    namespace rn = mptorch::pointwise::rn;
    for (int64_t k = 0; k < n; ++k)
    {
      const T x = static_cast<T>(a[k]);
      const T w = HAS_B ? static_cast<T>(b[k]) : y;
      if constexpr (OP == mptorch::pointwise::ADD)
        t[k] = rn::add(x, w);
      else if constexpr (OP == mptorch::pointwise::SUB)
        t[k] = rn::sub(x, w);
      else if constexpr (OP == mptorch::pointwise::MUL)
        t[k] = rn::mul(x, w);
      else if constexpr (OP == mptorch::pointwise::DIV)
        t[k] = rn::div(x, w);
      else if constexpr (OP == mptorch::pointwise::SQRT)
        t[k] = rn::sqrt(x);
      else if constexpr (OP == mptorch::pointwise::RSUB)
        t[k] = rn::sub(w, x);
      else
        t[k] = rn::div(w, x);
    }
  }

  template <int OP, typename scalar_t, typename T>
  inline void flat_steps(const scalar_t *a, const scalar_t *b, T y, T *t, int64_t n)
  {
    if (b)
      flat_steps<OP, true>(a, b, y, t, n);
    else
      flat_steps<OP, false>(a, b, y, t, n);
  }

  template <class C>
  void steps(const C &c, int64_t j0, int64_t n, carrier_t<typename C::scalar_type> *t)
  {
    using T = carrier_t<typename C::scalar_type>;
    if constexpr (std::is_same_v<typename C::step_type, mptorch::pointwise::ArithStep>)
    {
      if (c.g.flat && !c.step.odd)
      {
        const auto *a = c.a + j0;
        const auto *b = c.g.has_b ? c.b + j0 : nullptr;
        switch (c.step.op)
        {
        case mptorch::pointwise::ADD:
          return flat_steps<mptorch::pointwise::ADD>(a, b, c.y, t, n);
        case mptorch::pointwise::SUB:
          return flat_steps<mptorch::pointwise::SUB>(a, b, c.y, t, n);
        case mptorch::pointwise::MUL:
          return flat_steps<mptorch::pointwise::MUL>(a, b, c.y, t, n);
        case mptorch::pointwise::DIV:
          return flat_steps<mptorch::pointwise::DIV>(a, b, c.y, t, n);
        case mptorch::pointwise::SQRT:
          return flat_steps<mptorch::pointwise::SQRT>(a, b, c.y, t, n);
        case mptorch::pointwise::RSUB:
          return flat_steps<mptorch::pointwise::RSUB>(a, b, c.y, t, n);
        default:
          return flat_steps<mptorch::pointwise::RDIV>(a, b, c.y, t, n);
        }
      }
    }
    walk(c.g, j0, j0 + n, [&](int64_t j, int64_t oa, int64_t ob)
         {
      const T x = static_cast<T>(c.a[oa]);
      const T y = c.g.has_b ? static_cast<T>(c.b[ob]) : c.y;
      t[j - j0] = c.step(x, y); });
  }

  // Elements a chunk handles at a time: the steps into a buffer, then the
  // casts over it, each a loop of its own (the casts' loop is the
  // quantizers', quant_kernel's).
  constexpr int64_t POINTWISE_BLOCK = 1024;

  // A deterministic mode's loop. The call and the cast are copied into the
  // chunk, so that the stores through `o` cannot alias what the loop reads.
  template <bool IsSigned, class C, class Round>
  void run(const C &call, const Round &round)
  {
    using scalar_t = typename C::scalar_type;
    using T = carrier_t<scalar_t>;
    at::parallel_for(0, call.g.numel, quant_grain_size, [&](int64_t begin, int64_t end)
                     {
      const C c = call;
      const Round r = round;
      alignas(64) T t[POINTWISE_BLOCK];
      for (int64_t j0 = begin; j0 < end; j0 += POINTWISE_BLOCK)
      {
        const int64_t n = std::min(POINTWISE_BLOCK, end - j0);
        steps(c, j0, n, t);
        scalar_t *o = c.o + j0;
        for (int64_t k = 0; k < n; ++k)
          o[k] = static_cast<scalar_t>(r(t[k], IsSigned));
      } });
  }

  // RoundMode::SR's loop: element j takes the random word quant_kernel_sr
  // gives element j of a call drawing `seed`.
  template <bool IsSigned, class C, class Round>
  void run_sr(const C &call, uint64_t seed, const Round &round)
  {
    using scalar_t = typename C::scalar_type;
    using T = carrier_t<scalar_t>;
    at::parallel_for(0, call.g.numel, quant_grain_size, [&](int64_t begin, int64_t end)
                     {
      const C c = call;
      const Round r = round;
      alignas(64) T t[POINTWISE_BLOCK];
      PhiloxBlock blk;
      for (int64_t j0 = begin; j0 < end; j0 += POINTWISE_BLOCK)
      {
        const int64_t n = std::min(POINTWISE_BLOCK, end - j0);
        steps(c, j0, n, t);
        for (int64_t k = 0; k < n; ++k)
        {
          const int64_t j = j0 + k;
          if constexpr (std::is_same_v<scalar_t, double>)
          {
            if (j == begin || (j & 1) == 0)
              blk = philox_block(seed, (uint64_t)(j >> 1), 0);
            c.o[j] = static_cast<scalar_t>(r(t[k], blk.word64(2 * (int)(j & 1)), IsSigned));
          }
          else
          {
            if (j == begin || (j & 3) == 0)
              blk = philox_block(seed, (uint64_t)(j >> 2), 0);
            c.o[j] = static_cast<scalar_t>(r(t[k], blk.word((int)(j & 3)), IsSigned));
          }
        }
      } });
  }

  template <RoundMode RM>
  using Mode = std::integral_constant<RoundMode, RM>;

  // The call's round mode, made a template argument: `make(Mode<RM>{})`
  // returns the family's cast in RM.
  template <bool IsSigned, class C, class Make>
  void run_mode(const C &c, RoundMode rm, uint64_t seed, Make &&make)
  {
    switch (rm)
    {
    case RoundMode::RNE:
      run<IsSigned>(c, make(Mode<RoundMode::RNE>{}));
      break;
    case RoundMode::RNA:
      run<IsSigned>(c, make(Mode<RoundMode::RNA>{}));
      break;
    case RoundMode::RU:
      run<IsSigned>(c, make(Mode<RoundMode::RU>{}));
      break;
    case RoundMode::RD:
      run<IsSigned>(c, make(Mode<RoundMode::RD>{}));
      break;
    case RoundMode::RZ:
      run<IsSigned>(c, make(Mode<RoundMode::RZ>{}));
      break;
    case RoundMode::RO:
      run<IsSigned>(c, make(Mode<RoundMode::RO>{}));
      break;
    default: // SR, the one stochastic mode the entries leave
      run_sr<IsSigned>(c, seed, make(Mode<RoundMode::SR>{}));
      break;
    }
  }

  // A call's tensors as the kernels index them: the result, and the operands
  // with the geometry, or copies of their broadcast contiguous when the
  // geometry is too wide (make_geom). Nothing is indexed for an empty result.
  struct Operands
  {
    at::Tensor a, b, o;
    PointwiseGeom g;
  };

  inline Operands operands(const at::Tensor &a, const std::optional<at::Tensor> &b)
  {
    using namespace mptorch::pointwise;
    Operands t{a, b.has_value() ? *b : at::Tensor(), pointwise_result(a, b), PointwiseGeom{}};
    if (t.o.numel() == 0)
      return t;
    if (!make_geom(t.a, b.has_value() ? &t.b : nullptr, t.o, false, t.g))
    {
      t.a = a.expand(t.o.sizes()).contiguous();
      if (b.has_value())
        t.b = b->expand(t.o.sizes()).contiguous();
      make_geom(t.a, b.has_value() ? &t.b : nullptr, t.o, false, t.g);
    }
    return t;
  }

  // The seed under SR. It is drawn before the result's size is looked at,
  // as the quantizers' CPU bodies draw it, so that a call on an empty tensor
  // leaves the generator as the quantizer would.
  inline uint64_t seed_for(RoundMode rm) { return rm == RoundMode::SR ? draw_cpu_seed() : 0; }

  // The body of every family's arithmetic entry point, after its format
  // checks and with the round mode canonical (stochastic_host.h): the
  // operands' checks, the result, the seed, the geometry, and
  // `launch(call, seed)` in the dtype's dispatch.
  template <class Launch>
  at::Tensor run_pointwise(const char *name, const at::Tensor &a,
                           const std::optional<at::Tensor> &b, std::optional<double> scalar,
                           int64_t op, int64_t carrier_rounding, RoundMode rm, Launch &&launch)
  {
    using namespace mptorch::pointwise;
    check_pointwise(a, b, scalar, op, carrier_rounding, name);
    const Operands t = operands(a, b);
    const uint64_t seed = seed_for(rm);
    if (t.o.numel() == 0)
      return t.o;
    const bool odd = carrier_rounding == static_cast<int64_t>(RoundMode::RO);
    MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "pointwise_cpu", [&]
                                 {
      const Call<scalar_t, ArithStep> c{t.a.data_ptr<scalar_t>(),
                                        b.has_value() ? t.b.data_ptr<scalar_t>() : nullptr,
                                        t.o.data_ptr<scalar_t>(),
                                        t.g,
                                        number_in_carrier<carrier_t<scalar_t>, scalar_t>(scalar, odd),
                                        ArithStep{static_cast<int32_t>(op), odd}};
      launch(c, seed); });
    return t.o;
  }

  // The body of every family's table entry point, likewise: the table's
  // checks, the result, the seed, the geometry, and `launch(call, seed)`
  // with the table's step (common/pointwise.h's PwlStep).
  template <class Launch>
  at::Tensor run_pwl(const char *name, const at::Tensor &x, const at::Tensor &entries, double lo,
                     double hi, int64_t tails, bool rsqrt, RoundMode rm, Launch &&launch)
  {
    using namespace mptorch::pointwise;
    check_pwl(x, entries, lo, hi, tails, name);
    const Operands t = operands(x, std::nullopt);
    const uint64_t seed = seed_for(rm);
    if (t.o.numel() == 0)
      return t.o;
    const at::Tensor e = entries.contiguous();
    const int32_t segments = static_cast<int32_t>(e.numel() - 1);
    MPTORCH_DISPATCH_QUANT_TYPES(x.scalar_type(), "pwl_cpu", [&]
                                 {
      using T = carrier_t<scalar_t>;
      const PwlStep<T> step{e.data_ptr<T>(), segments, static_cast<T>(lo), static_cast<T>(hi),
                            static_cast<T>(static_cast<double>(segments) / (hi - lo)),
                            static_cast<int32_t>(tails), rsqrt};
      const Call<scalar_t, PwlStep<T>> c{t.a.data_ptr<scalar_t>(), nullptr,
                                         t.o.data_ptr<scalar_t>(), t.g, T(0), step};
      launch(c, seed); });
    return t.o;
  }

  // A stochastic call's round mode and random bits as a kernel takes them
  // (stochastic_host.h): SR_B and SR_C run as SR, with the variant recorded
  // in prng_bits.
  inline RoundMode canonical(int64_t round_mode, int64_t &prng_bits, const char *name)
  {
    int variant;
    round_mode = mptorch::sr_canonical(round_mode, variant);
    if (variant >= 0)
      prng_bits = mptorch::sr_encode(prng_bits, variant, name);
    return static_cast<RoundMode>(round_mode);
  }
} // namespace mptorch_cpu::pointwise
