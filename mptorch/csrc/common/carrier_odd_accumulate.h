#pragma once

// The carrier step rounded to odd under KAHAN, BLOCK and TREE: the
// custom_matmul_*_accumulated_odd ops, and the odd conv ops' calls past
// NAIVE (carrier_rounding=RoundMode.RO with an accumulate algorithm).
//
// The algorithms are gemm_accumulate.h's, written once. At each carrier
// operation whose result a format rounds, they test carrier_rounds_to_odd and
// call the odd twin of the plain expression (odd_sum, odd_diff, odd_fold),
// which this header defines for N's odd macs (carrier_odd.h) and which
// argument-dependent lookup finds when a kernel is instantiated. The rule is
// N's: a carrier operation a format cast rounds is rounded to odd, so the cast
// rounds the exact value; one that no format rounds stays the carrier's
// nearest operation. Per algorithm:
//
//   KAHAN   y = add(mul(a*b) - c)   the product odd; the difference odd over
//                                    a real adder, nearest over the identity
//           fused: add(fma(a, b, -c)), the fma odd
//           t = add(s + y), c = add(add(t - s) - y): odd over a real adder,
//                                    nearest over the identity
//   BLOCK   blk: the odd mac's step (N's)
//           tot = outer(tot + blk): odd when the outer format is on
//   TREE    p = mul(a*b): the product odd
//           v = add(level + v): odd over a real adder, nearest otherwise
//           tot = outer(tot + root): odd when the outer format is on
//
// A difference x - y is odd::add(x, -y): the negation is exact, and the
// zero signs are IEEE subtraction's (x - y and x + (-y) agree on every
// non-NaN input). A NaN is a NaN of the backend's, as for every odd
// primitive.
//
// TREE's product and KAHAN's term need the multiply to round the product to
// odd, which N's SplitMacOdd does inside its step only. accumulate_mac maps it
// to AccumulatedSplitMacOdd, whose `mul` does it on every call, so TREE's
// `mac.mul(a, b, rng)` needs no test of its own.
//
// The bounds that make each cast exact (ops._odd_findings) cover the outer
// format as well as the mac's.
//
// Only phase Y's units include this header, and the odd entry and Meta units
// that name their ops: everything here is a template that only Y's kernel
// units instantiate. It also holds AccumulateFamily<OddArgs<B>>, without which
// AccumulateArgs<OddArgs<B>> does not compile, so no unit can instantiate an
// odd accumulator without these specializations in view; without them the
// algorithms would take an odd mac's plain carrier operations, silently.

#include "carrier_odd.h"
#include "gemm_accumulate.h"

#if !defined(__METAL_VERSION__)

namespace mptorch::gemm
{
  // A multiplier whose product is rounded to odd before the multiply format's
  // cast, as SplitMacOdd's step forms it: the odd product handed to the
  // multiplier with 1 as the other factor, exact and folded away.
  template <class T, class Multiplier>
  struct OddProduct
  {
    using value_t = T;
    Multiplier inner{};

    CUDA_HOST_DEVICE_INLINE T operator()(T a, T b, PhiloxEngine &rng) const
    {
      return inner(odd::mul(a, b), T(1), rng);
    }
  };

  // SplitMacOdd with that multiplier: the same step, and a `mul` that the
  // algorithms can call on its own.
  template <class Multiplier, class Adder>
  struct AccumulatedSplitMacOdd
  {
    using value_t = typename Adder::value_t;
    OddProduct<value_t, Multiplier> mul{};
    Adder add{};

    AccumulatedSplitMacOdd() = default;
    CUDA_HOST_DEVICE_INLINE AccumulatedSplitMacOdd(const SplitMacOdd<Multiplier, Adder> &m)
        : mul{m.mul}, add(m.add)
    {
    }

    CUDA_HOST_DEVICE_INLINE value_t step(value_t a, value_t b, value_t acc, PhiloxEngine &rng) const
    {
      const value_t p = mul(a, b, rng);
      if constexpr (is_identity_adder<Adder>::value)
        return acc + p;
      else
        return add(odd::add(acc, p), rng);
    }
  };

  // The CUDA kernels take the accumulator, mac included, as a __global__
  // parameter, so the odd mac is the plain one's size and trivially copyable.
  template <class Multiplier, class Adder>
  inline constexpr bool accumulated_mac_fits_v =
      sizeof(AccumulatedSplitMacOdd<Multiplier, Adder>) == sizeof(SplitMac<Multiplier, Adder>) &&
      std::is_trivially_copyable_v<AccumulatedSplitMacOdd<Multiplier, Adder>>;

  // KAHAN's term: the product less the compensation, rounded to the
  // accumulate format.
  template <class Multiplier, class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  mac_term(const AccumulatedSplitMacOdd<Multiplier, Adder> &m, typename Adder::value_t a,
           typename Adder::value_t b, typename Adder::value_t c, PhiloxEngine &rng)
  {
    static_assert(accumulated_mac_fits_v<Multiplier, Adder>);
    const typename Adder::value_t p = m.mul(a, b, rng);
    if constexpr (is_identity_adder<Adder>::value)
      return m.add(p - c, rng);
    else
      return m.add(odd::add(p, -c), rng);
  }

  template <class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  mac_term(const FusedMacOdd<Adder> &m, typename Adder::value_t a, typename Adder::value_t b,
           typename Adder::value_t c, PhiloxEngine &rng)
  {
    return m.add(odd::fma(a, b, -c), rng);
  }

  // A sum or difference a format rounds next: odd over a real adder, the
  // carrier's nearest over the identity. A FusedMacOdd always has a real
  // adder (OddArgs builds none over the identity: the entry points refuse
  // fma_quant = false).
  template <class Multiplier, class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_sum(const AccumulatedSplitMacOdd<Multiplier, Adder> &, typename Adder::value_t x,
          typename Adder::value_t y)
  {
    if constexpr (is_identity_adder<Adder>::value)
      return x + y;
    else
      return odd::add(x, y);
  }

  template <class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_sum(const FusedMacOdd<Adder> &, typename Adder::value_t x, typename Adder::value_t y)
  {
    static_assert(!is_identity_adder<Adder>::value, "an odd fused step is always rounded again");
    return odd::add(x, y);
  }

  template <class Multiplier, class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_diff(const AccumulatedSplitMacOdd<Multiplier, Adder> &, typename Adder::value_t x,
           typename Adder::value_t y)
  {
    if constexpr (is_identity_adder<Adder>::value)
      return x - y;
    else
      return odd::add(x, -y);
  }

  template <class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_diff(const FusedMacOdd<Adder> &, typename Adder::value_t x, typename Adder::value_t y)
  {
    static_assert(!is_identity_adder<Adder>::value, "an odd fused step is always rounded again");
    return odd::add(x, -y);
  }

  // BLOCK's and TREE's fold of a block into the total: odd when the outer
  // format rounds it, the carrier's nearest sum otherwise.
  template <class Multiplier, class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_fold(const AccumulatedSplitMacOdd<Multiplier, Adder> &, bool outer_quant,
           typename Adder::value_t x, typename Adder::value_t y)
  {
    return outer_quant ? odd::add(x, y) : x + y;
  }

  template <class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_fold(const FusedMacOdd<Adder> &, bool outer_quant, typename Adder::value_t x,
           typename Adder::value_t y)
  {
    return outer_quant ? odd::add(x, y) : x + y;
  }

  // An odd op's accumulation is its twin's: the outer format's family and
  // whether there is a product to build a tree of. With it, AccumulateArgs
  // wraps the odd Args, and its with_accumulator hands the algorithms the
  // odd mac OddArgs builds (accumulate_mac's, below).
  template <class Base>
  struct AccumulateFamily<OddArgs<Base>> : AccumulateFamily<Base>
  {
  };
} // namespace mptorch::gemm

template <class Multiplier, class Adder>
struct accumulate_mac<mptorch::gemm::SplitMacOdd<Multiplier, Adder>>
{
  using type = mptorch::gemm::AccumulatedSplitMacOdd<Multiplier, Adder>;
};

template <class Multiplier, class Adder>
struct carrier_rounds_to_odd<mptorch::gemm::AccumulatedSplitMacOdd<Multiplier, Adder>>
{
  static constexpr bool value = true;
};

template <class Adder>
struct carrier_rounds_to_odd<mptorch::gemm::FusedMacOdd<Adder>>
{
  static constexpr bool value = true;
};

// BLOCK's unroll factor treats an odd fused mac as the fused mac it rounds
// like.
template <class Adder>
struct is_fused_mac<mptorch::gemm::FusedMacOdd<Adder>>
{
  static constexpr bool value = true;
};

#endif // !__METAL_VERSION__
