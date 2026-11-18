#pragma once

// The carrier step rounded to odd under KAHAN, BLOCK and TREE: the
// custom_matmul_*_accumulated_odd ops, and the odd conv ops' calls with an
// algorithm other than NAIVE (carrier_rounding=RoundMode.RO with an
// accumulate algorithm).
//
// The algorithms are gemm_accumulate.h's, written once. At each carrier
// operation whose result a format then rounds, they test
// carrier_rounds_to_odd and call the round-to-odd version of the plain
// expression (odd_sum, odd_diff, odd_fold). This header defines those for
// the odd macs of carrier_odd.h, and argument-dependent lookup finds them
// when a kernel is instantiated. The rule is the one the NAIVE odd ops
// follow: a carrier operation whose result a format's cast rounds is rounded
// to odd, so that the cast rounds the exact value, and one whose result no
// format rounds stays the carrier's operation rounded to nearest. Per
// algorithm:
//
//   KAHAN   y = add(mul(a*b) - c)   the product odd; the difference odd over
//                                    a real adder, nearest over the identity
//           fused: add(fma(a, b, -c)), the fma odd
//           t = add(s + y), c = add(add(t - s) - y): odd over a real adder,
//                                    nearest over the identity
//   BLOCK   blk: the odd mac's step (as for NAIVE)
//           tot = outer(tot + blk): odd when the outer format is on
//   TREE    p = mul(a*b): the product odd
//           v = add(level + v): odd over a real adder, nearest otherwise
//           tot = outer(tot + root): odd when the outer format is on
//
// A difference x - y is computed as odd::add(x, -y): the negation is exact,
// and the signs of zero results are IEEE subtraction's (x - y and x + (-y)
// agree on every input that is not a NaN). A NaN is whatever NaN the backend
// produces, as for every odd primitive.
//
// TREE's product and KAHAN's term need the multiply to round the product to
// odd, which SplitMacOdd does only inside its step. accumulate_mac maps it
// to AccumulatedSplitMacOdd, whose `mul` does so on every call, so TREE's
// `mac.mul(a, b, rng)` needs no test of its own.
//
// The bounds that make each cast exact (mptorch/quant/ops/_odd.py's
// _odd_findings) cover the outer format as well as the mac's formats.
//
// Only the 72 instantiation units of these ops, the odd ops' entry and Meta
// files and the MPS backend include this header; everything here is a
// template that only those kernel units instantiate. The MPS backend
// compiles it as Metal too (mps/odd.metal). There, the plain nearest
// operations (in the identity-adder cases, and in a fold with no outer
// format) are carrier_odd_metal.h's add_rn and sub_rn, which do not flush
// subnormals, and the odd product goes straight to the multiply format's
// cast; each of these is behind __METAL_VERSION__, with the host and CUDA
// statement, unchanged, in the other branch.
//
// This header also holds AccumulateFamily<OddArgs<B>>, without which
// AccumulateArgs<OddArgs<B>> does not compile. So no unit can instantiate an
// odd accumulator without also seeing the specializations here; without
// them, the algorithms would silently use the odd mac's plain carrier
// operations. tests/test_accumulators.py checks that each of the 72 units
// includes this header.

#include "carrier_odd.h"
#include "gemm_accumulate.h"

namespace mptorch::gemm
{
  // A multiplier whose product is rounded to odd before the multiply format's
  // cast, as SplitMacOdd's step computes it: the odd product is passed to
  // the multiplier with 1 as the other factor, which is exact and which the
  // compiler removes. (On Metal it goes straight to the multiplier's cast.)
  template <class T, class Multiplier>
  struct OddProduct
  {
    using value_t = T;
    Multiplier inner{};

    CUDA_HOST_DEVICE_INLINE T operator()(T a, T b, MPTORCH_THREAD PhiloxEngine &rng) const
    {
#if defined(__METAL_VERSION__)
      return odd::cast_product(inner, odd::mul(a, b), rng);
#else
      return inner(odd::mul(a, b), T(1), rng);
#endif
    }
  };

  // SplitMacOdd with that multiplier: the same step, plus a `mul` that the
  // algorithms can call on its own.
  template <class Multiplier, class Adder>
  struct AccumulatedSplitMacOdd
  {
    using value_t = typename Adder::value_t;
    OddProduct<value_t, Multiplier> mul{};
    Adder add{};

    AccumulatedSplitMacOdd() = default;
    CUDA_HOST_DEVICE_INLINE AccumulatedSplitMacOdd(const MPTORCH_THREAD SplitMacOdd<Multiplier, Adder> &m)
        : mul{m.mul}, add(m.add)
    {
    }

    CUDA_HOST_DEVICE_INLINE value_t step(value_t a, value_t b, value_t acc, MPTORCH_THREAD PhiloxEngine &rng) const
    {
      const value_t p = mul(a, b, rng);
      if constexpr (is_identity_adder<Adder>::value)
#if defined(__METAL_VERSION__)
        return odd::add_rn(acc, p);
#else
        return acc + p;
#endif
      else
        return add(odd::add(acc, p), rng);
    }
  };

  // The CUDA kernels take the accumulator, mac included, as a kernel
  // parameter, so the odd mac must have the plain mac's size and be
  // trivially copyable.
#if !defined(__METAL_VERSION__)
  template <class Multiplier, class Adder>
  inline constexpr bool accumulated_mac_fits_v =
      sizeof(AccumulatedSplitMacOdd<Multiplier, Adder>) == sizeof(SplitMac<Multiplier, Adder>) &&
      std::is_trivially_copyable_v<AccumulatedSplitMacOdd<Multiplier, Adder>>;
#endif

  // KAHAN's term: the product minus the compensation, rounded to the
  // accumulate format.
  template <class Multiplier, class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  mac_term(const MPTORCH_THREAD AccumulatedSplitMacOdd<Multiplier, Adder> &m, typename Adder::value_t a,
           typename Adder::value_t b, typename Adder::value_t c, MPTORCH_THREAD PhiloxEngine &rng)
  {
#if !defined(__METAL_VERSION__)
    static_assert(accumulated_mac_fits_v<Multiplier, Adder>);
#endif
    const typename Adder::value_t p = m.mul(a, b, rng);
    if constexpr (is_identity_adder<Adder>::value)
#if defined(__METAL_VERSION__)
      return m.add(odd::sub_rn(p, c), rng);
#else
      return m.add(p - c, rng);
#endif
    else
      return m.add(odd::add(p, -c), rng);
  }

  template <class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  mac_term(const MPTORCH_THREAD FusedMacOdd<Adder> &m, typename Adder::value_t a, typename Adder::value_t b,
           typename Adder::value_t c, MPTORCH_THREAD PhiloxEngine &rng)
  {
    return m.add(odd::fma(a, b, -c), rng);
  }

  // A sum or difference whose result a format rounds next: rounded to odd
  // over a real adder, and the carrier's nearest operation over the
  // identity. A FusedMacOdd always has a real adder (OddArgs builds none over
  // the identity, since the entry points refuse fma_quant = false).
  template <class Multiplier, class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_sum(const MPTORCH_THREAD AccumulatedSplitMacOdd<Multiplier, Adder> &, typename Adder::value_t x,
          typename Adder::value_t y)
  {
    if constexpr (is_identity_adder<Adder>::value)
#if defined(__METAL_VERSION__)
      return odd::add_rn(x, y);
#else
      return x + y;
#endif
    else
      return odd::add(x, y);
  }

  template <class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_sum(const MPTORCH_THREAD FusedMacOdd<Adder> &, typename Adder::value_t x, typename Adder::value_t y)
  {
    static_assert(!is_identity_adder<Adder>::value, "an odd fused step is always rounded again");
    return odd::add(x, y);
  }

  template <class Multiplier, class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_diff(const MPTORCH_THREAD AccumulatedSplitMacOdd<Multiplier, Adder> &, typename Adder::value_t x,
           typename Adder::value_t y)
  {
    if constexpr (is_identity_adder<Adder>::value)
#if defined(__METAL_VERSION__)
      return odd::sub_rn(x, y);
#else
      return x - y;
#endif
    else
      return odd::add(x, -y);
  }

  template <class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_diff(const MPTORCH_THREAD FusedMacOdd<Adder> &, typename Adder::value_t x, typename Adder::value_t y)
  {
    static_assert(!is_identity_adder<Adder>::value, "an odd fused step is always rounded again");
    return odd::add(x, -y);
  }

  // BLOCK's and TREE's addition of a block's sum to the total: rounded to
  // odd when the outer format rounds it, and the carrier's nearest sum
  // otherwise.
  template <class Multiplier, class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_fold(const MPTORCH_THREAD AccumulatedSplitMacOdd<Multiplier, Adder> &, bool outer_quant,
           typename Adder::value_t x, typename Adder::value_t y)
  {
#if defined(__METAL_VERSION__)
    return outer_quant ? odd::add(x, y) : odd::add_rn(x, y);
#else
    return outer_quant ? odd::add(x, y) : x + y;
#endif
  }

  template <class Adder>
  CUDA_HOST_DEVICE_INLINE typename Adder::value_t
  odd_fold(const MPTORCH_THREAD FusedMacOdd<Adder> &, bool outer_quant, typename Adder::value_t x,
           typename Adder::value_t y)
  {
#if defined(__METAL_VERSION__)
    return outer_quant ? odd::add(x, y) : odd::add_rn(x, y);
#else
    return outer_quant ? odd::add(x, y) : x + y;
#endif
  }

  // An odd op accumulates like its twin: the same family of outer format, and
  // the same answer to whether there is a separate product for TREE. With
  // this, AccumulateArgs can wrap the odd Args, and its with_accumulator
  // hands the algorithms the odd mac that OddArgs builds (mapped by
  // accumulate_mac, below).
#if defined(__METAL_VERSION__)
  // The same members, written out, since Metal has no derived classes.
  template <class Base>
  struct AccumulateFamily<OddArgs<Base>>
  {
    using Widths = typename AccumulateFamily<Base>::Widths;
    using Common = typename AccumulateFamily<Base>::Common;
    static constexpr MPTORCH_CONSTANT bool has_product = AccumulateFamily<Base>::has_product;
  };
#else
  template <class Base>
  struct AccumulateFamily<OddArgs<Base>> : AccumulateFamily<Base>
  {
  };
#endif
} // namespace mptorch::gemm

template <class Multiplier, class Adder>
struct accumulate_mac<mptorch::gemm::SplitMacOdd<Multiplier, Adder>>
{
  using type = mptorch::gemm::AccumulatedSplitMacOdd<Multiplier, Adder>;
};

template <class Multiplier, class Adder>
struct carrier_rounds_to_odd<mptorch::gemm::AccumulatedSplitMacOdd<Multiplier, Adder>>
{
  static constexpr MPTORCH_CONSTANT bool value = true;
};

template <class Adder>
struct carrier_rounds_to_odd<mptorch::gemm::FusedMacOdd<Adder>>
{
  static constexpr MPTORCH_CONSTANT bool value = true;
};

// BLOCK's unroll factor treats an odd fused mac like a fused mac, since it
// rounds once per step too.
template <class Adder>
struct is_fused_mac<mptorch::gemm::FusedMacOdd<Adder>>
{
  static constexpr MPTORCH_CONSTANT bool value = true;
};
