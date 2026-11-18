#pragma once

// The host side of the three stochastic modes (common/stochastic.h): where a
// call's RoundMode::SR_B or SR_C becomes RoundMode::SR, with the variant
// recorded next to N in the prng_bits of each of the call's formats. Only the
// entry points and the drivers do this, before the round mode becomes a
// template argument, so dispatch_round_mode and every `== RoundMode::SR`
// test after it see a single stochastic mode, and each kernel has a single
// stochastic instantiation.

#include "gemm_dtype.h"
#include "stochastic.h"
#include <c10/util/Exception.h>
#include <cstdint>

namespace mptorch
{
  // Whether a validated round mode is stochastic.
  inline bool is_stochastic_mode(int64_t round_mode)
  {
    return round_mode == static_cast<int64_t>(RoundMode::SR) ||
           round_mode == static_cast<int64_t>(RoundMode::SR_B) ||
           round_mode == static_cast<int64_t>(RoundMode::SR_C);
  }

  // Returns the round mode a kernel runs the call in, and sets `variant` to
  // the stochastic variant: 0 for SR, 1 for SR_B, 2 for SR_C, and -1 for a
  // deterministic mode.
  inline int64_t sr_canonical(int64_t round_mode, int &variant)
  {
    variant = is_stochastic_mode(round_mode) ? sr_variant_of(static_cast<RoundMode>(round_mode)) : -1;
    return variant >= 0 ? static_cast<int64_t>(RoundMode::SR) : round_mode;
  }

  // A format's prng_bits under a stochastic call: N, checked to fit in the
  // spec's low byte (a larger value from a direct torch.ops call would be read
  // as a variant), with the variant recorded above it.
  inline int sr_encode(int64_t prng_bits, int variant, const char *op)
  {
    TORCH_CHECK(prng_bits >= 0 && prng_bits <= 255, op, ": prng_bits must be in [0, 255], got ",
                prng_bits);
    return sr_spec(static_cast<int>(prng_bits), variant);
  }

  // Encodes every format of a GEMM's Args: the format slots of the plain,
  // palette and cross-family Args (mul_c, acc_c, fma_c), the outer format of
  // the accumulated ones, and, recursively, the Args the odd, accumulated,
  // conv and block wrappers hold (`base`, `inner`).
  template <class Args>
  void sr_encode_args(Args &args, int variant, const char *op)
  {
    if constexpr (requires { args.mul_c.prng_bits; })
      args.mul_c.prng_bits = sr_encode(args.mul_c.prng_bits, variant, op);
    if constexpr (requires { args.acc_c.prng_bits; })
      args.acc_c.prng_bits = sr_encode(args.acc_c.prng_bits, variant, op);
    if constexpr (requires { args.fma_c.prng_bits; })
      args.fma_c.prng_bits = sr_encode(args.fma_c.prng_bits, variant, op);
    if constexpr (requires { args.outer_c.prng_bits; })
      args.outer_c.prng_bits = sr_encode(args.outer_c.prng_bits, variant, op);
    if constexpr (requires { args.base; })
      sr_encode_args(args.base, variant, op);
    if constexpr (requires { args.inner; })
      sr_encode_args(args.inner, variant, op);
  }

  // A GEMM call's Args as its kernel runs them: canonicalizes round_mode in
  // place, and returns a copy of the Args whose formats are encoded when the
  // mode is stochastic.
  template <class Args>
  Args sr_canonical_args(const Args &args, int64_t &round_mode, const char *op)
  {
    int variant;
    round_mode = sr_canonical(round_mode, variant);
    Args out = args;
    if (variant >= 0)
      sr_encode_args(out, variant, op);
    return out;
  }
  // The same for a block format's descriptor (its int[] fields in
  // block_decode.h's BlockFmtField order; `prng_field` is ELEM_PRNG_BITS):
  // canonicalizes round_mode in place, and encodes the element format's
  // prng_bits, which checked_block_fmt has already validated.
  template <class Fmt>
  void sr_canonical_fmt(Fmt &f, int prng_field, int64_t &round_mode, const char *op)
  {
    int variant;
    round_mode = sr_canonical(round_mode, variant);
    if (variant >= 0)
      f[prng_field] = sr_encode(f[prng_field], variant, op);
  }
} // namespace mptorch
