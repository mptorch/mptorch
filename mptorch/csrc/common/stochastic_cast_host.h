#pragma once

// The host side of _stochastic_cast (common/stochastic_cast.h): its checks
// and its job, shared by the CPU kernel and the CUDA entry point.
//
//   _stochastic_cast(Tensor x, Tensor draws, int family, int[] format,
//                    int stochastic_mode) -> Tensor
//
// x is float32 or float64, draws an int64 tensor of x's shape and device
// holding each element's random word (the low 32 bits in binary32, all 64 in
// binary64), and format the family's fields as its quantizer's schema spells
// them: binaryK [K, P, bias, prng_bits, is_signed, saturation, subnormals],
// superfp [man_bits, exp_bits, normal_binades, bias, prng_bits, is_signed,
// saturation], fixed point [wl, fl, prng_bits, is_signed, symmetric].
// stochastic_mode is RoundMode::SR, SR_B or SR_C.

#include "stochastic_cast.h"
#include "stochastic_host.h"
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

namespace mptorch::stochastic_cast
{
  inline void check_tensors(const at::Tensor &x, const at::Tensor &draws)
  {
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
                "_stochastic_cast: x is float32 or float64, got ", x.scalar_type());
    TORCH_CHECK(draws.scalar_type() == at::kLong && draws.device() == x.device() &&
                    draws.sizes() == x.sizes(),
                "_stochastic_cast: draws is an int64 tensor of x's shape and device");
  }

  template <class T>
  Job<T> make_job(int64_t family, c10::IntArrayRef format, int64_t stochastic_mode)
  {
    constexpr const char *op = "_stochastic_cast";
    TORCH_CHECK(family >= 0 && family <= 2, op, ": family is 0 (binaryK), 1 (superfp) or 2 (fixed point), got ",
                family);
    TORCH_CHECK(is_round_mode(stochastic_mode) && is_stochastic_mode(stochastic_mode), op,
                ": stochastic_mode is RoundMode.SR, SR_B or SR_C, got ", stochastic_mode);
    static const int lengths[3] = {7, 7, 5};
    TORCH_CHECK(static_cast<int64_t>(format.size()) == lengths[family], op, ": a family ", family,
                " format has ", lengths[family], " fields, got ", format.size());
    int variant;
    sr_canonical(stochastic_mode, variant);
    Job<T> j{};
    j.family = static_cast<int>(family);
    if (family == 0)
    {
      const int K = static_cast<int>(format[0]), P = static_cast<int>(format[1]);
      j.is_signed = format[4] != 0;
      j.subnormals = static_cast<SubnormalsMode>(format[6]);
      j.spec = sr_encode(format[3], variant, op);
      // as the quantizer derives them: P - 1 mantissa bits, the rest after
      // the sign the exponent's
      j.bk = make_binaryK_params<T>(P - 1, j.is_signed ? K - P : K - P + 1, static_cast<int>(format[2]),
                                    j.is_signed, static_cast<SaturationMode>(format[5]),
                                    j.subnormals == SubnormalsMode::EXTENDED_NORMALS);
    }
    else if (family == 1)
    {
      j.is_signed = format[5] != 0;
      j.subnormals = SubnormalsMode::SUBNORMALS;
      j.spec = sr_encode(format[4], variant, op);
      j.sf = make_superfp_params<T>(static_cast<int>(format[0]), static_cast<int>(format[1]),
                                    static_cast<int>(format[2]), static_cast<int>(format[3]),
                                    static_cast<SaturationMode>(format[6]));
    }
    else
    {
      j.is_signed = format[3] != 0;
      j.subnormals = SubnormalsMode::SUBNORMALS;
      j.spec = sr_encode(format[2], variant, op);
      j.fx = make_fixedpoint_params<T>(static_cast<int>(format[0]), static_cast<int>(format[1]), j.is_signed,
                                       format[4] != 0);
    }
    return j;
  }
} // namespace mptorch::stochastic_cast
