#include "../common/cast_binaryK.h"
#include "../common/modes.h"
#include "utils.h"
#include <ATen/ATen.h>
#include <functional>

using namespace at;

namespace
{

  template <typename scalar_t>
  void binaryK_kernel_impl(scalar_t *a, scalar_t *o, int size, int K, int P, int bias,
                           bool is_signed, RoundMode round_mode,
                           SaturationMode saturation_mode,
                           SubnormalsMode subnormals_mode)
  {
    int man_bits, exp_bits;
    if (is_signed)
    {
      man_bits = P - 1;
      exp_bits = K - P;
    }
    else
    {
      man_bits = P - 1;
      exp_bits = K - P + 1;
    }

    std::function<scalar_t(scalar_t)> quantizer;
    switch (round_mode)
    {
    case RoundMode::RNE:
      quantizer = [man_bits, exp_bits, bias, is_signed, saturation_mode,
                   subnormals_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_binaryK_nearest_even(static_cast<float>(x), man_bits, exp_bits, bias, is_signed,
                                                               saturation_mode, subnormals_mode));
      };
      break;

    case RoundMode::RNA:
      quantizer = [man_bits, exp_bits, bias, is_signed, saturation_mode,
                   subnormals_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_binaryK_nearest_away(static_cast<float>(x), man_bits, exp_bits, bias, is_signed,
                                                               saturation_mode, subnormals_mode));
      };
      break;

    case RoundMode::RU:
      quantizer = [man_bits, exp_bits, bias, is_signed, saturation_mode,
                   subnormals_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_binaryK_up(static_cast<float>(x), man_bits, exp_bits, bias, is_signed,
                                                     saturation_mode, subnormals_mode));
      };
      break;

    case RoundMode::RD:
      quantizer = [man_bits, exp_bits, bias, is_signed, saturation_mode,
                   subnormals_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_binaryK_down(static_cast<float>(x), man_bits, exp_bits, bias, is_signed,
                                                       saturation_mode, subnormals_mode));
      };
      break;

    case RoundMode::RZ:
      quantizer = [man_bits, exp_bits, bias, is_signed, saturation_mode,
                   subnormals_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_binaryK_zero(static_cast<float>(x), man_bits, exp_bits, bias, is_signed,
                                                       saturation_mode, subnormals_mode));
      };
      break;

    default: // RO
      quantizer = [man_bits, exp_bits, bias, is_signed, saturation_mode,
                   subnormals_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_binaryK_odd(static_cast<float>(x), man_bits, exp_bits, bias, is_signed,
                                                      saturation_mode, subnormals_mode));
      };
      break;
    }

    quant_kernel(a, o, size, quantizer);
  }

  template <typename scalar_t>
  void binaryK_kernel_sr_impl(scalar_t *a, int *r, scalar_t *o, int size, int K, int P,
                              int bias, int prng_bits, bool is_signed,
                              RoundMode round_mode, SaturationMode saturation_mode,
                              SubnormalsMode subnormals_mode)
  {
    int man_bits, exp_bits;
    if (is_signed)
    {
      man_bits = P - 1;
      exp_bits = K - P;
    }
    else
    {
      man_bits = P - 1;
      exp_bits = K - P + 1;
    }

    std::function<scalar_t(scalar_t, uint32_t)> quantizer =
        [man_bits, exp_bits, bias, is_signed, prng_bits, saturation_mode,
         subnormals_mode](scalar_t x, uint32_t rv) -> scalar_t
    {
      return static_cast<scalar_t>(cast_binaryK_stochastic(static_cast<float>(x), rv, prng_bits, man_bits, exp_bits,
                                                           bias, is_signed, saturation_mode,
                                                           subnormals_mode));
    };

    quant_kernel(a, r, o, size, quantizer);
  }

} // namespace

Tensor binaryK_quantize_cpu(Tensor a, int64_t K, int64_t P, int64_t bias,
                            int64_t prng_bits, bool is_signed,
                            int64_t round_mode, int64_t saturation_mode,
                            int64_t subnormals_mode)
{
  auto o = empty_like(a);
  int size = a.numel();
  RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
  SubnormalsMode subnormals_mode_ =
      static_cast<SubnormalsMode>(subnormals_mode);
  SaturationMode saturation_mode_ =
      static_cast<SaturationMode>(saturation_mode);

  const int K_ = static_cast<int>(K);
  const int P_ = static_cast<int>(P);
  const int bias_ = static_cast<int>(bias);
  const int prng_bits_ = static_cast<int>(prng_bits);

  if (round_mode_ != RoundMode::SR)
  {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "binaryK_quantize_cpu", [&]
                                    {
      scalar_t *p_a = a.data_ptr<scalar_t>();
      scalar_t *p_o = o.data_ptr<scalar_t>();
      binaryK_kernel_impl<scalar_t>(p_a, p_o, size, K_, P_, bias_, is_signed, round_mode_,
                     saturation_mode_, subnormals_mode_); });
  }
  else
  {
    auto rand_ints = randint_like(a, INT_MAX, device(a.device()).dtype(kInt));
    int *p_r = rand_ints.data_ptr<int>();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "binaryK_quantize_cpu_sr", [&]
                                    {
      scalar_t *p_a = a.data_ptr<scalar_t>();
      scalar_t *p_o = o.data_ptr<scalar_t>();
      binaryK_kernel_sr_impl<scalar_t>(p_a, p_r, p_o, size, K_, P_, bias_, prng_bits_, is_signed,
                     round_mode_, saturation_mode_, subnormals_mode_); });
  }

  return o;
}
