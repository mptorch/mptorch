#include "../common/cast_superfp.h"
#include "../common/modes.h"
#include "utils.h"
#include <ATen/ATen.h>
#include <functional>

using namespace at;

namespace
{

  template <typename scalar_t>
  void superfp_kernel_impl(scalar_t *a, scalar_t *o, int size, int man_bits, int exp_bits,
                           int normal_binades, int bias, bool is_signed,
                           RoundMode round_mode, SaturationMode saturation_mode)
  {
    std::function<scalar_t(scalar_t)> quantizer;
    switch (round_mode)
    {
    case RoundMode::RNE:
      quantizer = [man_bits, exp_bits, normal_binades, bias, is_signed,
                   saturation_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_superfp_nearest_even(static_cast<float>(x), man_bits, exp_bits,
                                                                normal_binades, bias, is_signed, saturation_mode));
      };
      break;

    case RoundMode::RNA:
      quantizer = [man_bits, exp_bits, normal_binades, bias, is_signed,
                   saturation_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_superfp_nearest_away(static_cast<float>(x), man_bits, exp_bits,
                                                                normal_binades, bias, is_signed, saturation_mode));
      };
      break;

    case RoundMode::RU:
      quantizer = [man_bits, exp_bits, normal_binades, bias, is_signed,
                   saturation_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_superfp_up(static_cast<float>(x), man_bits, exp_bits,
                                                      normal_binades, bias, is_signed, saturation_mode));
      };
      break;

    case RoundMode::RD:
      quantizer = [man_bits, exp_bits, normal_binades, bias, is_signed,
                   saturation_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_superfp_down(static_cast<float>(x), man_bits, exp_bits,
                                                        normal_binades, bias, is_signed, saturation_mode));
      };
      break;

    case RoundMode::RZ:
      quantizer = [man_bits, exp_bits, normal_binades, bias, is_signed,
                   saturation_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_superfp_zero(static_cast<float>(x), man_bits, exp_bits,
                                                        normal_binades, bias, is_signed, saturation_mode));
      };
      break;

    default: // RO
      quantizer = [man_bits, exp_bits, normal_binades, bias, is_signed,
                   saturation_mode](scalar_t x) -> scalar_t
      {
        return static_cast<scalar_t>(cast_superfp_odd(static_cast<float>(x), man_bits, exp_bits,
                                                       normal_binades, bias, is_signed, saturation_mode));
      };
      break;
    }

    quant_kernel(a, o, size, quantizer);
  }

  template <typename scalar_t>
  void superfp_kernel_sr_impl(scalar_t *a, int *r, scalar_t *o, int size, int man_bits, int exp_bits,
                              int normal_binades, int bias, int prng_bits, bool is_signed,
                              SaturationMode saturation_mode)
  {
    std::function<scalar_t(scalar_t, uint32_t)> quantizer =
        [man_bits, exp_bits, normal_binades, bias, prng_bits, is_signed,
         saturation_mode](scalar_t x, uint32_t rv) -> scalar_t
    {
      return static_cast<scalar_t>(cast_superfp_stochastic(static_cast<float>(x), rv, prng_bits, man_bits,
                                                            exp_bits, normal_binades, bias, is_signed,
                                                            saturation_mode));
    };

    quant_kernel(a, r, o, size, quantizer);
  }

} // namespace

Tensor superfp_quantize_cpu(Tensor a, int64_t man_bits, int64_t exp_bits, int64_t normal_binades,
                            int64_t bias, int64_t prng_bits, bool is_signed,
                            int64_t round_mode, int64_t saturation_mode)
{
  auto o = empty_like(a);
  int size = a.numel();
  RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
  SaturationMode saturation_mode_ = static_cast<SaturationMode>(saturation_mode);

  const int man_bits_ = static_cast<int>(man_bits);
  const int exp_bits_ = static_cast<int>(exp_bits);
  const int normal_binades_ = static_cast<int>(normal_binades);
  const int bias_ = static_cast<int>(bias);
  const int prng_bits_ = static_cast<int>(prng_bits);

  if (round_mode_ != RoundMode::SR)
  {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "superfp_quantize_cpu", [&]
                                    {
      scalar_t *p_a = a.data_ptr<scalar_t>();
      scalar_t *p_o = o.data_ptr<scalar_t>();
      superfp_kernel_impl<scalar_t>(p_a, p_o, size, man_bits_, exp_bits_, normal_binades_, bias_,
                     is_signed, round_mode_, saturation_mode_); });
  }
  else
  {
    auto rand_ints = randint_like(a, INT_MAX, device(a.device()).dtype(kInt));
    int *p_r = rand_ints.data_ptr<int>();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "superfp_quantize_cpu_sr", [&]
                                    {
      scalar_t *p_a = a.data_ptr<scalar_t>();
      scalar_t *p_o = o.data_ptr<scalar_t>();
      superfp_kernel_sr_impl<scalar_t>(p_a, p_r, p_o, size, man_bits_, exp_bits_, normal_binades_, bias_,
                     prng_bits_, is_signed, saturation_mode_); });
  }

  return o;
}
