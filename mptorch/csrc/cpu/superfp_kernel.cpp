#include "../common/cast_superfp.h"
#include "../common/modes.h"
#include "utils.h"
#include <ATen/ATen.h>

using namespace at;

namespace
{

  // One body for all six deterministic round modes, and the superfp twin of
  // binaryK_kernel.cpp's binaryK_run: `Cast` is a per-arm closure type rather
  // than a function pointer, `IsSigned` is a template parameter, and `p` is a
  // precomputed SuperfpParams. See that function for what each is worth.
  template <typename scalar_t, bool IsSigned, class Cast>
  void superfp_run(const scalar_t *a, scalar_t *o, int64_t size, const SuperfpParams &p,
                   Cast cast)
  {
    quant_kernel(a, o, size,
                 [=](scalar_t x) -> scalar_t
                 {
                   return static_cast<scalar_t>(
                       cast(static_cast<float>(x), IsSigned, p));
                 });
  }

  template <typename scalar_t, bool IsSigned>
  void superfp_kernel_impl(const scalar_t *a, scalar_t *o, int64_t size, int man_bits,
                           int exp_bits, int normal_binades, int bias,
                           RoundMode round_mode, SaturationMode saturation_mode)
  {
    const SuperfpParams p = make_superfp_params(man_bits, exp_bits, normal_binades,
                                                bias, saturation_mode);

    switch (round_mode)
    {
    case RoundMode::RNE:
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](float v, bool sg, const SuperfpParams &q) { return cast_superfp_nearest_even(v, sg, q); });
      break;

    case RoundMode::RNA:
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](float v, bool sg, const SuperfpParams &q) { return cast_superfp_nearest_away(v, sg, q); });
      break;

    case RoundMode::RU:
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](float v, bool sg, const SuperfpParams &q) { return cast_superfp_up(v, sg, q); });
      break;

    case RoundMode::RD:
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](float v, bool sg, const SuperfpParams &q) { return cast_superfp_down(v, sg, q); });
      break;

    case RoundMode::RZ:
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](float v, bool sg, const SuperfpParams &q) { return cast_superfp_zero(v, sg, q); });
      break;

    default: // RO
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](float v, bool sg, const SuperfpParams &q) { return cast_superfp_odd(v, sg, q); });
      break;
    }
  }

  template <typename scalar_t, bool IsSigned>
  void superfp_kernel_sr_impl(const scalar_t *a, const int *r, scalar_t *o, int64_t size,
                              int man_bits, int exp_bits, int normal_binades, int bias,
                              int prng_bits, SaturationMode saturation_mode)
  {
    const SuperfpParams p = make_superfp_params(man_bits, exp_bits, normal_binades,
                                                bias, saturation_mode);

    quant_kernel(a, r, o, size,
                 [=](scalar_t x, uint32_t rv) -> scalar_t
                 {
                   return static_cast<scalar_t>(cast_superfp_stochastic(
                       static_cast<float>(x), rv, prng_bits, IsSigned, p));
                 });
  }

} // namespace

Tensor superfp_quantize_cpu(Tensor a, int64_t man_bits, int64_t exp_bits, int64_t normal_binades,
                            int64_t bias, int64_t prng_bits, bool is_signed,
                            int64_t round_mode, int64_t saturation_mode)
{
  // see binaryK_quantize_cpu for why the input is made contiguous here
  auto a_c = a.contiguous();
  auto o = empty_like(a_c);
  const int64_t size = a_c.numel(); // int would truncate past 2^31 elements
  RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
  SaturationMode saturation_mode_ = static_cast<SaturationMode>(saturation_mode);

  const int man_bits_ = static_cast<int>(man_bits);
  const int exp_bits_ = static_cast<int>(exp_bits);
  const int normal_binades_ = static_cast<int>(normal_binades);
  const int bias_ = static_cast<int>(bias);
  const int prng_bits_ = static_cast<int>(prng_bits);

  if (round_mode_ != RoundMode::SR)
  {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a_c.scalar_type(), "superfp_quantize_cpu", [&]
                                    {
      const scalar_t *p_a = a_c.data_ptr<scalar_t>();
      scalar_t *p_o = o.data_ptr<scalar_t>();
      if (is_signed)
        superfp_kernel_impl<scalar_t, true>(p_a, p_o, size, man_bits_, exp_bits_, normal_binades_,
                       bias_, round_mode_, saturation_mode_);
      else
        superfp_kernel_impl<scalar_t, false>(p_a, p_o, size, man_bits_, exp_bits_, normal_binades_,
                       bias_, round_mode_, saturation_mode_); });
  }
  else
  {
    auto rand_ints = randint_like(a_c, INT_MAX, device(a_c.device()).dtype(kInt));
    const int *p_r = rand_ints.data_ptr<int>();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a_c.scalar_type(), "superfp_quantize_cpu_sr", [&]
                                    {
      const scalar_t *p_a = a_c.data_ptr<scalar_t>();
      scalar_t *p_o = o.data_ptr<scalar_t>();
      if (is_signed)
        superfp_kernel_sr_impl<scalar_t, true>(p_a, p_r, p_o, size, man_bits_, exp_bits_,
                       normal_binades_, bias_, prng_bits_, saturation_mode_);
      else
        superfp_kernel_sr_impl<scalar_t, false>(p_a, p_r, p_o, size, man_bits_, exp_bits_,
                       normal_binades_, bias_, prng_bits_, saturation_mode_); });
  }

  return o;
}
