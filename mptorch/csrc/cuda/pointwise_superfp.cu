// superfp_pointwise on the device (cuda/pointwise_kernel.cuh holds the
// kernels): the step, then superfp_quant's cast of the format.

#include "pointwise_kernel.cuh"

namespace mptorch::pointwise_cuda
{
  void launch_superfp(const Launch &l, const SuperfpFormat &f)
  {
    dispatch(l, [&](auto tag, auto mode)
             {
      using scalar_t = decltype(tag);
      constexpr RoundMode RM = decltype(mode)::value;
      run<scalar_t, RM>(l, mptorch::pointwise::make_superfp_round<carrier_t<scalar_t>, RM>(
                               f.man_bits, f.exp_bits, f.normal_binades, f.bias, f.is_signed,
                               f.spec, f.saturation)); });
  }
} // namespace mptorch::pointwise_cuda
