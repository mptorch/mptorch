// fixedpoint_pointwise on the device (cuda/pointwise_kernel.cuh holds the
// kernels): the step, then fixedpoint_quant's cast of the format.

#include "pointwise_kernel.cuh"

namespace mptorch::pointwise_cuda
{
  void launch_fixedpoint(const Launch &l, const FixedPointFormat &f)
  {
    dispatch(l, [&](auto tag, auto mode)
             {
      using scalar_t = decltype(tag);
      constexpr RoundMode RM = decltype(mode)::value;
      run<scalar_t, RM>(l, mptorch::pointwise::make_fixedpoint_round<carrier_t<scalar_t>, RM>(
                               f.wl, f.fl, f.is_signed, f.symmetric, f.spec)); });
  }
} // namespace mptorch::pointwise_cuda
