// binaryK_pointwise on the device (cuda/pointwise_kernel.cuh holds the
// kernels): the step, then binaryK_quant's cast of the format.

#include "pointwise_kernel.cuh"

namespace mptorch::pointwise_cuda
{
  void launch_binaryK(const Launch &l, const BinaryKFormat &f)
  {
    dispatch(l, [&](auto tag, auto mode)
             {
      using scalar_t = decltype(tag);
      constexpr RoundMode RM = decltype(mode)::value;
      run<scalar_t, RM>(l, mptorch::pointwise::make_binaryK_round<carrier_t<scalar_t>, RM>(
                               f.K, f.P, f.bias, f.is_signed, f.spec, f.saturation, f.subnormals)); });
  }
} // namespace mptorch::pointwise_cuda
