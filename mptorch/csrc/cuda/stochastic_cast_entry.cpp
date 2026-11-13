// The CUDA entry point of _stochastic_cast (common/stochastic_cast_host.h);
// the kernel is cuda/stochastic_cast_kernel.cu. Test-only.

#include "../common/stochastic_cast_host.h"
#include "../quant_ops.h"
#include "stochastic_cast_kernel.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty_like.h>

using at::Tensor;

Tensor stochastic_cast_cuda(Tensor x, Tensor draws, int64_t family, c10::IntArrayRef format,
                            int64_t stochastic_mode)
{
  namespace sc = mptorch::stochastic_cast;
  sc::check_tensors(x, draws);
  Tensor x_c = x.contiguous(), d_c = draws.contiguous();
  Tensor o = at::empty_like(x_c);
  const int64_t n = o.numel();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == at::kFloat)
  {
    const auto j = sc::make_job<float>(family, format, stochastic_mode);
    if (n)
      mptorch::stochastic_cast_cuda::launch(j, x_c.data_ptr<float>(), d_c.data_ptr<int64_t>(),
                                            o.data_ptr<float>(), n, stream);
  }
  else
  {
    const auto j = sc::make_job<double>(family, format, stochastic_mode);
    if (n)
      mptorch::stochastic_cast_cuda::launch(j, x_c.data_ptr<double>(), d_c.data_ptr<int64_t>(),
                                            o.data_ptr<double>(), n, stream);
  }
  return o;
}
