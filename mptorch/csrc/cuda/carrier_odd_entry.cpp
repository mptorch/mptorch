// The CUDA entry point of _carrier_odd (cuda/carrier_odd_kernel.cu runs it):
// the tensors and the stream. A .cpp inside csrc/cuda/, so a CPU-only build
// skips it.

#include "../common/carrier_odd_host.h"
#include "../quant_ops.h"
#include "carrier_odd_kernel.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty_like.h>

using at::Tensor;

Tensor carrier_odd_cuda(Tensor a, Tensor b, Tensor c, int64_t op)
{
  mptorch::gemm::check_carrier_odd(a, b, c, op);
  Tensor a_c = a.contiguous(), b_c = b.contiguous(), c_c = c.contiguous();
  Tensor o = at::empty_like(a_c);
  const int64_t n = o.numel();
  if (n == 0)
    return o;
  TORCH_CHECK(n < (int64_t(1) << 31) * 256, "_carrier_odd: too many elements for one grid");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (a.scalar_type() == at::kFloat)
    mptorch::carrier_odd_cuda::launch(a_c.data_ptr<float>(), b_c.data_ptr<float>(),
                                      c_c.data_ptr<float>(), o.data_ptr<float>(), n, op, stream);
  else
    mptorch::carrier_odd_cuda::launch(a_c.data_ptr<double>(), b_c.data_ptr<double>(),
                                      c_c.data_ptr<double>(), o.data_ptr<double>(), n, op,
                                      stream);
  return o;
}
