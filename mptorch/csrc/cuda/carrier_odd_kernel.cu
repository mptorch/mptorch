#include "../common/carrier_odd.h"
#include "carrier_odd_kernel.h"

// _carrier_odd on the device: common/carrier_odd.h's directed-pair
// primitives applied elementwise, one element a thread. A test op.

namespace mptorch::carrier_odd_cuda
{
  namespace
  {
    template <class T>
    __global__ void carrier_odd_kernel(const T *a, const T *b, const T *c, T *o, int64_t n,
                                       int64_t op)
    {
      namespace odd = mptorch::gemm::odd;
      const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n)
        return;
      o[i] = op == 0 ? odd::fma(a[i], b[i], c[i]) : op == 1 ? odd::add(a[i], b[i])
                                                            : odd::mul(a[i], b[i]);
    }

    template <class T>
    void launch_as(const T *a, const T *b, const T *c, T *o, int64_t n, int64_t op,
                   cudaStream_t stream)
    {
      constexpr int block = 256;
      const int64_t grid = (n + block - 1) / block;
      carrier_odd_kernel<T><<<(unsigned)grid, block, 0, stream>>>(a, b, c, o, n, op);
    }
  } // namespace

  void launch(const float *a, const float *b, const float *c, float *o, int64_t n, int64_t op,
              cudaStream_t stream)
  {
    launch_as(a, b, c, o, n, op, stream);
  }
  void launch(const double *a, const double *b, const double *c, double *o, int64_t n,
              int64_t op, cudaStream_t stream)
  {
    launch_as(a, b, c, o, n, op, stream);
  }
} // namespace mptorch::carrier_odd_cuda
