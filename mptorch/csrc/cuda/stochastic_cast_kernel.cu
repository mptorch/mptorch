// The CUDA kernel of _stochastic_cast (common/stochastic_cast.h): one thread
// per element, its random word given. Test-only.

#include "stochastic_cast_kernel.h"

namespace mptorch::stochastic_cast_cuda
{
  namespace
  {
    template <class T, class W>
    __global__ void stochastic_cast_kernel(const mptorch::stochastic_cast::Job<T> j, const T *x,
                                           const int64_t *draws, T *o, int64_t n)
    {
      const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n)
        return;
      o[i] = mptorch::stochastic_cast::cast(j, x[i], static_cast<W>(static_cast<uint64_t>(draws[i])));
    }

    template <class T, class W>
    void launch_as(const mptorch::stochastic_cast::Job<T> &j, const T *x, const int64_t *draws, T *o, int64_t n,
                   cudaStream_t stream)
    {
      constexpr int block = 256;
      const int64_t grid = (n + block - 1) / block;
      stochastic_cast_kernel<T, W><<<(unsigned)grid, block, 0, stream>>>(j, x, draws, o, n);
    }
  } // namespace

  void launch(const mptorch::stochastic_cast::Job<float> &j, const float *x, const int64_t *draws, float *o,
              int64_t n, cudaStream_t stream)
  {
    launch_as<float, uint32_t>(j, x, draws, o, n, stream);
  }
  void launch(const mptorch::stochastic_cast::Job<double> &j, const double *x, const int64_t *draws,
              double *o, int64_t n, cudaStream_t stream)
  {
    launch_as<double, uint64_t>(j, x, draws, o, n, stream);
  }
} // namespace mptorch::stochastic_cast_cuda
