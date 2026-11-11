#pragma once

// The device half of _carrier_odd (cuda/carrier_odd_entry.cpp holds the
// tensors): ATen-free, like every .cu here.

#include <cstdint>
#include <cuda_runtime.h>

namespace mptorch::carrier_odd_cuda
{
  void launch(const float *a, const float *b, const float *c, float *o, int64_t n, int64_t op,
              cudaStream_t stream);
  void launch(const double *a, const double *b, const double *c, double *o, int64_t n,
              int64_t op, cudaStream_t stream);
} // namespace mptorch::carrier_odd_cuda
