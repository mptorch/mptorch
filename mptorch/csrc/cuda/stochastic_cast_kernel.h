#pragma once

// The device half of _stochastic_cast (cuda/stochastic_cast_entry.cpp holds
// the tensors): ATen-free, like every .cu here.

#include "../common/stochastic_cast.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace mptorch::stochastic_cast_cuda
{
  void launch(const mptorch::stochastic_cast::Job<float> &j, const float *x, const int64_t *draws, float *o,
              int64_t n, cudaStream_t stream);
  void launch(const mptorch::stochastic_cast::Job<double> &j, const double *x, const int64_t *draws,
              double *o, int64_t n, cudaStream_t stream);
} // namespace mptorch::stochastic_cast_cuda
