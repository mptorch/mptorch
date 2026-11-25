#pragma once

// The CUDA kernels of block_project (common/block_project.h), as
// block_project_entry.cpp launches them: raw pointers and plain structs, so
// that block_project.cu includes no ATen tensor header (cuda/block_kernel.h
// says why).

#include "../common/block_kernels.h"
#include "../common/block_project.h"
#include <ATen/cuda/PhiloxCudaState.h>
#include <cuda_runtime_api.h>

namespace mptorch::block_cuda
{
  // `rng` is read only under RoundMode::SR.
  void launch_block_project(const mptorch::block::BlockProjectJob &j, const mptorch::block::BlockFormatParams &p,
                            const mptorch::block::BlockCast &c, at::PhiloxCudaState rng, cudaStream_t stream);

#if !defined(MPTORCH_NO_FP64)
  // The same in binary64, for a float64 tensor.
  void launch_block_project(const mptorch::block::BlockProjectJob &j,
                            const mptorch::block::BlockFormatParamsT<double> &p,
                            const mptorch::block::BlockCastT<double> &c, at::PhiloxCudaState rng,
                            cudaStream_t stream);
#endif
} // namespace mptorch::block_cuda
