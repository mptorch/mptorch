#pragma once

// The CUDA kernels of the elementwise block ops, as block_entry.cpp launches
// them: raw pointers and plain structs (common/block_kernels.h), so that
// block_kernel.cu includes no ATen tensor header and compiles in a few
// seconds rather than paying ATen's front end (cuda/custom_matmul_kernel.cuh
// says what that costs).

#include "../common/block_kernels.h"
#include <ATen/cuda/PhiloxCudaState.h>
#include <cuda_runtime_api.h>

namespace mptorch::block_cuda
{
  // block_pack (j.quant false) and block_quant / block_quant_ (j.quant true).
  // `rng` is read only under RoundMode::SR.
  void launch_block_pack(const mptorch::block::BlockPackJob &j, const mptorch::block::BlockFormatParams &p,
                         const mptorch::block::BlockCast &c, at::PhiloxCudaState rng, cudaStream_t stream);

  void launch_block_unpack(const mptorch::block::BlockUnpackJob &j,
                           const mptorch::block::BlockFormatParams &p, cudaStream_t stream);
} // namespace mptorch::block_cuda
