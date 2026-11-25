#pragma once

// The drivers of the CUDA scaled quantizers (binaryK_scaled_kernel.cu,
// superfp_scaled_kernel.cu, fixedpoint_scaled_kernel.cu): utils.cuh's
// quant_kernel_all and quant_kernel_all_sr, with each element multiplied by
// its scale before the cast and divided by it after, in the same pass
// (common/scaled_quant_host.h states the operation and the layout). They
// load and store exactly as those kernels do, 16 bytes at a time, and under
// stochastic rounding give element i the random word those kernels give it,
// so a scaled call is bit for bit the elementwise quantizer of the scaled
// tensor.
//
// The multiply and the divide are spelled __fmul_rn / __fdiv_rn (__dmul_rn /
// __ddiv_rn in binary64): under nvcc's default --fmad=true a plain `x * s`
// may be contracted with the cast's first addition into an FMA, which would
// round once where torch's elementwise mul and the quantizer's cast round
// twice, and the _rn intrinsics are never contracted.

#include "../common/bit_helper.h"
#include "../common/scaled_quant_host.h"
#include "utils.cuh"
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mptorch_cuda_scaled
{

  __device__ __forceinline__ float mul_rn(float x, float s) { return __fmul_rn(x, s); }
  __device__ __forceinline__ double mul_rn(double x, double s) { return __dmul_rn(x, s); }
  __device__ __forceinline__ float div_rn(float x, float s) { return __fdiv_rn(x, s); }
  __device__ __forceinline__ double div_rn(double x, double s) { return __ddiv_rn(x, s); }

  // The scale layout as the kernels index it. A row's position fits in 32
  // bits (the launcher checks), so the per-lane division is a 32-bit one;
  // only the row of a vector's first element takes a 64-bit division.
  struct ScaleShape
  {
    int64_t cols;
    uint32_t cols32;
    uint32_t block;
    int64_t per_row;
  };

  // The row and column of one element, from which the scales of the
  // elements after it in a vector follow.
  struct ScaleCursor
  {
    int64_t row;
    uint32_t col;

    __device__ __forceinline__ ScaleCursor(int64_t r, uint32_t c) : row(r), col(c) {}

    __device__ __forceinline__ ScaleCursor(int64_t e, const ScaleShape &l)
        : row(e / l.cols), col(static_cast<uint32_t>(e - (e / l.cols) * l.cols)) {}

    // The index of the scale of the element `lane` places after this one.
    __device__ __forceinline__ int64_t index(int lane, const ScaleShape &l) const
    {
      uint32_t c = col + static_cast<uint32_t>(lane);
      int64_t r = row;
      while (c >= l.cols32)
      {
        c -= l.cols32;
        ++r;
      }
      return r * l.per_row + c / l.block;
    }
  };

  // o[i] = quant.scaled(a[i], s) for every element, where s is element i's
  // scale: the tensor's one scale when Single, and otherwise its row's and
  // block's. `quant.scaled(x, s)` takes and returns carrier values.
  template <typename scalar_t, bool Single, class Quant>
  __global__ __launch_bounds__(256, 2) void scaled_quant_kernel_all(
      const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s, ScaleShape l,
      Quant quant)
  {
    using T = carrier_t<scalar_t>;
    constexpr int vec_elems = SIMDTraits<scalar_t>::vec_elems;
    const int64_t vec_size = size / vec_elems;
    const int64_t rem_size = size % vec_elems;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const T s0 = s[0];

    for (int64_t i = idx; i < vec_size; i += stride)
    {
      int4 in_i4 = reinterpret_cast<const int4 *>(a)[i];
      float4 in = reinterpret_cast<const float4 &>(in_i4);
      float4 out;
      if constexpr (Single)
      {
        out = SIMDTraits<scalar_t>::process(in, [&](T val, int /*lane*/)
                                            { return quant.scaled(val, s0); });
      }
      else
      {
        const ScaleCursor c(i * vec_elems, l);
        out = SIMDTraits<scalar_t>::process(in, [&](T val, int lane)
                                            { return quant.scaled(val, s[c.index(lane, l)]); });
      }
      reinterpret_cast<int4 *>(o)[i] = reinterpret_cast<const int4 &>(out);
    }

    if (idx < rem_size)
    {
      const int64_t e = vec_size * vec_elems + idx;
      T sv = s0;
      if constexpr (!Single)
        sv = s[ScaleCursor(e, l).index(0, l)];
      o[e] = static_cast<scalar_t>(quant.scaled(static_cast<T>(a[e]), sv));
    }
  }

  // The RoundMode::SR twin, with quant_kernel_all_sr's random words:
  // element j takes word j & 3 of Philox block j >> 2 of the launch's (seed,
  // offset), and a float64 element words 2 * (j & 1) and the next of block
  // j >> 1. `quant.scaled_sr(x, s, word)` rounds stochastically.
  template <typename scalar_t, bool Single, class Quant>
  __global__ __launch_bounds__(256, 2) void scaled_quant_kernel_all_sr(
      const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s, ScaleShape l,
      Quant quant, at::PhiloxCudaState rng_args)
  {
    using T = carrier_t<scalar_t>;
    constexpr int vec_elems = SIMDTraits<scalar_t>::vec_elems;
    const int64_t vec_size = size / vec_elems;
    const int64_t rem_size = size % vec_elems;
    auto seeds = at::cuda::philox::unpack(rng_args);
    const uint64_t seed = std::get<0>(seeds);
    const uint64_t offset = std::get<1>(seeds);
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const T s0 = s[0];

    for (int64_t i = idx; i < vec_size; i += stride)
    {
      int4 in_i4 = reinterpret_cast<const int4 *>(a)[i];
      float4 in = reinterpret_cast<const float4 &>(in_i4);
      const int64_t base = i * vec_elems;
      ScaleCursor c(int64_t(0), 0u);
      if constexpr (!Single)
        c = ScaleCursor(base, l);
      auto scale_of = [&](int lane) -> T
      {
        if constexpr (Single)
          return s0;
        else
          return s[c.index(lane, l)];
      };
      float4 out;
      if constexpr (std::is_same_v<scalar_t, double>)
      {
        PhiloxBlock b0 = philox_block(seed, (uint64_t)i, offset);
        out = SIMDTraits<scalar_t>::process(in, [&](double val, int lane)
                                            { return quant.scaled_sr(val, scale_of(lane), b0.word64(2 * lane)); });
      }
      else
      {
        const uint64_t blk = (uint64_t)(base >> 2);
        PhiloxBlock b0 = philox_block(seed, blk, offset);
        PhiloxBlock b1;
        if constexpr (vec_elems > 4)
          b1 = philox_block(seed, blk + 1, offset);
        out = SIMDTraits<scalar_t>::process(in, [&](float val, int lane)
                                            {
          const int w = (int)((base + lane) & 3);
          uint32_t rv;
          if constexpr (vec_elems > 4)
            rv = (lane < 4) ? b0.word(w) : b1.word(w);
          else
            rv = b0.word(w);
          return quant.scaled_sr(val, scale_of(lane), rv); });
      }
      reinterpret_cast<int4 *>(o)[i] = reinterpret_cast<const int4 &>(out);
    }

    if (idx < rem_size)
    {
      const int64_t e = vec_size * vec_elems + idx;
      T sv = s0;
      if constexpr (!Single)
        sv = s[ScaleCursor(e, l).index(0, l)];
      if constexpr (std::is_same_v<scalar_t, double>)
      {
        PhiloxBlock b = philox_block(seed, (uint64_t)(e >> 1), offset);
        o[e] = static_cast<scalar_t>(
            quant.scaled_sr(static_cast<T>(a[e]), sv, b.word64(2 * (int)(e & 1))));
      }
      else
      {
        PhiloxBlock b = philox_block(seed, (uint64_t)(e >> 2), offset);
        o[e] = static_cast<scalar_t>(quant.scaled_sr(static_cast<T>(a[e]), sv, b.word((int)(e & 3))));
      }
    }
  }

  // The kernels' view of a layout. A tensor with one scale (one row, one
  // block) takes the Single kernels, which index nothing; any other layout
  // needs rows of fewer than 2^31 elements, so that a position in a row is a
  // 32-bit integer.
  inline ScaleShape scale_shape(const mptorch::ScaleLayout &l, int64_t size, bool &single,
                                const char *op)
  {
    single = l.cols == size && l.per_row == 1;
    TORCH_CHECK(single || l.cols <= std::numeric_limits<int32_t>::max(), op,
                ": a row of the tensor has ", l.cols, " elements, and the CUDA kernel takes rows "
                "of at most 2^31 - 1 elements when the row has several scales or the tensor "
                "several rows");
    return ScaleShape{l.cols, static_cast<uint32_t>(single ? 1 : l.cols),
                      static_cast<uint32_t>(single ? 1 : l.block), l.per_row};
  }

  // Launch the deterministic kernel for `quant` over a contiguous, 16-byte
  // aligned `a`, on the current stream.
  template <typename scalar_t, class Quant>
  void launch_scaled(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                     const mptorch::ScaleLayout &layout, Quant quant, const char *op)
  {
    constexpr int BLOCK_SIZE = 256;
    constexpr int64_t vec_elems = SIMDTraits<scalar_t>::vec_elems;
    bool single;
    const ScaleShape l = scale_shape(layout, size, single, op);
    const int grid = grid_for(std::max(size / vec_elems, size % vec_elems), BLOCK_SIZE);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (single)
      scaled_quant_kernel_all<scalar_t, true><<<grid, BLOCK_SIZE, 0, stream>>>(a, o, size, s, l, quant);
    else
      scaled_quant_kernel_all<scalar_t, false><<<grid, BLOCK_SIZE, 0, stream>>>(a, o, size, s, l, quant);
  }

  // Launch the RoundMode::SR kernel, with a (seed, offset) pair from ATen's
  // generator, which it advances as an elementwise quantizer's call does.
  template <typename scalar_t, class Quant>
  void launch_scaled_sr(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                        const mptorch::ScaleLayout &layout, Quant quant, const char *op)
  {
    constexpr int BLOCK_SIZE = 256;
    constexpr int64_t vec_elems = SIMDTraits<scalar_t>::vec_elems;
    bool single;
    const ScaleShape l = scale_shape(layout, size, single, op);
    const int grid = grid_for(std::max(size / vec_elems, size % vec_elems), BLOCK_SIZE);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    at::PhiloxCudaState rng_args = quant_rng_engine_inputs();
    if (single)
      scaled_quant_kernel_all_sr<scalar_t, true>
          <<<grid, BLOCK_SIZE, 0, stream>>>(a, o, size, s, l, quant, rng_args);
    else
      scaled_quant_kernel_all_sr<scalar_t, false>
          <<<grid, BLOCK_SIZE, 0, stream>>>(a, o, size, s, l, quant, rng_args);
  }

} // namespace mptorch_cuda_scaled
