#pragma once

// The CUDA kernels of the fused pointwise and table ops (common/pointwise.h
// states them), included by the three family units. A grid-stride loop over the
// result's elements, each read through the geometry (pointwise::offsets,
// with FastDivmod, when the geometry is not flat) and written densely.
//
// Under RoundMode::SR a thread takes a group of the elements that share a
// Philox block, four in binary32 and two in binary64, so that each block is
// generated once: element j takes word j & 3 of block j >> 2 (in binary64
// the 64-bit word 2 * (j & 1) of block j >> 1) of the launch's (seed,
// offset) stream, as cuda/utils.cuh's quant_kernel_all_sr gives it, so the
// result is that of the elementwise quantizer on the step's result.

#include "../common/philox.h"
#include "pointwise_kernel.h"
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <type_traits>

namespace mptorch::pointwise_cuda
{
  namespace
  {
    using mptorch::pointwise::PointwiseGeom;

    constexpr int BLOCK = 256;

    inline unsigned grid_for(int64_t work)
    {
      return static_cast<unsigned>(
          std::max<int64_t>(1, std::min<int64_t>((work + BLOCK - 1) / BLOCK, 2147483647)));
    }

    template <typename scalar_t>
    __device__ __forceinline__ void operands(const PointwiseGeom &g, const scalar_t *a,
                                             const scalar_t *b, carrier_t<scalar_t> y, int64_t j,
                                             carrier_t<scalar_t> &x_out, carrier_t<scalar_t> &y_out)
    {
      using T = carrier_t<scalar_t>;
      int64_t oa = j, ob = j;
      if (!g.flat)
        mptorch::pointwise::offsets(g, static_cast<int32_t>(j), oa, ob);
      x_out = static_cast<T>(a[oa]);
      y_out = g.has_b ? static_cast<T>(b[ob]) : y;
    }

    template <typename scalar_t, class Step, class Round>
    __global__ __launch_bounds__(BLOCK) void pointwise_kernel(const scalar_t *a, const scalar_t *b,
                                                              scalar_t *o, PointwiseGeom g,
                                                              carrier_t<scalar_t> y, Step step,
                                                              Round round)
    {
      using T = carrier_t<scalar_t>;
      const int64_t stride = (int64_t)gridDim.x * blockDim.x;
      for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < g.numel; j += stride)
      {
        T x, w;
        operands(g, a, b, y, j, x, w);
        o[j] = static_cast<scalar_t>(round(step(x, w), round.is_signed));
      }
    }

    template <typename scalar_t, class Step, class Round>
    __global__ __launch_bounds__(BLOCK) void pointwise_kernel_sr(
        const scalar_t *a, const scalar_t *b, scalar_t *o, PointwiseGeom g, carrier_t<scalar_t> y,
        Step step, Round round, at::PhiloxCudaState rng)
    {
      using T = carrier_t<scalar_t>;
      constexpr int G = std::is_same_v<scalar_t, double> ? 2 : 4;
      const auto seeds = at::cuda::philox::unpack(rng);
      const uint64_t seed = std::get<0>(seeds), offset = std::get<1>(seeds);
      const int64_t groups = (g.numel + G - 1) / G;
      const int64_t stride = (int64_t)gridDim.x * blockDim.x;
      for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; q < groups; q += stride)
      {
        const PhiloxBlock blk = philox_block(seed, (uint64_t)q, offset);
#pragma unroll
        for (int k = 0; k < G; ++k)
        {
          const int64_t j = q * G + k;
          if (j >= g.numel)
            break;
          T x, w;
          operands(g, a, b, y, j, x, w);
          const T v = step(x, w);
          if constexpr (G == 2)
            o[j] = static_cast<scalar_t>(round(v, blk.word64(2 * k), round.is_signed));
          else
            o[j] = static_cast<scalar_t>(round(v, blk.word(k), round.is_signed));
        }
      }
    }

    template <typename scalar_t, RoundMode RM, class Step, class Round>
    void launch_step(const Launch &l, const Step &step, const Round &round)
    {
      using T = carrier_t<scalar_t>;
      const auto *a = static_cast<const scalar_t *>(l.a);
      const auto *b = static_cast<const scalar_t *>(l.b);
      auto *o = static_cast<scalar_t *>(l.o);
      const T y = static_cast<T>(l.y);
      if constexpr (RM == RoundMode::SR)
      {
        constexpr int G = std::is_same_v<scalar_t, double> ? 2 : 4;
        pointwise_kernel_sr<scalar_t><<<grid_for((l.g.numel + G - 1) / G), BLOCK, 0, l.stream>>>(
            a, b, o, l.g, y, step, round, l.rng);
      }
      else
        pointwise_kernel<scalar_t><<<grid_for(l.g.numel), BLOCK, 0, l.stream>>>(a, b, o, l.g, y,
                                                                                step, round);
    }

    // The call's step: the arithmetic op's, or the table's, its constants in
    // the carrier (the scale computed in binary64 first, as torch rounds a
    // number).
    template <typename scalar_t, RoundMode RM, class Round>
    void run(const Launch &l, const Round &round)
    {
      using T = carrier_t<scalar_t>;
      if (l.pwl)
      {
        const mptorch::pointwise::PwlStep<T> step{
            static_cast<const T *>(l.entries), l.segments, static_cast<T>(l.lo),
            static_cast<T>(l.hi), static_cast<T>(static_cast<double>(l.segments) / (l.hi - l.lo)),
            l.tails, l.rsqrt};
        launch_step<scalar_t, RM>(l, step, round);
      }
      else
        launch_step<scalar_t, RM>(l, mptorch::pointwise::ArithStep{l.op, l.odd}, round);
    }

    template <RoundMode RM>
    using Mode = std::integral_constant<RoundMode, RM>;

    // f(scalar_t{}, Mode<RM>{}) for the call's dtype and canonical round mode.
    template <class F>
    void dispatch(const Launch &l, F &&f)
    {
      auto with_mode = [&](auto tag)
      {
        switch (l.rm)
        {
        case RoundMode::RNE:
          return f(tag, Mode<RoundMode::RNE>{});
        case RoundMode::RNA:
          return f(tag, Mode<RoundMode::RNA>{});
        case RoundMode::RU:
          return f(tag, Mode<RoundMode::RU>{});
        case RoundMode::RD:
          return f(tag, Mode<RoundMode::RD>{});
        case RoundMode::RZ:
          return f(tag, Mode<RoundMode::RZ>{});
        case RoundMode::RO:
          return f(tag, Mode<RoundMode::RO>{});
        default:
          return f(tag, Mode<RoundMode::SR>{});
        }
      };
      switch (l.dt)
      {
      case GemmDtype::Double:
        return with_mode(double{});
      case GemmDtype::Half:
        return with_mode(c10::Half{});
      case GemmDtype::BFloat16:
        return with_mode(c10::BFloat16{});
      default:
        return with_mode(float{});
      }
    }
  } // namespace
} // namespace mptorch::pointwise_cuda
