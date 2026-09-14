#pragma once

#include "../common/philox.h"
#include <ATen/core/Tensor.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <cstdint>
#include <mutex>
#include <type_traits>

// Elementwise quantization drivers.
//
// `Quant` is always a concrete functor type (a lambda), never a
// std::function: the cast has to inline into the loop body for the compiler
// to schedule its bit-twiddling across iterations, and an indirect call
// through std::function blocks that as well as costing a heap allocation per
// invocation. Callers therefore pass the lambda straight in rather than
// storing it first -- see binaryK_kernel.cpp's round-mode switch.
//
// at::parallel_for rather than a raw `#pragma omp parallel for`: it runs on
// ATen's own thread pool, so torch.set_num_threads applies, and it degrades
// to serial inside an enclosing parallel region instead of nesting. It does
// still need setup.py's -fopenmp under the AT_PARALLEL_OPENMP backend, since
// the pragma comes from a header template inlined into this translation
// unit. Chunks are disjoint and each output element depends only on the
// input element at the same index, so results are independent of the thread
// count -- RoundMode::SR included, since quant_kernel_sr keys each element's
// random draw on that same global index rather than on a per-thread stream.
namespace mptorch_cpu
{
  // A sixteenth of ATen's own default (at::internal::GRAIN_SIZE, 32768).
  // That default is calibrated for a few-instruction elementwise op, whereas
  // one format cast is tens of instructions, so 2048 elements is still far
  // above the thread pool's dispatch cost -- and it lets a layer-sized
  // tensor be split across threads instead of landing on one. Spelled out
  // rather than divided down, so this header needn't pull in
  // ATen/TensorIterator.h just for the constant.
  constexpr int64_t quant_grain_size = 2048;
}

template <typename scalar_t, class Quant>
void quant_kernel(const scalar_t *a, scalar_t *o, int64_t size, Quant quant)
{
  at::parallel_for(0, size, mptorch_cpu::quant_grain_size,
                   [=](int64_t begin, int64_t end)
                   {
                     for (int64_t i = begin; i < end; ++i)
                       o[i] = quant(a[i]);
                   });
}

// RoundMode::SR. This used to take a `const int *r` -- an int32 tensor of
// draws that binaryK_quantize_cpu / superfp_quantize_cpu materialized with
// randint_like before every call. Two things were wrong with that. It cost an
// allocation and a full extra read/write pass the size of the input, and
// ATen's CPU generator is serial, so the fill did not scale with
// torch.set_num_threads while the quantization around it did: at eight
// threads the draw was the majority of the call. See finding E1 in
// dev/gemm_perf_audit.md.
//
// Element `i` takes word `i & 3` of Philox block `i >> 2`, so one 10-round
// generate is amortized over four elements and each element's value is a
// function of its own index -- which is what keeps the result independent of
// the thread count and of where at::parallel_for happens to cut the chunks.
// `seed` comes from draw_cpu_seed() below, once per call.
//
// A float64 element rounds in binary64 and draws a 64-bit word: words
// `2 * (i & 1)` and the next of block `i >> 1`, the layout PhiloxBlock
// documents and the CUDA kernel uses, so `quant` takes a uint64_t there.
template <typename scalar_t, class Quant>
void quant_kernel_sr(const scalar_t *a, scalar_t *o, int64_t size, uint64_t seed, Quant quant)
{
  if constexpr (std::is_same_v<scalar_t, double>)
  {
    at::parallel_for(0, size, mptorch_cpu::quant_grain_size,
                     [=](int64_t begin, int64_t end)
                     {
                       PhiloxBlock blk;
                       for (int64_t i = begin; i < end; ++i)
                       {
                         if (i == begin || (i & 1) == 0)
                           blk = philox_block(seed, (uint64_t)(i >> 1), 0);
                         o[i] = quant(a[i], blk.word64(2 * (int)(i & 1)));
                       }
                     });
  }
  else
  {
  at::parallel_for(0, size, mptorch_cpu::quant_grain_size,
                   [=](int64_t begin, int64_t end)
                   {
                     PhiloxBlock blk;
                     for (int64_t i = begin; i < end; ++i)
                     {
                       if (i == begin || (i & 3) == 0)
                         blk = philox_block(seed, (uint64_t)(i >> 2), 0);
                       o[i] = quant(a[i], blk.word((int)(i & 3)));
                     }
                   });
  }
}

// One 64-bit seed from ATen's default CPU generator, so torch.manual_seed
// still governs a RoundMode::SR call exactly as it did when the draws came
// from randint_like. Shared with the custom_matmul_*.cpp GEMM entry points,
// which seed their per-output-element PhiloxEngine streams from it (see
// NaiveAccumulator::seed_rng in gemm_policy.h). Only called when
// RoundMode::SR is selected.
inline uint64_t draw_cpu_seed()
{
  auto gen = at::get_generator_or_default<at::CPUGeneratorImpl>(
      c10::nullopt, at::detail::getDefaultCPUGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  return gen->random64();
}
