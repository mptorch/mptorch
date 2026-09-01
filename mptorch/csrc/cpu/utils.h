#pragma once

#include <ATen/Parallel.h>
#include <cstdint>

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
// count -- RoundMode::SR included, since its randomness is drawn into a
// tensor up front rather than per thread.
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

template <typename scalar_t, class Quant>
void quant_kernel(const scalar_t *a, const int *r, scalar_t *o, int64_t size, Quant quant)
{
  at::parallel_for(0, size, mptorch_cpu::quant_grain_size,
                   [=](int64_t begin, int64_t end)
                   {
                     for (int64_t i = begin; i < end; ++i)
                       o[i] = quant(a[i], (uint32_t)r[i]);
                   });
}
