#pragma once

#include "../common/philox.h"
#include <ATen/core/Tensor.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <cstdint>
#include <mutex>
#include <type_traits>

// Elementwise quantization drivers shared by the CPU quantizer kernels: the
// parallel loop that applies a cast to every element, its stochastic-rounding
// twin that also hands each element a random word, and the seed draw.
//
// `Quant` is always a concrete functor type (a lambda), never a
// std::function: the cast has to inline into the loop body for the compiler
// to schedule its bit-twiddling across iterations, and an indirect call
// through std::function blocks that as well as costing a heap allocation per
// invocation. Callers therefore pass the lambda straight in rather than
// storing it first, one closure type per round mode.
//
// at::parallel_for rather than a raw `#pragma omp parallel for`: it runs on
// ATen's own thread pool, so torch.set_num_threads applies, and it degrades
// to serial inside an enclosing parallel region instead of nesting. It does
// still need setup.py's -fopenmp under the AT_PARALLEL_OPENMP backend, since
// the pragma comes from a header template compiled into this translation
// unit; without the flag the loop is silently serial. Chunks are disjoint
// and each output element depends only on the input element at the same
// index, so results are independent of the thread count, RoundMode::SR
// included, since quant_kernel_sr keys each element's random draw on that
// same global index rather than on a per-thread stream.
namespace mptorch_cpu
{
  // The parallel_for grain: a sixteenth of ATen's own default
  // (at::internal::GRAIN_SIZE, 32768). That default is calibrated for a
  // few-instruction elementwise op, whereas one format cast is tens of
  // instructions, so 2048 elements is still far above the thread pool's
  // dispatch cost, and it lets a layer-sized tensor be split across threads
  // instead of landing on one. Spelled out rather than divided down so this
  // header need not pull in ATen/TensorIterator.h just for the constant.
  constexpr int64_t quant_grain_size = 2048;
}

// o[i] = quant(a[i]) for i in [0, size), in parallel chunks of at least
// quant_grain_size elements. Both pointers must address contiguous storage.
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

// The RoundMode::SR driver: o[i] = quant(a[i], r_i), where r_i is a random
// word generated in place from `seed` and the element's index. The
// alternative, a tensor of draws made with randint_like before the call,
// costs an allocation and a full extra read/write pass the size of the
// input, and because ATen's CPU generator is serial that fill does not scale
// with torch.set_num_threads while the quantization around it does: at
// eight threads the draw was the majority of the call.
//
// Element `i` takes 32-bit word `i & 3` of Philox block `i >> 2`, so one
// 10-round generate is amortized over four elements and each element's word
// is a function of its own index alone, which is what keeps the result
// independent of the thread count and of where at::parallel_for cuts the
// chunks. The block is regenerated at the start of every chunk and at every
// fourth element after that. `seed` comes from draw_cpu_seed() below, once
// per call.
//
// A float64 element rounds in binary64 and draws a 64-bit word: the pair of
// words starting at `2 * (i & 1)` of block `i >> 1`, the layout PhiloxBlock
// documents and the CUDA kernel uses, so `quant` takes a uint64_t there and
// a uint32_t otherwise.
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
// governs a RoundMode::SR call the way it governs torch's own random ops.
// Also the seed of the CPU GEMM backend (gemm_backend.h), whose
// per-output-element Philox streams are keyed on it and the element's index
// (NaiveTile::seed_rng in gemm_policy.h). Only called when RoundMode::SR is
// selected, so a deterministic call leaves the generator untouched. The
// generator's mutex is held for the draw, as ATen's own callers do.
inline uint64_t draw_cpu_seed()
{
  auto gen = at::get_generator_or_default<at::CPUGeneratorImpl>(
      c10::nullopt, at::detail::getDefaultCPUGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  return gen->random64();
}
