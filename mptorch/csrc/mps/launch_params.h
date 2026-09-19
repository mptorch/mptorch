#pragma once

// What an MPS kernel takes per launch besides its tensors: the shape, the
// strides and the stochastic-rounding seed, passed as one small `constant`
// buffer. The host (metal_runtime.mm) fills these and the kernels
// (gemm.metal, quantize.metal) read them, and both compile this one
// definition, so the two layouts cannot drift apart; the size checks pin
// them anyway. Everything in them varies per call, which is why they are a
// buffer and not part of a kernel's source (the formats are, so that their
// constants fold; see metal_runtime.mm).

#if !defined(__METAL_VERSION__)
#include <cstdint>
#endif

namespace mptorch_mps
{

  // A GEMM, as common/gemm_args.h's GemmShape describes it: C[b] =
  // op(A[b]) @ op(B[b]), written densely as [batch, M, N], with per-operand
  // batch strides in elements (0 broadcasts that operand). The idx_* strides
  // read a mixed op's precision index, and are 0 on the single-format ops.
  struct GemmLaunch
  {
    uint32_t M, K, N, batch;
    uint32_t trans_a, trans_b, use_rng, unused;
    uint64_t stride_a, stride_b;
    uint64_t seed;
    uint64_t idx_row_stride, idx_col_stride, idx_batch_stride;
  };
  static_assert(sizeof(GemmLaunch) == 80, "GemmLaunch must have one layout on the host and the GPU");

  // An elementwise quantizer over `n` contiguous elements.
  struct QuantLaunch
  {
    uint64_t n;
    uint64_t seed;
  };
  static_assert(sizeof(QuantLaunch) == 16, "QuantLaunch must have one layout on the host and the GPU");

} // namespace mptorch_mps
