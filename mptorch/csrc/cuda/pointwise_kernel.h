#pragma once

// The device half of the fused pointwise and table ops
// (cuda/pointwise_entry.cpp holds the tensors): one launch per call, from raw
// pointers, the geometry and the format. Like the GEMM, block and narrowing kernels, the .cu files include
// no ATen tensor header.

#include "../common/gemm_dtype.h"
#include "../common/modes.h"
#include "../common/pointwise.h"
#include <ATen/cuda/PhiloxCudaState.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mptorch::pointwise_cuda
{
  // One call: the operands (b null for a number operand), the result, the
  // geometry, the op, whether the step rounds to odd, the number operand
  // already in the carrier (a binary32 value held exactly in a double),
  // the canonical round mode (stochastic_host.h), the Philox state under SR,
  // and the stream. A table op (`pwl`) has a step of its own instead of op
  // and odd: its entries in the carrier, the range, the tails and the
  // reduction (common/pointwise.h's PwlStep).
  struct Launch
  {
    const void *a = nullptr;
    const void *b = nullptr;
    void *o = nullptr;
    GemmDtype dt = GemmDtype::Float;
    mptorch::pointwise::PointwiseGeom g;
    int32_t op = 0;
    bool odd = false;
    double y = 0.0;
    RoundMode rm = RoundMode::RNE;
    at::PhiloxCudaState rng{};
    cudaStream_t stream = nullptr;
    bool pwl = false;
    const void *entries = nullptr;
    int32_t segments = 0;
    double lo = 0.0, hi = 0.0;
    int32_t tails = 0;
    bool rsqrt = false;
  };

  // Each family's format, in its quantizer's terms; `spec` is prng_bits with
  // the SR variant.
  struct BinaryKFormat
  {
    int K, P, bias, spec;
    bool is_signed;
    SaturationMode saturation;
    SubnormalsMode subnormals;
  };
  struct SuperfpFormat
  {
    int man_bits, exp_bits, normal_binades, bias, spec;
    bool is_signed;
    SaturationMode saturation;
  };
  struct FixedPointFormat
  {
    int wl, fl, spec;
    bool is_signed, symmetric;
  };

  // Defined in cuda/pointwise_{binaryK,superfp,fixedpoint}.cu.
  void launch_binaryK(const Launch &l, const BinaryKFormat &f);
  void launch_superfp(const Launch &l, const SuperfpFormat &f);
  void launch_fixedpoint(const Launch &l, const FixedPointFormat &f);
} // namespace mptorch::pointwise_cuda
