// The CUDA entry points of the two fixed-point conv ops, each one pass of a
// convolution (forward, input gradient or weight gradient) on the
// fixed-point GEMM's arithmetic (common/fixedpoint_gemm.h), as the other
// conv ops run their GEMM twins' (common/gemm_conv_host.h).
// cuda/cuda_ops.cpp registers them, through MPT_CUDA_IMPL.
//
// An entry point packs its schema arguments and names the backend, and the
// driver (common/fixedpoint_conv_host.h's run_custom_conv_fixedpoint) does
// the rest. The bodies are the same text as their CPU, MPS and Meta copies,
// which tests/test_entry_copies.py checks. The kernels are instantiated in
// cuda/inst/custom_conv_fixedpoint*.

#include "../common/fixedpoint_conv_host.h"
#include "../quant_ops.h"
#include "gemm_backend.h"

using at::Tensor;
using namespace mptorch::gemm;

namespace
{
  using Backend = mptorch::gemm_cuda::CudaBackend;
}

// The split mac's pass: p = Q_M(RN_c(a * b)), s = Q_A(s + p), over the
// gathered operands. `integer` runs the integer accumulator where the
// carrier path would give the same bits, a hook for the tests.
Tensor fixedpoint_conv_cuda(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
                            c10::IntArrayRef stride, c10::IntArrayRef padding,
                            c10::IntArrayRef dilation, int64_t groups, int64_t mul_wl,
                            int64_t mul_fl, bool mul_is_signed, bool mul_symmetric, int64_t acc_wl,
                            int64_t acc_fl, bool acc_is_signed, bool acc_symmetric,
                            int64_t round_mode, int64_t mul_prng_bits, int64_t acc_prng_bits,
                            bool wrap, bool integer)
{
  return run_custom_conv_fixedpoint<Backend>(
      "custom_conv_fixedpoint",
      pack_fixedpoint_split(mul_wl, mul_fl, mul_is_signed, mul_symmetric, acc_wl, acc_fl,
                            acc_is_signed, acc_symmetric, mul_prng_bits, acc_prng_bits, wrap,
                            integer),
      a, b, conv_pass, out_size, stride, padding, dilation, groups, round_mode);
}

// The fused mac's pass: s = Q_F(RN_c(a * b) + s), one rounding a step.
Tensor fixedpoint_conv_fma_cuda(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
                                c10::IntArrayRef stride, c10::IntArrayRef padding,
                                c10::IntArrayRef dilation, int64_t groups, int64_t fma_wl,
                                int64_t fma_fl, bool fma_is_signed, bool fma_symmetric,
                                int64_t round_mode, int64_t fma_prng_bits, bool wrap)
{
  return run_custom_conv_fixedpoint<Backend>(
      "custom_conv_fixedpoint_fma",
      pack_fixedpoint_fused(fma_wl, fma_fl, fma_is_signed, fma_symmetric, fma_prng_bits, wrap), a,
      b, conv_pass, out_size, stride, padding, dilation, groups, round_mode);
}
