// The CPU entry points of the two fixed-point GEMM ops (common/fixedpoint_gemm.h):
// a split mac, whose products are rounded to the multiply format and whose
// running sums to the accumulate format, and a fused one, whose steps are
// rounded once. Every sum is exact; the product is the carrier's. Each
// rounding saturates or, with `wrap`, wraps the accumulate format around.
// cpu/cpu_ops.cpp registers them.
//
// As in cpu/custom_matmul_entry.cpp, an entry point packs its schema
// arguments and names the backend, and the driver
// (common/fixedpoint_gemm_host.h's run_custom_matmul_fixedpoint) does the
// rest, including the choice between the carrier path and the integer
// accumulator. The bodies are the same text as their CUDA, MPS and Meta
// copies, which tests/test_entry_copies.py checks. The kernels are
// instantiated in cpu/inst/custom_matmul_fixedpoint*.

#include "../common/fixedpoint_gemm_host.h"
#include "../quant_ops.h"
#include "gemm_backend.h"

using at::Tensor;
using namespace mptorch::gemm;

namespace
{
  using Backend = mptorch::gemm_cpu::CpuBackend;
}

// The split mac: p = Q_M(RN_c(a * b)), s = Q_A(s + p). `integer` runs the
// integer accumulator where the carrier path would give the same bits, a
// hook for the tests that hold the two paths to each other.
Tensor fixedpoint_matmul_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t mul_wl,
                             int64_t mul_fl, bool mul_is_signed, bool mul_symmetric, int64_t acc_wl,
                             int64_t acc_fl, bool acc_is_signed, bool acc_symmetric,
                             int64_t round_mode, int64_t mul_prng_bits, int64_t acc_prng_bits,
                             bool wrap, bool integer)
{
  return run_custom_matmul_fixedpoint<Backend>(
      "custom_matmul_fixedpoint",
      pack_fixedpoint_split(mul_wl, mul_fl, mul_is_signed, mul_symmetric, acc_wl, acc_fl,
                            acc_is_signed, acc_symmetric, mul_prng_bits, acc_prng_bits, wrap,
                            integer),
      a, b, trans_a, trans_b, round_mode);
}

// The fused mac: s = Q_F(RN_c(a * b) + s), one rounding a step.
Tensor fixedpoint_matmul_fma_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t fma_wl,
                                 int64_t fma_fl, bool fma_is_signed, bool fma_symmetric,
                                 int64_t round_mode, int64_t fma_prng_bits, bool wrap)
{
  return run_custom_matmul_fixedpoint<Backend>(
      "custom_matmul_fixedpoint_fma",
      pack_fixedpoint_fused(fma_wl, fma_fl, fma_is_signed, fma_symmetric, fma_prng_bits, wrap), a,
      b, trans_a, trans_b, round_mode);
}
