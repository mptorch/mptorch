// The two CPU GEMM entry points for superfp formats with a split mac.
//
// Each is the two things that genuinely differ between the eight ops --
// packing the flat schema into an Args (common/gemm_args.h) and naming the
// backend -- wrapped around the one driver in common/gemm_host.h. The bodies
// are the same text as their CUDA twins in cuda/custom_matmul_entry.cpp,
// which is what finding H1 was for.
//
// Still one translation unit per (format family x mac mode): unlike the CUDA
// side, whose .cu files hold the kernels and whose entry points all fit in
// one .cpp, a CPU entry point instantiates the kernel it calls, so the B2
// split is what keeps these four objects compiling in parallel rather than
// one of them being the pole.

#include "custom_matmul_kernel.h"
#include "../common/gemm_host.h"
#include "../quant_ops.h"

using at::Tensor;
using namespace mptorch::gemm;

namespace
{
  using Backend = mptorch::gemm_cpu::CpuBackend;
}

Tensor superfp_matmul_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b,
                           int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                           int64_t mul_bias, bool mul_is_signed,
                           bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                           int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                           int64_t accumulate_algorithm, int64_t round_mode,
                           int64_t saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_superfp",
      pack_superfp_split(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                         accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades,
                         acc_bias, acc_is_signed, saturation_mode, mul_prng_bits, acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// superfp analogue of binaryK_matmul_mixed_cpu -- see its comment.
Tensor superfp_matmul_mixed_cpu(Tensor a, Tensor b, Tensor prec_idx,
                                 bool trans_a, bool trans_b,
                                 c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits,
                                 c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias,
                                 bool mul_is_signed,
                                 bool accumulate_quant, c10::IntArrayRef acc_man_bits,
                                 c10::IntArrayRef acc_exp_bits,
                                 c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias,
                                 bool acc_is_signed, int64_t accumulate_algorithm,
                                 int64_t round_mode, int64_t saturation_mode,
                                 int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  constexpr const char *op = "custom_matmul_superfp_mixed";
  return run_custom_matmul_mixed<Backend>(
      op,
      [&] {
        return pack_superfp_split_mixed(op, mul_man_bits, mul_exp_bits, mul_normal_binades,
                                        mul_bias, mul_is_signed, accumulate_quant, acc_man_bits,
                                        acc_exp_bits, acc_normal_binades, acc_bias, acc_is_signed,
                                        saturation_mode, mul_prng_bits, acc_prng_bits);
      },
      a, b, prec_idx, trans_a, trans_b, accumulate_algorithm, round_mode);
}
