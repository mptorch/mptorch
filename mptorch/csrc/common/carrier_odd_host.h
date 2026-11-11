#pragma once

// The host side of the custom_matmul_*_odd ops (phase N of
// dev/continuation_plan.md, common/carrier_odd.h): one body per op, written
// once on the backend, which {cpu,cuda}/custom_matmul_odd_entry.cpp name.
// Each is its twin's entry point with the packed Args wrapped in OddArgs:
// the schema, the checks and the driver are the twin's, so a call differs
// from the twin's only in the kernel it runs. The driver refuses every
// algorithm but NAIVE (check_naive_only), as for the twins.

#include "carrier_odd.h"
#include "gemm_host.h"

namespace mptorch::gemm
{
  // A fused step that no format rounds after is the plain fused op's: the
  // odd step exists to be rounded again.
  inline void check_fma_quant(bool fma_quant, const char *op_name)
  {
    TORCH_CHECK(fma_quant, op_name,
                ": fma_quant=false leaves the step unrounded, and a carrier step rounded to "
                "odd is only correct when a format rounds it again; use the plain fused op");
  }

  template <class Backend>
  at::Tensor run_binaryK_split_odd(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                   int64_t mul_K, int64_t mul_P, int64_t mul_bias,
                                   bool mul_is_signed, bool accumulate_quant, int64_t acc_K,
                                   int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
                                   int64_t accumulate_algorithm, int64_t round_mode,
                                   int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                   int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                   int64_t mul_prng_bits, int64_t acc_prng_bits)
  {
    return run_custom_matmul<Backend>(
        "custom_matmul_binaryK_odd",
        OddArgs<BinaryKSplitArgs>{pack_binaryK_split(
            mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P, acc_bias,
            acc_is_signed, mul_saturation_mode, mul_subnormals_mode, acc_saturation_mode,
            acc_subnormals_mode, mul_prng_bits, acc_prng_bits)},
        a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
  }

  template <class Backend>
  at::Tensor run_superfp_split_odd(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                   int64_t mul_man_bits, int64_t mul_exp_bits,
                                   int64_t mul_normal_binades, int64_t mul_bias,
                                   bool mul_is_signed, bool accumulate_quant,
                                   int64_t acc_man_bits, int64_t acc_exp_bits,
                                   int64_t acc_normal_binades, int64_t acc_bias,
                                   bool acc_is_signed, int64_t accumulate_algorithm,
                                   int64_t round_mode, int64_t mul_saturation_mode,
                                   int64_t acc_saturation_mode, int64_t mul_prng_bits,
                                   int64_t acc_prng_bits)
  {
    return run_custom_matmul<Backend>(
        "custom_matmul_superfp_odd",
        OddArgs<SuperfpSplitArgs>{pack_superfp_split(
            mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
            accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades, acc_bias,
            acc_is_signed, mul_saturation_mode, acc_saturation_mode, mul_prng_bits,
            acc_prng_bits)},
        a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
  }

  template <class Backend>
  at::Tensor run_binaryK_fused_odd(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                   bool fma_quant, int64_t fma_K, int64_t fma_P,
                                   int64_t fma_bias, bool fma_is_signed,
                                   int64_t accumulate_algorithm, int64_t round_mode,
                                   int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                   int64_t fma_prng_bits)
  {
    constexpr const char *op = "custom_matmul_binaryK_fma_odd";
    check_fma_quant(fma_quant, op);
    return run_custom_matmul<Backend>(
        op,
        OddArgs<BinaryKFusedArgs>{pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias,
                                                     fma_is_signed, fma_saturation_mode,
                                                     fma_subnormals_mode, fma_prng_bits)},
        a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
  }

  template <class Backend>
  at::Tensor run_superfp_fused_odd(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                   bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                   int64_t fma_normal_binades, int64_t fma_bias,
                                   bool fma_is_signed, int64_t accumulate_algorithm,
                                   int64_t round_mode, int64_t fma_saturation_mode,
                                   int64_t fma_prng_bits)
  {
    constexpr const char *op = "custom_matmul_superfp_fma_odd";
    check_fma_quant(fma_quant, op);
    return run_custom_matmul<Backend>(
        op,
        OddArgs<SuperfpFusedArgs>{pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits,
                                                     fma_normal_binades, fma_bias,
                                                     fma_is_signed, fma_saturation_mode,
                                                     fma_prng_bits)},
        a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
  }

  // _carrier_odd's checks: three tensors of one float32 or float64 dtype,
  // device and shape (c is ignored by add and mul, but must still match).
  inline void check_carrier_odd(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c,
                                int64_t op)
  {
    TORCH_CHECK(op >= 0 && op <= 2, "_carrier_odd: op is 0 (fma), 1 (add) or 2 (mul), got ",
                op);
    TORCH_CHECK(a.scalar_type() == at::kFloat || a.scalar_type() == at::kDouble,
                "_carrier_odd: float32 or float64 tensors, got ", a.scalar_type());
    for (const at::Tensor *t : {&b, &c})
    {
      TORCH_CHECK(t->scalar_type() == a.scalar_type() && t->device() == a.device() &&
                      t->sizes() == a.sizes(),
                  "_carrier_odd: a, b and c must share one dtype, device and shape");
    }
  }
} // namespace mptorch::gemm
