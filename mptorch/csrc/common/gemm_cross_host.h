#pragma once

// The host side of the six cross-family ops (gemm_cross.h): the packers that
// turn their schemas into a CrossSplitArgs. Otherwise the ops go through the
// shared drivers: run_custom_matmul for the NAIVE pair,
// run_custom_matmul_accumulated for the *_accumulated twins and
// run_custom_conv_single for the conv pair.
//
// Only these files include this header: the cross-family entry files,
// {cpu,cuda,mps}/custom_matmul_cross_entry.cpp; carrier_odd_gemm_host.h, for
// the odd cross-family ops; and meta_host.h. So the entry points of every
// other op compile exactly as they would without it.
//
// The schemas are the single-family split op's, with the accumulate format
// given in its own family and without accumulate_quant, since the sum is
// always rounded (gemm_cross.h says why):
//
//   custom_matmul_binaryK_superfp(a, b, trans_a, trans_b,
//       mul_K, mul_P, mul_bias, mul_is_signed,
//       acc_man_bits, acc_exp_bits, acc_normal_binades, acc_bias, acc_is_signed,
//       accumulate_algorithm, round_mode,
//       mul_saturation_mode, mul_subnormals_mode, acc_saturation_mode,
//       mul_prng_bits, acc_prng_bits)
//   custom_matmul_superfp_binaryK(a, b, trans_a, trans_b,
//       mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
//       acc_K, acc_P, acc_bias, acc_is_signed,
//       accumulate_algorithm, round_mode,
//       mul_saturation_mode, acc_saturation_mode, acc_subnormals_mode,
//       mul_prng_bits, acc_prng_bits)
//
// An *_accumulated twin appends a block size and an outer format in the
// accumulate format's family, as the single-family twins do; a conv op takes
// the *_accumulated schema with the geometry in place of the transpose
// flags.

#include "gemm_conv_host.h"
#include "gemm_cross.h"
#include "gemm_host.h"

namespace mptorch::gemm
{

  inline BinaryKSuperfpArgs pack_binaryK_superfp(
      int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits,
      int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
      int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
      int64_t mul_prng_bits, int64_t acc_prng_bits)
  {
    BinaryKSuperfpArgs args;
    args.mul = binaryK_widths(mul_K, mul_P, mul_bias, mul_is_signed);
    args.mul_c = BinaryKCommon{mul_is_signed, static_cast<SaturationMode>(mul_saturation_mode),
                               static_cast<SubnormalsMode>(mul_subnormals_mode),
                               narrow_to_int(mul_prng_bits)};
    args.acc = superfp_widths(acc_man_bits, acc_exp_bits, acc_normal_binades, acc_bias);
    args.acc_c = SuperfpCommon{acc_is_signed, static_cast<SaturationMode>(acc_saturation_mode),
                               narrow_to_int(acc_prng_bits)};
    return args;
  }

  inline SuperfpBinaryKArgs pack_superfp_binaryK(
      int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias,
      bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
      int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
      int64_t mul_prng_bits, int64_t acc_prng_bits)
  {
    SuperfpBinaryKArgs args;
    args.mul = superfp_widths(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias);
    args.mul_c = SuperfpCommon{mul_is_signed, static_cast<SaturationMode>(mul_saturation_mode),
                               narrow_to_int(mul_prng_bits)};
    args.acc = binaryK_widths(acc_K, acc_P, acc_bias, acc_is_signed);
    args.acc_c = BinaryKCommon{acc_is_signed, static_cast<SaturationMode>(acc_saturation_mode),
                               static_cast<SubnormalsMode>(acc_subnormals_mode),
                               narrow_to_int(acc_prng_bits)};
    return args;
  }

} // namespace mptorch::gemm
