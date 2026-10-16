// The MPS kernels behind mptorch::binaryK_quant and mptorch::superfp_quant:
// a new tensor of the input's shape and dtype, every element rounded by
// quantize.metal, in the same cast, the same rounding mode and, under
// RoundMode::SR, the same random word as the CPU kernel gives it
// (cpu/binaryK_kernel.cpp, cpu/superfp_kernel.cpp), so the result is the
// CPU's bit for bit.
//
// Each entry point observes what its CPU twin does, in the same order: the
// input is made contiguous, the output allocated, the seed drawn under SR
// (even for an empty tensor, which the CPU also draws for, so that the
// generator's later draws stay in step), and only then is the dtype checked.
//
// The in-place ops, binaryK_quant_ and superfp_quant_, run the same kernel
// with the tensor bound as both its input and its output. That is
// value-safe: mpt_quantize reads x[i] before it writes y[i] and touches no
// other element, and neither buffer is declared restrict. Like the CPU's,
// they refuse a strided tensor rather than make the copy the out-of-place op
// makes, which the kernel would write instead of the caller's tensor.

#include "../cpu/utils.h" // draw_cpu_seed
#include "../quant_ops.h"
#include "launch_params.h"
#include "metal_runtime.h"
#include <ATen/ops/empty_like.h>
#include <string>

using at::Tensor;

namespace
{

  // The CPU quantizers treat every integer that names no deterministic mode
  // as RO (the `default:` arm of their switch), and SR as SR; so does this,
  // so that an out-of-range round_mode rounds as it does on the CPU.
  int64_t cpu_round_mode(int64_t round_mode)
  {
    const auto sr = static_cast<int64_t>(RoundMode::SR);
    if (round_mode == sr || (round_mode >= 0 && round_mode <= static_cast<int64_t>(RoundMode::RZ)))
      return round_mode;
    return static_cast<int64_t>(RoundMode::RO);
  }

  // Launches quantize.metal over the contiguous `a` into `o`, which is `a`
  // itself for the in-place ops, with the instantiation's family and format
  // lines in `format`.
  void quantize_into(const char *op, const Tensor &a, const Tensor &o, int64_t round_mode,
                     bool is_signed, int64_t prng_bits, const std::string &format)
  {
    const int64_t rm = cpu_round_mode(round_mode);
    const bool sr = rm == static_cast<int64_t>(RoundMode::SR);
    const uint64_t seed = sr ? draw_cpu_seed() : 0;
    const char *storage = mptorch_mps::metal_storage_type(a.scalar_type(), op);
    const int64_t n = a.numel();
    if (n == 0)
      return;

    const std::string tail = std::string("using mpt_storage_t = ") + storage +
                             ";\n#define MPT_ROUND_MODE " + mptorch_mps::metal_round_mode(rm) +
                             "\n#define MPT_IS_SIGNED " + (is_signed ? "true" : "false") +
                             "\n#define MPT_PRNG_BITS " + std::to_string(static_cast<int>(prng_bits)) + "\n" +
                             format;
    const mptorch_mps::QuantLaunch q{static_cast<uint64_t>(n), seed};
    mptorch_mps::launch(mptorch_mps::KernelSource::Quantize, tail,
                        {a, o, mptorch_mps::KernelArg::params(q)},
                        static_cast<uint64_t>((n + 3) / 4));
  }

  Tensor quantize(const char *op, const Tensor &a, int64_t round_mode, bool is_signed,
                  int64_t prng_bits, const std::string &format)
  {
    Tensor a_c = a.contiguous();
    Tensor o = at::empty_like(a_c);
    quantize_into(op, a_c, o, round_mode, is_signed, prng_bits, format);
    return o;
  }

  // The format lines of a binaryK and a superfp instantiation.
  std::string binaryK_format(int64_t K, int64_t P, int64_t bias, bool is_signed,
                             int64_t saturation_mode, int64_t subnormals_mode)
  {
    // P3109's widths, narrowed to int and derived as the CPU kernel derives
    // them: P - 1 stored mantissa bits, and the exponent gets the rest after
    // the sign, if there is one.
    const int K_ = static_cast<int>(K), P_ = static_cast<int>(P);
    const int man_bits = P_ - 1;
    const int exp_bits = is_signed ? K_ - P_ : K_ - P_ + 1;
    const bool extended = static_cast<SubnormalsMode>(subnormals_mode) == SubnormalsMode::EXTENDED_NORMALS;
    return "#define MPT_SUPERFP 0\n#define MPT_SUBNORMALS SubnormalsMode(" +
           std::to_string(subnormals_mode) +
           ")\ninline BinaryKParams mpt_params()\n{\n    return make_binaryK_params<float>(" +
           std::to_string(man_bits) + ", " + std::to_string(exp_bits) + ", " +
           std::to_string(static_cast<int>(bias)) + ", " + (is_signed ? "true" : "false") +
           ", SaturationMode(" + std::to_string(saturation_mode) + "), " +
           (extended ? "true" : "false") + ");\n}\n";
  }

  std::string superfp_format(int64_t man_bits, int64_t exp_bits, int64_t normal_binades,
                             int64_t bias, int64_t saturation_mode)
  {
    return "#define MPT_SUPERFP 1\n#define MPT_SUBNORMALS SubnormalsMode::SUBNORMALS\n"
           "inline SuperfpParams mpt_params()\n{\n    return make_superfp_params<float>(" +
           std::to_string(static_cast<int>(man_bits)) + ", " + std::to_string(static_cast<int>(exp_bits)) +
           ", " + std::to_string(static_cast<int>(normal_binades)) + ", " +
           std::to_string(static_cast<int>(bias)) + ", SaturationMode(" +
           std::to_string(saturation_mode) + "));\n}\n";
  }

} // namespace

Tensor binaryK_quantize_mps(Tensor a, int64_t K, int64_t P, int64_t bias, int64_t prng_bits,
                            bool is_signed, int64_t round_mode, int64_t saturation_mode,
                            int64_t subnormals_mode)
{
  return quantize("binaryK_quantize_mps", a, round_mode, is_signed, prng_bits,
                  binaryK_format(K, P, bias, is_signed, saturation_mode, subnormals_mode));
}

Tensor superfp_quantize_mps(Tensor a, int64_t man_bits, int64_t exp_bits, int64_t normal_binades,
                            int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode,
                            int64_t saturation_mode)
{
  return quantize("superfp_quantize_mps", a, round_mode, is_signed, prng_bits,
                  superfp_format(man_bits, exp_bits, normal_binades, bias, saturation_mode));
}

// mptorch::binaryK_quant_ and mptorch::superfp_quant_ on an MPS tensor: the
// kernel above over `a` in place, after the CPU's contiguity check, worded
// as the CPU words it.
Tensor &binaryK_quantize_mps_(Tensor &a, int64_t K, int64_t P, int64_t bias, int64_t prng_bits,
                              bool is_signed, int64_t round_mode, int64_t saturation_mode,
                              int64_t subnormals_mode)
{
  TORCH_CHECK(a.is_contiguous(), "binaryK_quant_ writes its argument in place and needs a "
              "contiguous tensor, got strides ", a.strides(), " for sizes ", a.sizes(),
              ": use binaryK_quant, which copies a strided input");
  quantize_into("binaryK_quantize_mps_", a, a, round_mode, is_signed, prng_bits,
                binaryK_format(K, P, bias, is_signed, saturation_mode, subnormals_mode));
  return a;
}

Tensor &superfp_quantize_mps_(Tensor &a, int64_t man_bits, int64_t exp_bits, int64_t normal_binades,
                              int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode,
                              int64_t saturation_mode)
{
  TORCH_CHECK(a.is_contiguous(), "superfp_quant_ writes its argument in place and needs a "
              "contiguous tensor, got strides ", a.strides(), " for sizes ", a.sizes(),
              ": use superfp_quant, which copies a strided input");
  quantize_into("superfp_quantize_mps_", a, a, round_mode, is_signed, prng_bits,
                superfp_format(man_bits, exp_bits, normal_binades, bias, saturation_mode));
  return a;
}
