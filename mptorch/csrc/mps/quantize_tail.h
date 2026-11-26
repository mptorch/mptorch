#pragma once

// The tail quantize.metal's kernels are instantiated with (metal_runtime.h's
// launch): the storage type, the round mode, the sign, the random bits and the
// family's format lines, which define quantize_one's cast. quantize.cpp's
// quantizers and pointwise_entry.cpp's fused ops write it, so a fused op's
// cast is the MPS quantizer's, and so the CPU's.

#include "../common/modes.h"
#include "metal_runtime.h"
#include <cstdint>
#include <string>

namespace mptorch_mps
{
  // The storage type, the round mode, the sign and the random bits, then the
  // family's format lines.
  inline std::string quantize_tail(const char *storage, int64_t rm, bool is_signed,
                                   int64_t prng_bits, const std::string &format)
  {
    return std::string("using mpt_storage_t = ") + storage + ";\n#define MPT_ROUND_MODE " +
           metal_round_mode(rm) + "\n#define MPT_IS_SIGNED " + (is_signed ? "true" : "false") +
           "\n#define MPT_PRNG_BITS " + std::to_string(static_cast<int>(prng_bits)) + "\n" + format;
  }

  // The format lines of a binaryK, a superfp and a fixed-point
  // instantiation; MPT_FORMATS selects the family in quantize.metal.
  inline std::string binaryK_format_lines(int64_t K, int64_t P, int64_t bias, bool is_signed,
                                          int64_t saturation_mode, int64_t subnormals_mode)
  {
    // P3109's widths, narrowed to int and converted as the CPU kernel
    // converts them: P - 1 stored mantissa bits, and the exponent field takes
    // the remaining bits after the sign bit, if there is one.
    const int K_ = static_cast<int>(K), P_ = static_cast<int>(P);
    const int man_bits = P_ - 1;
    const int exp_bits = is_signed ? K_ - P_ : K_ - P_ + 1;
    const bool extended =
        static_cast<SubnormalsMode>(subnormals_mode) == SubnormalsMode::EXTENDED_NORMALS;
    return "#define MPT_FORMATS 0\n#define MPT_SUBNORMALS SubnormalsMode(" +
           std::to_string(subnormals_mode) +
           ")\ninline BinaryKParams mpt_params()\n{\n    return make_binaryK_params<float>(" +
           std::to_string(man_bits) + ", " + std::to_string(exp_bits) + ", " +
           std::to_string(static_cast<int>(bias)) + ", " + (is_signed ? "true" : "false") +
           ", SaturationMode(" + std::to_string(saturation_mode) + "), " +
           (extended ? "true" : "false") + ");\n}\n";
  }

  inline std::string superfp_format_lines(int64_t man_bits, int64_t exp_bits,
                                          int64_t normal_binades, int64_t bias,
                                          int64_t saturation_mode)
  {
    return "#define MPT_FORMATS 1\n#define MPT_SUBNORMALS SubnormalsMode::SUBNORMALS\n"
           "inline SuperfpParams mpt_params()\n{\n    return make_superfp_params<float>(" +
           std::to_string(static_cast<int>(man_bits)) + ", " +
           std::to_string(static_cast<int>(exp_bits)) + ", " +
           std::to_string(static_cast<int>(normal_binades)) + ", " +
           std::to_string(static_cast<int>(bias)) + ", SaturationMode(" +
           std::to_string(saturation_mode) + "));\n}\n";
  }

  inline std::string fixedpoint_format_lines(int64_t wl, int64_t fl, bool is_signed, bool symmetric)
  {
    return "#define MPT_FORMATS 2\n#define MPT_SUBNORMALS SubnormalsMode::SUBNORMALS\n"
           "inline FixedPointParams mpt_params()\n{\n    return make_fixedpoint_params<float>(" +
           std::to_string(static_cast<int>(wl)) + ", " + std::to_string(static_cast<int>(fl)) +
           ", " + (is_signed ? "true" : "false") + ", " + (symmetric ? "true" : "false") +
           ");\n}\n";
  }
} // namespace mptorch_mps
