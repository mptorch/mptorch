// The out-of-line definitions of the op boundary's format checks, which
// common/format_host.h and common/stochastic_host.h declare. They are
// compiled here, once, rather than as inline functions in every translation
// unit that calls them, because the code of a check compiled into a unit
// moved the code GCC generated for the unrelated quantizer loops in the
// same unit (format_host.h's check_format says by how much).

#include "common/stochastic_host.h"

namespace mptorch
{
  void check_format(const Format &f, CarrierBits c, const char *op)
  {
    const bool magnitude = f.family == FormatFamily::FixedPoint;
    const int64_t room = significand_room(magnitude, c);
    const auto shown = [&f](int64_t v) { return Shown{v, f.narrowed}; };
    if (magnitude)
    {
      TORCH_CHECK(f.bits >= 1 && f.bits <= room, op, ": the format has ", shown(f.bits), " magnitude bits, but ",
                  c.name, " holds 1 to ", room);
      TORCH_CHECK(f.fl >= f.bits - EXP_LIMIT && f.fl <= EXP_LIMIT, op, ": the format's fl is ", shown(f.fl),
                  ", but with ", f.bits, " magnitude bits the kernels take ", f.bits - EXP_LIMIT, " to ",
                  EXP_LIMIT, ": they compute -fl and mag_bits - fl in an int, and need both between -2^30 and 2^30");
      return;
    }
    TORCH_CHECK(f.exp_bits >= 0 && f.exp_bits <= MAX_EXP_BITS, op, ": the format has ", shown(f.exp_bits),
                " exponent bits, but the kernels take 0 to ", MAX_EXP_BITS, ": they compute 1 << exp_bits in an int");
    TORCH_CHECK(f.bits >= 0 && f.bits <= room, op, ": the format has ", shown(f.bits), " stored mantissa bits, but ",
                c.name, " holds 0 to ", room);
    const int64_t top_field = (int64_t(1) << f.exp_bits) - 1;
    TORCH_CHECK(f.bias >= top_field - EXP_LIMIT && f.bias <= EXP_LIMIT + 1, op, ": the format's bias is ",
                shown(f.bias), ", but with ", f.exp_bits, " exponent bits the kernels take ", top_field - EXP_LIMIT,
                " to ", EXP_LIMIT + 1,
                ": they compute 1 - bias and 2^exp_bits - 1 - bias in an int, and need both between -2^30 and 2^30");
    if (f.family == FormatFamily::Superfp)
      TORCH_CHECK(f.normal_binades >= 1 && f.normal_binades <= top_field, op, ": the format has ",
                  shown(f.normal_binades), " normal binades, but a superfp format with ", f.exp_bits,
                  " exponent bits has 1 to ", top_field);
  }

  void sr_check_prng(int64_t prng_bits, int64_t bits, bool magnitude, CarrierBits c, const char *op,
                     bool narrowed)
  {
    const int64_t most = significand_room(magnitude, c) - bits;
    TORCH_CHECK(prng_bits >= 0 && prng_bits <= most, op, ": prng_bits must be in [0, ", most, "] for ",
                magnitude ? "" : "a ", bits, magnitude ? " magnitude bits" : "-bit mantissa", " rounded in ",
                c.name, ", got ", Shown{prng_bits, narrowed});
  }

  void check_quantizer_format(const Format &f, CarrierBits c, int64_t prng_bits, int64_t round_mode,
                              const char *op)
  {
    check_format(f, c, op);
    if (is_stochastic_mode(round_mode))
      sr_check_prng(prng_bits, f.bits, f.family == FormatFamily::FixedPoint, c, op);
  }
} // namespace mptorch
