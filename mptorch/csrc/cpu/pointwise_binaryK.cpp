// binaryK_pointwise and binaryK_pwl on the CPU (cpu/pointwise_kernel.h holds
// the loops): the step, then binaryK_quant's cast of the format, and for the
// arithmetic ops under RNE, where the cast is cast_binaryK_rne_fast, the
// vector blocks of cpu/pointwise_rne.h.

#include "../common/format_host.h"
#include "../quant_ops.h"
#include "pointwise_kernel.h"
#include "pointwise_rne.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

using at::Tensor;

namespace
{
  namespace pw = mptorch_cpu::pointwise;
  using mptorch::pointwise::ArithStep;

  // Whether a store to `o` can stall a later load of `in` on 4K aliasing:
  // the CPU's first test of a load against earlier stores compares the low
  // 12 address bits, so a result that lies a little ahead of an operand,
  // modulo 4096, stalls the next vector loads. A float32 add of 2048 x 2048
  // whose result lay 64 bytes past its operand took 3.4 ms with AVX-512
  // against 1.6 ms elsewhere (one thread).
  inline bool aliases(const void *o, const void *in)
  {
    return in && ((reinterpret_cast<uintptr_t>(o) - reinterpret_cast<uintptr_t>(in)) & 4095) < 512;
  }

  // The flat, nearest-stepped RNE call of a format whose cast is
  // cast_binaryK_rne_fast (cpu/pointwise_rne.h), in blocks: the operands
  // converted to binary32 (read in place from a float32 tensor), the vector
  // block, and the scalar loop's statement for a block that left a NaN. A
  // float32 result is written in place, or through the stack buffer and
  // copied when it aliases an operand (`aliases`).
  template <typename scalar_t>
  void run_rne_blocks(const pw::Call<scalar_t, ArithStep> &call,
                      const mptorch::pointwise::BinaryKRound<float, RoundMode::RNE> &round)
  {
    at::parallel_for(0, call.g.numel, mptorch_cpu::quant_grain_size, [&](int64_t begin, int64_t end)
                     {
      const pw::Call<scalar_t, ArithStep> c = call;
      const auto r = round;
      const bool through = aliases(c.o, c.a) || aliases(c.o, c.b);
      constexpr int64_t B = 1024;
      alignas(64) float xb[B], yb[B], ob[B];
      for (int64_t i = begin; i < end; i += B)
      {
        const int64_t n = std::min(B, end - i);
        const float *x, *y = nullptr;
        float *o;
        if constexpr (std::is_same_v<scalar_t, float>)
        {
          x = c.a + i;
          y = c.g.has_b ? c.b + i : nullptr;
          o = through ? ob : c.o + i;
        }
        else
        {
          for (int64_t k = 0; k < n; ++k)
            xb[k] = static_cast<float>(c.a[i + k]);
          if (c.g.has_b)
            for (int64_t k = 0; k < n; ++k)
              yb[k] = static_cast<float>(c.b[i + k]);
          x = xb;
          y = c.g.has_b ? yb : nullptr;
          o = ob;
        }
        if (!pw::pointwise_rne.run(c.step.op, x, y, c.y, o, n, r.p))
          for (int64_t k = 0; k < n; ++k)
            o[k] = r(c.step(x[k], y ? y[k] : c.y), true);
        if constexpr (std::is_same_v<scalar_t, float>)
        {
          if (through)
            std::memcpy(c.o + i, ob, static_cast<size_t>(n) * sizeof(float));
        }
        else
          for (int64_t k = 0; k < n; ++k)
            c.o[i + k] = static_cast<scalar_t>(ob[k]);
      } });
  }

  // The format's cast in the call's round mode, around either step.
  struct Launch
  {
    int K, P, bias, spec;
    bool is_signed;
    SaturationMode saturation;
    SubnormalsMode subnormals;
    RoundMode rm;

    template <class C>
    void operator()(const C &c, uint64_t seed) const
    {
      using scalar_t = typename C::scalar_type;
      using T = carrier_t<scalar_t>;
      if constexpr (std::is_same_v<T, float> && std::is_same_v<typename C::step_type, ArithStep>)
      {
        if (rm == RoundMode::RNE && !c.step.odd && c.g.flat && is_signed &&
            subnormals == SubnormalsMode::SUBNORMALS && pw::pointwise_rne.run)
        {
          const auto r = mptorch::pointwise::make_binaryK_round<float, RoundMode::RNE>(
              K, P, bias, true, spec, saturation, subnormals);
          if (r.p.fast_rne)
            return run_rne_blocks(c, r);
        }
      }
      auto make = [&](auto m)
      {
        return mptorch::pointwise::make_binaryK_round<T, decltype(m)::value>(
            K, P, bias, is_signed, spec, saturation, subnormals);
      };
      if (is_signed)
        pw::run_mode<true>(c, rm, seed, make);
      else
        pw::run_mode<false>(c, rm, seed, make);
    }
  };

  // The quantizer's format checks, then the canonical round mode.
  Launch checked(const char *name, const Tensor &x, int64_t K, int64_t P, int64_t bias,
                 int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode,
                 int64_t subnormals_mode)
  {
    mptorch::check_quantizer_format(
        mptorch::binaryK_format(K, P, bias, is_signed, saturation_mode, subnormals_mode),
        mptorch::carrier_bits(x.scalar_type() == at::kDouble), prng_bits, round_mode, name);
    const RoundMode rm = pw::canonical(round_mode, prng_bits, name);
    return Launch{static_cast<int>(K),         static_cast<int>(P),
                  static_cast<int>(bias),      static_cast<int>(prng_bits),
                  is_signed,                   static_cast<SaturationMode>(saturation_mode),
                  static_cast<SubnormalsMode>(subnormals_mode), rm};
  }
} // namespace

Tensor binaryK_pointwise_cpu(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                             int64_t op, int64_t carrier_rounding, int64_t K, int64_t P,
                             int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode,
                             int64_t saturation_mode, int64_t subnormals_mode)
{
  constexpr const char *name = "binaryK_pointwise";
  const Launch l = checked(name, a, K, P, bias, prng_bits, is_signed, round_mode, saturation_mode,
                           subnormals_mode);
  return pw::run_pointwise(name, a, b, scalar, op, carrier_rounding, l.rm, l);
}

Tensor binaryK_pwl_cpu(Tensor x, Tensor entries, double lo, double hi, int64_t tails, bool rsqrt,
                       int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed,
                       int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
  constexpr const char *name = "binaryK_pwl";
  const Launch l = checked(name, x, K, P, bias, prng_bits, is_signed, round_mode, saturation_mode,
                           subnormals_mode);
  return pw::run_pwl(name, x, entries, lo, hi, tails, rsqrt, l.rm, l);
}
