// Every binary32 word through the branchless binaryK casts of
// common/cast_binaryK_word.h (RNA, RU, RD, RZ, RO), against the shipped casts
// of common/cast_binaryK.h, word for word.
//
// The word forms copy each shipped cast's arms as straight-line code and
// select one, so that the CPU's vector tiles can run them; this holds them
// to the shipped bodies on every input of every format below (NaNs with
// their payloads, infinities, both zeros, carrier subnormals, and every
// saturation mode, P3109's two domains included). `control` compares each
// word form against another mode's shipped cast (RU against RD, RD against
// RU, RZ against RU, RNA against RZ, RO against RZ), which must mismatch, so a
// sweep that compared nothing could not pass. Built with the extension's host
// flags:
//
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc dev/benchmarks/sweeps/cpu_word_cast_sweep.cpp \
//       -o /tmp/cpu_word_cast_sweep
//   /tmp/cpu_word_cast_sweep            # the sweep
//   /tmp/cpu_word_cast_sweep control    # the negative control
//
// About 2 min on 16 threads (13 formats x 5 modes x 2^32 words).

#include "common/cast_binaryK_word.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace
{
  struct Format
  {
    int man_bits, exp_bits, bias;
    SaturationMode sat;
    const char *name;
  };

  float bits_float(uint32_t w)
  {
    float f;
    std::memcpy(&f, &w, 4);
    return f;
  }
  uint32_t float_bits(float f)
  {
    uint32_t w;
    std::memcpy(&w, &f, 4);
    return w;
  }

  constexpr SubnormalsMode SUB = SubnormalsMode::SUBNORMALS;

  float shipped(RoundMode mode, float x, const BinaryKParams &p)
  {
    switch (mode)
    {
    case RoundMode::RNA:
      return cast_binaryK_nearest_away(x, true, SUB, p);
    case RoundMode::RU:
      return cast_binaryK_up(x, true, SUB, p);
    case RoundMode::RD:
      return cast_binaryK_down(x, true, SUB, p);
    case RoundMode::RZ:
      return cast_binaryK_zero(x, true, SUB, p);
    default:
      return cast_binaryK_odd(x, true, SUB, p);
    }
  }

  // The mode each word form is compared against: itself, or under `control`
  // another mode, which must disagree somewhere.
  RoundMode against(RoundMode mode, bool control)
  {
    if (!control)
      return mode;
    switch (mode)
    {
    case RoundMode::RU:
      return RoundMode::RD;
    case RoundMode::RD:
      return RoundMode::RU;
    case RoundMode::RZ:
      return RoundMode::RU;
    default:
      return RoundMode::RZ;
    }
  }

  template <RoundMode RM>
  uint64_t sweep(const BinaryKParams &p, RoundMode want, const char *label)
  {
    uint64_t bad = 0;
    uint32_t first = 0;
    // checked before the critical section, so that only the first mismatches
    // enter it: under `control` nearly every word mismatches
    std::atomic<bool> found{false};
#pragma omp parallel for reduction(+ : bad) schedule(dynamic, 64)
    for (int64_t hi = 0; hi < (int64_t(1) << 16); ++hi)
    {
      for (uint32_t lo = 0; lo < (1u << 16); ++lo)
      {
        const uint32_t w = (uint32_t(hi) << 16) | lo;
        const float x = bits_float(w);
        const uint32_t got = float_bits(cast_binaryK_word<RM>(x, p));
        const uint32_t ref = float_bits(shipped(want, x, p));
        if (got != ref)
        {
          ++bad;
          if (!found.load(std::memory_order_relaxed))
          {
#pragma omp critical
            if (!found.load(std::memory_order_relaxed))
            {
              first = w;
              found.store(true, std::memory_order_relaxed);
            }
          }
        }
      }
    }
    std::printf("  %-4s %12llu mismatches", label, (unsigned long long)bad);
    if (bad)
    {
      const float x = bits_float(first);
      std::printf("  (e.g. input 0x%08x: word form 0x%08x, shipped 0x%08x)", first,
                  float_bits(cast_binaryK_word<RM>(x, p)), float_bits(shipped(want, x, p)));
    }
    std::printf("\n");
    return bad;
  }
} // namespace

int main(int argc, char **argv)
{
  const bool control = argc > 1 && std::strcmp(argv[1], "control") == 0;
  const Format formats[] = {
      {3, 4, 7, SaturationMode::SAT_NONE, "e4m3-like, SAT_NONE"},
      {3, 4, 8, SaturationMode::SAT_FINITE, "binary8p4, SAT_FINITE"},
      {10, 5, 15, SaturationMode::SAT_NONE, "binary16"},
      {10, 5, 16, SaturationMode::SAT_FINITE_EXTENDED, "binary16p11, SAT_FINITE_EXTENDED"},
      {7, 7, 63, SaturationMode::SAT_NONE, "P = 8, 7 exponent bits"},
      {1, 5, 15, SaturationMode::SAT_NONE, "P = 2"},
      {2, 3, 4, SaturationMode::SAT_NONE, "binary6p3"},
      {22, 7, 63, SaturationMode::SAT_FINITE, "P = 23, 7 exponent bits"},
      {5, 6, 31, SaturationMode::SAT_NONE, "P = 6, 6 exponent bits"},
      {2, 5, 16, SaturationMode::SAT_NONE, "binary8p3 (e5m2-like)"},
      {7, 8, 127, SaturationMode::SAT_NONE,
       "bf16-like, bias 127 (its bottom below binary32's normals)"},
      {7, 8, 128, SaturationMode::SAT_PROPAGATE, "binary16p8, SAT_PROPAGATE"},
      {2, 4, 8, SaturationMode::SAT_FINITE, "P = 3, 4 exponent bits, SAT_FINITE"},
  };
  std::printf(control ? "negative control: each word form against another mode's cast\n"
                      : "word forms against the shipped casts, all 2^32 binary32 inputs\n");
  uint64_t total = 0, failed_controls = 0;
  for (const Format &f : formats)
  {
    const BinaryKParams p =
        make_binaryK_params<float>(f.man_bits, f.exp_bits, f.bias, true, f.sat, false);
    if (!word_cast_format(true, SUB, p))
    {
      std::printf("%s: not a word-cast format, skipped\n", f.name);
      continue;
    }
    std::printf("%s\n", f.name);
    const uint64_t counts[] = {
        sweep<RoundMode::RNA>(p, against(RoundMode::RNA, control), "RNA"),
        sweep<RoundMode::RU>(p, against(RoundMode::RU, control), "RU"),
        sweep<RoundMode::RD>(p, against(RoundMode::RD, control), "RD"),
        sweep<RoundMode::RZ>(p, against(RoundMode::RZ, control), "RZ"),
        sweep<RoundMode::RO>(p, against(RoundMode::RO, control), "RO"),
    };
    for (uint64_t c : counts)
    {
      total += c;
      failed_controls += (c == 0);
    }
  }
  if (control)
  {
    std::printf("%s: %llu of the comparisons found no mismatch\n",
                failed_controls ? "CONTROL FAILED" : "control ok",
                (unsigned long long)failed_controls);
    return failed_controls ? 1 : 0;
  }
  std::printf("%s: %llu mismatches in all\n", total ? "FAILED" : "ok", (unsigned long long)total);
  return total ? 1 : 0;
}
