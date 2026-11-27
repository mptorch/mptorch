// Every binary32 word through the elementwise quantizer's vector loops
// (cpu/binaryK_quant_vector.cpp, compiled in), against the scalar loop's casts
// (cast_binaryK_nearest_even, _nearest_away, _up, _down, _zero and _odd for a
// signed format under SUBNORMALS), word for word.
//
// The loops are the ones the extension runs, chosen as it chooses them, so
// this checks the vectorized code itself: the RNE loop's lanes over the
// shipped cast_binaryK_rne_fast, and the other modes' over
// common/cast_binaryK_word.h (whose casts cpu_word_cast_sweep.cpp holds to the
// shipped ones). The words go through in blocks of 2^16, alternately out of
// place and in place (the in-place op passes one buffer as both), each block
// starting at a different offset from a vector boundary. `control` compares
// each loop against another mode's scalar cast (RNE and RNA against RZ, RU
// against RD, RD against RU, RZ against RU, RO against RZ), which must
// mismatch, so a sweep that compared nothing could not pass. Built with the
// extension's host flags:
//
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc dev/benchmarks/sweeps/cpu_quant_vector_sweep.cpp \
//       mptorch/csrc/cpu/binaryK_quant_vector.cpp -o /tmp/cpu_quant_vector_sweep
//   /tmp/cpu_quant_vector_sweep                        # the widest loops the CPU has
//   MPTORCH_CPU_SIMD=avx2 /tmp/cpu_quant_vector_sweep  # the AVX2 loops on an AVX-512 CPU
//   /tmp/cpu_quant_vector_sweep control                # the negative control
//
// About 3 min on 16 threads (13 formats x 6 modes x 2^32 words).

#include "cpu/binaryK_quant_vector.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using mptorch_cpu::vector_quant_format;
using mptorch_cpu::vector_quants;
using mptorch_cpu::VectorQuantFn;

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

  // The scalar loop's cast (cpu/binaryK_kernel.cpp) for a signed format.
  float scalar(RoundMode mode, float x, const BinaryKParams &p)
  {
    switch (mode)
    {
    case RoundMode::RNE:
      return cast_binaryK_nearest_even(x, true, SUB, p);
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

  // The mode each loop is compared against: its own, or under `control`
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
    case RoundMode::RZ:
      return RoundMode::RU;
    default:
      return RoundMode::RZ;
    }
  }

  constexpr int64_t BLOCK = int64_t(1) << 16;
  constexpr int64_t PAD = 16; // the largest start offset, in floats

  uint64_t sweep(VectorQuantFn f, RoundMode want, const BinaryKParams &p, const char *label)
  {
    uint64_t bad = 0;
    uint32_t first = 0, first_got = 0;
    // checked before the critical section, so that only the first mismatches
    // enter it: under `control` nearly every word mismatches
    std::atomic<bool> found{false};
#pragma omp parallel reduction(+ : bad)
    {
      std::vector<float> in(BLOCK + PAD), out(BLOCK + PAD);
#pragma omp for schedule(dynamic, 64)
      for (int64_t hi = 0; hi < (int64_t(1) << 16); ++hi)
      {
        const int64_t off = hi % PAD;
        float *x = in.data() + off;
        for (int64_t lo = 0; lo < BLOCK; ++lo)
          x[lo] = bits_float(uint32_t(hi << 16 | lo));
        // odd blocks in place, as binaryK_quant_ runs
        float *y = (hi & 1) ? x : out.data() + (PAD - 1 - off);
        f(x, y, BLOCK, p);
        for (int64_t lo = 0; lo < BLOCK; ++lo)
        {
          const uint32_t w = uint32_t(hi << 16 | lo);
          const uint32_t got = float_bits(y[lo]);
          const uint32_t ref = float_bits(scalar(want, bits_float(w), p));
          if (got != ref)
          {
            ++bad;
            if (!found.load(std::memory_order_relaxed))
            {
#pragma omp critical
              if (!found.load(std::memory_order_relaxed))
              {
                first = w;
                first_got = got;
                found.store(true, std::memory_order_relaxed);
              }
            }
          }
        }
      }
    }
    std::printf("  %-4s %12llu mismatches", label, (unsigned long long)bad);
    if (bad)
      std::printf("  (e.g. input 0x%08x: loop 0x%08x, scalar 0x%08x)", first, first_got,
                  float_bits(scalar(want, bits_float(first), p)));
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
  const struct
  {
    RoundMode mode;
    const char *label;
  } modes[] = {{RoundMode::RNE, "RNE"}, {RoundMode::RNA, "RNA"}, {RoundMode::RU, "RU"},
               {RoundMode::RD, "RD"},   {RoundMode::RZ, "RZ"},   {RoundMode::RO, "RO"}};
  if (!vector_quants.run[static_cast<int>(RoundMode::RZ)])
  {
    std::printf("no vector loops on this CPU or under MPTORCH_CPU_SIMD=0: nothing to check\n");
    return 1;
  }
  std::printf("the %s loops%s\n", vector_quants.isa,
              control ? ", negative control: each against another mode's scalar cast"
                      : ", all 2^32 binary32 inputs against the scalar casts");
  uint64_t total = 0, failed_controls = 0, compared = 0;
  for (const Format &f : formats)
  {
    const BinaryKParams p =
        make_binaryK_params<float>(f.man_bits, f.exp_bits, f.bias, true, f.sat, false);
    std::printf("%s\n", f.name);
    for (const auto &m : modes)
    {
      const VectorQuantFn fn = vector_quants.run[static_cast<int>(m.mode)];
      if (!fn || !vector_quant_format(m.mode, SUB, p))
      {
        std::printf("  %-4s kept by the scalar loop, skipped\n", m.label);
        continue;
      }
      const uint64_t c = sweep(fn, against(m.mode, control), p, m.label);
      total += c;
      failed_controls += (c == 0);
      ++compared;
    }
  }
  if (control)
  {
    std::printf("%s: %llu of the %llu comparisons found no mismatch\n",
                failed_controls ? "CONTROL FAILED" : "control ok",
                (unsigned long long)failed_controls, (unsigned long long)compared);
    return failed_controls ? 1 : 0;
  }
  std::printf("%s: %llu mismatches in %llu comparisons\n", total ? "FAILED" : "ok",
              (unsigned long long)total, (unsigned long long)compared);
  return total ? 1 : 0;
}
