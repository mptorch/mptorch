// Every binary32 word through the vectorized word tiles of
// cpu/gemm_word_tile.cpp (RNA, RU, RD, RZ, RO), against the scalar step the
// CPU kernel runs in their place.
//
// The tiles are the shipped file, compiled in here, so this checks the code
// the extension runs: the compiler's vectorization of
// common/cast_binaryK_word.h, lane by lane, on every input (the casts
// themselves are held to the shipped casts by cpu_word_cast_sweep.cpp). Each
// word goes through three one-step tiles, as cpu_rne_tile_sweep.cpp's do:
//
//   fused     a = 0, b = -1, sum = x:  cast(fma(0, -1, x)) = cast(x) exactly
//   split     a = 1, b = x, sum = 0:   cast(0 + cast(1 * x))
//   mul-only  a = 1, b = x, sum = 0:   0 + cast(1 * x)
//
// and each result must be the scalar expression's word for word, on the
// shipped scalar casts. A block whose tile ends with a NaN is handed back:
// it must then be as it was, and each of its lanes is run alone, to match the
// scalar step or be handed back with a NaN the scalar step makes too.
//
// The odd macs' tiles (cpu/gemm_word_odd_tile.cpp, carrier_rounding=RO) take
// each word x with an inexact product, 2/3 times x, against a running sum y
// made from x's word (its halves swapped), so that both odd steps round:
//
//   odd fused     cast(odd::fma(2/3, x, y))
//   odd split     cast(odd::add(y, cast(odd::mul(2/3, x))))
//   odd mul-only  y + cast(odd::mul(2/3, x))
//
// on carrier_odd.h's scalar primitives, which the scalar kernel's odd step
// calls. `control` holds each mode's tiles to another mode's scalar step,
// which must mismatch. Built with the extension's host flags:
//
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc dev/benchmarks/sweeps/cpu_word_tile_sweep.cpp \
//       mptorch/csrc/cpu/gemm_word_tile.cpp mptorch/csrc/cpu/gemm_word_odd_tile.cpp \
//       -o /tmp/cpu_word_tile_sweep
//   /tmp/cpu_word_tile_sweep                         # the widest ISA the CPU has
//   MPTORCH_CPU_SIMD=avx2 /tmp/cpu_word_tile_sweep   # the AVX2 tiles on an AVX-512 CPU
//   /tmp/cpu_word_tile_sweep control                 # the negative control
//
// About 2 min per format on 16 threads, both tile families. MPTORCH_CPU_SIMD=0 leaves the tiles
// null, and the program then says it has nothing to check.

#include "cpu/gemm_word_odd_tile.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

using mptorch::gemm_cpu::RneTileFn;
using mptorch::gemm_cpu::word_mode_index;
using mptorch::gemm_cpu::word_odd_tiles;
using mptorch::gemm_cpu::word_tiles;
namespace odd = mptorch::gemm::odd;

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

  // the shipped scalar cast, as gemm_policy.h's policies call it
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

  // One tile call's results against the scalar step (cpu_rne_tile_sweep.cpp's
  // check): `in` the sums before and `out` after.
  template <class Call, class Want>
  uint64_t check(bool done, const float *in, const float *out, int64_t n, Call one_lane, Want want)
  {
    uint64_t bad = 0;
    if (done)
    {
      for (int64_t l = 0; l < n; ++l)
        bad += float_bits(out[l]) != float_bits(want(in[l]));
      return bad;
    }
    for (int64_t l = 0; l < n; ++l)
      bad += float_bits(out[l]) != float_bits(in[l]);
    for (int64_t l = 0; l < n; ++l)
    {
      float r;
      if (one_lane(in[l], r))
        bad += float_bits(r) != float_bits(want(in[l]));
      else
        bad += float_bits(r) != float_bits(in[l]) || !std::isnan(want(in[l]));
    }
    return bad;
  }

  // A row of words through a one-step split or product-only tile, a = 1 and
  // the sums zero, against `want`.
  template <class Want>
  uint64_t check_row(RneTileFn tile, const float *one_a, const float *x, int64_t tj,
                     const BinaryKParams &p, Want want)
  {
    uint64_t bad = 0;
    alignas(64) float out[32] = {};
    const bool done = tile(one_a, x, out, 1, 1, tj, p, p);
    const float one = 1.0f;
    for (int64_t j = 0; j < tj; ++j)
    {
      if (done)
        bad += float_bits(out[j]) != float_bits(want(x[j]));
      else
      {
        bad += float_bits(out[j]) != 0u; // handed back: the zeros as they were
        float r = 0.0f;
        if (tile(&one, &x[j], &r, 1, 1, 1, p, p))
          bad += float_bits(r) != float_bits(want(x[j]));
        else
          bad += float_bits(r) != 0u || !std::isnan(want(x[j]));
      }
    }
    return bad;
  }

  // Mismatches over every word for one format and mode, through the three
  // tiles, against the scalar step of mode `want_mode`.
  uint64_t sweep(const BinaryKParams &p, RoundMode mode, RoundMode want_mode)
  {
    constexpr int64_t TI = 32, TJ = 32, LANES = TI * TJ;
    const int r = word_mode_index(mode);
    const RneTileFn fused = word_tiles.fused[r], split = word_tiles.split[r],
                    mul_only = word_tiles.mul_only[r];
    auto cast = [&](float v) { return shipped(want_mode, v, p); };
    uint64_t bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(dynamic, 64)
    for (int64_t chunk = 0; chunk < (int64_t(1) << 32) / LANES; ++chunk)
    {
      alignas(64) float x[LANES], sums[LANES];
      alignas(64) float zero_a[TI], one_a[TI], minus_one_b[TJ];
      for (int i = 0; i < TI; ++i)
        zero_a[i] = 0.0f, one_a[i] = 1.0f;
      for (int j = 0; j < TJ; ++j)
        minus_one_b[j] = -1.0f;
      for (int64_t l = 0; l < LANES; ++l)
        x[l] = sums[l] = bits_float(static_cast<uint32_t>(chunk * LANES + l));
      const float zero = 0.0f, minus_one = -1.0f;

      const bool done = fused(zero_a, minus_one_b, sums, TI, 1, TJ, p, p);
      bad += check(
          done, x, sums, LANES,
          [&](float v, float &out)
          {
            out = v;
            return fused(&zero, &minus_one, &out, 1, 1, 1, p, p);
          },
          [&](float v) { return cast(std::fma(0.0f, -1.0f, v)); });
      for (int64_t i = 0; i < TI; ++i)
      {
        bad += check_row(split, one_a, x + i * TJ, TJ, p,
                         [&](float v) { return cast(0.0f + cast(1.0f * v)); });
        bad += check_row(mul_only, one_a, x + i * TJ, TJ, p,
                         [&](float v) { return 0.0f + cast(1.0f * v); });
      }
    }
    return bad;
  }

  // The odd tiles: one K-step of a = 2/3 against b = a row of words x, over
  // sums y = x's word with its halves swapped, each lane against the scalar
  // odd step. A block handed back must be as it was, and each of its lanes is
  // run alone, to match or be handed back with a NaN the scalar step makes.
  template <class Want>
  uint64_t check_odd_row(RneTileFn tile, const float *x, int64_t tj, const BinaryKParams &p,
                         Want want)
  {
    const float c = 2.0f / 3.0f;
    alignas(64) float y[32], out[32];
    for (int64_t j = 0; j < tj; ++j)
    {
      const uint32_t w = float_bits(x[j]);
      y[j] = out[j] = bits_float((w << 16) | (w >> 16));
    }
    uint64_t bad = 0;
    const bool done = tile(&c, x, out, 1, 1, tj, p, p);
    for (int64_t j = 0; j < tj; ++j)
    {
      if (done)
        bad += float_bits(out[j]) != float_bits(want(x[j], y[j]));
      else
      {
        bad += float_bits(out[j]) != float_bits(y[j]);
        float r = y[j];
        if (tile(&c, &x[j], &r, 1, 1, 1, p, p))
          bad += float_bits(r) != float_bits(want(x[j], y[j]));
        else
          bad += float_bits(r) != float_bits(y[j]) || !std::isnan(want(x[j], y[j]));
      }
    }
    return bad;
  }

  uint64_t sweep_odd(const BinaryKParams &p, RoundMode mode, RoundMode want_mode)
  {
    constexpr int64_t TJ = 32;
    const int r = word_mode_index(mode);
    const RneTileFn fused = word_odd_tiles.fused[r], split = word_odd_tiles.split[r],
                    mul_only = word_odd_tiles.mul_only[r];
    auto cast = [&](float v) { return shipped(want_mode, v, p); };
    const float c = 2.0f / 3.0f;
    uint64_t bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(dynamic, 64)
    for (int64_t chunk = 0; chunk < (int64_t(1) << 32) / TJ; ++chunk)
    {
      alignas(64) float x[TJ];
      for (int64_t j = 0; j < TJ; ++j)
        x[j] = bits_float(static_cast<uint32_t>(chunk * TJ + j));
      bad +=
          check_odd_row(fused, x, TJ, p, [&](float v, float y) { return cast(odd::fma(c, v, y)); });
      bad += check_odd_row(split, x, TJ, p, [&](float v, float y)
                           { return cast(odd::add(y, cast(odd::mul(c, v)))); });
      bad += check_odd_row(mul_only, x, TJ, p,
                           [&](float v, float y) { return y + cast(odd::mul(c, v)); });
    }
    return bad;
  }

  RoundMode other(RoundMode mode)
  {
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
} // namespace

int main(int argc, char **argv)
{
  const bool control = argc > 1 && std::strcmp(argv[1], "control") == 0;
  if (!word_tiles.split[0])
  {
    std::printf("no word tiles on this CPU or under MPTORCH_CPU_SIMD=0: nothing to check\n");
    return 0;
  }
  std::printf("the %s word tiles%s\n", word_tiles.isa,
              control ? ", negative control: each against another mode's scalar step" : "");
  const Format formats[] = {
      {3, 4, 7, SaturationMode::SAT_NONE, "e4m3-like, SAT_NONE"},
      {3, 4, 8, SaturationMode::SAT_FINITE, "binary8p4, SAT_FINITE"},
      {10, 5, 16, SaturationMode::SAT_FINITE_EXTENDED, "binary16p11, SAT_FINITE_EXTENDED"},
      {2, 5, 16, SaturationMode::SAT_NONE, "binary8p3 (e5m2-like)"},
      {7, 8, 127, SaturationMode::SAT_NONE, "bf16-like, bias 127"},
      {22, 7, 63, SaturationMode::SAT_FINITE, "P = 23, 7 exponent bits"},
  };
  const RoundMode modes[] = {RoundMode::RNA, RoundMode::RU, RoundMode::RD, RoundMode::RZ,
                             RoundMode::RO};
  const char *names[] = {"RNA", "RU", "RD", "RZ", "RO"};
  uint64_t total = 0, quiet_controls = 0;
  for (const Format &f : formats)
  {
    const BinaryKParams p =
        make_binaryK_params<float>(f.man_bits, f.exp_bits, f.bias, true, f.sat, false);
    std::printf("%s\n", f.name);
    for (int m = 0; m < 5; ++m)
    {
      const RoundMode want = control ? other(modes[m]) : modes[m];
      const uint64_t bad = sweep(p, modes[m], want);
      const uint64_t bad_odd = word_odd_tiles.split[0] ? sweep_odd(p, modes[m], want) : 0;
      std::printf("  %-4s %12llu mismatches, odd carrier %12llu\n", names[m],
                  (unsigned long long)bad, (unsigned long long)bad_odd);
      total += bad + bad_odd;
      quiet_controls += (bad == 0) + (word_odd_tiles.split[0] && bad_odd == 0);
    }
  }
  if (control)
  {
    std::printf("%s: %llu of the comparisons found no mismatch\n",
                quiet_controls ? "CONTROL FAILED" : "control ok",
                (unsigned long long)quiet_controls);
    return quiet_controls ? 1 : 0;
  }
  std::printf("%s: %llu mismatches in all\n", total ? "FAILED" : "ok", (unsigned long long)total);
  return total ? 1 : 0;
}
