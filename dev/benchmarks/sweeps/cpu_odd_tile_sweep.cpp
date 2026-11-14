// Every binary32 word through the vectorized odd tiles of cpu/gemm_odd_tile.cpp
// (carrier_rounding=RoundMode.RO), against the scalar step the CPU kernel runs
// in their place.
//
// The tiles are the shipped file, compiled in here, so this checks the code
// the extension runs: on AVX-512 the directed pair and its parity select, on
// AVX2 and NEON the compiler's vectorization of carrier_odd.h's binary32
// primitives, each followed by cast_binaryK_rne_fast. Each word goes through
// the three tiles once, against products that are never trivial:
//
//   fused     sum = x, product a_i * b_j:        cast(odd::fma(a_i, b_j, x))
//   split     sum = s_l, product a_i * x:        cast(odd::add(s_l, cast(odd::mul(a_i, x))))
//   mul-only  sum = s_l, product a_i * x:        s_l + cast(odd::mul(a_i, x))
//
// with a_i, b_j and s_l fixed spreads over the exponent range (zeros, signs,
// subnormals and values near the top included). Then the special cases of the
// odd step (zero signs, the largest finite value, subnormal parity, sums past
// the carrier, infinities) run through each tile as single lanes. Each result
// must be the scalar expression's word for word, on the scalar kernel's own
// cast (cast_binaryK_nearest_even, which takes the fast path for every format
// here); a block that is handed back must be as it was, and each of its lanes
// is then run alone, to match the scalar step or be handed back with a NaN
// the scalar step makes too. Built with the extension's host flags:
//
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc dev/benchmarks/sweeps/cpu_odd_tile_sweep.cpp \
//       mptorch/csrc/cpu/gemm_odd_tile.cpp -o /tmp/cpu_odd_tile_sweep
//   /tmp/cpu_odd_tile_sweep                         # the widest ISA the CPU has
//   MPTORCH_CPU_SIMD=avx2 /tmp/cpu_odd_tile_sweep   # the AVX2 tiles on an AVX-512 CPU
//
// About 15-25 s per format on 16 threads. MPTORCH_CPU_SIMD=0 leaves the tiles
// null, and the program then says it has nothing to check.

#include "cpu/gemm_odd_tile.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

using mptorch::gemm_cpu::odd_tiles;
using mptorch::gemm_cpu::RneTileFn;
namespace odd = mptorch::gemm::odd;

namespace
{
  constexpr int64_t TI = 32, TJ = 32, LANES = TI * TJ;

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

  float cast(float x, const BinaryKParams &p)
  {
    return cast_binaryK_nearest_even(x, true, SubnormalsMode::SUBNORMALS, p);
  }
  // The scalar kernel's steps (common/carrier_odd.h's SplitMacOdd and
  // FusedMacOdd over BinaryK*T<float, RNE>; the multiplier's x * 1 is exact).
  float want_fused(float a, float b, float s, const BinaryKParams &p)
  {
    return cast(odd::fma(a, b, s), p);
  }
  float want_split(float a, float b, float s, const BinaryKParams &p)
  {
    return cast(odd::add(s, cast(odd::mul(a, b) * 1.0f, p)), p);
  }
  float want_mul_only(float a, float b, float s, const BinaryKParams &p)
  {
    return s + cast(odd::mul(a, b) * 1.0f, p);
  }

  // Fixed spreads: a value per index, its exponent walking the range, with a
  // few special values mixed in.
  float spread(int64_t i, int salt)
  {
    switch ((i + salt) % 23)
    {
    case 0:
      return 0.0f;
    case 1:
      return -0.0f;
    case 2:
      return std::ldexp(1.0f, -149) * float(1 + i % 7); // subnormals
    case 3:
      return std::numeric_limits<float>::max() * (i % 2 ? -0.75f : 0.5f);
    case 4:
      return i % 2 ? -1.0f : 1.0f;
    default:
    {
      const float m = 1.0f + float((i * 2654435761u + salt * 40503u) % 8388608u) * 0x1p-23f;
      const int e = int((i * 37 + salt * 11) % 181) - 90;
      return std::ldexp((i + salt) % 3 ? m : -m, e);
    }
    }
  }

  // One lane alone: true if it ran, else it was handed back.
  bool one_lane(RneTileFn tile, float a, float b, float s, float &r, const BinaryKParams &p)
  {
    r = s;
    return tile(&a, &b, &r, 1, 1, 1, p, p);
  }

  template <class Want>
  uint64_t check_lane(RneTileFn tile, float a, float b, float s, const BinaryKParams &p, Want want)
  {
    float r;
    if (one_lane(tile, a, b, s, r, p))
      return float_bits(r) != float_bits(want(a, b, s, p));
    return float_bits(r) != float_bits(s) || !std::isnan(want(a, b, s, p));
  }

  // Mismatches over every word for one format, through the three tiles.
  uint64_t sweep(const BinaryKParams &p)
  {
    alignas(64) float a_row[TI], b_col[TJ], s_lane[LANES];
    for (int64_t i = 0; i < TI; ++i)
      a_row[i] = spread(i, 1);
    for (int64_t j = 0; j < TJ; ++j)
      b_col[j] = spread(j, 5);
    for (int64_t l = 0; l < LANES; ++l)
      s_lane[l] = spread(l, 9);
    uint64_t bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(dynamic, 64)
    for (int64_t chunk = 0; chunk < (int64_t(1) << 32) / LANES; ++chunk)
    {
      alignas(64) float x[LANES], sums[LANES];
      for (int64_t l = 0; l < LANES; ++l)
        x[l] = bits_float(static_cast<uint32_t>(chunk * LANES + l));

      // fused: one K-step of a_i * b_j over sum = x
      std::memcpy(sums, x, sizeof sums);
      if (odd_tiles.fused(a_row, b_col, sums, TI, 1, TJ, p, p))
      {
        for (int64_t l = 0; l < LANES; ++l)
          bad += float_bits(sums[l]) != float_bits(want_fused(a_row[l / TJ], b_col[l % TJ], x[l], p));
      }
      else
      {
        for (int64_t l = 0; l < LANES; ++l)
        {
          bad += float_bits(sums[l]) != float_bits(x[l]);
          bad += check_lane(odd_tiles.fused, a_row[l / TJ], b_col[l % TJ], x[l], p, want_fused);
        }
      }

      // split and mul-only: a_i against a row of words, over the sums s_l
      for (const auto &[tile, want] :
           {std::pair{odd_tiles.split, &want_split}, std::pair{odd_tiles.mul_only, &want_mul_only}})
        for (int64_t i = 0; i < TI; ++i)
        {
          alignas(64) float out[TJ];
          std::memcpy(out, s_lane + i * TJ, sizeof out);
          const float *row = x + i * TJ;
          if (tile(a_row + i, row, out, 1, 1, TJ, p, p))
          {
            for (int64_t j = 0; j < TJ; ++j)
              bad += float_bits(out[j]) != float_bits(want(a_row[i], row[j], s_lane[i * TJ + j], p));
          }
          else
          {
            for (int64_t j = 0; j < TJ; ++j)
            {
              bad += float_bits(out[j]) != float_bits(s_lane[i * TJ + j]);
              bad += check_lane(tile, a_row[i], row[j], s_lane[i * TJ + j], p, want);
            }
          }
        }
    }
    return bad;
  }

  // The binary32 vectors of the odd step's review (dev/continuation_plan.md's
  // phase N; tests/test_carrier_rounding.py's VECTORS), as (a, b, c).
  uint64_t special_cases(const BinaryKParams &p)
  {
    const float inf = std::numeric_limits<float>::infinity();
    const float omega = std::numeric_limits<float>::max();
    const float v[][3] = {
        {1, 1, 0x1p-30f}, {1, 1, -0x1p-30f}, {1, 1, 0x1p-22f},
        {1, 1 + 0x1p-23f, -0x1p-40f}, {1, 1 + 0x1p-22f, -0x1p-40f},
        {1 + 0x1p-12f, 1 + 0x1p-12f, -0x1p-11f},
        {0x1p-50f, 0x1p-50f, 1}, {1, 1, 0x1p-100f}, {0x1p-50f, -0x1p-50f, 1},
        {0x1p-75f, 0x1p-75f, 0}, {0x1p-75f, 0x1p-75f, -0x1p-149f},
        {7 * 0x1p-75f, 0x1p-75f, 0}, {9 * 0x1p-75f, 0x1p-75f, 0},
        {0x1p-63f * (1 - 0x1p-24f), 0x1p-63f, 0},
        {0x1p64f, 0x1p64f, 0}, {omega, 1, 0x1p103f}, {0x1p64f, 0x1p64f, -omega},
        {1, 0x1p100f, 0x1p-100f}, {1, -0x1p-100f, 0x1p100f},
        {1, 1, -1}, {-0.0f, 1, -0.0f}, {0.0f, 1, -0.0f}, {-1, 0, -0.0f},
        {0x1p64f, 0x1p64f, -inf}, {inf, 1, 1}, {inf, 0, 1}, {1, 1, inf},
        {std::nanf(""), 1, 1}, {1, 1, std::nanf("")},
    };
    uint64_t bad = 0;
    for (const auto &t : v)
    {
      bad += check_lane(odd_tiles.fused, t[0], t[1], t[2], p, want_fused);
      bad += check_lane(odd_tiles.split, t[0], t[1], t[2], p, want_split);
      bad += check_lane(odd_tiles.mul_only, t[0], t[1], t[2], p, want_mul_only);
    }
    return bad;
  }
} // namespace

int main()
{
  if (!odd_tiles.split)
  {
    std::printf("no vectorized odd tile on this CPU or build (isa %s): nothing to check\n",
                odd_tiles.isa);
    return 0;
  }
  const Format formats[] = {
      {3, 4, 7, SaturationMode::SAT_NONE, "e4m3-like, SAT_NONE"},
      {3, 4, 8, SaturationMode::SAT_FINITE, "binary8p4, SAT_FINITE"},
      {10, 5, 15, SaturationMode::SAT_NONE, "binary16"},
      {10, 5, 16, SaturationMode::SAT_FINITE_EXTENDED, "binary16p11, SAT_FINITE_EXTENDED"},
      {7, 7, 63, SaturationMode::SAT_NONE, "P = 8, 7 exponent bits"},
      {1, 5, 15, SaturationMode::SAT_NONE, "P = 2"},
      {2, 3, 4, SaturationMode::SAT_NONE, "binary6p3"},
      {21, 7, 63, SaturationMode::SAT_FINITE, "P = 22, 7 exponent bits"},
      {22, 7, 63, SaturationMode::SAT_FINITE, "P = 23, 7 exponent bits"},
      {5, 6, 31, SaturationMode::SAT_NONE, "P = 6, 6 exponent bits"},
  };
  std::printf("isa %s, every binary32 word through the fused, split and mul-only odd tiles\n",
              odd_tiles.isa);
  uint64_t total = 0;
  int checked = 0;
  for (const Format &f : formats)
  {
    const BinaryKParams p =
        make_binaryK_params<float>(f.man_bits, f.exp_bits, f.bias, true, f.sat, false);
    if (!p.fast_rne)
    {
      std::printf("  %-36s the fast gate is off: not the tile's case, skipped\n", f.name);
      continue;
    }
    const uint64_t bad = sweep(p) + special_cases(p);
    std::printf("  %-36s %llu mismatches\n", f.name, (unsigned long long)bad);
    total += bad;
    ++checked;
  }
  std::printf("%s: %d formats checked, %llu mismatches\n", total ? "FAIL" : "PASS", checked,
              (unsigned long long)total);
  return total ? 1 : 0;
}
