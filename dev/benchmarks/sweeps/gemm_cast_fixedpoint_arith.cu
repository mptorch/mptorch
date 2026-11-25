// The fixed-point GEMM's fast RNE cast (common/fixedpoint_gemm.h's
// cast_fixedpoint_rne_fast) over every binary32 input, three-way, so that a
// disagreement is attributable:
//
//   reference  reference_casts.h's ref_fixedpoint under RNE, written from the
//              format's definition
//   bitwise    cast_fixedpoint.h's cast_fixedpoint_nearest_even, the shipped
//              cast the elementwise quantizers run
//   fast       the fast path, for each format its gate admits
//
// The gate is the shipped one (make_fixedpoint_fast); the sweep reports
// whether it is honest, over a grid of formats: wl from 2 to 24 (signed,
// unsigned and symmetric), fl from -104 to 126, past the gate's ends on
// both sides. A format the gate refuses is listed, and its fast arm is not
// run. A step below 2^-125 (fl = 126) is outside what the bitwise cast
// rounds correctly in binary32, and the Python checks warn about such a
// format; the bitwise cast and the reference disagree there near zero, which
// is listed and does not fail the run.
//
// Build & run (from the repo root):
//   nvcc -O3 -std=c++20 -arch=sm_89 --expt-relaxed-constexpr -I mptorch/csrc \
//        -I dev/benchmarks/sweeps dev/benchmarks/sweeps/gemm_cast_fixedpoint_arith.cu \
//        -o /tmp/fx_arith && /tmp/fx_arith          # every format, ~2 min
// The same source builds as plain C++ with the extension's host flags, over
// OpenMP threads:
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc -I dev/benchmarks/sweeps -x c++ \
//       dev/benchmarks/sweeps/gemm_cast_fixedpoint_arith.cu -o /tmp/fx_arith_host \
//     && /tmp/fx_arith_host spot                    # a few formats, ~1 min
// `spot` is the honest host gate, as in gemm_cast_float_arith.cu: the formats
// reach the fast path as run-time values, so what the host answers is whether
// the compiler keeps its two operations apart, which a handful of formats
// over every input settles. Pass `all` (the default on CUDA) or `spot`.
//
// Exit status 0 when every admitted format matches on every input.
#include "common/cast_fixedpoint.h"
#include "common/fixedpoint_gemm.h"
#include "common/modes.h"
#include "reference_casts.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#if defined(__CUDACC__)
#define CHECK(x)                                                                                   \
  do                                                                                               \
  {                                                                                                \
    cudaError_t e_ = (x);                                                                          \
    if (e_ != cudaSuccess)                                                                         \
    {                                                                                              \
      printf("ERR %s %d\n", cudaGetErrorString(e_), __LINE__);                                     \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)
#endif

using mptorch::gemm::FixedPointFast;

struct Counts
{
  unsigned long long fast_vs_bitwise, fast_vs_ref, bitwise_vs_ref;
  unsigned first_bad;
};

struct Fmt
{
  int wl, fl;
  bool is_signed, symmetric;
};

CUDA_HOST_DEVICE_INLINE uint32_t word_of(float f)
{
  return FLOAT_TO_BITS(&f);
}

// NaN results compare as NaNs, whatever their payloads: every arm passes the
// input's NaN through.
CUDA_HOST_DEVICE_INLINE bool same(uint32_t a, uint32_t b)
{
  const bool na = (a & 0x7FFFFFFFu) > 0x7F800000u, nb = (b & 0x7FFFFFFFu) > 0x7F800000u;
  return na || nb ? na == nb : a == b;
}

CUDA_HOST_DEVICE_INLINE void one(uint32_t bits, const Fmt &f, const FixedPointParamsT<float> &p,
                                 const FixedPointFast &fast, unsigned long long &nfb,
                                 unsigned long long &nfr, unsigned long long &nbr, uint32_t &first)
{
  const float x = bits_to_float(bits);
  const uint32_t r = word_of(cast_fixedpoint_nearest_even(x, f.wl, f.fl, f.is_signed, f.symmetric));
  const uint32_t b = word_of(cast_fixedpoint_nearest_even(x, f.is_signed, p));
  if (!same(b, r))
    ++nbr;
#if defined(MPTORCH_FAST_CAST) // nvcc's device pass, or a host build with the extension's flags
  if (fast.on)
  {
    const uint32_t q = word_of(mptorch::gemm::cast_fixedpoint_rne_fast(x, fast));
    if (!same(q, b) && nfb++ == 0)
      first = bits;
    if (!same(q, r))
      ++nfr;
  }
#else
  (void)nfb, (void)nfr, (void)first;
#endif
}

#if defined(__CUDACC__)
constexpr int PER_THREAD = 64;

__global__ void exhaustive(Fmt f, FixedPointParamsT<float> p, FixedPointFast fast, Counts *c)
{
  const uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
  unsigned long long nfb = 0, nfr = 0, nbr = 0;
  uint32_t first = 0;
  for (int k = 0; k < PER_THREAD; ++k)
    one((uint32_t)(base + k), f, p, fast, nfb, nfr, nbr, first);
  if (nfb && atomicAdd(&c->fast_vs_bitwise, nfb) == 0)
    c->first_bad = first;
  if (nfr)
    atomicAdd(&c->fast_vs_ref, nfr);
  if (nbr)
    atomicAdd(&c->bitwise_vs_ref, nbr);
}

static Counts run(const Fmt &f, const FixedPointParamsT<float> &p, const FixedPointFast &fast)
{
  static Counts *dc = nullptr;
  if (!dc)
    CHECK(cudaMalloc(&dc, sizeof(Counts)));
  CHECK(cudaMemset(dc, 0, sizeof(Counts)));
  exhaustive<<<(unsigned)((1ull << 32) / (256 * PER_THREAD)), 256>>>(f, p, fast, dc);
  CHECK(cudaDeviceSynchronize());
  Counts h;
  CHECK(cudaMemcpy(&h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
  return h;
}
#else
static Counts run(const Fmt &f, const FixedPointParamsT<float> &p, const FixedPointFast &fast)
{
  unsigned long long nfb = 0, nfr = 0, nbr = 0;
  uint32_t first = 0xFFFFFFFFu;
#pragma omp parallel for schedule(static) reduction(+ : nfb, nfr, nbr) reduction(min : first)
  for (int64_t hi = 0; hi < (1 << 16); ++hi)
  {
    uint32_t mine = 0xFFFFFFFFu;
    for (uint32_t lo = 0; lo < (1u << 16); ++lo)
      one((uint32_t)(((uint64_t)hi << 16) | lo), f, p, fast, nfb, nfr, nbr, mine);
    first = mine < first ? mine : first;
  }
  return Counts{nfb, nfr, nbr, first};
}
#endif

int main(int argc, char **argv)
{
#if defined(__CUDACC__)
  bool spot = argc > 1 && !std::strcmp(argv[1], "spot");
#else
  bool spot = !(argc > 1 && !std::strcmp(argv[1], "all"));
#endif
  std::vector<Fmt> fmts;
  if (spot)
    fmts = {{8, 4, true, false},     {16, 8, true, true},    {23, 0, true, false},
            {12, -103, true, false}, {9, 125, false, false}, {24, 20, false, false}};
  else
    for (int wl = 2; wl <= 24; ++wl)
      for (int fl : {-104, -103, -40, -3, 0, 1, 4, 8, 12, 20, 40, 80, 120, 125, 126})
        for (int kind = 0; kind < 3; ++kind)
        {
          const bool is_signed = kind != 2, symmetric = kind == 1;
          const int mag = wl - (is_signed ? 1 : 0);
          // the carrier must hold the format's ends (the Python checks refuse
          // the rest): the top below 2^128, the step a binary32 value
          if (mag - fl > 127 || -fl < -149 || (is_signed && wl < 2))
            continue;
          fmts.push_back({wl, fl, is_signed, symmetric});
        }
  int admitted = 0, bad = 0, ref_bad = 0;
  for (const Fmt &f : fmts)
  {
    const auto p = make_fixedpoint_params<float>(f.wl, f.fl, f.is_signed, f.symmetric);
    const FixedPointFast fast = mptorch::gemm::make_fixedpoint_fast(f.wl, f.fl, f.is_signed, p);
    const Counts c = run(f, p, fast);
    admitted += fast.on;
    const bool fail = fast.on && (c.fast_vs_bitwise || c.fast_vs_ref);
    bad += fail;
    ref_bad += c.bitwise_vs_ref != 0 && f.fl <= 125; // past 125, a warned format
    if (fail || c.bitwise_vs_ref || spot)
    {
      std::printf("wl=%2d fl=%4d %-9s %-8s fast!=bitwise %llu  fast!=ref %llu  bitwise!=ref %llu",
                  f.wl, f.fl, f.is_signed ? (f.symmetric ? "symmetric" : "signed") : "unsigned",
                  fast.on ? "admitted" : "refused", c.fast_vs_bitwise, c.fast_vs_ref,
                  c.bitwise_vs_ref);
      if (c.fast_vs_bitwise)
        std::printf("  (one at word 0x%08x)", c.first_bad);
      std::printf("\n");
    }
  }
  std::printf(
      "%zu formats, %d admitted by the gate; %d admitted formats mismatch; %d formats inside the "
      "cast's domain where the bitwise cast and the reference disagree\n",
      fmts.size(), admitted, bad, ref_bad);
  return bad || ref_bad ? 1 : 0;
}
