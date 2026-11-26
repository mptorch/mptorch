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
// The `sr` modes check the GEMM's stochastic cast (fixedpoint_gemm.h's
// cast_fixedpoint_stochastic_gemm) against the shipped one the elementwise
// quantizers run (cast_fixedpoint_stochastic), whose results the test suite
// holds to P3109's definition: every binary32 input, under each of a set of
// draws, for a grid of formats (wl 2 to 24, signed, symmetric and unsigned,
// fl from -100 to 130, a carrier-subnormal step included), N random bits
// from 0 up to the carrier rule's bound (mag_bits + N <= 24), and the three
// variants. The cast reads only R, the top N bits of the draw's mantissa,
// so the draws are R's extremes, its two middle values and an odd and an
// even value (every R for N <= 2), with the draw's other bits random.
//   /tmp/fx_arith sr          # the grid: 216 formats, 3,240 cases, ~15 min
//   /tmp/fx_arith sr-short    # six formats, the gate for a change, ~10 s
//   /tmp/fx_arith sr-control  # the negative control: the GEMM's constants
//                             # built for StochasticA under every variant;
//                             # B and C must mismatch
// On the host, `sr-short` runs over OpenMP threads (~10 min on 16). The
// GEMM's cast is integer arithmetic, with no float operation a host
// compiler could fuse or widen, so the GPU's run is the gate.
//
// Results (2026-11-25, RTX 4060 and the Ryzen 7940HS): `sr` 0 of 3,240
// cases mismatch (15 min 24 s; a first, wider grid of 288 formats and 4,428
// cases, 21 min, also 0); `sr-short` 0 on the GPU and on the host (9 min 45 s
// on 16 threads); `sr-control` 1,941,258,420 mismatches.
//
// The `det` modes do the same for the GEMM's deterministic casts
// (fixedpoint_word_round, under RNE, RNA, RU, RD, RZ and RO) against the
// shipped bitwise casts, on the same grid of formats:
//   /tmp/fx_arith det          # 216 formats x 6 modes
//   /tmp/fx_arith det-short    # six formats, the gate for a change
//   /tmp/fx_arith det-control  # the line under RNE against the shipped RNA
//                              # (they differ only at ties) and under RU
//                              # against RD: both must mismatch
// The line has a device spelling and a host one (the same answers, each
// faster on its side), so run `det` on the GPU and `det-short` on the host.
//
// Results (2026-11-26): `det` 0 of 1,296 cases mismatch on the GPU (1 min 33
// s); `det-short` 0 on the GPU and on the host (1 min 18 s on 16 threads);
// `det-control` 68,478 tie mismatches and 8,321,159,945 RU-against-RD ones on
// both. A first version, which read a code's parity from the word's bit at
// the step, mismatched `det-short` under RNE and RO for FixedPoint(9, 125):
// under twice the step that bit is the exponent field's lowest bit.
//
// Exit status 0 when every admitted format matches on every input (in the
// `sr` and `det` modes, every case; in the controls, when they mismatch).
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

// --- the stochastic cast ---------------------------------------------------------

constexpr int MAX_DRAWS = 8;

struct Draws
{
  int n;
  uint32_t w[MAX_DRAWS];
};

struct SrCase
{
  FixedPointParamsT<float> p;
  mptorch::gemm::FixedPointStochasticT<float> st;
  int spec;
  bool is_signed;
};

CUDA_HOST_DEVICE_INLINE unsigned long long sr_one(uint32_t bits, const SrCase &c, const Draws &d,
                                                  uint32_t &first)
{
  const float x = bits_to_float(bits);
  unsigned long long bad = 0;
  for (int i = 0; i < d.n; ++i)
  {
    const uint32_t g = word_of(mptorch::gemm::cast_fixedpoint_stochastic_gemm(x, d.w[i], c.st, c.spec,
                                                                              c.is_signed, c.p));
    const uint32_t s = word_of(cast_fixedpoint_stochastic(x, d.w[i], c.spec, c.is_signed, c.p));
    if (!same(g, s) && bad++ == 0)
      first = bits;
  }
  return bad;
}

#if defined(__CUDACC__)
__global__ void exhaustive_sr(SrCase c, Draws d, Counts *out)
{
  const uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
  unsigned long long bad = 0;
  uint32_t first = 0;
  for (int k = 0; k < PER_THREAD; ++k)
    bad += sr_one((uint32_t)(base + k), c, d, first);
  if (bad && atomicAdd(&out->fast_vs_bitwise, bad) == 0)
    out->first_bad = first;
}

static Counts run_sr(const SrCase &c, const Draws &d)
{
  static Counts *dc = nullptr;
  if (!dc)
    CHECK(cudaMalloc(&dc, sizeof(Counts)));
  CHECK(cudaMemset(dc, 0, sizeof(Counts)));
  exhaustive_sr<<<(unsigned)((1ull << 32) / (256 * PER_THREAD)), 256>>>(c, d, dc);
  CHECK(cudaDeviceSynchronize());
  Counts h;
  CHECK(cudaMemcpy(&h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
  return h;
}
#else
static Counts run_sr(const SrCase &c, const Draws &d)
{
  unsigned long long bad = 0;
  uint32_t first = 0xFFFFFFFFu;
#pragma omp parallel for schedule(static) reduction(+ : bad) reduction(min : first)
  for (int64_t hi = 0; hi < (1 << 16); ++hi)
  {
    uint32_t mine = 0xFFFFFFFFu;
    for (uint32_t lo = 0; lo < (1u << 16); ++lo)
      bad += sr_one((uint32_t)(((uint64_t)hi << 16) | lo), c, d, mine);
    first = mine < first ? mine : first;
  }
  return Counts{bad, 0, 0, first};
}
#endif

// R's extremes, its middle values and an odd and an even one (every R for
// N <= 2), each at the top N bits of the mantissa, with the draw's other bits
// from a fixed LCG: the cast must ignore them.
static Draws draws_for(int n)
{
  static uint64_t lcg = 0x2545F4914F6CDD1Dull;
  auto next = [&] {
    lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(lcg >> 32);
  };
  Draws d{};
  const uint32_t top = n == 0 ? 0u : (1u << n) - 1u;
  std::vector<uint32_t> rs;
  if (n <= 2)
    for (uint32_t r = 0; r <= top; ++r)
      rs.push_back(r);
  else
    rs = {0u, 1u, top / 2u, top / 2u + 1u, top - 1u, top, 5u};
  for (uint32_t r : rs)
  {
    const uint32_t low = 23 - n >= 32 ? 0u : next() & ((1u << (23 - n)) - 1u);
    const uint32_t high = next() & 0xFF800000u; // sign and exponent bits, ignored too
    d.w[d.n++] = high | (n == 0 ? 0u : r << (23 - n)) | low;
  }
  return d;
}

static int sr_main(const char *mode)
{
  const bool shortrun = !std::strcmp(mode, "sr-short");
  const bool control = !std::strcmp(mode, "sr-control");
  std::vector<Fmt> fmts;
  if (shortrun || control)
    fmts = {{8, 4, true, false}, {16, 8, true, true}, {16, 4, true, false},
            {12, -20, false, false}, {9, 125, false, false}, {10, 127, true, false}};
  else
    for (int wl : {2, 8, 12, 16, 23, 24})
      for (int fl : {-100, -20, 0, 3, 8, 16, 40, 120, 125, 126, 127, 130})
        for (int kind = 0; kind < 3; ++kind)
        {
          const bool is_signed = kind != 2, symmetric = kind == 1;
          const int mag = wl - (is_signed ? 1 : 0);
          if (mag - fl > 127 || -fl < -149 || (is_signed && wl < 2) || mag < 1)
            continue;
          fmts.push_back({wl, fl, is_signed, symmetric});
        }
  int cases = 0, bad = 0;
  unsigned long long control_bc = 0;
  for (const Fmt &f : fmts)
  {
    const int mag = f.wl - (f.is_signed ? 1 : 0);
    const auto p = make_fixedpoint_params<float>(f.wl, f.fl, f.is_signed, f.symmetric);
    for (int n : {0, 1, 2, 4, 8, 12, 16, 23})
    {
      if (mag + n > 24 || ((shortrun || control) && n != 0 && n != 1 && n != 4))
        continue;
      const Draws d = draws_for(n);
      for (int variant = 0; variant < 3; ++variant)
      {
        SrCase c{p, {}, sr_spec(n, variant), f.is_signed};
        c.st = mptorch::gemm::make_fixedpoint_stochastic<float>(p, f.is_signed, control ? sr_spec(n, 0) : c.spec);
        const Counts r = run_sr(c, d);
        ++cases;
        if (control)
        {
          if (variant != 0)
            control_bc += r.fast_vs_bitwise;
          continue;
        }
        if (r.fast_vs_bitwise)
        {
          ++bad;
          std::printf("wl=%2d fl=%4d %-9s N=%2d %c: %llu mismatches over %d draws (one at word 0x%08x)\n",
                      f.wl, f.fl, f.is_signed ? (f.symmetric ? "symmetric" : "signed") : "unsigned", n,
                      "ABC"[variant], r.fast_vs_bitwise, d.n, r.first_bad);
        }
      }
    }
  }
  if (control)
  {
    std::printf("%d cases, StochasticB and C under A's constants: %llu mismatches (the control: some must)\n",
                cases, control_bc);
    return control_bc == 0 ? 1 : 0;
  }
  std::printf("%zu formats, %d cases (format, N, variant); %d cases mismatch\n", fmts.size(), cases, bad);
  return bad ? 1 : 0;
}

// --- the deterministic casts ------------------------------------------------------

using mptorch::gemm::FixedPointWordT;

// The line in mode RM against the shipped bitwise cast in mode REF (REF = RM
// but in the control).
template <RoundMode RM, RoundMode REF>
CUDA_HOST_DEVICE_INLINE bool det_differs(uint32_t bits, const FixedPointWordT<float> &c, bool is_signed,
                                         const FixedPointParamsT<float> &p)
{
  const float x = bits_to_float(bits);
  const uint32_t g = word_of(mptorch::gemm::fixedpoint_word_round<float, RM>(x, c, is_signed, p));
  const uint32_t s = word_of(mptorch::gemm::fixedpoint_bitwise_cast<float, REF>(x, is_signed, p));
  return !same(g, s);
}

#if defined(__CUDACC__)
template <RoundMode RM, RoundMode REF>
__global__ void exhaustive_det(FixedPointWordT<float> c, bool is_signed, FixedPointParamsT<float> p, Counts *out)
{
  const uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
  unsigned long long bad = 0;
  uint32_t first = 0;
  for (int k = 0; k < PER_THREAD; ++k)
    if (det_differs<RM, REF>((uint32_t)(base + k), c, is_signed, p) && bad++ == 0)
      first = (uint32_t)(base + k);
  if (bad && atomicAdd(&out->fast_vs_bitwise, bad) == 0)
    out->first_bad = first;
}

template <RoundMode RM, RoundMode REF>
static Counts run_det(const FixedPointWordT<float> &c, bool is_signed, const FixedPointParamsT<float> &p)
{
  static Counts *dc = nullptr;
  if (!dc)
    CHECK(cudaMalloc(&dc, sizeof(Counts)));
  CHECK(cudaMemset(dc, 0, sizeof(Counts)));
  exhaustive_det<RM, REF><<<(unsigned)((1ull << 32) / (256 * PER_THREAD)), 256>>>(c, is_signed, p, dc);
  CHECK(cudaDeviceSynchronize());
  Counts h;
  CHECK(cudaMemcpy(&h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
  return h;
}
#else
template <RoundMode RM, RoundMode REF>
static Counts run_det(const FixedPointWordT<float> &c, bool is_signed, const FixedPointParamsT<float> &p)
{
  unsigned long long bad = 0;
  uint32_t first = 0xFFFFFFFFu;
#pragma omp parallel for schedule(static) reduction(+ : bad) reduction(min : first)
  for (int64_t hi = 0; hi < (1 << 16); ++hi)
    for (uint32_t lo = 0; lo < (1u << 16); ++lo)
    {
      const uint32_t bits = (uint32_t)(((uint64_t)hi << 16) | lo);
      if (det_differs<RM, REF>(bits, c, is_signed, p))
      {
        ++bad;
        first = bits < first ? bits : first;
      }
    }
  return Counts{bad, 0, 0, first};
}
#endif

static Counts run_det_mode(RoundMode rm, const FixedPointWordT<float> &c, bool is_signed,
                           const FixedPointParamsT<float> &p)
{
  switch (rm)
  {
  case RoundMode::RNA:
    return run_det<RoundMode::RNA, RoundMode::RNA>(c, is_signed, p);
  case RoundMode::RU:
    return run_det<RoundMode::RU, RoundMode::RU>(c, is_signed, p);
  case RoundMode::RD:
    return run_det<RoundMode::RD, RoundMode::RD>(c, is_signed, p);
  case RoundMode::RZ:
    return run_det<RoundMode::RZ, RoundMode::RZ>(c, is_signed, p);
  case RoundMode::RO:
    return run_det<RoundMode::RO, RoundMode::RO>(c, is_signed, p);
  default:
    return run_det<RoundMode::RNE, RoundMode::RNE>(c, is_signed, p);
  }
}

static int det_main(const char *mode)
{
  const bool shortrun = !std::strcmp(mode, "det-short");
  const bool control = !std::strcmp(mode, "det-control");
  std::vector<Fmt> fmts;
  if (shortrun || control)
    fmts = {{8, 4, true, false}, {16, 8, true, true}, {16, 4, true, false},
            {12, -20, false, false}, {9, 125, false, false}, {10, 127, true, false}};
  else
    for (int wl : {2, 8, 12, 16, 23, 24})
      for (int fl : {-100, -20, 0, 3, 8, 16, 40, 120, 125, 126, 127, 130})
        for (int kind = 0; kind < 3; ++kind)
        {
          const bool is_signed = kind != 2, symmetric = kind == 1;
          const int mag = wl - (is_signed ? 1 : 0);
          if (mag - fl > 127 || -fl < -149 || (is_signed && wl < 2) || mag < 1)
            continue;
          fmts.push_back({wl, fl, is_signed, symmetric});
        }
  const RoundMode modes[] = {RoundMode::RNE, RoundMode::RNA, RoundMode::RU,
                             RoundMode::RD,  RoundMode::RZ,  RoundMode::RO};
  const char *names[] = {"RNE", "RNA", "RU", "RD", "RZ", "RO"};
  int cases = 0, bad = 0;
  unsigned long long control_ties = 0, control_sides = 0;
  for (const Fmt &f : fmts)
  {
    const auto p = make_fixedpoint_params<float>(f.wl, f.fl, f.is_signed, f.symmetric);
    const auto c = mptorch::gemm::make_fixedpoint_word<float>(p, f.is_signed);
    if (control)
    {
      control_ties += run_det<RoundMode::RNE, RoundMode::RNA>(c, f.is_signed, p).fast_vs_bitwise;
      control_sides += run_det<RoundMode::RU, RoundMode::RD>(c, f.is_signed, p).fast_vs_bitwise;
      cases += 2;
      continue;
    }
    for (int m = 0; m < 6; ++m)
    {
      const Counts r = run_det_mode(modes[m], c, f.is_signed, p);
      ++cases;
      if (r.fast_vs_bitwise)
      {
        ++bad;
        std::printf("wl=%2d fl=%4d %-9s %-3s: %llu mismatches (one at word 0x%08x)\n", f.wl, f.fl,
                    f.is_signed ? (f.symmetric ? "symmetric" : "signed") : "unsigned", names[m],
                    r.fast_vs_bitwise, r.first_bad);
      }
    }
  }
  if (control)
  {
    std::printf("%d cases: the line's RNE against the shipped RNA %llu mismatches, RU against RD "
                "%llu (the control: both must be nonzero)\n",
                cases, control_ties, control_sides);
    return control_ties != 0 && control_sides != 0 ? 0 : 1;
  }
  std::printf("%zu formats, %d cases (format, mode); %d cases mismatch\n", fmts.size(), cases, bad);
  return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
  if (argc > 1 && !std::strncmp(argv[1], "sr", 2))
    return sr_main(argv[1]);
  if (argc > 1 && !std::strncmp(argv[1], "det", 3))
    return det_main(argv[1]);
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
