// Every cast, every rounding mode, against dev/benchmarks/sweeps/reference_casts.h
// over all 2^32 float inputs.
//
// The three gemm_cast_*_arith.cu sweeps each check one cast -- the one with a
// float-arithmetic fast path -- and skip a configuration whose gate is off,
// because a path that is never entered has nothing to verify. That leaves the
// integer paths of the other five rounding modes, and of every format the
// gates exclude, checked only by the Python suite's sampled inputs. This
// program is the exhaustive half for all of them: one pass over the 2^32 bit
// patterns per configuration, computing the shipped cast and the reference
// cast for each of the nine rounding modes and counting
//
//   * mismatches, per mode, with the numerically smallest input that shows one;
//   * -0.0 results, per mode and per side. P3109's zero is code point 0 and it
//     is unsigned -- in a signed binaryK the encoding that would hold -0.0 is
//     the format's NaN -- and superfp has no -0.0 either, so neither side may
//     ever return one. The shipped casts drop the sign where a zero is made,
//     inside the clips and the region arms; the reference drops it on the
//     value at its return. Two different rules, compared here from opposite
//     ends.
//
// Fixed point (cast_fixedpoint.h) is a third family with a section of its own
// below: `short`, `short-host`, `host` and `sr-mean` each run its rows too, and
// `fixed` / `fixed-host` run them alone (a change to cast_fixedpoint.h only).
//
// Stochastic rounding gets the same random word on both sides, a hash of the
// input pattern: SR's business is which of two neighbours it picks, and that
// is a function of (value, random word) that both spellings must agree on.
//
// Build & run (from the repo root):
//   nvcc -O3 -std=c++17 -arch=sm_89 --expt-relaxed-constexpr -I mptorch/csrc \
//        -I dev/benchmarks/sweeps dev/benchmarks/sweeps/cast_all_modes_sweep.cu \
//        -o /tmp/cast_all_modes_sweep && /tmp/cast_all_modes_sweep spot
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc -I dev/benchmarks/sweeps -x c++ dev/benchmarks/sweeps/cast_all_modes_sweep.cu \
//       -o /tmp/cast_all_modes_sweep_host && /tmp/cast_all_modes_sweep_host host
// Those host flags are setup.py's, which is what makes the host run a check of
// the build that ships. Pass `short` (~40 edge and middle formats, the gate for
// a change; `short-host` for the host build), `spot` (the grid below),
// `spot-pairs` (its axes covered pairwise, 96, for `spot`'s record in a
// quarter of the time beside `short`), `sweep`
// (that grid over every man_bits), `quick` (its two end man_bits) or `host`;
// the image64 modes further down have the same split. `sr-mean` is a different
// check -- SR's exact distribution against P3109, not against the reference --
// and takes a fraction of a second on either build.
//
// `host` is not a thinner grid, it is a different question. A configuration
// costs ~100 s of 16 host threads against ~2 s of a GPU, which makes the grid
// a four-hour run there. What varies with the format -- which arm an input
// takes, what the clips return -- is integer arithmetic, identical on both
// backends, and the GPU grid settles it. What is open on the host is g++'s
// codegen for that arithmetic and for the fast paths it compiles (binaryK's
// RNE, and superfp's RNE in a spelling of the host's own, which the GPU grid
// does not run), and nothing about that varies with man_bits. So `host` is
// thirteen configurations chosen to enter every arm, each still over all 2^32
// inputs.
//
// Since cf69807 added SR_B and SR_C (9 modes, the SR references written from
// P3109's definition) a full host configuration costs ~18 minutes of 16
// threads, so `host` is a six-and-a-half-hour run; `host-sampled` runs its 22
// configurations on every high half of the word and one low half in 60, with
// the low halves where ties, sticky bits and carries sit, in 6.4 minutes.
//
// man_bits stops at 23, the finest grid binary32 has. Above it the two
// spellings genuinely disagree and neither is this program's business:
// make_normal_range_params clamps the top of the range to binary32's grid
// (`const int man = (man_bits < 23) ? man_bits : 23`), while the reference
// shifts a code point by `23 - man_bits`, which goes negative. Raising the
// bound reports ~1e9 mismatches per configuration at man_bits 24 for that
// reason alone, on any tree; the formats a binary32 kernel can simulate all
// lie at or below 23.
//
// The configurations are chosen for the arms, not for coverage of the format
// space -- the formats have their own sweeps. What has to be reached here is
// every branch a zero can come out of: the subnormal clip's underflow and
// fall-through, the normal clip's underflow with and without its offset, the
// superfp supernormal arm with and without a float32 subnormal in it, and the
// directed modes' negation of a magnitude that rounded away. That is what the
// odd biases below are for: bias 125/127/128 at man_bits 3 put the two clips'
// min_exponent_store at 0, 1 and negative, which is where a rounded word can
// have a sign and no magnitude.
#include "common/cast_binaryK.h"
#include "common/cast_fixedpoint.h"
#include "common/cast_superfp.h"
#include "common/modes.h"
#include "old_sr_casts.h"
#include "reference_casts.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if defined(__CUDACC__)
#define CHECK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){printf("ERR %s %d\n",cudaGetErrorString(e_),__LINE__);exit(1);} }while(0)
#endif

constexpr int NMODE = 9;
static const char *MODE_NAME[NMODE] = {"RNE", "RNA", "RU", "RD", "RZ", "RO", "SR", "SRB", "SRC"};

struct Counts
{
    unsigned long long bad[NMODE];
    unsigned long long nzero_shipped[NMODE];
    unsigned long long nzero_reference[NMODE];
    unsigned first_bad[NMODE];
};

// One configuration, either family. `is_binaryK` picks which half of the
// fields is live; passing the struct whole keeps one kernel signature.
struct Cfg
{
    bool is_binaryK;
    int man_bits, exp_bits, bias;
    int normal_binades; // superfp only
    bool is_signed;
    SaturationMode sat;
    SubnormalsMode subm; // binaryK only
    int rand_bits;       // SR
};

CUDA_HOST_DEVICE_INLINE uint32_t bits_of(float f) { return FLOAT_TO_BITS(&f); }

// A random word per input, the same on both sides. splitmix32's finalizer.
CUDA_HOST_DEVICE_INLINE uint32_t rand_for(uint32_t bits)
{
    uint32_t z = bits + 0x9E3779B9u;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

// The shipped casts for either carrier -- binary32 for the sweep proper, both
// for the image sweep at the end of this file.
template <class T>
CUDA_HOST_DEVICE_INLINE T shipped_cast(int mode, T x, typename FloatTraits<T>::word_t rnd, const Cfg &c,
                                       const BinaryKParamsT<T> &bp, const SuperfpParamsT<T> &sp)
{
    if (c.is_binaryK)
        switch (mode)
        {
        case 0: return cast_binaryK_nearest_even(x, c.is_signed, c.subm, bp);
        case 1: return cast_binaryK_nearest_away(x, c.is_signed, c.subm, bp);
        case 2: return cast_binaryK_up(x, c.is_signed, c.subm, bp);
        case 3: return cast_binaryK_down(x, c.is_signed, c.subm, bp);
        case 4: return cast_binaryK_zero(x, c.is_signed, c.subm, bp);
        case 5: return cast_binaryK_odd(x, c.is_signed, c.subm, bp);
        default: return cast_binaryK_stochastic(x, rnd, sr_spec(c.rand_bits, mode - 6), c.is_signed, c.subm, bp);
        }
    switch (mode)
    {
    case 0: return cast_superfp_nearest_even(x, c.is_signed, sp);
    case 1: return cast_superfp_nearest_away(x, c.is_signed, sp);
    case 2: return cast_superfp_up(x, c.is_signed, sp);
    case 3: return cast_superfp_down(x, c.is_signed, sp);
    case 4: return cast_superfp_zero(x, c.is_signed, sp);
    case 5: return cast_superfp_odd(x, c.is_signed, sp);
    default: return cast_superfp_stochastic(x, rnd, sr_spec(c.rand_bits, mode - 6), c.is_signed, sp);
    }
}

CUDA_HOST_DEVICE_INLINE float reference_cast(int mode, float x, uint32_t rnd, const Cfg &c)
{
    const int m = c.man_bits, e = c.exp_bits, b = c.bias, nb = c.normal_binades;
    if (c.is_binaryK)
        switch (mode)
        {
        case 0: return cast_binaryK_nearest_even(x, m, e, b, c.is_signed, c.sat, c.subm);
        case 1: return cast_binaryK_nearest_away(x, m, e, b, c.is_signed, c.sat, c.subm);
        case 2: return cast_binaryK_up(x, m, e, b, c.is_signed, c.sat, c.subm);
        case 3: return cast_binaryK_down(x, m, e, b, c.is_signed, c.sat, c.subm);
        case 4: return cast_binaryK_zero(x, m, e, b, c.is_signed, c.sat, c.subm);
        case 5: return cast_binaryK_odd(x, m, e, b, c.is_signed, c.sat, c.subm);
        default: return cast_binaryK_stochastic(x, rnd, sr_spec(c.rand_bits, mode - 6), m, e, b, c.is_signed, c.sat, c.subm);
        }
    switch (mode)
    {
    case 0: return cast_superfp_nearest_even(x, m, e, nb, b, c.is_signed, c.sat);
    case 1: return cast_superfp_nearest_away(x, m, e, nb, b, c.is_signed, c.sat);
    case 2: return cast_superfp_up(x, m, e, nb, b, c.is_signed, c.sat);
    case 3: return cast_superfp_down(x, m, e, nb, b, c.is_signed, c.sat);
    case 4: return cast_superfp_zero(x, m, e, nb, b, c.is_signed, c.sat);
    case 5: return cast_superfp_odd(x, m, e, nb, b, c.is_signed, c.sat);
    default: return cast_superfp_stochastic(x, rnd, sr_spec(c.rand_bits, mode - 6), m, e, nb, b, c.is_signed, c.sat);
    }
}

// The body both backends run for one input pattern.
CUDA_HOST_DEVICE_INLINE void one_input(uint32_t bits, const Cfg &c, const BinaryKParams &bp,
                                       const SuperfpParams &sp, unsigned long long *bad,
                                       unsigned long long *nzs, unsigned long long *nzr,
                                       uint32_t *first)
{
    float x = bits_to_float(bits);
    uint32_t rnd = rand_for(bits);
    for (int mode = 0; mode < NMODE; ++mode)
    {
        uint32_t s = bits_of(shipped_cast(mode, x, rnd, c, bp, sp));
        uint32_t r = bits_of(reference_cast(mode, x, rnd, c));
        // NaN payloads are not part of the contract on either side, but the
        // pattern is preserved by both, so compare words and let a NaN that
        // differs show up rather than hiding the class.
        if (s != r) { ++bad[mode]; if (bits < first[mode]) first[mode] = bits; }
        if (s == 0x80000000u) ++nzs[mode];
        if (r == 0x80000000u) ++nzr[mode];
    }
}

#if defined(__CUDACC__)
constexpr int PER_THREAD = 64;

__global__ void exhaustive(Cfg c, BinaryKParams bp, SuperfpParams sp, Counts *out)
{
    uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
    unsigned long long bad[NMODE] = {0}, nzs[NMODE] = {0}, nzr[NMODE] = {0};
    uint32_t first[NMODE];
    for (int m = 0; m < NMODE; ++m) first[m] = 0xFFFFFFFFu;
    for (int k = 0; k < PER_THREAD; ++k)
        one_input((uint32_t)(base + k), c, bp, sp, bad, nzs, nzr, first);
    for (int m = 0; m < NMODE; ++m)
    {
        if (bad[m]) { atomicAdd(&out->bad[m], bad[m]); atomicMin(&out->first_bad[m], first[m]); }
        if (nzs[m]) atomicAdd(&out->nzero_shipped[m], nzs[m]);
        if (nzr[m]) atomicAdd(&out->nzero_reference[m], nzr[m]);
    }
}

static void run_exhaustive(const Cfg &c, const BinaryKParams &bp, const SuperfpParams &sp, Counts *h)
{
    static Counts *dc = nullptr;
    if (!dc) CHECK(cudaMalloc(&dc, sizeof(Counts)));
    Counts zero;
    memset(&zero, 0, sizeof zero);
    for (int m = 0; m < NMODE; ++m) zero.first_bad[m] = 0xFFFFFFFFu; // atomicMin needs a top
    CHECK(cudaMemcpy(dc, &zero, sizeof zero, cudaMemcpyHostToDevice));
    exhaustive<<<(unsigned)((1ull << 32) / (256 * PER_THREAD)), 256>>>(c, bp, sp, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
}
#else
// `host-sampled` (and `host`'s other sampled callers): every high half of the
// word (sign, exponent, the top seven mantissa bits) with every
// host_lo_stride-th low half, plus the low halves where ties, sticky bits and
// carries sit (0-3, 0x7FFF-0x8001, 0xFFFC-0xFFFF). 1 is every input.
static int host_lo_stride = 1;

static void run_exhaustive(const Cfg &c, const BinaryKParams &bp, const SuperfpParams &sp, Counts *h)
{
    unsigned long long bad[NMODE] = {0}, nzs[NMODE] = {0}, nzr[NMODE] = {0};
    uint32_t first[NMODE];
    for (int m = 0; m < NMODE; ++m) first[m] = 0xFFFFFFFFu;
    std::vector<uint32_t> los;
    for (uint32_t lo = 0; lo < (1u << 16); ++lo)
        if (lo % (uint32_t)host_lo_stride == 0 || lo <= 3 || (lo >= 0x7FFF && lo <= 0x8001) || lo >= 0xFFFC)
            los.push_back(lo);
#pragma omp parallel
    {
        unsigned long long b[NMODE] = {0}, s[NMODE] = {0}, r[NMODE] = {0};
        uint32_t f[NMODE];
        for (int m = 0; m < NMODE; ++m) f[m] = 0xFFFFFFFFu;
#pragma omp for schedule(static)
        for (int64_t hi = 0; hi < (1 << 16); ++hi)
            for (uint32_t lo : los)
                one_input((uint32_t)(((uint64_t)hi << 16) | lo), c, bp, sp, b, s, r, f);
#pragma omp critical
        for (int m = 0; m < NMODE; ++m)
        {
            bad[m] += b[m]; nzs[m] += s[m]; nzr[m] += r[m];
            if (f[m] < first[m]) first[m] = f[m];
        }
    }
    for (int m = 0; m < NMODE; ++m)
    {
        h->bad[m] = bad[m]; h->nzero_shipped[m] = nzs[m]; h->nzero_reference[m] = nzr[m];
        h->first_bad[m] = first[m];
    }
}
#endif

static const char *sat_name(SaturationMode s)
{
    return s == SaturationMode::SAT_NONE ? "SAT_NONE"
         : s == SaturationMode::SAT_FINITE ? "SAT_FINITE"
         : s == SaturationMode::SAT_FINITE_EXTENDED ? "SAT_FINITE_EXTENDED" : "SAT_PROPAGATE";
}
static const char *sub_name(SubnormalsMode s)
{
    return s == SubnormalsMode::SUBNORMALS ? "SUBNORMALS"
         : s == SubnormalsMode::NORMALS ? "NORMALS" : "EXTENDED_NORMALS";
}

static int failures = 0, checked = 0;

static void run_case(const Cfg &c)
{
    BinaryKParams bp = make_binaryK_params(c.man_bits, c.exp_bits, c.bias, c.is_signed, c.sat,
                                           c.subm == SubnormalsMode::EXTENDED_NORMALS);
    SuperfpParams sp = make_superfp_params(c.man_bits, c.exp_bits, c.normal_binades, c.bias, c.sat);
    Counts h;
    memset(&h, 0, sizeof(h));
    run_exhaustive(c, bp, sp, &h);
    ++checked;

    char head[160];
    if (c.is_binaryK)
        snprintf(head, sizeof head, "binaryK m%-2d e%d b%-4d %s %-13s %-16s", c.man_bits, c.exp_bits,
                 c.bias, c.is_signed ? "signed  " : "unsigned", sat_name(c.sat), sub_name(c.subm));
    else
        snprintf(head, sizeof head, "superfp m%-2d e%d n%-3d b%-4d %s %-13s", c.man_bits, c.exp_bits,
                 c.normal_binades, c.bias, c.is_signed ? "signed  " : "unsigned", sat_name(c.sat));

    bool ok = true;
    for (int m = 0; m < NMODE; ++m)
        if (h.bad[m] || h.nzero_shipped[m] || h.nzero_reference[m]) ok = false;
    if (ok)
        return;
    ++failures;
    printf("  %s\n", head);
    for (int m = 0; m < NMODE; ++m)
    {
        if (!h.bad[m] && !h.nzero_shipped[m] && !h.nzero_reference[m]) continue;
        printf("      %-3s  mismatches %-12llu  -0.0 shipped %-10llu  reference %-10llu", MODE_NAME[m],
               h.bad[m], h.nzero_shipped[m], h.nzero_reference[m]);
        if (h.bad[m])
            printf("  e.g. 0x%08x = %g", h.first_bad[m], bits_to_float(h.first_bad[m]));
        printf("\n");
    }
}

// ===========================================================================
// Fixed point: cast_fixedpoint.h against reference_casts.h's ref_fixedpoint,
// every binary32 input in every mode, in a section of its own -- its format
// has none of the fields Cfg carries, and its cast takes a FixedPointParams
// where the other two take theirs. There is no image64 half: the cast rounds
// on the word, nothing in it is carrier arithmetic, so the binary64 cast is
// checked against the same reference by cast_double_sweep.cu instead.
// ===========================================================================
struct FxCfg
{
    int wl, fl;
    bool is_signed, symmetric;
    int rand_bits; // SR's N; mag_bits + N <= 23 is the carrier check
};

CUDA_HOST_DEVICE_INLINE float fx_shipped(int mode, float x, uint32_t rnd, const FxCfg &c, const FixedPointParams &p)
{
    switch (mode)
    {
    case 0: return cast_fixedpoint_nearest_even(x, c.is_signed, p);
    case 1: return cast_fixedpoint_nearest_away(x, c.is_signed, p);
    case 2: return cast_fixedpoint_up(x, c.is_signed, p);
    case 3: return cast_fixedpoint_down(x, c.is_signed, p);
    case 4: return cast_fixedpoint_zero(x, c.is_signed, p);
    case 5: return cast_fixedpoint_odd(x, c.is_signed, p);
    default: return cast_fixedpoint_stochastic(x, rnd, sr_spec(c.rand_bits, mode - 6), c.is_signed, p);
    }
}

CUDA_HOST_DEVICE_INLINE float fx_reference(int mode, float x, uint32_t rnd, const FxCfg &c)
{
    switch (mode)
    {
    case 0: return cast_fixedpoint_nearest_even(x, c.wl, c.fl, c.is_signed, c.symmetric);
    case 1: return cast_fixedpoint_nearest_away(x, c.wl, c.fl, c.is_signed, c.symmetric);
    case 2: return cast_fixedpoint_up(x, c.wl, c.fl, c.is_signed, c.symmetric);
    case 3: return cast_fixedpoint_down(x, c.wl, c.fl, c.is_signed, c.symmetric);
    case 4: return cast_fixedpoint_zero(x, c.wl, c.fl, c.is_signed, c.symmetric);
    case 5: return cast_fixedpoint_odd(x, c.wl, c.fl, c.is_signed, c.symmetric);
    default: return ref_fixedpoint(x, c.wl, c.fl, c.is_signed, c.symmetric, (RoundMode)mode, rnd, c.rand_bits);
    }
}

CUDA_HOST_DEVICE_INLINE void fx_one_input(uint32_t bits, const FxCfg &c, const FixedPointParams &p,
                                          unsigned long long *bad, unsigned long long *nzs,
                                          unsigned long long *nzr, uint32_t *first)
{
    float x = bits_to_float(bits);
    uint32_t rnd = rand_for(bits);
    for (int mode = 0; mode < NMODE; ++mode)
    {
        uint32_t s = bits_of(fx_shipped(mode, x, rnd, c, p));
        uint32_t r = bits_of(fx_reference(mode, x, rnd, c));
        if (s != r) { ++bad[mode]; if (bits < first[mode]) first[mode] = bits; }
        if (s == 0x80000000u) ++nzs[mode];
        if (r == 0x80000000u) ++nzr[mode];
    }
}

#if defined(__CUDACC__)
__global__ void fx_exhaustive(FxCfg c, FixedPointParams p, Counts *out)
{
    uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
    unsigned long long bad[NMODE] = {0}, nzs[NMODE] = {0}, nzr[NMODE] = {0};
    uint32_t first[NMODE];
    for (int m = 0; m < NMODE; ++m) first[m] = 0xFFFFFFFFu;
    for (int k = 0; k < PER_THREAD; ++k)
        fx_one_input((uint32_t)(base + k), c, p, bad, nzs, nzr, first);
    for (int m = 0; m < NMODE; ++m)
    {
        if (bad[m]) { atomicAdd(&out->bad[m], bad[m]); atomicMin(&out->first_bad[m], first[m]); }
        if (nzs[m]) atomicAdd(&out->nzero_shipped[m], nzs[m]);
        if (nzr[m]) atomicAdd(&out->nzero_reference[m], nzr[m]);
    }
}

static void fx_run_exhaustive(const FxCfg &c, const FixedPointParams &p, Counts *h)
{
    static Counts *dc = nullptr;
    if (!dc) CHECK(cudaMalloc(&dc, sizeof(Counts)));
    Counts zero;
    memset(&zero, 0, sizeof zero);
    for (int m = 0; m < NMODE; ++m) zero.first_bad[m] = 0xFFFFFFFFu;
    CHECK(cudaMemcpy(dc, &zero, sizeof zero, cudaMemcpyHostToDevice));
    fx_exhaustive<<<(unsigned)((1ull << 32) / (256 * PER_THREAD)), 256>>>(c, p, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
}
#else
static void fx_run_exhaustive(const FxCfg &c, const FixedPointParams &p, Counts *h)
{
    unsigned long long bad[NMODE] = {0}, nzs[NMODE] = {0}, nzr[NMODE] = {0};
    uint32_t first[NMODE];
    for (int m = 0; m < NMODE; ++m) first[m] = 0xFFFFFFFFu;
#pragma omp parallel
    {
        unsigned long long b[NMODE] = {0}, s[NMODE] = {0}, r[NMODE] = {0};
        uint32_t f[NMODE];
        for (int m = 0; m < NMODE; ++m) f[m] = 0xFFFFFFFFu;
#pragma omp for schedule(static)
        for (int64_t hi = 0; hi < (1 << 16); ++hi)
            for (uint32_t lo = 0; lo < (1u << 16); ++lo)
                fx_one_input((uint32_t)(((uint64_t)hi << 16) | lo), c, p, b, s, r, f);
#pragma omp critical
        for (int m = 0; m < NMODE; ++m)
        {
            bad[m] += b[m]; nzs[m] += s[m]; nzr[m] += r[m];
            if (f[m] < first[m]) first[m] = f[m];
        }
    }
    for (int m = 0; m < NMODE; ++m)
    {
        h->bad[m] = bad[m]; h->nzero_shipped[m] = nzs[m]; h->nzero_reference[m] = nzr[m];
        h->first_bad[m] = first[m];
    }
}
#endif

static void fx_head(const FxCfg &c, char *head, size_t n)
{
    snprintf(head, n, "fixed   wl%-2d fl%-5d %s rand%-2d", c.wl, c.fl,
             !c.is_signed ? "unsigned " : c.symmetric ? "symmetric" : "signed   ", c.rand_bits);
}

static void fx_run_case(const FxCfg &c)
{
    const FixedPointParams p = make_fixedpoint_params(c.wl, c.fl, c.is_signed, c.symmetric);
    Counts h;
    memset(&h, 0, sizeof(h));
    fx_run_exhaustive(c, p, &h);
    ++checked;
    bool ok = true;
    for (int m = 0; m < NMODE; ++m)
        if (h.bad[m] || h.nzero_shipped[m] || h.nzero_reference[m]) ok = false;
    if (ok)
        return;
    ++failures;
    char head[160];
    fx_head(c, head, sizeof head);
    printf("  %s\n", head);
    for (int m = 0; m < NMODE; ++m)
    {
        if (!h.bad[m] && !h.nzero_shipped[m] && !h.nzero_reference[m]) continue;
        printf("      %-3s  mismatches %-12llu  -0.0 shipped %-10llu  reference %-10llu", MODE_NAME[m],
               h.bad[m], h.nzero_shipped[m], h.nzero_reference[m]);
        if (h.bad[m])
        {
            const float x = bits_to_float(h.first_bad[m]);
            const uint32_t rnd = rand_for(h.first_bad[m]);
            printf("  e.g. 0x%08x = %g: shipped %g, reference %g", h.first_bad[m], x,
                   fx_shipped(m, x, rnd, c, p), fx_reference(m, x, rnd, c));
        }
        printf("\n");
    }
}

// The fixed-point edges, for `short`: each end of the word (wl 1 unsigned, 2
// signed, mag_bits 23 at the carrier's mantissa), the three ranges, fl
// negative, zero, inside the word and past it, the step at 2^-125 (the last
// one inside the carrier check's warning) and the top at 2^127 (the largest
// mag_bits - fl it admits), and SR with no random bits and with all of them.
static void fx_short_list()
{
    const int before = checked, before_bad = failures;
    printf("short, fixed point -- 2^32 inputs x 9 rounding modes per configuration\n");
    const FxCfg rows[] = {
        {4, 2, true, false, 20},   {4, 2, true, true, 20},   {4, 2, false, false, 19},
        {2, 0, true, false, 22},   {1, 0, false, false, 0},  {1, 3, false, false, 22},
        {8, 6, true, true, 16},    {8, 6, true, false, 0},   {8, 0, true, false, 16},
        {8, -3, true, false, 12},  {8, 12, true, true, 16},  {16, 8, true, false, 8},
        {16, 15, false, false, 7}, {24, 20, true, false, 0}, {24, 0, true, true, 0},
        {23, 22, false, false, 0}, {22, 30, true, false, 2}, {8, 125, true, false, 16},
        {8, 125, false, false, 15}, {8, -120, true, false, 16}, {8, -120, true, true, 16},
        {16, -111, false, false, 7},
    };
    for (const FxCfg &c : rows) fx_run_case(c);
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before, failures - before_bad);
}

// Formats past mptorch.number's fixed-point checks as they stood when SR was
// rewritten (2026-11-09): mag_bits 24, mag_bits + rand_bits 24, and a top at
// 2^127 with no two's complement end below it -- bounds only the old SR's
// shift needed. For the decision whether to relax those checks: `fixed-beyond`.
static void fx_beyond_list()
{
    const int before = checked, before_bad = failures;
    printf("beyond the checks, fixed point -- 2^32 inputs x 9 rounding modes per configuration\n");
    const FxCfg rows[] = {
        {25, 20, true, false, 0}, {24, 20, false, false, 0}, {25, 0, true, true, 0},
        {24, 22, true, false, 1}, {16, 8, true, false, 9},   {8, -121, true, true, 16},
        {7, -121, false, false, 17},
    };
    for (const FxCfg &c : rows) fx_run_case(c);
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before, failures - before_bad);
}

// The host's codegen of the same casts: one configuration per arm -- the hand
// table's format, a symmetric one, SR at the mantissa's edge, an unsigned one,
// and the step at 2^-125.
static void fx_host_list()
{
    const int before = checked, before_bad = failures;
    printf("host, fixed point -- 2^32 inputs x 9 rounding modes per configuration\n");
    const FxCfg rows[] = {
        {4, 2, true, false, 20}, {8, 6, true, true, 16}, {24, 20, true, false, 0},
        {16, 15, false, false, 7}, {8, 125, true, false, 16},
    };
    for (const FxCfg &c : rows) fx_run_case(c);
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before, failures - before_bad);
}

// SR's distribution, exactly: every draw R of N bits for inputs around the
// draw's own levels, where P3109's StochasticA rounds up iff floor(eta 2^N) +
// R >= 2^N, so the count of draws that go up must be floor(eta 2^N) itself,
// not merely close to it. eta = (|x| - lo) / step is exact in double for every
// binary32 x (lo is 0 or at least one step, which bounds |x| - lo to 2^23 of
// x's ulps). A cast that rounds |x| in the carrier before its draw -- the
// float add of binaryK's subnormal arm, which this cast used to make -- fails
// here at the mantissa's edge, which is what the negative control shows.
struct FxSrResult
{
    int inputs, off_grid, wrong;
    float worst_x;
    long long worst_up, worst_expect;
};

static FxSrResult fx_sr_case(const FxCfg &c)
{
    const FixedPointParams p = make_fixedpoint_params(c.wl, c.fl, c.is_signed, c.symmetric);
    const int N = c.rand_bits;
    const double step = ldexp(1.0, -c.fl);
    const int mag_bits = c.wl - (c.is_signed ? 1 : 0);
    const double top = ldexp((double)((1ull << mag_bits) - 1), -c.fl);

    std::vector<double> xs;
    const double codes[] = {0.0, 1.0, 3.0, (double)((1ull << mag_bits) / 2), (double)((1ull << mag_bits) - 2)};
    std::vector<double> fracs = {0.25, 0.3, 0.5, 0.75, 0.999};
    for (int k : {1, 2, 3})
        if (N >= 2) fracs.push_back(ldexp((double)k, -N));
    if (N >= 1) fracs.push_back(1.0 - ldexp(1.0, -N));
    for (double j : codes)
        for (double f : fracs)
        {
            const float x = (float)((j + f) * step);
            xs.push_back(x);
            xs.push_back(nextafterf(x, 0.0f));
            xs.push_back(nextafterf(x, 3.0e38f));
        }
    for (int e : {-1, -9, -20, -30}) xs.push_back((float)(ldexp(3.0, e - 2) * step)); // under the step

    FxSrResult r{0, 0, 0, 0.0f, 0, 0};
    long long worst = -1;
    // The draw's top N bits, which is where cast_fixedpoint_stochastic reads
    // them. The negative control (-DFX_DRAW_BELOW_MAG_BITS) puts them where
    // the cast before 2026-11-09 read them, directly below its kept mag_bits
    // bits, so that what it shows is that cast's rounding and not the move.
#if defined(FX_DRAW_BELOW_MAG_BITS)
    const int shift = 23 - mag_bits - N;
#else
    const int shift = 23 - N;
#endif
    for (double xd0 : xs)
        for (double sgn : {1.0, -1.0})
        {
            if (!c.is_signed && sgn < 0) continue;
            const float x = (float)(sgn * xd0);
            const double ax = fabs((double)x);
            if (x == 0.0f || !(ax < top)) continue; // the top saturates: another test's
            const double lo = floor(ax / step) * step;
            const double eta = (ax - lo) / step; // exact
            const double X = ldexp(eta, N);
            // the draws StochasticA, B and C send up: floor(X), floor(X + 1/2), RNITE(X)
            const long long expects[3] = {(long long)floor(X), (long long)floor(X + 0.5), (long long)nearbyint(X)};
            for (int v = 0; v < 3; ++v)
            {
                const long long expect = expects[v];
                long long up = 0, stray = 0;
                for (uint32_t k = 0; k < (1u << N); ++k)
                {
                    const double y = fabs((double)fx_shipped(6 + v, x, k << shift, c, p));
                    if (y == lo + step && eta != 0) ++up;
                    else if (y != lo) ++stray;
                }
                ++r.inputs;
                if (stray) { ++r.off_grid; continue; }
                const long long err = up > expect ? up - expect : expect - up;
                if (err) ++r.wrong;
                if (err > worst) worst = err, r.worst_x = x, r.worst_up = up, r.worst_expect = expect;
            }
        }
    return r;
}

static void fx_sr_mean()
{
    printf("sr-mean, fixed point -- every draw; the counts up must be floor(X), floor(X + 1/2), RNITE(X) exactly\n");
    const FxCfg rows[] = {
        {24, 20, true, false, 0}, {23, 20, true, false, 1}, {22, 20, true, false, 2}, {23, 22, false, false, 0},
        {4, 2, true, false, 12},  {8, 6, true, true, 12},   {16, 14, false, false, 7}, {8, -3, true, false, 8},
        {12, 30, true, false, 10}, {8, 125, true, false, 12},
    };
    for (const FxCfg &c : rows)
    {
        const FxSrResult r = fx_sr_case(c);
        ++checked;
        char head[160];
        fx_head(c, head, sizeof head);
        const bool ok = !r.wrong && !r.off_grid;
        printf("  %s  %-4s %d inputs", head, ok ? "ok" : "FAIL", r.inputs);
        if (!ok)
            printf("  [%d wrong, %d off the neighbours; e.g. %g: %lld of 2^%d draws up, %lld expected]", r.wrong,
                   r.off_grid, r.worst_x, r.worst_up, c.rand_bits, r.worst_expect);
        printf("\n");
        if (!ok) ++failures;
    }
}

// The grid, for the GPU. `stride` thins the man_bits axis; see host_list
// below for why the host runs a list instead.
// The host list: every arm, once. Three subnormals modes and three saturation
// modes over binaryK 8p4 at two biases -- 7, the OCP one, and 125, which puts
// the subnormal clip's min_exponent_store at zero and so reaches the
// fall-through a -0.0 word used to take -- plus man_bits 0 for the
// zero-argument round formulas, bias 128 for a dead underflow test, and
// man_bits 10 for a wide mantissa. Then four superfp formats: one with an
// underflow region inside binary32 (m3e4n1b7), one without (m8e4n2b7), one
// with normal_binades == 2^exp_bits, and man_bits 0. Signed and unsigned each
// appear; the unsigned early return is the only thing signedness changes here.
static void host_list()
{
    const SaturationMode O = SaturationMode::SAT_NONE, F = SaturationMode::SAT_FINITE,
                         P = SaturationMode::SAT_PROPAGATE, X = SaturationMode::SAT_FINITE_EXTENDED;
    const SubnormalsMode S = SubnormalsMode::SUBNORMALS, N = SubnormalsMode::NORMALS,
                         E = SubnormalsMode::EXTENDED_NORMALS;
    printf("binaryK -- 2^32 inputs x 9 rounding modes per configuration\n");
    run_case(Cfg{true, 3, 4, 7, 1, true, O, S, 8});
    run_case(Cfg{true, 3, 4, 7, 1, true, F, N, 8});
    run_case(Cfg{true, 3, 4, 7, 1, false, P, E, 8});
    run_case(Cfg{true, 3, 8, 125, 1, true, O, S, 8});
    run_case(Cfg{true, 3, 8, 125, 1, true, O, N, 8});
    run_case(Cfg{true, 0, 4, 7, 1, true, O, S, 8});
    run_case(Cfg{true, 0, 8, 128, 1, false, F, S, 8});
    run_case(Cfg{true, 10, 5, 15, 1, true, O, S, 8});
    // SatFinite in the extended domain: the fast path (signed, SUBNORMALS),
    // the integer path's clip (NORMALS) and the unsigned format's two
    // reserved codes.
    run_case(Cfg{true, 3, 4, 7, 1, true, X, S, 8});
    run_case(Cfg{true, 3, 4, 7, 1, true, X, N, 8});
    run_case(Cfg{true, 3, 4, 7, 1, false, X, E, 8});
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked, failures);
    int before = checked, before_bad = failures;
    printf("superfp -- 2^32 inputs x 9 rounding modes per configuration\n");
    run_case(Cfg{false, 3, 4, 7, 1, true, O, S, 8});
    run_case(Cfg{false, 3, 4, 7, 1, false, F, S, 8});
    run_case(Cfg{false, 8, 4, 7, 2, true, O, S, 8});
    run_case(Cfg{false, 0, 4, 7, 1, true, P, S, 8});
    run_case(Cfg{false, 2, 4, 7, 16, true, O, S, 8});
    run_case(Cfg{false, 3, 4, 7, 1, true, X, S, 8});
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before,
           failures - before_bad);
    fx_host_list();
}

struct GridEB { int e, bias; };
// biases chosen for the clips' corners, not for the formats: at man_bits 3,
// bias 125 puts the subnormal clip's min_exponent_store at 0, bias 127 puts
// the normal clip's at 1 (its offset arm) and bias 128 puts it below zero
// (its underflow arm dead, which is where a -0.0 word used to fall through).
static const GridEB GRID_EBS[] = {{4, 7}, {5, 15}, {8, 127}, {8, 125}, {8, 128}, {3, 1}, {6, 60}, {2, -8}};
// Saturation and subnormals as a list rather than a cross product: the
// subnormals mode picks which arm a small input takes and the saturation
// mode what a large one becomes, and no arm depends on both.
struct GridMS { SaturationMode sat; SubnormalsMode sub; };
static const GridMS GRID_MS[] = {
    {SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS},
    {SaturationMode::SAT_FINITE, SubnormalsMode::SUBNORMALS},
    {SaturationMode::SAT_PROPAGATE, SubnormalsMode::SUBNORMALS},
    {SaturationMode::SAT_NONE, SubnormalsMode::NORMALS},
    {SaturationMode::SAT_NONE, SubnormalsMode::EXTENDED_NORMALS},
    {SaturationMode::SAT_FINITE, SubnormalsMode::NORMALS},
};

static void grid(int stride)
{
    using EB = GridEB;
    using MS = GridMS;
    const auto &ebs = GRID_EBS;
    const auto &ms = GRID_MS;

    printf("binaryK -- 2^32 inputs x 9 rounding modes per configuration\n");
    for (int man = 0; man <= 23; man += stride)
        for (const EB &eb : ebs)
            for (bool sgn : {true, false})
                for (const MS &m : ms)
                {
                    run_case(Cfg{true, man, eb.e, eb.bias, 1, sgn, m.sat, m.sub, 8});
                }
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked, failures);

    int before = checked, before_bad = failures;
    printf("superfp -- 2^32 inputs x 9 rounding modes per configuration\n");
    // normal_binades 1 and 2 are the ordinary cases; 2^exp_bits is the
    // degenerate one where the supernormal region is empty and the cutoffs
    // invert. man_bits 4 and up at exp_bits 4 puts supernormal_cutoff below
    // -127, which is the arm where a float32 subnormal rounds to a signed zero.
    for (int man = 0; man <= 13; man += stride)
        for (const EB &eb : ebs)
            for (int nb : {1, 2, 1 << eb.e})
                for (bool sgn : {true, false})
                    for (const SaturationMode sat : {SaturationMode::SAT_NONE,
                                                     SaturationMode::SAT_FINITE,
                                                     SaturationMode::SAT_PROPAGATE})
                    {
                        run_case(Cfg{false, man, eb.e, eb.bias, nb, sgn, sat, SubnormalsMode::SUBNORMALS, 8});
                    }
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before, failures - before_bad);
}

// `spot-pairs`: the `spot` grid's axes covered pairwise instead of crossed.
//
// `spot` crosses everything -- binaryK's mantissa widths 0/8/16 x eight
// exponent/bias shapes x six saturation/subnormals pairs x both signs, and
// superfp's widths 0/8 x the shapes x three normal_binades x both signs x
// three saturations -- 576 configurations at ~4 s each. But a defect in a cast
// is a property of an arm, and which arm an input reaches is decided by at
// most two of those axes together (a clip's corner by the bias and the
// subnormals mode, a saturated top by the bias and the saturation, and so
// on). So every value of each axis meets every value of every other axis
// here at least once, which is 48 configurations a family:
//
//   binaryK  one per (shape k, mode pair j); mantissa width (k + j) % 3 and
//            signedness (k + j) % 2, which give every (width, sign) pair
//            too, since both are functions of (k + j) % 6.
//   superfp  one per (shape k, normal_binades n, sign t); width
//            (k + n + t) % 2 and saturation (k + n + t) % 3, again every
//            (width, saturation) pair, as functions of (k + n + t) % 6.
//
// Pair coverage is not the edges, which `short` holds by name (man_bits 1, 22
// and 23, each floor on the faithful edge, the clips' stores at 0 and past
// the carrier); the two together are the trimmed record.
static void grid_pairs()
{
    const int bk_mans[] = {0, 8, 16};
    printf("binaryK, pairwise -- 2^32 inputs x 9 rounding modes per configuration\n");
    for (int k = 0; k < 8; ++k)
        for (int j = 0; j < 6; ++j)
        {
            const GridEB &eb = GRID_EBS[k];
            const GridMS &m = GRID_MS[j];
            run_case(Cfg{true, bk_mans[(k + j) % 3], eb.e, eb.bias, 1, (k + j) % 2 == 0, m.sat, m.sub, 8});
        }
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked, failures);

    int before = checked, before_bad = failures;
    const SaturationMode sats[] = {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE,
                                   SaturationMode::SAT_PROPAGATE};
    printf("superfp, pairwise -- 2^32 inputs x 9 rounding modes per configuration\n");
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 3; ++n)
            for (int t = 0; t < 2; ++t)
            {
                const GridEB &eb = GRID_EBS[k];
                const int nb = n == 0 ? 1 : n == 1 ? 2 : 1 << eb.e;
                const int s = k + n + t;
                run_case(Cfg{false, s % 2 == 0 ? 0 : 8, eb.e, eb.bias, nb, t == 0, sats[s % 3],
                             SubnormalsMode::SUBNORMALS, 8});
            }
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before, failures - before_bad);
}

// ===========================================================================
// image64: the binary64 casts, held to the binary32 ones.
//
// Every shipped cast is a template on its carrier (bit_helper.h, since
// 90beb61). This mode is the first of two checks, the other being
// cast_double_sweep.cu, written to prove the binary64 casts before anything in
// the extension computed in binary64 (the elementwise quantizers first did at
// c8677b0). A binary32 value is also a binary64 value, and for a format binary32
// can simulate faithfully the format value it rounds to does not depend on
// which carrier held the input -- so for all 2^32 float words x and all seven
// modes,
//
//     cast<double>((double)x)  ==  (double)cast<float>(x),   bit for bit.
//
// The right-hand side is the cast the sweep above verifies against the
// reference over every input, so this carries that verification across to the
// binary64 instantiation wherever both carriers are exact: every arm, the
// clips, the saturations, NaN and infinity, and man_bits == 0's tie, which
// breaks on the carrier's exponent-field parity and holds only because 1023 is
// odd like 127. Where binary32 is not faithful the two legitimately differ,
// so configurations outside number.py's bounds are skipped, not reported
// (faithful32 below transcribes them).
//
// Stochastic rounding gets the same draw on both sides: binary32 reads the low
// 23 bits of its word below the binary point and binary64 the low 52 of its
// own, so the binary32 word's 23 go into the top 23 of the binary64 one, and
// every rand_bits <= 23 masks the same bits. SR is also the one place the
// identity is not the whole truth. Two of its arms do arithmetic in the carrier
// -- the subnormal (superfp: underflow) arm adds and subtracts the smallest
// normal around the rounding, and the NORMALS/EXTENDED_NORMALS floor compares
// |x| with u * min -- and binary32 rounds those intermediates where binary64
// does not. There, and only for inputs below the region's top, a binary64
// result that differs is accepted if it is one of x's two neighbours on the
// format's grid, which are the RD and RU results this same pass computed (and
// held to the identity); every such input is counted and printed as
// "excused". cast_double_sweep.cu checks those arms exactly, against the
// reference's binary64 instantiation.
//
// The first time it ran, 2394 of 2406 faithful configurations were identical.
// The other twelve were two format families on the NORMALS/EXTENDED_NORMALS
// edge that number.py then called faithful and that binary32 was not, and in
// each the binary64 answer was P3109's:
//
//   man_bits 0, e7, bias 127 (floor 2^-126), NORMALS or EXTENDED_NORMALS, any
//     saturation or signedness. RNE of 2^-127, the tie between zero and the
//     floor, is the even zero; binary32 returns 2^-126, because the binary32
//     subnormal 0x00400000 goes through the man_bits == 0 round as if its
//     exponent field were a binade's, which makes it a tie between 2^-127 and
//     2^-126, and the round carries it to the floor before the clip can see
//     it. EXTENDED_NORMALS also failed SR: make_normal_range_params tested the
//     floor's range on min_exponent_store, which that mode lowers by a binade
//     whatever man_bits is, so at bias 127 it read 0 and left binary32's floor
//     dead, where man_bits == 0's floor is NORMALS' 2^-126.
//   man_bits 23, e7, bias 126, EXTENDED_NORMALS. RNA of 2^-127 is zero, being
//     below the half-floor 2^-127 + 2^-150; binary32 holds that half-floor
//     as 2^-127 and returns the floor.
//
// The dead floor was a bug and is fixed: the range test reads the floor's own
// exponent field. The other two are binary32 not holding the point the
// compare needs, so number.py's bound now stops those two families a binade
// short (bias 126 and 125) and faithful32 below follows it. `image64-edges`
// keeps the twelve: outside the bound, each must now differ from binary64 at
// exactly the documented input and mode and nowhere else, which is what says
// the fix took and that nothing else is wrong at those formats.
// cast_double_sweep.cu's `edges` checks the binary64 casts on the same formats.
// ===========================================================================

struct ImageCounts
{
    unsigned long long bad[NMODE];
    unsigned long long nzero[NMODE]; // -0.0 from the binary64 cast
    unsigned long long excused;      // SR, in a carrier-arithmetic region, on a neighbour
    unsigned first_bad[NMODE];
};

CUDA_HOST_DEVICE_INLINE uint64_t bits64_of(double d)
{
    uint64_t w;
    memcpy(&w, &d, sizeof w);
    return w;
}

CUDA_HOST_DEVICE_INLINE void image_input(uint32_t bits, const Cfg &c, double sr_below, const BinaryKParams &bp32,
                                         const SuperfpParams &sp32, const BinaryKParamsT<double> &bp64,
                                         const SuperfpParamsT<double> &sp64, unsigned long long *bad,
                                         unsigned long long *nzero, unsigned long long *excused, uint32_t *first)
{
    float x = bits_to_float(bits);
    double xd = x;
    uint32_t r32 = rand_for(bits);
    uint64_t r64 = (uint64_t)(r32 & 0x007FFFFFu) << 29;
    uint64_t wide[NMODE];
    for (int mode = 0; mode < NMODE; ++mode)
    {
        wide[mode] = bits64_of(shipped_cast(mode, xd, r64, c, bp64, sp64));
        uint64_t image = bits64_of((double)shipped_cast(mode, x, r32, c, bp32, sp32));
        if (wide[mode] == 0x8000000000000000u)
            ++nzero[mode];
        if (wide[mode] == image)
            continue;
        // SR, exact in both carriers since cf69807, has no excuse left: the
        // counter stays for the report line, at zero
        (void)sr_below;
        (void)excused;
        ++bad[mode];
        if (bits < first[mode])
            first[mode] = bits;
    }
}

#if defined(__CUDACC__)
__global__ void exhaustive_image(Cfg c, double sr_below, BinaryKParams bp32, SuperfpParams sp32,
                               BinaryKParamsT<double> bp64, SuperfpParamsT<double> sp64, ImageCounts *out)
{
    uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
    unsigned long long bad[NMODE] = {0}, nzero[NMODE] = {0}, excused = 0;
    uint32_t first[NMODE];
    for (int m = 0; m < NMODE; ++m) first[m] = 0xFFFFFFFFu;
    for (int k = 0; k < PER_THREAD; ++k)
        image_input((uint32_t)(base + k), c, sr_below, bp32, sp32, bp64, sp64, bad, nzero, &excused, first);
    for (int m = 0; m < NMODE; ++m)
    {
        if (bad[m]) { atomicAdd(&out->bad[m], bad[m]); atomicMin(&out->first_bad[m], first[m]); }
        if (nzero[m]) atomicAdd(&out->nzero[m], nzero[m]);
    }
    if (excused) atomicAdd(&out->excused, excused);
}

static void run_image(const Cfg &c, double sr_below, const BinaryKParams &bp32, const SuperfpParams &sp32,
                      const BinaryKParamsT<double> &bp64, const SuperfpParamsT<double> &sp64, ImageCounts *h)
{
    static ImageCounts *dc = nullptr;
    if (!dc) CHECK(cudaMalloc(&dc, sizeof(ImageCounts)));
    ImageCounts zero;
    memset(&zero, 0, sizeof zero);
    for (int m = 0; m < NMODE; ++m) zero.first_bad[m] = 0xFFFFFFFFu;
    CHECK(cudaMemcpy(dc, &zero, sizeof zero, cudaMemcpyHostToDevice));
    exhaustive_image<<<(unsigned)((1ull << 32) / (256 * PER_THREAD)), 256>>>(c, sr_below, bp32, sp32, bp64, sp64, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, dc, sizeof(ImageCounts), cudaMemcpyDeviceToHost));
}
#else
static void run_image(const Cfg &c, double sr_below, const BinaryKParams &bp32, const SuperfpParams &sp32,
                      const BinaryKParamsT<double> &bp64, const SuperfpParamsT<double> &sp64, ImageCounts *h)
{
    memset(h, 0, sizeof *h);
    for (int m = 0; m < NMODE; ++m) h->first_bad[m] = 0xFFFFFFFFu;
#pragma omp parallel
    {
        unsigned long long b[NMODE] = {0}, z[NMODE] = {0}, e = 0;
        uint32_t f[NMODE];
        for (int m = 0; m < NMODE; ++m) f[m] = 0xFFFFFFFFu;
#pragma omp for schedule(static)
        for (int64_t hi = 0; hi < (1 << 16); ++hi)
            for (uint32_t lo = 0; lo < (1u << 16); ++lo)
                image_input((uint32_t)(((uint64_t)hi << 16) | lo), c, sr_below, bp32, sp32, bp64, sp64, b, z, &e, f);
#pragma omp critical
        {
            for (int m = 0; m < NMODE; ++m)
            {
                h->bad[m] += b[m]; h->nzero[m] += z[m];
                if (f[m] < h->first_bad[m]) h->first_bad[m] = f[m];
            }
            h->excused += e;
        }
    }
}
#endif

// number.py's "faithful in binary32" (`_width_error`, `_range_findings` and
// the two `_*_extent`s), transcribed: the formats for which the identity above
// is a claim at all.
static bool faithful32(const Cfg &c)
{
    if (c.man_bits < 0 || c.man_bits > 23 || c.man_bits + c.rand_bits > 23 || c.exp_bits < 1 || c.exp_bits > 30)
        return false;
    const int binades = 1 << c.exp_bits;
    const int reserved = (c.sat != SaturationMode::SAT_FINITE) + (c.is_binaryK && !c.is_signed ? 1 : 0);
    const int codes = 1 << c.man_bits;
    int top_code = codes - 1 - reserved, top_field = binades - 1;
    while (top_code < 0)
    {
        top_code += codes;
        --top_field;
    }
    const int top_exp = top_field - c.bias;
    if (top_field < 1 || top_exp < -126 || top_exp > 127)
        return false;
    long long bottom, step, min_bottom = -125;
    if (c.is_binaryK)
    {
        const int min_exp = 1 - c.bias;
        if (c.subm == SubnormalsMode::SUBNORMALS)
            bottom = step = min_exp - c.man_bits;
        else
        {
            if (c.subm == SubnormalsMode::NORMALS)
                bottom = min_exp, step = min_exp - c.man_bits;
            else
                bottom = c.man_bits ? min_exp - 1 : min_exp, step = bottom - c.man_bits;
            // a binade short of 2^-126 where half the floor is out of reach:
            // man_bits 0 rounds it by field, and EXTENDED_NORMALS cannot hold it
            // below a 23-bit mantissa
            min_bottom = c.man_bits == 0 ? -125
                         : c.subm == SubnormalsMode::EXTENDED_NORMALS && c.man_bits - 148 > -126 ? c.man_bits - 148
                                                                                                  : -126;
        }
    }
    else
    {
        if (c.normal_binades >= binades)
            return false;
        const long long normal_cutoff = binades - c.bias - c.normal_binades;
        bottom = normal_cutoff - (long long)(binades - c.normal_binades) * (1LL << c.man_bits) + 1;
        step = bottom < normal_cutoff - c.man_bits ? bottom : normal_cutoff - c.man_bits;
    }
    return step >= -149 && bottom >= min_bottom;
}

static int image_checked = 0, image_failures = 0, image_skipped = 0;
static unsigned long long image_excused = 0;

static void image_run(const Cfg &c, ImageCounts *h)
{
    const bool ext = c.subm == SubnormalsMode::EXTENDED_NORMALS;
    BinaryKParams bp32 = make_binaryK_params(c.man_bits, c.exp_bits, c.bias, c.is_signed, c.sat, ext);
    SuperfpParams sp32 = make_superfp_params(c.man_bits, c.exp_bits, c.normal_binades, c.bias, c.sat);
    BinaryKParamsT<double> bp64 = make_binaryK_params<double>(c.man_bits, c.exp_bits, c.bias, c.is_signed, c.sat, ext);
    SuperfpParamsT<double> sp64 = make_superfp_params<double>(c.man_bits, c.exp_bits, c.normal_binades, c.bias, c.sat);

    // the top of SR's carrier-arithmetic region, as a magnitude
    double sr_below;
    if (!c.is_binaryK)
        sr_below = ldexp(1.0, sp64.supernormal_cutoff < -1100 ? -1100 : sp64.supernormal_cutoff);
    else if (c.subm == SubnormalsMode::SUBNORMALS)
        sr_below = ldexp(1.0, 1 - c.bias);
    else
        sr_below = bits_to_float(bp32.min_num); // the floor, whose compare is the arithmetic

    run_image(c, sr_below, bp32, sp32, bp64, sp64, h);
}

static void image_head(const Cfg &c, char *head, size_t n)
{
    if (c.is_binaryK)
        snprintf(head, n, "binaryK m%-2d e%d b%-4d %s %-13s %-16s r%d", c.man_bits, c.exp_bits, c.bias,
                 c.is_signed ? "signed  " : "unsigned", sat_name(c.sat), sub_name(c.subm), c.rand_bits);
    else
        snprintf(head, n, "superfp m%-2d e%d n%-3d b%-4d %s %-13s r%d", c.man_bits, c.exp_bits, c.normal_binades,
                 c.bias, c.is_signed ? "signed  " : "unsigned", sat_name(c.sat), c.rand_bits);
}

static void image_case(Cfg c)
{
    if (!faithful32(c))
    {
        ++image_skipped;
        return;
    }
    ImageCounts h;
    image_run(c, &h);
    ++image_checked;
    image_excused += h.excused;

    char head[160];
    image_head(c, head, sizeof head);

    bool ok = true;
    for (int m = 0; m < NMODE; ++m)
        if (h.bad[m] || h.nzero[m]) ok = false;
    printf("  %s  %-5s excused %llu\n", head, ok ? "ok" : "FAIL", h.excused);
    if (ok)
        return;
    ++image_failures;
    for (int m = 0; m < NMODE; ++m)
    {
        if (!h.bad[m] && !h.nzero[m]) continue;
        printf("      %-3s  mismatches %-12llu  -0.0 %-10llu", MODE_NAME[m], h.bad[m], h.nzero[m]);
        if (h.bad[m])
            printf("  e.g. 0x%08x = %g", h.first_bad[m], bits_to_float(h.first_bad[m]));
        printf("\n");
    }
}

static Cfg image_cfg(bool binaryK, int man, int e, int bias, int nb, bool sgn, SaturationMode sat, SubnormalsMode sub)
{
    return Cfg{binaryK, man, e, bias, nb, sgn, sat, sub, man + 8 <= 23 ? 8 : 23 - man};
}

// The biases that put a binaryK format's floor on binary32's faithful edge --
// its smallest value at 2^-125 under SUBNORMALS, at 2^-126 under the other two
// except where half of that floor is out of reach, which faithful32 puts at
// 2^-125 too -- next to one a binade clear of it, and its top at 2^127.
static int floor_edge_bias(int man, SubnormalsMode sub)
{
    if (sub == SubnormalsMode::SUBNORMALS)
        return 126 - man;
    if (man == 0)
        return 126; // 1 - bias >= -125, both modes: the floor is NORMALS'
    if (sub == SubnormalsMode::NORMALS)
        return 127; // 1 - bias >= -126
    return man == 23 ? 125 : 126; // EXTENDED_NORMALS: -bias >= -126, or -125 at 23
}

// The twelve configurations the first image64-sweep failed, all now outside
// faithful32 by the bound that came of them. Each must differ from binary64 in
// exactly one mode, at 2^-127 and (signed) its negation, and nowhere else --
// not in SR, which the dead floor used to fail, and with no -0.0.
static bool image_edges_signed_only = false; // image64-short runs half of them

static void image_edges()
{
    const SaturationMode O = SaturationMode::SAT_NONE, F = SaturationMode::SAT_FINITE,
                         P = SaturationMode::SAT_PROPAGATE;
    const SubnormalsMode N = SubnormalsMode::NORMALS, E = SubnormalsMode::EXTENDED_NORMALS;
    constexpr int RNE = 0, RNA = 1;
    struct Edge { int man, bias; SaturationMode sat; SubnormalsMode sub; int mode; };
    const Edge edges[] = {{0, 127, F, N, RNE}, {0, 127, O, N, RNE}, {0, 127, P, E, RNE},
                          {0, 127, O, E, RNE}, {23, 126, P, E, RNA}, {23, 126, O, E, RNA}};
    printf("image64, the binary32 edges -- expected to differ at 2^-127 only\n");
    for (const Edge &ed : edges)
        for (bool sgn : {true, false})
        {
            if (!sgn && image_edges_signed_only)
                continue;
            Cfg c = image_cfg(true, ed.man, 7, ed.bias, 1, sgn, ed.sat, ed.sub);
            if (faithful32(c))
            {
                printf("  faithful32 accepts a known binary32 edge -- the bound moved back\n");
                ++image_failures;
                continue;
            }
            ImageCounts h;
            image_run(c, &h);
            ++image_checked;
            image_excused += h.excused;
            bool ok = true;
            for (int m = 0; m < NMODE; ++m)
            {
                const unsigned long long want = m == ed.mode ? (sgn ? 2 : 1) : 0;
                if (h.bad[m] != want || h.nzero[m] || (want && h.first_bad[m] != 0x00400000u))
                    ok = false;
            }
            char head[160];
            image_head(c, head, sizeof head);
            printf("  %s  %-5s %s at 2^-127: %llu  excused %llu\n", head, ok ? "ok" : "FAIL", MODE_NAME[ed.mode],
                   h.bad[ed.mode], h.excused);
            if (ok)
                continue;
            ++image_failures;
            for (int m = 0; m < NMODE; ++m)
                if (h.bad[m] || h.nzero[m])
                    printf("      %-3s  mismatches %-12llu  -0.0 %-10llu  e.g. 0x%08x\n", MODE_NAME[m], h.bad[m],
                           h.nzero[m], h.first_bad[m]);
        }
}

// `stride` 1 is `image64-sweep`: every faithful combination of the axes below.
// The default is a thinned grid of about 50; `image64-host` is a list of 13
// that enters every arm, for the same reason `host` above is a list.
static void image_grid(int stride)
{
    const SaturationMode O = SaturationMode::SAT_NONE, F = SaturationMode::SAT_FINITE,
                         P = SaturationMode::SAT_PROPAGATE;
    const SubnormalsMode S = SubnormalsMode::SUBNORMALS, N = SubnormalsMode::NORMALS,
                         E = SubnormalsMode::EXTENDED_NORMALS;
    struct EB { int e, bias; }; // bias 9999: floor_edge_bias
    const EB ebs[] = {{2, -8}, {4, 7}, {7, 63}, {7, 0}, {7, 9999}, {3, 1}, {5, 15}, {6, 31}};
    const int n_ebs = stride == 1 ? 8 : 5;
    struct MS { SaturationMode sat; SubnormalsMode sub; };
    const MS ms[] = {{O, S}, {F, N}, {P, E}, {F, S}, {O, N}, {O, E}};
    const int n_ms = stride == 1 ? 6 : 3;
    const int mans_thin[] = {0, 3, 10, 23};

    printf("image64, binaryK -- 2^32 float inputs x 7 rounding modes per configuration\n");
    int count = stride == 1 ? 24 : 4;
    for (int i = 0; i < count; ++i)
    {
        const int man = stride == 1 ? i : mans_thin[i];
        for (int k = 0; k < n_ebs; ++k)
            for (int j = 0; j < n_ms; ++j)
                for (bool sgn : {true, false})
                {
                    if (stride != 1 && sgn != ((i + k + j) % 2 == 0))
                        continue; // the thin grid alternates signedness rather than doubling
                    const int bias = ebs[k].bias == 9999 ? floor_edge_bias(man, ms[j].sub) : ebs[k].bias;
                    image_case(image_cfg(true, man, ebs[k].e, bias, 1, sgn, ms[j].sat, ms[j].sub));
                }
    }
    printf("image64, superfp\n");
    struct SB { int e, bias, nb; };
    const SB sbs[] = {{4, 7, 1}, {4, 7, 15}, {2, 1, 3}, {3, 3, 1}, {4, 7, 2}};
    const int mans_superfp[] = {0, 1, 3, 6};
    for (int man : mans_superfp)
        for (const SB &sb : sbs)
            for (const SaturationMode sat : {O, F, P})
                for (bool sgn : {true, false})
                {
                    if (stride != 1 && (sat != (man % 3 == 0 ? O : man % 3 == 1 ? F : P) || sgn != (sb.nb != 15)))
                        continue;
                    image_case(image_cfg(false, man, sb.e, sb.bias, sb.nb, sgn, sat, S));
                }
}

static void image_host_list()
{
    const SaturationMode O = SaturationMode::SAT_NONE, F = SaturationMode::SAT_FINITE,
                         P = SaturationMode::SAT_PROPAGATE;
    const SubnormalsMode S = SubnormalsMode::SUBNORMALS, N = SubnormalsMode::NORMALS,
                         E = SubnormalsMode::EXTENDED_NORMALS;
    printf("image64, host list -- 2^32 float inputs x 7 rounding modes per configuration\n");
    image_case(image_cfg(true, 3, 4, 7, 1, true, O, S));
    image_case(image_cfg(true, 3, 4, 7, 1, true, F, N));
    image_case(image_cfg(true, 3, 4, 7, 1, false, P, E));
    image_case(image_cfg(true, 0, 4, 7, 1, true, O, S));
    image_case(image_cfg(true, 10, 5, 15, 1, true, O, S));
    image_case(image_cfg(true, 23, 7, 63, 1, true, F, S));
    image_case(image_cfg(true, 3, 7, floor_edge_bias(3, S), 1, true, O, S));
    image_case(image_cfg(true, 3, 7, floor_edge_bias(3, N), 1, true, O, N));
    image_case(image_cfg(true, 3, 7, floor_edge_bias(3, E), 1, false, F, E));
    image_case(image_cfg(true, 3, 7, 0, 1, true, F, S));
    image_case(image_cfg(false, 3, 4, 7, 1, true, O, S));
    image_case(image_cfg(false, 0, 4, 7, 1, true, P, S));
    image_case(image_cfg(false, 6, 4, 7, 15, false, F, S));
}

// ===========================================================================
// The short lists: the per-change gate.
//
// The grids above are the exhaustive record and cost accordingly (`spot` is 576
// configurations, `image64-sweep` 2406, and a host configuration ~100 s). What a
// change to the casts usually breaks, though, is an edge -- a mantissa width the
// bit formulas special-case, a floor or a top on the carrier's boundary, a clip
// corner, an empty or degenerate region -- and the edges are few. So `short`,
// `image64-short` and their `-host` twins run those, each named for the edge it
// is there for, beside a handful of ordinary formats to show the middle did not
// move. The grids run once, at the end of a piece of work.
// ===========================================================================

// binary32, shipped against the reference, every input.
static void short_list()
{
    const SaturationMode O = SaturationMode::SAT_NONE, F = SaturationMode::SAT_FINITE,
                         P = SaturationMode::SAT_PROPAGATE, X = SaturationMode::SAT_FINITE_EXTENDED;
    const SubnormalsMode S = SubnormalsMode::SUBNORMALS, N = SubnormalsMode::NORMALS,
                         E = SubnormalsMode::EXTENDED_NORMALS;
    auto bk = [](int man, int e, int bias, bool sgn, SaturationMode sat, SubnormalsMode sub) {
        run_case(image_cfg(true, man, e, bias, 1, sgn, sat, sub));
    };
    auto sf = [](int man, int e, int nb, int bias, bool sgn, SaturationMode sat) {
        run_case(image_cfg(false, man, e, bias, nb, sgn, sat, SubnormalsMode::SUBNORMALS));
    };
    printf("short, binaryK -- 2^32 inputs x 9 rounding modes per configuration\n");
    // SatFinite in the extended domain: the fast path, the integer clip, and an
    // unsigned format's two reserved codes; then a superfp.
    bk(3, 4, 8, true, X, S);
    bk(3, 4, 8, true, X, N);
    bk(2, 5, 16, false, X, E);
    sf(3, 4, 1, 7, true, X);
    // the middle: P3109's Binary8p4, OCP's E4M3 and E5M2 shapes, a 16-bit format
    bk(3, 4, 8, true, O, S);
    bk(3, 4, 7, true, F, S);
    bk(2, 5, 15, false, P, S);
    bk(10, 5, 15, true, O, S);
    // man_bits 0: the zero-argument round, its tie parity, and the floor's carry
    bk(0, 4, 7, true, O, S);
    bk(0, 7, 126, false, F, N);   // P = 1's faithful floor, 2^-125
    bk(0, 7, 127, true, O, E);    // min_exponent_store 0, the floor NORMALS' (missing before 69a9eae)
    bk(0, 2, -127, true, O, E);   // min_exponent_store 254, the floor past the carrier (inf before 69a9eae)
    // man_bits 1: the reserved codes outnumber the top binade's
    bk(1, 4, 7, false, O, S);
    bk(1, 3, 3, true, P, N);
    // man_bits 22 and 23: the last mask, the bypass, rand_bits 1 and 0
    bk(22, 7, 104, true, O, S);   // SUBNORMALS' floor at 2^-125
    bk(23, 7, 103, false, F, S);
    bk(23, 7, 125, true, O, E);   // EXTENDED_NORMALS' P = 24 floor, 2^-125
    bk(23, 7, 126, true, P, E);   // a binade below it, the half-floor binary32 does not hold
    // the clips' corners at an ordinary width
    bk(3, 8, 123, true, O, S);    // the smallest value at 2^-125
    bk(3, 8, 124, true, F, S);    // a binade below it
    bk(3, 8, 125, true, O, S);    // the subnormal clip's min_exponent_store at 0
    bk(3, 7, 127, true, O, N);    // the normal clip's floor at 2^-126
    bk(3, 7, 128, false, F, N);   // its underflow test dead
    bk(3, 7, 126, true, O, E);
    bk(3, 7, 127, false, P, E);
    bk(3, 2, -126, true, O, N);   // the floor on the top field, 2^127
    // the top: exactly binary32's, past it, and no finite normal at all
    bk(3, 4, -112, true, O, S);
    bk(3, 4, -113, true, F, S);
    bk(3, 4, -113, false, P, N);
    bk(3, 4, 200, true, O, S);
    bk(0, 2, -8, true, F, S);     // a format with nothing below 1
    // NORMALS' floor at 2^-126 with man + N = 23: a binary32 subnormal takes
    // its arm with sh = -1, and goes up (reference_casts.h's ref_sr_coarse_floor)
    run_case(Cfg{true, 0, 7, 127, 1, true, O, N, 23});
    run_case(Cfg{true, 0, 7, 127, 1, false, F, N, 22}); // sh = 0: the exact arm
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked, failures);

    const int before = checked, before_bad = failures;
    printf("short, superfp -- 2^32 inputs x 9 rounding modes per configuration\n");
    sf(3, 4, 1, 7, true, O);      // the middle
    sf(2, 5, 2, 15, false, F);
    sf(0, 4, 1, 7, true, P);      // man_bits 0
    sf(4, 4, 1, 7, true, O);      // the supernormal cutoff below -127: a binary32 subnormal in that arm
    sf(3, 4, 15, 7, true, F);     // one supernormal binade left
    sf(2, 4, 16, 7, true, O);     // none left: the regions invert
    sf(3, 4, 1, 21, true, O);     // the supernormal floor at 2^-125
    sf(3, 4, 1, 22, false, F);    // a binade below it
    sf(13, 8, 2, 127, true, O);   // a cutoff far past int, clamped (superfp_region_cutoffs, 90beb61)
    sf(2, 3, 1, -120, true, F);   // the top at binary32's
    sf(2, 3, 1, -121, true, P);   // past it
    // the supernormal floor 2^c between binary32's smallest subnormal and its
    // normals, where a binary32 subnormal takes the arm below 2^c with the
    // shift sh = 148 + c - N: exact while it is not negative, and up when it
    // is (reference_casts.h's ref_sr_coarse_floor)
    sf(0, 8, 1, 146, true, O);    // c = -145, N = 8: sh = -5
    sf(2, 4, 2, 104, false, F);   // c = -145 again, through man_bits 2
    sf(0, 8, 1, 149, true, O);    // c = -148, the window's bottom: sh = -8
    run_case(Cfg{false, 0, 8, 146, 1, true, O, SubnormalsMode::SUBNORMALS, 20}); // sh = -17
    run_case(Cfg{false, 0, 8, 146, 1, true, O, SubnormalsMode::SUBNORMALS, 2});  // sh = 1: the exact arm
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before, failures - before_bad);
    fx_short_list();
}

// What a host run adds is g++'s codegen for the same casts, which does not vary
// with the format -- so one configuration per kind of arm: the host's binaryK
// RNE fast path, the man_bits == 0 formulas at the moved floor, the 23-bit
// bypass under EXTENDED_NORMALS, the subnormal clip's corner, and superfp's
// supernormal arm with a binary32 subnormal in it. The last four are the host's
// own spelling of superfp's RNE fast path (cast_superfp.h, since f9a6aa9),
// which the GPU grids do not compile unless asked to
// (MPTORCH_SUPERFP_RNE_HOST_SPELLING): its rare branch with an underflow region
// below it, its range test's floor at the lowest one the gate admits, its
// normal arm saturating in the extended domain, and the range one binade wide.
static void short_host_list()
{
    const SaturationMode O = SaturationMode::SAT_NONE, F = SaturationMode::SAT_FINITE,
                         P = SaturationMode::SAT_PROPAGATE, X = SaturationMode::SAT_FINITE_EXTENDED;
    const SubnormalsMode S = SubnormalsMode::SUBNORMALS;
    printf("short host list -- 2^32 inputs x 9 rounding modes per configuration\n");
    run_case(image_cfg(true, 3, 4, 8, 1, true, X, S));
    run_case(image_cfg(true, 3, 4, 8, 1, true, O, S));
    run_case(image_cfg(true, 0, 7, 127, 1, false, F, SubnormalsMode::EXTENDED_NORMALS));
    run_case(image_cfg(true, 23, 7, 125, 1, true, P, SubnormalsMode::EXTENDED_NORMALS));
    run_case(image_cfg(true, 3, 8, 125, 1, true, F, S));
    run_case(image_cfg(false, 4, 4, 7, 1, true, O, S));
    run_case(image_cfg(false, 3, 4, 7, 1, true, O, S));  // an underflow region, floor 2^-111
    run_case(image_cfg(false, 3, 4, 21, 1, true, F, S));  // the floor at 2^-125
    run_case(image_cfg(false, 3, 4, 7, 1, true, X, S));  // SatFinite in the extended domain
    run_case(image_cfg(false, 3, 4, 7, 15, true, F, S)); // one supernormal binade
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked, failures);
    fx_host_list();
}

// binary64 against binary32, every input, formats inside faithful32 only -- each
// listed one is checked to be, so a bound that moves shows up here rather than
// as a silent skip. The floors at each subnormals mode's faithful edge across
// the mantissa widths that edge depends on, the top at 2^127, the middle, and
// the signed half of image_edges' six formats, which must still differ at
// 2^-127 only.
static void image_short(bool host)
{
    const SaturationMode O = SaturationMode::SAT_NONE, F = SaturationMode::SAT_FINITE,
                         P = SaturationMode::SAT_PROPAGATE;
    const SubnormalsMode S = SubnormalsMode::SUBNORMALS, N = SubnormalsMode::NORMALS,
                         E = SubnormalsMode::EXTENDED_NORMALS;
    auto one = [](Cfg c) {
        if (!faithful32(c))
        {
            char head[160];
            image_head(c, head, sizeof head);
            printf("  %s  FAIL  listed as faithful, and faithful32 says it is not\n", head);
            ++image_failures;
            return;
        }
        image_case(c);
    };
    auto bk = [&](int man, int e, int bias, bool sgn, SaturationMode sat, SubnormalsMode sub) {
        one(image_cfg(true, man, e, bias, 1, sgn, sat, sub));
    };
    auto edge = [&](int man, bool sgn, SaturationMode sat, SubnormalsMode sub) {
        bk(man, 7, floor_edge_bias(man, sub), sgn, sat, sub);
    };
    if (host)
    {
        printf("image64, short host list -- 2^32 float inputs x 7 rounding modes per configuration\n");
        bk(3, 4, 8, true, O, S);
        edge(0, false, P, N);
        edge(23, true, O, E);
        one(image_cfg(false, 3, 4, 7, 1, true, F, S));
        return;
    }
    printf("image64, short -- 2^32 float inputs x 7 rounding modes per configuration\n");
    bk(3, 4, 8, true, O, S);       // the middle
    bk(2, 5, 15, false, F, S);
    bk(10, 5, 15, true, P, N);
    edge(0, true, F, S);           // the floors on binary32's faithful edge
    edge(1, false, O, S);
    edge(3, true, P, S);
    edge(22, false, F, S);
    edge(23, true, O, S);
    edge(0, false, P, N);
    edge(1, true, F, N);
    edge(3, false, O, N);
    edge(23, true, P, N);
    edge(0, true, O, E);
    edge(1, false, P, E);
    edge(3, true, F, E);
    edge(22, false, O, E);
    edge(23, true, P, E);
    bk(0, 7, 0, true, F, S);       // the top at 2^127
    bk(3, 4, -112, true, O, S);
    bk(23, 7, 0, false, F, S);
    bk(1, 4, -112, false, P, N);   // with the reserved codes outnumbering the top binade's
    bk(0, 2, -8, true, F, S);      // small
    bk(3, 3, 1, false, O, E);
    printf("image64, short, superfp\n");
    one(image_cfg(false, 3, 4, 7, 1, true, O, S));
    one(image_cfg(false, 0, 4, 7, 1, true, P, S));
    one(image_cfg(false, 3, 4, 7, 15, false, F, S));
    one(image_cfg(false, 3, 4, 21, 1, true, O, S));   // the supernormal floor at 2^-125
    one(image_cfg(false, 4, 4, 7, 12, true, O, S));
    one(image_cfg(false, 2, 3, -120, 1, true, F, S)); // the top at 2^127
}

// ===========================================================================
// sr-mean: stochastic rounding's distribution, exactly, against P3109 rather
// than against the reference.
//
// Every other mode here compares two spellings of a rule, which cannot see a
// rule both spell wrong -- and SR's rule is a distribution, not a value. So
// this enumerates the draw instead of sampling it: the cast reads N =
// rand_bits bits R of its random word, directly below the format's
// significand, and taking all 2^N of them counts exactly how many go to x's
// upper neighbour. P3109 (arXiv:2606.04028, Fig. 2) says how many, with lo
// and hi the neighbours -- this pass's own RZ and RU results, which the sweeps
// above verify over every input -- and X = (|x| - lo) / (hi - lo) * 2^N:
// floor(X) under StochasticA, floor(X + 1/2) under B (the R with floor(2X) +
// 2R + 1 >= 2^(N+1)), and RNITE(X) under C. Every draw's result must be lo or
// hi, and the count must be that one exactly.
//
// X is exact in double wherever hi - lo is a power of two; an EXTENDED_NORMALS
// floor, which is not one, takes reference_casts.h's exact search. The inputs
// are where SR has more than one arm to pass through: every 64th of the floor
// below it, the cell just under the floor and the one just over it,
// EXTENDED_NORMALS' hole, and a few ordinary cells above. `sr-mean-old` runs
// the same against the casts as they stood at 4164cee (old_sr_casts.h), which
// must fail: that is the check's negative control.
// ===========================================================================

struct SrMeanResult
{
    int inputs, off_grid, wrong;
    float worst_x;   // a failing input
    int worst_mode;  // its mode
    long long worst_up, worst_expect;
};

// floor(X), floor(X + 1/2) and RNITE(X): the number of the 2^N draws that
// StochasticA, B and C send up.
static void sr_expect(float x, double lo, double hi, int n, long long expect[3])
{
    const double ax = fabs((double)x);
    int e;
    const double m = frexp(hi - lo, &e);
    if (m == 0.5)
    {
        const double X = ldexp((ax - lo) / (hi - lo), n); // both exact
        expect[0] = (long long)floor(X);
        expect[1] = (long long)floor(X + 0.5);
        expect[2] = (long long)nearbyint(X); // the default rounding: ties to even
        return;
    }
    // lo = 0 and hi not a power of two: |x| = s * 2^ex against hi = hm * 2^eh
    int ex, eh;
    const double fx = frexp(ax, &ex), fh = frexp(hi, &eh);
    unsigned long long s = (unsigned long long)ldexp(fx, 53), hm = (unsigned long long)ldexp(fh, 53);
    int sx = ex - 53, sh = eh - 53;
    while (s && !(s & 1)) s >>= 1, ++sx;
    while (hm && !(hm & 1)) hm >>= 1, ++sh;
    const RefSrX X = ref_sr_x_scaled(s, sx, hm, sh, n);
    expect[0] = (long long)X.floor_x;
    expect[1] = (long long)((X.floor_2x + 1) / 2);
    expect[2] = (long long)X.rnite_x;
}

static SrMeanResult sr_mean_case(const Cfg &c, bool old)
{
    const BinaryKParams bp = make_binaryK_params(c.man_bits, c.exp_bits, c.bias, c.is_signed, c.sat,
                                                 c.subm == SubnormalsMode::EXTENDED_NORMALS);
    const SuperfpParams sp = make_superfp_params(c.man_bits, c.exp_bits, c.normal_binades, c.bias, c.sat);
    constexpr int RU = 2, RZ = 4, SR = 6;
    const int n = c.rand_bits;

    // the floor, as the format defines it rather than as the params spell it
    const int min_exp = 1 - c.bias;
    double floor_val = ldexp(1.0, min_exp);
    if (c.is_binaryK && c.subm == SubnormalsMode::SUBNORMALS)
        floor_val = ldexp(1.0, min_exp - c.man_bits);
    else if (c.is_binaryK && c.subm == SubnormalsMode::EXTENDED_NORMALS && c.man_bits)
        floor_val = ldexp(1.0 + ldexp(1.0, -c.man_bits), min_exp - 1);
    const double hole = ldexp(1.0, min_exp - 1);

    std::vector<double> xs;
    for (int k = 1; k < 64; ++k) xs.push_back(floor_val * k / 64.0);
    for (double f : {0.9, 0.93, 0.95, 0.97, 0.99, 0.999, 1.001, 1.01, 1.05, 1.3, 1.7, 2.6, 3.3})
        xs.push_back(floor_val * f);
    for (double f : {1.001, 1.03, 1.07, 1.11}) xs.push_back(hole * f);
    for (double v : {0.3, 0.7, 1.1, 1.9, 3.14159}) xs.push_back(v);
    // a hair either side of every draw level under the floor, where a cast
    // that rounds |x| in the carrier first goes wrong
    if (n >= 1 && n <= 12)
        for (int k = 1; k < (1 << n); k += (1 << n) / 4 > 0 ? (1 << n) / 4 : 1)
        {
            const float at = (float)(floor_val * ldexp((double)k, -n));
            xs.push_back(nextafterf(at, 0.0f));
            xs.push_back(nextafterf(at, 3.0e38f));
        }

    SrMeanResult r{0, 0, 0, 0.0f, 0, 0, 0};
    const int shift = 23 - c.man_bits - n;
    for (double xd : xs)
    {
        const float x = (float)xd;
        if (x == 0.0f || !(fabs(x) < 3.0e38f))
            continue;
        const float lo = fabs(shipped_cast(RZ, x, 0u, c, bp, sp));
        const float hi = fabs(shipped_cast(RU, x, 0u, c, bp, sp));
        if (!(hi < 3.5e38f))
            continue; // past the top: the upper neighbour is an overflow, and saturation is another test's
        long long expect[3] = {0, 0, 0};
        if (hi != lo)
            sr_expect(x, lo, hi, n, expect);
        for (int v = 0; v < (old ? 1 : 3); ++v)
        {
            ++r.inputs;
            long long up = 0, stray = 0;
            for (uint32_t k = 0; k < (1u << n); ++k)
            {
                const uint32_t rnd = k << shift;
                float y;
                if (old)
                    y = c.is_binaryK ? old_sr::cast_binaryK(x, rnd, n, c.is_signed, c.subm, bp)
                                     : old_sr::cast_superfp(x, rnd, n, c.is_signed, sp);
                else
                    y = shipped_cast(SR + v, x, rnd, c, bp, sp);
                y = fabs(y);
                if (y == hi && hi != lo) ++up;
                else if (y != lo) ++stray;
            }
            if (stray)
            {
                ++r.off_grid;
                if (!r.worst_x) r.worst_x = x, r.worst_mode = SR + v, r.worst_up = -1, r.worst_expect = expect[v];
                continue;
            }
            if (up != expect[v])
            {
                if (!r.wrong) r.worst_x = x, r.worst_mode = SR + v, r.worst_up = up, r.worst_expect = expect[v];
                ++r.wrong;
            }
        }
    }
    return r;
}

static void sr_mean(bool old = false)
{
    const SaturationMode O = SaturationMode::SAT_NONE;
    const SubnormalsMode S = SubnormalsMode::SUBNORMALS, N = SubnormalsMode::NORMALS,
                         E = SubnormalsMode::EXTENDED_NORMALS;
    printf("sr-mean%s -- every draw, for inputs around each region's edges; the counts that go up must be\n"
           "P3109's exactly (floor(X), floor(X + 1/2), RNITE(X) for StochasticA, B, C)\n", old ? "-old" : "");
    struct Row { bool bk; int man, e, bias, nb, rb; SubnormalsMode sub; };
    const Row rows[] = {
        {true, 0, 4, 8, 1, 12, S},  {true, 0, 4, 8, 1, 12, N},  {true, 0, 4, 8, 1, 12, E},
        {true, 1, 4, 8, 1, 12, S},  {true, 1, 4, 8, 1, 12, N},  {true, 1, 4, 8, 1, 12, E},
        {true, 3, 4, 8, 1, 12, S},  {true, 3, 4, 8, 1, 12, N},  {true, 3, 4, 8, 1, 12, E},
        {true, 8, 5, 15, 1, 8, S},  {true, 8, 5, 15, 1, 8, N},  {true, 8, 5, 15, 1, 8, E},
        {true, 0, 7, 126, 1, 12, N}, {true, 3, 7, 127, 1, 12, N}, {true, 3, 7, 126, 1, 12, E},
        {false, 3, 4, 7, 1, 12, S}, {false, 0, 4, 7, 1, 12, S},
        // few random bits, where B is round-half-away at N = 0 and C rounds
        // half toward zero, and the bound (man + N = 23), where B and C are A
        {true, 3, 4, 8, 1, 0, S},   {true, 3, 4, 8, 1, 1, N},   {true, 3, 4, 8, 1, 2, E},
        {false, 3, 4, 7, 1, 0, S},  {false, 3, 4, 7, 1, 2, S},  {true, 3, 5, 15, 1, 20, S},
    };
    for (const Row &w : rows)
    {
        Cfg c{w.bk, w.man, w.e, w.bias, w.nb, true, O, w.sub, w.rb};
        const SrMeanResult r = sr_mean_case(c, old);
        ++checked;
        char head[160];
        image_head(c, head, sizeof head);
        const bool ok = !r.wrong && !r.off_grid;
        printf("  %s  %-4s %d inputs", head, ok ? "ok" : "FAIL", r.inputs);
        if (!ok)
            printf("  [%d wrong, %d off the neighbours; e.g. %g under %s: %lld of 2^%d draws up, %lld expected]",
                   r.wrong, r.off_grid, r.worst_x, MODE_NAME[r.worst_mode], r.worst_up, c.rand_bits,
                   r.worst_expect);
        printf("\n");
        if (!ok) ++failures;
    }
    if (!old)
        fx_sr_mean();
}

int main(int argc, char **argv)
{
    std::string what = argc > 1 ? argv[1] : "spot";
    if (what == "sr-mean")
    {
        sr_mean();
        printf("\n%s: %d configurations checked, %d bad\n", failures ? "FAIL" : "PASS", checked, failures);
        return failures ? 1 : 0;
    }
    if (what == "sr-mean-old")
    {
        // the negative control: the casts as they stood at 4164cee must fail
        sr_mean(true);
        printf("\n%s: %d configurations checked, %d bad (the old casts must fail)\n",
               failures ? "EXPECTED FAIL" : "UNEXPECTED PASS", checked, failures);
        return failures ? 0 : 1;
    }
    if (what == "image64-short" || what == "image64-short-host")
    {
        image_short(what == "image64-short-host");
        if (what == "image64-short")
        {
            // the signed half of image64-edges, which is where faithful32's bound
            // at 2^-125 and the floor restored at 69a9eae are held
            const int before_checked = image_checked, before_failures = image_failures;
            image_edges_signed_only = true;
            image_edges();
            printf("  %d edge configurations, %d bad\n", image_checked - before_checked,
                   image_failures - before_failures);
        }
        printf("\n%s: %d configurations checked, %d bad; %llu SR inputs excused in carrier-arithmetic regions\n",
               image_failures ? "FAIL" : "PASS", image_checked, image_failures, image_excused);
        return image_failures ? 1 : 0;
    }
    if (what == "fixed" || what == "fixed-host")
    {
        // the fixed-point rows alone, for a change to cast_fixedpoint.h only
        if (what == "fixed") fx_short_list();
        else fx_host_list();
        printf("\n%s: %d configurations checked, %d bad\n", failures ? "FAIL" : "PASS", checked, failures);
        return failures ? 1 : 0;
    }
    if (what == "fixed-beyond")
    {
        fx_beyond_list();
        printf("\n%s: %d configurations checked, %d bad\n", failures ? "FAIL" : "PASS", checked, failures);
        return failures ? 1 : 0;
    }
    if (what == "short" || what == "short-host")
    {
        if (what == "short") short_list();
        else short_host_list();
        printf("\n%s: %d configurations checked, %d bad\n", failures ? "FAIL" : "PASS", checked, failures);
        return failures ? 1 : 0;
    }
    if (what.rfind("image64", 0) == 0)
    {
        if (what == "image64-sweep") image_grid(1);
        else if (what == "image64-host") image_host_list();
        else if (what == "image64-edges") image_edges();
        else image_grid(8);
        printf("\n%s: %d configurations checked (%d outside binary32's faithful bounds skipped), %d bad; "
               "%llu SR inputs excused in carrier-arithmetic regions\n",
               image_failures ? "FAIL" : "PASS", image_checked, image_skipped, image_failures, image_excused);
        return image_failures ? 1 : 0;
    }
    if (what == "sweep") grid(1);
    else if (what == "spot-pairs") grid_pairs();
    else if (what == "host") host_list();
#if !defined(__CUDACC__)
    else if (what == "host-sampled")
    {
        // host_list's configurations on a sample of each one's inputs (1 in
        // 60 low halves): a configuration costs ~18 min of 16 threads at 9
        // modes in full (the definition-based SR references), ~17 s like
        // this; the 22 run in 6.4 min
        host_lo_stride = 61;
        host_list();
    }
#endif
    else if (what == "quick") grid(24);
    else grid(8);
    printf("\n%s: %d configurations checked, %d bad\n", failures ? "FAIL" : "PASS", checked, failures);
    return failures ? 1 : 0;
}
